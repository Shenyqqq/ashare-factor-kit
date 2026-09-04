"""
research/market_regime.py — CSI300 HMM market regime breakdown (daily / weekly / yearly).

Usage:
    python -m research.market_regime
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import BACKTEST_START, BACKTEST_END, RAW_DIR
from factors.factor import _fit_hmm_regime

OUTPUT_DIR = ROOT / "research" / "output"


def load_csi300_close() -> pd.Series:
    """Load CSI300 close — same fallback order as run.py."""
    for fname in ("csi300.parquet", "index_000300.parquet"):
        path = RAW_DIR / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "close" in df.columns:
            close = df["close"]
        else:
            close = df.iloc[:, 0]
        close = close.squeeze()
        close.index = pd.to_datetime(close.index)
        return close.sort_index().astype(float)
    raise FileNotFoundError(
        f"CSI300 data not found in {RAW_DIR}. "
        "Expected csi300.parquet or index_000300.parquet."
    )


def assign_regime(strong: pd.Series, weak: pd.Series) -> pd.Series:
    """Discrete regime from HMM posterior probabilities."""
    regime = pd.Series("震荡", index=strong.index, dtype="object")
    bull = (strong > 0.4) & (strong > weak)
    bear = (weak > 0.4) & (weak > strong)
    regime[bull] = "强势"
    regime[bear] = "弱势"
    return regime


def compute_daily(close: pd.Series) -> pd.DataFrame:
    """Daily HMM probs + context features + regime label."""
    idx_ret = close.pct_change()
    hmm = _fit_hmm_regime(idx_ret)
    hmm = hmm.rename(columns={
        "HMM_强势概率": "HMM_强势",
        "HMM_弱势概率": "HMM_弱势",
    })
    hmm["HMM_震荡"] = 1.0 - hmm["HMM_强势"] - hmm["HMM_弱势"]

    ma60 = close.rolling(60).mean()
    df = pd.DataFrame({
        "close": close,
        "市场动量_60d": close.pct_change(60),
        "市场MA偏离_60d": close / ma60.replace(0, np.nan) - 1,
        "市场波动率_20d": idx_ret.rolling(20).std(),
    })
    df = df.join(hmm)
    df["regime"] = assign_regime(df["HMM_强势"], df["HMM_弱势"])
    return df


def resample_last(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Last observation per calendar period (W-FRI / month / year)."""
    if freq == "W-FRI":
        periods = df.index.to_period("W-FRI")
    elif freq == "M":
        periods = df.index.to_period("M")
    elif freq == "Y":
        periods = df.index.to_period("Y")
    else:
        raise ValueError(freq)
    out = df.groupby(periods).last()
    out.index = df.groupby(periods).apply(lambda g: g.index[-1])
    out.index.name = "date"
    return out


def regime_pct_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    counts = df["regime"].value_counts()
    total = counts.sum()
    pct = (counts / total * 100).round(1)
    out = pd.DataFrame({"count": counts, "pct": pct})
    out.index.name = label
    return out


def monthly_dominant_table(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = resample_last(daily, "M")
    rows = []
    for period, row in monthly.iterrows():
        rows.append({
            "month": str(period),
            "regime": row["regime"],
            "HMM_强势": round(row["HMM_强势"], 3),
            "HMM_弱势": round(row["HMM_弱势"], 3),
            "close": round(row["close"], 1),
        })
    return pd.DataFrame(rows)


def find_regime_stretches(series: pd.Series, min_len: int = 8) -> list[dict]:
    """Longest consecutive runs of each regime (weekly observations)."""
    if series.empty:
        return []
    runs: list[dict] = []
    start = series.index[0]
    prev = series.iloc[0]
    length = 1
    for dt, val in zip(series.index[1:], series.iloc[1:]):
        if val == prev:
            length += 1
        else:
            if length >= min_len:
                runs.append({
                    "regime": prev,
                    "start": start,
                    "end": series.index[series.index.get_loc(dt) - 1],
                    "weeks": length,
                })
            start, prev, length = dt, val, 1
    if length >= min_len:
        runs.append({"regime": prev, "start": start, "end": series.index[-1], "weeks": length})
    return sorted(runs, key=lambda x: x["weeks"], reverse=True)


def weekly_switch_stats(weekly: pd.DataFrame) -> dict:
    changed = weekly["regime"].ne(weekly["regime"].shift()).iloc[1:]
    n_weeks = len(weekly) - 1
    n_switches = int(changed.sum())
    return {
        "n_weeks": n_weeks,
        "n_switches": n_switches,
        "switch_rate_pct": round(n_switches / n_weeks * 100, 1) if n_weeks else 0.0,
        "avg_weeks_between_switches": round(n_weeks / n_switches, 1) if n_switches else float("inf"),
    }


def write_summary_md(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    yearly: pd.DataFrame,
    path: Path,
) -> None:
    start, end = daily.index.min().date(), daily.index.max().date()
    yearly_pct = (
        daily.groupby(daily.index.year)["regime"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mul(100)
        .round(1)
    )
    for col in ["强势", "震荡", "弱势"]:
        if col not in yearly_pct.columns:
            yearly_pct[col] = 0.0
    yearly_pct = yearly_pct[["强势", "震荡", "弱势"]]
    yearly_pct.index.name = "year"

    monthly_dom = monthly_dominant_table(daily)
    switch = weekly_switch_stats(weekly)
    stretches = find_regime_stretches(weekly["regime"], min_len=8)

    lines = [
        "# CSI300 HMM Market Regime Summary",
        "",
        f"Period: {start} → {end} ({len(daily)} trading days)",
        "",
        "## Yearly regime % (daily observations)",
        "",
        yearly_pct.to_markdown(),
        "",
        "## Weekly switch statistics",
        "",
        f"- Weeks analysed: {switch['n_weeks']}",
        f"- Regime switches: {switch['n_switches']} ({switch['switch_rate_pct']}% of weeks)",
        f"- Avg weeks between switches: {switch['avg_weeks_between_switches']}",
        "",
        "## Long weekly stretches (≥8 weeks)",
        "",
    ]
    if stretches:
        for s in stretches[:12]:
            lines.append(
                f"- **{s['regime']}** {s['start'].date()} → {s['end'].date()} "
                f"({s['weeks']} weeks)"
            )
    else:
        lines.append("- None ≥8 weeks")

    lines += [
        "",
        "## Monthly end-of-month regime",
        "",
        monthly_dom.to_markdown(index=False),
        "",
        "## Training window notes",
        "",
        _training_window_advice(yearly_pct, switch, stretches),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _training_window_advice(yearly_pct: pd.DataFrame, switch: dict, stretches: list) -> str:
    """Heuristic guidance for lgbm walk-forward window length."""
    avg_gap = switch["avg_weeks_between_switches"]
    avg_gap_months = avg_gap / 4.33 if np.isfinite(avg_gap) else 12

    bear_years = yearly_pct.index[yearly_pct.get("弱势", 0) > 40].tolist()
    chop_years = yearly_pct.index[yearly_pct.get("震荡", 0) > 50].tolist()

    lines = [
        f"- Weekly regime switches ~every **{avg_gap:.0f} weeks** "
        f"(~{avg_gap_months:.1f} months); a **6-month** train window captures "
        f"~{6 / max(avg_gap_months, 0.1):.0f} regime segment(s).",
        f"- **12-month** window spans ~{12 / max(avg_gap_months, 0.1):.0f} switches — "
        "often mixes bull/chop/bear; good when regimes are short-lived.",
        f"- **24-month** window risks stale bear/chop samples when dominated by "
        f"weak/chop years {bear_years + chop_years}.",
    ]
    long_bear = [s for s in stretches if s["regime"] == "弱势" and s["weeks"] >= 16]
    if long_bear:
        lines.append(
            f"- Long **弱势** runs ({long_bear[0]['weeks']}w max) favour shorter "
            "validation windows or time-decay so old bull data does not dominate."
        )
    lines.append(
        "- **Practical**: h5 weekly rebalance → prefer **6–12m** train; "
        "h20 monthly → **12m** default OK, consider **6m** after extended 弱势/震荡."
    )
    return "\n".join(lines)


def plot_timeline(daily: pd.DataFrame, path: Path) -> None:
    colors = {"强势": "#2ecc71", "震荡": "#f39c12", "弱势": "#e74c3c"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    ax0, ax1, ax2 = axes
    ax0.plot(daily.index, daily["close"], color="#2c3e50", lw=0.8)
    for regime, grp in daily.groupby(
        (daily["regime"] != daily["regime"].shift()).cumsum()
    ):
        r = grp["regime"].iloc[0]
        ax0.axvspan(grp.index[0], grp.index[-1], alpha=0.18, color=colors.get(r, "#95a5a6"))
    ax0.set_ylabel("CSI300")
    ax0.set_title("CSI300 HMM Regime Timeline")

    ax1.fill_between(daily.index, 0, daily["HMM_强势"], alpha=0.6, color=colors["强势"], label="强势")
    ax1.fill_between(daily.index, 0, daily["HMM_弱势"], alpha=0.6, color=colors["弱势"], label="弱势")
    ax1.set_ylabel("HMM prob")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(daily.index, daily["市场MA偏离_60d"], color="#8e44ad", lw=0.7)
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.set_ylabel("MA60 dev")

    patches = [mpatches.Patch(color=c, label=r, alpha=0.5) for r, c in colors.items()]
    ax0.legend(handles=patches, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    close = load_csi300_close()
    close = close.loc[BACKTEST_START:BACKTEST_END]
    if close.empty:
        raise ValueError(f"No CSI300 data in [{BACKTEST_START}, {BACKTEST_END}]")

    daily = compute_daily(close)
    weekly = resample_last(daily, "W-FRI")
    yearly = resample_last(daily, "Y")

    daily_path = OUTPUT_DIR / "market_regime_daily.csv"
    weekly_path = OUTPUT_DIR / "market_regime_weekly.csv"
    daily.to_csv(daily_path, encoding="utf-8-sig")
    weekly.to_csv(weekly_path, encoding="utf-8-sig")

    summary_path = OUTPUT_DIR / "market_regime_summary.md"
    write_summary_md(daily, weekly, yearly, summary_path)

    chart_path = OUTPUT_DIR / "market_regime_timeline.png"
    plot_timeline(daily, chart_path)

    yearly_pct = (
        daily.groupby(daily.index.year)["regime"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mul(100)
        .round(1)
    )
    switch = weekly_switch_stats(weekly)

    print(f"Saved: {daily_path}")
    print(f"Saved: {weekly_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {chart_path}")
    print("\nYearly regime % (daily):")
    print(yearly_pct.to_string())
    print(f"\nWeekly switches: {switch['n_switches']}/{switch['n_weeks']} "
          f"({switch['switch_rate_pct']}%), avg gap {switch['avg_weeks_between_switches']}w")


if __name__ == "__main__":
    main()
