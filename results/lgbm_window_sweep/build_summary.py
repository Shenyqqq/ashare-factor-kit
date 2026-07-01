"""Build summary.md from lgbm_window_sweep artifacts."""
import json
import math
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def win_key(s: str):
    if not s:
        return (0, 0)
    a, b = s.split(",")
    return (int(a), int(b))


def top30_ann_from_nav(nav_path: Path) -> float | None:
    nav = pd.read_csv(nav_path, index_col=0, parse_dates=True)
    if "Top30" not in nav.columns:
        return None
    s = nav["Top30"].dropna()
    if len(s) < 2:
        return None
    total = s.iloc[-1] / s.iloc[0] - 1
    years = (s.index[-1] - s.index[0]).days / 365.25
    return round((1 + total) ** (1 / max(years, 0.1)) - 1, 4)


def monotonicity_from_annual(annual_path: Path) -> float | None:
    ann = pd.read_csv(annual_path, index_col=0)
    q_cols = sorted([c for c in ann.columns if re.match(r"Q[1-5]", str(c))])
    if len(q_cols) < 5:
        return None
    ranks = list(range(1, len(q_cols) + 1))
    yearly = []
    for _, row in ann.iterrows():
        rets = [row[c] for c in q_cols]
        if any(pd.isna(rets)):
            continue
        corr = pd.Series(ranks).corr(pd.Series(rets), method="spearman")
        if corr is not None and not math.isnan(corr):
            yearly.append(corr)
    if not yearly:
        return None
    return round(float(pd.Series(yearly).mean()), 4)


def parse_windows(tag: str) -> str:
    m = re.search(r"_w([\d-]+)", tag)
    return m.group(1).replace("-", ",") if m else ""


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return v


def main() -> None:
    log_failures = []
    log_path = BASE / "sweep.log"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Run failed:"):
                log_failures.append(line.replace("Run failed: ", ""))

    rows = []
    for mp in sorted(BASE.glob("model_metrics_lgbm_h*.json")):
        with open(mp, encoding="utf-8") as f:
            d = json.load(f, parse_constant=lambda x: float("nan") if x == "NaN" else None)
        tag = d.get("tag", mp.stem.replace("model_metrics_", ""))
        h = re.search(r"_h(\d+)", tag)
        horizon = int(h.group(1)) if h else None
        bt_nav = BASE / f"backtest_{tag}_nav.csv"
        bt_ann = BASE / f"backtest_{tag}_annual.csv"
        n_pred = d.get("预测期数") or 0
        status = "OK" if n_pred and bt_nav.exists() else "FAILED (no backtest)"
        mono = monotonicity_from_annual(bt_ann) if bt_ann.exists() else None
        top30 = top30_ann_from_nav(bt_nav) if bt_nav.exists() else None
        rows.append({
            "horizon": horizon,
            "windows (months)": parse_windows(tag),
            "IC mean": d.get("IC均值"),
            "ICIR": d.get("ICIR"),
            "IC>0%": d.get("IC>0胜率"),
            "monotonicity": mono,
            "top30 ann return": top30,
            "status": status,
        })
    rows.sort(key=lambda r: (r["horizon"] or 0, win_key(r["windows (months)"])))

    ok_count = sum(1 for r in rows if r["status"] == "OK")
    lines = [
        "# LGBM train-window sweep",
        "",
        "Batch: `results/lgbm_window_sweep/run_sweep.ps1` | log: `sweep.log`",
        "",
        f"**Completed with backtest:** {ok_count}/6 runs.",
        "",
        "| horizon | windows (months) | IC mean | ICIR | IC>0% | monotonicity | top30 ann return | status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['horizon']} | {r['windows (months)']} | {fmt(r['IC mean'])} | {fmt(r['ICIR'])} | "
            f"{fmt(r['IC>0%'])} | {fmt(r['monotonicity'])} | {fmt(r['top30 ann return'])} | {r['status']} |"
        )

    lines.extend(["", "## Failures / notes", ""])
    notes = [
        "Initial script used `--skip-factor-build` on run 4 before h20 factor cache existed (fixed: per-horizon skip).",
        "h20 + windows 3,6: Walk-Forward had 0 prediction dates; backtest failed (empty factor_scores).",
    ]
    for x in sorted(set(log_failures)):
        notes.append(x)
    for n in notes:
        lines.append(f"- {n}")

    lines.extend([
        "",
        "## Script fix",
        "",
        "`--skip-factor-build` applies after the first **successful** run per horizon (h5 / h20), not after run 1 globally.",
        "",
        "Monotonicity and top30 ann return come from backtest CSVs (not in `model_metrics_*.json`).",
    ])

    (BASE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary: {ok_count}/6 OK")


if __name__ == "__main__":
    main()
