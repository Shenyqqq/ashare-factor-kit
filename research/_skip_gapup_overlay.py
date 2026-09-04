"""一次性 overlay：模型 TopN 买入日「去高开」——不改生产引擎。

旗舰五日分数 + 与 mon_wed_hold 同一套 holdings / EW 宇宙。
规则（开盘后才知道）：买入日隔夜 open[entry]/close[entry-1]-1 ≥ 100bp 则剔除。
  A 剔除后等权剩余（篮子变小）
  B 从下一名补足到原 N
可选一行：再剔除低开 ≤ −300bp，补足。

持有：① 生产：持有到下个 W-FRI 信号  ② 实盘候选：open[t+1]→close[t+3]
成本：引擎公式 |Δ|/(2∪) × (佣金+印花+10bp half-spread)。overlay 是
close[sell]/open[buy] 截面等权，无涨跌停卡住/停牌日路径，属近似。

用法（仓库根）:
    .venv\\Scripts\\python.exe research/_skip_gapup_overlay.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.encoding_bootstrap import bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()

import numpy as np
import pandas as pd
from loguru import logger

from backtest.execution import BacktestConfig, total_cost_fraction
from backtest.risk_metrics import compute_risk_metrics
from backtest.turnover import compute_turnover
from backtest.portfolio import PortfolioState
from config.settings import RAW_DIR, RISK_FREE_RATE
from data.clean import clean_ohlc_aligned

OUT = ROOT / "results" / "lgbm_h5_nolongshare_w104_decay0_20260814"
TAG = "lgbm_h5_w104_p_sparse_rt4"
TXT = ROOT / "research" / "_skip_gapup_overlay_out.txt"
BT_FREQ = "W-FRI"
GAP_UP_BP = 100.0
GAP_DN_BP = 300.0
NS = (100, 30)
HOLDS = ("to_next", "mon_wed")  # 生产 / 实盘候选
FILTERS = ("raw", "skip_refill", "skip_norefill", "skip_dn_refill")


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except UnicodeEncodeError:
                f.write(data.encode("utf-8", errors="replace").decode("utf-8"))
            try:
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            f.flush()


def _parse_holdings(path: Path) -> dict[pd.Timestamp, list[str]]:
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        dt = pd.Timestamp(r["信号日"])
        names = [x.strip().zfill(6) for x in str(r["标的"]).split("|") if x.strip()]
        out[dt] = names
    return out


def _reconstruct_universe(path: Path, group: str) -> dict[pd.Timestamp, list[str]]:
    td = pd.read_csv(path)
    g = td[td["group"] == group].copy()
    g["signal_date"] = pd.to_datetime(g["signal_date"])
    g = g.sort_values("signal_date")
    holdings: set[str] = set()
    out: dict[pd.Timestamp, list[str]] = {}
    for _, r in g.iterrows():
        sells = {
            x.strip().zfill(6)
            for x in str(r["sells"]).split("|")
            if x.strip() and x.strip().lower() != "nan"
        }
        buys = {
            x.strip().zfill(6)
            for x in str(r["buys"]).split("|")
            if x.strip() and x.strip().lower() != "nan"
        }
        holdings = (holdings - sells) | buys
        out[pd.Timestamp(r["signal_date"])] = sorted(holdings)
    return out


def _year_prod(rets: pd.Series) -> pd.Series:
    out = {}
    for year, grp in rets.groupby(rets.index.year):
        out[int(year)] = float((1.0 + grp.fillna(0.0)).prod() - 1.0)
    return pd.Series(out)


def _metrics(port: pd.Series, bm: pd.Series, col: str = "port") -> dict:
    aligned_bm = bm.reindex(port.index)
    nav = pd.DataFrame(
        {
            col: (1.0 + port.fillna(0.0)).cumprod(),
            "benchmark": (1.0 + aligned_bm.fillna(0.0)).cumprod(),
        }
    )
    rm = compute_risk_metrics(nav, rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE)
    y_port = _year_prod(port)
    y_bm = _year_prod(aligned_bm.fillna(0.0))
    w = port.loc["2022-01-01":"2024-12-31"]
    b = aligned_bm.reindex(w.index)
    if w.empty:
        w22 = {"ann": np.nan, "excess": np.nan, "sharpe": np.nan}
    else:
        w22 = _metrics_ann(w, b, col)
    return {
        "n_periods": int(len(port)),
        "ann": float(rm.loc[col, "年化收益"]),
        "sharpe": float(rm.loc[col, "Sharpe"]),
        "mdd": float(rm.loc[col, "最大回撤"]),
        "bm_ann": float(rm.loc["benchmark", "年化收益"]),
        "excess": float(rm.loc[col, "年化收益"] - rm.loc["benchmark", "年化收益"]),
        "y2025": float(y_port.get(2025, np.nan)),
        "y2026": float(y_port.get(2026, np.nan)),
        "bm_y2025": float(y_bm.get(2025, np.nan)),
        "bm_y2026": float(y_bm.get(2026, np.nan)),
        "ann_2224": float(w22["ann"]),
        "xs_2224": float(w22["excess"]),
        "sh_2224": float(w22["sharpe"]),
        "y2022": float(y_port.get(2022, np.nan)),
        "y2023": float(y_port.get(2023, np.nan)),
        "y2024": float(y_port.get(2024, np.nan)),
    }


def _metrics_ann(port: pd.Series, bm: pd.Series, col: str) -> dict:
    aligned_bm = bm.reindex(port.index)
    nav = pd.DataFrame(
        {
            col: (1.0 + port.fillna(0.0)).cumprod(),
            "benchmark": (1.0 + aligned_bm.fillna(0.0)).cumprod(),
        }
    )
    rm = compute_risk_metrics(nav, rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE)
    return {
        "ann": float(rm.loc[col, "年化收益"]),
        "excess": float(rm.loc[col, "年化收益"] - rm.loc["benchmark", "年化收益"]),
        "sharpe": float(rm.loc[col, "Sharpe"]),
    }


def _ew_px_ret(
    names: list[str],
    den_panel: pd.DataFrame,
    den_dt: pd.Timestamp,
    num_panel: pd.DataFrame,
    num_dt: pd.Timestamp,
) -> tuple[float, int, pd.Series]:
    cols = [c for c in names if c in den_panel.columns and c in num_panel.columns]
    if not cols or den_dt not in den_panel.index or num_dt not in num_panel.index:
        return float("nan"), 0, pd.Series(dtype=float)
    den = den_panel.loc[den_dt, cols].astype(float)
    num = num_panel.loc[num_dt, cols].astype(float)
    r = (num / den - 1.0).replace([np.inf, -np.inf], np.nan)
    r = r[np.isfinite(r)]
    if r.empty:
        return float("nan"), 0, r
    return float(r.mean()), int(len(r)), r


def _apply_cost(gross: float, cost: float) -> float:
    if not np.isfinite(gross):
        return 0.0
    c = 0.0 if not np.isfinite(cost) else float(cost)
    return float((1.0 + gross) * (1.0 - c) - 1.0)


def _overnight(
    names: list[str],
    open_: pd.DataFrame,
    close: pd.DataFrame,
    entry: pd.Timestamp,
    prev: pd.Timestamp,
) -> pd.Series:
    cols = [c for c in names if c in open_.columns and c in close.columns]
    if not cols or entry not in open_.index or prev not in close.index:
        return pd.Series(dtype=float)
    o = open_.loc[entry, cols].astype(float)
    c = close.loc[prev, cols].astype(float)
    g = (o / c - 1.0).replace([np.inf, -np.inf], np.nan)
    return g


def _pass_gap(gap: pd.Series, *, skip_up: bool, skip_dn: bool) -> pd.Series:
    ok = gap.notna() & np.isfinite(gap)
    if skip_up:
        ok &= gap < (GAP_UP_BP / 1e4)
    if skip_dn:
        ok &= gap > -(GAP_DN_BP / 1e4)
    return ok


def _pick(
    ranked: list[str],
    gap: pd.Series,
    n: int,
    *,
    skip_up: bool,
    skip_dn: bool,
    refill: bool,
) -> tuple[list[str], list[str], list[str]]:
    """返回 (篮子, 踢掉的原TopN, 补进来的)。"""
    top = ranked[:n]
    g_top = gap.reindex(top)
    keep_mask = _pass_gap(g_top, skip_up=skip_up, skip_dn=skip_dn)
    kept = [s for s in top if bool(keep_mask.get(s, False))]
    kicked = [s for s in top if s not in set(kept)]
    if not skip_up and not skip_dn:
        return top, [], []
    if not refill:
        return kept, kicked, []
    need = n - len(kept)
    filled: list[str] = []
    if need > 0:
        rest = ranked[n:]
        g_rest = gap.reindex(rest)
        ok_rest = _pass_gap(g_rest, skip_up=skip_up, skip_dn=skip_dn)
        for s in rest:
            if not bool(ok_rest.get(s, False)):
                continue
            filled.append(s)
            if len(filled) >= need:
                break
    return kept + filled, kicked, filled


def _ranked_pool(
    holdings_n100: list[str],
    scores_row: pd.Series,
    universe: set[str],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in holdings_n100:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    sc = scores_row.dropna()
    idx = [c for c in sc.index if c in universe and c not in seen]
    if not idx:
        return out
    tmp = sc.loc[idx].rename("score").to_frame()
    tmp["_code"] = tmp.index.astype(str)
    ranked = tmp.sort_values(["score", "_code"], ascending=[False, True])
    out.extend(list(ranked.index.astype(str)))
    return out


def _fmt_row(label: str, m: dict) -> str:
    def pct(x, nd=1):
        if x is None or not np.isfinite(x):
            return "    —"
        return f"{x*100:6.1f}%"

    def pp(x):
        if x is None or not np.isfinite(x):
            return "    —"
        return f"{x*100:6.1f}pp"

    def sh(x):
        if x is None or not np.isfinite(x):
            return "   —"
        return f"{x:5.2f}"

    return (
        f"{label:<28} {pct(m['ann'])} {pct(m['bm_ann'])} {pp(m['excess'])} "
        f"{sh(m['sharpe'])}  {pct(m['ann_2224'])} {pp(m['xs_2224'])}  "
        f"{pct(m['y2025'])} {pct(m['y2026'])}"
    )


def main() -> None:
    out_f = open(TXT, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out_f)

    holdings = _parse_holdings(OUT / f"holdings_top100_{TAG}.csv")
    sigs = pd.DatetimeIndex(sorted(holdings))
    universe = _reconstruct_universe(OUT / f"turnover_detail_{TAG}.csv", "benchmark")
    scores = pd.read_parquet(OUT / f"factor_scores_{TAG}.parquet")
    scores.index = pd.to_datetime(scores.index)
    scores.columns = scores.columns.astype(str).str.zfill(6)

    close_raw = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    open_raw = pd.read_parquet(RAW_DIR / "open_hfq.parquet")
    close, open_, _, _ = clean_ohlc_aligned(close_raw, open_raw, None, None)
    close.columns = close.columns.astype(str).str.zfill(6)
    open_.columns = open_.columns.astype(str).str.zfill(6)
    cal = pd.DatetimeIndex(close.index).sort_values()
    pos = {d: i for i, d in enumerate(cal)}

    cfg = BacktestConfig(bid_ask_spread_bps=10.0)
    roundtrip = total_cost_fraction(1.0, cfg)
    logger.info(
        f"periods={len(sigs)} {sigs.min().date()}→{sigs.max().date()}  "
        f"full-RT cost={roundtrip:.6f}"
    )

    # key = (n, filt, hold) → list of (t, gross, names, n_kick, n_fill, kicked, filled)
    baskets: dict[tuple, list] = defaultdict(list)
    bm_rows: dict[str, list] = {h: [] for h in HOLDS}
    kick_stats = []

    skipped = 0
    for i, t in enumerate(sigs):
        if t not in pos:
            skipped += 1
            continue
        loc = pos[t]
        if loc + 3 >= len(cal):
            skipped += 1
            continue
        t1 = cal[loc + 1]
        t3 = cal[loc + 3]
        if i + 1 < len(sigs):
            next_sig = sigs[i + 1]
        else:
            skipped += 1
            continue
        if next_sig not in pos:
            skipped += 1
            continue

        names100 = holdings[t]
        univ = set(universe.get(pd.Timestamp(t), []))
        if t in scores.index:
            sc_row = scores.loc[t]
            if isinstance(sc_row, pd.DataFrame):
                sc_row = sc_row.iloc[0]
        else:
            sc_row = pd.Series(dtype=float)
        ranked = _ranked_pool(names100, sc_row, univ)

        # gap on a generous candidate set (Top100 + next 200)
        cand = ranked[: max(400, len(names100) + 200)]
        gap = _overnight(cand, open_, close, t1, t)

        sell = {"to_next": next_sig, "mon_wed": t3}
        bm_names = universe.get(pd.Timestamp(t), [])
        for hold, sell_dt in sell.items():
            g, n, _ = _ew_px_ret(bm_names, open_, t1, close, sell_dt)
            bm_rows[hold].append((t, g, n))

        for n in NS:
            specs = [
                ("raw", False, False, True),
                ("skip_refill", True, False, True),
                ("skip_norefill", True, False, False),
                ("skip_dn_refill", True, True, True),
            ]
            for filt, skip_up, skip_dn, refill in specs:
                picked, kicked, filled = _pick(
                    ranked, gap, n, skip_up=skip_up, skip_dn=skip_dn, refill=refill,
                )
                for hold, sell_dt in sell.items():
                    g, n_ok, r_ser = _ew_px_ret(picked, open_, t1, close, sell_dt)
                    baskets[(n, filt, hold)].append(
                        {
                            "t": t,
                            "gross": g,
                            "names": picked,
                            "n_ok": n_ok,
                            "n_kick": len(kicked),
                            "n_fill": len(filled),
                            "kicked": kicked,
                            "filled": filled,
                        }
                    )
                if filt != "raw":
                    kick_g, _, kick_r = _ew_px_ret(kicked, open_, t1, close, t3)
                    fill_g, _, fill_r = _ew_px_ret(filled, open_, t1, close, t3)
                    keep_g, _, _ = _ew_px_ret(
                        [s for s in names100[:n] if s not in set(kicked)],
                        open_, t1, close, t3,
                    )
                    kick_stats.append(
                        {
                            "t": t,
                            "n": n,
                            "filt": filt,
                            "n_kick": len(kicked),
                            "n_fill": len(filled),
                            "kick_3d": kick_g,
                            "fill_3d": fill_g,
                            "keep_3d": keep_g,
                            "kick_gap": float(gap.reindex(kicked).mean())
                            if kicked
                            else np.nan,
                            "fill_gap": float(gap.reindex(filled).mean())
                            if filled
                            else np.nan,
                        }
                    )

    logger.info(f"skipped signal dates={skipped}")

    # apply cost via engine turnover of consecutive baskets
    rets: dict[tuple, pd.Series] = {}
    cost_mean: dict[tuple, float] = {}
    n_mean: dict[tuple, float] = {}
    kick_mean: dict[tuple, float] = {}
    fill_mean: dict[tuple, float] = {}
    for key, rows in baskets.items():
        prev = PortfolioState()
        net = []
        idx = []
        costs = []
        ns, kicks, fills = [], [], []
        for row in rows:
            new = PortfolioState(holdings=frozenset(row["names"]))
            to = compute_turnover(prev, new)
            c = total_cost_fraction(to, cfg) if row["names"] else 0.0
            net.append(_apply_cost(row["gross"], c))
            idx.append(row["t"])
            costs.append(c)
            ns.append(len(row["names"]))
            kicks.append(row["n_kick"])
            fills.append(row["n_fill"])
            prev = new
        rets[key] = pd.Series(net, index=pd.DatetimeIndex(idx), dtype=float)
        cost_mean[key] = float(np.mean(costs)) if costs else np.nan
        n_mean[key] = float(np.mean(ns)) if ns else np.nan
        kick_mean[key] = float(np.mean(kicks)) if kicks else np.nan
        fill_mean[key] = float(np.mean(fills)) if fills else np.nan

    bm: dict[str, pd.Series] = {}
    prev_bm = PortfolioState()
    for hold in HOLDS:
        net, idx = [], []
        prev_bm = PortfolioState()
        for t, g, _n in bm_rows[hold]:
            names = universe.get(pd.Timestamp(t), [])
            new = PortfolioState(holdings=frozenset(names))
            to = compute_turnover(prev_bm, new)
            c = total_cost_fraction(to, cfg)
            net.append(_apply_cost(g, c))
            idx.append(t)
            prev_bm = new
        bm[hold] = pd.Series(net, index=pd.DatetimeIndex(idx), dtype=float)

    filt_label = {
        "raw": "不过滤",
        "skip_refill": "去高开(补足)",
        "skip_norefill": "去高开(不补)",
        "skip_dn_refill": "去高开+深低开(补)",
    }
    hold_label = {"to_next": "五日持有(到下信号)", "mon_wed": "三天持有(一开→三收)"}

    print("=" * 108)
    print("模型 TopN × 去高开 overlay（不改生产引擎）")
    print("=" * 108)
    print(f"分数/持仓 : {OUT.name}")
    print(f"样本     : {sigs.min().date()} → {sigs.max().date()}  期数={len(next(iter(rets.values())))}")
    print("隔夜     : open[entry]/close[entry-1]-1，entry=信号日下一交易日（通常周一）")
    print(f"主规则   : 高开 ≥ {GAP_UP_BP:.0f}bp 剔除；对照 A 不补 / B 补足到原 N")
    print(f"可选     : 再剔除低开 ≤ −{GAP_DN_BP:.0f}bp，补足")
    print("EW 基准  : 不过滤的同一宇宙（生产 benchmark 重建），过滤后不换基准")
    print(
        "成本     : overlay 截面 close[sell]/open[buy] 等权后扣 "
        "引擎 total_cost_fraction(|Δ|/(2∪)，佣金+印花+10bp)。"
    )
    print(
        "          无一字板卡住/停牌日路径；三天持有按周频换篮扣费"
        "（未额外把重叠票当周四周五空仓全卖）。"
    )
    print()
    hdr = (
        f"{'组合':<28} {'年化':>7} {'EW':>7} {'超额':>8} {'Sh':>5}  "
        f"{'22–24年化':>8} {'22–24超额':>9}  {'2025':>7} {'2026':>7}"
    )
    print(hdr)
    print("-" * 108)

    summary_rows = []
    for n in NS:
        for hold in HOLDS:
            print(f"\n--- Top{n}  ×  {hold_label[hold]} ---")
            for filt in FILTERS:
                key = (n, filt, hold)
                if key not in rets:
                    continue
                m = _metrics(rets[key], bm[hold], col=f"Top{n}")
                lab = f"Top{n} {filt_label[filt]}"
                print(_fmt_row(lab, m))
                print(
                    f"{'':28}   期均N={n_mean[key]:.1f}  踢={kick_mean[key]:.2f}  "
                    f"补={fill_mean[key]:.2f}  期均成本={cost_mean[key]*1e4:.2f}bp"
                )
                summary_rows.append(
                    {
                        **m,
                        "n": n,
                        "hold": hold,
                        "filt": filt,
                        "label": lab,
                        "n_mean": n_mean[key],
                        "kick_mean": kick_mean[key],
                        "fill_mean": fill_mean[key],
                        "cost_bp": cost_mean[key] * 1e4,
                    }
                )

    ks = pd.DataFrame(kick_stats)
    print("\n" + "=" * 108)
    print("踢出 / 补进（三天持有毛收益，同周；仅有踢出的期）")
    print("=" * 108)
    for n in NS:
        for filt in ("skip_refill", "skip_norefill", "skip_dn_refill"):
            sub = ks[(ks["n"] == n) & (ks["filt"] == filt)]
            if sub.empty:
                continue
            with_k = sub[sub["n_kick"] > 0]
            print(
                f"Top{n} {filt_label[filt]}: 期均踢 {sub['n_kick'].mean():.2f}  "
                f"期均补 {sub['n_fill'].mean():.2f}  "
                f"有踢出的期 {len(with_k)}/{len(sub)}"
            )
            if with_k.empty:
                continue
            kg = with_k["kick_3d"].dropna()
            fg = with_k["fill_3d"].dropna()
            kp = with_k["keep_3d"].dropna()
            if len(kg):
                print(
                    f"  被踢票 3d 期均 {kg.mean()*1e4:+.1f}bp  "
                    f"胜率 {(kg>0).mean()*100:.1f}%  n={len(kg)}"
                )
            if len(kp):
                print(
                    f"  原篮留下 3d 期均 {kp.mean()*1e4:+.1f}bp  "
                    f"胜率 {(kp>0).mean()*100:.1f}%  n={len(kp)}"
                )
            if len(fg):
                print(
                    f"  补进票 3d 期均 {fg.mean()*1e4:+.1f}bp  "
                    f"胜率 {(fg>0).mean()*100:.1f}%  n={len(fg)}"
                )
            if len(kg) and len(fg):
                both = with_k.dropna(subset=["kick_3d", "fill_3d"])
                if len(both):
                    d = both["fill_3d"] - both["kick_3d"]
                    print(
                        f"  补−踢  期均 {d.mean()*1e4:+.1f}bp  "
                        f"补>踢 {(d>0).mean()*100:.1f}%"
                    )

    # decision helper: refill excess vs raw, and 2022-24 not worse
    print("\n" + "=" * 108)
    print("是否写入引擎（规则：补足后超额稳定好于不过滤，且 2022–24 不更差）")
    print("=" * 108)
    for n in NS:
        for hold in HOLDS:
            raw = next(
                r for r in summary_rows if r["n"] == n and r["hold"] == hold and r["filt"] == "raw"
            )
            ref = next(
                r
                for r in summary_rows
                if r["n"] == n and r["hold"] == hold and r["filt"] == "skip_refill"
            )
            better = ref["excess"] > raw["excess"] + 1e-4
            not_worse_2224 = ref["xs_2224"] >= raw["xs_2224"] - 1e-4
            flag = "过" if (better and not_worse_2224) else "不过"
            print(
                f"Top{n} {hold_label[hold]}: 全样本超额 {raw['excess']*100:.1f}→"
                f"{ref['excess']*100:.1f}pp  ({'更好' if better else '未更好'})  "
                f"22–24超额 {raw['xs_2224']*100:.1f}→{ref['xs_2224']*100:.1f}pp  "
                f"({ '不更差' if not_worse_2224 else '更差' })  → {flag}"
            )

    pd.DataFrame(summary_rows).to_csv(
        OUT / "skip_gapup_overlay_summary.csv", index=False, encoding="utf-8-sig"
    )
    ks.to_csv(OUT / "skip_gapup_overlay_kicks.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUT / 'skip_gapup_overlay_summary.csv'}")
    print(f"wrote {TXT}")
    out_f.close()


if __name__ == "__main__":
    main()
