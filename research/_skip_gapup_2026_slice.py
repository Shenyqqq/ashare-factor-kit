"""2026 切片：去高开 overlay 是否只在 H1 失效、6 月后回暖。

复用 research/_skip_gapup_overlay.py 同一套规则/成本；不改引擎。
归属按**信号日 t**（W-FRI）；买入日 entry=t+1（通常周一）。
H1 = 信号日 2026-01-01…05-31；Jun+ = 信号日 ≥2026-06-01；样本末=最后一期信号。
"""
from __future__ import annotations

import importlib.util
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
from backtest.portfolio import PortfolioState
from backtest.turnover import compute_turnover
from config.settings import RAW_DIR
from data.clean import clean_ohlc_aligned

OUT = ROOT / "results" / "lgbm_h5_nolongshare_w104_decay0_20260814"
TAG = "lgbm_h5_w104_p_sparse_rt4"
TXT = ROOT / "research" / "_skip_gapup_2026_slice_out.txt"
PPY = 52
NS = (100, 30)
HOLDS = ("to_next", "mon_wed")
FILTERS = ("raw", "skip_norefill", "skip_refill")  # 不补为主，补足一行


def _load_overlay():
    p = ROOT / "research" / "_skip_gapup_overlay.py"
    spec = importlib.util.spec_from_file_location("skip_gapup_overlay", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
            try:
                f.flush()
            except Exception:
                pass


def _cum(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if x.empty:
        return float("nan")
    return float((1.0 + x).prod() - 1.0)


def _ann(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    n = int(len(x))
    if n == 0:
        return float("nan")
    c = float((1.0 + x).prod())
    if c <= 0:
        return float("nan")
    return float(c ** (PPY / n) - 1.0)


def _mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.mean()) if len(x) else float("nan")


def _fmt_pct(x, nd=1):
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x*100:{nd+5}.{nd}f}%"


def _fmt_pp(x, nd=1):
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x*100:{nd+5}.{nd}f}pp"


def _fmt_bp(x):
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x*1e4:+7.1f}bp"


def main() -> None:
    sg = _load_overlay()
    out_f = open(TXT, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out_f)

    holdings = sg._parse_holdings(OUT / f"holdings_top100_{TAG}.csv")
    sigs = pd.DatetimeIndex(sorted(holdings))
    universe = sg._reconstruct_universe(OUT / f"turnover_detail_{TAG}.csv", "benchmark")
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

    last_2025 = sigs[sigs < "2026-01-01"].max()
    use_sigs = sigs[sigs >= last_2025]  # 2025 末周只为换手链
    logger.info(f"warmup last_2025={last_2025.date()}  through {sigs.max().date()}  n={len(use_sigs)}")

    baskets: dict[tuple, list] = defaultdict(list)
    bm_rows: dict[str, list] = {h: [] for h in HOLDS}
    kick_rows = []
    meta_rows = []

    for i, t in enumerate(sigs):
        if t < last_2025:
            continue
        if t not in pos:
            continue
        loc = pos[t]
        if loc + 3 >= len(cal):
            continue
        t1 = cal[loc + 1]
        t3 = cal[loc + 3]
        # next signal in full calendar (not just use_sigs)
        nxt = sigs[sigs > t]
        if len(nxt) == 0:
            continue
        next_sig = nxt[0]
        if next_sig not in pos:
            continue

        names100 = holdings[t]
        univ = set(universe.get(pd.Timestamp(t), []))
        if t in scores.index:
            sc_row = scores.loc[t]
            if isinstance(sc_row, pd.DataFrame):
                sc_row = sc_row.iloc[0]
        else:
            sc_row = pd.Series(dtype=float)
        ranked = sg._ranked_pool(names100, sc_row, univ)
        cand = ranked[: max(400, len(names100) + 200)]
        gap = sg._overnight(cand, open_, close, t1, t)
        sell = {"to_next": next_sig, "mon_wed": t3}
        bm_names = universe.get(pd.Timestamp(t), [])
        for hold, sell_dt in sell.items():
            g, n_ok, _ = sg._ew_px_ret(bm_names, open_, t1, close, sell_dt)
            bm_rows[hold].append((t, g, n_ok))

        meta_rows.append(
            {
                "t": t,
                "entry": t1,
                "t3": t3,
                "next_sig": next_sig,
                "sig_month": f"{t.year:04d}-{t.month:02d}",
                "entry_month": f"{t1.year:04d}-{t1.month:02d}",
                "entry_wd": int(t1.dayofweek),
            }
        )

        for n in NS:
            for filt, skip_up, refill in (
                ("raw", False, True),
                ("skip_norefill", True, False),
                ("skip_refill", True, True),
            ):
                picked, kicked, filled = sg._pick(
                    ranked, gap, n, skip_up=skip_up, skip_dn=False, refill=refill,
                )
                for hold, sell_dt in sell.items():
                    g, n_ok, _ = sg._ew_px_ret(picked, open_, t1, close, sell_dt)
                    baskets[(n, filt, hold)].append(
                        {
                            "t": t,
                            "gross": g,
                            "names": picked,
                            "n_kick": len(kicked),
                            "n_fill": len(filled),
                        }
                    )
                kick_3d, _, _ = sg._ew_px_ret(kicked, open_, t1, close, t3)
                keep_3d, _, _ = sg._ew_px_ret(
                    [s for s in names100[:n] if s not in set(kicked)],
                    open_, t1, close, t3,
                )
                fill_3d, _, _ = sg._ew_px_ret(filled, open_, t1, close, t3)
                kick_5d, _, _ = sg._ew_px_ret(kicked, open_, t1, close, next_sig)
                keep_5d, _, _ = sg._ew_px_ret(
                    [s for s in names100[:n] if s not in set(kicked)],
                    open_, t1, close, next_sig,
                )
                fill_5d, _, _ = sg._ew_px_ret(filled, open_, t1, close, next_sig)
                if filt != "raw":
                    kick_rows.append(
                        {
                            "t": t,
                            "n": n,
                            "filt": filt,
                            "n_kick": len(kicked),
                            "n_fill": len(filled),
                            "kick_3d": kick_3d,
                            "keep_3d": keep_3d,
                            "fill_3d": fill_3d,
                            "kick_5d": kick_5d,
                            "keep_5d": keep_5d,
                            "fill_5d": fill_5d,
                        }
                    )

    rets: dict[tuple, pd.Series] = {}
    kicks_s: dict[tuple, pd.Series] = {}
    for key, rows in baskets.items():
        prev = PortfolioState()
        net, idx, nk = [], [], []
        for row in rows:
            new = PortfolioState(holdings=frozenset(row["names"]))
            to = compute_turnover(prev, new)
            c = total_cost_fraction(to, cfg) if row["names"] else 0.0
            net.append(sg._apply_cost(row["gross"], c))
            idx.append(row["t"])
            nk.append(row["n_kick"])
            prev = new
        rets[key] = pd.Series(net, index=pd.DatetimeIndex(idx), dtype=float)
        kicks_s[key] = pd.Series(nk, index=pd.DatetimeIndex(idx), dtype=float)

    bm: dict[str, pd.Series] = {}
    for hold in HOLDS:
        prev = PortfolioState()
        net, idx = [], []
        for t, g, _n in bm_rows[hold]:
            names = universe.get(pd.Timestamp(t), [])
            new = PortfolioState(holdings=frozenset(names))
            to = compute_turnover(prev, new)
            c = total_cost_fraction(to, cfg)
            net.append(sg._apply_cost(g, c))
            idx.append(t)
            prev = new
        bm[hold] = pd.Series(net, index=pd.DatetimeIndex(idx), dtype=float)

    meta = pd.DataFrame(meta_rows).set_index("t")
    ks = pd.DataFrame(kick_rows)
    # drop warmup week from reporting
    for k in list(rets):
        rets[k] = rets[k].loc[rets[k].index >= "2026-01-01"]
        kicks_s[k] = kicks_s[k].loc[kicks_s[k].index >= "2026-01-01"]
    for hold in HOLDS:
        bm[hold] = bm[hold].loc[bm[hold].index >= "2026-01-01"]
    meta = meta.loc[meta.index >= "2026-01-01"]
    ks = ks[ks["t"] >= "2026-01-01"]

    last_sig = meta.index.max()
    last_entry = meta["entry"].max()
    n_mismatch = int((meta["sig_month"] != meta["entry_month"]).sum())
    hold_label = {"to_next": "五日(到下信号)", "mon_wed": "三天(一开→三收)"}
    filt_label = {
        "raw": "不过滤",
        "skip_norefill": "去高开不补",
        "skip_refill": "去高开补足",
    }

    windows = [
        ("2026全年", "2026-01-01", "2026-12-31"),
        ("H1(1–5月)", "2026-01-01", "2026-05-31"),
        ("6月及之后", "2026-06-01", "2026-12-31"),
    ]

    print("=" * 100)
    print("2026 去高开切片（同一套 overlay，不改引擎）")
    print("=" * 100)
    print(f"旗舰     : {OUT.name}")
    print(f"归属     : 信号日 t（W-FRI）；买入日 entry=下一交易日（通常周一）")
    print(f"样本末   : 最后信号 {last_sig.date()}，最后买入 {pd.Timestamp(last_entry).date()}")
    print(f"信号月≠买入月: {n_mismatch}/{len(meta)} 期（按信号日切月/H1）")
    print(f"H1       : 信号日 2026-01-01 … 2026-05-31")
    print(f"6月及之后 : 信号日 ≥ 2026-06-01 至样本末")
    print("成本     : 与上次 overlay 相同（2025 末周作换手链，不计入表）")
    print("短窗     : 先看累计；年化=52 期几何，6–7 月只有几周，仅作对照")
    print()

    # list weeks
    h1 = meta.loc["2026-01-01":"2026-05-31"]
    jp = meta.loc["2026-06-01":]
    print(f"H1 期数={len(h1)}  首末信号 {h1.index.min().date()} … {h1.index.max().date()}")
    print(f"Jun+ 期数={len(jp)}  信号日: {', '.join(d.strftime('%m-%d') for d in jp.index)}")
    print()

    summary = []
    print(
        f"{'窗口':<12} {'组合':<22} {'n':>3} {'累计':>8} {'年化':>8} "
        f"{'EW累计':>8} {'vsEW':>8} {'vs不过滤累计':>12} {'vs不过滤年化':>12} {'周均踢':>6}"
    )
    print("-" * 100)
    for wname, lo, hi in windows:
        for n in NS:
            for hold in HOLDS:
                raw = rets[(n, "raw", hold)].loc[lo:hi]
                b = bm[hold].reindex(raw.index)
                for filt in FILTERS:
                    s = rets[(n, filt, hold)].loc[lo:hi]
                    nk = kicks_s[(n, filt, hold)].loc[lo:hi]
                    row = {
                        "window": wname,
                        "n": n,
                        "hold": hold,
                        "filt": filt,
                        "n_weeks": int(len(s)),
                        "cum": _cum(s),
                        "ann": _ann(s),
                        "ew_cum": _cum(b),
                        "ew_ann": _ann(b),
                        "xs_ann": _ann(s) - _ann(b),
                        "vs_raw_cum": _cum(s) - _cum(raw),
                        "vs_raw_ann": _ann(s) - _ann(raw),
                        "kick_mean": float(nk.mean()) if filt != "raw" else 0.0,
                    }
                    summary.append(row)
                    lab = f"Top{n} {hold_label[hold][:2]} {filt_label[filt]}"
                    print(
                        f"{wname:<12} {lab:<22} {row['n_weeks']:3d} "
                        f"{_fmt_pct(row['cum'])} {_fmt_pct(row['ann'])} "
                        f"{_fmt_pct(row['ew_cum'])} {_fmt_pp(row['xs_ann'])} "
                        f"{_fmt_pp(row['vs_raw_cum'])} {_fmt_pp(row['vs_raw_ann'])} "
                        f"{row['kick_mean']:6.2f}"
                    )
                print()

    # monthly: unfiltered vs no-refill, Top30 and Top100, both holds
    print("=" * 100)
    print("2026 月度（按信号日月份；主看去高开不补 − 不过滤）")
    print("=" * 100)
    months = sorted(meta["sig_month"].unique())
    monthly_rows = []
    print(
        f"{'月':<8} {'n周':>3} {'Top30三天不过滤':>14} {'不补':>8} {'差额':>8} {'踢':>5}  "
        f"{'Top30五日不过滤':>14} {'不补':>8} {'差额':>8}  "
        f"{'Top100三天差额':>12} {'Top100五日差额':>12}"
    )
    print("-" * 100)
    for m in months:
        idx = meta.index[meta["sig_month"] == m]
        n_w = len(idx)
        def mc(n, filt, hold):
            return _cum(rets[(n, filt, hold)].reindex(idx))

        def mk(n, hold):
            return float(kicks_s[(n, "skip_norefill", hold)].reindex(idx).mean())

        t30_3_raw, t30_3_nr = mc(30, "raw", "mon_wed"), mc(30, "skip_norefill", "mon_wed")
        t30_5_raw, t30_5_nr = mc(30, "raw", "to_next"), mc(30, "skip_norefill", "to_next")
        t100_3_d = mc(100, "skip_norefill", "mon_wed") - mc(100, "raw", "mon_wed")
        t100_5_d = mc(100, "skip_norefill", "to_next") - mc(100, "raw", "to_next")
        print(
            f"{m:<8} {n_w:3d} {_fmt_pct(t30_3_raw)} {_fmt_pct(t30_3_nr)} "
            f"{_fmt_pp(t30_3_nr - t30_3_raw)} {mk(30,'mon_wed'):5.1f}  "
            f"{_fmt_pct(t30_5_raw)} {_fmt_pct(t30_5_nr)} "
            f"{_fmt_pp(t30_5_nr - t30_5_raw)}  "
            f"{_fmt_pp(t100_3_d)} {_fmt_pp(t100_5_d)}"
        )
        monthly_rows.append(
            {
                "month": m,
                "n_weeks": n_w,
                "signals": "|".join(d.strftime("%Y-%m-%d") for d in idx),
                "top30_3d_raw": t30_3_raw,
                "top30_3d_norefill": t30_3_nr,
                "top30_3d_diff": t30_3_nr - t30_3_raw,
                "top30_3d_kick": mk(30, "mon_wed"),
                "top30_5d_raw": t30_5_raw,
                "top30_5d_norefill": t30_5_nr,
                "top30_5d_diff": t30_5_nr - t30_5_raw,
                "top100_3d_diff": t100_3_d,
                "top100_5d_diff": t100_5_d,
                "top100_3d_kick": mk(100, "mon_wed"),
            }
        )

    print()
    print("=" * 100)
    print("被踢高开票 vs 留下的票（有踢出的周；毛收益，未扣费）")
    print("=" * 100)
    print("差 = 踢 − 留；负号 = 高开更弱（过滤器同号有效）；翻号 = 高开更强")
    kwin = [
        ("2026全年", "2026-01-01", "2026-12-31"),
        ("H1(1–5月)", "2026-01-01", "2026-05-31"),
        ("6月及之后", "2026-06-01", "2026-12-31"),
    ]
    print(
        f"{'窗口':<12} {'篮':>5} {'有踢周':>6} {'周均踢':>6}  "
        f"{'踢3d':>9} {'留3d':>9} {'踢-留3d':>9}  "
        f"{'踢5d':>9} {'留5d':>9} {'踢-留5d':>9}  3d翻号?"
    )
    print("-" * 100)
    kick_sum = []
    for wname, lo, hi in kwin:
        for n in NS:
            sub = ks[(ks["n"] == n) & (ks["filt"] == "skip_norefill")]
            sub = sub[(sub["t"] >= lo) & (sub["t"] <= hi)]
            wk = sub[sub["n_kick"] > 0]
            d3 = (wk["kick_3d"] - wk["keep_3d"]).dropna()
            d5 = (wk["kick_5d"] - wk["keep_5d"]).dropna()
            flip3 = "是(高开更强)" if (len(d3) and float(d3.mean()) > 0) else "否"
            print(
                f"{wname:<12} Top{n:<3} {len(wk):6d} {sub['n_kick'].mean():6.2f}  "
                f"{_fmt_bp(_mean(wk['kick_3d']))} {_fmt_bp(_mean(wk['keep_3d']))} "
                f"{_fmt_bp(_mean(d3) if len(d3) else np.nan)}  "
                f"{_fmt_bp(_mean(wk['kick_5d']))} {_fmt_bp(_mean(wk['keep_5d']))} "
                f"{_fmt_bp(_mean(d5) if len(d5) else np.nan)}  {flip3}"
            )
            kick_sum.append(
                {
                    "window": wname,
                    "n": n,
                    "n_kick_weeks": len(wk),
                    "kick_mean": float(sub["n_kick"].mean()) if len(sub) else np.nan,
                    "kick_3d": _mean(wk["kick_3d"]),
                    "keep_3d": _mean(wk["keep_3d"]),
                    "d3": _mean(d3) if len(d3) else np.nan,
                    "kick_5d": _mean(wk["kick_5d"]),
                    "keep_5d": _mean(wk["keep_5d"]),
                    "d5": _mean(d5) if len(d5) else np.nan,
                }
            )

    # one-line verdict helpers
    print()
    print("=" * 100)
    print("对照：6 月后 去高开不补 相对不过滤（累计差；正=增厚）")
    print("=" * 100)
    for n in NS:
        for hold in HOLDS:
            jp_raw = rets[(n, "raw", hold)].loc["2026-06-01":]
            jp_nr = rets[(n, "skip_norefill", hold)].loc["2026-06-01":]
            h1_raw = rets[(n, "raw", hold)].loc["2026-01-01":"2026-05-31"]
            h1_nr = rets[(n, "skip_norefill", hold)].loc["2026-01-01":"2026-05-31"]
            print(
                f"Top{n} {hold_label[hold]}: H1 Δ累计={_fmt_pp(_cum(h1_nr)-_cum(h1_raw))}  "
                f"Jun+ Δ累计={_fmt_pp(_cum(jp_nr)-_cum(jp_raw))}  "
                f"Jun+ n={len(jp_nr)}"
            )

    pd.DataFrame(summary).to_csv(
        OUT / "skip_gapup_2026_slice_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(monthly_rows).to_csv(
        OUT / "skip_gapup_2026_monthly.csv", index=False, encoding="utf-8-sig"
    )
    ks.to_csv(OUT / "skip_gapup_2026_kicks.csv", index=False, encoding="utf-8-sig")
    # by-date nets for audit
    by = pd.DataFrame({"t": rets[(30, "raw", "mon_wed")].index})
    by = by.set_index("t")
    for n in NS:
        for hold in HOLDS:
            for filt in FILTERS:
                by[f"Top{n}_{hold}_{filt}"] = rets[(n, filt, hold)]
            by[f"Top{n}_{hold}_kick"] = kicks_s[(n, "skip_norefill", hold)]
        by[f"EW_{hold}"] = bm[hold]
    by = by.join(meta[["entry", "sig_month", "entry_month"]])
    by.to_csv(OUT / "skip_gapup_2026_by_date.csv", encoding="utf-8-sig")
    print(f"\nwrote {TXT}")
    out_f.close()


if __name__ == "__main__":
    main()
