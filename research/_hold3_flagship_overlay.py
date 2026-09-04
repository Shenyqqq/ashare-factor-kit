"""旗舰 xgb 得分不变：三日持仓 overlay（不重训）。

同一套 ``factor_scores``（完整 WF），调仓日历仍 W-FRI。
引擎 ``hold_period=3``：周五信号 → 下周一开买 → 周三收盘卖
（``close[t+3]/open[t+1]``），不拿到下周五。

对照：同一得分上 hold_period=None（持有到下一信号日，旗舰生产口径）。
CLI ``--hold-period 3`` 会改训练标签，这里不用。

用法（仓库根，单进程）:
    .venv\\Scripts\\python.exe research/_hold3_flagship_overlay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.encoding_bootstrap import add_utf8_file_sink, bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()

import numpy as np
import pandas as pd
from loguru import logger

from backtest.execution import (
    BacktestConfig,
    build_delist_dates_from_stock_list,
    build_listing_dates_from_stock_list,
    build_st_schedule,
)
from backtest.quantile import run_quantile_backtest
from backtest.report import (
    export_holdings,
    export_turnover_detail,
    plot_quantile_result,
    print_quantile_summary,
)
from backtest.risk_metrics import (
    compute_risk_metrics,
    export_risk_metrics,
    period_returns_from_nav,
)
from config.settings import (
    BACKTEST_END,
    BACKTEST_START,
    RAW_DIR,
    RISK_FREE_RATE,
    UNIVERSE_DIR,
)
from data.download import is_b_share_code
from research.ic.universe import mask_scores_for_backtest
from run import _load_data, _load_indices

SCORE_DIR = ROOT / "results" / "xgb_h5_sizeind_sparse_w156_decay0_nobshare_20260817"
SCORE_FILE = SCORE_DIR / "factor_scores_xgb_h5_w156_p_sparse_rt4.parquet"
NAV100_5D = SCORE_DIR / "backtest_xgb_h5_w156_p_sparse_rt4_nav.csv"
NAV30_5D = SCORE_DIR / "backtest_top30_xgb_h5_w156_p_sparse_rt4_nav.csv"
IC_FILE = SCORE_DIR / "ic_series_xgb_h5_w156_p_sparse_rt4.csv"
METRICS_FILE = SCORE_DIR / "model_metrics_xgb_h5_w156_p_sparse_rt4.json"

OUT = ROOT / "results" / "xgb_h5_sizeind_w156_nob_hold3_20260818"
TAG = "xgb_h5_w156_p_sparse_rt4_hold3"
BT_FREQ = "W-FRI"
HOLD3 = 3
MASK_HOLD = 5  # 与旗舰 mask_scores 同口径，排名宇宙不变
TOPNS = (100, 30, 10)


def year_returns(nav: pd.Series) -> pd.Series:
    s = pd.to_numeric(nav, errors="coerce")
    s.index = pd.DatetimeIndex(s.index)
    rets = period_returns_from_nav(s)
    out = {}
    for y, g in rets.groupby(rets.index.year):
        out[int(y)] = float((1.0 + g.fillna(0.0)).prod() - 1.0)
    return pd.Series(out)


def summarize_track(nav: pd.DataFrame, col: str) -> dict:
    m = compute_risk_metrics(nav, rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE)
    row = m.loc[col] if col in m.index else None
    r_s = year_returns(nav[col])
    r_b = year_returns(nav["benchmark"]) if "benchmark" in nav.columns else pd.Series(dtype=float)
    years = sorted(set(r_s.index) & set(r_b.index)) if len(r_b) else sorted(r_s.index)
    xs_y = {int(y): float(r_s[y] - r_b[y]) for y in years} if len(r_b) else {}
    bm_ann = float(m.loc["benchmark", "年化收益"]) if "benchmark" in m.index else np.nan
    ann = float(row["年化收益"]) if row is not None else np.nan
    return {
        "ann": ann,
        "sharpe": float(row["Sharpe"]) if row is not None else np.nan,
        "mdd": float(row["最大回撤"]) if row is not None else np.nan,
        "win_rate": float(row["胜率"]) if row is not None else np.nan,
        "beat_bm": float(row["超额胜率"]) if row is not None else np.nan,
        "bm_ann": bm_ann,
        "excess_ann": (ann - bm_ann) if np.isfinite(ann) and np.isfinite(bm_ann) else np.nan,
        "year": {int(y): float(r_s[y]) for y in r_s.index},
        "bm_year": {int(y): float(r_b[y]) for y in r_b.index},
        "xs_year": xs_y,
    }


def _ser(d):
    if isinstance(d, dict):
        return {str(k): _ser(v) for k, v in d.items()}
    if isinstance(d, float):
        return None if (d != d) else round(d, 6)
    if isinstance(d, (np.floating,)):
        d = float(d)
        return None if (d != d) else round(d, 6)
    return d


def _fmt_pct(x, nd=1) -> str:
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x * 100:.{nd}f}%"


def _fmt_pp(x, nd=1) -> str:
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x * 100:+.{nd}f}pp"


def _ic_yearly(path: Path) -> dict:
    if not path.exists():
        return {}
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    out = {"all_mean": float(s.mean()), "all_ir": float(s.mean() / s.std()) if s.std() else np.nan,
           "pos_rate": float((s > 0).mean()), "n": int(s.notna().sum())}
    for y, g in s.groupby(s.index.year):
        out[f"y{int(y)}"] = float(g.mean())
        out[f"y{int(y)}_ir"] = float(g.mean() / g.std()) if g.std() else np.nan
    return out


def _load_bt_meta(prices: pd.DataFrame):
    sl_path = UNIVERSE_DIR / "stock_list.parquet"
    sl_df = pd.read_parquet(sl_path)
    stock_names = sl_df.set_index("code")["name"]
    stock_names.index = stock_names.index.astype(str).str.zfill(6)
    delist_dates = build_delist_dates_from_stock_list(sl_df) or None
    listing_dates = build_listing_dates_from_stock_list(sl_df) or None
    is_st = sl_df.set_index("code")["is_st_current"] if "is_st_current" in sl_df.columns else None
    if is_st is not None:
        is_st.index = is_st.index.astype(str).str.zfill(6)
    st_hist = pd.read_parquet(RAW_DIR / "st_history.parquet") if (RAW_DIR / "st_history.parquet").exists() else None
    st_schedule = build_st_schedule(
        stock_names, prices.index, is_st_current=is_st,
        delist_dates=delist_dates, st_history=st_hist,
    )
    return stock_names, delist_dates, listing_dates, is_st, st_hist, st_schedule


def _run_one(
    *,
    prices, bt_scores, open_, masks, indices, cfg,
    stock_names, listing_dates, volume, st_schedule, delist_dates,
    clean_ret, top_n, hold_period, tag_suffix, plot: bool,
):
    result = run_quantile_backtest(
        prices, bt_scores,
        n_quantiles=5, rebalance_freq=BT_FREQ,
        start=BACKTEST_START, end=BACKTEST_END,
        open_prices=open_, masks=masks, indices=indices,
        config=cfg, stock_names=stock_names,
        listing_dates=listing_dates, volume=volume,
        st_schedule=st_schedule, delist_dates=delist_dates,
        eligible_mask=None, top_n=top_n,
        returns=clean_ret, hold_period=hold_period,
    )
    stem = f"backtest_{TAG}{tag_suffix}"
    print_quantile_summary(result, rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE)
    if plot:
        plot_quantile_result(
            result,
            title=f"Q1-Q5  |  xgb flagship scores  hold={hold_period}  Top{top_n}",
            save_path=str(OUT / f"{stem}.png"),
            rebalance_freq=BT_FREQ,
            rf=RISK_FREE_RATE,
        )
    result.nav.to_csv(OUT / f"{stem}_nav.csv", encoding="utf-8-sig")
    result.annual_returns.to_csv(OUT / f"{stem}_annual.csv", encoding="utf-8-sig")
    result.long_short_nav.to_csv(OUT / f"{stem}_longshort.csv", header=True)
    export_risk_metrics(
        result.nav, save_path=str(OUT / f"{stem}_risk_metrics.csv"),
        rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE,
    )
    export_holdings(result, save_path=str(OUT / f"holdings_top{top_n}_{TAG}{tag_suffix}.csv"))
    export_turnover_detail(result, save_path=str(OUT / f"turnover_detail_{TAG}{tag_suffix}.csv"))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    add_utf8_file_sink(OUT / "run.log")
    if not SCORE_FILE.exists():
        raise FileNotFoundError(f"旗舰全样本得分不存在: {SCORE_FILE}")

    logger.info(
        "三日持仓 overlay：得分不变，不重训。"
        f" scores={SCORE_FILE}  →  {OUT}"
    )
    logger.info(
        "CLI --hold-period 3 会改训练标签，本脚本只调 "
        "run_quantile_backtest(hold_period=3)；调仓日历仍 W-FRI。"
    )

    logger.info("load data...")
    (
        prices, prices_raw, financial, volume, amount,
        open_, high, low, clean_ret, masks,
        market_prices, industry_map,
        margin, moneyflow, northbound, institution,
        total_mv, circ_mv, turnover_rate,
    ) = _load_data(skip_download=True, sample=0)
    indices = _load_indices()

    scores = pd.read_parquet(SCORE_FILE)
    scores.index = pd.to_datetime(scores.index)
    scores.columns = scores.columns.astype(str).str.zfill(6)
    n_b = sum(is_b_share_code(c) for c in scores.columns)
    if n_b:
        keep = [c for c in scores.columns if not is_b_share_code(c)]
        logger.warning(f"scores 含 B 股 {n_b} 列，已剔除")
        scores = scores.loc[:, keep]
    logger.info(
        f"scores shape={scores.shape}  "
        f"{scores.index.min().date()}→{scores.index.max().date()}  "
        f"B股列={n_b}  BJ92="
        f"{sum(str(c).startswith('92') for c in scores.columns)}"
    )

    stock_names, delist_dates, listing_dates, is_st, st_hist, st_schedule = _load_bt_meta(prices)
    bt_scores = mask_scores_for_backtest(
        scores, prices, open_=open_, hold_period=MASK_HOLD, volume=volume, masks=masks,
        stock_names=stock_names, listing_dates=listing_dates,
        delist_dates=delist_dates, is_st_current=is_st, st_history=st_hist,
        score_universe="strict",
    )
    n_train = int(scores.notna().sum().sum())
    n_bt = int(bt_scores.notna().sum().sum())
    logger.info(f"bt_score_universe=strict  mask_hold={MASK_HOLD}: {n_train} → {n_bt}")

    cfg = BacktestConfig(bid_ask_spread_bps=10.0)

    report: dict = {
        "method": "overlay_hold3_same_scores",
        "retrain": False,
        "scores": str(SCORE_FILE),
        "calendar": BT_FREQ,
        "signal": "周五收盘后",
        "buy": "下周一开盘 (t+1 open)",
        "sell_hold3": "周三收盘 (t+3 close)",
        "sell_flagship": "持有到下一信号日（约下周五）",
        "costs": "佣金+印花税+10bp bid-ask；换手按周一调仓名变化计（非整周全清）",
        "ic_flagship": json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else {},
        "ic_yearly": _ic_yearly(IC_FILE),
    }

    # 旗舰 5 日（已落盘 Top100 / Top30）
    if NAV100_5D.exists():
        nav100 = pd.read_csv(NAV100_5D, index_col=0, parse_dates=True)
        report["flagship_5d_Top100"] = summarize_track(nav100, "Top100")
        report["flagship_5d_Q5"] = summarize_track(nav100, "Q5")
        report["flagship_5d_benchmark"] = summarize_track(nav100, "benchmark")
    if NAV30_5D.exists():
        nav30 = pd.read_csv(NAV30_5D, index_col=0, parse_dates=True)
        report["flagship_5d_Top30"] = summarize_track(nav30, "Top30")

    for n in TOPNS:
        logger.info(f"hold3 backtest Top{n}...")
        result = _run_one(
            prices=prices, bt_scores=bt_scores, open_=open_, masks=masks,
            indices=indices, cfg=cfg, stock_names=stock_names,
            listing_dates=listing_dates, volume=volume, st_schedule=st_schedule,
            delist_dates=delist_dates, clean_ret=clean_ret,
            top_n=n, hold_period=HOLD3, tag_suffix=f"_top{n}",
            plot=(n == 100),
        )
        col = f"Top{n}"
        report[f"hold3_{col}"] = summarize_track(result.nav, col)
        if n == 100:
            report["hold3_Q5"] = summarize_track(result.nav, "Q5")
            report["hold3_benchmark"] = summarize_track(result.nav, "benchmark")
            report["hold3_ic_monotonicity"] = float(result.ic_monotonicity)

    # Top10 五日对照：旗舰目录没有 Top10 nav
    logger.info("flagship 5d (hold to next signal) Top10 for comparison...")
    r10 = _run_one(
        prices=prices, bt_scores=bt_scores, open_=open_, masks=masks,
        indices=indices, cfg=cfg, stock_names=stock_names,
        listing_dates=listing_dates, volume=volume, st_schedule=st_schedule,
        delist_dates=delist_dates, clean_ret=clean_ret,
        top_n=10, hold_period=None, tag_suffix="_top10_hold5",
        plot=False,
    )
    report["flagship_5d_Top10"] = summarize_track(r10.nav, "Top10")

    outp = OUT / "hold3_vs_flagship5d.json"
    outp.write_text(json.dumps(_ser(report), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"wrote {outp}")

    lines = ["=" * 78, "  旗舰 xgb 三日持仓 overlay（不重训）", "=" * 78]
    lines.append("  实现: overlay。同一套 factor_scores，run_quantile_backtest(hold_period=3)。")
    lines.append("  未走 CLI --hold-period 3（那条会改 3 日训练标签）。")
    lines.append("  日历: 仍 W-FRI。周五信号 → 下周一开买 → 周三收盘卖；Thu–Fri 现金（期收益不计）。")
    lines.append("  成本: 佣金+印花+10bp bid-ask；换手按周一调仓名变化（重叠票不额外双边）。")
    ic = report.get("ic_flagship") or {}
    icy = report.get("ic_yearly") or {}
    lines.append(
        f"  训练 IC（5 日标签，得分不变）: mean={ic.get('IC均值')}  "
        f"ICIR={ic.get('ICIR')}  胜率={ic.get('IC>0胜率')}  n={ic.get('预测期数')}"
    )
    if icy:
        yic = "  ".join(
            f"{y}={_fmt_pct(icy.get(f'y{y}'), 2)}"
            for y in (2021, 2022, 2023, 2024, 2025, 2026)
            if f"y{y}" in icy
        )
        lines.append(f"  IC 分年均值: {yic}")
    lines.append("-" * 78)

    def _block(title: str, h3: dict | None, h5: dict | None):
        lines.append(f"  [{title}]")
        if h5:
            lines.append(
                f"    5日  年化={_fmt_pct(h5.get('ann'))}  超额={_fmt_pct(h5.get('excess_ann'))}  "
                f"Sharpe={h5.get('sharpe', float('nan')):.2f}  MDD={_fmt_pct(h5.get('mdd'))}"
            )
        if h3:
            lines.append(
                f"    3日  年化={_fmt_pct(h3.get('ann'))}  超额={_fmt_pct(h3.get('excess_ann'))}  "
                f"Sharpe={h3.get('sharpe', float('nan')):.2f}  MDD={_fmt_pct(h3.get('mdd'))}"
            )
        if h3 and h5:
            lines.append(
                f"    Δ(3−5) 年化={_fmt_pp(h3['ann'] - h5['ann'])}  "
                f"超额={_fmt_pp(h3['excess_ann'] - h5['excess_ann'])}  "
                f"MDD={_fmt_pp(h3['mdd'] - h5['mdd'])}"
            )
            y5 = h5.get("year") or {}
            y3 = h3.get("year") or {}
            xs5 = h5.get("xs_year") or {}
            xs3 = h3.get("xs_year") or {}
            for y in (2021, 2022, 2023, 2024, 2025, 2026):
                k = str(y)
                if k not in y3 and y not in y3 and k not in y5:
                    continue
                a3 = y3.get(k, y3.get(y))
                a5 = y5.get(k, y5.get(y))
                e3 = xs3.get(k, xs3.get(y))
                e5 = xs5.get(k, xs5.get(y))
                d_a = (a3 - a5) if (a3 is not None and a5 is not None
                                    and np.isfinite(a3) and np.isfinite(a5)) else np.nan
                d_e = (e3 - e5) if (e3 is not None and e5 is not None
                                    and np.isfinite(e3) and np.isfinite(e5)) else np.nan
                lines.append(
                    f"      {y}: 3日收益={_fmt_pct(a3)} 超额={_fmt_pct(e3)}  |  "
                    f"5日收益={_fmt_pct(a5)} 超额={_fmt_pct(e5)}  |  "
                    f"Δ收益={_fmt_pp(d_a)} Δ超额={_fmt_pp(d_e)}"
                )

    _block("Top100", report.get("hold3_Top100"), report.get("flagship_5d_Top100"))
    _block("Top30", report.get("hold3_Top30"), report.get("flagship_5d_Top30"))
    _block("Top10", report.get("hold3_Top10"), report.get("flagship_5d_Top10"))
    _block("Q5", report.get("hold3_Q5"), report.get("flagship_5d_Q5"))
    _block("等权基准", report.get("hold3_benchmark"), report.get("flagship_5d_benchmark"))
    lines.append("=" * 78)
    text = "\n".join(lines)
    (OUT / "hold3_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
