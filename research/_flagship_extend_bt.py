"""用 last-window 模型给旗舰全样本得分接上缺失 W-FRI 调仓日，再做 5 日持有到下期回测。

不重训、不改旗舰超参。得分目录：
  旧全样本 WF: results/xgb_h5_sizeind_sparse_w156_decay0_nobshare_20260817
  模型: results/xgb_h5_sizeind_w156_nob_20260823/models（2026-07-31 last-window）
  产出: results/xgb_h5_sizeind_w156_nob_20260823
"""
from __future__ import annotations

import json
import os
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
from models.wf.metrics import spearman_ic
from research.ic.universe import mask_scores_for_backtest
from run import _load_data, _load_indices
from utils.rebalance_dates import get_rebalance_dates as _rebalance_dates

OLD_SCORE_DIR = ROOT / "results" / "xgb_h5_sizeind_sparse_w156_decay0_nobshare_20260817"
OLD_SCORE_FILE = OLD_SCORE_DIR / "factor_scores_xgb_h5_w156_p_sparse_rt4.parquet"
OLD_METRICS = OLD_SCORE_DIR / "model_metrics_xgb_h5_w156_p_sparse_rt4.json"
OLD_IC = OLD_SCORE_DIR / "ic_series_xgb_h5_w156_p_sparse_rt4.csv"

_OUT_NAME = os.environ.get("FLAGSHIP_EXT_OUT", "xgb_h5_sizeind_w156_nob_20260829")
OUT = ROOT / "results" / _OUT_NAME
MODEL_DIR = OUT
TAG = "xgb_h5_sizeind_w156_nob"
BT_FREQ = "W-FRI"
MASK_HOLD = 5
TOPNS = (100, 30, 10)
DENSE_YAML = ROOT / "config" / "factor_configs_h5_sizeind_20260815.yaml"


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


def _fmt_pct(x, nd=1) -> str:
    if x is None or not np.isfinite(x):
        return "    —"
    return f"{x * 100:.{nd}f}%"


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


def _fwd_h5(prices: pd.DataFrame, open_: pd.DataFrame, date) -> pd.Series:
    """close[t+5]/open[t+1]-1，与训练标签同口径。"""
    idx = pd.DatetimeIndex(prices.index).sort_values()
    d = pd.Timestamp(date)
    loc = idx.get_indexer([d], method="pad")[0]
    if loc < 0 or loc + 5 >= len(idx):
        return pd.Series(dtype=float)
    t1, t5 = idx[loc + 1], idx[loc + 5]
    o = open_.reindex(index=[t1], columns=prices.columns).iloc[0]
    c = prices.reindex(index=[t5], columns=prices.columns).iloc[0]
    return (c / o - 1.0).replace([np.inf, -np.inf], np.nan)


def score_missing_fridays(old_scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    from live.daily_update import (
        DEFAULT_WARMUP_DAYS,
        append_factor_panels,
        build_feature_matrix,
        compute_barra_warmup,
        load_clean_panels,
        load_latest_models,
        neutralize_as_of,
        predict_scores,
        resolve_feature_names,
        slice_warmup,
    )

    last_old = pd.Timestamp(old_scores.index.max())
    data_end = pd.Timestamp(prices.index.max())
    cal = _rebalance_dates(prices.index, BT_FREQ)
    missing = [d for d in cal if d > last_old and d <= data_end]
    logger.info(
        f"旧得分末日={last_old.date()} 数据末日={data_end.date()} "
        f"待接调仓日={[d.date() for d in missing]}"
    )
    if not missing:
        return old_scores

    panels = load_clean_panels(sample=0)
    model_entries, manifest_fn, manifest_nc = load_latest_models(
        str(MODEL_DIR), prefer_model="xgb", prefer_window=156,
    )
    feature_names, fn_source = resolve_feature_names(
        model_entries, str(DENSE_YAML), horizon=5,
    )
    logger.info(f"特征 {len(feature_names)} 个（{fn_source}） neut={manifest_nc}")

    warmup = slice_warmup(panels, data_end, DEFAULT_WARMUP_DAYS)
    factor_panels = append_factor_panels(feature_names, warmup, data_end)
    barra, weights = compute_barra_warmup(warmup)
    universe = panels["prices"].columns.astype(str).str.zfill(6)

    rows = []
    for d in missing:
        logger.info(f"last-window 出分 {d.date()}")
        neut_rows = None
        if manifest_fn:
            neut_rows = neutralize_as_of(
                factor_panels, barra, weights, panels["industry_map"], d,
                universe=universe, prices=panels["prices"],
                neut_controls=manifest_nc,
            )
        X = build_feature_matrix(
            feature_names, neut_rows or {}, factor_panels, d, universe=universe,
        )
        s = predict_scores(model_entries, X)
        s.name = d
        rows.append(s)
        n = int(pd.to_numeric(s, errors="coerce").notna().sum())
        logger.info(f"  {d.date()} n_scored={n}")

    extra = pd.DataFrame(rows)
    extra.index = pd.DatetimeIndex([r.name for r in rows])
    extra.columns = extra.columns.astype(str).str.zfill(6)
    all_cols = old_scores.columns.union(extra.columns)
    old_a = old_scores.reindex(columns=all_cols)
    extra_a = extra.reindex(columns=all_cols)
    out = pd.concat([old_a, extra_a]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    logger.info(
        f"拼接得分 {old_scores.shape} + {extra.shape} → {out.shape} "
        f"{out.index.min().date()}→{out.index.max().date()}"
    )
    return out


def _run_one(*, prices, bt_scores, open_, masks, indices, cfg,
             stock_names, listing_dates, volume, st_schedule, delist_dates,
             clean_ret, top_n, plot: bool):
    result = run_quantile_backtest(
        prices, bt_scores,
        n_quantiles=5, rebalance_freq=BT_FREQ,
        start=BACKTEST_START, end=BACKTEST_END,
        open_prices=open_, masks=masks, indices=indices,
        config=cfg, stock_names=stock_names,
        listing_dates=listing_dates, volume=volume,
        st_schedule=st_schedule, delist_dates=delist_dates,
        eligible_mask=None, top_n=top_n,
        returns=clean_ret, hold_period=None,
    )
    stem = f"backtest_{TAG}" if top_n == 100 else f"backtest_top{top_n}_{TAG}"
    print_quantile_summary(result, rebalance_freq=BT_FREQ, rf=RISK_FREE_RATE)
    if plot:
        plot_quantile_result(
            result,
            title=f"Q1-Q5  |  {TAG}  hold-to-next  Top{top_n}",
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
    export_holdings(result, save_path=str(OUT / f"holdings_top{top_n}_{TAG}.csv"))
    export_turnover_detail(result, save_path=str(OUT / f"turnover_detail_{stem}.csv"))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    add_utf8_file_sink(OUT / "extend_bt.log")
    logger.info("旗舰得分续接 + 5日持有到下期回测（不重训全 WF）")

    logger.info("load data...")
    (
        prices, prices_raw, financial, volume, amount,
        open_, high, low, clean_ret, masks,
        market_prices, industry_map,
        margin, moneyflow, northbound, institution,
        total_mv, circ_mv, turnover_rate,
    ) = _load_data(skip_download=True, sample=0)
    indices = _load_indices()

    old = pd.read_parquet(OLD_SCORE_FILE)
    old.index = pd.to_datetime(old.index)
    old.columns = old.columns.astype(str).str.zfill(6)
    n_b = sum(is_b_share_code(c) for c in old.columns)
    if n_b:
        old = old.loc[:, [c for c in old.columns if not is_b_share_code(c)]]
        logger.warning(f"旧得分含 B 股 {n_b} 列，已剔除")

    scores = score_missing_fridays(old, prices)
    scores.to_parquet(OUT / f"factor_scores_{TAG}.parquet")

    stock_names, delist_dates, listing_dates, is_st, st_hist, st_schedule = _load_bt_meta(prices)
    bt_scores = mask_scores_for_backtest(
        scores, prices, open_=open_, hold_period=MASK_HOLD, volume=volume, masks=masks,
        stock_names=stock_names, listing_dates=listing_dates,
        delist_dates=delist_dates, is_st_current=is_st, st_history=st_hist,
        score_universe="strict",
    )
    logger.info(
        f"bt_score_universe=strict: {int(scores.notna().sum().sum())} → "
        f"{int(bt_scores.notna().sum().sum())}"
    )

    ic_rows = []
    for d in scores.index:
        y = _fwd_h5(prices, open_, d)
        s = scores.loc[d]
        common = s.dropna().index.intersection(y.dropna().index)
        if len(common) < 50:
            continue
        ic = spearman_ic(s.reindex(common).values, y.reindex(common).values)
        ic_rows.append((d, ic))
    ic = pd.Series({d: v for d, v in ic_rows}, dtype=float).sort_index()
    ic.to_csv(OUT / f"ic_series_{TAG}.csv", header=["ic"])
    ic_metrics = {
        "IC均值": round(float(ic.mean()), 4) if len(ic) else None,
        "IC标准差": round(float(ic.std()), 4) if len(ic) else None,
        "ICIR": round(float(ic.mean() / ic.std()), 4) if len(ic) and ic.std() else None,
        "IC>0胜率": round(float((ic > 0).mean()), 4) if len(ic) else None,
        "预测期数": int(len(ic)),
        "得分末日": str(scores.index.max().date()),
        "数据末日": str(prices.index.max().date()),
        "续接": "last-window xgb_w156_20260731 + 旧全样本 WF",
        "old_ic": json.loads(OLD_METRICS.read_text(encoding="utf-8")) if OLD_METRICS.exists() else {},
    }
    (OUT / f"model_metrics_{TAG}.json").write_text(
        json.dumps(ic_metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(f"IC mean={ic_metrics['IC均值']} IR={ic_metrics['ICIR']} n={ic_metrics['预测期数']}")

    cfg = BacktestConfig(bid_ask_spread_bps=10.0)
    report = {"ic": ic_metrics, "old_full_sample": {}}
    if OLD_METRICS.exists():
        report["old_full_sample"] = json.loads(OLD_METRICS.read_text(encoding="utf-8"))

    for n in TOPNS:
        logger.info(f"backtest Top{n} hold-to-next...")
        result = _run_one(
            prices=prices, bt_scores=bt_scores, open_=open_, masks=masks,
            indices=indices, cfg=cfg, stock_names=stock_names,
            listing_dates=listing_dates, volume=volume, st_schedule=st_schedule,
            delist_dates=delist_dates, clean_ret=clean_ret,
            top_n=n, plot=(n == 100),
        )
        col = f"Top{n}"
        report[col] = summarize_track(result.nav, col)
        if n == 100:
            report["Q5"] = summarize_track(result.nav, "Q5")
            report["benchmark"] = summarize_track(result.nav, "benchmark")

    (OUT / "extend_bt_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    lines = ["旗舰续接回测  hold=到下期  bid-ask=10bp",
             f"  得分 {scores.index.min().date()} → {scores.index.max().date()}  n={len(scores)}",
             f"  IC={_fmt_pct(ic_metrics['IC均值'], 2)}  ICIR={ic_metrics['ICIR']}  n={ic_metrics['预测期数']}"]
    for n in TOPNS:
        t = report.get(f"Top{n}") or {}
        lines.append(
            f"  Top{n}: 年化={_fmt_pct(t.get('ann'))}  超额={_fmt_pct(t.get('excess_ann'))}  "
            f"MDD={_fmt_pct(t.get('mdd'))}  2025={_fmt_pct((t.get('year') or {}).get(2025))}  "
            f"2026={_fmt_pct((t.get('year') or {}).get(2026))}"
        )
    bm = report.get("benchmark") or {}
    lines.append(f"  等权: 年化={_fmt_pct(bm.get('ann'))}  MDD={_fmt_pct(bm.get('mdd'))}")
    text = "\n".join(lines)
    (OUT / "extend_bt_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
