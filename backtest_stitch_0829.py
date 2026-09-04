"""一次性：用 0817 历史 WF 得分 + 0829 last-window 08-21 得分拼接，旗舰参数回测到最新可回测日。

不重训全 WF；单进程。产物落 results/xgb_h5_sizeind_w156_nob_20260829_bt/。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAIN_MAX_WORKERS", "1")

from config.encoding_bootstrap import bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()

OLD_DIR = ROOT / "results" / "xgb_h5_sizeind_sparse_w156_decay0_nobshare_20260817"
LIVE_DIR = ROOT / "results" / "xgb_h5_sizeind_w156_nob_20260829"
OUT_DIR = ROOT / "results" / "xgb_h5_sizeind_w156_nob_20260829_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
add = None
from config.encoding_bootstrap import add_utf8_file_sink
add_utf8_file_sink(OUT_DIR / "run.log")

OLD_SCORES = OLD_DIR / "ml_factor_scores_xgb_h5_w156_p_sparse_rt4.parquet"
LIVE_SCORES = LIVE_DIR / "ml_factor_scores_xgb_h5_sizeind_w156_nob.parquet"
TAG = "xgb_h5_w156_p_sizeind_rt4"

# ── 1. 加载数据面板（与 run.py 同口径：drop 8开头BJ + B股，保留92）──────────────
from run import _load_data, _load_indices
from config.settings import BACKTEST_START, BACKTEST_END, RISK_FREE_RATE

logger.info("Step 1: 加载并清洗数据（skip-download）")
(prices, prices_raw, financial, volume, amount,
 open_, high, low, clean_ret, masks,
 market_prices, industry_map,
 margin, moneyflow, northbound, institution,
 total_mv, circ_mv, turnover_rate) = _load_data(skip_download=True, sample=0)

logger.info(f"prices: {prices.shape} 末日={prices.index.max().date()}")

# ── 2. 拼接得分 ────────────────────────────────────────────────────────────
old = pd.read_parquet(OLD_SCORES)
live = pd.read_parquet(LIVE_SCORES)
logger.info(f"old scores: {old.shape} {old.index.min().date()}~{old.index.max().date()}")
logger.info(f"live scores: {live.shape} {live.index.min().date()}~{live.index.max().date()}")

# 列对齐到 zfill(6)
def _zfill_cols(df):
    df = df.copy()
    df.columns = [str(c).zfill(6) for c in df.columns]
    return df

old = _zfill_cols(old)
live = _zfill_cols(live)

# 去掉 live 中与 old 末尾重叠/早于 old 末日的日期，避免重复
old_end = old.index.max()
live_new = live.loc[live.index > old_end]
logger.info(f"live 新增调仓日: {list(live_new.index)}")

# 列取并集，行 concat
all_cols = sorted(set(old.columns) | set(live_new.columns))
old = old.reindex(columns=all_cols)
live_new = live_new.reindex(columns=all_cols)
scores = pd.concat([old, live_new], axis=0)
scores.index = pd.DatetimeIndex(pd.to_datetime(scores.index)).normalize()
scores = scores.sort_index()
# 仅保留有得分的日期
scores = scores.loc[scores.notna().any(axis=1)]
logger.info(f"拼接后 scores: {scores.shape} {scores.index.min().date()}~{scores.index.max().date()}  n_dates={len(scores)}")

# ── 3. 回测（复刻 run.py main() 回测块，旗舰参数）────────────────────────────
from backtest.quantile import run_quantile_backtest
from backtest.report import (
    plot_quantile_result, print_quantile_summary,
    export_holdings, export_turnover_detail,
)
from backtest.risk_metrics import export_risk_metrics as _export_risk_metrics
from backtest.execution import (
    BacktestConfig,
    build_st_schedule,
    build_delist_dates_from_stock_list,
    build_listing_dates_from_stock_list,
)
from research.ic.universe import mask_scores_for_backtest

indices = _load_indices()
bt_config = BacktestConfig(
    turnover_limit=1.0,
    rank_change_threshold=0.0,
    portfolio_opt="ew",
    max_weight=None,
    cov_lookback=60,
    risk_aversion=1.0,
    bid_ask_spread_bps=10.0,
)
logger.info(f"bt_config: bid_ask={bt_config.bid_ask_spread_bps}bp opt={bt_config.portfolio_opt}")

# ST / 退市 / 上市 元数据
stock_names_ser = None
st_schedule = None
delist_dates = None
listing_dates = None
is_st_ser = None
st_history_df = None
try:
    from config.settings import UNIVERSE_DIR, RAW_DIR
    st_hist_path = RAW_DIR / "st_history.parquet"
    if st_hist_path.exists():
        st_history_df = pd.read_parquet(st_hist_path)
        logger.info(f"ST 历史: {len(st_history_df)} 段, {st_history_df['code'].nunique()} 只")
except Exception as e:
    logger.warning(f"ST 历史加载失败: {e}")
    st_history_df = None
try:
    sl_path = UNIVERSE_DIR / "stock_list.parquet"
    if sl_path.exists():
        sl_df = pd.read_parquet(sl_path)
        if "code" in sl_df.columns and "name" in sl_df.columns:
            stock_names_ser = sl_df.set_index("code")["name"]
            stock_names_ser.index = stock_names_ser.index.astype(str).str.zfill(6)
            delist_dates = build_delist_dates_from_stock_list(sl_df) or None
            listing_dates = build_listing_dates_from_stock_list(sl_df) or None
            is_st_ser = (
                sl_df.set_index("code")["is_st_current"]
                if "is_st_current" in sl_df.columns else None
            )
            if is_st_ser is not None:
                is_st_ser.index = is_st_ser.index.astype(str).str.zfill(6)
            st_schedule = build_st_schedule(
                stock_names_ser, prices.index,
                is_st_current=is_st_ser,
                delist_dates=delist_dates,
                st_history=st_history_df,
            )
            logger.info(f"ST 时间序列: {st_schedule.shape if st_schedule is not None else None}")
except Exception as e:
    logger.warning(f"stock_list 元数据加载失败: {e}")

# 回测得分宇宙裁剪（strict）
n_scored_train = int(scores.notna().sum().sum())
bt_scores = mask_scores_for_backtest(
    scores, prices, open_=open_,
    hold_period=5, volume=volume, masks=masks,
    stock_names=stock_names_ser,
    listing_dates=listing_dates, delist_dates=delist_dates,
    is_st_current=is_st_ser, st_history=st_history_df,
    score_universe="strict",
)
n_scored_bt = int(bt_scores.notna().sum().sum())
logger.info(f"bt_score_universe=strict: 得分格子 {n_scored_train} -> {n_scored_bt}")

result = run_quantile_backtest(
    prices, bt_scores,
    n_quantiles=5,
    rebalance_freq="W-FRI",
    start=BACKTEST_START,
    end=BACKTEST_END,
    open_prices=open_,
    masks=masks,
    indices=indices,
    config=bt_config,
    stock_names=stock_names_ser,
    listing_dates=listing_dates,
    volume=volume,
    st_schedule=st_schedule,
    delist_dates=delist_dates,
    eligible_mask=None,
    top_n=100,
    position_regime=None,
    returns=clean_ret,
    hold_period=5,
)
print_quantile_summary(result, rebalance_freq="W-FRI", rf=RISK_FREE_RATE)
plot_quantile_result(
    result,
    title="Q1-Q5 分组回测  |  flagship xgb_h5 sizeind w156 nob  (stitch 0817+0829)",
    save_path=str(OUT_DIR / f"backtest_{TAG}.png"),
    rebalance_freq="W-FRI",
    rf=RISK_FREE_RATE,
)
result.nav.to_csv(OUT_DIR / f"backtest_{TAG}_nav.csv", encoding="utf-8-sig")
result.annual_returns.to_csv(OUT_DIR / f"backtest_{TAG}_annual.csv", encoding="utf-8-sig")
result.long_short_nav.to_csv(OUT_DIR / f"backtest_{TAG}_longshort.csv", header=True)
_export_risk_metrics(
    result.nav,
    save_path=str(OUT_DIR / f"backtest_{TAG}_risk_metrics.csv"),
    rebalance_freq="W-FRI",
    rf=RISK_FREE_RATE,
)
export_holdings(result, save_path=str(OUT_DIR / f"holdings_top100_{TAG}.csv"))
export_turnover_detail(result, save_path=str(OUT_DIR / f"turnover_detail_{TAG}.csv"))
# 落盘拼接得分（供复查）
scores.to_parquet(OUT_DIR / f"factor_scores_{TAG}_stitch.parquet")
logger.info(f"完成 -> {OUT_DIR}")
