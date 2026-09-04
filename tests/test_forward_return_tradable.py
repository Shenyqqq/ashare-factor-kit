"""
tests/test_forward_return_tradable.py

最小单测：
  1. build_forward_return 在买日一字涨停时为 NaN（ML 应复用此函数）
  2. listing_dates 传入后次新被 build_ic_tradability_mask 剔除
  3. strategies.ml 不再维护无 masks 的本地 forward_return 分叉
  4. delist_dates：退市日之后 tradable=False（与 TradeRules.is_delisted 一致）
  5. BacktestConfig.min_listing_days == MIN_LISTING_DAYS（无 252 vs 60 双标准）
  6. OHLC 加载路径经 clean_prices（run.py / load_ic_data）
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research.ic.forward_return import build_forward_return
from research.ic.universe import build_ic_tradability_mask


DATES = pd.date_range("2024-01-02", periods=10, freq="B")
CODES = ["000001", "000002"]


def _prices() -> pd.DataFrame:
    # 单调上涨，便于算 forward return
    base = np.arange(1, len(DATES) + 1, dtype=float)
    return pd.DataFrame(
        {c: base * (1.0 + 0.01 * i) for i, c in enumerate(CODES)},
        index=DATES,
    )


def test_build_forward_return_masks_buy_day_limit_up_open():
    """买日 t+1 一字涨停 → 信号日 t 的 forward_return 为 NaN。"""
    prices = _prices()
    open_ = prices.copy() * 0.99
    period = 3

    # 第 4 个交易日（index=3）对 000001 一字涨停 → 信号日 index=2 应被屏蔽
    limit_up_open = pd.DataFrame(False, index=DATES, columns=CODES)
    limit_up_open.iloc[3, 0] = True
    any_limit = pd.DataFrame(False, index=DATES, columns=CODES)
    masks = {"limit_up_open": limit_up_open, "any_limit": any_limit}

    fwd = build_forward_return(prices, open_, period, masks=masks, apply_exec_mask=True)
    assert pd.isna(fwd.iloc[2, 0]), "买日一字涨停应对齐到信号日置 NaN"
    # 另一只股票同日未涨停，应仍有有效值
    assert pd.notna(fwd.iloc[2, 1])


def test_build_forward_return_masks_sell_day_any_limit():
    """卖日 t+N 涨跌停 → 信号日 t 的 forward_return 为 NaN。"""
    prices = _prices()
    open_ = prices.copy() * 0.99
    period = 3

    any_limit = pd.DataFrame(False, index=DATES, columns=CODES)
    any_limit.iloc[5, 1] = True  # 卖日 = 信号日+3 → 信号日 index=2
    masks = {
        "limit_up_open": pd.DataFrame(False, index=DATES, columns=CODES),
        "any_limit": any_limit,
    }

    fwd = build_forward_return(prices, open_, period, masks=masks, apply_exec_mask=True)
    assert pd.isna(fwd.iloc[2, 1])
    assert pd.notna(fwd.iloc[2, 0])


def test_listing_dates_excludes_new_listings():
    """listing_dates + min_listing_days 应剔除次新股。"""
    prices = _prices()
    volume = pd.DataFrame(1000.0, index=DATES, columns=CODES)
    listing_dates = {
        "000001": pd.Timestamp("2020-01-01"),  # 老股
        "000002": pd.Timestamp("2023-12-01"),  # 相对 DATES 次新
    }
    tradable = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=None,
        stock_names=None,
        min_listing_days=252,
        listing_dates=listing_dates,
        is_st_current=None,
    )
    # 000001 全程可交易；000002 在 listed+252 之前不可交易
    assert tradable["000001"].all()
    assert not tradable["000002"].any()


def test_delist_dates_excludes_after_delist():
    """delist_dates：date > delist_date → tradable=False（与 is_delisted 一致）。"""
    prices = _prices()
    volume = pd.DataFrame(1000.0, index=DATES, columns=CODES)
    # 000002 在第 5 个交易日（index=4）退市
    delist_day = DATES[4]
    delist_dates = {"000002": delist_day}

    tradable = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=None,
        stock_names=None,
        delist_dates=delist_dates,
    )
    # 退市日当日仍可交易；之后不可交易
    assert tradable.loc[delist_day, "000002"]
    assert not tradable.loc[DATES[5]:, "000002"].any()
    # 未退市股全程可交易
    assert tradable["000001"].all()

    # 与 TradeRules.is_delisted 语义对齐
    from backtest.execution import BacktestConfig, TradeRules
    rules = TradeRules(
        config=BacktestConfig(),
        delist_dates=delist_dates,
    )
    assert not rules.is_delisted("000002", delist_day)
    assert rules.is_delisted("000002", DATES[5])


def test_ml_delegates_forward_return_to_ic():
    """strategies.ml 应复用 build_forward_return，无本地无-masks 分叉。"""
    import strategies.ml as ml

    src = inspect.getsource(ml.build_factor_dataset)
    assert "build_forward_return" in src
    assert "_compute_forward_return" not in src
    assert not hasattr(ml, "_compute_forward_return")


def test_trade_rules_passes_listing_filter():
    """Backtest TradeRules.passes_listing_filter 使用 listing_dates + 共享阈值。"""
    from backtest.execution import BacktestConfig, TradeRules
    from config.settings import MIN_LISTING_DAYS

    cfg = BacktestConfig()  # 默认 = MIN_LISTING_DAYS
    assert cfg.min_listing_days == MIN_LISTING_DAYS
    rules = TradeRules(
        config=cfg,
        listing_dates={"000001": pd.Timestamp("2024-01-01")},
    )
    # 未满 MIN_LISTING_DAYS 天不可过
    early = pd.Timestamp("2024-01-01") + pd.Timedelta(days=MIN_LISTING_DAYS - 1)
    ok = pd.Timestamp("2024-01-01") + pd.Timedelta(days=MIN_LISTING_DAYS)
    assert not rules.passes_listing_filter("000001", early)
    assert rules.passes_listing_filter("000001", ok)
    # 无 listing 记录的股票默认放行
    assert rules.passes_listing_filter("999999", pd.Timestamp("2024-01-30"))


def test_min_listing_days_unified_ic_ml_backtest():
    """IC / BacktestConfig 共享 MIN_LISTING_DAYS，无 252 vs 60 双标准。"""
    from backtest.execution import BacktestConfig
    from config.settings import IC_MIN_LISTING_DAYS, MIN_LISTING_DAYS

    assert MIN_LISTING_DAYS == IC_MIN_LISTING_DAYS
    assert BacktestConfig().min_listing_days == MIN_LISTING_DAYS
    assert BacktestConfig().min_listing_days != 60 or MIN_LISTING_DAYS == 60


def test_ohlc_load_paths_call_clean_prices():
    """run.py::_load_data 与 load_ic_data 对 OHLC 走 clean_ohlc_aligned（联合刺针）。"""
    import run as run_mod
    from research.ic import load_data as ic_load

    run_src = inspect.getsource(run_mod._load_data)
    assert "clean_ohlc_aligned" in run_src
    assert "_open_raw" in run_src and "_high_raw" in run_src and "_low_raw" in run_src

    ic_src = inspect.getsource(ic_load.load_ic_data)
    assert "clean_ohlc_aligned" in ic_src
    assert "_open_raw" in ic_src and "_high_raw" in ic_src and "_low_raw" in ic_src


def test_clean_prices_applies_to_ohlc_panels():
    """OHLC：零价→NaN；默认不 ffill；联合刺针以 close 判定后四价一并 NaN。"""
    from data.clean import clean_prices, clean_ohlc_aligned

    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    # 第 3 日为零价 → 置 NaN；默认不 ffill，保持 NaN
    raw = pd.DataFrame(
        {"000001": [10.0, 10.1, 0.0, 10.2, 10.3, 10.4]},
        index=idx,
    )
    cleaned = clean_prices(raw.copy(), label="open_hfq")
    assert cleaned.iloc[2, 0] != 0
    assert pd.isna(cleaned.iloc[2, 0])
    # 显式开启 ffill 时才前填
    filled = clean_prices(raw.copy(), label="open_hfq", ffill_limit=5)
    assert filled.iloc[2, 0] == pytest.approx(filled.iloc[1, 0])
    assert (cleaned.dropna()["000001"] > 0).all()

    # 联合清洗：close 异常日（±100%/刺针）→ open 同步 NaN
    close = pd.DataFrame(
        {"000001": [10.0, 10.0, 100.0, 10.0, 10.0, 10.0]},  # day2 |ret|>100%
        index=idx,
    )
    open_ = close.copy()
    open_.iloc[2, 0] = 10.0  # open 本身正常，但应随 close 一并 NaN
    c2, o2, _, _ = clean_ohlc_aligned(close, open_, None, None)
    assert pd.isna(c2.iloc[2, 0])
    assert pd.isna(o2.iloc[2, 0])
