"""Data loading for IC analysis v2 (mirrors ic_analysis v1 / run.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import RAW_DIR
from data.clean import (
    clean_amount,
    clean_aux_panel,
    clean_financial,
    clean_market_cap,
    clean_ohlc_aligned,
    clean_ohlcv,
    clean_prices,
    clean_volume,
    mask_post_delist,
    validate_amount_units,
)
from research.ic.universe import load_delist_dates, load_is_st_current, load_stock_names


@dataclass
class ICDataBundle:
    prices: pd.DataFrame
    financial: pd.DataFrame | None
    prices_raw: pd.DataFrame | None
    volume: pd.DataFrame | None
    amount: pd.DataFrame | None
    open_: pd.DataFrame | None
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    margin: pd.DataFrame | None
    moneyflow: pd.DataFrame | None
    northbound: pd.DataFrame | None
    institution: pd.DataFrame | None
    market_prices: pd.DataFrame | None
    industry_map_df: pd.DataFrame | None
    clean_ret: pd.DataFrame
    masks: dict
    stock_names: pd.Series | None
    is_st_current: pd.Series | None = None
    st_history: pd.DataFrame | None = None
    circ_mv: pd.DataFrame | None = None
    total_mv: pd.DataFrame | None = None
    turnover_rate: pd.DataFrame | None = None


def _opt_parquet(fname: str) -> pd.DataFrame | None:
    p = RAW_DIR / fname
    return pd.read_parquet(p) if p.exists() else None


def load_ic_data() -> ICDataBundle:
    """Load OHLCV + auxiliary panels；清洗栈对齐 run.py::_load_data。"""
    _close_raw = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    fin_path = RAW_DIR / "financial_indicators.parquet"
    financial = (
        clean_financial(pd.read_parquet(fin_path)) if fin_path.exists() else None
    )

    _prices_raw = _opt_parquet("prices_raw.parquet")
    prices_raw = (
        clean_prices(_prices_raw, label="prices_raw")
        if _prices_raw is not None else None
    )
    _vol_raw = _opt_parquet("volume.parquet")
    volume = clean_volume(_vol_raw, name="volume") if _vol_raw is not None else None
    _amt_raw = _opt_parquet("amount.parquet")
    amount = clean_amount(_amt_raw, name="amount") if _amt_raw is not None else None

    _open_raw = _opt_parquet("open_hfq.parquet")
    _high_raw = _opt_parquet("high_hfq.parquet")
    _low_raw = _opt_parquet("low_hfq.parquet")
    prices, open_, high, low = clean_ohlc_aligned(
        _close_raw, _open_raw, _high_raw, _low_raw,
    )

    if amount is not None and volume is not None:
        validate_amount_units(amount, volume, prices)

    delist_dates = load_delist_dates()
    if delist_dates:
        prices = mask_post_delist(prices, delist_dates)
        prices_raw = mask_post_delist(prices_raw, delist_dates)
        open_ = mask_post_delist(open_, delist_dates)
        high = mask_post_delist(high, delist_dates)
        low = mask_post_delist(low, delist_dates)
        volume = mask_post_delist(volume, delist_dates)
        amount = mask_post_delist(amount, delist_dates)

    _margin_raw = _opt_parquet("margin_balance.parquet")
    margin = (
        clean_aux_panel(_margin_raw, name="margin")
        if _margin_raw is not None else None
    )
    _moneyflow_raw = _opt_parquet("moneyflow_large.parquet")
    moneyflow = (
        clean_aux_panel(_moneyflow_raw, name="moneyflow")
        if _moneyflow_raw is not None else None
    )
    # 北向下线：不加载进 IC 默认路径（parquet 归档保留）
    # 披露约 2024-08-19 停更；勿把停更后空档当信号
    northbound = None
    logger.warning(
        "北向持股未加载（已停更≈2024-08-19；默认因子/白名单不含北向）。"
        "归档仍见 data/raw/northbound_*.parquet"
    )
    institution = _opt_parquet("institution_holding.parquet")
    market_prices = _opt_parquet("csi_all.parquet")
    if market_prices is None:
        market_prices = _opt_parquet("index_000300.parquet")
    if market_prices is None:
        market_prices = _opt_parquet("csi300.parquet")
    industry_map_df = _opt_parquet("industry_map.parquet")

    from data.mv_panels import load_mv_raw

    _total_mv_raw = load_mv_raw("total_mv")
    total_mv = (
        clean_market_cap(_total_mv_raw, name="total_mv")
        if _total_mv_raw is not None else None
    )
    _circ_mv_raw = load_mv_raw("circ_mv")
    circ_mv = (
        clean_market_cap(_circ_mv_raw, name="circ_mv")
        if _circ_mv_raw is not None else None
    )
    # Barra_Liquidity 用换手率（非成交量）；仍由 compute_market_cap 产出
    _turnover_raw = _opt_parquet("turnover_rate.parquet")
    turnover_rate = (
        clean_aux_panel(_turnover_raw, name="turnover_rate")
        if _turnover_raw is not None else None
    )

    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    from data.download import drop_excluded_universe_columns

    prices = drop_excluded_universe_columns(prices, name="prices")
    prices_raw = drop_excluded_universe_columns(prices_raw)
    open_ = drop_excluded_universe_columns(open_)
    high = drop_excluded_universe_columns(high)
    low = drop_excluded_universe_columns(low)
    volume = drop_excluded_universe_columns(volume)
    amount = drop_excluded_universe_columns(amount)
    clean_ret = drop_excluded_universe_columns(clean_ret)
    margin = drop_excluded_universe_columns(margin)
    moneyflow = drop_excluded_universe_columns(moneyflow)
    institution = drop_excluded_universe_columns(institution)
    total_mv = drop_excluded_universe_columns(total_mv)
    circ_mv = drop_excluded_universe_columns(circ_mv)
    turnover_rate = drop_excluded_universe_columns(turnover_rate)
    if masks:
        masks = {
            k: (drop_excluded_universe_columns(v) if isinstance(v, pd.DataFrame) else v)
            for k, v in masks.items()
        }

    stock_names = load_stock_names()
    is_st_current = load_is_st_current()

    st_history = None
    st_path = RAW_DIR / "st_history.parquet"
    if st_path.exists():
        st_history = pd.read_parquet(st_path)

    return ICDataBundle(
        prices=prices,
        financial=financial,
        prices_raw=prices_raw,
        volume=volume,
        amount=amount,
        open_=open_,
        high=high,
        low=low,
        margin=margin,
        moneyflow=moneyflow,
        northbound=northbound,
        institution=institution,
        market_prices=market_prices,
        industry_map_df=industry_map_df,
        clean_ret=clean_ret,
        masks=masks,
        stock_names=stock_names,
        is_st_current=is_st_current,
        st_history=st_history,
        circ_mv=circ_mv,
        total_mv=total_mv,
        turnover_rate=turnover_rate,
    )


def resolve_industry_map(
    industry_map_df: pd.DataFrame | None,
    need: bool = False,
) -> pd.Series | None:
    if industry_map_df is not None and "sw_l2" in industry_map_df.columns:
        return industry_map_df["sw_l2"]
    if not need:
        return None
    try:
        from data.industry.download_industry import load_industry_map
        return load_industry_map()["sw_l2"]
    except FileNotFoundError:
        return None


def load_industry_panel(*, required: bool = False) -> pd.DataFrame | None:
    """
    加载 PIT 行业面板（industry_map_panel.parquet）。

    Parameters
    ----------
    required : bool
        True 时文件不存在直接 raise（严格默认，禁止静默静态 fallback）。
        False 时返回 None（仅 debug / ``--allow-static-industry``）。
    """
    from data.industry.download_industry import (
        PANEL_PATH,
        load_industry_panel as _load,
    )
    if not PANEL_PATH.exists():
        if required:
            raise FileNotFoundError(
                f"缺少 PIT 行业面板: {PANEL_PATH}。"
                "请先运行 `python -m data.industry.download_industry`；"
                "禁止静默回退静态 industry_map（PIT 泄漏）。"
                "仅 debug 可加 --allow-static-industry。"
            )
        return None
    return _load()


def require_industry_panel(
    *,
    allow_static: bool = False,
) -> pd.DataFrame | None:
    """Barra / 行业中性化入口：默认强制 PIT panel。

    allow_static=True 时允许缺失（返回 None，下游可走静态 map），并打 ERROR 级日志。
    """
    panel = load_industry_panel(required=not allow_static)
    if panel is None and allow_static:
        try:
            from loguru import logger
            logger.error(
                "industry_map_panel.parquet 缺失，已启用 --allow-static-industry；"
                "行业哑变量可能含 PIT 泄漏，结果不可与严格 PIT 口径比较"
            )
        except Exception:
            pass
    return panel
