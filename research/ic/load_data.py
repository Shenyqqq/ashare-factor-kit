"""Data loading for IC analysis v2 (mirrors ic_analysis v1 / run.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import RAW_DIR
from data.clean import clean_ohlcv
from research.ic.universe import load_is_st_current, load_stock_names


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


def _opt_parquet(fname: str) -> pd.DataFrame | None:
    p = RAW_DIR / fname
    return pd.read_parquet(p) if p.exists() else None


def load_ic_data() -> ICDataBundle:
    """Load OHLCV + auxiliary panels and run clean_ohlcv."""
    prices = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    fin_path = RAW_DIR / "financial_indicators.parquet"
    financial = pd.read_parquet(fin_path) if fin_path.exists() else None

    prices_raw = _opt_parquet("prices_raw.parquet")
    volume = _opt_parquet("volume.parquet")
    amount = _opt_parquet("amount.parquet")
    open_ = _opt_parquet("open_hfq.parquet")
    high = _opt_parquet("high_hfq.parquet")
    low = _opt_parquet("low_hfq.parquet")
    margin = _opt_parquet("margin_balance.parquet")
    moneyflow = _opt_parquet("moneyflow_large.parquet")
    northbound = _opt_parquet("northbound_holding.parquet")
    institution = _opt_parquet("institution_holding.parquet")
    market_prices = _opt_parquet("csi_all.parquet")
    if market_prices is None:
        market_prices = _opt_parquet("index_000300.parquet")
    if market_prices is None:
        market_prices = _opt_parquet("csi300.parquet")
    industry_map_df = _opt_parquet("industry_map.parquet")

    clean_ret, masks = clean_ohlcv(prices, open_, high, low)
    stock_names = load_stock_names()
    is_st_current = load_is_st_current()

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


def load_industry_panel() -> pd.DataFrame | None:
    """
    加载 PIT 行业面板（industry_map_panel.parquet）。

    返回长表 DataFrame[code, effective_date, sw_l1, sw_l2, end_date]；
    若文件不存在（旧版下载产物或 fallback 路径下未生成），返回 None。
    下游 barra.py / ic_analysis.py 在 panel 为 None 时自动回退到静态 industry_map。
    """
    from data.industry.download_industry import (
        PANEL_PATH,
        load_industry_panel as _load,
    )
    if not PANEL_PATH.exists():
        return None
    return _load()
