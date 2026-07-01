"""Tradable universe masks for IC cross-sections (quantile backtest philosophy)."""
from __future__ import annotations

import pandas as pd

from backtest.execution import build_st_schedule, infer_st_codes
from config.settings import EXCLUDE_ST, IC_MIN_LISTING_DAYS, UNIVERSE_DIR


def load_stock_names() -> pd.Series | None:
    """code → name from universe/stock_list.parquet when available."""
    path = UNIVERSE_DIR / "stock_list.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "code" not in df.columns or "name" not in df.columns:
        return None
    s = df.set_index("code")["name"]
    s.index = s.index.astype(str).str.zfill(6)
    return s


def load_is_st_current() -> pd.Series | None:
    """code → is_st_current bool from universe/stock_list.parquet when available."""
    path = UNIVERSE_DIR / "stock_list.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "code" not in df.columns or "is_st_current" not in df.columns:
        return None
    s = df.set_index("code")["is_st_current"]
    s.index = s.index.astype(str).str.zfill(6)
    return s


def build_ic_tradability_mask(
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    masks: dict | None = None,
    stock_names: pd.Series | None = None,
    min_listing_days: int | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    is_st_current: pd.Series | None = None,
) -> pd.DataFrame:
    """
    bool DataFrame aligned to *prices*; True = stock tradable on signal date for IC.

    Filters (signal date):
      - ST：优先用时间序列 st_schedule（按日期精确查询，M4 修复）；
            无元数据时回退到 stock_names 含 ST 的静态集合。
      - limit up / limit down (any_limit from clean_ohlcv masks)
      - suspension: NaN/zero close or zero volume
      - optional min listing days (stub when listing_dates absent)
    """
    tradable = pd.DataFrame(True, index=prices.index, columns=prices.columns)

    px = prices.reindex(columns=tradable.columns)
    tradable &= px.notna() & (px > 0)

    if volume is not None:
        vol = volume.reindex(index=tradable.index, columns=tradable.columns)
        tradable &= vol.notna() & (vol > 0)

    if masks is not None:
        any_limit = masks.get("any_limit")
        if any_limit is not None:
            lim = any_limit.reindex(index=tradable.index, columns=tradable.columns)
            tradable &= ~lim.fillna(False)

    if EXCLUDE_ST:
        # M4 修复：构建时间序列 ST 状态，按日期精确剔除
        schedule = build_st_schedule(
            stock_names, prices.index, is_st_current=is_st_current,
        )
        if schedule is not None:
            st_aligned = schedule.reindex(
                index=tradable.index, columns=tradable.columns
            ).fillna(False)
            tradable &= ~st_aligned
        else:
            st_codes = infer_st_codes(stock_names)
            if st_codes:
                st_cols = [c for c in tradable.columns if str(c) in st_codes]
                if st_cols:
                    tradable.loc[:, st_cols] = False

    min_days = IC_MIN_LISTING_DAYS if min_listing_days is None else min_listing_days
    if min_days > 0 and listing_dates:
        for stock, listed in listing_dates.items():
            if stock not in tradable.columns:
                continue
            too_new = tradable.index < (listed + pd.Timedelta(days=min_days))
            tradable.loc[too_new, stock] = False

    return tradable


def apply_tradable_mask(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    tradable: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tradable is None:
        return factor, forward_return
    common_idx = factor.index.intersection(forward_return.index).intersection(tradable.index)
    common_cols = factor.columns.intersection(forward_return.columns).intersection(tradable.columns)
    t = tradable.reindex(index=common_idx, columns=common_cols)
    f = factor.reindex(index=common_idx, columns=common_cols).where(t)
    r = forward_return.reindex(index=common_idx, columns=common_cols).where(t)
    return f, r
