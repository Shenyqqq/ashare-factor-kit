"""Tradable universe masks for IC cross-sections (quantile backtest philosophy)."""
from __future__ import annotations

import pandas as pd

from backtest.execution import build_st_schedule, infer_st_codes
from config.settings import (
    EXCLUDE_ST,
    MIN_LISTING_DAYS,
    UNIVERSE_DIR,
    resolve_exclude_limit_on_signal,
)


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


def load_listing_dates() -> dict[str, pd.Timestamp] | None:
    """code → list_date from universe/stock_list.parquet when available."""
    path = UNIVERSE_DIR / "stock_list.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "code" not in df.columns or "list_date" not in df.columns:
        return None
    from backtest.execution import build_listing_dates_from_stock_list
    out = build_listing_dates_from_stock_list(df)
    return out or None


def load_delist_dates() -> dict[str, pd.Timestamp] | None:
    """code → delist_date from universe/stock_list.parquet when available.

    复用 backtest.execution.build_delist_dates_from_stock_list，与回测
    TradeRules.is_delisted 同口径。
    """
    path = UNIVERSE_DIR / "stock_list.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "code" not in df.columns or "delist_date" not in df.columns:
        return None
    from backtest.execution import build_delist_dates_from_stock_list
    out = build_delist_dates_from_stock_list(df)
    return out or None


def load_st_history() -> pd.DataFrame | None:
    """加载真实 ST 历史长表（data/raw/st_history.parquet），无文件时返回 None。"""
    from config.settings import RAW_DIR
    path = RAW_DIR / "st_history.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def build_ic_tradability_mask(
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    masks: dict | None = None,
    stock_names: pd.Series | None = None,
    min_listing_days: int | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    is_st_current: pd.Series | None = None,
    small_cap_mask: pd.DataFrame | None = None,
    st_history: pd.DataFrame | None = None,
    exclude_limit_on_signal: bool | None = None,
    tradable_limit_mode: str | None = None,
) -> pd.DataFrame:
    """
    bool DataFrame aligned to *prices*; True = stock tradable on signal date for IC.

    Filters (signal date):
      - ST：优先用真实 st_history 时间序列；否则 build_st_schedule 保守实现；
            再回退到 stock_names 含 ST 的静态集合。
      - limit up / limit down (any_limit) — **仅 strict 模式**；research 保留涨跌停
      - suspension: NaN/zero close or zero volume
      - optional min listing days (stub when listing_dates absent)
      - delisted: date > delist_date → 不可交易（与 TradeRules.is_delisted 一致）

    exclude_limit_on_signal : 信号日是否因 any_limit 剔除（默认 False=research）
    tradable_limit_mode : strict|research 便捷别名（两者均 True/False 时覆盖上述开关）
    """
    exclude_limit = resolve_exclude_limit_on_signal(
        exclude_limit_on_signal, tradable_limit_mode
    )
    tradable = pd.DataFrame(True, index=prices.index, columns=prices.columns)

    px = prices.reindex(columns=tradable.columns)
    tradable &= px.notna() & (px > 0)

    if volume is not None:
        vol = volume.reindex(index=tradable.index, columns=tradable.columns)
        tradable &= vol.notna() & (vol > 0)

    if exclude_limit and masks is not None:
        any_limit = masks.get("any_limit")
        if any_limit is not None:
            lim = any_limit.reindex(index=tradable.index, columns=tradable.columns)
            tradable &= ~lim.fillna(False)

    if EXCLUDE_ST:
        # 与回测 TradeRules 同口径：st_history → st_schedule 按日期精确剔除
        hist = st_history if st_history is not None else load_st_history()
        schedule = build_st_schedule(
            stock_names,
            prices.index,
            is_st_current=is_st_current,
            delist_dates=delist_dates,
            st_history=hist,
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

    min_days = MIN_LISTING_DAYS if min_listing_days is None else min_listing_days
    if min_days > 0 and listing_dates:
        for stock, listed in listing_dates.items():
            if stock not in tradable.columns:
                continue
            too_new = tradable.index < (listed + pd.Timedelta(days=min_days))
            tradable.loc[too_new, stock] = False

    # 与 TradeRules.is_delisted 同语义：date > delist_date → 不可交易
    if delist_dates:
        for stock, d in delist_dates.items():
            if stock not in tradable.columns or d is None or pd.isna(d):
                continue
            after = tradable.index > pd.Timestamp(d)
            tradable.loc[after, stock] = False

    if small_cap_mask is not None:
        sc = small_cap_mask.reindex(
            index=tradable.index, columns=tradable.columns
        ).fillna(False)
        tradable &= sc

    return tradable


def mask_scores_for_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    open_: pd.DataFrame | None = None,
    hold_period: int | None = None,
    volume: pd.DataFrame | None = None,
    masks: dict | None = None,
    stock_names: pd.Series | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    is_st_current: pd.Series | None = None,
    st_history: pd.DataFrame | None = None,
    score_universe: str = "strict",
) -> pd.DataFrame:
    """Restrict ML score panel to the backtest ranking / EW-benchmark universe.

    Research tradable (signal-day keeps limits) + no label exec-mask correctly
    expand **training labels**.  ``get_cross_section`` also gates **prediction**
    on non-NaN labels, so research mode silently enlarges the score panel
    (~100 names/day).  Those names then enter Q1–Q5 / equal-weight benchmark
    even though execution still blocks limit-up opens — inflating every track
    including the benchmark.

    Default ``score_universe='strict'`` restores the pre-research-v2 prediction
    gate on the score panel **after** training:

      strict signal-day tradable (any_limit excluded)
      ∩ label-exec-mask availability (buy-day 一字涨停 / sell-day 涨跌停 → NaN)

    so backtest coverage matches old ``--tradable-strict --label-exec-mask``
    runs.  Pass ``'train'`` to keep the training tradable coverage end-to-end.
    """
    mode = str(score_universe or "strict").strip().lower()
    if mode in ("train", "research", "as_trained", "none", "off"):
        return scores
    if mode not in ("strict", "backtest", "signal_strict"):
        raise ValueError(
            f"score_universe must be 'strict' or 'train', got {score_universe!r}"
        )
    tradable = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=masks,
        stock_names=stock_names,
        listing_dates=listing_dates,
        delist_dates=delist_dates,
        is_st_current=is_st_current,
        st_history=st_history,
        exclude_limit_on_signal=True,
    )
    gate = tradable.reindex(index=scores.index, columns=scores.columns).fillna(False)

    # Match legacy get_cross_section gate: y = exec-masked forward return
    if hold_period is not None and hold_period > 0 and masks is not None:
        from research.ic.forward_return import build_forward_return

        fwd = build_forward_return(
            prices,
            open_,
            int(hold_period),
            masks=masks,
            apply_exec_mask=True,
        )
        gate = gate & fwd.reindex(index=scores.index, columns=scores.columns).notna()

    return scores.where(gate)


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
