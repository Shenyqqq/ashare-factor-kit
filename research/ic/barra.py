"""Barra pure-factor IC (orthogonalized alpha)."""
from __future__ import annotations

import gc
from typing import Callable

import numpy as np
import pandas as pd

from config.settings import INDUSTRY_REFERENCE, MIN_IC_STOCKS
from research.ic.ic_series import _to_float32_panel, spearman_ic_numpy


def _industry_dummies(
    industry_map: pd.Series,
    stock_index: pd.Index,
    reference: str | None = None,
    industry_panel: pd.DataFrame | None = None,
    date: pd.Timestamp | None = None,
) -> dict[str, pd.Series]:
    """
    Industry dummy columns for cross-section OLS.

    INDUSTRY_REFERENCE:
      drop_first — drop first sorted category (default, matches v1)
      <label>    — drop named category as reference

    PIT 支持：当 industry_panel 不为 None 且 date 不为 None 时，按当期行业
    （load_industry_as_of(panel, date)）取每只股票的 sw_l2，避免用当前静态
    映射回填历史截面（PIT 泄漏）。panel 为 None 时回退到静态 industry_map。
    """
    if industry_panel is not None and date is not None:
        from data.industry.download_industry import load_industry_as_of
        ind_series = load_industry_as_of(industry_panel, date, level="sw_l2")
        ind = ind_series.reindex(stock_index).fillna("未分类")
    else:
        ind = industry_map.reindex(stock_index).fillna("未分类")
    cats = sorted(ind.unique())
    if len(cats) <= 1:
        return {}
    ref = reference if reference and reference != "drop_first" else cats[0]
    if ref not in cats:
        ref = cats[0]
    return {
        f"_ind_{grp}": (ind == grp).astype(float)
        for grp in cats if grp != ref
    }


def precompute_ctrl_matrices(
    barra_factors: dict,
    forward_return: pd.DataFrame,
    industry_map: pd.Series | None = None,
    dates: pd.DatetimeIndex | None = None,
    industry_reference: str | None = None,
    industry_panel: pd.DataFrame | None = None,
) -> dict:
    """
    Cache control matrices per rebalance date: (ctrl_arr, ctrl_idx, fwd_arr).

    PIT 支持：当 industry_panel 不为 None 时，逐调仓日按 load_industry_as_of
    取当期行业构建哑变量，避免用当前静态 industry_map 回填历史截面（PIT 泄漏）。
    panel 为 None 时回退到静态 industry_map 一次构建 + 按位置索引的快速路径。

    TODO: cache QR / (X'X)^-1 X' per date for multi-factor Barra pass —
    currently recomputes lstsq per factor per date (acceptable for ~30 factors).
    """
    target_dates = dates if dates is not None else forward_return.index
    ref = industry_reference or INDUSTRY_REFERENCE
    use_pit = industry_panel is not None

    # 静态路径：行业哑变量只构建一次
    ind_arr = None
    ind_index = None
    if not use_pit and industry_map is not None:
        ind_full = industry_map.fillna("未分类")
        cats = sorted(ind_full.unique())
        drop = cats[0] if ref == "drop_first" else ref
        if drop not in cats:
            drop = cats[0]
        ind_cols = {
            f"_ind_{grp}": (ind_full == grp).astype(np.float32)
            for grp in cats if grp != drop
        }
        if ind_cols:
            ind_df = pd.DataFrame(ind_cols)
            ind_index = ind_df.index
            ind_arr = ind_df.values.astype(np.float32)

    if use_pit:
        from data.industry.download_industry import load_industry_as_of

    date_ctrl = {}
    for date in target_dates:
        if date not in forward_return.index:
            continue
        ctrl_cols = {}
        for bname, bdf in barra_factors.items():
            if date in bdf.index:
                ctrl_cols[bname] = bdf.loc[date].astype(np.float32)
        if not ctrl_cols:
            continue

        barra_df = pd.DataFrame(ctrl_cols).fillna(0.0)

        if use_pit:
            # PIT 路径：按当期行业构建哑变量（每调仓日独立 OLS，列可不同）
            ind_series = load_industry_as_of(industry_panel, date, level="sw_l2")
            ind = ind_series.reindex(barra_df.index).fillna("未分类")
            cats_d = sorted(ind.unique())
            if len(cats_d) > 1:
                drop_d = cats_d[0] if ref == "drop_first" else ref
                if drop_d not in cats_d:
                    drop_d = cats_d[0]
                cols_d = {
                    f"_ind_{g}": (ind == g).astype(np.float32)
                    for g in cats_d if g != drop_d
                }
                if cols_d:
                    ind_part = (
                        pd.DataFrame(cols_d, index=barra_df.index)
                        .values.astype(np.float32)
                    )
                    ctrl_arr = np.hstack([
                        barra_df.values.astype(np.float32), ind_part
                    ])
                else:
                    ctrl_arr = barra_df.values.astype(np.float32)
            else:
                ctrl_arr = barra_df.values.astype(np.float32)
        elif ind_arr is not None:
            positions = ind_index.get_indexer(barra_df.index)
            valid = positions >= 0
            if not valid.all():
                barra_df = barra_df.iloc[valid]
                positions = positions[valid]
            ind_part = ind_arr[positions]
            ctrl_arr = np.hstack([barra_df.values.astype(np.float32), ind_part])
        else:
            ctrl_arr = barra_df.values.astype(np.float32)

        ctrl_idx = barra_df.index
        fwd_arr = (
            forward_return.loc[date]
            .reindex(ctrl_idx)
            .values.astype(np.float32)
        )
        date_ctrl[date] = (ctrl_arr, ctrl_idx, fwd_arr)

    return date_ctrl


def compute_pure_ic_fast(
    factor: pd.DataFrame,
    date_ctrl: dict,
    rebalance_dates: pd.DatetimeIndex,
    min_stocks: int | None = None,
) -> pd.Series:
    """Pure alpha IC on rebalance dates using precomputed control matrices."""
    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    results = {}
    factor_f32 = _to_float32_panel(factor)
    if factor_f32.empty or len(factor_f32.columns) == 0:
        return pd.Series(dtype=float)
    factor_arr = factor_f32.to_numpy()
    date_to_row = {d: i for i, d in enumerate(factor_f32.index)}
    factor_cols = factor_f32.columns

    for date in rebalance_dates:
        cached = date_ctrl.get(date)
        if cached is None:
            continue
        ctrl_arr, ctrl_idx, fwd_arr = cached
        row_i = date_to_row.get(date)
        if row_i is None:
            continue

        f_row = factor_arr[row_i]
        col_pos = factor_cols.get_indexer(ctrl_idx)
        f_vals = f_row[np.maximum(col_pos, 0)]
        valid = (col_pos >= 0) & np.isfinite(f_vals) & np.isfinite(fwd_arr)
        if valid.sum() < min_n:
            continue

        f_v = f_vals[valid].astype(np.float32, copy=False)
        X = ctrl_arr[valid]
        y_fwd = fwd_arr[valid]
        A = np.column_stack([np.ones(len(f_v), dtype=np.float32), X])

        try:
            coef, _, _, _ = np.linalg.lstsq(A, f_v, rcond=None)
            resid = f_v - A @ coef
        except Exception:
            continue

        ic = spearman_ic_numpy(resid, y_fwd, min_n)
        if np.isfinite(ic):
            results[date] = ic

    return pd.Series(results)


def run_barra_pure_ic(
    registry: dict,
    summary_index: pd.Index,
    prices: pd.DataFrame,
    financial: pd.DataFrame | None,
    forward_return: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    industry_map: pd.Series | None,
    volume: pd.DataFrame | None,
    market_prices: pd.DataFrame | None,
    parallel_fn: Callable,
    barra_workers: int,
    log_fn: Callable[[str, float], float] | None = None,
    industry_panel: pd.DataFrame | None = None,
    names_sink: list | None = None,
    clean_ret: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute mean pure IC per factor; returns {name: mean_pure_ic}.

    industry_panel: 若提供 PIT 行业面板，则按当期行业构建哑变量（消除 PIT 泄漏）；
                    为 None 时回退到静态 industry_map。
    names_sink:     若提供，将实际使用的 Barra 控制因子名追加进该列表（供 JSON 元数据记录）。
    clean_ret:      涨跌停日 return=NaN 的清洁收益；传入后 Barra_Beta/ResVol/Momentum
                    用其替代 prices.pct_change()，避免涨跌停截断污染。
    """
    from factors.barra_risk import get_barra_factors

    mkt = market_prices
    if mkt is None:
        from config.settings import RAW_DIR
        for fname in ("index_000300.parquet", "index_399006.parquet"):
            p = RAW_DIR / fname
            if p.exists():
                mkt = pd.read_parquet(p)
                break

    # industry_map 透传给 get_barra_factors 做 Barra 因子自身行业中性化（P1-3），
    # 让控制变量先剔除行业残余成分再残差化 alpha 因子；PIT 模式下保持 None 不做静态中性化。
    barra_factors = get_barra_factors(
        prices=prices,
        financial=financial,
        market_prices=mkt,
        volume=volume,
        clean_ret=clean_ret,
        industry_map=industry_map,
    )
    if not barra_factors:
        return {}

    if names_sink is not None:
        names_sink.extend(list(barra_factors.keys()))

    if log_fn:
        log_fn("Barra 风格因子", 0)

    date_ctrl = precompute_ctrl_matrices(
        barra_factors,
        forward_return,
        industry_map,
        dates=rebalance_dates,
        industry_panel=industry_panel,
    )
    del barra_factors
    gc.collect()

    barra_names = [n for n in summary_index if n in registry]

    def _one(name):
        pure = compute_pure_ic_fast(registry[name], date_ctrl, rebalance_dates)
        mean_val = pure.mean() if len(pure) > 0 else np.nan
        return name, mean_val

    pure_ic_means = {}
    for name, mean_val in parallel_fn(_one, barra_names, barra_workers, progress_every=10):
        pure_ic_means[name] = mean_val

    del date_ctrl
    gc.collect()
    return pure_ic_means
