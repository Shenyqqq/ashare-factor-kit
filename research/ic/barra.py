"""Barra pure-factor IC (orthogonalized alpha)."""
from __future__ import annotations

import gc
from typing import Callable

import numpy as np
import pandas as pd

from config.settings import INDUSTRY_REFERENCE, MIN_IC_STOCKS
from research.ic.ic_series import _to_float32_panel, spearman_ic_numpy
from utils.wls import wls_residual


# ── barra_pure checkpoint 增量合并 / 指纹 ────────────────────────────────────

def barra_pure_cache_version() -> str:
    """与 ``factors.barra_risk.BARRA_CACHE_VERSION`` 对齐；改定义时 bump → 失效 pure ckpt。"""
    from factors.barra_risk import BARRA_CACHE_VERSION
    return str(BARRA_CACHE_VERSION)


def pack_barra_pure_ckpt(
    means: dict,
    names: list,
    series: dict,
    quantile_df: pd.DataFrame | None = None,
    *,
    version: str | None = None,
) -> tuple:
    """落盘格式：``(means, barra_ctrl_names, series, quantile_df, meta)``。"""
    meta = {"barra_version": version or barra_pure_cache_version()}
    qdf = quantile_df if isinstance(quantile_df, pd.DataFrame) else pd.DataFrame()
    return (means, list(names), series, qdf, meta)


def unpack_barra_pure_ckpt(
    ckpt,
) -> tuple[dict, list, dict, pd.DataFrame, dict] | None:
    """解析 barra_pure checkpoint；缺 pure 序列则返回 None（须重算）。"""
    if not isinstance(ckpt, tuple) or len(ckpt) < 3:
        return None
    if not isinstance(ckpt[2], dict):
        return None
    means = ckpt[0] if isinstance(ckpt[0], dict) else {}
    names = list(ckpt[1]) if ckpt[1] is not None else []
    series = ckpt[2]
    qdf = (
        ckpt[3]
        if len(ckpt) >= 4 and isinstance(ckpt[3], pd.DataFrame)
        else pd.DataFrame()
    )
    meta = ckpt[4] if len(ckpt) >= 5 and isinstance(ckpt[4], dict) else {}
    return means, names, series, qdf, meta


def barra_pure_version_ok(meta: dict | None, *, for_incremental: bool) -> bool:
    """指纹校验：版本不匹配 → 必须全量 pure。

    - 增量合并：无版本元数据也视为不兼容（避免错误复用旧口径）。
    - 普通 resume：无版本时祖父兼容（旧 3/4-tuple ckpt 仍可续跑）。
    """
    ver = (meta or {}).get("barra_version")
    current = barra_pure_cache_version()
    if ver is None:
        return not for_incremental
    return str(ver) == current


def missing_barra_pure_names(
    wanted,
    pure_series: dict,
) -> list[str]:
    """``wanted`` 中尚无有效 pure IC 序列的因子名。"""
    out: list[str] = []
    for n in wanted:
        s = pure_series.get(n) if pure_series else None
        if s is None or not hasattr(s, "__len__") or len(s) == 0:
            out.append(n)
    return out


def merge_barra_pure_results(
    base_means: dict,
    base_series: dict,
    base_qdf: pd.DataFrame | None,
    new_means: dict,
    new_series: dict,
    new_qdf: pd.DataFrame | None,
) -> tuple[dict, dict, pd.DataFrame]:
    """把新区 pure 结果合并进已有库（同名以新值为准）。"""
    means = {**(base_means or {}), **(new_means or {})}
    series = {**(base_series or {}), **(new_series or {})}
    base = base_qdf if isinstance(base_qdf, pd.DataFrame) else pd.DataFrame()
    new = new_qdf if isinstance(new_qdf, pd.DataFrame) else pd.DataFrame()
    if new.empty:
        qdf = base.copy() if not base.empty else pd.DataFrame()
    elif base.empty:
        qdf = new.copy()
    else:
        overlap = new.index.intersection(base.index)
        kept = base.drop(index=overlap, errors="ignore")
        qdf = pd.concat([kept, new], axis=0)
        qdf = qdf[~qdf.index.duplicated(keep="last")]
    return means, series, qdf


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


def unpack_date_ctrl(cached) -> tuple:
    """``date_ctrl[date]`` → ``(ctrl_arr, ctrl_idx, fwd_arr, w_arr)``。

    历史缓存 / 单测里的 3 元组（无回归权重）自动补 ``w_arr=None``（等权 OLS），
    保持向后兼容。
    """
    if cached is None:
        return None, None, None, None
    if len(cached) >= 4:
        return cached[0], cached[1], cached[2], cached[3]
    ctrl_arr, ctrl_idx, fwd_arr = cached
    return ctrl_arr, ctrl_idx, fwd_arr, None


def precompute_ctrl_matrices(
    barra_factors: dict,
    forward_return: pd.DataFrame,
    industry_map: pd.Series | None = None,
    dates: pd.DatetimeIndex | None = None,
    industry_reference: str | None = None,
    industry_panel: pd.DataFrame | None = None,
    weight_panel: pd.DataFrame | None = None,
) -> dict:
    """
    Cache control matrices per rebalance date: (ctrl_arr, ctrl_idx, fwd_arr, w_arr).

    PIT 支持：当 industry_panel 不为 None 时，逐调仓日按 load_industry_as_of
    取当期行业构建哑变量，避免用当前静态 industry_map 回填历史截面（PIT 泄漏）。
    panel 为 None 时回退到静态 industry_map 一次构建 + 按位置索引的快速路径。

    weight_panel: 截面回归权重面板（date × stock，本仓库口径 = **√市值**，
    见 ``factors/barra_risk.py::barra_regression_weights``）。传入后残差化
    改用 WLS，抑制小微盘噪声主导风格系数；None 时退化为等权 OLS。

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
        w_arr = None
        if weight_panel is not None and date in weight_panel.index:
            w_arr = (
                weight_panel.loc[date]
                .reindex(ctrl_idx)
                .values.astype(np.float32)
            )
        date_ctrl[date] = (ctrl_arr, ctrl_idx, fwd_arr, w_arr)

    return date_ctrl


def compute_pure_ic_fast(
    factor: pd.DataFrame,
    date_ctrl: dict,
    rebalance_dates: pd.DatetimeIndex,
    min_stocks: int | None = None,
) -> pd.Series:
    """Pure alpha IC on rebalance dates using precomputed control matrices.

    残差化用 WLS（权重 = √市值，由 ``date_ctrl`` 携带）；无权重时退化等权 OLS。
    """
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
        ctrl_arr, ctrl_idx, fwd_arr, w_arr = unpack_date_ctrl(cached)
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
        w_v = w_arr[valid] if w_arr is not None else None

        resid = wls_residual(f_v, X, w_v)
        if resid is None:
            continue

        ic = spearman_ic_numpy(
            resid.astype(np.float32, copy=False), y_fwd, min_n,
        )
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
    prices_raw: pd.DataFrame | None = None,
    quantile_decomp: bool = False,
    quantile_y_mode: str = "residual",
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    turnover_rate: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict[str, pd.Series], pd.DataFrame]:
    """Compute pure IC per factor; optionally Q1/Q5 long-short contribution.

    Returns
    -------
    (pure_ic_means, pure_ic_series, quantile_df)
        ``pure_ic_means``: {name: mean_pure_ic}
        ``pure_ic_series``: {name: pure IC Series on rebalance dates}
        ``quantile_df``: 多头/空头贡献表（``quantile_decomp=False`` 时为空 DataFrame）
        近窗新兴/衰减/逆转标注须用 ``pure_ic_series``，与全样本纯 IC 同口径。

    industry_panel: 若提供 PIT 行业面板，则按当期行业构建哑变量（消除 PIT 泄漏）；
                    为 None 时回退到静态 industry_map。
    names_sink:     若提供，将实际使用的 Barra 控制因子名追加进该列表（供 JSON 元数据记录）。
    clean_ret:      涨跌停日 return=NaN 的清洁收益；传入后 Barra_Beta/ResVol/Momentum
                    用其替代 prices.pct_change()，避免涨跌停截断污染。
    quantile_decomp:
        True 时在同一 OLS/WLS 残差上做 Q1/Q5 多空贡献分解（见 ``quantile_decomp``
        模块）。分组前按时序 pure IC 均值对齐方向（负则翻转 resid_x），使 Q5=
        预测收益更高一侧；「无效」仅数值病态，不因负 IC 未翻转。
    quantile_y_mode:
        ``residual``（默认）对 y 再残差化；``raw`` 用 pure IC 同款 forward return。
    circ_mv / total_mv:
        日频流通市值 / 总市值面板。用途有两个：(1) ``Barra_Size`` 的主数据源
        （log 流通市值）；(2) 截面残差化回归的 **WLS 权重 = √市值**。缺失时
        Size 降级为 log(total_assets) 且回归退化等权 OLS，均会 warning。
    turnover_rate / amount:
        日频换手率 / 成交额面板，供 ``Barra_Liquidity``（63/252 日换手率等权平均）。
    """
    from factors.barra_risk import barra_regression_weights, get_barra_factors
    from research.ic.quantile_decomp import (
        compute_quantile_ls_from_resid_loop,
        quantile_decomp_table,
        summarize_quantile_ls,
    )

    mkt = market_prices
    if mkt is None:
        # 市场代理优先中证全指（与 run.py / load_ic_data 同口径）
        from config.settings import RAW_DIR
        for fname in ("csi_all.parquet", "index_000300.parquet",
                      "index_399006.parquet"):
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
        prices_raw=prices_raw,
        circ_mv=circ_mv,
        total_mv=total_mv,
        turnover_rate=turnover_rate,
        amount=amount,
    )
    if not barra_factors:
        return {}, {}, pd.DataFrame()

    # WLS 权重 = √市值（同一日截面）；None → 等权 OLS（barra_regression_weights 已 warning）
    weight_panel = barra_regression_weights(
        prices, circ_mv=circ_mv, total_mv=total_mv,
    )

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
        weight_panel=weight_panel,
    )
    del barra_factors, weight_panel
    gc.collect()

    barra_names = [n for n in summary_index if n in registry]
    do_q = bool(quantile_decomp)
    y_mode = (quantile_y_mode or "residual").lower().strip()

    def _one(name):
        fac = registry[name]
        if do_q:
            pure, daily_q = compute_quantile_ls_from_resid_loop(
                fac, date_ctrl, rebalance_dates, y_mode=y_mode,
            )
            q_sum = summarize_quantile_ls(daily_q)
        else:
            pure = compute_pure_ic_fast(fac, date_ctrl, rebalance_dates)
            q_sum = None
        mean_val = pure.mean() if len(pure) > 0 else np.nan
        return name, mean_val, pure, q_sum

    pure_ic_means: dict[str, float] = {}
    pure_ic_series: dict[str, pd.Series] = {}
    q_summaries: dict[str, dict] = {}
    for name, mean_val, series, q_sum in parallel_fn(
        _one, barra_names, barra_workers, progress_every=10
    ):
        pure_ic_means[name] = mean_val
        if series is not None and len(series) > 0:
            pure_ic_series[name] = series
        if q_sum is not None:
            q_summaries[name] = q_sum

    quantile_df = (
        quantile_decomp_table(q_summaries, y_mode=y_mode) if q_summaries else pd.DataFrame()
    )

    del date_ctrl
    gc.collect()
    return pure_ic_means, pure_ic_series, quantile_df
