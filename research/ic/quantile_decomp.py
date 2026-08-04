"""Barra pure-IC 多头/空头贡献（Q1 vs Q5）分解。

口径（与 ``research/ic/barra.py`` pure IC 对齐）
--------------------------------------------
- 分组变量：因子对 Barra 风格 + 行业哑变量 OLS/WLS 残差 ``resid_x``
  （与 pure IC 的 x 相同）。
- **IC 方向对齐**：分位分组前，若该因子时序 pure IC 均值 < 0，则对
  ``resid_x`` 取反，使「高分 = 预测收益更高的一侧」；此后 Q5=名义多头、
  Q1=名义空头。pure IC 序列本身**不**取反（仍保留原始符号）。
- 收益 ``y``（默认 ``residual``）：同一控制矩阵上对 forward return 再残差化
  得到 ``resid_y``，再算各组等权均值——刻画风格中性后的多空贡献。
  可选 ``raw``：直接用 pure IC 的 forward return（未对 y 残差化）。
- 日期：调仓日（与 pure IC 相同）；样本需因子与 y 均有限（与 date_ctrl 一致）。
  CLI 若已对 forward_return 做 tradable mask + winsor，则与 IC 可交易池同口径。

单日指标（均在「越高越好」对齐方向上）
------------------------------------
- ``Q5_mean`` / ``Q1_mean`` / ``CS_mean``：各组 / 截面等权下期收益
- ``spread`` = Q5 - Q1
- ``多头超额`` = Q5 - CS_mean
- ``空头贡献`` = CS_mean - Q1  （空头组弱于截面 → 正贡献）
- ``long_share`` = 多头超额 / (多头超额 + 空头贡献)，仅在二者之和 > 0 时有定义

时序汇总后按 ``long_share`` 判 ``多空来源``：多头主导 / 空头主导 / 双边。
「无效」仅表示数值病态（非有限或分母≤0），**不**因负 IC 未翻转而无效。

历史 CSV：列名不变，但负 IC 因子的 Q5/Q1/long_share/多空来源语义按对齐后口径；
需重跑 ``--barra`` IC 才会更新已落盘的 ``ic_quantile_ls_*.csv`` / pure 表追加列。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    IC_QUANTILE_BILATERAL_HI,
    IC_QUANTILE_BILATERAL_LO,
    MIN_IC_STOCKS,
)
from utils.wls import wls_residual

Q_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")


def ic_align_sign(pure_ic: pd.Series | np.ndarray | float) -> float:
    """由时序 pure IC 均值得到分组对齐符号：均值 < 0 → -1，否则 +1。"""
    if isinstance(pure_ic, pd.Series):
        if pure_ic.empty:
            return 1.0
        mu = float(pure_ic.mean())
    elif np.ndim(pure_ic) > 0:
        arr = np.asarray(pure_ic, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 1.0
        mu = float(np.mean(arr))
    else:
        mu = float(pure_ic)
    if np.isfinite(mu) and mu < 0:
        return -1.0
    return 1.0


def classify_ls_source(
    long_excess: float,
    short_contrib: float,
    *,
    bilateral_lo: float | None = None,
    bilateral_hi: float | None = None,
    eps: float = 1e-12,
) -> str:
    """根据多头超额 / 空头贡献判定多空来源。

    Parameters
    ----------
    long_excess, short_contrib
        时序平均后的 ``Q5-CS``、``CS-Q1``（须已在「越高越好」方向上对齐）。
    bilateral_lo / bilateral_hi
        ``long_share`` 落在 ``[lo, hi]`` 判为双边；两侧之外分别为空头/多头主导。

    Notes
    -----
    「无效」仅当输入非有限，或 ``long_excess + short_contrib ≤ 0``（对齐后仍无正
    多空合计超额）。负 IC 因子应先翻转分组变量，不应因此被标无效。
    """
    lo = IC_QUANTILE_BILATERAL_LO if bilateral_lo is None else bilateral_lo
    hi = IC_QUANTILE_BILATERAL_HI if bilateral_hi is None else bilateral_hi
    le = float(long_excess) if np.isfinite(long_excess) else np.nan
    sc = float(short_contrib) if np.isfinite(short_contrib) else np.nan
    if not np.isfinite(le) or not np.isfinite(sc):
        return "无效"
    denom = le + sc
    if denom <= eps:
        # 对齐后多空合计仍非正：噪声/无单调，无法归因
        return "无效"
    share = le / denom
    if share > hi:
        return "多头主导"
    if share < lo:
        return "空头主导"
    return "双边"


def _qcut_labels_rank(
    scores: np.ndarray,
    n_quantiles: int = 5,
    min_per_bin: int = 1,
) -> np.ndarray | None:
    """对残差因子打 Q1..Qn 标签（0-based）。

    与回测 ``assign_quantile_groups`` 一致：先按值升序、再按位置打破平局，
    对 ``rank(method='first')`` 做 ``qcut``，避免 ties 导致 bins 塌缩。
    调用方须保证 ``scores`` 已按 IC 方向对齐（越高 = 预测收益更高）。
    """
    n = len(scores)
    if n < n_quantiles * max(1, min_per_bin):
        return None
    # 稳定排序：值升序，相同值按原始下标
    order = np.lexsort((np.arange(n), scores))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    try:
        # pd.qcut 接受 Series；labels=False → 0..k-1
        cats = pd.qcut(ranks, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return None
    labels = np.asarray(cats, dtype=np.float64)
    if not np.isfinite(labels).all():
        return None
    n_bins = int(np.nanmax(labels)) + 1
    if n_bins < n_quantiles:
        # 极端 ties / 样本不足导致 bins 少于请求数 → 跳过该日
        return None
    return labels.astype(np.int8)


def quantile_ls_one_cross_section(
    resid_x: np.ndarray,
    y: np.ndarray,
    *,
    n_quantiles: int = 5,
    min_stocks: int | None = None,
) -> dict[str, float] | None:
    """单截面：按已对齐的 resid_x 分 Q1–Q5，汇总等权收益与多空贡献。

    ``resid_x`` 须已是「越高越好」方向（负 IC 因子由调用方先取反）。
    """
    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    rx = np.asarray(resid_x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(rx) & np.isfinite(yy)
    if valid.sum() < min_n:
        return None
    rx_v = rx[valid]
    y_v = yy[valid]
    labels = _qcut_labels_rank(rx_v, n_quantiles=n_quantiles, min_per_bin=1)
    if labels is None:
        return None

    cs_mean = float(np.mean(y_v))
    q_means: dict[str, float] = {}
    for i, qname in enumerate(Q_LABELS[:n_quantiles]):
        mask = labels == i
        if not mask.any():
            return None
        q_means[qname] = float(np.mean(y_v[mask]))

    q1 = q_means["Q1"]
    q5 = q_means[Q_LABELS[n_quantiles - 1]]
    long_excess = q5 - cs_mean
    short_contrib = cs_mean - q1
    spread = q5 - q1
    denom = long_excess + short_contrib
    long_share = float(long_excess / denom) if denom > 1e-12 else np.nan

    return {
        "Q1_mean": q1,
        "Q5_mean": q5,
        "CS_mean": cs_mean,
        "spread": spread,
        "多头超额": long_excess,
        "空头贡献": short_contrib,
        "long_share": long_share,
        "n_stocks": float(len(y_v)),
    }


def residualize_y(
    y: np.ndarray, A: np.ndarray, weights: np.ndarray | None = None,
) -> np.ndarray:
    """``y ~ A`` 加权最小二乘残差（权重 = √市值）；失败时退回原 y。

    ``A`` 已含常数列，故 ``add_const=False``。``weights=None`` 时退化等权 OLS。
    """
    y64 = np.asarray(y, dtype=np.float64)
    A64 = np.asarray(A, dtype=np.float64)
    if y64.ndim != 1 or A64.ndim != 2 or len(y64) != A64.shape[0]:
        return y64
    if len(y64) < 2 * (A64.shape[1] + 1):
        return y64
    resid = wls_residual(y64, A64, weights, add_const=False)
    return y64 if resid is None else resid


def summarize_quantile_ls(
    daily: pd.DataFrame,
    *,
    bilateral_lo: float | None = None,
    bilateral_hi: float | None = None,
) -> dict[str, Any]:
    """对单因子逐日分位表做时序平均，并判定多空来源。"""
    empty = {
        "Q5_mean": np.nan,
        "Q1_mean": np.nan,
        "spread": np.nan,
        "多头超额": np.nan,
        "空头贡献": np.nan,
        "long_share": np.nan,
        "多空来源": "无效",
        "n_days": 0,
    }
    if daily is None or daily.empty:
        return empty

    cols = ["Q5_mean", "Q1_mean", "spread", "多头超额", "空头贡献", "long_share"]
    out: dict[str, Any] = {}
    for c in cols:
        if c in daily.columns:
            out[c] = float(daily[c].mean())
        else:
            out[c] = np.nan
    out["n_days"] = int(len(daily))
    # long_share 用平均超额重算，避免对日度 share 直接平均（非线性）
    out["long_share"] = (
        float(out["多头超额"] / (out["多头超额"] + out["空头贡献"]))
        if (
            np.isfinite(out["多头超额"])
            and np.isfinite(out["空头贡献"])
            and (out["多头超额"] + out["空头贡献"]) > 1e-12
        )
        else np.nan
    )
    out["多空来源"] = classify_ls_source(
        out["多头超额"],
        out["空头贡献"],
        bilateral_lo=bilateral_lo,
        bilateral_hi=bilateral_hi,
    )
    return out


def compute_quantile_ls_from_resid_loop(
    factor: pd.DataFrame,
    date_ctrl: dict,
    rebalance_dates: pd.DatetimeIndex,
    *,
    y_mode: str = "residual",
    min_stocks: int | None = None,
    n_quantiles: int = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    """在与 ``compute_pure_ic_fast`` 相同的 date_ctrl 上算 pure IC + 分位多空序列。

    分位分组前按时序 pure IC 均值对齐 ``resid_x``（均值 < 0 则取反），再算
    Q5/Q1 / long_share；pure IC 序列保持原始符号。

    Returns
    -------
    pure_ic : Series
        调仓日 pure Spearman IC（resid_x vs **raw** y，与现有 pure IC 一致；未翻转）。
    daily_q : DataFrame
        逐日分位贡献（收益侧按 ``y_mode``；分组已方向对齐）。
        ``attrs["ic_sign"]`` 为对齐符号（+1 / -1）。
    """
    from research.ic.barra import unpack_date_ctrl
    from research.ic.ic_series import _to_float32_panel, spearman_ic_numpy

    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    y_mode = (y_mode or "residual").lower().strip()
    if y_mode not in ("residual", "raw"):
        raise ValueError(f"y_mode must be 'residual' or 'raw', got {y_mode!r}")

    ic_results: dict = {}
    # 先收集逐日 resid_x / y，再按时序 IC 均值决定对齐符号后做分位
    day_bufs: list[tuple] = []

    factor_f32 = _to_float32_panel(factor)
    if factor_f32.empty or len(factor_f32.columns) == 0:
        empty_q = pd.DataFrame()
        empty_q.attrs["ic_sign"] = 1.0
        return pd.Series(dtype=float), empty_q

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
        if int(valid.sum()) < min_n:
            continue

        f_v = f_vals[valid].astype(np.float64, copy=False)
        X = ctrl_arr[valid]
        y_fwd = fwd_arr[valid].astype(np.float64, copy=False)
        w_v = w_arr[valid] if w_arr is not None else None
        A = np.column_stack([np.ones(len(f_v), dtype=np.float64), X.astype(np.float64)])

        # x 侧残差化与 compute_pure_ic_fast 同口径（WLS，权重 = √市值）
        resid_x = wls_residual(f_v, A, w_v, add_const=False)
        if resid_x is None:
            continue

        # pure IC 始终对 raw y（与现网 compute_pure_ic_fast 一致）；不翻转
        ic = spearman_ic_numpy(
            resid_x.astype(np.float32), y_fwd.astype(np.float32), min_n
        )
        if np.isfinite(ic):
            ic_results[date] = float(ic)

        y_use = residualize_y(y_fwd, A, w_v) if y_mode == "residual" else y_fwd
        day_bufs.append((date, resid_x, y_use))

    pure_ic = pd.Series(ic_results)
    sign = ic_align_sign(pure_ic)

    rows: list[dict] = []
    for date, resid_x, y_use in day_bufs:
        # 负 IC → 取反，使 Q5 = 预测收益更高一侧
        q = quantile_ls_one_cross_section(
            sign * resid_x, y_use, n_quantiles=n_quantiles, min_stocks=min_stocks
        )
        if q is not None:
            rows.append({"date": date, **q})

    daily_q = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    daily_q.attrs["ic_sign"] = float(sign)
    return pure_ic, daily_q


def quantile_decomp_table(
    summaries: dict[str, dict[str, Any]],
    *,
    y_mode: str = "residual",
) -> pd.DataFrame:
    """将 {factor: summarize_quantile_ls(...)} 整理为输出表。

    表中 Q5/多头超额等均在「越高越好」对齐方向上；负 IC 因子已翻转后再分解。
    """
    if not summaries:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(summaries, orient="index")
    df.index.name = "因子"
    # 稳定列序
    col_order = [
        "多头超额",
        "空头贡献",
        "long_share",
        "多空来源",
        "Q5_mean",
        "Q1_mean",
        "spread",
        "n_days",
    ]
    cols = [c for c in col_order if c in df.columns] + [
        c for c in df.columns if c not in col_order
    ]
    df = df[cols]
    df.attrs["y_mode"] = y_mode
    return df
