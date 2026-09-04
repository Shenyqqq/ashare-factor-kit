"""
utils/wls.py — 截面加权最小二乘（Barra 口径）

为什么需要 WLS
--------------
Barra 风格/行业截面回归若用朴素 OLS，全市场每只股票等权。A 股 5000+ 只里
绝大多数是小微盘，其收益噪声远大于大盘股，等权 OLS 让因子暴露的估计被
小盘噪声主导，风格系数不稳、残差（"纯 alpha"）里残留系统性成分。

行业标准（MSCI Barra CNE5/CNE6、Axioma）用 **回归权重 ∝ √市值**：
    minimize  Σ_i w_i · e_i²      其中 w_i = √(市值_i)

实现上等价于把设计矩阵与被解释变量同乘 √w_i = 市值_i^(1/4) 后做普通 lstsq，
残差再还原到原始尺度（**不**返回加权空间的残差——下游要的是可比的因子值）。

本模块被 3 处共用，保证 IC 纯化与 ML 特征中性化同口径：
    research/ic/barra.py           纯因子 IC 残差化
    research/ic/quantile_decomp.py Q1/Q5 多空分解
    models/wf/labels.py            residualize_panel / residual_return_label
"""
from __future__ import annotations

import numpy as np


def normalize_weights(w: np.ndarray | None, n: int) -> np.ndarray | None:
    """把回归权重归一到均值 1；非法（全 NaN / 全 ≤0）时返回 None（调用方退化为等权）。

    归一只影响数值条件数，不影响 WLS 解。
    """
    if w is None:
        return None
    arr = np.asarray(w, dtype=np.float64).ravel()
    if arr.size != n:
        return None
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    if not np.isfinite(arr).any():
        return None
    # 缺权重的个股用截面中位数补，避免整条截面退化为等权
    med = np.nanmedian(arr)
    if not np.isfinite(med) or med <= 0:
        return None
    arr = np.where(np.isfinite(arr), arr, med)
    mean = arr.mean()
    if not np.isfinite(mean) or mean <= 0:
        return None
    return arr / mean


def wls_residual(
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    add_const: bool = True,
) -> np.ndarray | None:
    """``y ~ [1, X]`` 的加权最小二乘残差（原始尺度）。

    Parameters
    ----------
    y : (n,) 被解释变量（因子值或 forward return）
    X : (n, k) 控制矩阵（Barra 风格 + 行业哑变量），**不含**常数列
    weights : (n,) | None
        回归权重 w_i（本仓库口径 = √市值）。None 或非法 → 退化为等权 OLS。
    add_const : bool
        是否自动拼常数列。

    Returns
    -------
    np.ndarray | None
        残差（长度 n）；lstsq 失败返回 None。
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()
    X64 = np.asarray(X, dtype=np.float64)
    if X64.ndim == 1:
        X64 = X64[:, None]
    n = len(y64)
    if X64.shape[0] != n:
        return None

    A = np.column_stack([np.ones(n, dtype=np.float64), X64]) if add_const else X64

    w = normalize_weights(weights, n)
    try:
        if w is None:
            coef, _, _, _ = np.linalg.lstsq(A, y64, rcond=None)
        else:
            # WLS ≡ 对 (A, y) 同乘 √w 后做 OLS
            rw = np.sqrt(w)
            coef, _, _, _ = np.linalg.lstsq(A * rw[:, None], y64 * rw, rcond=None)
    except Exception:
        return None
    # 残差始终回到原始（未加权）尺度
    return y64 - A @ coef
