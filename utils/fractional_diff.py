"""
utils/fractional_diff.py — AFML Ch5 Fractional Differencing

分数阶差分：用 d∈(0,1) 部分差分价格序列，保留长期记忆同时平稳。
d=1 等价于 pct_change（完全差分，丢失记忆）；d=0 等价于原始价格（不平稳）。
d=0.3-0.5 通常在保留记忆和平稳性之间取得好平衡。

实现 AFML Eq 5.5-5.7 的宽表向量化版本：
  x_t = Σ_k w_k * x_{t-k}
  w_k = (-1)^k * C(d, k)  （二项系数）
  权重 w_k 衰减，超过 threshold 后截断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# 权重序列
# ══════════════════════════════════════════════════════════════════════════════

def frac_diff_weights(d: float, threshold: float = 1e-5,
                      max_lag: int = 100) -> np.ndarray:
    """
    计算分数差分权重序列 w_k = (-1)^k * C(d, k)。

    利用递推关系 w_k = -w_{k-1} * (d - k + 1) / k，避免直接算阶乘。
    当 |w_k| < threshold 时停止增长（权重已可忽略）。
    max_lag 作为硬上限，防止极端 d 下无限循环。

    Returns
    -------
    np.ndarray, shape=(L,), L 为达到 threshold 时的权重长度（含 w_0=1）
    """
    if not (0.0 <= d <= 1.0):
        logger.warning(f"frac_diff_weights: d={d} 不在 [0,1]，仍按公式计算")

    w = [1.0]
    for k in range(1, max_lag + 1):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            # 此时权重已小到可忽略，停止扩展
            break
        w.append(w_k)
    return np.array(w, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Expanding-window 分数差分（AFML Snippet 5.1）
# ══════════════════════════════════════════════════════════════════════════════

def frac_diff(series: pd.DataFrame, d: float, threshold: float = 1e-5,
              max_lag: int = 100) -> pd.DataFrame:
    """
    Expanding-window 分数差分（AFML Eq 5.5）。

    对宽表 DataFrame（index=日期, columns=股票）做分数差分：
      x_t = Σ_{k=0}^{min(t, L-1)} w_k * x_{t-k}
    其中 L 为 frac_diff_weights(d, threshold) 的长度。
    早期行（t < L-1）使用部分权重（expanding），完整窗口从 t=L-1 开始；
    然而为了与 AFML 标准实现一致并保证早期权重不完整时数据可用性，
    这里前 L-1 行设为 NaN（权重不完整，避免偏差）。

    Returns
    -------
    pd.DataFrame, 与输入同 shape，前 L-1 行为 NaN
    """
    weights = frac_diff_weights(d, threshold=threshold, max_lag=max_lag)
    L = len(weights)

    arr = series.values.astype(np.float64)  # (T, N)
    T, N = arr.shape

    # 短序列保护：权重窗口超过数据长度时截断
    if L > T:
        logger.warning(
            f"frac_diff: 权重窗口 L={L} > 数据行数 T={T}，截断权重到 T。"
        )
        weights = weights[:T]
        L = T

    out = np.full_like(arr, np.nan)

    # 对每个时间点 t，做加权求和（仅当 t >= L-1 时权重完整）
    for t in range(L - 1, T):
        window = arr[t - L + 1: t + 1]      # (L, N)，最早到最近
        # weights[0] 对应 x_{t-L+1}（最远），weights[-1] 对应 x_t（最近）
        # 即 x_t = Σ_k w_k * x_{t-k}，k=0 对应最新
        # 因此需要把 weights 反转后与 window 逐元素相乘
        out[t] = (window[::-1] * weights[:, None]).sum(axis=0)

    return pd.DataFrame(out, index=series.index, columns=series.columns)


# ══════════════════════════════════════════════════════════════════════════════
# FFD: Fixed-Width Window 分数差分（AFML Snippet 5.2）
# ══════════════════════════════════════════════════════════════════════════════

def frac_diff_ffd(series: pd.DataFrame, d: float,
                  threshold: float = 1e-5) -> pd.DataFrame:
    """
    FFD（Fixed-Width Window）分数差分。

    与 expanding-window 不同，FFD 用 threshold 一次性确定固定窗口宽度 L，
    之后所有时间点都用同一组长度为 L 的权重做卷积，速度更快且对长序列稳定。
    AFML Snippet 5.2 的实现。

    前 L-1 行为 NaN（窗口不完整）。

    Returns
    -------
    pd.DataFrame, 与输入同 shape，前 L-1 行为 NaN
    """
    weights = frac_diff_weights(d, threshold=threshold, max_lag=10_000)
    L = len(weights)

    arr = series.values.astype(np.float64)  # (T, N)
    T, N = arr.shape

    # 短序列保护：若权重窗口超过数据长度，截断权重到 T 并告警。
    # 此时实际 threshold 被放松，但避免抛错；正常长序列（T >> L）不受影响。
    if L > T:
        logger.warning(
            f"frac_diff_ffd: 权重窗口 L={L} > 数据行数 T={T}，"
            f"截断权重到 T（threshold 实际被放松）。"
            f"建议加大 threshold 或使用更长序列。"
        )
        weights = weights[:T]
        L = T

    if L < 1:
        # 不可能（weights 至少含 w_0=1.0），保险起见
        return pd.DataFrame(np.nan, index=series.index, columns=series.columns)

    # 向量化卷积：用 sliding_window_view + einsum，避免 Python 循环
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view(arr, L, axis=0) -> (T-L+1, N, L)
    windows = sliding_window_view(arr, window_shape=L, axis=0)  # (T-L+1, N, L)
    # windows[i, :, m] 对应 arr[i + m]，m=0 是最早（应乘以 weights[0]?）
    # 我们要 out[t] = Σ_k w_k * arr[t-k]，t = i + L - 1
    # 即 out[i+L-1] = Σ_{k=0}^{L-1} w_k * arr[i+L-1-k]
    #              = Σ_{m=0}^{L-1} w_{L-1-m} * arr[i+m]
    # 所以 windows[i, :, m] * weights[L-1-m]，对 m 求和
    out = (windows * weights[::-1][None, None, :]).sum(axis=-1)  # (T-L+1, N)

    full = np.full_like(arr, np.nan)
    full[L - 1:] = out

    return pd.DataFrame(full, index=series.index, columns=series.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 最优 d 搜索（ADF 平稳性检验）
# ══════════════════════════════════════════════════════════════════════════════

def optimal_d(series: pd.DataFrame, d_range: np.ndarray = None,
              threshold: float = 1e-5,
              p_value_threshold: float = 0.01,
              max_stocks: int = 20) -> tuple[float, pd.DataFrame]:
    """
    搜索最优 d：使序列平稳的最小分数差分阶数。

    对 d ∈ d_range（默认 [0.0, 0.1, ..., 1.0]）依次做 FFD，
    对多只股票（最多 max_stocks 只，避免太慢）做 ADF 检验，
    取中位 p-value；返回第一个使 median_p < p_value_threshold 的 d。

    若所有 d 都不平稳（极端情况），返回最后一个 d=1.0。

    Returns
    -------
    (best_d, adf_table)
      best_d : float
      adf_table : pd.DataFrame, columns=['d', 'median_p_value', 'n_stocks']
    """
    from statsmodels.tsa.stattools import adfuller

    if d_range is None:
        d_range = np.arange(0.0, 1.01, 0.1)

    # 选 max_stocks 只非NaN较多的股票，避免稀疏列拉低检验功效
    stock_counts = series.notna().sum().sort_values(ascending=False)
    selected = stock_counts.head(max_stocks).index.tolist()
    sub = series[selected]

    rows = []
    best_d = float(d_range[-1])

    for d in d_range:
        fd = frac_diff_ffd(sub, d=d, threshold=threshold)
        pvals = []
        for col in fd.columns:
            s = fd[col].dropna()
            # ADF 需要足够样本，且不能全相同
            if len(s) < 30 or s.nunique() < 2:
                continue
            try:
                # autolag=None 用固定 lag=1 提速；或用默认 AIC
                p = adfuller(s, autolag="AIC")[1]
                pvals.append(p)
            except Exception as e:
                logger.debug(f"ADF 失败 d={d} col={col}: {e}")
                continue

        if not pvals:
            rows.append({"d": d, "median_p_value": np.nan,
                         "n_stocks": 0})
            continue

        median_p = float(np.median(pvals))
        rows.append({"d": d, "median_p_value": median_p,
                     "n_stocks": len(pvals)})

        if median_p < p_value_threshold and best_d == float(d_range[-1]):
            best_d = float(d)

    adf_table = pd.DataFrame(rows)
    logger.info(
        f"optimal_d: 选择 d={best_d:.2f} (p<{p_value_threshold})，"
        f"ADF 表:\n{adf_table.to_string(index=False)}"
    )
    return best_d, adf_table
