"""
factors/alpha_ops.py — Alpha158 / Alpha191 / Alpha101 共用时序与截面算子

纯 pandas/numpy，不引入 Qlib / jq / phandas。
约定：面板为 date × code；截面算子沿 axis=1；时序算子沿 axis=0（rolling）。

内存注意
--------
- 算子尽量返回新面板而不隐式保留输入副本；调用方算完应 ``del`` 中间结果。
- 禁止在 import 时预计算；无模块级大矩阵缓存。
- ``rolling.apply`` 已改为向量化 / ``sliding_window_view``（按列分块），
  避免对全市场面板做 Python 逐窗回调产生巨临时对象。
- 输出默认为输入同 dtype；需要 float32 时由调用方 / ``_normalize`` 转换。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-12
# 滑动窗口按列分块，限制 (window × chunk) 临时峰值
_COL_CHUNK = 128


def rank(sr: pd.DataFrame) -> pd.DataFrame:
    """截面百分比排名 [0, 1]。"""
    return sr.rank(axis=1, pct=True)


def delay(sr: pd.DataFrame, period: int) -> pd.DataFrame:
    return sr.shift(period)


def delta(sr: pd.DataFrame, period: int) -> pd.DataFrame:
    return sr.diff(period)


def corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return x.rolling(window, min_periods=mp).corr(y)


def cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return x.rolling(window, min_periods=mp).cov(y)


def ts_sum(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(1, int(window * 0.7))
    return sr.rolling(window, min_periods=mp).sum()


def ts_mean(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(1, int(window * 0.5))
    return sr.rolling(window, min_periods=mp).mean()


def ts_std(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.5))
    return sr.rolling(window, min_periods=mp).std()


def ts_max(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(1, int(window * 0.7))
    return sr.rolling(window, min_periods=mp).max()


def ts_min(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(1, int(window * 0.7))
    return sr.rolling(window, min_periods=mp).min()


def ts_prod(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动乘积：正值用 exp(sum(log))；否则分块 sliding 乘积（含 NaN 窗→NaN）。"""
    mp = max(1, int(window * 0.7))
    vals = sr.to_numpy(dtype=np.float64, copy=False)
    finite = vals[np.isfinite(vals)]
    if finite.size > 0 and finite.min() > 0:
        log_sr = np.log(sr)
        return np.exp(log_sr.rolling(window, min_periods=mp).sum())

    n_rows, n_cols = vals.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=sr.index, columns=sr.columns)
    for c0 in range(0, n_cols, _COL_CHUNK):
        c1 = min(c0 + _COL_CHUNK, n_cols)
        block = vals[:, c0:c1]
        sw = sliding_window_view(block, window_shape=window, axis=0)
        has_nan = np.isnan(sw).any(axis=-1)
        prod = np.prod(sw, axis=-1)
        prod = np.where(has_nan, np.nan, prod)
        # min_periods：非 NaN 计数
        cnt = np.sum(~np.isnan(sw), axis=-1)
        prod = np.where(cnt >= mp, prod, np.nan)
        start = window - 1
        out[start : start + sw.shape[0], c0:c1] = prod
        del sw
    return pd.DataFrame(out, index=sr.index, columns=sr.columns)


def ts_rank(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """窗口末值在窗口内的百分比排名（与国泰君安 Tsrank 一致量级用 pct）。"""
    mp = max(2, int(window * 0.7))
    return sr.rolling(window, min_periods=mp).rank(pct=True)


def sign(sr: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    return np.sign(sr)


def abs_(sr: pd.DataFrame) -> pd.DataFrame:
    return sr.abs()


def log(sr: pd.DataFrame) -> pd.DataFrame:
    return np.log(sr.replace(0, np.nan))


def maximum(a, b):
    return np.maximum(a, b)


def minimum(a, b):
    return np.minimum(a, b)


def sma_gtja(sr: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """国泰君安 SMA：ewm(alpha=m/n)。"""
    return sr.ewm(alpha=m / n, adjust=False).mean()


def sequence(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.float64)




def _rolling_argext(sr: pd.DataFrame, window: int, which: str) -> pd.DataFrame:
    """
    窗口内 argmax/argmin（0=窗口起点，window-1=最新）。
    分块 sliding_window_view，峰值约 O(window × chunk_cols)。
    """
    vals = np.asarray(sr.to_numpy(dtype=np.float64, copy=False))
    n_rows, n_cols = vals.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=sr.index, columns=sr.columns)

    nan_fill = -np.inf if which == "max" else np.inf
    argfn = np.argmax if which == "max" else np.argmin

    for c0 in range(0, n_cols, _COL_CHUNK):
        c1 = min(c0 + _COL_CHUNK, n_cols)
        block = vals[:, c0:c1]
        sw = sliding_window_view(block, window_shape=window, axis=0)
        all_nan = np.isnan(sw).all(axis=-1)
        filled = np.where(np.isnan(sw), nan_fill, sw)
        idx = argfn(filled, axis=-1).astype(np.float64)
        idx[all_nan] = np.nan
        start = window - 1
        out[start : start + sw.shape[0], c0:c1] = idx
        del sw, filled
    return pd.DataFrame(out, index=sr.index, columns=sr.columns)


def _rolling_weighted_mean(sr: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """固定权重滚动加权均值；含 NaN 的窗返回 NaN（与旧 apply 一致）。"""
    window = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    vals = np.asarray(sr.to_numpy(dtype=np.float64, copy=False))
    n_rows, n_cols = vals.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=sr.index, columns=sr.columns)

    for c0 in range(0, n_cols, _COL_CHUNK):
        c1 = min(c0 + _COL_CHUNK, n_cols)
        block = vals[:, c0:c1]
        sw = sliding_window_view(block, window_shape=window, axis=0)
        has_nan = np.isnan(sw).any(axis=-1)
        # (T', C', W) · (W,) → (T', C')
        dot = np.einsum("tcw,w->tc", np.nan_to_num(sw, nan=0.0), w) / wsum
        dot[has_nan] = np.nan
        start = window - 1
        out[start : start + sw.shape[0], c0:c1] = dot
        del sw
    return pd.DataFrame(out, index=sr.index, columns=sr.columns)


def _rolling_ols_time(sr: pd.DataFrame, window: int, mode: str) -> pd.DataFrame:
    """
    对时间 0..W-1 的滚动 OLS。
    mode: 'slope' | 'rsquare' | 'resi'（最新点残差）。
    用封闭公式 + 滚动和，避免 rolling.apply。
    """
    n = window
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    ssx = float(((x - x_mean) ** 2).sum())
    if ssx == 0:
        return pd.DataFrame(np.nan, index=sr.index, columns=sr.columns)

    vals = np.asarray(sr.to_numpy(dtype=np.float64, copy=False))
    n_rows, n_cols = vals.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < n:
        return pd.DataFrame(out, index=sr.index, columns=sr.columns)

    # sum_y / sum_xy via sliding windows per chunk
    for c0 in range(0, n_cols, _COL_CHUNK):
        c1 = min(c0 + _COL_CHUNK, n_cols)
        block = vals[:, c0:c1]
        sw = sliding_window_view(block, window_shape=n, axis=0)  # (T', C', W)
        has_nan = np.isnan(sw).any(axis=-1)
        y = np.nan_to_num(sw, nan=0.0)
        sum_y = y.sum(axis=-1)
        sum_xy = np.einsum("tcw,w->tc", y, x)
        y_mean = sum_y / n
        beta = (sum_xy - y_mean * x.sum()) / ssx

        if mode == "slope":
            res = beta
        elif mode == "rsquare":
            # y_hat = y_mean + beta * (x - x_mean)
            y_hat = y_mean[..., None] + beta[..., None] * (x - x_mean)
            sse = ((y - y_hat) ** 2).sum(axis=-1)
            ssy = ((y - y_mean[..., None]) ** 2).sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                res = 1.0 - sse / ssy
            res = np.where(ssy == 0, np.nan, res)
        else:  # resi
            y_hat_last = y_mean + beta * (x[-1] - x_mean)
            res = sw[:, :, -1] - y_hat_last  # use original last (may be nan)
            # if window had nan, mark nan
        res = np.where(has_nan, np.nan, res)
        start = n - 1
        out[start : start + sw.shape[0], c0:c1] = res
        del sw, y
    return pd.DataFrame(out, index=sr.index, columns=sr.columns)


def regbeta(sr: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    """对固定自变量 x（长度=window）做滚动 OLS 斜率。"""
    window = len(x)
    x = np.asarray(x, dtype=np.float64)
    x_mean = x.mean()
    x_var = float(((x - x_mean) ** 2).sum())
    if x_var == 0:
        return pd.DataFrame(np.nan, index=sr.index, columns=sr.columns)

    vals = np.asarray(sr.to_numpy(dtype=np.float64, copy=False))
    n_rows, n_cols = vals.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=sr.index, columns=sr.columns)

    for c0 in range(0, n_cols, _COL_CHUNK):
        c1 = min(c0 + _COL_CHUNK, n_cols)
        block = vals[:, c0:c1]
        sw = sliding_window_view(block, window_shape=window, axis=0)
        has_nan = np.isnan(sw).any(axis=-1)
        y = np.nan_to_num(sw, nan=0.0)
        y_mean = y.mean(axis=-1)
        beta = np.einsum("tcw,w->tc", y - y_mean[..., None], x - x_mean) / x_var
        beta = np.where(has_nan, np.nan, beta)
        start = window - 1
        out[start : start + sw.shape[0], c0:c1] = beta
        del sw, y
    return pd.DataFrame(out, index=sr.index, columns=sr.columns)


def decay_linear(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权：权重 1..window。"""
    weights = np.arange(1, window + 1, dtype=np.float64)
    return _rolling_weighted_mean(sr, weights)


def wma_gtja(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """国泰君安 Wma：权重 0.9^k。"""
    weights = np.power(0.9, np.arange(window - 1, -1, -1, dtype=np.float64))
    return _rolling_weighted_mean(sr, weights)


def lowday(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """距窗口内最低点的天数（1=今天最低）。"""
    # argmin 0=起点 → days_from_end = window - argmin；旧实现 len-nanargmin
    arg = _rolling_argext(sr, window, "min")
    return window - arg


def highday(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """距窗口内最高点的天数（1=今天最高）。"""
    arg = _rolling_argext(sr, window, "max")
    return window - arg


def count(cond: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(1, int(window * 0.7))
    return cond.astype(np.float64).rolling(window, min_periods=mp).sum()


def sumif(sr: pd.DataFrame, window: int, cond: pd.DataFrame) -> pd.DataFrame:
    masked = sr.where(cond, 0.0)
    return ts_sum(masked, window)


def row_max(sr: pd.DataFrame) -> pd.Series:
    return sr.max(axis=1)


def row_min(sr: pd.DataFrame) -> pd.Series:
    return sr.min(axis=1)


def ts_argmax(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """窗口内最大值位置（0=窗口起点，window-1=最新）。"""
    return _rolling_argext(sr, window, "max")


def ts_argmin(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_argext(sr, window, "min")


def ts_quantile(sr: pd.DataFrame, window: int, q: float) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return sr.rolling(window, min_periods=mp).quantile(q)


def ts_slope(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """对时间 0..N-1 的滚动 OLS 斜率。"""
    return _rolling_ols_time(sr, window, "slope")


def ts_rsquare(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """对时间的滚动 R²。"""
    return _rolling_ols_time(sr, window, "rsquare")


def ts_resi(sr: pd.DataFrame, window: int) -> pd.DataFrame:
    """对时间回归的当日残差（最新点）。"""
    return _rolling_ols_time(sr, window, "resi")


def safe_div(a, b, eps: float = _EPS):
    """元素级除法，分母加 eps 防除零。"""
    return a / (b + eps)


def typical_price(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return (high + low + close) / 3.0


def empty_like(df: pd.DataFrame, fill=np.nan) -> pd.DataFrame:
    """空面板（不复制源数据），供条件赋值替代 ``df.copy(deep=True)``。"""
    return pd.DataFrame(fill, index=df.index, columns=df.columns, dtype=np.float64)


def panel_select(
    conditions: list,
    choices: list,
    default=np.nan,
    index=None,
    columns=None,
) -> pd.DataFrame:
    """``np.select`` 包装为 DataFrame，避免多次 deep copy。"""
    cond_arr = [np.asarray(c) if not hasattr(c, "to_numpy") else c.to_numpy(copy=False)
                for c in conditions]
    choice_arr = []
    for ch in choices:
        if np.isscalar(ch):
            choice_arr.append(ch)
        elif hasattr(ch, "to_numpy"):
            choice_arr.append(ch.to_numpy(copy=False))
        else:
            choice_arr.append(np.asarray(ch))
    result = np.select(cond_arr, choice_arr, default=default)
    return pd.DataFrame(result, index=index, columns=columns)


def compute_vwap(
    amount: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """
    VWAP = amount / volume；缺 amount/volume 时用典型价 (H+L+C)/3 近似。
    返回 (vwap, note)。
    """
    if amount is not None and volume is not None:
        return amount / volume.replace(0, np.nan), "amount/volume"
    if high is not None and low is not None and close is not None:
        return typical_price(high, low, close), "typical_price(H+L+C)/3"
    return None, "unavailable"
