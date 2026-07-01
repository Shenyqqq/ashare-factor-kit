"""
IC summary statistics for research/ic_analysis_v2.

ICIR: mean(IC) / std(IC, ddof=0) — population std used consistently here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import IC_CLIP, IC_WINSORIZE_PCT


def prepare_ic_for_stats(ic: pd.Series) -> pd.Series:
    """Clip or winsorize IC series before ICIR / t-stats."""
    s = ic.dropna().copy()
    if len(s) == 0:
        return s
    if IC_CLIP and IC_CLIP > 0:
        return s.clip(-IC_CLIP, IC_CLIP)
    if IC_WINSORIZE_PCT is not None:
        lo, hi = IC_WINSORIZE_PCT
        ql, qh = s.quantile([lo, hi])
        return s.clip(ql, qh)
    return s


def icir(ic: pd.Series, ddof: int = 0) -> float:
    """ICIR = mean(IC) / std(IC, ddof). Default ddof=0 (population)."""
    s = prepare_ic_for_stats(ic)
    if len(s) == 0:
        return np.nan
    std = s.std(ddof=ddof)
    if std <= 0 or not np.isfinite(std):
        return 0.0
    return float(s.mean() / std)


def _default_nw_lags(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def newey_west_t(ic: pd.Series, lags: int | None = None) -> float:
    """
    HAC t-stat for H0: mean(IC)=0.

    Default lags ≈ floor(4*(n/100)^(2/9)); uses statsmodels OLS(HAC) when available.
    """
    s = prepare_ic_for_stats(ic)
    y = s.values.astype(float)
    n = len(y)
    if n < 3:
        return np.nan
    if lags is None:
        lags = _default_nw_lags(n)

    X = np.ones((n, 1))
    try:
        from statsmodels.regression.linear_model import OLS

        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(model.tvalues[0])
    except Exception:
        return _simple_hac_t(y, lags)


def _simple_hac_t(y: np.ndarray, lags: int) -> float:
    """Fallback Newey-West style t-stat without statsmodels."""
    n = len(y)
    mean = y.mean()
    u = y - mean
    gamma0 = (u ** 2).sum() / n
    lr_var = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        cov = (u[lag:] * u[:-lag]).sum() / n
        lr_var += 2 * w * cov
    if lr_var <= 0:
        return 0.0
    se = np.sqrt(lr_var / n)
    return float(mean / se) if se > 0 else 0.0


def win_rates(ic: pd.Series) -> tuple[float, float, float]:
    """
    Returns (sign_aligned_win_rate, positive_rate, negative_rate).

    Sign-aligned: if mean(IC)<0 use (IC<0).mean(), else (IC>0).mean().
    """
    s = ic.dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    pos = float((s > 0).mean())
    neg = float((s < 0).mean())
    aligned = neg if s.mean() < 0 else pos
    return aligned, pos, neg


def ic_stability_metrics(ic: pd.Series, rolling_window: int = 12) -> dict:
    """Rolling IC std, rolling ICIR, and fraction of years with same sign as full-sample mean."""
    s = ic.dropna()
    if len(s) == 0:
        return {
            "IC滚动标准差": np.nan,
            "同向年份占比": np.nan,
            "IC滚动ICIR": np.nan,
        }
    min_p = max(3, rolling_window // 2)
    roll_std = s.rolling(rolling_window, min_periods=min_p).std(ddof=0)
    roll_mean = s.rolling(rolling_window, min_periods=min_p).mean()
    roll_icir = roll_mean / roll_std.where(roll_std > 0)
    mean_sign = np.sign(s.mean())
    yearly = s.groupby(s.index.year).mean()
    if mean_sign == 0 or len(yearly) == 0:
        sign_pct = np.nan
    else:
        sign_pct = float((np.sign(yearly) == mean_sign).mean())
    return {
        "IC滚动标准差": round(float(roll_std.mean()), 4) if roll_std.notna().any() else np.nan,
        "同向年份占比": round(sign_pct, 4) if np.isfinite(sign_pct) else np.nan,
        "IC滚动ICIR": round(float(roll_icir.mean()), 4) if roll_icir.notna().any() else np.nan,
    }


def ic_stats(ic: pd.Series) -> dict:
    """Full-period IC statistics (v2)."""
    if len(ic.dropna()) == 0:
        return {
            "IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
            "胜率": np.nan, "正IC占比": np.nan, "负IC占比": np.nan,
            "|IC|均值": np.nan, "t统计量": np.nan, "NW_t统计量": np.nan,
            "样本数": 0,
        }
    s = prepare_ic_for_stats(ic)
    std0 = float(s.std(ddof=0))
    t_iid = s.mean() / (std0 / np.sqrt(len(s))) if std0 > 0 else 0.0
    aligned, pos, neg = win_rates(ic)
    stab = ic_stability_metrics(ic)
    return {
        "IC均值": round(float(s.mean()), 4),
        "IC标准差": round(std0, 4),
        "ICIR": round(icir(ic), 4),
        "胜率": round(aligned, 4),
        "正IC占比": round(pos, 4),
        "负IC占比": round(neg, 4),
        "|IC|均值": round(float(s.abs().mean()), 4),
        "t统计量": round(float(t_iid), 2),
        "NW_t统计量": round(float(newey_west_t(ic)), 2),
        "样本数": len(s),
        **stab,
    }


def ic_by_year(ic: pd.Series) -> pd.Series:
    return ic.groupby(ic.index.year).mean().round(4)


def benjamini_hochberg(t_stats: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Given t-statistics, compute two-sided p-values via scipy.stats.norm and
    return a boolean array indicating which hypotheses are significant at
    FDR <= alpha.

    Returns
    -------
    np.ndarray  shape (len(t_stats),), dtype=bool — True if rejected (significant).
    """
    from scipy.stats import norm

    t = np.asarray(t_stats, dtype=float)
    n = t.size
    if n == 0:
        return np.array([], dtype=bool)

    p = 2.0 * (1.0 - norm.cdf(np.abs(t)))
    order = np.argsort(p)                       # ascending p
    p_sorted = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha  # BH critical values

    # step-up: find largest k s.t. p_sorted[k-1] <= thresh[k-1]
    below = p_sorted <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    k_max = int(np.max(np.nonzero(below)[0]))   # index in sorted array
    # all hypotheses with rank <= k_max+1 are rejected
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: k_max + 1] = True

    out = np.zeros(n, dtype=bool)
    out[order] = rejected_sorted
    return out
