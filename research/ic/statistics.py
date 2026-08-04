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


def ic_direction_sign(mean_ic: float, *, eps: float = 1e-12) -> float:
    """Full-sample IC direction: ``+1`` / ``-1``, or ``0`` if mean≈0 / non-finite.

    Sparse gates use this so negative-IC factors are treated as if flipped to
    「higher = better」. Fallback when ``0``: skip sparse gate (selection) or
    use unsigned ``f>0`` raw trigger (:func:`trigger_cs_payoff`).
    """
    if not np.isfinite(mean_ic) or abs(float(mean_ic)) < eps:
        return 0.0
    return 1.0 if float(mean_ic) > 0 else -1.0


def win_rates(ic: pd.Series) -> tuple[float, float, float]:
    """
    Returns (sign_aligned_win_rate, positive_rate, negative_rate).

    Let ``s = sign(mean_IC)``. Sign-aligned win-rate is
    ``mean(sign(IC_t) == s)`` i.e. ``mean(IC_t * s > 0)``.
    If mean≈0, aligned falls back to positive_rate (raw).
    """
    s = ic.dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    pos = float((s > 0).mean())
    neg = float((s < 0).mean())
    direction = ic_direction_sign(float(s.mean()))
    if direction < 0:
        aligned = neg
    else:
        # direction > 0, or ≈0 → raw positive rate
        aligned = pos
    return aligned, pos, neg


def ic_payoff_ratio(ic: pd.Series) -> float:
    """旧版 IC 盈亏比（已弃用，仅兼容测试）。

    ``mean(IC|IC>0) / mean(|IC||IC<0)``。稀疏轨已改用
    :func:`trigger_cs_payoff`（触发日相对截面均值胜率）。
    """
    s = ic.dropna()
    if len(s) == 0:
        return np.nan
    pos = s[s > 0]
    neg = s[s < 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    num = float(pos.mean())
    den = float(neg.abs().mean())
    if den <= 0 or not np.isfinite(den) or not np.isfinite(num):
        return np.nan
    return num / den


def trigger_cs_payoff(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    *,
    min_trigger: int = 5,
    dates: pd.DatetimeIndex | None = None,
    direction: float = 1.0,
    tradable: pd.DataFrame | None = None,
) -> dict[str, float]:
    """触发日相对截面均值的「盈亏」指标（稀疏轨主门槛）。

    与全样本 IC 符号对齐（等价于把因子翻成「越高越好」）。每个有足够触发
    样本的交易日 ``t``：

    - ``s = sign(mean_IC)``（传入 ``direction``）；``s≈0`` 时回退 unsigned ``f>0``
    - 触发集：当日 ``f * s > 0`` 且 ``forward_return`` 有效的股票
      （``s<0`` 时触发侧为 ``f<0``）
    - ``edge_t = mean(y | f*s > 0) - mean(y)``
      其中 ``mean(y)`` = 全市场可交易 EW（可含「无因子值但可交易」的股票）
    - ``hit_t = 1{ edge_t > 0 }``
    - 截面宇宙与 IC 同口径：传入 ``tradable`` 时对 factor/y 做
      :func:`research.ic.universe.apply_tradable_mask`（ST/次新/涨跌停等）

    日期索引必须与普通 IC / 胜率对齐：传入该因子 ``ic_series`` 的有效日期
    （``compute_ic_series`` 产出，即有足够股票参与 IC 的全部交易日）。
    **禁止**只在 ``rebalance_dates`` 上算。

    汇总（对有效触发日取平均）::

        payoff_hit  = mean_t(hit_t)     # 触发日击败截面均值的胜率（主门槛）
        payoff_edge = mean_t(edge_t)    # 平均超额（诊断）

    Parameters
    ----------
    min_trigger
        单日触发且有有效 ``y`` 的最少股票数，否则跳过该日。
    dates
        评估日期；应传该因子 IC 序列索引（与胜率同一套有效交易日）。
        ``None`` 时用 factor ∩ forward 全部交易日（仍非调仓日子集）。
    direction
        全样本 IC 符号（``+1`` / ``-1``）。``≈0`` 或非有限时用 raw ``f>0``。
    tradable
        IC 可交易池 mask；与 ``compute_ic_series`` 同口径。``None`` 时依赖
        调用方已将不可交易样本的 ``y`` 置 NaN（例如 winsor 前 mask）。
    """
    if factor is None or forward_return is None or factor.empty or forward_return.empty:
        return {"payoff_hit": np.nan, "payoff_edge": np.nan, "n_days": 0}

    if tradable is not None:
        from research.ic.universe import apply_tradable_mask
        factor, forward_return = apply_tradable_mask(factor, forward_return, tradable)

    idx = factor.index.intersection(forward_return.index)
    if dates is not None:
        idx = idx.intersection(pd.DatetimeIndex(dates))
    if len(idx) == 0:
        return {"payoff_hit": np.nan, "payoff_edge": np.nan, "n_days": 0}

    cols = factor.columns.intersection(forward_return.columns)
    if len(cols) == 0:
        return {"payoff_hit": np.nan, "payoff_edge": np.nan, "n_days": 0}

    s = ic_direction_sign(float(direction)) if np.isfinite(direction) else 0.0
    # s==0 → raw unsigned f>0 (same as s=+1 for the trigger mask)
    trig_sign = s if s != 0.0 else 1.0

    f = factor.loc[idx, cols]
    y = forward_return.loc[idx, cols]
    hits: list[float] = []
    edges: list[float] = []
    for dt in idx:
        fv = f.loc[dt]
        yv = y.loc[dt]
        valid = yv.notna()
        if int(valid.sum()) < min_trigger:
            continue
        trig = valid & ((fv * trig_sign) > 0)
        n_trig = int(trig.sum())
        if n_trig < min_trigger:
            continue
        cs_mean = float(yv[valid].mean())
        trig_mean = float(yv[trig].mean())
        if not (np.isfinite(cs_mean) and np.isfinite(trig_mean)):
            continue
        edge = trig_mean - cs_mean
        edges.append(edge)
        hits.append(1.0 if edge > 0 else 0.0)

    if not hits:
        return {"payoff_hit": np.nan, "payoff_edge": np.nan, "n_days": 0}
    return {
        "payoff_hit": float(np.mean(hits)),
        "payoff_edge": float(np.mean(edges)),
        "n_days": len(hits),
    }


def recent_window_stats(
    ic: pd.Series,
    lookback: int,
) -> dict[str, float]:
    """最近 ``lookback`` 期 IC 均值 / ICIR / 正 IC 占比（诊断；新兴主门用 retention+FDR）。"""
    s = ic.dropna()
    if lookback <= 0 or len(s) == 0:
        return {"recent_ic": np.nan, "recent_icir": np.nan, "recent_pos_rate": np.nan, "n": 0}
    tail = s.iloc[-lookback:]
    n = len(tail)
    if n < 3:
        return {"recent_ic": np.nan, "recent_icir": np.nan, "recent_pos_rate": np.nan, "n": n}
    mean = float(tail.mean())
    std0 = float(tail.std(ddof=0))
    icir_v = mean / std0 if std0 > 0 else np.nan
    return {
        "recent_ic": mean,
        "recent_icir": icir_v,
        "recent_pos_rate": float((tail > 0).mean()),
        "n": n,
    }


def recent_past_icir_retention(
    ic: pd.Series,
    recent_periods: int,
) -> dict[str, float]:
    """同持仓期 IC 序列上的近期/历史 ICIR 保留率。

    ::

        IC_recent   = mean(最后 recent_periods 期 IC)
        ICIR_recent = ICIR(最后 recent_periods 期)
        ICIR_past   = ICIR(此前全部期)
        R = |ICIR_recent| / |ICIR_past|   （分母≈0 → NaN）

    用于衰减标注（非剔除）：
    ``(R < retention_min ∧ |ICIR_recent| < recent_icir_max) ∧ |IC_recent| < recent_ic_max``
    → 衰减因子（合取；默认见 ``IC_DECAY_*``，当前 0.50/0.20/0.010）。
    """
    s = prepare_ic_for_stats(ic) if ic is not None else pd.Series(dtype=float)
    empty = {
        "ic_recent": np.nan,
        "icir_recent": np.nan,
        "icir_past": np.nan,
        "retention": np.nan,
        "n_recent": 0,
        "n_past": 0,
    }
    if recent_periods <= 0 or len(s) < recent_periods + 3:
        return empty
    recent = s.iloc[-recent_periods:]
    past = s.iloc[:-recent_periods]
    if len(past) < 3 or len(recent) < 3:
        return empty
    ic_r = float(recent.mean())
    icir_r = icir(recent)
    icir_p = icir(past)
    if not (np.isfinite(icir_r) and np.isfinite(icir_p)):
        return {
            "ic_recent": ic_r,
            "icir_recent": icir_r,
            "icir_past": icir_p,
            "retention": np.nan,
            "n_recent": len(recent),
            "n_past": len(past),
        }
    den = abs(float(icir_p))
    retention = abs(float(icir_r)) / den if den > 1e-12 else np.nan
    return {
        "ic_recent": ic_r,
        "icir_recent": float(icir_r),
        "icir_past": float(icir_p),
        "retention": float(retention) if np.isfinite(retention) else np.nan,
        "n_recent": len(recent),
        "n_past": len(past),
    }


def segment_metric_trend(
    ic: pd.Series,
    segment_periods: int,
    n_segments: int = 3,
    *,
    metric: str = "icir",
    eps: float = 0.02,
) -> tuple[bool, list[float]]:
    """近 ``n_segments`` 段（每段 ``segment_periods`` 期）指标是否逐步增强。

    默认 metric=``icir`` → 各段 ``|ICIR|``；也可 ``ic`` → 各段 ``|mean IC|``。
    通过条件（允许小噪声）::

        vals[-1] > vals[0]
        ∧  ∀i: vals[i+1] + eps >= vals[i]   （单调不降）

    新兴定池可选开关；减轻但不消灭全样本选择偏差。
    """
    s = prepare_ic_for_stats(ic) if ic is not None else pd.Series(dtype=float)
    need = int(segment_periods) * int(n_segments)
    if segment_periods <= 0 or n_segments < 2 or len(s) < need:
        return False, []
    tail = s.iloc[-need:]
    vals: list[float] = []
    for i in range(n_segments):
        seg = tail.iloc[i * segment_periods:(i + 1) * segment_periods]
        if len(seg) < max(3, segment_periods // 2):
            return False, vals
        if metric == "ic":
            v = abs(float(seg.mean()))
        else:
            v = abs(float(icir(seg)))
        if not np.isfinite(v):
            return False, vals
        vals.append(v)
    if vals[-1] <= vals[0]:
        return False, vals
    for i in range(len(vals) - 1):
        if vals[i + 1] + float(eps) < vals[i]:
            return False, vals
    return True, vals


def style_reversal_fraction(
    ic: pd.Series,
    quarter_periods: int,
    *,
    abs_ic_min: float = 0.015,
) -> float:
    """最近一季内「与全样本 IC 符号相反且 |IC|>阈值」的占比。

    ::

        frac = mean_t 1{ sign(IC_t) != sign(mean_IC_full) AND |IC_t| > abs_ic_min }

    在最近 ``quarter_periods`` 个 IC 期上计算；``frac > 0.75`` → 风格逆转标签。
    """
    s = ic.dropna() if ic is not None else pd.Series(dtype=float)
    if quarter_periods <= 0 or len(s) < max(4, quarter_periods):
        return np.nan
    mean_sign = float(np.sign(s.mean()))
    if mean_sign == 0.0:
        return np.nan
    recent = s.iloc[-quarter_periods:]
    opp = (np.sign(recent.to_numpy()) != mean_sign) & (recent.abs().to_numpy() > abs_ic_min)
    return float(np.mean(opp))


def early_late_ic_ratio(ic: pd.Series) -> float:
    """早期半段 |IC均值| 相对近期半段的比：late/early（旧衰减代理，已弃用）。

    ``< 1`` 表示近期弱于早期（衰减）；过低视为衰减过大。
    """
    s = ic.dropna()
    if len(s) < 8:
        return np.nan
    mid = len(s) // 2
    early = float(s.iloc[:mid].mean())
    late = float(s.iloc[mid:].mean())
    early_abs = abs(early)
    if early_abs < 1e-12:
        return np.nan
    return abs(late) / early_abs


def ic_stability_metrics(ic: pd.Series, rolling_window: int = 12) -> dict:
    """Rolling IC std, rolling ICIR, and fraction of years with same sign as full-sample mean."""
    s = ic.dropna()
    if len(s) == 0:
        return {
            "IC滚动标准差": np.nan,
            "同向年份占比": np.nan,
            "IC滚动ICIR": np.nan,
            "最差12期IC均值": np.nan,
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
        "最差12期IC均值": round(float(roll_mean.min()), 4) if roll_mean.notna().any() else np.nan,
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
