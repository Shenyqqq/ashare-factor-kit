"""Rolling-window IC stats for pool hard gates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorStats:
    """Signed + absolute moments over a causal lookback window."""

    mean: float
    icir: float
    abs_mean: float
    abs_icir: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "icir": self.icir,
            "abs_mean": self.abs_mean,
            "abs_icir": self.abs_icir,
            "n": self.n,
        }


_NAN_STATS = FactorStats(
    mean=np.nan, icir=np.nan, abs_mean=np.nan, abs_icir=np.nan, n=0,
)


def compute_factor_stats(
    ic_series: pd.Series,
    end_date: pd.Timestamp | str,
    window: int = 52,
    *,
    ddof: int = 0,
    include_end: bool = False,
) -> FactorStats:
    """
    近 ``window`` 周（**严格早于** ``end_date``）的因果 IC 统计。

    为什么不含当日
    --------------
    决策日 t 的 IC_t = corr(factor_t, forward_return_t)，其中 forward_return_t
    需要 t → t+h 的收益，在 t 当日尚未实现。若把 IC_t 纳入门槛/排序，池的
    构成就用到了未来信息（前视）。因此生成侧只用 ``index < end_date``。
    消费侧 ``active_factors_by_rebalance`` 的 asof（``decision_date <= 调仓日``）
    因此是安全的：决策日 t 的池在 t 当日即可算出。

    Parameters
    ----------
    ic_series : 周频（或可索引的）signed IC 序列
    end_date  : 决策日；默认只用 ``index < end_date`` 的观测（不含当日）
    window    : 回看周数（取末尾最多 window 个非 NaN 点）
    ddof      : ICIR 分母 std 的自由度；默认 0，与 ``research.ic`` 一致
    include_end : **仅诊断用**。True 时退回旧的「含当日」口径（有前视），
        生产 schedule 生成路径不得开启。

    Returns
    -------
    FactorStats with mean, icir, abs_mean, abs_icir, n
    """
    if ic_series is None or len(ic_series) == 0:
        return _NAN_STATS
    end = pd.Timestamp(end_date)
    s = ic_series.dropna().astype(float).sort_index()
    s = s.loc[s.index <= end] if include_end else s.loc[s.index < end]
    if window is not None and window > 0:
        s = s.tail(int(window))
    n = int(len(s))
    if n == 0:
        return _NAN_STATS
    mu = float(s.mean())
    if n == 1:
        return FactorStats(
            mean=mu,
            icir=np.nan,
            abs_mean=abs(mu),
            abs_icir=np.nan,
            n=n,
        )
    sd = float(s.std(ddof=ddof))
    if not np.isfinite(sd) or sd <= 0:
        icir = np.nan
    else:
        icir = float(mu / sd)
    return FactorStats(
        mean=mu,
        icir=icir,
        abs_mean=abs(mu) if np.isfinite(mu) else np.nan,
        abs_icir=abs(icir) if np.isfinite(icir) else np.nan,
        n=n,
    )


def passes_hard_gate(
    stats: FactorStats,
    *,
    abs_mean_min: float = 0.015,
    abs_icir_min: float = 0.3,
    min_periods: int = 52,
) -> bool:
    """硬门：``|mean| > abs_mean_min`` 且 ``|ICIR| > abs_icir_min``，且样本数够。"""
    if stats.n < int(min_periods):
        return False
    if not (np.isfinite(stats.abs_mean) and np.isfinite(stats.abs_icir)):
        return False
    return bool(stats.abs_mean > abs_mean_min and stats.abs_icir > abs_icir_min)


def sort_key(stats: FactorStats) -> tuple[float, float]:
    """排序键：``|ICIR|`` 降序，并列 ``|mean|``（调用方 reverse=True）。"""
    a = float(stats.abs_icir) if np.isfinite(stats.abs_icir) else -np.inf
    b = float(stats.abs_mean) if np.isfinite(stats.abs_mean) else -np.inf
    return (a, b)
