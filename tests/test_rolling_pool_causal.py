"""
rolling-pool schedule 生成侧因果性：决策日 t 的池只能用 ``index < t`` 的 IC。

IC_t = corr(factor_t, forward_return_t) 需要 t → t+h 的收益，在 t 当日尚未实现；
若纳入硬门 / 排序 / 去重，池的构成就前视了。这里锁定「不含当日」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.rolling_pool.dedup import dedup_by_ic_corr
from research.rolling_pool.schedule import (
    PoolParams,
    build_pool_schedule,
    infer_rebalance_dates,
)
from research.rolling_pool.stats import compute_factor_stats


def _weekly(n: int, start: str = "2020-01-03") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="W-FRI")


def test_compute_factor_stats_excludes_decision_date():
    idx = _weekly(5)
    s = pd.Series([0.01, 0.01, 0.01, 0.01, 99.0], index=idx)
    st = compute_factor_stats(s, idx[-1], window=10)
    assert st.n == 4
    assert st.mean == 0.01
    # 显式 include_end 仅诊断用，会带回前视
    st_leaky = compute_factor_stats(s, idx[-1], window=10, include_end=True)
    assert st_leaky.n == 5


def test_compute_factor_stats_window_counts_only_history():
    idx = _weekly(10)
    s = pd.Series(np.arange(10, dtype=float) / 100.0, index=idx)
    st = compute_factor_stats(s, idx[-1], window=3)
    # 末 3 个严格早于决策日的点 = 第 6,7,8 号观测
    assert st.n == 3
    assert np.isclose(st.mean, np.mean([0.06, 0.07, 0.08]))


def test_dedup_by_ic_corr_excludes_decision_date():
    idx = _weekly(6)
    base = [0.02, -0.01, 0.03, -0.02, 0.01, 0.0]
    ic = {
        "a": pd.Series(base, index=idx),
        # 历史上与 a 完全相反（|corr|=1 → 应被去重），仅在决策日当天数值不同
        "b": pd.Series([-x for x in base[:-1]] + [5.0], index=idx),
    }
    kept = dedup_by_ic_corr(["a", "b"], ic, thr=0.7, end_date=idx[-1], window=10)
    assert len(kept) == 1, "决策日当天的 IC 不应参与相关计算"


def test_infer_rebalance_dates_warmup_leaves_room_for_history():
    idx = _weekly(60)
    ic = {f"f{i}": pd.Series(np.random.RandomState(i).randn(60) / 100, index=idx)
          for i in range(4)}
    dates = infer_rebalance_dates(ic, window=10)
    assert dates[0] == idx[10], "位置 i 只有 i 个历史点，需 i>=window"
    assert len(dates) == 50


def _synthetic_ic(n_dates: int = 40, n_factors: int = 6) -> dict[str, pd.Series]:
    idx = _weekly(n_dates)
    out = {}
    for i in range(n_factors):
        rs = np.random.RandomState(100 + i)
        # 不同因子给不同均值，保证部分过硬门
        vals = rs.randn(n_dates) * 0.01 + (0.03 - 0.008 * i)
        out[f"f{i}"] = pd.Series(vals, index=idx)
    return out


def test_schedule_pool_ignores_same_day_ic():
    """把最后一个决策日当天的 IC 换成极端值，该日的池必须不变。"""
    ic = _synthetic_ic()
    params = PoolParams(
        window=8, min_periods=8, abs_mean_min=0.005, abs_icir_min=0.1,
        k_max=3, cooldown_periods=0,
    )
    dates = infer_rebalance_dates(ic, window=params.window)
    assert len(dates) >= 5

    sched_a, _ = build_pool_schedule(ic, dates, params=params, progress_every=0)

    last = dates[-1]
    ic_b = {k: v.copy() for k, v in ic.items()}
    for i, (k, v) in enumerate(ic_b.items()):
        v.loc[last] = 9.0 if i % 2 == 0 else -9.0

    sched_b, _ = build_pool_schedule(ic_b, dates, params=params, progress_every=0)

    pool_a = sorted(sched_a.loc[sched_a["date"] == last, "factor"])
    pool_b = sorted(sched_b.loc[sched_b["date"] == last, "factor"])
    assert pool_a == pool_b, "决策日当天 IC 改变却影响了当期池 → 存在前视"
    assert pool_a, "冒烟：该期池不应为空"


def test_schedule_meta_flags_causality():
    ic = _synthetic_ic(30)
    params = PoolParams(
        window=8, min_periods=8, abs_mean_min=0.005, abs_icir_min=0.1, k_max=3,
    )
    dates = infer_rebalance_dates(ic, window=params.window)
    _, meta = build_pool_schedule(ic, dates, params=params, progress_every=0)
    assert meta["ic_window_excludes_decision_date"] is True
