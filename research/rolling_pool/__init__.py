"""
research.rolling_pool — 周频 Barra pure IC 轮动定池。

规则摘要
--------
- 窗口：决策日前近 ``window`` 周 pure IC（``index < t``，不含当日；默认 52≈1y，两年用 ``--lookback 2y``）
- 硬门：``|mean(IC)| > abs_mean_min`` 且 ``|ICIR| > abs_icir_min``
  （ICIR = mean/std，signed 序列上算再取绝对值；std 默认 ddof=0）
- 排序：``|ICIR|`` 降序，并列 ``|mean|``
- 去重：① IC 序列相关贪心；② 因子截面相关（``FactorCorrProvider``，有 panel 时启用）
- 容量：``K_max``，无下限
- 换手：``target_out = max(n_fail, floor(0.2 * n))``；不足则末位再踢
- 冷却：本轮出局默认 1 期不回流

截面去重说明见 ``corr_provider`` 模块文档：无 panel 时第二道自动跳过并在元数据中标记。

用法::

    python -m research.rolling_pool

接入回测（``run.py --rolling-pool-schedule``，默认 lazy）见 ``schedule_load`` / ``lazy``。
"""
from __future__ import annotations

from research.rolling_pool.corr_provider import (
    CachedPanelCorrProvider,
    FactorCorrProvider,
    NullFactorCorrProvider,
)
from research.rolling_pool.dedup import dedup_by_factor_cs_corr, dedup_by_ic_corr
from research.rolling_pool.lazy import (
    RollingPoolPanelStore,
    assert_panel_files_exist,
    pool_features_for_date,
    union_active_factors,
)
from research.rolling_pool.schedule import build_pool_schedule
from research.rolling_pool.schedule_load import (
    apply_schedule_mask,
    cs_zscore_sparse_rows,
    load_pool_schedule,
    load_panels_prefer_cache,
    schedule_pools_by_date,
    schedule_union,
)
from research.rolling_pool.stats import compute_factor_stats, passes_hard_gate

__all__ = [
    "CachedPanelCorrProvider",
    "FactorCorrProvider",
    "NullFactorCorrProvider",
    "RollingPoolPanelStore",
    "apply_schedule_mask",
    "assert_panel_files_exist",
    "build_pool_schedule",
    "compute_factor_stats",
    "cs_zscore_sparse_rows",
    "dedup_by_factor_cs_corr",
    "dedup_by_ic_corr",
    "load_panels_prefer_cache",
    "load_pool_schedule",
    "passes_hard_gate",
    "pool_features_for_date",
    "schedule_pools_by_date",
    "schedule_union",
    "union_active_factors",
]
