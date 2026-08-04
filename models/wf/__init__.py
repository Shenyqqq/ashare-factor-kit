"""Walk-forward training submodules (trainer v2)."""

from models.wf.splits import (
    get_window_splits,
    purge_train_indices,
    embargo_train_end,
    hold_period_to_embargo_periods,
)
from models.wf.labels import transform_labels
from models.wf.ensemble import combine_model_scores, select_window_weights
from models.wf.metrics import spearman_ic, compute_drift_flags, diagnostics_to_dataframe
from models.wf.two_stage import (
    DEFAULT_STAGE2_POOL_FRAC,
    apply_two_stage_ridge,
    pool_cs_winsor_zscore,
    top_frac_index,
)
from models.wf.stage1_cache import (
    build_pool_mask,
    load_stage1_cache,
    save_stage1_cache,
)

__all__ = [
    "get_window_splits",
    "purge_train_indices",
    "embargo_train_end",
    "hold_period_to_embargo_periods",
    "transform_labels",
    "combine_model_scores",
    "select_window_weights",
    "spearman_ic",
    "compute_drift_flags",
    "diagnostics_to_dataframe",
    "DEFAULT_STAGE2_POOL_FRAC",
    "apply_two_stage_ridge",
    "pool_cs_winsor_zscore",
    "top_frac_index",
    "build_pool_mask",
    "load_stage1_cache",
    "save_stage1_cache",
]
