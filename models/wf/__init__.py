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
]
