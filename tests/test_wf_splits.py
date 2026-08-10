"""Unit tests for walk-forward split utilities."""
import pandas as pd
import pytest

from models.wf.splits import (
    get_window_splits,
    purge_train_indices,
    embargo_train_end,
    hold_period_to_embargo_periods,
)
from models.trainer import months_to_rebalance_periods, WalkForwardTrainer


def test_shared_recent_val_same_for_both_windows():
    """Default: both train windows share val [idx-V, idx); train abuts val left."""
    idx, val_w = 100, 9  # ~2 months weekly
    ts6, te6, vs6, ve6 = get_window_splits(
        idx, 26, val_w, 200, min_train_window=26, window_specific_val=True,
    )
    ts12, te12, vs12, ve12 = get_window_splits(
        idx, 52, val_w, 200, min_train_window=26, window_specific_val=True,
    )
    # Shared val glued to pred
    assert (vs6, ve6) == (idx - val_w, idx)
    assert (vs12, ve12) == (vs6, ve6)
    # Train right end = val left end
    assert te6 == vs6
    assert te12 == vs12
    # Only W differs
    assert te6 - ts6 == 26
    assert te12 - ts12 == 52
    assert ts12 < ts6


def test_val_window_zero_train_abuts_pred():
    """val_window=0: train [idx-W, idx), empty val; purge still applies vs pred."""
    idx, W = 100, 26
    ts, te, vs, ve = get_window_splits(
        idx, W, 0, 200, min_train_window=W, window_specific_val=True,
    )
    assert (ts, te) == (idx - W, idx)
    assert vs == ve == idx  # empty val slice
    assert te == vs  # train right end glued to pred (no gap for val)

    # Both windows still share empty val and abut pred
    ts12, te12, vs12, ve12 = get_window_splits(
        idx, 52, 0, 200, min_train_window=26, window_specific_val=True,
    )
    assert (te12, vs12, ve12) == (idx, idx, idx)
    assert te12 - ts12 == 52


def test_months_to_periods_preserves_zero_val():
    """Weekly conversion must not lift val_window=0 to 1 period."""
    assert months_to_rebalance_periods(0, "W-FRI") == 0
    assert months_to_rebalance_periods(0, "ME") == 0
    assert months_to_rebalance_periods(2, "W-FRI") == 9


def test_multi_window_val0_requires_average():
    """Multi-window + val=0 + ic_weighted must raise (no silent NaN IC weights)."""
    with pytest.raises(ValueError, match="average"):
        WalkForwardTrainer(
            train_windows=[6, 12],
            val_window=0,
            train_window_units="periods",
            wf_selection="ic_weighted",
            model_types=["ridge"],
        )
    # average is allowed
    t = WalkForwardTrainer(
        train_windows=[6, 12],
        val_window=0,
        train_window_units="periods",
        wf_selection="average",
        model_types=["ridge"],
    )
    assert t.val_window == 0


def test_single_window_val0_ok():
    """Single window + val=0: no val IC needed (identity)."""
    t = WalkForwardTrainer(
        train_windows=[13],
        val_window=0,
        train_window_units="periods",
        wf_selection="ic_weighted",
        model_types=["ridge"],
    )
    assert t.val_window == 0
    assert t.train_windows == [13]


def test_legacy_offset_val_differs_by_window():
    """Deprecated window_specific_val=False: longer W pushes val earlier."""
    idx, val_w = 100, 6
    ts6, te6, vs6, ve6 = get_window_splits(
        idx, 6, val_w, 120, min_train_window=6, window_specific_val=False,
    )
    ts12, te12, vs12, ve12 = get_window_splits(
        idx, 12, val_w, 120, min_train_window=6, window_specific_val=False,
    )
    assert ve6 == idx
    assert ve12 == idx - val_w
    assert te12 < te6
    assert te6 == vs6
    assert te12 == vs12


def test_embargo_train_end():
    assert embargo_train_end(50, 3) == 47
    assert embargo_train_end(2, 5) == 0


def test_purge_removes_overlapping_labels():
    dates = pd.date_range("2020-01-01", periods=20, freq="W-FRI")
    train = dates[:10].tolist()
    val = dates[10:12].tolist()
    purged = purge_train_indices(train, val, dates[12], dates.tolist(), hold_period_days=5)
    assert len(purged) <= len(train)


def test_purge_empty_val_still_purges_pred():
    """val=[] must still purge train labels that overlap pred (val_window=0)."""
    dates = pd.date_range("2020-01-01", periods=20, freq="W-FRI").tolist()
    train = dates[:12]
    pred = dates[12]
    # hold≈2 周 → 尾部训练日标签与 pred 重叠，应被 purge
    purged = purge_train_indices(train, [], pred, dates, hold_period_days=10)
    assert len(purged) < len(train)
    assert purged[-1] < pred


def test_hold_period_to_embargo_periods():
    dates = pd.date_range("2020-01-01", periods=30, freq="W-FRI").tolist()
    assert hold_period_to_embargo_periods(5, dates) >= 1


def test_is_retrain_step_default_every_period():
    from models.trainer import is_retrain_step
    for off in range(20):
        assert is_retrain_step(off, 1, has_cached_models=True) is True
        assert is_retrain_step(off, 1, has_cached_models=False) is True


def test_is_retrain_step_quarterly():
    from models.trainer import is_retrain_step
    every = 13
    # no cache → always fit
    assert is_retrain_step(1, every, has_cached_models=False) is True
    # with cache: only offsets 0, 13, 26, ...
    assert is_retrain_step(0, every, has_cached_models=True) is True
    assert is_retrain_step(1, every, has_cached_models=True) is False
    assert is_retrain_step(12, every, has_cached_models=True) is False
    assert is_retrain_step(13, every, has_cached_models=True) is True
    assert is_retrain_step(26, every, has_cached_models=True) is True


def test_retrain_every_init_and_reject_zero():
    t = WalkForwardTrainer(
        train_windows=[13],
        val_window=0,
        train_window_units="periods",
        model_types=["ridge"],
        retrain_every=13,
    )
    assert t.retrain_every == 13
    with pytest.raises(ValueError, match="retrain_every"):
        WalkForwardTrainer(
            train_windows=[13],
            val_window=0,
            train_window_units="periods",
            model_types=["ridge"],
            retrain_every=0,
        )


if __name__ == "__main__":
    test_shared_recent_val_same_for_both_windows()
    test_val_window_zero_train_abuts_pred()
    test_months_to_periods_preserves_zero_val()
    test_multi_window_val0_requires_average()
    test_single_window_val0_ok()
    test_legacy_offset_val_differs_by_window()
    test_embargo_train_end()
    test_purge_removes_overlapping_labels()
    test_purge_empty_val_still_purges_pred()
    test_hold_period_to_embargo_periods()
    print("all ok")
