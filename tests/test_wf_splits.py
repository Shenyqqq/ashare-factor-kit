"""Unit tests for walk-forward split utilities."""
import pandas as pd

from models.wf.splits import (
    get_window_splits,
    purge_train_indices,
    embargo_train_end,
    hold_period_to_embargo_periods,
)


def test_window_specific_val_differs_by_window():
    idx, val_w = 100, 6
    ts6, te6, vs6, ve6 = get_window_splits(idx, 6, val_w, 120, min_train_window=6)
    ts12, te12, vs12, ve12 = get_window_splits(idx, 12, val_w, 120, min_train_window=6)
    assert ve6 == idx
    assert ve12 == idx - val_w
    assert te12 < te6


def test_v1_compat_shared_val():
    idx, val_w = 100, 6
    ts, te, vs, ve = get_window_splits(
        idx, 12, val_w, 120, window_specific_val=False,
    )
    assert ve == idx
    assert vs == idx - val_w
    assert te == vs


def test_embargo_train_end():
    assert embargo_train_end(50, 3) == 47
    assert embargo_train_end(2, 5) == 0


def test_purge_removes_overlapping_labels():
    dates = pd.date_range("2020-01-01", periods=20, freq="W-FRI")
    train = dates[:10].tolist()
    val = dates[10:12].tolist()
    purged = purge_train_indices(train, val, dates[12], dates.tolist(), hold_period_days=5)
    assert len(purged) <= len(train)


def test_hold_period_to_embargo_periods():
    dates = pd.date_range("2020-01-01", periods=30, freq="W-FRI").tolist()
    assert hold_period_to_embargo_periods(5, dates) >= 1


if __name__ == "__main__":
    test_window_specific_val_differs_by_window()
    test_v1_compat_shared_val()
    test_embargo_train_end()
    test_purge_removes_overlapping_labels()
    test_hold_period_to_embargo_periods()
    print("all ok")
