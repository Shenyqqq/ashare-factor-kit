"""Unit tests for Learning-to-Rank fine-rank label prep (objective=rank)."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from models.wf.labels import cross_sectional_rank
from models.wf.models import fit_model, prepare_rank_labels


def test_prepare_rank_labels_matches_cs_rank_round():
    rng = np.random.default_rng(0)
    y_raw = rng.normal(size=100)
    cs = cross_sectional_rank(y_raw)
    got = prepare_rank_labels(cs, [100])
    expected = np.round(cs * 99).astype(np.int32)
    np.testing.assert_array_equal(got, expected)
    assert got.min() == 0
    assert got.max() == 99
    assert len(np.unique(got)) == 100


def test_prepare_rank_labels_multi_group_independent():
    g1 = cross_sectional_rank(np.array([0.3, -0.1, 0.5, 0.0]))
    g2 = cross_sectional_rank(np.array([1.0, 2.0]))
    y = np.concatenate([g1, g2])
    got = prepare_rank_labels(y, [4, 2])
    np.testing.assert_array_equal(
        got[:4], np.round(g1 * 3).astype(np.int32)
    )
    np.testing.assert_array_equal(
        got[4:], np.round(g2 * 1).astype(np.int32)
    )
    assert got[:4].max() == 3
    assert got[4:].max() == 1


def test_prepare_rank_labels_dense_rank_on_raw():
    """Raw returns → per-group argsort dense ranks (same path as cs_rank)."""
    y = np.array([0.1, -0.5, 2.0, 10.0, 0.0])
    got = prepare_rank_labels(y, [3, 2])
    # group0: -0.5, 0.1, 2.0 → ranks 0, 1, 2
    np.testing.assert_array_equal(got[:3], [1, 0, 2])
    # group1: 10.0, 0.0 → ranks 1, 0
    np.testing.assert_array_equal(got[3:], [1, 0])


def test_prepare_rank_labels_group_size_mismatch_raises():
    with pytest.raises(ValueError, match="sum\\(group_sizes\\)"):
        prepare_rank_labels(np.array([0.1, 0.2, 0.3]), [2])


def test_three_rankers_share_prepare_rank_labels():
    """LGBM / XGB / Cat fit paths all call the same prepare_rank_labels."""
    src = inspect.getsource(fit_model)
    assert src.count("prepare_rank_labels(") >= 3
    assert "np.digitize" not in src
    assert "N_REL" not in src

    rng = np.random.default_rng(1)
    y = np.concatenate([
        cross_sectional_rank(rng.normal(size=30)),
        cross_sectional_rank(rng.normal(size=20)),
    ])
    groups = [30, 20]
    # Single shared transform: any ranker path would see identical labels
    labels = prepare_rank_labels(y, groups)
    for _model in ("lgbm", "xgb", "cat"):
        np.testing.assert_array_equal(prepare_rank_labels(y, groups), labels)
    assert labels.dtype == np.int32
    assert set(labels[:30].tolist()) == set(range(30))
    assert set(labels[30:].tolist()) == set(range(20))
