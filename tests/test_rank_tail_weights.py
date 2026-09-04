"""截面分位 U 形 sample_weight：中间 0.6、两端 1.0。"""
from __future__ import annotations

import numpy as np

from models.wf.labels import rank_tail_sample_weights


def test_rank_tail_endpoints_and_mid():
    y = np.array([0.0, 0.5, 1.0])
    w = rank_tail_sample_weights(y, mid_weight=0.6)
    assert w.shape == y.shape
    assert abs(float(w[0]) - 1.0) < 1e-9
    assert abs(float(w[1]) - 0.6) < 1e-9
    assert abs(float(w[2]) - 1.0) < 1e-9


def test_rank_tail_raw_return_is_ranked_first():
    y_raw = np.array([-2.0, 0.0, 3.0])
    w = rank_tail_sample_weights(y_raw, mid_weight=0.6)
    assert abs(float(w[0]) - 1.0) < 1e-9
    assert abs(float(w[1]) - 0.6) < 1e-9
    assert abs(float(w[2]) - 1.0) < 1e-9


def test_rank_tail_mid_one_is_off():
    y = np.array([0.0, 0.5, 1.0])
    w = rank_tail_sample_weights(y, mid_weight=1.0)
    assert np.allclose(w, 1.0)
