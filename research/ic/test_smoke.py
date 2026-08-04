"""Smoke tests for IC v2 statistics (run: python -m research.ic.test_smoke)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.statistics import (
    ic_direction_sign,
    ic_stats,
    icir,
    newey_west_t,
    prepare_ic_for_stats,
    win_rates,
)


def _assert_close(a, b, tol=1e-4, msg=""):
    assert abs(a - b) < tol, f"{msg}: {a} vs {b}"


def test_icir_ddof0():
    ic = pd.Series([0.05, 0.03, -0.02, 0.04, 0.01])
    expected = ic.mean() / ic.std(ddof=0)
    _assert_close(icir(ic), expected, msg="ICIR ddof=0")


def test_win_rate_sign_aware():
    pos = pd.Series([0.05, 0.03, -0.01, 0.02, 0.04])
    neg = pd.Series([-0.05, -0.03, 0.01, -0.02, -0.04])
    aligned_p, pos_r, neg_r = win_rates(pos)
    aligned_n, _, _ = win_rates(neg)
    _assert_close(aligned_p, pos_r, msg="positive mean → positive win rate")
    _assert_close(aligned_n, (neg < 0).mean(), msg="negative mean → negative win rate")
    assert pos_r > 0.5 and neg_r < 0.5
    assert ic_direction_sign(float(pos.mean())) == 1.0
    assert ic_direction_sign(float(neg.mean())) == -1.0
    assert ic_direction_sign(0.0) == 0.0


def test_ic_clip():
    ic = pd.Series(np.linspace(-1, 1, 50))
    clipped = prepare_ic_for_stats(ic)
    assert clipped.abs().max() <= 0.3 + 1e-9


def test_newey_west_t_finite():
    rng = np.random.default_rng(42)
    ic = pd.Series(rng.normal(0.03, 0.05, 120), index=pd.date_range("2020-01-01", periods=120, freq="ME"))
    t = newey_west_t(ic)
    assert np.isfinite(t)
    assert abs(t) > 0


def test_ic_stats_keys():
    ic = pd.Series([0.04, 0.02, -0.01, 0.03], index=pd.date_range("2023-01-31", periods=4, freq="ME"))
    stats = ic_stats(ic)
    for k in ("ICIR", "NW_t统计量", "胜率", "正IC占比", "负IC占比", "IC滚动标准差"):
        assert k in stats


def run_all():
    test_icir_ddof0()
    test_win_rate_sign_aware()
    test_ic_clip()
    test_newey_west_t_finite()
    test_ic_stats_keys()
    print("research.ic.test_smoke: all passed")


if __name__ == "__main__":
    run_all()
