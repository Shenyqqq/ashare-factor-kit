"""Barra pure checkpoint：增量 merge / 指纹校验（不跑真实残差化）。"""
from __future__ import annotations

import pandas as pd
import pytest

from research.ic.barra import (
    barra_pure_cache_version,
    barra_pure_version_ok,
    merge_barra_pure_results,
    missing_barra_pure_names,
    pack_barra_pure_ckpt,
    unpack_barra_pure_ckpt,
)


def _series(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="W-FRI")
    return pd.Series(vals, index=idx, dtype=float)


def test_pack_unpack_roundtrip_with_version():
    means = {"A": 0.02, "B": -0.01}
    series = {"A": _series([0.01, 0.03]), "B": _series([-0.02, 0.0])}
    qdf = pd.DataFrame(
        {"long_share": [0.5, 0.4]},
        index=pd.Index(["A", "B"], name="factor"),
    )
    ckpt = pack_barra_pure_ckpt(means, ["Size", "Beta"], series, qdf)
    assert isinstance(ckpt, tuple) and len(ckpt) == 5
    unpacked = unpack_barra_pure_ckpt(ckpt)
    assert unpacked is not None
    m, names, s, q, meta = unpacked
    assert m == means
    assert names == ["Size", "Beta"]
    assert set(s) == {"A", "B"}
    assert list(q.index) == ["A", "B"]
    assert meta.get("barra_version") == barra_pure_cache_version()


def test_unpack_rejects_no_series():
    assert unpack_barra_pure_ckpt(({"A": 0.1}, ["Size"])) is None
    assert unpack_barra_pure_ckpt("bad") is None


def test_version_ok_resume_vs_incremental():
    cur = barra_pure_cache_version()
    assert barra_pure_version_ok({"barra_version": cur}, for_incremental=True)
    assert barra_pure_version_ok({"barra_version": cur}, for_incremental=False)
    assert not barra_pure_version_ok(
        {"barra_version": "barra_ancient"}, for_incremental=True,
    )
    assert not barra_pure_version_ok(
        {"barra_version": "barra_ancient"}, for_incremental=False,
    )
    # 无版本：resume 祖父兼容；增量必须拒绝（防错误复用）
    assert barra_pure_version_ok({}, for_incremental=False)
    assert barra_pure_version_ok(None, for_incremental=False)
    assert not barra_pure_version_ok({}, for_incremental=True)
    assert not barra_pure_version_ok(None, for_incremental=True)


def test_missing_barra_pure_names():
    series = {
        "A": _series([0.1, 0.2]),
        "B": _series([]),  # 空序列视为缺失
        "C": None,
    }
    miss = missing_barra_pure_names(["A", "B", "C", "D"], series)
    assert miss == ["B", "C", "D"]


def test_merge_barra_pure_results_keeps_old_adds_new():
    base_means = {"A": 0.02, "B": 0.01}
    base_series = {"A": _series([0.02]), "B": _series([0.01])}
    base_q = pd.DataFrame(
        {"long_share": [0.55, 0.45]},
        index=pd.Index(["A", "B"]),
    )
    new_means = {"C": 0.03, "B": 0.011}  # B 被新区覆盖
    new_series = {"C": _series([0.03]), "B": _series([0.011])}
    new_q = pd.DataFrame(
        {"long_share": [0.46, 0.60]},
        index=pd.Index(["B", "C"]),
    )
    means, series, qdf = merge_barra_pure_results(
        base_means, base_series, base_q,
        new_means, new_series, new_q,
    )
    assert set(means) == {"A", "B", "C"}
    assert means["B"] == pytest.approx(0.011)
    assert means["C"] == pytest.approx(0.03)
    assert set(series) == {"A", "B", "C"}
    assert set(qdf.index) == {"A", "B", "C"}
    assert qdf.loc["B", "long_share"] == pytest.approx(0.46)
    assert qdf.loc["A", "long_share"] == pytest.approx(0.55)
    assert qdf.loc["C", "long_share"] == pytest.approx(0.60)


def test_merge_empty_quantile_sides():
    means, series, qdf = merge_barra_pure_results(
        {"A": 0.1}, {"A": _series([0.1])}, pd.DataFrame(),
        {"B": 0.2}, {"B": _series([0.2])},
        pd.DataFrame({"long_share": [0.5]}, index=["B"]),
    )
    assert set(means) == {"A", "B"}
    assert list(qdf.index) == ["B"]

    means2, series2, qdf2 = merge_barra_pure_results(
        {"A": 0.1}, {"A": _series([0.1])},
        pd.DataFrame({"long_share": [0.4]}, index=["A"]),
        {"B": 0.2}, {"B": _series([0.2])}, pd.DataFrame(),
    )
    assert set(means2) == {"A", "B"}
    assert list(qdf2.index) == ["A"]


def test_incremental_scenario_existing_plus_new_only():
    """模拟：已有 A/B pure → 只算 C → merge 后库含 A/B/C。"""
    existing = pack_barra_pure_ckpt(
        {"A": 0.02, "B": 0.01},
        ["Size"],
        {"A": _series([0.02, 0.01]), "B": _series([0.01, 0.00])},
        pd.DataFrame({"long_share": [0.5, 0.4]}, index=["A", "B"]),
    )
    unpacked = unpack_barra_pure_ckpt(existing)
    assert unpacked is not None
    means, names, series, qdf, meta = unpacked
    assert barra_pure_version_ok(meta, for_incremental=True)

    wanted = ["A", "B", "C"]
    miss = missing_barra_pure_names(wanted, series)
    assert miss == ["C"]

    # 假装只对 C 跑了 run_barra_pure_ic
    new_means = {"C": 0.025}
    new_series = {"C": _series([0.02, 0.03])}
    new_q = pd.DataFrame({"long_share": [0.52]}, index=["C"])
    merged_m, merged_s, merged_q = merge_barra_pure_results(
        means, series, qdf, new_means, new_series, new_q,
    )
    assert missing_barra_pure_names(wanted, merged_s) == []
    assert set(merged_m) == {"A", "B", "C"}
    # 旧因子均值未被触碰
    assert merged_m["A"] == 0.02 and merged_m["B"] == 0.01

    packed = pack_barra_pure_ckpt(merged_m, names, merged_s, merged_q)
    again = unpack_barra_pure_ckpt(packed)
    assert again is not None
    assert set(again[2]) == {"A", "B", "C"}
