"""PIT industry panel smoke test (synthetic, no network)."""
import pandas as pd
import numpy as np

from research.ic.barra import _industry_dummies, precompute_ctrl_matrices
from data.industry.download_industry import (
    build_industry_panel,
    load_industry_as_of,
    _snapshot_from_panel,
)


def test_as_of():
    raw = pd.DataFrame([
        {"symbol": "A", "start_date": "2015-01-01", "industry_code": "440101"},
        {"symbol": "A", "start_date": "2020-01-01", "industry_code": "440201"},
        {"symbol": "B", "start_date": "2015-01-01", "industry_code": "640101"},
    ])
    panel = build_industry_panel(raw)
    assert panel.loc[panel["code"] == "A", "end_date"].iloc[0] == pd.Timestamp("2019-12-31")
    assert pd.isna(panel.loc[panel["code"] == "A", "end_date"].iloc[1])
    asof_2019 = load_industry_as_of(panel, "2019-06-01").to_dict()
    asof_2021 = load_industry_as_of(panel, "2021-06-01").to_dict()
    assert asof_2019 == {"A": "4401", "B": "6401"}, asof_2019
    assert asof_2021 == {"A": "4402", "B": "6401"}, asof_2021
    snap = _snapshot_from_panel(panel)
    assert snap.loc["A", "sw_l2"] == "4402"
    assert snap.loc["B", "sw_l2"] == "6401"
    print("[OK] load_industry_as_of / build_industry_panel / snapshot")


def test_dummies_pit_vs_static():
    panel = pd.DataFrame([
        {"code": "A", "effective_date": "2015-01-01", "sw_l1": "44", "sw_l2": "4401", "end_date": "2019-12-31"},
        {"code": "A", "effective_date": "2020-01-01", "sw_l1": "44", "sw_l2": "4402", "end_date": pd.NaT},
        {"code": "B", "effective_date": "2015-01-01", "sw_l1": "64", "sw_l2": "6401", "end_date": pd.NaT},
        {"code": "C", "effective_date": "2015-01-01", "sw_l1": "44", "sw_l2": "4401", "end_date": pd.NaT},
    ])
    idx = pd.Index(["A", "B", "C"])
    d_2019 = _industry_dummies(None, idx, industry_panel=panel, date=pd.Timestamp("2019-06-01"))
    d_2021 = _industry_dummies(None, idx, industry_panel=panel, date=pd.Timestamp("2021-06-01"))
    # 2019: A,C=4401 (ref), B=6401 -> only _ind_6401 dummy
    assert set(d_2019.keys()) == {"_ind_6401"}, d_2019.keys()
    # 2021: A=4402, C=4401 (ref), B=6401 -> _ind_4402 and _ind_6401
    assert set(d_2021.keys()) == {"_ind_4402", "_ind_6401"}, d_2021.keys()
    assert d_2021["_ind_4402"]["A"] == 1.0
    assert d_2021["_ind_4402"]["C"] == 0.0
    # static fallback
    static = pd.Series({"A": "4402", "B": "6401", "C": "4401"})
    d_static = _industry_dummies(static, idx)
    assert set(d_static.keys()) == {"_ind_4402", "_ind_6401"}, d_static.keys()
    print("[OK] _industry_dummies PIT + static fallback")


def test_precompute_pit_and_static():
    panel = pd.DataFrame([
        {"code": "A", "effective_date": "2015-01-01", "sw_l1": "44", "sw_l2": "4401", "end_date": "2019-12-31"},
        {"code": "A", "effective_date": "2020-01-01", "sw_l1": "44", "sw_l2": "4402", "end_date": pd.NaT},
        {"code": "B", "effective_date": "2015-01-01", "sw_l1": "64", "sw_l2": "6401", "end_date": pd.NaT},
        {"code": "C", "effective_date": "2015-01-01", "sw_l1": "44", "sw_l2": "4401", "end_date": pd.NaT},
    ])
    dates = pd.DatetimeIndex(["2019-06-03", "2021-06-03"])
    fwd = pd.DataFrame(
        {"A": [0.01, 0.02], "B": [0.0, 0.03], "C": [0.05, 0.04]},
        index=dates,
    )
    barra = {"size": pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
        index=dates,
    )}

    # PIT path
    dc = precompute_ctrl_matrices(
        barra, fwd, industry_map=None, dates=dates, industry_panel=panel,
    )
    assert len(dc) == 2
    arr_2019, idx_2019, _ = dc[pd.Timestamp("2019-06-03")]
    arr_2021, idx_2021, _ = dc[pd.Timestamp("2021-06-03")]
    # 2019: 1 barra col + 1 ind dummy (_ind_6401) = 2 cols
    assert arr_2019.shape == (3, 2), arr_2019.shape
    # 2021: 1 barra col + 2 ind dummies (_ind_4402, _ind_6401) = 3 cols
    assert arr_2021.shape == (3, 3), arr_2021.shape
    print(f"[OK] precompute PIT: 2019 shape={arr_2019.shape}, 2021 shape={arr_2021.shape}")

    # Static path
    static = pd.Series({"A": "4402", "B": "6401", "C": "4401"})
    dc2 = precompute_ctrl_matrices(
        barra, fwd, industry_map=static, dates=dates,
    )
    arr_s, _, _ = dc2[pd.Timestamp("2019-06-03")]
    # Static: 1 barra col + 2 ind dummies (4401 ref dropped, 4402 & 6401 present) = 3 cols
    assert arr_s.shape == (3, 3), arr_s.shape
    print(f"[OK] precompute static: shape={arr_s.shape}")


def test_no_industry():
    """No industry_map and no panel -> ctrl_arr = barra only."""
    dates = pd.DatetimeIndex(["2021-06-03"])
    fwd = pd.DataFrame({"A": [0.01], "B": [0.0]}, index=dates)
    barra = {"size": pd.DataFrame({"A": [1.0], "B": [3.0]}, index=dates)}
    dc = precompute_ctrl_matrices(barra, fwd, industry_map=None, dates=dates)
    arr, _, _ = dc[pd.Timestamp("2021-06-03")]
    assert arr.shape == (2, 1), arr.shape
    print(f"[OK] precompute no-industry: shape={arr.shape}")


if __name__ == "__main__":
    test_as_of()
    test_dummies_pit_vs_static()
    test_precompute_pit_and_static()
    test_no_industry()
    print("\nALL PIT SMOKE TESTS PASSED")
