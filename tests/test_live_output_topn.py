"""live.daily_update.output_topn 候选列：申万二级 + 流通市值。"""
from pathlib import Path

import pandas as pd
import numpy as np

from live.daily_update import output_topn, CANDIDATE_COLS


def test_output_topn_adds_industry_and_circ_mv(tmp_path: Path):
    as_of = pd.Timestamp("2026-08-13")
    codes = ["000001", "600519", "000002"]
    scores = pd.Series([1.2, 3.4, 0.5], index=codes)
    mask = pd.Series([True, True, True], index=codes)
    names = pd.Series({"000001": "平安银行", "600519": "贵州茅台", "000002": "万科A"})
    sw = pd.Series({"000001": "4803", "600519": "3405", "000002": "4301"})
    circ = pd.DataFrame(
        {
            "000001": [2.5e11],
            "600519": [1.6e12],
            "000002": [1.8e11],
        },
        index=pd.DatetimeIndex([as_of]),
    )
    out = tmp_path / "candidates.csv"
    top = output_topn(
        scores, mask, top_n=2, cap_band="all", as_of=as_of, output_path=out,
        stock_names=names, circ_mv=circ, sw_l2=sw,
    )
    assert list(top["code"]) == ["600519", "000001"]
    assert list(top["rank"]) == [1, 2]
    assert top.loc[0, "sw_l2"] == "白酒Ⅱ"
    assert top.loc[1, "sw_l2"] == "股份制银行Ⅱ"
    assert abs(top.loc[0, "circ_mv_yi"] - 16000.0) < 1e-6
    assert abs(top.loc[0, "circ_mv"] / 1e8 - top.loc[0, "circ_mv_yi"]) < 1e-9
    for col in CANDIDATE_COLS:
        assert col in top.columns
    csv = pd.read_csv(out, encoding="utf-8-sig", dtype={"code": str})
    assert "sw_l2" in csv.columns and "circ_mv_yi" in csv.columns
    md = (tmp_path / "candidates.md").read_text(encoding="utf-8")
    assert "白酒Ⅱ" in md
    assert "circ_mv_yi" in md
    assert "亿元" in md
