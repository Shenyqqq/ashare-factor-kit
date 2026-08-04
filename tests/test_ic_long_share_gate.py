"""稠密门：long_share 与 |IC|∧|ICIR| 合取（corr-dedup 之前）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.selection import (
    _long_share_gate_reason,
    overlay_long_share,
    select_factors_raw,
)


def _row(ic=0.03, icir=0.5, t=3.0, nw=3.0, long_share=0.5):
    return {
        "IC均值": ic,
        "ICIR": icir,
        "t统计量": t,
        "NW_t统计量": nw,
        "胜率": 0.55,
        "正IC占比": 0.55,
        "负IC占比": 0.45,
        "样本数": 100,
        "同向年份占比": np.nan,
        "IC滚动ICIR": np.nan,
        "最差12期IC均值": np.nan,
        "long_share": long_share,
    }


def _summary(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def test_long_share_gate_reason_pass_fail_disable():
    assert _long_share_gate_reason(pd.Series({"long_share": 0.41}), 0.4) is None
    assert _long_share_gate_reason(pd.Series({"long_share": 0.4}), 0.4) is not None
    assert _long_share_gate_reason(pd.Series({"long_share": 0.3}), 0.4) is not None
    assert _long_share_gate_reason(pd.Series({"long_share": np.nan}), 0.4) is not None
    assert _long_share_gate_reason(pd.Series({"long_share": 0.1}), 0) is None
    assert _long_share_gate_reason(pd.Series({"long_share": 0.1}), None) is None


def test_select_raw_conjunction_ic_then_long_share():
    """过 IC∧ICIR∧t 后因 long_share 被剔；关闭门则保留。"""
    df = _summary({
        "A_ok": _row(long_share=0.55),
        "B_low_ls": _row(long_share=0.30),
        "C_weak_ic": _row(ic=0.005, icir=0.1, long_share=0.80),
    })
    # 无 FDR；NW_t 门用阈值 2.5
    kept_on, ex_on = select_factors_raw(
        df, all_ic={},
        ic_threshold=0.015, icir_threshold=0.30,
        t_threshold=2.5, nw_t_threshold=2.5,
        use_fdr=False, corr_dedup=False,
        min_long_share=0.4,
    )
    assert kept_on == ["A_ok"]
    assert "B_low_ls" in ex_on and "long_share" in ex_on["B_low_ls"]
    assert "C_weak_ic" in ex_on and "|IC|∧|ICIR|" in ex_on["C_weak_ic"]

    kept_off, _ = select_factors_raw(
        df, all_ic={},
        ic_threshold=0.015, icir_threshold=0.30,
        t_threshold=2.5, nw_t_threshold=2.5,
        use_fdr=False, corr_dedup=False,
        min_long_share=0.0,
    )
    assert set(kept_off) == {"A_ok", "B_low_ls"}


def test_select_raw_missing_long_share_fails_when_gate_on():
    row = _row()
    del row["long_share"]
    df = _summary({"X": row})
    kept, ex = select_factors_raw(
        df, all_ic={},
        ic_threshold=0.015, icir_threshold=0.30,
        t_threshold=2.5, nw_t_threshold=2.5,
        use_fdr=False, corr_dedup=False,
        min_long_share=0.4,
    )
    assert kept == []
    assert "long_share" in ex["X"] and "缺失" in ex["X"]


def test_overlay_long_share_csv(tmp_path):
    summary = _summary({
        "f1": _row(long_share=0.2),
        "f2": _row(long_share=0.2),
    })
    csv_path = tmp_path / "aligned.csv"
    pd.DataFrame({
        "因子": ["f1", "f2"],
        "long_share": [0.55, 0.25],
        "多空来源": ["双边", "空头主导"],
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")
    out = overlay_long_share(summary, long_share_csv=str(csv_path))
    assert abs(float(out.loc["f1", "long_share"]) - 0.55) < 1e-9
    assert abs(float(out.loc["f2", "long_share"]) - 0.25) < 1e-9

    kept, _ = select_factors_raw(
        out, all_ic={},
        ic_threshold=0.015, icir_threshold=0.30,
        t_threshold=2.5, nw_t_threshold=2.5,
        use_fdr=False, corr_dedup=False,
        min_long_share=0.4,
    )
    assert kept == ["f1"]
