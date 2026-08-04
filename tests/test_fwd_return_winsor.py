"""
tests/test_fwd_return_winsor.py

forward_return 截面 1%/99% winsorize：极端收益被夹到分位边界，中间值不变。
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research.ic.forward_return import winsorize_forward_return


def test_winsorize_clips_extremes_keeps_middle():
    """单截面：1–2 个极端收益被夹到 1%/99% 边界，中间值不变。"""
    # 100 只股票：线性收益 -0.5 … +0.49，再塞入两个极端值
    rng = np.linspace(-0.5, 0.49, 100)
    rng[0] = -10.0   # 远低于 1% 分位
    rng[-1] = 10.0   # 远高于 99% 分位
    codes = [f"{i:06d}" for i in range(100)]
    fwd = pd.DataFrame(
        [rng, rng * 0.5],  # 两行截面，第二行同样有极端（缩放后）
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=codes,
    )

    out = winsorize_forward_return(fwd, lower=0.01, upper=0.99)

    lo = pd.Series(rng).quantile(0.01)
    hi = pd.Series(rng).quantile(0.99)
    # 极端值被夹到分位边界
    assert out.iloc[0, 0] == pytest.approx(lo)
    assert out.iloc[0, -1] == pytest.approx(hi)
    # 中间值不变
    assert out.iloc[0, 50] == pytest.approx(rng[50])
    # 截尾后无超出边界的值
    assert out.iloc[0].min() >= lo - 1e-12
    assert out.iloc[0].max() <= hi + 1e-12
    # 保留样本：无新增 NaN
    assert out.notna().sum().sum() == fwd.notna().sum().sum()
    # 保持 index / columns
    assert list(out.index) == list(fwd.index)
    assert list(out.columns) == list(fwd.columns)


def test_winsorize_skips_all_nan_row():
    """全 NaN 行安全跳过，不报错且仍为全 NaN。"""
    fwd = pd.DataFrame(
        {
            "a": [np.nan, 0.1, 0.2],
            "b": [np.nan, 0.3, 10.0],
            "c": [np.nan, -5.0, 0.4],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    out = winsorize_forward_return(fwd, lower=0.01, upper=0.99)
    assert out.iloc[0].isna().all()
    assert out.iloc[1].notna().all()


def test_winsorize_empty_frame():
    out = winsorize_forward_return(pd.DataFrame(), lower=0.01, upper=0.99)
    assert out.empty


def test_ml_and_ic_share_winsorize_helper():
    """IC cli 与 ML build_factor_dataset 均调用同一 winsorize_forward_return。"""
    import strategies.ml as ml
    from research.ic import cli as ic_cli

    ml_src = inspect.getsource(ml.build_factor_dataset)
    assert "winsorize_forward_return" in ml_src

    cli_src = inspect.getsource(ic_cli.run)
    assert "winsorize_forward_return" in cli_src
