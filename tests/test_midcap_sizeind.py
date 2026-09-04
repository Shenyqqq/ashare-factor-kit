"""中盘 30–100 亿宇宙 / 池内 restan / membership residualize 单测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.wf.labels import (
    collapse_rare_industries,
    inpool_log_mcap_control,
    residualize_panel,
)
from research.ic.universe import restan_within_mask
from utils.universe import YI_TO_YUAN, build_mcap_yi_band_mask


DATES = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
CODES = [f"{i:06d}" for i in range(6)]


def test_yi_to_yuan_constant():
    assert YI_TO_YUAN == 1e8
    assert 30 * YI_TO_YUAN == 30e8
    assert 100 * YI_TO_YUAN == 100e8


def test_mcap_yi_band_inclusive_and_nan():
    """30 亿 / 100 亿含边界；缺市值剔除；无成交额过滤。"""
    circ = pd.DataFrame(
        [
            [29.9e8, 30e8, 50e8, 100e8, 100.1e8, np.nan],
            [40e8, 40e8, 40e8, 40e8, 40e8, 40e8],
        ],
        index=DATES,
        columns=CODES,
    )
    mask = build_mcap_yi_band_mask(circ, min_yi=30.0, max_yi=100.0)
    row0 = mask.loc[DATES[0]]
    assert bool(row0.iloc[0]) is False  # 29.9 亿
    assert bool(row0.iloc[1]) is True   # 30 亿
    assert bool(row0.iloc[2]) is True
    assert bool(row0.iloc[3]) is True   # 100 亿
    assert bool(row0.iloc[4]) is False  # 100.1 亿
    assert bool(row0.iloc[5]) is False  # NaN


def test_restan_within_mask_drops_outsiders_and_rezscores():
    idx = pd.DatetimeIndex(["2024-06-03"])
    cols = [f"{i:06d}" for i in range(8)]
    # 全市场 z 已做好：成员是 0..3，池外是极大值
    panel = pd.DataFrame(
        [[0.0, 1.0, 2.0, 3.0, 50.0, 50.0, 50.0, 50.0]],
        index=idx,
        columns=cols,
    )
    mem = pd.DataFrame(
        [[True, True, True, True, False, False, False, False]],
        index=idx,
        columns=cols,
    )
    out = restan_within_mask(panel, mem, winsor_pct=0.0)
    assert out.loc[idx[0], cols[4:]].isna().all()
    z = out.loc[idx[0], cols[:4]].to_numpy(dtype=float)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(z.std(ddof=0), 1.0, atol=1e-5)
    # 单调：原 0<1<2<3 仍保持
    assert np.all(np.diff(z) > 0)


def test_collapse_rare_industries():
    ind = pd.Series(["A"] * 12 + ["B"] * 9 + ["C"] * 2)
    out = collapse_rare_industries(ind, min_n=8)
    assert (out == "C").sum() == 0
    assert (out == "其他").sum() == 2
    assert (out == "A").sum() == 12


def test_inpool_log_mcap_not_full_market_z():
    members = pd.Index(["a", "b", "c", "d"])
    # 池内 40–80 亿；池外 5000 亿不应进入控制
    circ = pd.Series(
        {"a": 40e8, "b": 50e8, "c": 60e8, "d": 80e8, "out": 5000e8},
    )
    s = inpool_log_mcap_control(circ, members)
    assert "out" not in s.index
    assert np.isfinite(s).all()
    np.testing.assert_allclose(float(s.mean()), 0.0, atol=1e-6)


def test_residualize_membership_estimates_beta_in_pool_only():
    """全市场 WLS 再切片 ≠ 池内估 β。池外 Size 极大时 β 会被带偏。"""
    rng = np.random.default_rng(0)
    n_mem, n_out = 24, 24
    cols = [f"{i:06d}" for i in range(n_mem + n_out)]
    mem_cols = cols[:n_mem]
    out_cols = cols[n_mem:]
    dates = pd.DatetimeIndex(["2024-03-01"])
    size_m = rng.normal(size=n_mem)
    size_o = np.full(n_out, 8.0)  # 池外 Size 极大
    size = pd.DataFrame(
        [np.concatenate([size_m, size_o])], index=dates, columns=cols,
    )
    # 成员 y≈0.2*Size；池外 y≈3*Size → 全市场 β 被池外主导
    y_m = 0.2 * size_m + rng.normal(scale=0.05, size=n_mem)
    y_o = 3.0 * size_o + rng.normal(scale=0.05, size=n_out)
    fac = pd.DataFrame(
        [np.concatenate([y_m, y_o])], index=dates, columns=cols,
    )
    circ = pd.DataFrame(np.exp(size.to_numpy()), index=dates, columns=cols)
    mask = pd.DataFrame(False, index=dates, columns=cols)
    mask.loc[dates[0], mem_cols] = True
    ind = pd.Series(["G1"] * (n_mem // 2) + ["G2"] * (n_mem - n_mem // 2) + ["G1"] * n_out, index=cols)

    full = residualize_panel(
        fac, {"Barra_Size": size}, ind, dates, min_stocks=10,
    )
    pooled = residualize_panel(
        fac, {"Barra_Size": size}, ind, dates, min_stocks=10,
        membership_mask=mask, circ_mv=circ, min_industry_n=8,
    )
    assert pooled.loc[dates[0], out_cols].isna().all()
    assert int(pooled.loc[dates[0], mem_cols].notna().sum()) >= 10
    a = full.loc[dates[0], mem_cols]
    b = pooled.loc[dates[0], mem_cols]
    corr = float(a.corr(b))
    assert corr < 0.999, f"池内残差不应等于全市场再切片 corr={corr:.6f}"


def test_neutralize_one_factor_forwards_membership(monkeypatch):
    """训练路径 neutralize_one_factor 必须把 membership 传进 residualize_panel。"""
    from research.rolling_pool.neut_cache import neutralize_one_factor

    dates = pd.DatetimeIndex(["2024-01-02"])
    cols = [f"{i:06d}" for i in range(6)]
    panel = pd.DataFrame(np.arange(6, dtype=float).reshape(1, -1), index=dates, columns=cols)
    size = panel.copy()
    mask = pd.DataFrame([[True, True, True, True, False, False]], index=dates, columns=cols)
    seen = {}

    def fake_resid(fac, barra, ind, dates_use, **kwargs):
        seen["membership"] = kwargs.get("membership_mask")
        seen["circ_mv"] = kwargs.get("circ_mv")
        seen["min_industry_n"] = kwargs.get("min_industry_n")
        return fac

    monkeypatch.setattr("models.wf.labels.residualize_panel", fake_resid)
    out = neutralize_one_factor(
        panel, "动量_20d",
        barra_factors={"Barra_Size": size},
        industry_map=pd.Series(["A"] * 6, index=cols),
        dates_use=dates,
        weight_panel=None,
        zscore_fn=lambda x: x,
        membership_mask=mask,
        circ_mv=size,
        min_industry_n=10,
        restan_in_universe=False,
    )
    assert seen["membership"] is mask
    assert seen["min_industry_n"] == 10
    assert out.shape == panel.shape


def test_midcap_ckpt_paths_isolated_from_full_market():
    """亿元带 checkpoint 进 mcap 子目录，且文件名含 mcap + nc_size_industry + 日历。"""
    from research.ic import cli as ic_cli

    ic_cli._set_ckpt_tag("")
    ic_cli._set_ckpt_neut_tag("")
    ic_cli._set_ckpt_dir(None)
    full_barra = ic_cli._ckpt_path(5, "barra_pure")
    full_ic = ic_cli._ckpt_path(5, "ic_series")
    assert full_barra == ic_cli._CKPT_DIR_DEFAULT / "barra_pure_h5.pkl"
    assert full_ic == ic_cli._CKPT_DIR_DEFAULT / "ic_series_h5.pkl"

    tag = ic_cli._universe_tag(
        "all", 0.3, None, "all",
        mcap_min_yi=30.0, mcap_max_yi=100.0, calendar_fp="abc123",
    )
    assert tag == "mcap30_100_abc123"
    ic_cli._set_ckpt_tag(f"{tag}_tmr_v2")
    ic_cli._set_ckpt_neut_tag("nc_size_industry")
    ic_cli._set_ckpt_dir(ic_cli._CKPT_DIR_DEFAULT / "mcap30_100")
    mid_barra = ic_cli._ckpt_path(5, "barra_pure")
    mid_ic = ic_cli._ckpt_path(5, "ic_series")
    assert mid_barra.parent.name == "mcap30_100"
    assert "mcap30_100" in mid_barra.name
    assert "nc_size_industry" in mid_barra.name
    assert "abc123" in mid_barra.name
    assert mid_barra != full_barra
    assert mid_ic != full_ic
    assert not str(mid_barra).endswith("barra_pure_h5.pkl")
    ic_cli._set_ckpt_tag("")
    ic_cli._set_ckpt_neut_tag("")
    ic_cli._set_ckpt_dir(None)


def test_midcap_yaml_path_does_not_clobber_flagship():
    from research.midcap_ic import (
        _PROTECTED_YAML_NAMES,
        _default_save_suffix,
        write_midcap_yaml,
    )

    suf = _default_save_suffix(mcap_min_yi=30.0, mcap_max_yi=100.0)
    assert suf.startswith("midcap30_100_sizeind_")
    yaml_name = f"factor_configs_h5_{suf}.yaml"
    assert yaml_name not in _PROTECTED_YAML_NAMES
    assert yaml_name != "factor_configs_h5_nolongshare_20260804.yaml"
    assert yaml_name != "factor_configs_h5_sizeind_20260815.yaml"

    import pytest
    from pathlib import Path

    with pytest.raises(SystemExit, match="拒绝覆盖"):
        write_midcap_yaml(
            Path("research/output/selected_factors_h5.json"),
            Path("config/factor_configs_h5_sizeind_20260815.yaml"),
            period=5,
        )
