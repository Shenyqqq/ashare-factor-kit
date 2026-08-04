"""
tests/test_universe.py  —  utils/universe.py 单测

用合成数据验证：
  1. 市值区间边界（含/不含上下界）
  2. 流动性过滤（20日均成交额）
  3. NaN/空值处理（NaN 市值不算 eligible）
  4. 时变性（不同日期 universe 不同）
  5. fallback 到 total_mv 时正常工作
  6. apply_small_cap_mask 正确剔除
"""
import numpy as np
import pandas as pd
import pytest

from utils.universe import (
    build_small_cap_universe,
    apply_small_cap_mask,
    small_cap_universe_size,
)


# ── 测试夹具 ──────────────────────────────────────────────────────────────────

DATES = pd.date_range("2024-01-01", periods=30, freq="B")
CODES = ["000001", "000002", "000003", "000004", "000005"]


def _make_mv(values: list[list[float]]) -> pd.DataFrame:
    """values: 30×5 的二维列表 → DataFrame。"""
    return pd.DataFrame(values, index=DATES, columns=CODES, dtype=float)


def _make_amount_constant(level: float) -> pd.DataFrame:
    """每只股票每天成交额都是 level。"""
    return pd.DataFrame(
        np.full((len(DATES), len(CODES)), level, dtype=float),
        index=DATES, columns=CODES,
    )


# 流动性测试用的常量成交额：3000万，高于默认 min_amount=2000万
AMOUNT_LIQUID = 3000e4  # 3000万元


# ── 1. 边界测试 ───────────────────────────────────────────────────────────────

def test_boundary_inclusive():
    """上下界应包含等号（[lower, upper]）。"""
    # 全部股票每天成交额 3000万（高于 min_amount 2000万）
    amount = _make_amount_constant(AMOUNT_LIQUID)
    # 5 只股票流通市值分别为：低于下界、等于下界、中位、等于上界、高于上界
    circ_mv = _make_mv([
        [5e8, 8e8, 80e8, 150e8, 200e8] for _ in DATES
    ])
    mask = build_small_cap_universe(
        circ_mv=circ_mv, amount=amount,
        upper=150e8, lower=8e8, min_amount=2000e4,
    )
    # 前 9 天 rolling min_periods=10 → NaN → False；第 10 天起全 True
    expected_vals = [False] * 9 + [True] * (len(DATES) - 9)
    expected = pd.DataFrame(
        [[False, t, t, t, False] for t in expected_vals],
        index=DATES, columns=CODES,
    )
    pd.testing.assert_frame_equal(mask, expected)


def test_liquidity_filter():
    """20 日均成交额 < min_amount 的股票应被剔除。"""
    # 全部股票市值都在区间内
    circ_mv = _make_mv([[80e8] * 5 for _ in DATES])
    # 5 只股票成交额分别为：0、刚好 2000万、3000万、1亿、NaN
    amount_vals = [0.0, 2000e4, 3000e4, 1e8, np.nan]
    amount = pd.DataFrame(
        np.tile(amount_vals, (len(DATES), 1)),
        index=DATES, columns=CODES, dtype=float,
    )
    mask = build_small_cap_universe(
        circ_mv=circ_mv, amount=amount,
        upper=150e8, lower=8e8, min_amount=2000e4,
        amount_window=20,
    )
    # 第 10 天起 rolling 满足 min_periods：
    # 000001 False (0), 000002 True (=2000万), 000003 True, 000004 True, 000005 False (NaN)
    last = mask.iloc[-1]
    assert last["000001"] == False
    assert last["000002"] == True   # 等于阈值
    assert last["000003"] == True
    assert last["000004"] == True
    assert last["000005"] == False  # NaN 不算 eligible


# ── 2. NaN 处理 ───────────────────────────────────────────────────────────────

def test_nan_market_cap_excluded():
    """市值 NaN 的格子不应算 eligible（即使成交额满足）。"""
    amount = _make_amount_constant(AMOUNT_LIQUID)
    circ_mv = _make_mv([[80e8] * 5 for _ in DATES])
    circ_mv.iloc[10:, 2] = np.nan  # 000003 后半段市值缺失
    mask = build_small_cap_universe(
        circ_mv=circ_mv, amount=amount,
        upper=150e8, lower=8e8, min_amount=2000e4,
    )
    # 000003 前 10 天 (rows 0-9) 市值在区间 → 第 9 天 rolling 满足 min_periods=10 → True
    # 第 10 天起市值 NaN → False
    assert mask.iloc[9, 2] == True
    assert mask.iloc[11, 2] == False
    # 其他股票从第 9 天起 True
    assert mask.iloc[15, 0] == True
    assert mask.iloc[15, 1] == True


def test_empty_inputs():
    """空 DataFrame 应返回空 mask，不抛错。"""
    empty = pd.DataFrame()
    mask = build_small_cap_universe(
        circ_mv=empty, amount=None,
        upper=150e8, lower=8e8, min_amount=2000e4,
    )
    assert mask.empty


# ── 3. 时变性 ─────────────────────────────────────────────────────────────────

def test_time_varying():
    """同一只股票在不同日期可能进/出 universe。"""
    amount = _make_amount_constant(AMOUNT_LIQUID)
    # 000001 前 15 天市值 50亿（小盘），后 15 天 200亿（大盘）
    circ_mv_vals = []
    for i, d in enumerate(DATES):
        circ_mv_vals.append([
            50e8 if i < 15 else 200e8,
            80e8, 80e8, 80e8, 80e8,
        ])
    circ_mv = pd.DataFrame(circ_mv_vals, index=DATES, columns=CODES, dtype=float)
    mask = build_small_cap_universe(
        circ_mv=circ_mv, amount=amount,
        upper=150e8, lower=8e8, min_amount=2000e4,
    )
    # 000001 前 15 天 True，后 15 天 False（rolling 满足后）
    assert mask.iloc[10, 0] == True
    assert mask.iloc[20, 0] == False
    # 时变计数：前 15 天 5 只（rolling 满足后），后 15 天 4 只
    sizes = small_cap_universe_size(mask)
    assert sizes.iloc[10] == 5
    assert sizes.iloc[20] == 4


# ── 4. fallback 到 total_mv ───────────────────────────────────────────────────

def test_fallback_to_total_mv():
    """circ_mv=None 时用 total_mv 近似。"""
    amount = _make_amount_constant(AMOUNT_LIQUID)
    total_mv = _make_mv([[80e8] * 5 for _ in DATES])
    # 不传 circ_mv，应自动用 total_mv
    mask = build_small_cap_universe(
        circ_mv=None, amount=amount, total_mv=total_mv,
        upper=150e8, lower=8e8, min_amount=2000e4,
    )
    # 80亿在 [8, 150] 区间，amount=3000万>2000万 → 第 10 天起全 True
    assert mask.iloc[-1, 0] == True


def test_no_mv_raises():
    """circ_mv 和 total_mv 都 None 时应抛 ValueError。"""
    with pytest.raises(ValueError, match="至少需提供"):
        build_small_cap_universe(
            circ_mv=None, amount=None, total_mv=None,
        )


# ── 5. apply_small_cap_mask ───────────────────────────────────────────────────

def test_apply_mask():
    """apply_small_cap_mask 应将被剔除的格子置 NaN。"""
    # 构造 universe mask：只保留 000001、000002
    mask = pd.DataFrame(
        [[True, True, False, False, False] for _ in DATES],
        index=DATES, columns=CODES,
    )
    # 构造得分面板：所有股票都有值（30×5）
    panel = pd.DataFrame(
        np.arange(150, dtype=float).reshape(30, 5),
        index=DATES, columns=CODES,
    )
    out = apply_small_cap_mask(panel, mask)
    # 000001/000002 保留原值，000003/4/5 全 NaN
    assert out.iloc[10, 0] == panel.iloc[10, 0]
    assert out.iloc[10, 1] == panel.iloc[10, 1]
    assert np.isnan(out.iloc[10, 2])
    assert np.isnan(out.iloc[10, 3])
    assert np.isnan(out.iloc[10, 4])


def test_cap_band_micro_30_no_shell_floor():
    """micro_30 / micro_lt30: lower=0.0 无 8 亿地板；None lower 才会回退 8 亿。"""
    from utils.universe import build_cap_band_mask
    from config.settings import CAP_BANDS, SHELL_CAP_LOWER

    assert CAP_BANDS["micro_30"] == (0.0, 30e8)
    assert CAP_BANDS["micro_lt30"] == (0.0, 30e8)
    # 显式 0 ≠ None 回退
    assert CAP_BANDS["micro_30"][0] is not None
    assert CAP_BANDS["micro_30"][0] != SHELL_CAP_LOWER

    circ = _make_mv(
        [[5e8, 20e8, 40e8, 8e8, 25e8] for _ in DATES]
    )
    amount = _make_amount_constant(AMOUNT_LIQUID)
    mask = build_cap_band_mask("micro_30", circ_mv=circ, amount=amount)
    assert mask is not None
    # 5亿（<8亿壳股地板）仍应入池；40亿超上限剔除
    assert bool(mask.iloc[-1, 0]) is True   # 5e8
    assert bool(mask.iloc[-1, 1]) is True   # 20e8
    assert bool(mask.iloc[-1, 2]) is False  # 40e8
    assert bool(mask.iloc[-1, 3]) is True   # 8e8
    assert bool(mask.iloc[-1, 4]) is True   # 25e8
    # alias 同口径
    mask_alias = build_cap_band_mask("micro_lt30", circ_mv=circ, amount=amount)
    pd.testing.assert_frame_equal(mask, mask_alias)


def test_apply_mask_none_passthrough():
    """mask=None 时 apply 应原样返回 panel。"""
    panel = pd.DataFrame(
        np.arange(10, dtype=float).reshape(2, 5),
        index=DATES[:2], columns=CODES,
    )
    out = apply_small_cap_mask(panel, None)
    pd.testing.assert_frame_equal(out, panel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
