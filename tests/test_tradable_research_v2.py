"""Research v2 tradable/limit semantics + 涨跌停状态 factor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    FWD_RETURN_EXEC_MASK,
    TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY,
    apply_label_exec_mask_for_mode,
    normalize_tradable_limit_mode,
    tradable_ckpt_tag,
)
from factors.factor_limit import factor_limit_state
from research.ic.forward_return import build_forward_return
from research.ic.universe import build_ic_tradability_mask


DATES = pd.date_range("2024-01-02", periods=10, freq="B")
CODES = ["000001", "000002"]


def _prices() -> pd.DataFrame:
    base = np.arange(1, len(DATES) + 1, dtype=float)
    return pd.DataFrame(
        {c: base * (1.0 + 0.01 * i) for i, c in enumerate(CODES)},
        index=DATES,
    )


def _limit_masks(limit_up_idx: int, col: int = 0) -> dict:
    limit_up = pd.DataFrame(False, index=DATES, columns=CODES)
    limit_down = pd.DataFrame(False, index=DATES, columns=CODES)
    limit_up.iloc[limit_up_idx, col] = True
    any_limit = limit_up | limit_down
    limit_up_open = pd.DataFrame(False, index=DATES, columns=CODES)
    return {
        "limit_up": limit_up,
        "limit_down": limit_down,
        "any_limit": any_limit,
        "limit_up_open": limit_up_open,
    }


def test_default_tradable_limit_mode_is_research():
    assert not TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY
    assert not FWD_RETURN_EXEC_MASK
    assert normalize_tradable_limit_mode(None) == "research"
    assert not apply_label_exec_mask_for_mode(None)
    assert apply_label_exec_mask_for_mode("strict")


def test_tradable_ckpt_tag_bumps_by_mode():
    assert tradable_ckpt_tag() == "tmr_v2"
    assert tradable_ckpt_tag(tradable_limit_mode="strict") == "tmr_strict"


def test_research_tradable_keeps_limit_up_signal_day():
    prices = _prices()
    volume = pd.DataFrame(1000.0, index=DATES, columns=CODES)
    masks = _limit_masks(limit_up_idx=3, col=0)

    tradable = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=masks,
        exclude_limit_on_signal=False,
    )
    assert tradable.iloc[3, 0], "research 模式信号日涨停股应仍为可交易"

    tradable_strict = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=masks,
        exclude_limit_on_signal=True,
    )
    assert not tradable_strict.iloc[3, 0], "strict 模式信号日涨停股应剔除"


def test_research_forward_return_skips_exec_mask():
    prices = _prices()
    open_ = prices.copy() * 0.99
    period = 3
    limit_up_open = pd.DataFrame(False, index=DATES, columns=CODES)
    limit_up_open.iloc[3, 0] = True
    any_limit = pd.DataFrame(False, index=DATES, columns=CODES)
    any_limit.iloc[5, 1] = True
    masks = {"limit_up_open": limit_up_open, "any_limit": any_limit}

    fwd_research = build_forward_return(
        prices, open_, period, masks=masks, apply_exec_mask=False,
    )
    fwd_strict = build_forward_return(
        prices, open_, period, masks=masks, apply_exec_mask=True,
    )
    assert pd.notna(fwd_research.iloc[2, 0]), "research 标签不应屏蔽买日一字涨停"
    assert pd.isna(fwd_strict.iloc[2, 0]), "strict 标签仍应屏蔽买日一字涨停"
    assert pd.notna(fwd_research.iloc[2, 1]), "research 标签不应屏蔽卖日涨跌停"
    assert pd.isna(fwd_strict.iloc[2, 1]), "strict 标签仍应屏蔽卖日涨跌停"


def test_factor_limit_state_encoding():
    masks = _limit_masks(limit_up_idx=2, col=0)
    masks["limit_down"].iloc[4, 1] = True
    masks["limit_up"].iloc[5, 0] = True
    masks["limit_down"].iloc[5, 0] = True
    masks["any_limit"] = masks["limit_up"] | masks["limit_down"]

    # 编码逻辑（normalize 前）
    state = pd.DataFrame(2.0, index=DATES, columns=CODES)
    state = state.mask(masks["limit_down"], 1.0)
    state = state.mask(masks["limit_up"], 3.0)
    assert state.iloc[2, 0] == pytest.approx(3.0)
    assert state.iloc[4, 1] == pytest.approx(1.0)
    assert state.iloc[0, 0] == pytest.approx(2.0)
    assert state.iloc[5, 0] == pytest.approx(3.0), "同日涨跌停应编码为涨停(3)"

    panel = factor_limit_state(masks)
    assert panel.shape == state.shape


def test_mask_scores_for_backtest_strict_drops_signal_limit():
    from research.ic.universe import mask_scores_for_backtest

    prices = _prices()
    open_ = prices.copy() * 0.99
    volume = pd.DataFrame(1000.0, index=DATES, columns=CODES)
    masks = _limit_masks(limit_up_idx=3, col=0)
    # buy-day一字涨停 on day 3 → signal day 2 label exec-masked
    masks["limit_up_open"] = pd.DataFrame(False, index=DATES, columns=CODES)
    masks["limit_up_open"].iloc[3, 0] = True
    scores = pd.DataFrame(1.0, index=DATES, columns=CODES)

    kept_train = mask_scores_for_backtest(
        scores, prices, volume=volume, masks=masks, score_universe="train",
    )
    assert kept_train.notna().all().all()

    masked = mask_scores_for_backtest(
        scores, prices, open_=open_, hold_period=3,
        volume=volume, masks=masks, score_universe="strict",
    )
    assert pd.isna(masked.iloc[3, 0]), "strict 回测宇宙应去掉信号日涨停得分"
    assert pd.isna(masked.iloc[2, 0]), "strict 回测宇宙应去掉买日一字涨停标签门控"
    assert pd.notna(masked.iloc[3, 1]), "未触及涨跌停的股票应保留"
    # day0 → sell day3 has any_limit on code0, so exec-mask also drops day0/code0
    assert pd.isna(masked.iloc[0, 0])
    assert pd.notna(masked.iloc[0, 1])


def test_normalize_tradable_limit_mode_rejects_invalid():
    with pytest.raises(ValueError, match="strict\\|research"):
        normalize_tradable_limit_mode("legacy")
