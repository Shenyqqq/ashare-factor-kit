"""
tests/test_position_regime.py — 仓位体制信号 / 敞口映射 / ML 不再注入市场广播
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.regime import (
    PositionRegimeConfig,
    apply_exposure,
    compute_position_regime,
    exposure_at,
)
from backtest.quantile import run_quantile_backtest


def _synth_market(n_days: int = 300, n_stocks: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    # 温和上涨的市场
    mkt = 1000.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
    market = pd.DataFrame({"close": mkt}, index=dates)
    stock_rets = rng.normal(0.0003, 0.02, (n_days, n_stocks))
    prices = pd.DataFrame(
        10.0 * np.cumprod(1 + stock_rets, axis=0),
        index=dates, columns=codes,
    )
    clean_ret = pd.DataFrame(stock_rets, index=dates, columns=codes)
    circ_mv = pd.DataFrame(
        rng.uniform(1e9, 1e11, (n_days, n_stocks)),
        index=dates, columns=codes,
    )
    return market, prices, clean_ret, circ_mv


def test_apply_exposure_cash_zero():
    assert apply_exposure(0.10, 0.5) == pytest.approx(0.05)
    assert apply_exposure(-0.08, 0.25) == pytest.approx(-0.02)
    assert np.isnan(apply_exposure(np.nan, 0.5))


def test_force_exposure_override():
    market, prices, clean_ret, circ_mv = _synth_market()
    cfg = PositionRegimeConfig(force_exposure=0.5, e_min=0.3)
    df = compute_position_regime(
        market, prices=prices, clean_ret=clean_ret, circ_mv=circ_mv, config=cfg,
    )
    assert (df["target_exposure"].dropna() == 0.5).all()


def test_exposure_in_range_and_shifted():
    market, prices, clean_ret, circ_mv = _synth_market()
    cfg = PositionRegimeConfig(e_min=0.3)
    df = compute_position_regime(
        market, prices=prices, clean_ret=clean_ret, circ_mv=circ_mv, config=cfg,
    )
    exp = df["target_exposure"].dropna()
    assert exp.min() >= 0.3 - 1e-9
    assert exp.max() <= 1.0 + 1e-9
    # PIT：首日应为 NaN（shift 后）
    assert pd.isna(df["mkt_trend"].iloc[0])


def test_score_mapping_extremes():
    """3 个 risk-on → exposure=1；0 个 → e_min。"""
    cfg = PositionRegimeConfig(e_min=0.3)
    # 手工构造已 shift 的布尔计分路径：通过强制市场形态
    dates = pd.bdate_range("2020-01-01", periods=200)
    # 强势上涨 + 低波动
    close = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    market = pd.DataFrame({"close": close})
    # 全部股票站上 MA（价格单调升）
    prices = pd.DataFrame(
        {f"{i:06d}": close.values * (1 + 0.01 * i) for i in range(1, 21)},
        index=dates,
    )
    df = compute_position_regime(market, prices=prices, config=cfg)
    # 后半段应接近满仓
    late = df["target_exposure"].iloc[-40:].mean()
    assert late > 0.85


def test_exposure_at_lookup():
    dates = pd.bdate_range("2023-01-01", periods=5)
    regime = pd.DataFrame(
        {"target_exposure": [0.3, 0.5, 0.7, 0.9, 1.0]},
        index=dates,
    )
    assert exposure_at(regime, dates[2]) == pytest.approx(0.7)
    assert exposure_at(regime, dates[0] - pd.Timedelta(days=10), default=1.0) == 1.0


def test_quantile_bit_identical_when_regime_off():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-01", periods=80)
    codes = [f"{i:06d}" for i in range(1, 31)]
    close = pd.DataFrame(
        10.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(dates), len(codes))), axis=0),
        index=dates, columns=codes,
    )
    open_ = close.shift(1).bfill() * 0.999
    scores = pd.DataFrame(rng.normal(0, 1, close.shape), index=dates, columns=codes)
    r1 = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
    )
    r2 = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
        position_regime=None, position_exposure=None,
    )
    pd.testing.assert_frame_equal(r1.nav, r2.nav)
    assert r1.position_exposure is None


def test_quantile_scales_with_force_exposure():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-01", periods=80)
    codes = [f"{i:06d}" for i in range(1, 31)]
    close = pd.DataFrame(
        10.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(dates), len(codes))), axis=0),
        index=dates, columns=codes,
    )
    open_ = close.shift(1).bfill() * 0.999
    scores = pd.DataFrame(rng.normal(0, 1, close.shape), index=dates, columns=codes)
    base = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
    )
    # 常数半仓
    exp = pd.Series(0.5, index=dates, name="target_exposure")
    half = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
        position_exposure=exp,
    )
    # 每期 r→0.5r 后 cumprod，期末超额应严格小于满仓（同号时）且敞口日志恒为 0.5
    top_col = [c for c in base.nav.columns if c.startswith("Top")][0]
    base_ex = base.nav[top_col].iloc[-1] - 1.0
    half_ex = half.nav[top_col].iloc[-1] - 1.0
    if base_ex > 0.02:
        assert 0 < half_ex < base_ex
    assert half.position_exposure is not None
    assert (half.position_exposure == 0.5).all()
    # 单期核对：相邻 NAV 的 period return 应满足 r_half ≈ 0.5 * r_full
    r_full = base.nav[top_col].pct_change().iloc[1]
    r_half = half.nav[top_col].pct_change().iloc[1]
    if np.isfinite(r_full) and abs(r_full) > 1e-8:
        assert r_half / r_full == pytest.approx(0.5, rel=1e-6)


def test_ml_registry_has_no_market_or_ludong_inject(monkeypatch):
    """include_regime / regime_cs 不应再把 市场*/HMM_*/轮动_* 注入 feature_names。"""
    import strategies.ml as ml

    dates = pd.bdate_range("2023-01-01", periods=40)
    codes = ["000001", "000002", "000003"]
    prices = pd.DataFrame(
        np.linspace(10, 12, len(dates))[:, None] * np.ones((1, 3)),
        index=dates, columns=codes,
    )
    mom = prices.pct_change(5).astype("float32")

    def fake_registry(**kwargs):
        return {"动量_20d": mom}

    monkeypatch.setattr(ml, "get_factor_registry", fake_registry)
    monkeypatch.setattr(
        ml, "_load_or_compute_registry",
        lambda *a, **k: {"动量_20d": mom},
    )

    ds = ml.build_factor_dataset(
        prices, pd.DataFrame(),
        hold_period=5,
        factor_whitelist=["动量_20d"],
        include_regime=True,   # 即使请求也应 no-op
        regime_cs=True,
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
    )
    assert not any(n.startswith("市场") or n.startswith("HMM_") for n in ds.feature_names)
    assert not any(n.startswith("轮动_") for n in ds.feature_names)
    assert "动量_20d" in ds.feature_names
