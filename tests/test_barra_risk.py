"""Barra 风格因子口径回归测试（2026-07 重写后）。

钉死的口径（改动这些断言前请先确认是有意变更）：
  Size      = log(流通市值)，不是 log(总资产)、也不是流通盘比例
  Liquidity = 63/252 日换手率等权平均，不是 log 成交量
  Growth    = 营收 YoY 50% + 净利润 YoY 50%
  Leverage  = 单一 DTOA
  Beta/ResVol = 对中证全指 close 的半衰期加权回归（β + HSIGMA）
  回归      = WLS，权重 = √市值
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.barra_risk import (
    _halflife_weighted_mean,
    barra_growth,
    barra_leverage,
    barra_liquidity,
    barra_regression_weights,
    barra_size,
    barra_value,
    get_barra_factors,
    market_return,
    pick_market_cap,
)
from utils.wls import normalize_weights, wls_residual

N_STOCKS = 120
CODES = [f"{i:06d}" for i in range(N_STOCKS)]
DATES = pd.bdate_range("2021-01-01", periods=420)


def _panel(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=CODES)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7)
    ret = rng.normal(0, 0.02, (len(DATES), N_STOCKS))
    prices = _panel(100 * np.exp(np.cumsum(ret, axis=0)))
    # 市值跨 3 个数量级，保证 log 与 √ 权重都有区分度
    base_mv = np.logspace(8.5, 11.5, N_STOCKS)
    circ_mv = _panel(np.tile(base_mv, (len(DATES), 1)) * (1 + 0.01 * ret.cumsum(0)))
    total_mv = circ_mv * 1.4
    turnover = _panel(np.abs(rng.normal(0.02, 0.008, (len(DATES), N_STOCKS))))
    volume = _panel(np.abs(rng.normal(1e6, 2e5, (len(DATES), N_STOCKS))))
    mkt = pd.DataFrame(
        {
            "open": np.arange(len(DATES)) + 3000.0,
            "high": np.arange(len(DATES)) + 3010.0,
            "low": np.arange(len(DATES)) + 2990.0,
            "close": 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DATES)))),
            "volume": np.full(len(DATES), 1e9),
        },
        index=DATES,
    )
    return dict(prices=prices, circ_mv=circ_mv, total_mv=total_mv,
                turnover=turnover, volume=volume, mkt=mkt)


def _financial() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for q in pd.date_range("2020-03-31", "2026-03-31", freq="QE"):
        for c in CODES:
            rows.append({
                "code": c,
                "trade_date": q,
                "total_assets": float(rng.uniform(1e8, 1e11)),
                "debt_ratio": float(rng.uniform(5, 80)),
                "revenue_growth": float(rng.normal(10, 30)),
                "net_profit_growth": float(rng.normal(8, 60)),
                "bvps": float(rng.uniform(1, 20)),
            })
    return pd.DataFrame(rows)


# ── Size ─────────────────────────────────────────────────────────────────────

def test_size_uses_circulating_market_cap_not_total_assets(synthetic):
    """Size 必须由流通市值驱动；即使财务表有 total_assets 也不得走降级路径。"""
    fin = _financial()
    size = barra_size(
        synthetic["prices"], circ_mv=synthetic["circ_mv"],
        total_mv=synthetic["total_mv"], financial=fin,
    )
    assert size is not None
    d = DATES[-1]
    log_mv = np.log(synthetic["circ_mv"].loc[d])
    rho = size.loc[d].corr(log_mv, method="spearman")
    assert rho > 0.99, f"Size 与 log(流通市值) 秩相关应≈1，实际 {rho:.4f}"

    # 与 log(total_assets) 应基本无关（证明没退回旧路径）
    ta = fin[fin["trade_date"] <= d].groupby("code")["total_assets"].last()
    rho_ta = size.loc[d].reindex(ta.index).corr(np.log(ta), method="spearman")
    assert abs(rho_ta) < 0.35, f"Size 疑似仍用 total_assets（rho={rho_ta:.3f}）"


def test_size_prefers_circ_over_total_mv(synthetic):
    mv, src = pick_market_cap(
        synthetic["prices"], circ_mv=synthetic["circ_mv"],
        total_mv=synthetic["total_mv"],
    )
    assert src == "circ_mv"
    mv2, src2 = pick_market_cap(synthetic["prices"], circ_mv=None,
                                total_mv=synthetic["total_mv"])
    assert src2 == "total_mv"


def test_size_falls_back_to_total_assets_only_without_market_cap(synthetic):
    size = barra_size(synthetic["prices"], financial=_financial())
    assert size is not None, "无市值面板时应 warning 降级而不是直接崩"
    assert barra_size(synthetic["prices"], financial=None) is None


# ── Liquidity ────────────────────────────────────────────────────────────────

def test_liquidity_uses_turnover_not_volume(synthetic):
    liq = barra_liquidity(
        synthetic["prices"], turnover_rate=synthetic["turnover"],
        volume=synthetic["volume"],
    )
    assert liq is not None
    d = DATES[-1]
    turn_avg = synthetic["turnover"].rolling(63).mean().loc[d]
    vol_avg = synthetic["volume"].rolling(20).mean().loc[d]
    rho_turn = liq.loc[d].corr(turn_avg, method="spearman")
    rho_vol = liq.loc[d].corr(vol_avg, method="spearman")
    assert rho_turn > 0.5, f"Liquidity 应主要由换手率驱动（rho={rho_turn:.3f}）"
    assert rho_turn > abs(rho_vol)


def test_liquidity_derives_turnover_from_amount_over_circ_mv(synthetic):
    amount = synthetic["turnover"] * synthetic["circ_mv"]
    liq = barra_liquidity(
        synthetic["prices"], turnover_rate=None, amount=amount,
        circ_mv=synthetic["circ_mv"],
    )
    assert liq is not None
    ref = barra_liquidity(synthetic["prices"], turnover_rate=synthetic["turnover"])
    d = DATES[-1]
    assert liq.loc[d].corr(ref.loc[d]) > 0.99


def test_liquidity_averages_both_windows(synthetic):
    """63/252 双窗等权：结果须同时和两个窗口相关，且不等于任一单窗。"""
    both = barra_liquidity(synthetic["prices"], turnover_rate=synthetic["turnover"])
    only63 = barra_liquidity(
        synthetic["prices"], turnover_rate=synthetic["turnover"], windows=(63,),
    )
    only252 = barra_liquidity(
        synthetic["prices"], turnover_rate=synthetic["turnover"], windows=(252,),
    )
    d = DATES[-1]
    assert both.loc[d].corr(only63.loc[d]) < 0.999
    assert both.loc[d].corr(only252.loc[d]) < 0.999
    assert both.loc[d].corr(only63.loc[d]) > 0.3
    assert both.loc[d].corr(only252.loc[d]) > 0.3


# ── Growth ───────────────────────────────────────────────────────────────────

def test_growth_blends_revenue_and_profit_yoy(synthetic):
    fin = _financial()
    both = barra_growth(fin, synthetic["prices"])
    rev_only = barra_growth(fin.drop(columns=["net_profit_growth"]),
                            synthetic["prices"])
    profit_only = barra_growth(fin.drop(columns=["revenue_growth"]),
                               synthetic["prices"])
    d = DATES[-1]
    c_rev = both.loc[d].corr(rev_only.loc[d])
    c_pro = both.loc[d].corr(profit_only.loc[d])
    assert c_rev > 0.4 and c_pro > 0.4, "两腿都应有实质贡献"
    assert c_rev < 0.99 and c_pro < 0.99, "不应退化成单腿"
    # 等权：两腿贡献度接近
    assert abs(c_rev - c_pro) < 0.25


def test_financial_barra_drops_b_share_absent_from_prices():
    """财务长表含 200xxx 时，prices 无 B → normalize 后列不含 200。"""
    dates = pd.bdate_range("2021-01-04", periods=8)
    a_codes = ["000001", "600000", "300001"]
    prices = pd.DataFrame(10.0, index=dates, columns=a_codes)
    rows = []
    for q in pd.date_range("2020-03-31", "2021-03-31", freq="QE"):
        for i, c in enumerate(a_codes + ["200001"]):
            rows.append({
                "code": c,
                "trade_date": q,
                "debt_ratio": 99.0 if c.startswith("200") else 20.0 + 10.0 * i,
                "revenue_growth": 50.0 if c.startswith("200") else 5.0 + 8.0 * i,
                "net_profit_growth": 80.0 if c.startswith("200") else 4.0 + 6.0 * i,
                "pb": 0.1 if c.startswith("200") else 1.5 + 0.5 * i,
            })
    fin = pd.DataFrame(rows)
    for name, df in (
        ("Leverage", barra_leverage(fin, prices)),
        ("Growth", barra_growth(fin, prices)),
        ("Value", barra_value(fin, prices)),
    ):
        assert df is not None, name
        cols = [str(c) for c in df.columns]
        assert "200001" not in cols, name
        assert list(df.columns) == a_codes, name


# ── 市场收益 / Beta ───────────────────────────────────────────────────────────

def test_market_return_picks_close_column(synthetic):
    r = market_return(synthetic["mkt"])
    assert isinstance(r, pd.Series)
    expected = synthetic["mkt"]["close"].pct_change()
    pd.testing.assert_series_equal(r, expected, check_names=False)


def test_beta_resvol_masked_after_long_gap(synthetic):
    """退市/长停后 ewm 不得无限 carry forward 旧 beta。"""
    prices = synthetic["prices"].copy()
    ret = prices.pct_change()
    dead = CODES[0]
    # 前期有足够样本算出 beta，最后 200 天停牌 → 窗内有效日 < 126，必须置 NaN
    ret.loc[:, dead] = ret[dead].where(ret.index < DATES[-200])
    fac = get_barra_factors(
        prices=prices, financial=None, market_prices=synthetic["mkt"],
        clean_ret=ret, circ_mv=synthetic["circ_mv"],
        turnover_rate=synthetic["turnover"],
    )
    beta = fac["Barra_Beta"].reindex(columns=CODES)
    resvol = fac["Barra_ResVol"].reindex(columns=CODES)
    assert beta[dead].notna().any(), "停牌前应算得出 beta"
    assert np.isnan(beta.loc[DATES[-1], dead])
    assert np.isnan(resvol.loc[DATES[-1], dead])
    assert beta.loc[DATES[-1]].notna().sum() > N_STOCKS // 2


# ── 半衰期加权 ───────────────────────────────────────────────────────────────

def test_halflife_weighted_mean_matches_bruteforce():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(80, 4)))
    window, hl = 20, 5.0
    out = _halflife_weighted_mean(df, window, hl, min_frac=0.5)

    lam = 0.5 ** (1 / hl)
    w = lam ** np.arange(window)
    t = 60
    for c in range(4):
        seg = df.iloc[t - window + 1:t + 1, c].to_numpy()[::-1]  # 近→远
        expected = float((w * seg).sum() / w.sum())
        assert out.iloc[t, c] == pytest.approx(expected, rel=1e-4)


def test_halflife_weighted_mean_renormalizes_over_valid_obs():
    """NaN（涨跌停）应从分母剔除，而不是当 0 收益拉低动量。"""
    df = pd.DataFrame({"a": [1.0] * 40})
    df.iloc[35, 0] = np.nan
    out = _halflife_weighted_mean(df, 20, 5.0, min_frac=0.5)
    assert out.iloc[39, 0] == pytest.approx(1.0, rel=1e-5)


def test_halflife_weighted_mean_respects_min_frac():
    df = pd.DataFrame({"a": [np.nan] * 30 + [1.0] * 3})
    out = _halflife_weighted_mean(df, 20, 5.0, min_frac=0.9)
    assert np.isnan(out.iloc[-1, 0])


# ── WLS ──────────────────────────────────────────────────────────────────────

def test_wls_residual_matches_closed_form():
    rng = np.random.default_rng(5)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    w = rng.uniform(0.5, 40, n)
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.5, n)

    resid = wls_residual(y, X, w)
    A = np.column_stack([np.ones(n), X])
    W = np.diag(normalize_weights(w, n))
    beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    np.testing.assert_allclose(resid, y - A @ beta, rtol=1e-8, atol=1e-8)


def test_wls_residual_orthogonal_under_weights():
    """WLS 残差应与设计矩阵在加权内积下正交（等权 OLS 则不然）。"""
    rng = np.random.default_rng(6)
    n = 300
    X = rng.normal(size=(n, 2))
    w = np.exp(rng.normal(0, 1.5, n))
    y = X[:, 0] * 2 + rng.normal(0, 1, n)

    r_wls = wls_residual(y, X, w)
    wn = normalize_weights(w, n)
    assert abs(float((wn * r_wls) @ X[:, 0])) < 1e-6
    r_ols = wls_residual(y, X, None)
    assert abs(float((wn * r_ols) @ X[:, 0])) > 1e-4
    assert not np.allclose(r_wls, r_ols)


def test_wls_falls_back_to_ols_on_bad_weights():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 2))
    y = rng.normal(size=60)
    ols = wls_residual(y, X, None)
    for bad in (np.zeros(60), np.full(60, np.nan), np.full(59, 2.0)):
        np.testing.assert_allclose(wls_residual(y, X, bad), ols, atol=1e-10)


def test_regression_weights_are_sqrt_market_cap(synthetic):
    w = barra_regression_weights(
        synthetic["prices"], circ_mv=synthetic["circ_mv"],
        total_mv=synthetic["total_mv"],
    )
    d = DATES[-1]
    np.testing.assert_allclose(
        w.loc[d].to_numpy(dtype=float),
        np.sqrt(synthetic["circ_mv"].loc[d].to_numpy(dtype=float)),
        rtol=1e-5,
    )
    assert barra_regression_weights(synthetic["prices"]) is None


# ── residualize_panel 端到端 ─────────────────────────────────────────────────

def test_residualize_panel_wls_differs_from_ols(synthetic):
    from models.wf.labels import residualize_panel

    rng = np.random.default_rng(9)
    fac = get_barra_factors(
        prices=synthetic["prices"], financial=_financial(),
        market_prices=synthetic["mkt"], circ_mv=synthetic["circ_mv"],
        total_mv=synthetic["total_mv"], turnover_rate=synthetic["turnover"],
    )
    ind = pd.Series(
        np.where(np.arange(N_STOCKS) % 2 == 0, "银行", "地产"), index=CODES,
    )
    alpha = _panel(rng.normal(size=(len(DATES), N_STOCKS)))
    dates = DATES[-6:]
    w = barra_regression_weights(
        synthetic["prices"], circ_mv=synthetic["circ_mv"],
    )

    r_ols = residualize_panel(alpha, fac, ind, dates)
    r_wls = residualize_panel(alpha, fac, ind, dates, weight_panel=w)
    a, b = r_ols.loc[dates[-1]], r_wls.loc[dates[-1]]
    assert a.notna().sum() > 50
    assert not np.allclose(a.dropna(), b.dropna())
    assert a.corr(b) > 0.8
