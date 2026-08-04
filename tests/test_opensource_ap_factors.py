"""
tests/test_opensource_ap_factors.py — OpenSourceAP Batch-1/2 因子单测

合成季报/价量面板验证方向、构造与 registry 登记。
"""
from __future__ import annotations

import os

# 合成面板签名易与磁盘缓存脏结果碰撞；单测强制重算
os.environ["FACTOR_CACHE_DISABLE"] = "1"

import numpy as np
import pandas as pd

from factors.factor_opensource_ap import (
    OPENSOURCE_AP_ACRONYM,
    OPENSOURCE_AP_FACTOR_NAMES,
    _earnings_consistency_quarterly,
    _mean_rank_rev_growth_quarterly,
    _num_earn_increase_quarterly,
    _yoy_growth,
    factor_accruals_cf,
    factor_asset_growth,
    factor_assets_to_market,
    factor_cfp,
    factor_cheq,
    factor_ch_asset_turnover_approx,
    factor_comp_equ_iss,
    factor_coskewness,
    factor_earnings_consistency,
    factor_firm_age,
    factor_max_ret,
    factor_mean_rank_rev_growth,
    factor_mom_season,
    factor_num_earn_increase,
    factor_op_leverage_approx,
    factor_oper_prof_approx,
    factor_pct_acc,
    factor_residual_momentum,
    factor_return_skew,
    factor_share_iss,
    factor_sp_approx,
    factor_xfin_approx,
    get_opensource_ap_factors,
)
from factors.factor import get_factor_names, get_factor_registry, factor_quality_accrual


# 交易日覆盖季报 PIT 后窗口；季报 ≥24 期以覆盖 MeanRank 5 年滞后
DATES = pd.date_range("2015-01-02", periods=2800, freq="B")
Q_DATES = pd.date_range("2015-03-31", periods=40, freq="QE")
CODES = ["A", "B", "C", "D"]


def _synth_prices():
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(index=DATES, columns=CODES, dtype=float)
    for i, c in enumerate(CODES):
        rets = rng.normal(0.0003, 0.012, len(DATES))
        prices[c] = (10.0 + i) * np.cumprod(1.0 + rets)
    return prices


def _synth_clean_ret(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def _synth_financial():
    """构造有差异的资产/权益/EPS/营收增长序列。"""
    rows = []
    at0 = {"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0}
    bv0 = {"A": 5.0, "B": 5.0, "C": 5.0, "D": 5.0}
    at_g = {"A": 0.20, "B": -0.05, "C": 0.05, "D": 0.02}
    bv_g = {"A": 0.15, "B": -0.03, "C": 0.04, "D": 0.01}
    for t, q in enumerate(Q_DATES):
        for c in CODES:
            at0[c] *= (1.0 + at_g[c] / 4)
            bv0[c] *= (1.0 + bv_g[c] / 4)
            if c == "A":
                eps = 1.0 + 0.05 * t  # 稳定连增
            elif c == "B":
                eps = 1.0 + (0.3 if (t // 4) % 2 == 0 else -0.3)
            else:
                eps = 1.0 + 0.02 * t
            rev_g = {"A": 40.0, "B": -10.0, "C": 10.0, "D": 5.0}[c]
            if c == "A":
                rev_g = 50.0 - 2.0 * t
            rows.append({
                "trade_date": q,
                "code": c,
                "total_assets": at0[c],
                "bvps": max(bv0[c], 0.1),
                "eps": eps,
                "operating_cashflow": 0.5 + 0.1 * (CODES.index(c)),
                "revenue_growth": rev_g,
                "roe": 10.0,
                "gross_profit_margin": {"A": 40.0, "B": 10.0, "C": 25.0, "D": 20.0}[c],
                "net_profit_margin": {"A": 12.0, "B": 3.0, "C": 8.0, "D": 6.0}[c],
                "debt_ratio": {"A": 60.0, "B": 20.0, "C": 40.0, "D": 35.0}[c],
            })
    return pd.DataFrame(rows)


def _synth_mcap(prices: pd.DataFrame) -> pd.DataFrame:
    scale = {"A": 1e9, "B": 2e9, "C": 5e9, "D": 2e10}
    return pd.DataFrame({c: scale[c] for c in CODES}, index=prices.index, dtype=float)


def _synth_shares(prices: pd.DataFrame) -> pd.DataFrame:
    """人为制造 A 股本扩张、B 股本收缩。"""
    sh = pd.DataFrame(1e8, index=prices.index, columns=CODES, dtype=float)
    # A: 线性扩张；B: 收缩
    t = np.arange(len(prices), dtype=float)
    sh["A"] = 1e8 * (1.0 + 0.5 * t / len(prices))
    sh["B"] = 1e8 * (1.0 - 0.2 * t / len(prices))
    return sh


# ── Batch-1 ──────────────────────────────────────────────────────────────────

def test_acronym_mapping_complete():
    assert OPENSOURCE_AP_FACTOR_NAMES == set(OPENSOURCE_AP_ACRONYM)
    assert OPENSOURCE_AP_ACRONYM["资产增长"] == "AssetGrowth"
    assert OPENSOURCE_AP_ACRONYM["应计占比"] == "PctAcc"
    assert OPENSOURCE_AP_ACRONYM["盈利连增期数"] == "NumEarnIncrease"
    assert OPENSOURCE_AP_ACRONYM["应计资产比"] == "Accruals"
    assert OPENSOURCE_AP_ACRONYM["协偏度"] == "Coskewness"
    assert OPENSOURCE_AP_ACRONYM["残差动量"] == "ResidualMomentum"
    assert len(OPENSOURCE_AP_FACTOR_NAMES) == 30


def test_yoy_growth_four_quarters():
    idx = pd.date_range("2020-03-31", periods=8, freq="QE")
    level = pd.DataFrame({"X": [100, 110, 120, 130, 200, 210, 220, 230]}, index=idx)
    g = _yoy_growth(level, 4)
    assert np.isnan(g.iloc[3, 0])
    assert abs(g.iloc[4, 0] - 1.0) < 1e-9


def test_asset_growth_low_expansion_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_asset_growth(fin, prices)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["B"] > last["A"]


def test_cheq_low_equity_growth_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_cheq(fin, prices)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["B"] > last["A"]


def test_am_high_assets_to_mcap_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial()
    fin = fin.copy()
    fin["total_assets"] = 1e10
    mv = _synth_mcap(prices)
    panel = factor_assets_to_market(fin, prices, circ_mv=mv)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["A"] > last["D"]


def test_cfp_higher_ocf_per_price_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial()
    px = pd.DataFrame(10.0, index=prices.index, columns=prices.columns)
    fin = fin.copy()
    fin.loc[fin["code"] == "D", "operating_cashflow"] = 5.0
    fin.loc[fin["code"] == "A", "operating_cashflow"] = 0.1
    panel = factor_cfp(fin, prices, prices_raw=px)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["D"] > last["A"]


def test_earnings_consistency_filters_sign_flip():
    idx = pd.date_range("2015-03-31", periods=24, freq="QE")
    stable = np.linspace(1.0, 2.0, len(idx))
    flip = np.array([1.0 if (i // 4) % 2 == 0 else -1.0 for i in range(len(idx))], dtype=float)
    eps = pd.DataFrame({"S": stable, "F": flip}, index=idx)
    cons = _earnings_consistency_quarterly(eps)
    late = cons.iloc[-4:]
    assert late["S"].notna().sum() >= late["F"].notna().sum()


def test_mean_rank_high_rank_is_low_growth():
    idx = pd.date_range("2015-03-31", periods=24, freq="QE")
    g = pd.DataFrame({"H": 50.0, "M": 10.0, "L": -5.0}, index=idx)
    raw = _mean_rank_rev_growth_quarterly(g)
    late = raw.iloc[-1]
    assert late["L"] > late["H"]


def test_mean_rank_factor_panel_shape():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_mean_rank_rev_growth(fin, prices)
    assert panel is not None
    assert panel.shape == prices.shape


def test_earnings_consistency_factor_runs():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_earnings_consistency(fin, prices)
    assert panel is not None
    assert panel.shape == prices.shape


# ── Batch-2 ──────────────────────────────────────────────────────────────────

def test_num_earn_increase_streak():
    idx = pd.date_range("2015-03-31", periods=20, freq="QE")
    # 稳定上行 → 连增；震荡 → 低连增
    up = np.linspace(1.0, 3.0, len(idx))
    zig = np.array([1.0 + (0.5 if (i // 2) % 2 == 0 else -0.5) for i in range(len(idx))])
    eps = pd.DataFrame({"UP": up, "ZIG": zig}, index=idx)
    n = _num_earn_increase_quarterly(eps)
    # 晚期 UP 连增期数应 ≥ ZIG
    assert n.iloc[-1]["UP"] >= n.iloc[-1]["ZIG"]


def test_num_earn_increase_factor_panel():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_num_earn_increase(fin, prices)
    assert panel is not None
    assert panel.shape == prices.shape
    assert panel.iloc[-1].notna().any()


def test_share_iss_low_issuance_scores_higher():
    prices = _synth_prices()
    sh = _synth_shares(prices)
    panel = factor_share_iss(prices, years=1, shares=sh)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    # B 收缩（负发行）取负后应高于扩张的 A
    assert last["B"] > last["A"]


def test_pct_acc_low_accrual_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial().copy()
    # A: 高应计 eps>>ocf；B: 低应计 ocf≈eps
    fin.loc[fin["code"] == "A", "eps"] = 2.0
    fin.loc[fin["code"] == "A", "operating_cashflow"] = 0.1
    fin.loc[fin["code"] == "B", "eps"] = 1.0
    fin.loc[fin["code"] == "B", "operating_cashflow"] = 0.95
    panel = factor_pct_acc(fin, prices)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["B"] > last["A"]


def test_quality_accrual_eps_fallback():
    """financial_indicators 无 net_profit 时 质量_应计 应走 eps fallback。"""
    prices = _synth_prices()
    fin = _synth_financial()
    assert "net_profit" not in fin.columns
    panel = factor_quality_accrual(fin, prices)
    assert panel is not None
    assert panel.shape == prices.shape


def test_firm_age_younger_scores_higher():
    prices = _synth_prices()
    listing = {
        "A": prices.index[0],
        "B": prices.index[500],
        "C": prices.index[1000],
        "D": prices.index[1500],
    }
    panel = factor_firm_age(prices, listing_dates=listing)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    # D 最年轻 → 取负年龄后最高分
    assert last["D"] > last["A"]


def test_max_ret_and_skew_run():
    prices = _synth_prices()
    cr = _synth_clean_ret(prices)
    mx = factor_max_ret(prices, clean_ret=cr)
    sk = factor_return_skew(prices, clean_ret=cr)
    assert mx is not None and sk is not None
    assert mx.shape == prices.shape
    assert sk.iloc[-1].notna().any()


def test_sp_approx_runs():
    prices = _synth_prices()
    fin = _synth_financial()
    panel = factor_sp_approx(fin, prices, prices_raw=prices)
    assert panel is not None
    assert panel.iloc[-1].notna().any()


def test_comp_equ_iss_runs():
    prices = _synth_prices()
    cr = _synth_clean_ret(prices)
    mv = _synth_mcap(prices)
    panel = factor_comp_equ_iss(prices, clean_ret=cr, circ_mv=mv)
    assert panel is not None
    # 需要 ~60 个月历史，末段应有值
    assert panel.iloc[-1].notna().any()


def test_get_opensource_ap_factors_batch2_subset():
    prices = _synth_prices()
    fin = _synth_financial()
    mv = _synth_mcap(prices)
    cr = _synth_clean_ret(prices)
    sh = _synth_shares(prices)
    # monkeypatch shares loader via explicit factor calls covered above;
    # batch entry for accounting factors:
    out = get_opensource_ap_factors(
        prices, financial=fin, circ_mv=mv, prices_raw=prices, clean_ret=cr,
        factor_names={"盈利连增期数", "应计占比", "营收市值比", "上市年龄", "月最大收益"},
    )
    assert "盈利连增期数" in out
    assert "应计占比" in out
    assert "营收市值比" in out
    assert "上市年龄" in out
    assert "月最大收益" in out
    for p in out.values():
        assert p.shape == prices.shape


def test_get_factor_names_includes_batch2():
    prices = _synth_prices()
    fin = _synth_financial()
    names = get_factor_names(
        prices=prices, financial=fin, circ_mv=_synth_mcap(prices),
        clean_ret=_synth_clean_ret(prices),
    )
    for n in (
        "盈利连增期数", "应计占比", "上市年龄", "月最大收益",
        "营收市值比", "净债务市值比", "综合债务融资",
    ):
        assert n in names, n


def test_registry_computes_osap_batch1():
    prices = _synth_prices()
    fin = _synth_financial()
    mv = _synth_mcap(prices)
    batch1 = {
        "资产增长", "资产市值比", "现金流市值比",
        "权益增长", "盈利一致性", "营收增长秩",
    }
    reg = get_factor_registry(
        prices=prices,
        financial=fin,
        prices_raw=prices,
        circ_mv=mv,
        factor_names=list(batch1),
    )
    assert batch1 <= set(reg.keys())
    for name, panel in reg.items():
        assert panel.shape[0] == len(prices)
        assert panel.notna().any().any()


def test_registry_computes_batch2_accounting():
    prices = _synth_prices()
    # 错开各股首个有效价，避免无 list_date 时 FirmAge 截面恒等 → zscore 全 NaN
    for i, c in enumerate(CODES):
        prices.loc[prices.index[: i * 200], c] = np.nan
    fin = _synth_financial()
    mv = _synth_mcap(prices)
    cr = _synth_clean_ret(prices)
    names = ["盈利连增期数", "应计占比", "营收市值比", "上市年龄", "月最大收益", "收益偏度"]
    reg = get_factor_registry(
        prices=prices,
        financial=fin,
        prices_raw=prices,
        circ_mv=mv,
        clean_ret=cr,
        factor_names=names,
    )
    for n in names:
        assert n in reg, n
        assert reg[n].notna().any().any(), n


# ── Batch-3 ──────────────────────────────────────────────────────────────────

def _synth_market(prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0002, 0.008, len(prices))
    close = 1000.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({"close": close}, index=prices.index)


def test_accruals_cf_low_accrual_scores_higher():
    prices = _synth_prices()
    fin = _synth_financial().copy()
    sh = _synth_shares(prices)
    fin.loc[fin["code"] == "A", "eps"] = 2.0
    fin.loc[fin["code"] == "A", "operating_cashflow"] = 0.1
    fin.loc[fin["code"] == "B", "eps"] = 1.0
    fin.loc[fin["code"] == "B", "operating_cashflow"] = 0.95
    panel = factor_accruals_cf(fin, prices, shares=sh)
    assert panel is not None
    last = panel.iloc[-1].dropna()
    assert last["B"] > last["A"]


def test_oper_prof_and_op_leverage_run():
    prices = _synth_prices()
    fin = _synth_financial()
    sh = _synth_shares(prices)
    mv = _synth_mcap(prices)
    op = factor_oper_prof_approx(fin, prices, shares=sh, circ_mv=mv)
    opl = factor_op_leverage_approx(fin, prices, shares=sh)
    assert op is not None and opl is not None
    assert op.iloc[-1].notna().any()
    assert opl.iloc[-1].notna().any()


def test_xfin_and_ch_ato_run():
    prices = _synth_prices()
    fin = _synth_financial()
    sh = _synth_shares(prices)
    xfin = factor_xfin_approx(fin, prices, shares=sh, prices_raw=prices)
    ato = factor_ch_asset_turnover_approx(fin, prices, shares=sh)
    assert xfin is not None and ato is not None
    assert xfin.shape == prices.shape
    assert ato.iloc[-1].notna().any()


def test_coskew_resmom_momseason_run():
    prices = _synth_prices()
    cr = _synth_clean_ret(prices)
    # 注入市场相关收益，避免残差/协偏度全空
    mkt = _synth_market(prices)
    mret = mkt["close"].pct_change()
    cr = cr.add(mret * 0.8, axis=0)
    cosk = factor_coskewness(prices, clean_ret=cr, market_prices=mkt)
    resm = factor_residual_momentum(
        prices, clean_ret=cr, market_prices=mkt,
        beta_window=252, mom_window=120, skip=21, min_beta=180, min_mom=60,
    )
    seas = factor_mom_season(prices, clean_ret=cr)
    assert cosk is not None and resm is not None and seas is not None
    assert cosk.iloc[-1].notna().any()
    assert resm.iloc[-1].notna().any()
    assert seas.iloc[-1].notna().any()


def test_get_opensource_ap_factors_batch3(monkeypatch):
    prices = _synth_prices()
    fin = _synth_financial()
    mv = _synth_mcap(prices)
    cr = _synth_clean_ret(prices)
    mkt = _synth_market(prices)
    sh = _synth_shares(prices)
    monkeypatch.setattr(
        "factors.factor_opensource_ap._shares_panel",
        lambda *a, **k: sh,
    )
    mret = mkt["close"].pct_change()
    cr = cr.add(mret * 0.5, axis=0)
    out = get_opensource_ap_factors(
        prices, financial=fin, circ_mv=mv, prices_raw=prices,
        clean_ret=cr, market_prices=mkt,
        factor_names={
            "应计资产比", "经营利润权益比", "外部融资资产比",
            "资产周转变化", "经营杠杆", "协偏度", "残差动量", "季节动量",
        },
    )
    for n in (
        "应计资产比", "经营利润权益比", "外部融资资产比",
        "资产周转变化", "经营杠杆", "协偏度", "季节动量",
    ):
        assert n in out, n
        assert out[n].shape == prices.shape
    # 残差动量默认窗较长；合成面板末段应有值（若无则缩短窗已在单测覆盖）
    assert "残差动量" in out


def test_get_factor_names_includes_batch3():
    prices = _synth_prices()
    fin = _synth_financial()
    mkt = _synth_market(prices)
    names = get_factor_names(
        prices=prices, financial=fin, circ_mv=_synth_mcap(prices),
        clean_ret=_synth_clean_ret(prices), market_prices=mkt,
    )
    for n in (
        "应计资产比", "经营利润权益比", "外部融资资产比",
        "资产周转变化", "经营杠杆", "协偏度", "残差动量", "季节动量",
    ):
        assert n in names, n


def test_registry_computes_batch3(monkeypatch):
    prices = _synth_prices()
    fin = _synth_financial()
    mv = _synth_mcap(prices)
    cr = _synth_clean_ret(prices)
    mkt = _synth_market(prices)
    sh = _synth_shares(prices)
    monkeypatch.setattr(
        "factors.factor_opensource_ap._shares_panel",
        lambda *a, **k: sh,
    )
    mret = mkt["close"].pct_change()
    cr = cr.add(mret * 0.5, axis=0)
    names = [
        "应计资产比", "经营利润权益比", "协偏度", "季节动量",
    ]
    reg = get_factor_registry(
        prices=prices, financial=fin, prices_raw=prices,
        circ_mv=mv, clean_ret=cr, market_prices=mkt,
        factor_names=names,
    )
    for n in names:
        assert n in reg, n
        assert reg[n].notna().any().any(), n
