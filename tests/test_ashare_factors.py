"""
tests/test_ashare_factors.py — A 股特色因子单测（合成数据，不打外网）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.factor_ashare import (
    ASHARE_DENSE_FACTOR_NAMES,
    ASHARE_EVENT_FACTOR_NAMES,
    ASHARE_SPARSE_FACTOR_NAMES,
    _cs_residual,
    _lhb_reason_bucket,
    _parse_lhb_institution,
    factor_block_discount_seat_quality,
    factor_block_discount_vol,
    factor_block_inst_takeover,
    factor_block_seller_inst_pressure,
    factor_cb_conversion_dilution,
    factor_lhb_inst_net_buy,
    factor_lhb_net_buy_pct,
    factor_lhb_reason_down_avoid,
    factor_lhb_reason_turnover,
    factor_lhb_reason_up,
    factor_lockup_adv_pressure,
    factor_lockup_incentive_pressure,
    factor_lockup_placement_pressure,
    factor_margin_balance_to_float,
    factor_margin_buy_to_amount,
    factor_margin_net_buy,
    factor_moneyflow_residual,
    factor_rating_downgrade_avoid,
    factor_rating_upgrade,
    factor_repurchase_completion,
    factor_repurchase_intensity,
    factor_research_coverage,
    factor_research_eps_dispersion,
    factor_research_eps_slope,
    factor_research_eps_upgrade,
    factor_restricted_listing_supply,
    factor_short_sell_avoid,
    factor_target_price_upside,
    factor_yjbb_surprise_raw,
    get_ashare_factors,
)
from factors.sparse_factors import SPARSE_FACTOR_NAMES
from factors.factor import EVENT_OVERLAY_FACTOR_NAMES, get_factor_names


DATES = pd.date_range("2024-01-02", periods=40, freq="B")
CODES = ["000001", "000002", "000003"]


def _prices() -> pd.DataFrame:
    base = np.linspace(10.0, 20.0, len(DATES))
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {c: base * (1.0 + 0.01 * i) + rng.normal(0, 0.05, len(DATES))
         for i, c in enumerate(CODES)},
        index=DATES,
    )


def test_parse_lhb_institution():
    assert _parse_lhb_institution("3家机构买入，成功率40%") == (3.0, 0.0)
    assert _parse_lhb_institution("2家机构卖出，成功率30%") == (0.0, 2.0)
    assert _parse_lhb_institution("主力做T") == (0.0, 0.0)


def test_cs_residual_orthogonal():
    """残差应与控制变量近似正交（截面）。"""
    idx = DATES[:10]
    cols = CODES
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    noise = pd.DataFrame(rng.normal(scale=0.1, size=x.shape), index=idx, columns=cols)
    y = 2.0 * x + noise
    resid = _cs_residual(y, x, min_obs=3)
    # 多数日期残差与 x 相关应接近 0
    corrs = []
    for dt in resid.index:
        r = resid.loc[dt].dropna()
        xx = x.loc[dt].reindex(r.index)
        if len(r) >= 3:
            corrs.append(np.corrcoef(r, xx)[0, 1])
    assert np.nanmean(np.abs(corrs)) < 0.25


def test_moneyflow_residual_shape():
    prices = _prices()
    mf = pd.DataFrame(
        np.random.default_rng(2).normal(size=prices.shape),
        index=prices.index, columns=prices.columns,
    )
    amt = pd.DataFrame(
        np.abs(np.random.default_rng(3).normal(1e8, 1e7, size=prices.shape)),
        index=prices.index, columns=prices.columns,
    )
    ret = prices.pct_change()
    panel = factor_moneyflow_residual(mf, amount=amt, clean_ret=ret, prices=prices)
    assert panel.shape == prices.shape
    assert panel.notna().any().any()


def test_rating_upgrade_from_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rf = pd.DataFrame({
        "code": ["000001", "000002", "000001"],
        "announce_date": [DATES[5], DATES[6], DATES[10]],
        "rating_change": ["调高", "维持", "调高"],
        "is_first": ["不是首次评级"] * 3,
        "rating": ["买入", "增持", "买入"],
        "institute": ["A", "B", "C"],
    })
    rf.to_parquet(tmp_path / "rank_forecast.parquet")
    panel = factor_rating_upgrade(prices, window=20)
    assert panel.shape == prices.shape
    # 000001 应有正信号
    assert panel["000001"].notna().any() or (panel["000001"].fillna(0) != 0).any()


def test_research_eps_upgrade(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rr = pd.DataFrame({
        "code": ["000001", "000001", "000002"],
        "announce_date": [DATES[3], DATES[8], DATES[5]],
        "eps_forecast": [1.0, 1.2, 0.8],
        "institute": ["A", "A", "B"],
        "title": ["t1", "t2", "t3"],
        "rating": ["买入"] * 3,
    })
    rr.to_parquet(tmp_path / "research_report.parquet")
    panel = factor_research_eps_upgrade(prices, window=20)
    assert panel.shape == prices.shape


def test_lhb_inst_net_buy(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    lhb = pd.DataFrame({
        "code": ["000001", "000002", "000001"],
        "lhb_date": [DATES[4], DATES[4], DATES[7]],
        "interpretation": ["2家机构买入，成功率40%", "1家机构卖出，成功率30%", "3家机构买入"],
        "net_buy": [1e7, -5e6, 2e7],
        "reason": ["a", "b", "c"],
    })
    lhb.to_parquet(tmp_path / "lhb_detail.parquet")
    panel = factor_lhb_inst_net_buy(prices, window=20)
    assert panel.shape == prices.shape
    assert panel.notna().any().any()


def test_repurchase_intensity(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rep = pd.DataFrame({
        "code": ["000001", "000002"],
        "announce_date": [DATES[5], DATES[8]],
        "plan_pct_hi": [2.0, 1.0],
        "plan_amt_hi": [np.nan, np.nan],
        "progress": ["董事会预案", "实施中"],
    })
    rep.to_parquet(tmp_path / "repurchase.parquet")
    panel = factor_repurchase_intensity(prices, window=20)
    assert panel.shape == prices.shape


def test_block_discount_seat_quality(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    bt = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "trade_date": [DATES[5], DATES[5], DATES[6]],
        "discount_rate": [-0.1, -0.05, 0.02],
        "buyer_branch": ["机构专用", "某某营业部", "机构专用"],
    })
    bt.to_parquet(tmp_path / "block_trade.parquet")
    panel = factor_block_discount_seat_quality(prices, window=10)
    assert panel.shape == prices.shape


def test_yjbb_surprise_raw(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    yjbb = pd.DataFrame({
        "code": ["000001", "000002"],
        "announce_date": [DATES[10], DATES[12]],
        "report_date": ["20241231", "20241231"],
        "net_profit_yoy": [50.0, -10.0],
    })
    yjbb.to_parquet(tmp_path / "yjbb.parquet")
    yjyg = pd.DataFrame({
        "code": ["000001", "000002"],
        "report_date": ["20241231", "20241231"],
        "change_pct": [30.0, 0.0],
        "announce_date": [DATES[5], DATES[5]],
    })
    yjyg.to_parquet(tmp_path / "yjyg.parquet")
    panel = factor_yjbb_surprise_raw(prices, window=10)
    assert panel.shape == prices.shape
    # 000001 surprise ≈ 50-30 = 20
    assert panel.loc[DATES[10], "000001"] == pytest.approx(20.0)


def test_registry_names_include_ashare():
    assert "大单残差净流入_5d" in ASHARE_DENSE_FACTOR_NAMES
    assert "评级上修_20d" in ASHARE_SPARSE_FACTOR_NAMES
    assert "业绩快报超预期" in ASHARE_EVENT_FACTOR_NAMES
    assert "业绩快报超预期" in EVENT_OVERLAY_FACTOR_NAMES
    for n in ASHARE_SPARSE_FACTOR_NAMES:
        assert n in SPARSE_FACTOR_NAMES
    # get_factor_names 在有 moneyflow 时应枚举残差因子
    prices = _prices()
    mf = prices.copy()
    names = get_factor_names(
        prices, financial=pd.DataFrame(), moneyflow=mf, factor_names=None,
    )
    assert "大单残差净流入_5d" in names
    for n in ASHARE_SPARSE_FACTOR_NAMES:
        assert n in names


def test_get_ashare_factors_missing_data_no_crash(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    out = get_ashare_factors(prices, factor_names={"评级上修_20d", "股份回购强度_60d"})
    # 无文件时返回空 dict 或跳过，不抛错
    assert isinstance(out, dict)


def test_special_sparse_pack_includes_ashare():
    from factors.special_factors import SPECIAL_FACTOR_PACKS
    sparse = SPECIAL_FACTOR_PACKS["sparse"]
    assert "龙虎榜机构净买入_20d" in sparse.factor_names
    assert "评级上修_20d" in sparse.factor_names


def test_lhb_net_buy_pct(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    lhb = pd.DataFrame({
        "code": ["000001", "000002", "000001"],
        "lhb_date": [DATES[4], DATES[4], DATES[7]],
        "net_buy_pct": [5.0, -2.0, 3.0],
        "interpretation": ["a", "b", "c"],
        "net_buy": [1e7, -1e6, 2e7],
    })
    lhb.to_parquet(tmp_path / "lhb_detail.parquet")
    panel = factor_lhb_net_buy_pct(prices, window=20)
    assert panel.shape == prices.shape
    assert panel.notna().any().any()


def test_target_price_upside(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rf = pd.DataFrame({
        "code": ["000001", "000002"],
        "announce_date": [DATES[5], DATES[6]],
        "rating_change": ["维持", "调高"],
        "is_first": ["不是首次评级"] * 2,
        "rating": ["买入", "增持"],
        "target_high": [30.0, 25.0],
        "target_low": [20.0, 18.0],
        "institute": ["A", "B"],
    })
    rf.to_parquet(tmp_path / "rank_forecast.parquet")
    panel = factor_target_price_upside(prices, prices_raw=prices, hold=10)
    assert panel.shape == prices.shape


def test_new_ashare_names_registered():
    margin_dense = (
        "融资买入占成交额_5d",
        "融资净买入_5d",
        "融券卖出规避_5d",
        "融资余额流通市值比",
    )
    for n in margin_dense:
        assert n in ASHARE_DENSE_FACTOR_NAMES
        assert n not in ASHARE_SPARSE_FACTOR_NAMES
        assert n not in SPARSE_FACTOR_NAMES
    for n in (
        "龙虎榜净买占比_20d",
        "目标价上行空间",
        "解禁流动性压力_60d",
        "大宗机构接盘_20d",
        "评级下调规避_20d",
        "回购完成进度_60d",
        "研报覆盖热度_20d",
        "龙虎榜涨幅上榜_20d",
        "龙虎榜换手上榜_20d",
        "龙虎榜跌幅上榜规避_20d",
        "解禁定增压力_60d",
        "解禁激励压力_60d",
        "转债转股稀释_60d",
        "激励行权稀释_60d",
        "限售上市供给_60d",
        "研报EPS斜率",
        "研报EPS分歧度",
        "大宗卖方机构抛压_20d",
        "大宗折溢价波动_20d",
    ):
        assert n in ASHARE_SPARSE_FACTOR_NAMES
        assert n in SPARSE_FACTOR_NAMES
    prices = _prices()
    names = get_factor_names(prices, financial=pd.DataFrame())
    assert "THS_资金净流入强度_即时" not in names
    assert "THS_资金净流入强度_5d" not in names
    assert "THS_资金净流入强度_20d" not in names
    assert "大宗机构接盘_20d" in names
    assert "融资净买入_5d" in names
    assert "融资买入占成交额_5d" in names
    assert "融券卖出规避_5d" in names
    assert "融资余额流通市值比" in names
    assert "研报EPS斜率" in names
    # 日频 PE/PB / ST 事件明确不注册
    assert "日频PE_TTM" not in names
    assert "日频PB" not in names
    assert "ST戴帽" not in names
    assert "ST摘帽" not in names


def test_lhb_reason_bucket():
    assert _lhb_reason_bucket("日涨幅偏离值达到7%的前5只证券") == "up"
    assert _lhb_reason_bucket("日跌幅偏离值达到7%的前5只证券") == "down"
    assert _lhb_reason_bucket("日换手率达到20%的前5只证券") == "turnover"
    assert _lhb_reason_bucket("退市整理期") == "other"


def test_margin_net_buy_and_short_avoid(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rows = []
    for i, code in enumerate(CODES):
        for j, dt in enumerate(DATES[5:12]):
            rows.append({
                "date": dt,
                "code": code,
                "margin_balance": 1e9 * (1 + 0.5 * i + 0.05 * j),
                "margin_buy_amount": 1e7 * (1 + 0.5 * i + 0.1 * j),
                "margin_repay_amount": 2e6 * (1 + 0.2 * i),
                "short_sell_volume": 1e5 * (1 + 0.3 * i + 0.05 * j),
                "short_repay_volume": 1e4,
                "short_balance_amount": 1e6,
            })
    md = pd.DataFrame(rows)
    md.to_parquet(tmp_path / "margin_detail.parquet")
    circ = pd.DataFrame({c: 1e10 for c in CODES}, index=DATES)
    circ.to_parquet(tmp_path / "circ_mv.parquet")
    amt = pd.DataFrame({c: 1e8 for c in CODES}, index=DATES)
    amt.to_parquet(tmp_path / "amount.parquet")
    net = factor_margin_net_buy(prices, amount=amt, window=3)
    short = factor_short_sell_avoid(prices, amount=amt, window=3)
    ratio = factor_margin_balance_to_float(prices)
    assert net.shape == prices.shape
    assert short.shape == prices.shape
    assert ratio.shape == prices.shape
    assert net.notna().any().any()
    assert short.notna().any().any()
    assert ratio.notna().any().any()


def test_lhb_reason_factors(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    lhb = pd.DataFrame({
        "code": ["000001", "000002", "000001", "000003"],
        "lhb_date": [DATES[4], DATES[4], DATES[7], DATES[8]],
        "reason": [
            "日涨幅偏离值达到7%的前5只证券",
            "日跌幅偏离值达到7%的前5只证券",
            "日换手率达到20%的前5只证券",
            "连续三个交易日内，涨幅偏离值累计达到20%的证券",
        ],
        "net_buy": [1e7, -1e6, 2e6, 3e6],
    })
    lhb.to_parquet(tmp_path / "lhb_detail.parquet")
    up = factor_lhb_reason_up(prices, window=20)
    turn = factor_lhb_reason_turnover(prices, window=20)
    down = factor_lhb_reason_down_avoid(prices, window=20)
    assert up.shape == prices.shape
    assert turn.notna().any().any()
    assert down.notna().any().any()


def test_lockup_type_and_share_change(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    lr = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "release_date": [DATES[20], DATES[25], DATES[22]],
        "release_type": ["定向增发机构配售股份", "股权激励限售股份", "首发原股东限售股份"],
        "actual_release_value": [1e8, 5e7, 2e8],
    })
    lr.to_parquet(tmp_path / "lockup_release.parquet")
    circ = pd.DataFrame({c: 1e10 for c in CODES}, index=DATES)
    circ.to_parquet(tmp_path / "circ_mv.parquet")
    place = factor_lockup_placement_pressure(prices, horizon=30)
    ince = factor_lockup_incentive_pressure(prices, horizon=30)
    assert place.shape == prices.shape
    assert ince.notna().any().any()

    sc = pd.DataFrame({
        "code": ["000001", "000002", "000003", "000001"],
        "announce_date": [DATES[5], DATES[6], DATES[8], DATES[10]],
        "change_date": [DATES[6], DATES[7], DATES[9], DATES[11]],
        "total_shares": [1e9, 2e9, 3e9, 1.05e9],
        "circ_shares": [5e8, 1e9, 1.5e9, 5.2e8],
        "change_reason": ["可转债转股", "期权行权", "限售股份上市", "定期报告"],
    })
    sc.to_parquet(tmp_path / "share_change.parquet")
    cb = factor_cb_conversion_dilution(prices, window=20)
    ex = factor_restricted_listing_supply(prices, window=20)
    assert cb.shape == prices.shape
    assert ex.notna().any().any()


def test_research_slope_dispersion(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    rr = pd.DataFrame({
        "code": ["000001", "000001", "000002", "000002"],
        "announce_date": [DATES[5], DATES[5], DATES[8], DATES[8]],
        "eps_forecast": [1.0, 1.2, 0.8, 1.5],
        "eps_2026": [1.0, 1.2, 0.8, 1.5],
        "eps_2027": [1.3, 1.5, 0.9, 1.8],
        "eps_2028": [1.6, 1.8, 1.0, 2.0],
        "institute": ["A", "B", "A", "B"],
        "title": ["t1", "t2", "t3", "t4"],
        "rating": ["买入"] * 4,
    })
    rr.to_parquet(tmp_path / "research_report.parquet")
    slope = factor_research_eps_slope(prices, hold=10)
    disp = factor_research_eps_dispersion(prices, hold=10)
    assert slope.shape == prices.shape
    assert disp.shape == prices.shape
    assert slope.loc[DATES[5]:, "000001"].notna().any()
    assert disp.loc[DATES[5]:, "000001"].notna().any()


def test_block_seller_and_discount_vol(tmp_path, monkeypatch):
    monkeypatch.setattr("factors.factor_ashare.RAW_DIR", tmp_path)
    prices = _prices()
    bt = pd.DataFrame({
        "code": ["000001", "000002", "000001", "000003"],
        "trade_date": [DATES[5], DATES[5], DATES[6], DATES[7]],
        "discount_rate": [-0.1, -0.05, 0.02, -0.08],
        "amount": [1e7, 2e7, 1.5e7, 3e7],
        "buyer_branch": ["某某营业部", "机构专用", "游资", "机构专用"],
        "seller_branch": ["机构专用", "某某营业部", "机构专用", "游资席位"],
    })
    bt.to_parquet(tmp_path / "block_trade.parquet")
    circ = pd.DataFrame({c: 1e10 for c in CODES}, index=DATES)
    circ.to_parquet(tmp_path / "circ_mv.parquet")
    sell = factor_block_seller_inst_pressure(prices, window=10)
    vol = factor_block_discount_vol(prices, window=10)
    assert sell.shape == prices.shape
    assert vol.shape == prices.shape
    assert sell.notna().any().any()
