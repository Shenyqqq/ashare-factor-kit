"""
tests/test_risk_metrics.py — 单测 backtest/risk_metrics.compute_risk_metrics

用合成净值（已知 Sharpe / 回撤 / 胜率）验证：
  1. 完美单调上升 nav → 波动=0, Sharpe=NaN（除零保护）, 最大回撤=0, Calmar=NaN
  2. 已知回撤序列 → max_dd 精确匹配
  3. 已知波动序列 → 年化波动 / Sharpe 量级合理
  4. IR vs benchmark 计算正确
  5. 全 NaN track 不崩溃
  6. _periods_per_year freq 映射正确
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.risk_metrics import (
    _periods_per_year,
    compute_risk_metrics,
    format_risk_metrics_table,
    export_risk_metrics,
    period_returns_from_nav,
)


def _nav_from_rets(rets: dict[str, list[float]], start="2023-01-31", freq="ME") -> pd.DataFrame:
    """合成 nav：每列给一串 period 收益，cumprod 成累计净值，首行=1。"""
    dates = pd.date_range(start, periods=max(len(r) for r in rets.values()), freq=freq)
    df = {}
    for track, r in rets.items():
        df[track] = (1 + pd.Series(r)).cumprod().values
    return pd.DataFrame(df, index=dates)


# ── _periods_per_year ────────────────────────────────────────────────────────

def test_periods_per_year_mapping():
    assert _periods_per_year("W-FRI") == 52
    assert _periods_per_year("ME") == 12
    assert _periods_per_year("2W-FRI") == 26
    assert _periods_per_year("3D") == 84
    assert _periods_per_year("5D") == 50
    assert _periods_per_year("20D") == 12   # 月频（按交易日，dict 显式映射）
    assert _periods_per_year("10D") == 25
    assert _periods_per_year("60D") == 4
    assert _periods_per_year("unknown") == 12  # default
    assert _periods_per_year(None) == 12


# ── 完美单调上升：波动=0，Sharpe=NaN，回撤=0 ────────────────────────────────

def test_perfect_uptrend_no_volatility():
    # 每期 +1%，nav 单调上升，无波动
    rets = {"Q5": [0.01] * 12, "benchmark": [0.005] * 12}
    nav = _nav_from_rets(rets)
    m = compute_risk_metrics(nav, rebalance_freq="ME")

    # 年化收益 = (1.01^12)^(12/12) - 1 = 1.01^12 - 1 ≈ 0.1268
    assert abs(m.loc["Q5", "年化收益"] - (1.01 ** 12 - 1)) < 1e-9
    # 波动=0
    assert m.loc["Q5", "年化波动"] == 0.0
    # Sharpe = 0/0 → NaN（除零保护）
    assert np.isnan(m.loc["Q5", "Sharpe"])
    # Sortino 同样 NaN（无下行波动）
    assert np.isnan(m.loc["Q5", "Sortino"])
    # 最大回撤 = 0（单调上升）
    assert m.loc["Q5", "最大回撤"] == 0.0
    # Calmar = ann_return / 0 → NaN
    assert np.isnan(m.loc["Q5", "Calmar"])
    # 胜率 = 100%
    assert m.loc["Q5", "胜率"] == 1.0


# ── 已知回撤序列 ────────────────────────────────────────────────────────────

def test_known_drawdown():
    # nav: 1.0 → 1.2 → 1.0 → 1.1（最大回撤发生在第3期：1.0/1.2 - 1 = -16.67%）
    dates = pd.date_range("2023-01-31", periods=4, freq="ME")
    nav = pd.DataFrame({"Q5": [1.0, 1.2, 1.0, 1.1]}, index=dates)
    m = compute_risk_metrics(nav, rebalance_freq="ME")

    max_dd = m.loc["Q5", "最大回撤"]
    assert abs(max_dd - (-1 / 6)) < 1e-6, f"max_dd={max_dd}, want -0.16667"


# ── 已知波动序列：Sharpe 量级合理 ───────────────────────────────────────────

def test_sharpe_magnitude_reasonable():
    rng = np.random.default_rng(42)
    n = 60  # 5 年月频
    # Q5: 月均 1.5%, 月波动 4%；benchmark: 月均 0.5%, 月波动 4%
    rets = {
        "Q5": rng.normal(0.015, 0.04, n).tolist(),
        "benchmark": rng.normal(0.005, 0.04, n).tolist(),
    }
    nav = _nav_from_rets(rets)
    m = compute_risk_metrics(nav, rebalance_freq="ME")

    # 年化波动 ≈ 0.04 * sqrt(12) ≈ 0.1386
    assert 0.10 < m.loc["Q5", "年化波动"] < 0.18
    # Sharpe 量级在合理区间（-3, 3）
    assert -3.0 < m.loc["Q5", "Sharpe"] < 3.0
    # Sortino >= Sharpe（下行波动 <= 总波动）
    if not np.isnan(m.loc["Q5", "Sortino"]):
        assert m.loc["Q5", "Sortino"] >= m.loc["Q5", "Sharpe"] - 1e-6
    # 最大回撤在 [-100%, 0]
    assert -1.0 <= m.loc["Q5", "最大回撤"] <= 0.0


# ── IR vs benchmark ─────────────────────────────────────────────────────────

def test_information_ratio_vs_benchmark():
    rng = np.random.default_rng(1)
    n = 60
    bench = rng.normal(0.005, 0.04, n)
    # Q5 系统性超额 0.5%/月，波动相近
    strat = bench + 0.005 + rng.normal(0, 0.01, n)
    rets = {"Q5": strat.tolist(), "benchmark": bench.tolist()}
    nav = _nav_from_rets(rets)
    m = compute_risk_metrics(nav, rebalance_freq="ME")

    # Q5 的 IR 应为正（系统超额）
    assert m.loc["Q5", "IR"] > 0
    # benchmark 自身 IR 应为 NaN
    assert np.isnan(m.loc["benchmark", "IR"])


# ── 全 NaN track 不崩溃 ───────────────────────────────────────────────────

def test_all_nan_track_does_not_crash():
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")
    nav = pd.DataFrame({
        "Q5": [1.0] * 12,
        "Q1": [np.nan] * 12,
    }, index=dates)
    m = compute_risk_metrics(nav, rebalance_freq="ME")
    # Q1 全 NaN → 所有指标 NaN
    assert np.isnan(m.loc["Q1", "年化收益"])
    assert np.isnan(m.loc["Q1", "Sharpe"])
    # Q5 常数列 → 波动=0, Sharpe=NaN（不崩溃）
    assert m.loc["Q5", "年化波动"] == 0.0
    assert np.isnan(m.loc["Q5", "Sharpe"])


# ── 空 nav 不崩溃 ───────────────────────────────────────────────────────

def test_empty_nav():
    m = compute_risk_metrics(pd.DataFrame(), rebalance_freq="ME")
    assert m.empty
    assert list(m.columns) == ["年化收益", "年化波动", "Sharpe", "Sortino",
                                "最大回撤", "Calmar", "IR", "胜率", "超额胜率"]


# ── 格式化表可读 ───────────────────────────────────────────────────────────

def test_format_table_returns_string():
    rng = np.random.default_rng(0)
    n = 24
    rets = {
        "Q1": rng.normal(0.003, 0.04, n).tolist(),
        "Q5": rng.normal(0.012, 0.035, n).tolist(),
        "Top100": rng.normal(0.010, 0.035, n).tolist(),
        "benchmark": rng.normal(0.006, 0.038, n).tolist(),
    }
    nav = _nav_from_rets(rets)
    m = compute_risk_metrics(nav, rebalance_freq="ME")
    txt = format_risk_metrics_table(m)
    assert isinstance(txt, str)
    assert "Sharpe" in txt
    assert "最大回撤" in txt
    assert "Q5" in txt
    # 至少 4 行（header + sep + 4 tracks）
    assert txt.count("\n") >= 5
    assert "最大回撤" in m.columns


def test_export_csv_has_max_drawdown(tmp_path):
    """CSV 英文字段 max_drawdown 对应中文「最大回撤」。"""
    rets = {"Q5": [0.02, -0.05, 0.01, 0.03] * 3,
            "benchmark": [0.01, -0.02, 0.01, 0.01] * 3}
    nav = _nav_from_rets(rets)
    out = tmp_path / "risk_mdd.csv"
    m = export_risk_metrics(nav, save_path=str(out), rebalance_freq="ME")
    assert "最大回撤" in m.columns
    df = pd.read_csv(out)
    assert "max_drawdown" in df.columns
    assert (df["max_drawdown"] <= 0).all()


def test_format_win_rate_is_percent_not_fraction():
    """胜率存 0–1，展示应 ×100 成约 XX.X%（不是 0.X%）。"""
    rets = {"Q5": [0.01, -0.005, 0.02, 0.01, -0.01, 0.015] * 2}
    nav = _nav_from_rets(rets)
    m = compute_risk_metrics(nav, rebalance_freq="ME")
    # 原始指标仍是小数
    assert 0.0 <= m.loc["Q5", "胜率"] <= 1.0
    txt = format_risk_metrics_table(m)
    # 展示行应含几十百分点，例如 "  66.7%"，不应是 "   0.7%"
    import re
    m_win = re.search(r"Q5\s+.*?(\d+\.\d)%\s*$", txt, re.M)
    # 更稳：整表中不应出现 "0.X%" 作为胜率列（胜率≈几十）
    assert m.loc["Q5", "胜率"] * 100 >= 10
    # 格式化后数字应接近 win_rate*100
    wr_pct = m.loc["Q5", "胜率"] * 100
    assert f"{wr_pct:.1f}%" in txt, f"expected {wr_pct:.1f}% in:\n{txt}"


# ── export CSV 写盘 ──────────────────────────────────────────────────────────

def test_export_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 12
    rets = {"Q5": rng.normal(0.01, 0.04, n).tolist(),
            "benchmark": rng.normal(0.005, 0.04, n).tolist()}
    nav = _nav_from_rets(rets)
    out = tmp_path / "risk.csv"
    m = export_risk_metrics(nav, save_path=str(out), rebalance_freq="ME")
    assert out.exists()
    # CSV 应有英文表头
    df = pd.read_csv(out)
    assert "track" in df.columns
    assert "sharpe" in df.columns
    assert "ann_return" in df.columns
    assert "win_rate" in df.columns
    assert "beat_bm_rate" in df.columns
    # 返回的 metrics 与 compute 一致
    assert "Sharpe" in m.columns


def test_beat_bm_rate_distinct_from_win_rate():
    """超额胜率 = 期收益>benchmark，与「收益>0」胜率分列；fillna(0) 含首期。"""
    # W-FRI 调仓：Top100 每期都赢基准，但第 3 期绝对收益为负
    rets = {
        "Q5": [0.02, 0.01, -0.01, 0.03],
        "Top100": [0.03, 0.02, -0.01, 0.04],
        "benchmark": [0.01, 0.00, -0.02, 0.02],
    }
    nav = _nav_from_rets(rets, start="2023-01-06", freq="W-FRI")
    # 还原期收益应含首期、与手工 fillna(0) 一致
    recovered = period_returns_from_nav(nav["Top100"])
    assert list(np.round(recovered.values, 10)) == [0.03, 0.02, -0.01, 0.04]

    m = compute_risk_metrics(nav, rebalance_freq="W-FRI")
    # Top100 期胜率 3/4；超额 4/4（含负收益那期仍打赢基准）
    assert m.loc["Top100", "胜率"] == pytest.approx(0.75)
    assert m.loc["Top100", "超额胜率"] == pytest.approx(1.0)
    assert np.isnan(m.loc["benchmark", "超额胜率"])
    # CSV 英文字段不与 win_rate 混名
    txt = format_risk_metrics_table(m)
    assert "超额胜率" in txt
    assert "Top100" in txt


def test_print_quantile_summary_highlights_top100_win_rates():
    """摘要表打印 Top100 期胜率与超额胜率，不只藏在 CSV。"""
    from unittest.mock import patch

    from backtest.quantile import QuantileResult
    from backtest.report import print_quantile_summary

    rets = {
        "Q1": [-0.02, 0.01, -0.01, 0.00],
        "Q5": [0.02, 0.01, 0.00, 0.03],
        "Top100": [0.03, 0.02, -0.01, 0.04],
        "benchmark": [0.01, 0.00, -0.02, 0.02],
    }
    nav = _nav_from_rets(rets, start="2023-01-06", freq="W-FRI")
    ls = (1 + (nav["Q5"] / nav["Q5"].shift(1).fillna(1.0) - 1)
          - (nav["Q1"] / nav["Q1"].shift(1).fillna(1.0) - 1)).cumprod()
    result = QuantileResult(
        nav=nav,
        annual_returns=pd.DataFrame(
            {"Q1": [0.1], "Q5": [0.2], "Top100": [0.25]}, index=[2023],
        ),
        ic_monotonicity=0.9,
        long_short_nav=ls,
        turnover=pd.DataFrame(),
        top_holdings={},
    )
    captured: list[str] = []
    with patch("backtest.report.logger") as log:
        log.info.side_effect = lambda msg: captured.append(str(msg))
        print_quantile_summary(result, rebalance_freq="W-FRI")
    text = "\n".join(captured)
    assert "期胜率" in text
    assert "超额胜率" in text
    assert "★ Top100" in text
    assert "实操参考" in text
    # 75.0% 期胜率、100.0% 超额
    assert "75.0%" in text
    assert "100.0%" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
