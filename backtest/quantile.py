"""
backtest/quantile.py — 分组回测（Q1-Q5）模块化引擎（原 quantile_v2，已合并）

验证因子得分的单调性：Q5（高分）是否持续跑赢Q1（低分）？

时间线（统一命名）：
  signal_date       — 周期末收盘后观察因子得分
  execution_date    — T+1 开盘买入（传入 open_prices 时）
  next_signal_date  — 下一调仓信号
  hold window       — [execution_date, next_signal_date]

执行模型（贴近实盘）：
  • 真实 buy-and-hold：每股 NAV 从 exec-open 起，组合 = Σ w×stock_NAV
  • 调仓成本：执行日 NAV × (1 − cost)
  • 一字涨停：跳过并按排名向后回填；一字跌停：无法卖出，继续持有
  • 停牌：当日收益记 0（NAV 持平）
  • ST / 上市天数 / 涨跌停 masks 全部走 TradeRules

历史：原 v1 单体 quantile.py 已删除；本文件保留 v2 模块化实现，
子模块 execution / portfolio / return_engine / turnover / benchmark / report 不变。
公共 API（run_quantile_backtest / QuantileResult）保持与 run.py 直接对接。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.benchmark import index_period_return
from backtest.execution import (
    BacktestConfig,
    TradeRules,
    hold_dates_between,
    infer_st_codes,
    resolve_execution_date,
    total_cost_fraction,
)
from backtest.portfolio import (
    PortfolioState,
    apply_turnover_control,
    assign_quantile_groups,
    rebalance_holdings,
    select_top_n,
)
from backtest.optimize import normalize_method, optimize_weights
from backtest.return_engine import simulate_period
from backtest.turnover import TurnoverRecord, compute_turnover, make_turnover_record
from utils.rebalance_dates import get_rebalance_dates


@dataclass
class QuantileResult:
    """分组回测结果；turnover_detail 用于审计 CSV 导出。"""

    nav: pd.DataFrame
    annual_returns: pd.DataFrame
    ic_monotonicity: float
    long_short_nav: pd.Series
    turnover: pd.DataFrame
    top_holdings: dict = None
    turnover_detail: pd.DataFrame | None = field(default=None)
    position_exposure: pd.Series | None = field(default=None)
    position_regime: pd.DataFrame | None = field(default=None)


def run_quantile_backtest(
    prices: pd.DataFrame,
    factor_scores: pd.DataFrame,
    n_quantiles: int = 5,
    rebalance_freq: str = "ME",
    start: str = None,
    end: str = None,
    cost_bps: float = 3.5,
    min_stocks: int = 5,
    open_prices: pd.DataFrame = None,
    masks: dict = None,
    indices: dict = None,
    top_n: int = 100,
    stock_names: pd.Series | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    volume: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    st_schedule: pd.DataFrame | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    eligible_mask: pd.DataFrame | None = None,
    position_regime: pd.DataFrame | None = None,
    position_exposure: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
) -> QuantileResult:
    """
    Run Q1-Q5 + Top-N quantile backtest.

    Parameters
    ----------
    prices, factor_scores : 后复权收盘价 / 模型预测得分
    open_prices : 后复权开盘价（可选）。传入时启用 T+1 开盘执行 + 一字涨停剔除
    masks : clean_ohlcv() 产生的 mask dict（limit_up_open / limit_down_open 等）
    cost_bps : 单边成本 bp 兜底（仅当 commission/stamp/slippage 全 0 时使用）
    indices : {"沪深300": pd.Series, "创业板指": pd.Series}
    top_n : 每期保存得分最高的 top N 标的（默认 100，与 settings.N_STOCKS 一致）
    stock_names, listing_dates, volume : 可选 universe 过滤
    config : BacktestConfig 覆盖（成本率、ST、上市天数等）
    st_schedule : M4 时间序列 ST 状态（wide bool: index=date, columns=stock）
    delist_dates : M4 code → delist_date 字典，回测时按日期判断已退市
    position_regime : 可选，``compute_position_regime`` 输出（含 target_exposure）。
        与 ``position_exposure`` 皆为 None 时不缩放（与旧实验 bit-identical）。
    position_exposure : 可选，调仓日 → 敞口标量；优先于从 regime 表按日查找。
    returns : 可选日频收益面板（优先 ``clean_ret``），供 ``mv``/``invvol``/``rp``
        估计 Σ/σ；None 时由 ``prices.pct_change()`` 推导。
    """
    if factor_scores.empty or not isinstance(factor_scores.index, pd.DatetimeIndex):
        raise ValueError(
            f"factor_scores 为空或 index 不是 DatetimeIndex（dtype={factor_scores.index.dtype}）"
        )

    cfg = config or BacktestConfig(cost_bps=cost_bps)
    cfg.cost_bps = cost_bps
    cfg.use_open_execution = open_prices is not None

    # scores 只决定信号日；T+1 / 持有窗 / 指数一律用完整日频 prices 日历。
    # 禁止 prices ∩ scores 裁薄日历（周频 scores 会把持有窗塌成 1 日）。
    common_cols = prices.columns.intersection(factor_scores.columns)
    prices_a = prices.reindex(columns=common_cols)
    scores_a = factor_scores.reindex(columns=common_cols)
    signal_dates_idx = scores_a.index.intersection(prices_a.index)
    if start:
        t0 = pd.Timestamp(start)
        prices_a = prices_a.loc[prices_a.index >= t0]
        signal_dates_idx = signal_dates_idx[signal_dates_idx >= t0]
    if end:
        t1 = pd.Timestamp(end)
        # 价格日历可略超出 end，以便末日信号的 T+1 执行日仍在索引内
        signal_dates_idx = signal_dates_idx[signal_dates_idx <= t1]
    scores_a = scores_a.loc[signal_dates_idx]
    open_a = (
        open_prices.reindex(index=prices_a.index, columns=prices_a.columns)
        if open_prices is not None else None
    )

    rules = TradeRules(
        masks=masks,
        st_codes=infer_st_codes(stock_names),
        st_schedule=st_schedule,
        listing_dates=listing_dates,
        delist_dates=delist_dates,
        volume=volume,
        config=cfg,
    )

    rebalance_dates = get_rebalance_dates(scores_a.index, rebalance_freq)
    if len(rebalance_dates) < 2:
        raise ValueError("调仓日期不足，请检查数据范围或调仓频率")

    q_labels = [f"Q{i + 1}" for i in range(n_quantiles)]
    top_label = f"Top{top_n}"
    track_labels = q_labels + [top_label, "benchmark"]

    group_returns: dict[str, list] = {k: [] for k in track_labels}
    turnover_rows: dict[str, list] = {k: [] for k in track_labels}
    all_turnover_records: list[TurnoverRecord] = []
    period_meta: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    top_holdings_log: dict = {}

    states: dict[str, PortfolioState] = {k: PortfolioState() for k in track_labels}

    # 仓位体制：仅当传入 exposure / regime 时启用；否则完全跳过（bit-identical）
    use_pos_regime = position_exposure is not None or position_regime is not None
    exposure_log: list[float] = []
    if use_pos_regime:
        from backtest.regime import apply_exposure, exposure_at

    # 组合优化：默认 ew → 不传入 weights，simulate_period 走旧等权路径
    opt_method = normalize_method(getattr(cfg, "portfolio_opt", "ew"))
    use_opt = opt_method != "ew"
    ret_panel: pd.DataFrame | None = None
    if use_opt and opt_method in ("mv", "invvol", "rp"):
        if returns is not None and not returns.empty:
            ret_panel = returns.reindex(index=prices_a.index, columns=prices_a.columns)
        else:
            ret_panel = prices_a.pct_change()

    for i in range(len(rebalance_dates) - 1):
        signal_date = rebalance_dates[i]
        next_signal_date = rebalance_dates[i + 1]

        use_open = cfg.use_open_execution
        execution_date = resolve_execution_date(signal_date, prices_a.index, use_open)
        if execution_date is None:
            continue

        hold_dates = hold_dates_between(prices_a.index, execution_date, next_signal_date)
        if len(hold_dates) == 0:
            continue

        # 本期敞口（benchmark 不缩放）
        period_exposure = 1.0
        if use_pos_regime:
            if position_exposure is not None and signal_date in position_exposure.index:
                val = position_exposure.loc[signal_date]
                if np.isfinite(val):
                    period_exposure = float(np.clip(val, 0.0, 1.0))
            elif position_regime is not None:
                period_exposure = exposure_at(position_regime, signal_date, default=1.0)

        scores_today = scores_a.loc[signal_date].dropna()
        # Drop execution-day unbuyable names before qcut (limit-up open, ST, …)
        cand = [s for s in scores_today.index if s in prices_a.columns]
        if cand:
            mask = rules.buyable_mask(cand, execution_date, prices_a)
            buyable = [s for s, m in zip(cand, mask) if m]
        else:
            buyable = []
        scores_for_groups = scores_today.loc[buyable] if buyable else scores_today.iloc[:0]
        group_map = assign_quantile_groups(scores_for_groups, n_quantiles, q_labels, min_stocks)
        if group_map is None:
            continue

        period_meta.append((signal_date, next_signal_date, execution_date))
        if use_pos_regime:
            exposure_log.append(period_exposure)
        eligible = [s for s in scores_today.index if s in prices_a.columns]
        # cap-band / 自定义 universe mask：仅保留 mask 为 True 的股票（PIT 安全，
        # mask 用 signal_date 当日流通市值构造）。mask=None 时不过滤（全市场）。
        if eligible_mask is not None and signal_date in eligible_mask.index:
            allowed = eligible_mask.loc[signal_date]
            allowed = allowed[allowed.fillna(False)].index
            eligible = [s for s in eligible if s in allowed]

        top_stocks = select_top_n(scores_today, top_n, rules, execution_date, prices_a)
        top_holdings_log[signal_date] = top_stocks

        targets: dict[str, set[str]] = {
            q: set(stocks) for q, stocks in group_map.items() if stocks
        }
        targets[top_label] = set(top_stocks)
        targets["benchmark"] = set(eligible)

        for label in track_labels:
            target = targets.get(label, set())
            if label == "benchmark":
                if len(eligible) < min_stocks:
                    group_returns[label].append(np.nan)
                    turnover_rows[label].append({"turnover": np.nan, "cost": 0.0})
                    continue
            elif len(target) < min_stocks:
                group_returns[label].append(np.nan)
                turnover_rows[label].append({"turnover": np.nan, "cost": 0.0})
                continue

            # ── Turnover 控制（任务3）──────────────────────────────────────────
            # 仅对非 benchmark track 生效（benchmark = 全样本等权，无主动选股）
            # 当 turnover_limit < 1.0 或 rank_change_threshold > 0 时，
            # 在 rebalance 前调整 target：保留排名仍高的上期持仓，限制换手。
            if (
                label != "benchmark"
                and (cfg.turnover_limit < 1.0 or cfg.rank_change_threshold > 0.0)
            ):
                prev_holdings = set(states[label].holdings)
                if prev_holdings:
                    target = apply_turnover_control(
                        prev_holdings,
                        target,
                        scores_today,
                        turnover_limit=cfg.turnover_limit,
                        rank_change_threshold=cfg.rank_change_threshold,
                    )

            prev = states[label]
            prev_set = set(prev.holdings)
            new_state, sold, bought = rebalance_holdings(
                prev, target, execution_date, rules, prices_a,
            )
            # Bug 2 fix: stocks that couldn't be sold on exec day (one-word
            # limit down) are retried daily for sale inside simulate_period.
            # Compute the stuck set (intended sells minus actually sold) and
            # pass it down so the engine can freeze their NAV at the first
            # sellable day instead of carrying them full-window at close/close.
            stuck = (prev_set - target) - sold
            turnover = compute_turnover(prev, new_state)
            cost = total_cost_fraction(turnover, cfg)
            rec = make_turnover_record(
                signal_date, execution_date, prev, new_state, sold, bought, cfg,
            )
            all_turnover_records.append((label, rec))
            turnover_rows[label].append({"turnover": turnover, "cost": cost})

            stocks = list(new_state.holdings)
            if not stocks:
                group_returns[label].append(np.nan)
                states[label] = new_state
                continue

            # 权重：benchmark 始终等权；非 ew 时在 invested book 上优化（和=1）
            # 仓位体制只缩放 period_return，不折进权重（避免双重缩放）
            period_weights = None
            if use_opt and label != "benchmark":
                period_weights = optimize_weights(
                    method=opt_method,
                    stocks=stocks,
                    scores=scores_today,
                    returns=ret_panel,
                    asof=signal_date,
                    lookback=int(getattr(cfg, "cov_lookback", 60) or 60),
                    risk_aversion=float(getattr(cfg, "risk_aversion", 1.0) or 1.0),
                    target_sum=1.0,
                    max_weight=getattr(cfg, "max_weight", None),
                )

            period_ret, _, sold_mid = simulate_period(
                prices_a, hold_dates, stocks, execution_date, open_a, rules, cost,
                stuck_stocks=stuck if stuck else None,
                weights=period_weights,
            )
            # 仓位体制：非 benchmark track 按 target_exposure 缩放（现金收益 0）
            if use_pos_regime and label != "benchmark" and period_exposure != 1.0:
                period_ret = apply_exposure(period_ret, period_exposure)
            group_returns[label].append(period_ret)
            # Drop mid-window-sold stuck names from carried holdings so next
            # period doesn't re-attempt an already-exited position.
            if sold_mid:
                new_holdings = set(new_state.holdings) - sold_mid
                new_state = PortfolioState(holdings=frozenset(new_holdings))
            states[label] = new_state

    if not period_meta:
        raise ValueError("无有效调仓周期")

    signal_dates = [m[0] for m in period_meta]
    period_rets = pd.DataFrame(group_returns, index=pd.DatetimeIndex(signal_dates))

    if indices:
        for idx_name, idx_price in indices.items():
            idx_rets = []
            for signal_date, next_signal_date, exec_d in period_meta:
                hdates = hold_dates_between(prices_a.index, exec_d, next_signal_date)
                idx_rets.append(index_period_return(idx_price, hdates, exec_d))
            period_rets[idx_name] = idx_rets

    nav = (1 + period_rets.fillna(0)).cumprod()

    ls_ret = period_rets[q_labels[-1]] - period_rets[q_labels[0]]
    long_short_nav = (1 + ls_ret.fillna(0)).cumprod()

    annual_rets = {}
    for year, grp in period_rets[q_labels + [top_label]].groupby(period_rets.index.year):
        annual_rets[year] = (1 + grp.fillna(0)).prod() - 1
    annual_returns = pd.DataFrame(annual_rets).T

    total_returns = nav[q_labels].iloc[-1]
    rank_corr = total_returns.rank().corr(
        pd.Series(range(1, n_quantiles + 1), index=q_labels, dtype=float)
    )

    turnover_df = pd.DataFrame(
        {label: [r["turnover"] for r in turnover_rows[label]] for label in track_labels},
        index=period_rets.index,
    )

    detail_rows = []
    for group_label, rec in all_turnover_records:
        detail_rows.append({
            "group": group_label,
            "signal_date": rec.signal_date,
            "execution_date": rec.execution_date,
            "sells": "|".join(rec.sells),
            "buys": "|".join(rec.buys),
            "turnover": rec.turnover,
            "cost": rec.cost,
        })
    turnover_detail = pd.DataFrame(detail_rows) if detail_rows else None

    pos_exp_series = None
    if use_pos_regime and exposure_log:
        pos_exp_series = pd.Series(
            exposure_log,
            index=pd.DatetimeIndex(signal_dates),
            name="target_exposure",
        )

    return QuantileResult(
        nav=nav,
        annual_returns=annual_returns,
        ic_monotonicity=rank_corr,
        long_short_nav=long_short_nav,
        turnover=turnover_df,
        top_holdings=top_holdings_log,
        turnover_detail=turnover_detail,
        position_exposure=pos_exp_series,
        position_regime=position_regime,
    )


def _smoke_test_2stock_math() -> None:
    """Verify buy-hold math: equal-weight 2 stocks, zero cost."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame(
        {"A": [10.0, 11.0, 11.55], "B": [20.0, 21.0, 21.0]},
        index=dates,
    )
    open_ = pd.DataFrame({"A": [10.0, 11.0, 11.55], "B": [20.0, 21.0, 21.0]}, index=dates)
    rules = TradeRules(config=BacktestConfig(use_open_execution=True))
    hold = dates
    ret, nav, _ = simulate_period(close, hold, ["A", "B"], dates[0], open_, rules, 0.0)
    expected = 0.5 * 1.155 + 0.5 * 1.05 - 1.0
    assert abs(ret - expected) < 1e-9, f"got {ret}, want {expected}"
    print(f"2-stock buy-hold OK: period_return={ret:.4%} (expected {expected:.4%})")


def _smoke_test_full_loop() -> None:
    """Minimal multi-period backtest on synthetic panel."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=120)
    codes = [f"{i:06d}" for i in range(1, 51)]
    close = pd.DataFrame(
        10.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(dates), len(codes))), axis=0),
        index=dates, columns=codes,
    )
    open_ = close.shift(1).bfill() * 0.999
    scores = pd.DataFrame(rng.normal(0, 1, close.shape), index=dates, columns=codes)
    result = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=10, min_stocks=3,
    )
    assert not result.nav.empty
    assert result.turnover_detail is not None
    print(f"Full loop OK: nav shape={result.nav.shape}, monotonicity={result.ic_monotonicity:.3f}")


if __name__ == "__main__":
    _smoke_test_2stock_math()
    _smoke_test_full_loop()
    print("quantile smoke test passed.")
