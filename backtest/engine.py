"""
backtest/engine.py  —  纯 pandas 回测引擎，无第三方框架依赖

核心逻辑:
    每个调仓日 → 用因子得分选 top N 股票 → 等权买入 → 计算持仓市值
    处理: 手续费、印花税、涨跌停无法成交、退市

用法:
    from backtest.engine import run_backtest
    result = run_backtest(prices, factor_scores)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    INITIAL_CAPITAL, REBALANCE_FREQ, N_STOCKS,
    COMMISSION_RATE, SLIPPAGE_RATE, STAMP_DUTY,
    MIN_PRICE, BACKTEST_START, BACKTEST_END,
)


# ── 结果对象 ──────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    portfolio_value: pd.Series       # 每日组合总市值
    holdings_history: list           # 每次调仓的持仓快照
    turnover: pd.Series              # 每次调仓的换手率
    benchmark: pd.Series = None      # 基准（如沪深300）

    @property
    def returns(self) -> pd.Series:
        return self.portfolio_value.pct_change().dropna()

    @property
    def total_return(self) -> float:
        return self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1

    @property
    def annual_return(self) -> float:
        n_years = len(self.portfolio_value) / 252
        return (1 + self.total_return) ** (1 / n_years) - 1

    @property
    def sharpe(self) -> float:
        r = self.returns
        if r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        cum = self.portfolio_value / self.portfolio_value.cummax()
        return (cum - 1).min()

    @property
    def calmar(self) -> float:
        dd = abs(self.max_drawdown)
        return self.annual_return / dd if dd > 0 else np.nan

    def summary(self) -> pd.Series:
        return pd.Series({
            "总收益率":   f"{self.total_return:.1%}",
            "年化收益率": f"{self.annual_return:.1%}",
            "夏普比率":   f"{self.sharpe:.2f}",
            "最大回撤":   f"{self.max_drawdown:.1%}",
            "卡玛比率":   f"{self.calmar:.2f}",
            "平均换手率": f"{self.turnover.mean():.1%}",
        })


# ── 核心回测函数 ──────────────────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,           # index=日期, columns=股票代码
    factor_scores: pd.DataFrame,    # 同上，值=因子得分（越高越好）
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    initial_capital: float = INITIAL_CAPITAL,
    n_stocks: int = N_STOCKS,
    rebalance_freq: str = REBALANCE_FREQ,
) -> BacktestResult:

    # ── 时间范围裁剪 ──────────────────────────────────────────────────────────
    prices = prices.loc[start:end]
    factor_scores = factor_scores.loc[start:end]
    all_dates = prices.index

    # ── 调仓日序列 ────────────────────────────────────────────────────────────
    rebalance_dates = pd.date_range(start, end, freq=rebalance_freq)
    rebalance_dates = [d for d in rebalance_dates if d in all_dates]

    # ── 状态变量 ──────────────────────────────────────────────────────────────
    cash = initial_capital
    holdings: dict[str, float] = {}     # {code: shares}
    portfolio_values = {}
    holdings_history = []
    turnovers = {}

    prev_portfolio_set: set = set()

    # ── 主循环：按交易日推进 ──────────────────────────────────────────────────
    for date in all_dates:
        current_prices = prices.loc[date].dropna()

        # 1. 过滤停牌/退市（当日无价格）
        tradable = set(current_prices.index)
        active_holdings = {c: s for c, s in holdings.items() if c in tradable}

        # 2. 计算当前组合市值
        stock_value = sum(
            active_holdings.get(c, 0) * current_prices.get(c, 0)
            for c in active_holdings
        )
        total_value = cash + stock_value

        # 3. 调仓日才执行交易
        if date in rebalance_dates and date in factor_scores.index:
            scores_today = factor_scores.loc[date].dropna()

            # 只考虑可交易、价格满足最低要求的股票
            valid = scores_today[
                scores_today.index.isin(tradable) &
                (current_prices.reindex(scores_today.index).fillna(0) >= MIN_PRICE)
            ]

            new_portfolio = set(valid.nlargest(n_stocks).index.tolist())

            # ── 卖出不在新组合里的股票 ────────────────────────────────────────
            to_sell = prev_portfolio_set - new_portfolio
            for code in to_sell:
                if code in holdings and code in current_prices.index:
                    price = current_prices[code] * (1 - SLIPPAGE_RATE)
                    proceeds = holdings[code] * price
                    cost = proceeds * (COMMISSION_RATE + STAMP_DUTY)
                    cash += proceeds - cost
                    del holdings[code]

            # ── 买入新进组合的股票 ────────────────────────────────────────────
            to_buy = new_portfolio - prev_portfolio_set
            # 先重新计算总值（卖出后 cash 变了）
            stock_value = sum(
                holdings.get(c, 0) * current_prices.get(c, 0)
                for c in holdings
            )
            total_value = cash + stock_value
            target_per_stock = total_value / n_stocks

            for code in new_portfolio:
                price = current_prices.get(code)
                if price is None or price <= 0:
                    continue
                price_with_slip = price * (1 + SLIPPAGE_RATE)
                current_pos_value = holdings.get(code, 0) * price
                diff = target_per_stock - current_pos_value
                if diff > 100:   # 最小买入阈值，避免碎股
                    shares = diff / price_with_slip
                    cost = diff * COMMISSION_RATE
                    if cash >= diff + cost:
                        holdings[code] = holdings.get(code, 0) + shares
                        cash -= diff + cost

            # ── 换手率计算 ────────────────────────────────────────────────────
            turnover = len(to_sell | to_buy) / max(len(prev_portfolio_set | new_portfolio), 1)
            turnovers[date] = turnover

            holdings_history.append({
                "date": date,
                "holdings": list(new_portfolio),
                "n_buy": len(to_buy),
                "n_sell": len(to_sell),
            })

            prev_portfolio_set = new_portfolio

        # 4. 记录每日总市值
        stock_value = sum(
            holdings.get(c, 0) * current_prices.get(c, 0)
            for c in holdings if c in current_prices.index
        )
        portfolio_values[date] = cash + stock_value

    result = BacktestResult(
        portfolio_value=pd.Series(portfolio_values),
        holdings_history=holdings_history,
        turnover=pd.Series(turnovers),
    )

    logger.info("=== 回测结果 ===")
    print(result.summary().to_string())
    return result


# ── 可视化 ────────────────────────────────────────────────────────────────────

def plot_result(result: BacktestResult, benchmark: pd.Series = None, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Backtest Result", fontsize=14)

    # NAV curve
    ax = axes[0]
    nav = result.portfolio_value / result.portfolio_value.iloc[0]
    ax.plot(nav.index, nav.values, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0]
        ax.plot(bench_norm.index, bench_norm.values, label="Benchmark", linewidth=1, alpha=0.7)
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    dd = (result.portfolio_value / result.portfolio_value.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)

    # Monthly returns bar chart
    ax = axes[2]
    monthly = result.portfolio_value.resample("ME").last().pct_change().dropna() * 100
    colors = ["green" if v >= 0 else "red" for v in monthly.values]
    ax.bar(monthly.index, monthly.values, color=colors, width=20)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Monthly Return (%)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表保存至 {save_path}")
    else:
        plt.show()
