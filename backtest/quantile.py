"""
backtest/quantile.py  —  分组回测（Q1-Q5）

验证因子得分的单调性：Q5（高分）是否持续跑赢Q1（低分）？

执行模型（尽量贴近实盘）：
  - 信号日（月末收盘后）看因子得分 → 次日开盘执行
  - 执行价 = 次日开盘价（open_prices 传入时生效）
  - 一字涨停股（execution day 开盘即封死）剔除出买入列表
  - 一字跌停股（想卖卖不出）：当期继续持有，下期尝试卖出（简化：不强制剔出）
  - 持仓期收益用真实收盘价追踪（涨跌停日的 return 是真实发生的，不 mask）

注意：执行价和持仓收益的 mask 概念不同：
  - 因子计算 clean_ret：mask 涨跌停日（价格不反映真实供需）
  - 持仓追踪 daily_returns：用真实 return（你持股涨停了就是赚了）
  - 执行约束：一字板当天根本买不进去，从候选股中排除
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class QuantileResult:
    nav: pd.DataFrame          # 各组NAV，columns=[Q1,Q2,...,Q5,benchmark]
    annual_returns: pd.DataFrame   # 逐年收益，index=year, columns=[Q1,...,Q5]
    ic_monotonicity: float     # 组别rank vs 组别收益的相关性（越接近1越好）
    long_short_nav: pd.Series  # Q5/Q1 多空组合NAV
    turnover: pd.DataFrame     # 各组换手率


def _get_rebalance_dates(factor_scores: pd.DataFrame,
                         rebalance_freq: str) -> pd.DatetimeIndex:
    """按频率生成调仓日期"""
    return (
        pd.Series(index=factor_scores.index, dtype=float)
        .resample(rebalance_freq)
        .last()
        .dropna()
        .index
        .intersection(factor_scores.index)
    )


def _next_trading_day(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """返回 index 中严格晚于 date 的第一个交易日，不存在则返回 None"""
    later = index[index > date]
    return later[0] if len(later) > 0 else None


def run_quantile_backtest(
    prices: pd.DataFrame,
    factor_scores: pd.DataFrame,
    n_quantiles: int = 5,
    rebalance_freq: str = "ME",
    start: str = None,
    end: str = None,
    cost_bps: float = 3,
    min_stocks: int = 5,
    open_prices: pd.DataFrame = None,
    masks: dict = None,
) -> QuantileResult:
    """
    分组回测核心逻辑。

    prices:       后复权收盘价，用于持仓期收益追踪
    open_prices:  后复权开盘价（可选）。传入时：
                    - 执行价从 sig_date 收盘改为次日开盘
                    - 一字涨停股（open==close==high，开盘即封死）从买入列表剔除
    masks:        clean_ohlcv() 产生的 mask dict（可选）。
                  用于识别 limit_up_open（一字涨停）以过滤不可买股票。
    cost_bps:     单边成本 bp（佣金0.01%+印花税摊半0.025%≈0.035%，即3.5bp）
    """
    # ── 对齐时间范围 ──────────────────────────────────────────────────────────
    common_dates = prices.index.intersection(factor_scores.index)
    if start:
        common_dates = common_dates[common_dates >= start]
    if end:
        common_dates = common_dates[common_dates <= end]

    prices_aligned = prices.loc[common_dates]
    scores_aligned = factor_scores.loc[common_dates]

    use_open = (open_prices is not None)
    if use_open:
        open_aligned = open_prices.reindex(index=prices_aligned.index,
                                           columns=prices_aligned.columns)

    rebalance_dates = _get_rebalance_dates(scores_aligned, rebalance_freq)
    if len(rebalance_dates) < 2:
        raise ValueError("调仓日期不足，请检查数据范围或调仓频率")

    q_labels = [f"Q{i+1}" for i in range(n_quantiles)]
    group_returns: dict[str, list] = {q: [] for q in q_labels}
    group_returns["benchmark"] = []
    date_index = []

    # 持仓期收益用真实收盘价（不 mask，涨跌停日的 P&L 是真实发生的）
    daily_returns = prices_aligned.pct_change().fillna(0)
    cost_rate = cost_bps / 10000

    prev_holdings: dict[str, set] = {q: set() for q in q_labels}

    for i in range(len(rebalance_dates) - 1):
        sig_date = rebalance_dates[i]
        hold_end = rebalance_dates[i + 1]

        # 执行日：次日开盘（有 open_prices 时），否则同日收盘
        exec_date = _next_trading_day(sig_date, prices_aligned.index) if use_open else sig_date

        # 当日因子得分（过滤NaN）
        scores_today = scores_aligned.loc[sig_date].dropna()

        # ── 过滤不可买股票（一字涨停：开盘即封死，无法成交）──
        if use_open and exec_date is not None and masks is not None:
            lu_open = masks.get("limit_up_open")
            if lu_open is not None and exec_date in lu_open.index:
                # 一字涨停：当日 open==close==high，完全无法买入
                cant_buy = lu_open.loc[exec_date]
                cant_buy = cant_buy[cant_buy].index
                scores_today = scores_today.drop(
                    labels=[c for c in cant_buy if c in scores_today.index]
                )

        if len(scores_today) < n_quantiles * min_stocks:
            continue

        # 分组
        try:
            groups = pd.qcut(scores_today, n_quantiles,
                             labels=q_labels, duplicates="drop")
        except ValueError:
            continue

        # ── 持仓区间：从执行日（含）到下一执行日（不含）──
        # 用开盘价执行时：收益 = open[exec+1] / open[exec] * close_path - 1
        # 简化：持仓区间仍用收盘价追踪（开盘买入当天日内 return 忽略）
        hold_dates = prices_aligned.index[
            (prices_aligned.index > sig_date) &
            (prices_aligned.index <= hold_end)
        ]
        if len(hold_dates) == 0:
            continue

        date_index.append((sig_date, hold_end))
        period_returns_all = daily_returns.loc[hold_dates]

        # 若使用开盘价执行，首日 return 替换为 close[exec] / open[exec] - 1（当日日内）
        if use_open and exec_date is not None and exec_date in hold_dates:
            open_exec = open_aligned.loc[exec_date]
            close_exec = prices_aligned.loc[exec_date]
            intraday_ret = (close_exec / open_exec.replace(0, np.nan) - 1).fillna(0)
            period_returns_all = period_returns_all.copy()
            period_returns_all.loc[exec_date] = intraday_ret

        for q in q_labels:
            stocks = groups[groups == q].index.tolist()
            stocks_in_prices = [s for s in stocks if s in prices_aligned.columns]
            if len(stocks_in_prices) < min_stocks:
                group_returns[q].append(np.nan)
                continue

            prev = prev_holdings[q]
            curr = set(stocks_in_prices)
            turnover = (len(prev.symmetric_difference(curr)) / (2 * len(prev | curr))
                        if prev else 1.0)
            prev_holdings[q] = curr

            ret_series = period_returns_all[stocks_in_prices].mean(axis=1)
            cum_ret = (1 + ret_series).prod() - 1
            cum_ret -= turnover * cost_rate
            group_returns[q].append(cum_ret)

        # 全样本等权基准
        all_stocks = [s for s in scores_today.index if s in prices_aligned.columns]
        if all_stocks:
            ret_bm = period_returns_all[all_stocks].mean(axis=1)
            group_returns["benchmark"].append((1 + ret_bm).prod() - 1)
        else:
            group_returns["benchmark"].append(np.nan)

    if not date_index:
        raise ValueError("无有效调仓周期")

    # ── 构建NAV ──────────────────────────────────────────────────────────────
    period_rets = pd.DataFrame(group_returns,
                               index=pd.MultiIndex.from_tuples(date_index))
    period_rets.index = [t[0] for t in date_index]
    period_rets.index = pd.DatetimeIndex(period_rets.index)

    nav = (1 + period_rets.fillna(0)).cumprod()

    # ── 多空NAV：Q5 / Q1 ─────────────────────────────────────────────────────
    ls_ret = period_rets[q_labels[-1]] - period_rets[q_labels[0]]
    long_short_nav = (1 + ls_ret.fillna(0)).cumprod()

    # ── 逐年收益 ──────────────────────────────────────────────────────────────
    annual_rets = {}
    for year, grp in period_rets[q_labels].groupby(period_rets.index.year):
        # 年化：复利连乘
        annual_rets[year] = (
            (1 + grp.fillna(0)).prod() - 1
        )
    annual_returns = pd.DataFrame(annual_rets).T

    # ── 单调性检验：组别排名 vs 全周期收益 ───────────────────────────────────
    total_returns = nav[q_labels].iloc[-1]
    rank_corr = total_returns.rank().corr(
        pd.Series(range(1, n_quantiles + 1), index=q_labels, dtype=float)
    )

    # ── 换手率（简化：只保存最后一轮，作为代表值）───────────────────────────
    turnover_df = pd.DataFrame(index=period_rets.index,
                               columns=q_labels, dtype=float)

    return QuantileResult(
        nav=nav,
        annual_returns=annual_returns,
        ic_monotonicity=rank_corr,
        long_short_nav=long_short_nav,
        turnover=turnover_df,
    )


def plot_quantile_result(result: QuantileResult,
                         title: str = "分组回测（Q1-Q5）",
                         save_path: str = None):
    """
    4图布局：
    1. 各组NAV走势（含基准）
    2. 多空NAV
    3. 逐年分组收益柱状图
    4. 各组总收益条形图（直观看单调性）
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    q_labels = [c for c in result.nav.columns if c.startswith("Q")]
    n = len(q_labels)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n))  # Q1红→Q5绿

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # ── 图1: 各组NAV ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for i, q in enumerate(q_labels):
        result.nav[q].plot(ax=ax, color=colors[i], lw=1.5, label=q)
    if "benchmark" in result.nav.columns:
        result.nav["benchmark"].plot(ax=ax, color="gray", lw=1.5,
                                     ls="--", label="等权基准")
    ax.set_title("各分组净值走势")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylabel("净值")

    # ── 图2: 多空NAV ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    result.long_short_nav.plot(ax=ax, color="steelblue", lw=1.5)
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    # 超额收益为正的区域填充
    ls = result.long_short_nav.copy()
    ax.fill_between(ls.index, 1, ls, where=(ls > 1),
                    alpha=0.3, color="green", label="Q5>Q1")
    ax.fill_between(ls.index, 1, ls, where=(ls < 1),
                    alpha=0.3, color="red",   label="Q5<Q1")
    monotonicity = result.ic_monotonicity
    mono_color = "green" if monotonicity > 0.8 else "orange" if monotonicity > 0.5 else "red"
    ax.set_title(
        f"多空组合 (Q5 - Q1)  |  单调性={monotonicity:.3f}",
        color=mono_color
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylabel("净值")

    # ── 图3: 逐年分组收益柱状图 ──────────────────────────────────────────────
    ax = axes[1, 0]
    x = np.arange(len(result.annual_returns))
    width = 0.8 / n
    for i, q in enumerate(q_labels):
        if q in result.annual_returns.columns:
            vals = result.annual_returns[q].values
            ax.bar(x + i * width - 0.4 + width / 2, vals * 100,
                   width, label=q, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(result.annual_returns.index, rotation=45)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("逐年分组收益（%）")
    ax.legend(fontsize=8, ncol=n)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("收益率（%）")

    # ── 图4: 总收益单调性条形图 ──────────────────────────────────────────────
    ax = axes[1, 1]
    total_rets = (result.nav[q_labels].iloc[-1] - 1) * 100
    ax.bar(q_labels, total_rets.values, color=colors, alpha=0.85, edgecolor="white")
    for i, v in enumerate(total_rets.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    if "benchmark" in result.nav.columns:
        bm_ret = (result.nav["benchmark"].iloc[-1] - 1) * 100
        ax.axhline(bm_ret, color="gray", ls="--", lw=1.2,
                   label=f"基准 {bm_ret:.1f}%")
        ax.legend(fontsize=9)
    ax.set_title("各组总收益（单调性直观图）")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("总收益（%）")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def print_quantile_summary(result: QuantileResult):
    """打印分组回测摘要到终端"""
    q_labels = [c for c in result.nav.columns if c.startswith("Q")]
    total_rets = (result.nav[q_labels].iloc[-1] - 1) * 100

    print("\n" + "=" * 60)
    print("  分组回测摘要")
    print("=" * 60)
    print(f"  {'组别':<8} {'累计收益':>10} {'年化收益':>10}")
    print("-" * 60)
    n_years = (result.nav.index[-1] - result.nav.index[0]).days / 365
    for q in q_labels:
        cum  = total_rets[q]
        ann  = ((1 + cum / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
        bar  = "█" * max(0, int(ann / 5))
        print(f"  {q:<8} {cum:>9.1f}%  {ann:>9.1f}%  {bar}")
    if "benchmark" in result.nav.columns:
        bm_cum = (result.nav["benchmark"].iloc[-1] - 1) * 100
        bm_ann = ((1 + bm_cum / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
        print(f"\n  {'基准':<8} {bm_cum:>9.1f}%  {bm_ann:>9.1f}%")
    print(f"\n  多空累计: {(result.long_short_nav.iloc[-1]-1)*100:.1f}%")
    mono = result.ic_monotonicity
    color = "\033[92m" if mono > 0.8 else "\033[93m" if mono > 0.5 else "\033[91m"
    print(f"  单调性:   {color}{mono:.3f}{chr(27)}[0m"
          f"  ({'优秀' if mono>0.8 else '一般' if mono>0.5 else '较差'})")
    print("=" * 60)
