"""
backtest/portfolio.py — holdings state and rebalance selection.

Maintains actual_position across periods; limit-down stocks that cannot be
sold carry into the next rebalance. Top-N backfill walks the ranked list until
*N* eligible names are filled.

Turnover control (任务3):
  ``apply_turnover_control`` 在 rebalance 前对 target 做约束式调整：
    1. rank_change_threshold > 0 → 上期持仓中 rank_pct 仍 ≥ (1 - threshold)
       的股票强制保留（即使跌出新 target），避免频繁换掉「只是排名略降」的票
    2. turnover_limit < 1.0 → 每期 |sells|+|buys| ≤ 2 × turnover_limit × |target|
       约束总换手率，控制交易成本
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.execution import TradeRules


@dataclass
class PortfolioState:
    """Actual holdings carried across rebalance periods."""

    holdings: frozenset[str] = field(default_factory=frozenset)

    def copy(self) -> PortfolioState:
        return PortfolioState(holdings=frozenset(self.holdings))


def equal_weights(stocks: list[str]) -> dict[str, float]:
    """Equal weight among held names (sum = 1)."""
    if not stocks:
        return {}
    w = 1.0 / len(stocks)
    return {s: w for s in stocks}


def select_top_n(
    scores: pd.Series,
    n: int,
    rules: TradeRules,
    execution_date: pd.Timestamp,
    close: pd.DataFrame,
) -> list[str]:
    """
    Rank by score descending; skip unbuyable names and backfill 31, 32, …
    until *n* stocks selected or candidates exhausted.
    """
    ranked = scores.sort_values(ascending=False)
    selected: list[str] = []
    for stock in ranked.index:
        if len(selected) >= n:
            break
        if stock not in close.columns:
            continue
        if rules.can_buy(stock, execution_date, close):
            selected.append(stock)
    return selected


def rebalance_holdings(
    prev: PortfolioState,
    target: set[str],
    execution_date: pd.Timestamp,
    rules: TradeRules,
    close: pd.DataFrame,
) -> tuple[PortfolioState, set[str], set[str]]:
    """
    Apply sell-then-buy with A-share constraints.

    Limit-down open → cannot sell → stuck in actual_position.
    Returns (new_state, sold_set, bought_set).
    """
    prev_set = set(prev.holdings)
    intended_sells = prev_set - target
    sold = {s for s in intended_sells if rules.can_sell(s, execution_date, close)}
    stuck = intended_sells - sold

    remaining = (prev_set - sold) | stuck
    intended_buys = target - remaining
    bought = {s for s in intended_buys if rules.can_buy(s, execution_date, close)}

    new_holdings = remaining | bought
    return PortfolioState(holdings=frozenset(new_holdings)), sold, bought


def apply_turnover_control(
    prev_holdings: set[str],
    target: set[str],
    scores: pd.Series,
    turnover_limit: float = 1.0,
    rank_change_threshold: float = 0.0,
) -> set[str]:
    """
    在 rebalance 前调整 target，控制换手率与排名变动阈值。

    逻辑：
      1. **No-swap keepers**: prev ∩ target —— 这些股票本来就在新 target 里，
         无需换仓，优先全部保留。
      2. **Rank-change forced keep** (rank_change_threshold > 0):
         上期持仓中 rank_pct ≥ (1 - threshold) 的股票视为「排名未显著下降」，
         即使不在新 target 也强制保留，避免「只跌几名就换」的浪费。
      3. **Turnover cap on sells** (turnover_limit < 1.0):
         计划卖出的 prev 持仓（prev - stay）若超过 ``floor(L × |prev|)``，
         将 rank_pct 最高的「计划卖出」股票加回 stay（即使不在 target），
         限制单期卖出数量。这样在 prev 与 target 完全不相交时仍能平滑过渡，
         而不是返回空集。
      4. **Trim overflow**: stay 超过 target_size 时按 rank_pct 降序裁剪。
      5. **Turnover cap on buys + fill remaining**:
         从 target - stay 中按 score 降序选新股补齐，且 n_buys ≤
         ``floor(L × target_size)``。

    单期换手率（与 ``compute_turnover`` 一致）：

        turnover = (|sells| + |buys|) / (2 × |union|)

    当 prev_size ≈ target_size 且 prev ∩ target 较大时，本函数的 cap
    (``L × prev_size`` 卖 + ``L × target_size`` 买) 使实际 turnover 接近 L；
    极端不相交情况下 turnover 可能略高于 L，但保证了组合不会变成空集。

    Parameters
    ----------
    prev_holdings : set[str]
        上期实际持仓。
    target : set[str]
        原始 target（quantile 分组 / Top-N / benchmark）。
    scores : pd.Series
        当期得分（越大越优先）。
    turnover_limit : float
        每期最大换手率，1.0=无限制。
    rank_change_threshold : float
        排名变动阈值，0.0=不启用。

    Returns
    -------
    set[str]
        调整后的 target。
    """
    if not prev_holdings:
        return set(target)
    if turnover_limit >= 1.0 and rank_change_threshold <= 0.0:
        return set(target)

    target = set(target)
    target_size = len(target)
    if target_size == 0:
        return target

    prev_set = set(prev_holdings)
    prev_size = len(prev_set)

    if len(scores) == 0:
        return target
    rank_pct = scores.rank(pct=True)

    # ── Step 1: no-swap keepers (prev ∩ target) ───────────────────────────────
    stay = prev_set & target

    # ── Step 2: rank-change forced keep ───────────────────────────────────────
    if rank_change_threshold > 0.0:
        keep_cutoff = 1.0 - rank_change_threshold
        forced_keep = {
            s for s in prev_set
            if s in rank_pct.index and float(rank_pct.loc[s]) >= keep_cutoff
        }
        stay = stay | forced_keep

    # ── Step 3: turnover cap on sells ─────────────────────────────────────────
    # 计划卖出 = prev - stay。若超过 floor(L × prev_size)，把最高 rank 的
    # 「计划卖出」股票保留下来（即使不在 target），以限制单期卖出数量。
    if turnover_limit < 1.0:
        max_sells = int(np.floor(turnover_limit * prev_size))
        planned_sells = prev_set - stay
        if len(planned_sells) > max_sells:
            n_keep_back = len(planned_sells) - max_sells
            planned_sells_sorted = sorted(
                planned_sells,
                key=lambda s: float(rank_pct.loc[s]) if s in rank_pct.index else -1.0,
                reverse=True,
            )
            stay = stay | set(planned_sells_sorted[:n_keep_back])

    # ── Step 4: trim overflow to target_size (drop lowest rank) ───────────────
    if len(stay) > target_size:
        sorted_stay = sorted(
            stay,
            key=lambda s: float(rank_pct.loc[s]) if s in rank_pct.index else -1.0,
            reverse=True,
        )
        stay = set(sorted_stay[:target_size])

    remaining_slots = target_size - len(stay)
    if remaining_slots <= 0:
        return stay

    # ── Step 5: turnover cap on buys + fill from target ───────────────────────
    if turnover_limit < 1.0:
        max_buys = int(np.floor(turnover_limit * target_size))
        n_buys = min(remaining_slots, max_buys)
    else:
        n_buys = remaining_slots

    if n_buys <= 0:
        return stay

    candidates = [
        s for s in target
        if s not in stay and s in scores.index
    ]
    candidates.sort(key=lambda s: float(scores.loc[s]), reverse=True)
    new_buys = set(candidates[:n_buys])
    return stay | new_buys


def assign_quantile_groups(
    scores: pd.Series,
    n_quantiles: int,
    q_labels: list[str],
    min_stocks: int,
) -> dict[str, list[str]] | None:
    """Return {Q1: [codes], …} or None if insufficient names."""
    if len(scores) < n_quantiles * min_stocks:
        return None
    try:
        groups = pd.qcut(scores, n_quantiles, labels=q_labels, duplicates="drop")
    except ValueError:
        return None
    result: dict[str, list[str]] = {q: [] for q in q_labels}
    for code, label in groups.items():
        result[str(label)].append(code)
    return result
