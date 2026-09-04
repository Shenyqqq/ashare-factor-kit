"""
backtest/turnover.py — turnover metrics and per-rebalance audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.execution import BacktestConfig, total_cost_fraction
from backtest.portfolio import PortfolioState


@dataclass
class TurnoverRecord:
    """One rebalance event for a portfolio track."""

    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    sells: list[str]
    buys: list[str]
    turnover: float
    cost: float


def compute_turnover(prev: PortfolioState, new: PortfolioState) -> float:
    """
    One-way turnover fraction: |Δ| / (2 × union_size).

    Empty → previous: turnover = 1.0 (full deploy).
    """
    prev_set = set(prev.holdings)
    new_set = set(new.holdings)
    if not prev_set:
        return 1.0 if new_set else 0.0
    union = prev_set | new_set
    if not union:
        return 0.0
    return len(prev_set.symmetric_difference(new_set)) / (2.0 * len(union))


def make_turnover_record(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    prev: PortfolioState,
    new: PortfolioState,
    sold: set[str],
    bought: set[str],
    cfg: BacktestConfig,
) -> TurnoverRecord:
    turnover = compute_turnover(prev, new)
    cost = total_cost_fraction(turnover, cfg)
    return TurnoverRecord(
        signal_date=signal_date,
        execution_date=execution_date,
        sells=sorted(sold),
        buys=sorted(bought),
        turnover=turnover,
        cost=cost,
    )


def records_to_dataframe(records: list[TurnoverRecord], label: str = "") -> pd.DataFrame:
    """Flatten turnover records for CSV export."""
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        rows.append({
            "group": label,
            "signal_date": r.signal_date,
            "execution_date": r.execution_date,
            "sells": "|".join(r.sells),
            "buys": "|".join(r.buys),
            "turnover": r.turnover,
            "cost": r.cost,
        })
    return pd.DataFrame(rows).set_index("signal_date")
