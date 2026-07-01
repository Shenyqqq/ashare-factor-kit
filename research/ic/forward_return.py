"""Forward return construction (aligned with ML / backtest)."""
from __future__ import annotations

import pandas as pd


def build_forward_return(
    prices: pd.DataFrame,
    open_: pd.DataFrame | None,
    period: int,
) -> pd.DataFrame:
    """
    有开盘价 → close[t+N]/open[t+1]-1（信号日收盘后次日开盘买入）
    无开盘价 → close[t+N]/close[t]-1
    """
    if open_ is not None:
        buy = open_.shift(-1)
        sell = prices.shift(-period)
        return sell / buy.replace(0, pd.NA) - 1
    return prices.pct_change(period).shift(-period)


def forward_return_label(open_: pd.DataFrame | None) -> str:
    return "open[t+1]→close[t+N]" if open_ is not None else "close→close"
