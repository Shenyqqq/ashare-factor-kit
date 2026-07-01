"""Industry-level IC breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.ic_series import compute_ic_series


def compute_ic_industry(
    factor_registry: dict,
    forward_return: pd.DataFrame,
    industry_map: pd.Series,
    tradable: pd.DataFrame | None = None,
    min_stocks: int = 20,
) -> pd.DataFrame:
    industry_groups = industry_map.groupby(industry_map).apply(lambda x: x.index.tolist())
    all_factor_stocks = set()
    for factor in factor_registry.values():
        all_factor_stocks.update(factor.columns)

    rows = []
    for ind_code, stocks in industry_groups.items():
        available = [s for s in stocks if s in all_factor_stocks]
        if len(available) < min_stocks:
            continue
        row = {"行业": ind_code, "股票数": len(available)}
        for fname, factor in factor_registry.items():
            cols = [s for s in available if s in factor.columns]
            if len(cols) < min_stocks:
                row[fname] = np.nan
                continue
            ic = compute_ic_series(factor[cols], forward_return.reindex(columns=cols), tradable=tradable)
            row[fname] = round(ic.mean(), 4) if len(ic) > 0 else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("行业")
