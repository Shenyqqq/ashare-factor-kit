"""Industry-level IC breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.ic_series import compute_ic_series


def compute_ic_industry(
    factor_registry,
    forward_return: pd.DataFrame,
    industry_map: pd.Series,
    tradable: pd.DataFrame | None = None,
    min_stocks: int = 20,
    names: list | None = None,
) -> pd.DataFrame:
    """
    分行业 IC。

    ``factor_registry`` 可以是普通 dict（向后兼容）或 `_LazyFactorRegistry`：
      - LazyRegistry 模式下逐因子 __getitem__ 加载面板 → 算各行业 IC → 释放全面板
        （峰值 = 1 个面板 ~48MB），而非同时持有全部面板。
        此模式需传 names 列表（或从 LazyRegistry._names 推导）。

    股票池取 ``forward_return.columns``（与因子面板同 universe，因子已 reindex
    到 prices.columns，等价于原 ``all_factor_stocks`` 并集，结果不变）。
    """
    industry_groups = industry_map.groupby(industry_map).apply(lambda x: x.index.tolist())
    all_factor_stocks = set(forward_return.columns)

    is_lazy = hasattr(factor_registry, "release_cache") and hasattr(factor_registry, "__getitem__")

    # 预过滤有效行业 + 各行业可用股票（基于 forward_return 全 universe）
    valid_industries: list[tuple[str, list]] = []
    for ind_code, stocks in industry_groups.items():
        available = [s for s in stocks if s in all_factor_stocks]
        if len(available) >= min_stocks:
            valid_industries.append((ind_code, available))

    if not valid_industries:
        return pd.DataFrame()

    if is_lazy:
        names_iter = names if names is not None else (
            list(factor_registry._names) if hasattr(factor_registry, "_names") else []
        )
    else:
        names_iter = list(factor_registry.keys())

    # rows: 行业 -> {因子: IC均值, "股票数": N}
    rows: dict[str, dict] = {
        ind_code: {"行业": ind_code, "股票数": len(available)}
        for ind_code, available in valid_industries
    }

    for fname in names_iter:
        if is_lazy:
            if fname not in factor_registry:
                continue
            try:
                factor = factor_registry[fname]
            except KeyError:
                continue
        else:
            factor = factor_registry[fname]
        if factor is None or factor.empty:
            for ind_code, _ in valid_industries:
                rows[ind_code][fname] = np.nan
            continue
        for ind_code, available in valid_industries:
            cols = [s for s in available if s in factor.columns]
            if len(cols) < min_stocks:
                rows[ind_code][fname] = np.nan
                continue
            ic = compute_ic_series(
                factor[cols], forward_return.reindex(columns=cols), tradable=tradable
            )
            rows[ind_code][fname] = round(ic.mean(), 4) if len(ic) > 0 else np.nan
        del factor

    if is_lazy:
        factor_registry.release_cache()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(list(rows.values())).set_index("行业")
