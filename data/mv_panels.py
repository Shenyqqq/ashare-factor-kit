"""
data/mv_panels.py — 市值宽表面板路径约定

主路径（Size / WLS √市值 / 对数市值）：
  ``total_mv.parquet`` / ``circ_mv.parquet``
  ← ``python -m data.download_stock_value_em``（东财日频，单位=元）

兜底/校验（自算）：
  ``total_mv_computed.parquet`` / ``circ_mv_computed.parquet``
  ← ``python -m data.compute_market_cap``（shares × prices_raw）

换手率仍由自算产出：
  ``turnover_rate.parquet`` ← ``compute_market_cap``（与 Size 解耦）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import RAW_DIR

PRIMARY = {
    "total_mv": RAW_DIR / "total_mv.parquet",
    "circ_mv": RAW_DIR / "circ_mv.parquet",
}
COMPUTED = {
    "total_mv": RAW_DIR / "total_mv_computed.parquet",
    "circ_mv": RAW_DIR / "circ_mv_computed.parquet",
}


def resolve_mv_path(kind: str, *, allow_computed_fallback: bool = True) -> Path | None:
    """返回可用的市值面板路径。优先东财主文件，可选回退自算。"""
    if kind not in PRIMARY:
        raise ValueError(f"unknown mv kind: {kind}")
    primary = PRIMARY[kind]
    if primary.exists():
        return primary
    if allow_computed_fallback:
        fb = COMPUTED[kind]
        if fb.exists():
            logger.warning(
                f"{kind}: 缺少东财主面板 {primary.name}，回退自算 {fb.name}；"
                f"请跑 `python -m data.download_stock_value_em`"
            )
            return fb
    return None


def load_mv_raw(kind: str, *, allow_computed_fallback: bool = True) -> pd.DataFrame | None:
    path = resolve_mv_path(kind, allow_computed_fallback=allow_computed_fallback)
    if path is None:
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str).str.zfill(6)
    return df
