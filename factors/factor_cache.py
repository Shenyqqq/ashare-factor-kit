"""
factors/factor_cache.py  —  因子面板磁盘缓存

缓存键：horizon、rebalance_freq、因子白名单 hash、BACKTEST 起止日期。
存储：PROCESSED_DIR/factor_panel_h{h}_{freq}_{start}_{end}_{hash}.parquet
      + 同名 .meta.json（因子名列表等元数据）
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import BACKTEST_START, BACKTEST_END, PROCESSED_DIR


def is_regime_factor(name: str) -> bool:
    return name.startswith("市场") or name.startswith("HMM_")


def whitelist_hash(factor_whitelist: list | None) -> str:
    if factor_whitelist is None:
        payload = "__all__"
    else:
        payload = "|".join(sorted(factor_whitelist))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def factor_cache_path(
    hold_period: int,
    rebalance_freq: str,
    factor_whitelist: list | None,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """返回因子面板 parquet 路径（不含 forward_return）。"""
    start = start or BACKTEST_START
    end = end or BACKTEST_END
    freq = (rebalance_freq or "ME").replace("/", "-")
    h = whitelist_hash(factor_whitelist)
    name = f"factor_panel_h{hold_period}_{freq}_{start}_{end}_{h}.parquet"
    return PROCESSED_DIR / name


def panel_to_frame(factor_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name in sorted(factor_panel):
        df = factor_panel[name].astype("float32")
        df.columns = pd.MultiIndex.from_product(
            [[name], df.columns], names=["factor", "code"]
        )
        parts.append(df)
    return pd.concat(parts, axis=1)


def frame_to_panel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factors = df.columns.get_level_values("factor").unique()
    return {str(f): df[f].astype("float32") for f in factors}


def save_factor_panel(
    path: Path,
    factor_panel: dict[str, pd.DataFrame],
    *,
    hold_period: int,
    rebalance_freq: str,
    factor_whitelist: list | None,
    start: str | None = None,
    end: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_to_frame(factor_panel).to_parquet(path)
    meta = {
        "hold_period": hold_period,
        "rebalance_freq": rebalance_freq,
        "backtest_start": start or BACKTEST_START,
        "backtest_end": end or BACKTEST_END,
        "whitelist_hash": whitelist_hash(factor_whitelist),
        "factor_names": sorted(factor_panel.keys()),
        "n_factors": len(factor_panel),
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"因子面板缓存已保存: {path} ({len(factor_panel)} 个因子)")


def load_factor_panel(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    panel = frame_to_panel(df)
    logger.info(f"因子面板缓存命中: {path} ({len(panel)} 个因子)")
    return panel


def cache_exists(path: Path) -> bool:
    return path.exists() and path.with_suffix(".meta.json").exists()
