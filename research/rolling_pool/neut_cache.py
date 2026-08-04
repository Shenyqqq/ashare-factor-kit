"""
feature-neutralize 残差面板磁盘缓存（急切路径与 rolling-pool lazy 共用）。

落盘目录：``data/processed/factor_panels/factor_panel_neut_*.parquet``
手动清缓存：删除该 glob（或整目录 ``factor_panels/``）；改 Barra/残差算法后
也会因 ``NEUT_CACHE_VERSION`` / 指纹变化自动失效。

环境变量 ``FACTOR_CACHE_DISABLE=1`` 时跳过读写（强制重算）。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# 改残差算法 / 键字段时 bump，旧 factor_panel_neut_*.parquet 自动失效。
# v5：东财市值 + WLS；v6：键补调仓日历 / Barra·行业·权重指纹（急切路径对齐 lazy）。
NEUT_CACHE_VERSION = "neut_v6"


def _cache_disabled() -> bool:
    return os.getenv("FACTOR_CACHE_DISABLE", "0") == "1"


def universe_sig(prices: pd.DataFrame) -> str:
    """样本宇宙指纹（sample 子集 ≠ 全市场，中性化缓存不可混用）。"""
    cols = prices.columns
    c0 = str(cols[0]) if len(cols) else ""
    c1 = str(cols[-1]) if len(cols) else ""
    return f"{prices.shape[0]}x{prices.shape[1]}_{c0}_{c1}"


def dates_sig(dates: pd.DatetimeIndex | None) -> str:
    """调仓日历指纹：长度 + 首/中/末日。"""
    if dates is None:
        return "na"
    d = pd.DatetimeIndex(dates)
    if len(d) == 0:
        return "empty"
    mid = d[len(d) // 2]
    return f"{len(d)}_{d[0].date()}_{mid.date()}_{d[-1].date()}"


def _panel_sniff(df: pd.DataFrame | None) -> str:
    """轻量内容嗅探（shape + 首/中/末行有限值统计），避免全表 hash。"""
    if df is None or getattr(df, "empty", True):
        return "empty"
    try:
        n_r, n_c = int(df.shape[0]), int(df.shape[1])
        idx = df.index
        positions = sorted({0, n_r // 2, n_r - 1})
        parts = [f"{n_r}x{n_c}"]
        arr = df.to_numpy(dtype=np.float64, copy=False)
        for i in positions:
            row = arr[i]
            finite = np.isfinite(row)
            n_fin = int(finite.sum())
            s = float(np.nansum(row)) if n_fin else 0.0
            parts.append(f"{i}:{n_fin}:{s:.6g}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "err"


def barra_bundle_sig(
    barra_factors: dict[str, pd.DataFrame] | None,
    *,
    industry_map: pd.Series | pd.DataFrame | None = None,
    weight_panel: pd.DataFrame | None = None,
) -> str:
    """Barra 面板 + 行业映射 + WLS 权重指纹（影响残差结果的控制变量侧）。"""
    parts: list[str] = []
    if barra_factors:
        for name in sorted(barra_factors.keys()):
            parts.append(f"{name}:{_panel_sniff(barra_factors.get(name))}")
    else:
        parts.append("barra:none")
    if isinstance(industry_map, pd.DataFrame):
        if "sw_l2" in industry_map.columns:
            ind = industry_map["sw_l2"]
        else:
            ind = industry_map.iloc[:, 0] if industry_map.shape[1] else None
    else:
        ind = industry_map
    if ind is not None and len(ind):
        try:
            vc = pd.Series(ind).astype(str).value_counts()
            top = ",".join(f"{k}:{int(v)}" for k, v in vc.head(5).items())
            parts.append(f"ind:{len(ind)}:{top}")
        except Exception:
            parts.append(f"ind:{len(ind)}")
    else:
        parts.append("ind:none")
    parts.append(f"w:{_panel_sniff(weight_panel)}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]


def neut_cache_path(
    name: str,
    prices: pd.DataFrame,
    *,
    hold_period: int,
    rebalance_freq: str,
    rebalance_dates: pd.DatetimeIndex | None = None,
    ctrl_sig: str | None = None,
) -> Path:
    """
    残差面板落盘路径。

    缓存键 = 因子名 + 版本 + hold_period + rebalance_freq + 宇宙指纹
    + 调仓日历指纹 + 控制变量指纹（Barra/行业/WLS 权重）。

    **跨 horizon / 跨调仓频率 / 跨样本宇宙绝不可共用。**
    手动清理：``data/processed/factor_panels/factor_panel_neut_*.parquet``
    """
    from factors.factor_cache import FACTOR_CACHE_DIR

    raw = (
        f"{name}|{NEUT_CACHE_VERSION}|h{int(hold_period)}|{str(rebalance_freq)}"
        f"|{universe_sig(prices)}|{dates_sig(rebalance_dates)}"
        f"|{ctrl_sig or 'ctrl:na'}"
    )
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return FACTOR_CACHE_DIR / f"factor_panel_neut_{h}.parquet"


def try_load_neut_panel(
    path: Path,
    *,
    prices: pd.DataFrame,
    name: str = "",
) -> pd.DataFrame | None:
    """读盘并对齐 prices；失败返回 None。打 HIT/MISS 日志。"""
    if _cache_disabled():
        return None
    if not path.exists():
        logger.info(f"neut cache MISS {name or path.stem}: {path}")
        return None
    try:
        panel = pd.read_parquet(path)
        if not (
            panel.index.equals(prices.index)
            and panel.columns.equals(prices.columns)
        ):
            panel = panel.reindex(index=prices.index, columns=prices.columns)
        panel = panel.astype(np.float32, copy=False)
        # inf → NaN（与 build_factor_dataset / PanelStore._prepare 同口径）
        arr = panel.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(arr).all() and np.isinf(arr).any():
            panel = panel.replace([np.inf, -np.inf], np.nan)
        logger.info(f"neut cache HIT {name or path.stem}: {path}")
        return panel
    except Exception as e:
        logger.info(f"neut cache MISS {name or path.stem} (read fail: {e}): {path}")
        return None


def save_neut_panel(path: Path, panel: pd.DataFrame, *, name: str = "") -> None:
    """float32 原子写盘。"""
    if _cache_disabled() or panel is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        panel.astype(np.float32, copy=False).to_parquet(tmp)
        os.replace(tmp, path)
        logger.info(f"neut cache SAVE {name or path.stem}: {path}")
    except Exception as e:
        logger.warning(f"neut cache SAVE fail {name or path.stem}: {e}")
        try:
            tmp = path.with_suffix(".tmp.parquet")
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def neutralize_one_factor(
    panel: pd.DataFrame,
    name: str,
    *,
    barra_factors: dict[str, pd.DataFrame],
    industry_map: pd.Series,
    dates_use: pd.DatetimeIndex,
    weight_panel: pd.DataFrame | None,
    zscore_fn,
) -> pd.DataFrame:
    """残差化 + re-zscore（与 lazy ``_maybe_neutralize`` 同口径）。"""
    from models.wf.labels import residualize_panel

    resid = residualize_panel(
        panel, barra_factors, industry_map, dates_use,
        weight_panel=weight_panel,
    )
    return zscore_fn(resid)
