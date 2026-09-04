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

# live 日更专用缓存键前缀：单日截面残差，不含调仓日历（仅 as_of 一日）。
# 与训练 neut 缓存（NEUT_CACHE_VERSION）隔离，避免键冲突 / 口径漂移。
LIVE_NEUT_CACHE_VERSION = "live_neut_v1"


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


def _industry_panel_sniff(panel: pd.DataFrame | None) -> str:
    """PIT 行业长表指纹：行数 / code 数 / 生效日首尾。"""
    if panel is None or getattr(panel, "empty", True):
        return "pit:none"
    try:
        n = int(len(panel))
        nc = int(panel["code"].nunique()) if "code" in panel.columns else 0
        eff = panel["effective_date"] if "effective_date" in panel.columns else None
        if eff is None and "start_date" in panel.columns:
            eff = panel["start_date"]
        if eff is not None:
            e = pd.to_datetime(eff, errors="coerce")
            return f"pit:{n}:{nc}:{e.min()}:{e.max()}"
        return f"pit:{n}:{nc}"
    except Exception:
        return "pit:err"


def barra_bundle_sig(
    barra_factors: dict[str, pd.DataFrame] | None,
    *,
    industry_map: pd.Series | pd.DataFrame | None = None,
    weight_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
) -> str:
    """Barra 面板 + 行业映射 + WLS 权重指纹（影响残差结果的控制变量侧）。

    有 PIT ``industry_panel`` 时指纹含 panel，避免与静态回填残差共用缓存。
    """
    parts: list[str] = []
    if barra_factors:
        for name in sorted(barra_factors.keys()):
            parts.append(f"{name}:{_panel_sniff(barra_factors.get(name))}")
    else:
        parts.append("barra:none")
    if industry_panel is not None and not getattr(industry_panel, "empty", True):
        parts.append(_industry_panel_sniff(industry_panel))
    elif isinstance(industry_map, pd.DataFrame):
        if "sw_l2" in industry_map.columns:
            ind = industry_map["sw_l2"]
        else:
            ind = industry_map.iloc[:, 0] if industry_map.shape[1] else None
        if ind is not None and len(ind):
            try:
                vc = pd.Series(ind).astype(str).value_counts()
                top = ",".join(f"{k}:{int(v)}" for k, v in vc.head(5).items())
                parts.append(f"ind:{len(ind)}:{top}")
            except Exception:
                parts.append(f"ind:{len(ind)}")
        else:
            parts.append("ind:none")
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


def _neut_controls_key_infix(neut_controls: str) -> str:
    """默认 ``barra`` 不加 ``nc:``，与 13b 及更早 ``neut_v6`` 文件兼容。

    非默认控制集合（如 ``size_industry``）才插入 ``|nc:{mode}``，
    避免与 9 风格残差共用 ``factor_panel_neut_*`` / ``live_neut_*``。
    """
    mode = str(neut_controls or "barra").strip().lower() or "barra"
    if mode == "barra":
        return ""
    return f"|nc:{mode}"


def _safe_cache_subdir(tag: str) -> str:
    """目录名只保留字母数字和下划线，避免 ``mcap30_100`` 以外的奇怪字符。"""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tag))
    return cleaned.strip("_") or "extra"


def neut_cache_path(
    name: str,
    prices: pd.DataFrame,
    *,
    hold_period: int,
    rebalance_freq: str,
    rebalance_dates: pd.DatetimeIndex | None = None,
    ctrl_sig: str | None = None,
    neut_controls: str = "barra",
    universe_tag: str = "",
) -> Path:
    """
    残差面板落盘路径。

    缓存键 = 因子名 + 版本 + hold_period + rebalance_freq
    + 宇宙指纹 + 调仓日历指纹 + 控制变量指纹（Barra/行业/WLS 权重）。
    默认 ``barra`` 不加 ``nc:``（与旧 ``neut_v6`` 文件同键）；
    仅非默认集合插入 ``|nc:{neut_controls}``。
    非空 ``universe_tag``（如 ``mcap30_100``）再插入 ``|u:``，并落到子目录，
    禁止与全市场 ``factor_panel_neut_*``（无 mcap 后缀/子目录）同名覆盖。

    **跨 horizon / 跨调仓频率 / 跨样本宇宙 / 跨控制变量集合绝不可共用。**
    ``size_industry`` 与 9 风格 ``barra`` 必须落到不同 ``factor_panel_neut_*``。
    手动清理：``data/processed/factor_panels/factor_panel_neut_*.parquet``
    """
    from factors.factor_cache import FACTOR_CACHE_DIR

    u_infix = f"|u:{universe_tag}" if universe_tag else ""
    raw = (
        f"{name}|{NEUT_CACHE_VERSION}{_neut_controls_key_infix(neut_controls)}"
        f"|h{int(hold_period)}|{str(rebalance_freq)}"
        f"|{universe_sig(prices)}|{dates_sig(rebalance_dates)}"
        f"|{ctrl_sig or 'ctrl:na'}{u_infix}"
    )
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    fname = f"factor_panel_neut_{h}.parquet"
    if universe_tag:
        return FACTOR_CACHE_DIR / _safe_cache_subdir(universe_tag) / fname
    return FACTOR_CACHE_DIR / fname


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
    industry_panel: pd.DataFrame | None = None,
    membership_mask: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    min_industry_n: int = 0,
    restan_in_universe: bool = False,
) -> pd.DataFrame:
    """残差化 + re-zscore（与 lazy ``_maybe_neutralize`` 同口径）。

    ``membership_mask is None`` 时与旧全市场 WLS 完全一致。提供时先（可选）
    在当日宇宙 restan，再只对成员估 β；残差在池外为 NaN，随后 ``zscore_fn``
    只在有限值上标准化（等价池内 zscore）。
    """
    from models.wf.labels import residualize_panel

    src = panel
    if restan_in_universe and membership_mask is not None:
        from research.ic.universe import restan_within_mask
        src = restan_within_mask(panel, membership_mask, dates=dates_use)
    resid = residualize_panel(
        src, barra_factors, industry_map, dates_use,
        weight_panel=weight_panel,
        industry_panel=industry_panel,
        membership_mask=membership_mask,
        circ_mv=circ_mv,
        min_industry_n=int(min_industry_n or 0),
    )
    return zscore_fn(resid)


# ── live 日更专用 neut 缓存 ────────────────────────────────────────────────────
#
# live/daily_update.neutralize_as_of 每次重算当日 neut 行不落盘，热身窗内多因子
# 重复残差化耗时显著。此处提供单日行缓存：键 = 因子名 + live_neut_v1 + as_of
# + 宇宙指纹 + 控制变量指纹（Barra/行业/WLS），不含调仓日历（仅单日）。
# 与训练 neut 缓存（NEUT_CACHE_VERSION）键前缀隔离，落盘文件名
# live_neut_<hash>.parquet，单日行 DataFrame(index=[as_of], columns=universe)。


def live_neut_cache_path(
    name: str,
    as_of,
    prices: pd.DataFrame,
    *,
    ctrl_sig: str | None = None,
    neut_controls: str = "barra",
) -> Path:
    """live 单日 neut 缓存路径。

    缓存键 = 因子名 + live_neut_v1 + as_of_date + 宇宙指纹 + 控制变量指纹。
    默认 ``barra`` 不加 ``nc:``（与旧 live 文件同键）；仅非默认集合
    插入 ``|nc:{neut_controls}``。不含调仓日历（仅单日）。
    ``size_industry`` 与 9 风格 ``barra`` 不得共用 live_neut 缓存。
    """
    from factors.factor_cache import FACTOR_CACHE_DIR

    as_of_str = pd.Timestamp(as_of).strftime("%Y%m%d")
    raw = (
        f"{name}|{LIVE_NEUT_CACHE_VERSION}"
        f"{_neut_controls_key_infix(neut_controls)}|{as_of_str}"
        f"|{universe_sig(prices)}|{ctrl_sig or 'ctrl:na'}"
    )
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return FACTOR_CACHE_DIR / f"live_neut_{h}.parquet"


def try_load_live_neut(
    path: Path,
    *,
    as_of,
    universe: pd.Index,
    name: str = "",
) -> pd.Series | None:
    """读 live neut 单日缓存并对齐 universe；命中返回 Series，未命中返回 None。

    缓存文件是单日行 DataFrame(index=[as_of], columns=universe)。
    """
    if _cache_disabled():
        return None
    if not path.exists():
        logger.info(f"live neut MISS {name or path.stem}: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts not in df.index:
            logger.info(f"live neut MISS {name or path.stem} (as_of 不匹配): {path}")
            return None
        row = df.loc[as_of_ts]
        # 对齐 universe
        row = row.reindex(universe.astype(str).str.zfill(6))
        row = row.astype(np.float32, copy=False)
        arr = row.to_numpy(dtype=np.float32, copy=False)
        if np.isinf(arr).any():
            row = row.replace([np.inf, -np.inf], np.nan)
        logger.info(f"live neut HIT {name or path.stem}: {path}")
        return row
    except Exception as e:
        logger.info(f"live neut MISS {name or path.stem} (read fail: {e}): {path}")
        return None


def save_live_neut(
    path: Path,
    row: pd.Series,
    *,
    as_of,
    name: str = "",
) -> None:
    """写 live neut 单日缓存（单行 DataFrame，原子写盘）。"""
    if _cache_disabled() or row is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        as_of_ts = pd.Timestamp(as_of)
        df = row.to_frame().T
        df.index = pd.DatetimeIndex([as_of_ts])
        df = df.astype(np.float32, copy=False)
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp)
        os.replace(tmp, path)
        logger.info(f"live neut SAVE {name or path.stem}: {path}")
    except Exception as e:
        logger.warning(f"live neut SAVE fail {name or path.stem}: {e}")
        try:
            tmp = path.with_suffix(".tmp.parquet")
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
