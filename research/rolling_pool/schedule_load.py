"""
Load rolling-pool schedule parquet and apply date×factor masks for ML / WF.

Schedule 长表列：``date``, ``factor``（可选 ``abs_icir`` / ``abs_mean``）。

接入回测（默认 lazy，见 ``research.rolling_pool.lazy``）：
- Schedule 预计算；并集 ``U`` 仅用于面板存在性检查
- WF 每期特征列 = 调仓日 t 的 ``pool_t``；train/val/pred 共用，禁止窗内并集
- 运行时 ``ensure(pool_t)``（约 ≤50 列），禁止一次性 materialize U×T×N
- 旧路径（``--no-rolling-pool-lazy``）：并集 U 预先进 dataset + 按日 mask（过时，易 OOM）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def load_pool_schedule(path: str | Path) -> pd.DataFrame:
    """读取 schedule 长表，规范 ``date`` / ``factor`` 列。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rolling pool schedule 不存在: {p}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(p)
    else:
        raise ValueError(f"不支持的 schedule 格式: {p.suffix}（期望 .parquet / .csv）")
    if "date" not in df.columns or "factor" not in df.columns:
        raise ValueError(
            f"schedule 需含 date/factor 列，实际列={list(df.columns)}"
        )
    out = df[["date", "factor"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["factor"] = out["factor"].astype(str)
    out = out.dropna(subset=["date", "factor"]).drop_duplicates()
    if out.empty:
        raise ValueError(f"schedule 为空: {p}")
    return out.sort_values(["date", "factor"]).reset_index(drop=True)


def schedule_union(schedule: pd.DataFrame) -> list[str]:
    """并集 U = ∪ pools，稳定排序。"""
    return sorted(schedule["factor"].unique().tolist())


def schedule_pools_by_date(schedule: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    """date → 当期因子集合。"""
    pools: dict[pd.Timestamp, set[str]] = {}
    for d, g in schedule.groupby("date", sort=True):
        pools[pd.Timestamp(d)] = set(g["factor"].tolist())
    return pools


def schedule_tag(path: str | Path, *, max_len: int = 14) -> str:
    """
    schedule 文件 → 产物 tag 片段，形如 ``_rp{abbr}-{hash6}``。

    用于 ``run.py`` 的实验 tag，避免同参数的「滚动定池」与「固定池」实验
    互相覆盖 ``results/<tag>/``。abbr 取文件名去掉 ``rolling_pool_schedule``
    前缀后的可辨识片段；hash 取解析后绝对路径的 md5 前 6 位。
    """
    import hashlib
    import re

    p = Path(path)
    stem = p.stem
    for pref in ("rolling_pool_schedule_", "rolling_pool_schedule", "schedule_"):
        if stem.startswith(pref):
            stem = stem[len(pref):]
            break
    abbr = re.sub(r"[^0-9A-Za-z]+", "", stem)[:max_len].lower()
    try:
        key = str(p.resolve())
    except OSError:
        key = str(p)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:6]
    return f"_rp{abbr}-{h}" if abbr else f"_rp{h}"


def active_factors_by_rebalance(
    schedule: pd.DataFrame,
    rebalance_dates: list | pd.DatetimeIndex,
) -> dict[pd.Timestamp, list[str]]:
    """
    将 schedule 决策日 asof 对齐到调仓日：取 ``decision_date <= rebalance_date`` 的最近一期池。
    调仓日早于首个决策日 → 空列表。

    asof 用 ``<=`` 是安全的：schedule 生成侧决策日 t 的池只用 ``index < t``
    的 IC（见 ``research.rolling_pool.schedule`` 模块 docstring），故 t 当日
    即可知该池；此处**不再**额外 shift 一期。
    """
    pools = schedule_pools_by_date(schedule)
    if not pools:
        return {}
    pool_dates = pd.DatetimeIndex(sorted(pools.keys()))
    out: dict[pd.Timestamp, list[str]] = {}
    for d in rebalance_dates:
        dt = pd.Timestamp(d)
        pos = int(pool_dates.searchsorted(dt, side="right") - 1)
        if pos < 0:
            out[dt] = []
        else:
            out[dt] = sorted(pools[pool_dates[pos]])
    return out


def apply_schedule_mask(
    registry: dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    *,
    only_names: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    按 asof 池成员资格，把非当期池因子行置 NaN。

    - 仅处理 ``only_names``（默认 = schedule 并集 ∩ registry）中的因子
    - 首个决策日之前的行全部置 NaN（无池信息）
    - 决策日之间：沿用上一决策日池（ffill membership）
    - Barra / special 等不在 only_names 的列保持不变
    """
    if not registry:
        return registry
    pools = schedule_pools_by_date(schedule)
    if not pools:
        return registry

    union = set(schedule_union(schedule))
    names = [n for n in registry if n in union]
    if only_names is not None:
        names = [n for n in names if n in only_names]
    if not names:
        logger.warning("apply_schedule_mask: registry 与 schedule 并集无交集，跳过 mask")
        return registry

    pool_dates = pd.DatetimeIndex(sorted(pools.keys()))
    name_set = set(names)
    # membership: decision_date × factor（向量化构造）
    rows = []
    for d, facs in pools.items():
        for f in facs:
            if f in name_set:
                rows.append((d, f, True))
    if rows:
        long = pd.DataFrame(rows, columns=["date", "factor", "active"])
        memb = (
            long.pivot_table(
                index="date", columns="factor", values="active", aggfunc="max",
            )
            .reindex(index=pool_dates, columns=names)
            .fillna(False)
            .astype(bool)
        )
    else:
        memb = pd.DataFrame(False, index=pool_dates, columns=names)

    out = dict(registry)
    n_masked_factors = 0
    for name in names:
        panel = registry[name]
        if panel is None or panel.empty:
            continue
        # asof ffill onto panel index；首决策日前 → False
        active = memb[name].reindex(panel.index, method="ffill")
        mask = active.fillna(False).to_numpy(dtype=bool)
        if mask.all():
            continue
        arr = panel.to_numpy(dtype=np.float32, copy=True)
        arr[~mask, :] = np.nan
        out[name] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
        n_masked_factors += 1

    n_dates = len(pool_dates)
    mean_pool = float(np.mean([len(v) for v in pools.values()]))
    logger.info(
        f"rolling_pool mask: |U|={len(union)}, schedule_dates={n_dates}, "
        f"mean_pool={mean_pool:.1f}, masked_panels={n_masked_factors}/{len(names)}"
    )
    return out


def cs_zscore_sparse_rows(
    panel: pd.DataFrame,
    clip: float = 3.0,
    min_stocks: int = 2,
) -> pd.DataFrame:
    """
    截面 z-score，仅处理含有限值的行（rolling-pool mask 后绝大多数行为全 NaN）。

    与 ``factors.factor.cross_sectional_zscore`` 同口径（总体 std、±clip），
    但跳过全 NaN 行，避免 ``DataFrame.apply`` 在大并集上过慢。
    """
    if panel is None or panel.empty:
        return panel
    arr = panel.to_numpy(dtype=np.float64, copy=False)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        n = int(valid.sum())
        if n < min_stocks:
            continue
        v = row[valid]
        std = float(v.std(ddof=0))
        if std < 1e-12:
            out[i, valid] = 0.0
            continue
        z = (v - float(v.mean())) / std
        out[i, valid] = np.clip(z, -clip, clip).astype(np.float32)
    return pd.DataFrame(out, index=panel.index, columns=panel.columns)


def load_panels_prefer_cache(
    names: list[str],
    prices: pd.DataFrame,
    registry_kwargs: dict,
    *,
    compute_missing: bool = True,
    quiet: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    优先从 ``data/processed/factor_panels/`` 读单因子面板，reindex 到 ``prices``；
    缺失名再经 ``get_factor_registry`` 计算。

    注意：磁盘面板可能按全市场签名落盘；``--sample`` 时仅取列子集，
    截面 z-score 口径仍是缓存生成时的宇宙（冒烟可接受；全量同宇宙则一致）。

    quiet : bool
        True 或单名加载时用 debug 日志（lazy 按折高频调用）。
    """
    from factors.factor_cache import _cache_paths, _load_panel
    from factors.factor import get_factor_registry

    wanted = list(dict.fromkeys(names))
    registry: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for n in wanted:
        pq, _ = _cache_paths(n)
        if not pq.exists():
            missing.append(n)
            continue
        try:
            panel = _load_panel(pq)
            panel = panel.reindex(index=prices.index, columns=prices.columns)
            registry[n] = panel.astype(np.float32, copy=False)
        except Exception as e:
            logger.warning(f"rolling_pool panel 读取失败，将重算: {n} ({e})")
            missing.append(n)

    _log = logger.debug if (quiet or len(wanted) <= 3) else logger.info
    _log(
        f"rolling_pool panels: cache_hit={len(registry)}/{len(wanted)}, "
        f"missing={len(missing)}"
    )

    if missing and compute_missing:
        computed = get_factor_registry(
            **registry_kwargs, factor_names=missing, include_regime=False,
        )
        # 仅保留请求名（registry 可能附带其它）
        for n in missing:
            if n in computed and computed[n] is not None:
                registry[n] = computed[n].reindex(
                    index=prices.index, columns=prices.columns,
                ).astype(np.float32, copy=False)
        still = [n for n in missing if n not in registry]
        if still:
            logger.warning(
                f"rolling_pool: {len(still)} 个因子无法加载/计算 "
                f"（示例: {still[:8]}）"
            )
    elif missing and not compute_missing:
        logger.warning(
            f"rolling_pool: skip compute，{len(missing)} 个因子缺失"
        )

    return registry
