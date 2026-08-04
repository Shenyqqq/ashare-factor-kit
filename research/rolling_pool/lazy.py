"""
Lazy rolling-pool factor panels for ML / Walk-Forward.

避免一次性 materialize 并集 U（~395）×T×N：
- Schedule 仍预计算；U 仅用于存在性检查 / 磁盘面板校验
- 每一期调仓日 t 使用当日因子池 ``pool_t``（schedule asof，约 ≤50）
- 该期 train / val / pred **全部只用 pool_t**（同一组列）；禁止窗内并集
- 运行时 ``ensure(pool_t)`` 按需加载，峰值约 |pool_t|（+ always_on），≪ |U|
- 中性化结果落盘复用，避免 LRU 逐出后重复 OLS

正确性约定
----------
1. **neut 缓存键**（``research.rolling_pool.neut_cache``，与急切
   ``build_factor_dataset`` 共用）：名 + ``neut_v6`` + hold_period +
   rebalance_freq + 宇宙指纹 + 调仓日历指纹 + Barra/行业/权重指纹。
   **跨 horizon / 跨调仓频率 / 跨样本绝不可共用。**
   清缓存：删 ``data/processed/factor_panels/factor_panel_neut_*.parquet``。
2. **所有面板同口径清洗**：磁盘面板、registry 现算、``seed_panels``（always_on
   的 Barra / special）都过 ``_prepare``（reindex prices → float32 → inf→NaN）。
   ``should_skip_neutralize`` 的名字跳过残差化 / re-zscore（不双重 zscore），
   但仍过 inf 清洗。
3. **fail-fast**：``strict=True``（默认）下面板缺失 / 日期对不上直接 raise，
   禁止整列 NaN→``fillna(0)`` 静默变常数特征。
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from research.rolling_pool.neut_cache import (
    NEUT_CACHE_VERSION,
    barra_bundle_sig,
    neut_cache_path as _neut_cache_path,
    save_neut_panel,
    try_load_neut_panel,
    universe_sig as _universe_sig,
)


def process_rss_gb() -> float:
    """当前进程 RSS（GB）；失败时返回 nan。无需 psutil。"""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 3)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb,
        )
        if ok:
            return float(counters.WorkingSetSize) / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return float("nan")


def log_rss(tag: str) -> float:
    rss = process_rss_gb()
    if np.isfinite(rss):
        logger.info(f"[mem] {tag}: RSS={rss:.2f} GB")
    else:
        logger.info(f"[mem] {tag}: RSS=n/a")
    return rss


def union_active_factors(
    active_factors: dict[pd.Timestamp, list[str]] | None,
    dates: Iterable,
    *,
    always_on: list[str] | None = None,
) -> list[str]:
    """
    多日活跃因子并集，稳定排序；``always_on`` 追加在末尾去重。

    仅用于元数据 / 存在性检查。WF 特征列请用 ``pool_features_for_date``
    （单日 pool_t），禁止对 train∪val∪pred 取并集喂模型。
    """
    names: set[str] = set()
    if active_factors:
        for d in dates:
            dt = pd.Timestamp(d)
            facs = active_factors.get(dt)
            if facs is None:
                facs = active_factors.get(d, [])
            names.update(facs)
    out = sorted(names)
    if always_on:
        for n in always_on:
            if n not in names:
                out.append(n)
    return out


def pool_features_for_date(
    active_factors: dict[pd.Timestamp, list[str]] | None,
    date,
    *,
    always_on: list[str] | None = None,
) -> list[str]:
    """
    调仓日 ``date`` 的因子池 pool_t（+ always_on），稳定排序。

    该期训练 / 验证 / 预测均应使用这一组列（历史截面也取同列真实值，
    不再按「历史日是否入池」二次 mask）。
    """
    return union_active_factors(active_factors, [date], always_on=always_on)


class MissingFactorPanelError(FileNotFoundError):
    """schedule 里的因子在磁盘 / registry 中找不到（严格模式 fail-fast）。"""


def assert_panel_files_exist(
    names: list[str],
    *,
    allow_missing: bool = False,
) -> tuple[list[str], list[str]]:
    """
    检查 ``data/processed/factor_panels/`` 是否有对应 parquet。

    缺文件默认 **raise**（``MissingFactorPanelError``）：schedule 里的因子名与
    磁盘面板对不上，通常意味着因子改名 / 未预计算，静默放行会让该列整段
    NaN→fillna(0)，变成一列常数特征而无人察觉。

    Parameters
    ----------
    allow_missing : bool
        **仅 debug**。True 时只 warning 放行（缺失面板会在首次 ``get`` 时尝试
        现算，若仍失败则该列在严格模式下仍会 fail-fast）。

    Returns
    -------
    (present, missing)
    """
    from factors.factor_cache import _cache_paths

    present, missing = [], []
    for n in names:
        pq, _ = _cache_paths(n)
        if pq.exists():
            present.append(n)
        else:
            missing.append(n)
    if missing:
        msg = (
            f"rolling_pool: {len(missing)}/{len(names)} 个因子面板缺失 "
            f"（示例: {missing[:8]}）"
        )
        if allow_missing:
            logger.warning(msg + "；allow_missing=True（仅 debug），将在首次 get 时尝试计算")
        else:
            raise MissingFactorPanelError(
                msg
                + "。请先生成对应 factor_panels/ 缓存或重生成 schedule；"
                  "确认要放行时显式传 allow_missing=True / "
                  "--no-rolling-pool-strict（仅 debug）"
            )
    return present, missing


# 兼容旧 import：``from research.rolling_pool.lazy import NEUT_CACHE_VERSION``
# / ``_neut_cache_path`` / ``_universe_sig``（实现见 neut_cache.py）。


class RollingPoolPanelStore:
    """
    按需加载单因子面板的 LRU 缓存。

    - 原始面板：磁盘 parquet / registry 现算 → ``_prepare`` 统一清洗
    - ``_prepare``：reindex 到 ``prices`` index/columns → float32 → ±inf→NaN
      （与固定池路径 ``build_factor_dataset`` 的 inf 清洗出口同口径）
    - 可选：Barra+行业残差化 + 稀疏行 re-zscore；结果写 ``factor_panel_neut_*.parquet``
      （缓存键含 hold_period + rebalance_freq，见 ``_neut_cache_path``）
    - ``seed_panels``（Barra-as-features / special packs 等 always_on）：**不**直接
      塞 ``_cache``，同样经 ``_prepare`` + ``_maybe_neutralize``；``should_skip_neutralize``
      的名字仍跳过残差化与 re-zscore（避免双重 zscore），但一定过 inf 清洗
    - ``ensure(names)`` 会自动把 ``max_cached`` 抬到至少 ``len(names)``，
      以容纳当期 pool_t（+ always_on）
    - **sticky LRU**：WF 主循环每期只 ``ensure(pool_t)``，不每期清空。池换手
      仅 ~20%，跨期保留可让多数因子直接命中内存（省读盘 + 省残差化）；
      常驻规模由 ``max_cached`` 封顶，超限淘汰最久未用者。
      把 ``--rolling-pool-max-cached`` 调到 ≈|pool_t| 即退回「每期几乎全换」
    - ``strict=True``（默认）：面板加载失败直接 raise，禁止整列 NaN→fillna(0)
      静默变成常数特征
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        registry_kwargs: dict,
        *,
        compute_missing: bool = True,
        feature_neutralize: bool = False,
        barra_factors: dict[str, pd.DataFrame] | None = None,
        industry_map: pd.Series | None = None,
        weight_panel: pd.DataFrame | None = None,
        active_dates_by_factor: dict[str, pd.DatetimeIndex] | None = None,
        rebalance_dates: pd.DatetimeIndex | None = None,
        max_cached: int = 160,
        seed_panels: dict[str, pd.DataFrame] | None = None,
        neut_disk_cache: bool = True,
        hold_period: int | None = None,
        rebalance_freq: str | None = None,
        strict: bool = True,
    ):
        self.prices = prices
        self.registry_kwargs = registry_kwargs
        self.compute_missing = compute_missing
        self.feature_neutralize = feature_neutralize
        self.barra_factors = barra_factors
        self.industry_map = industry_map
        # 截面残差化的 WLS 权重面板（= √市值），None → 等权 OLS
        self.weight_panel = weight_panel
        self.active_dates_by_factor = active_dates_by_factor or {}
        self.rebalance_dates = (
            pd.DatetimeIndex(rebalance_dates)
            if rebalance_dates is not None
            else pd.DatetimeIndex([])
        )
        self.max_cached = max(8, int(max_cached))
        self.neut_disk_cache = bool(neut_disk_cache)
        self.hold_period = hold_period
        self.rebalance_freq = rebalance_freq
        self.strict = bool(strict)
        if self.feature_neutralize and self.neut_disk_cache and (
            hold_period is None or rebalance_freq is None
        ):
            raise ValueError(
                "neut 磁盘缓存需要 hold_period + rebalance_freq（缓存键必须区分 "
                "horizon / 调仓频率，否则 h20 会命中 h5 的残差面板）；"
                "如确实不需要落盘请传 neut_disk_cache=False"
            )
        # 控制变量指纹：与急切 build_factor_dataset 共用同一键规则
        self._ctrl_sig = (
            barra_bundle_sig(
                barra_factors if feature_neutralize else None,
                industry_map=industry_map if feature_neutralize else None,
                weight_panel=weight_panel if feature_neutralize else None,
            )
            if feature_neutralize
            else "ctrl:na"
        )
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._pin: set[str] | None = None
        self._failed: set[str] = set()
        # seed（always_on）原始面板：首次 get 时与磁盘面板走同一清洗管线
        self._seed_raw: dict[str, pd.DataFrame] = {
            n: p for n, p in (seed_panels or {}).items() if p is not None
        }
        self.n_loads = 0
        self.n_hits = 0
        self.n_neut_disk_hits = 0
        self.n_seed_loads = 0
        self.peak_cached = 0

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def cached_names(self) -> list[str]:
        return list(self._cache.keys())

    def seed_names(self) -> list[str]:
        return list(self._seed_raw.keys())

    def get(self, name: str) -> pd.DataFrame | None:
        if name in self._failed:
            if self.strict:
                raise MissingFactorPanelError(
                    f"rolling_pool: 因子面板不可用 {name!r}（此前加载失败）"
                )
            return None
        if name in self._cache:
            self._cache.move_to_end(name)
            self.n_hits += 1
            return self._cache[name]

        seed = self._seed_raw.get(name)
        panel = None
        # seed 面板已在内存，不走 neut 磁盘缓存（避免把未中性化面板写进 neut 键）
        if seed is None and self.feature_neutralize and self.neut_disk_cache:
            panel = self._try_load_neut_disk(name)

        if panel is None:
            raw = seed if seed is not None else self._load_raw(name)
            if raw is None:
                self._failed.add(name)
                if self.strict:
                    raise MissingFactorPanelError(
                        f"rolling_pool: 因子面板缺失且无法计算 {name!r}。"
                        "schedule 中的因子名与磁盘 factor_panels/ 或 registry 对不上；"
                        "请重生成 schedule / 预计算面板，或（仅 debug）关闭严格模式"
                    )
                logger.warning(
                    f"rolling_pool: 因子面板缺失 {name!r}（非严格模式，该列将为 NaN）"
                )
                return None
            panel = self._prepare(raw)
            if self.feature_neutralize:
                panel = self._maybe_neutralize(name, panel)
                if self.neut_disk_cache and seed is None:
                    self._save_neut_disk(name, panel)
            if seed is not None:
                self.n_seed_loads += 1

        self._cache[name] = panel
        self.n_loads += 1
        self.peak_cached = max(self.peak_cached, len(self._cache))
        self._evict()
        return panel

    def ensure(self, names: Iterable[str]) -> list[str]:
        """预热缓存；自动抬高 max_cached，并 pin 住本批避免中途逐出。"""
        wanted = list(dict.fromkeys(names))
        if len(wanted) > self.max_cached:
            logger.info(
                f"PanelStore: bump max_cached {self.max_cached} → {len(wanted)} "
                f"（当期 pool_t）"
            )
            self.max_cached = len(wanted)
        self._pin = set(wanted)
        ok = []
        try:
            for n in wanted:
                if self.get(n) is not None:
                    ok.append(n)
        finally:
            self._pin = None
        self._evict()
        return ok

    def release_except(self, keep: Iterable[str] | None = None) -> int:
        """
        释放不在 keep 中的面板；keep=None 清空。返回释放数量。

        WF 主循环**不**每期调用它（sticky LRU：跨期保留提高命中率，常驻规模由
        ``max_cached`` 封顶）。保留此方法用于手动/异常路径下的显式回收。
        """
        keep_set = set(keep) if keep is not None else set()
        drop = [n for n in self._cache if n not in keep_set]
        for n in drop:
            del self._cache[n]
        return len(drop)

    def _evict(self) -> None:
        while len(self._cache) > self.max_cached:
            victim = None
            for k in self._cache.keys():
                if self._pin is not None and k in self._pin:
                    continue
                victim = k
                break
            if victim is None:
                break
            del self._cache[victim]

    def _load_raw(self, name: str) -> pd.DataFrame | None:
        from research.rolling_pool.schedule_load import load_panels_prefer_cache

        reg = load_panels_prefer_cache(
            [name],
            self.prices,
            self.registry_kwargs,
            compute_missing=self.compute_missing,
            quiet=True,
        )
        panel = reg.get(name)
        if panel is None:
            return None
        return panel.astype(np.float32, copy=False)

    def _neut_path(self, name: str) -> Path:
        dates = self.active_dates_by_factor.get(name)
        if dates is None or len(dates) == 0:
            dates = self.rebalance_dates
        return _neut_cache_path(
            name,
            self.prices,
            hold_period=int(self.hold_period or 0),
            rebalance_freq=str(self.rebalance_freq or "na"),
            rebalance_dates=dates,
            ctrl_sig=self._ctrl_sig,
        )

    def _try_load_neut_disk(self, name: str) -> pd.DataFrame | None:
        path = self._neut_path(name)
        panel = try_load_neut_panel(path, prices=self.prices, name=name)
        if panel is None:
            return None
        self.n_neut_disk_hits += 1
        return self._prepare(panel)

    def _save_neut_disk(self, name: str, panel: pd.DataFrame) -> None:
        save_neut_panel(self._neut_path(name), panel, name=name)

    def _prepare(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        统一入口清洗：对齐 ``prices`` index/columns → float32 → ±inf→NaN。

        磁盘面板、registry 现算面板、``seed_panels``（Barra / special always_on）
        与 neut 磁盘缓存全部走这里，保证与固定池路径
        （``strategies.ml.build_factor_dataset`` 的 inf 清洗出口）同口径。
        不做额外 winsor / zscore：因子面板出厂已 winsor+cs_zscore，
        残差化后的 re-zscore 由 ``_maybe_neutralize`` 负责，避免双重标准化。
        """
        if panel is None:
            return panel
        if not (
            panel.index.equals(self.prices.index)
            and panel.columns.equals(self.prices.columns)
        ):
            panel = panel.reindex(
                index=self.prices.index, columns=self.prices.columns,
            )
        panel = panel.astype(np.float32, copy=False)
        return self._clean_inf(panel)

    @staticmethod
    def _clean_inf(panel: pd.DataFrame) -> pd.DataFrame:
        try:
            arr = panel.to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(arr).all() and np.isinf(arr).any():
                return panel.replace([np.inf, -np.inf], np.nan)
        except Exception:
            return panel.replace([np.inf, -np.inf], np.nan)
        return panel

    def _maybe_neutralize(self, name: str, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Barra+行业残差化 + 稀疏行 re-zscore。

        ``should_skip_neutralize``（``Barra_*`` / special pack）原样返回：
        这些面板已是 z-score 口径，不再残差化也不再 re-zscore（禁止双重 zscore），
        但已在 ``_prepare`` 过 inf 清洗与 dtype/对齐。
        """
        from factors.special_factors import should_skip_neutralize
        from models.wf.labels import residualize_panel
        from research.rolling_pool.schedule_load import cs_zscore_sparse_rows

        if should_skip_neutralize(name):
            return panel
        if self.barra_factors is None or self.industry_map is None:
            return panel

        dates_use = self.active_dates_by_factor.get(name)
        if dates_use is None or len(dates_use) == 0:
            dates_use = self.rebalance_dates
        if dates_use is None or len(dates_use) == 0:
            return panel

        resid = residualize_panel(
            panel, self.barra_factors, self.industry_map, dates_use,
            weight_panel=self.weight_panel,
        )
        return cs_zscore_sparse_rows(resid)

    def stats_summary(self) -> str:
        return (
            f"loads={self.n_loads}, hits={self.n_hits}, "
            f"neut_disk_hits={self.n_neut_disk_hits}, "
            f"seed_loads={self.n_seed_loads}, "
            f"cached={len(self._cache)}, peak_cached={self.peak_cached}, "
            f"max_cached={self.max_cached}, failed={len(self._failed)}"
        )


def _report_all_nan_columns(
    store: "RollingPoolPanelStore",
    dt: pd.Timestamp,
    X_raw: pd.DataFrame,
    *,
    strict: bool,
) -> None:
    """
    当日整列 NaN 的特征：会被 ``fillna(0)`` 变成常数列。

    整截面全为 NaN（所有列都空）在严格模式下 raise；个别稀疏因子当日无值
    是合法的，按因子名去重 warning 一次，避免静默。
    """
    empty = [c for c in X_raw.columns if not X_raw[c].notna().any()]
    if not empty:
        return
    if strict and len(empty) == X_raw.shape[1]:
        raise MissingFactorPanelError(
            f"rolling_pool: {dt.date()} 全部 {len(empty)} 个特征当日整列 NaN"
            f"（示例: {empty[:8]}）；疑似因子名 / 日期对不上"
        )
    seen = getattr(store, "_warned_all_nan", None)
    if seen is None:
        seen = set()
        try:
            store._warned_all_nan = seen
        except Exception:
            return
    fresh = [c for c in empty if c not in seen]
    if not fresh:
        return
    seen.update(fresh)
    logger.warning(
        f"rolling_pool: {len(fresh)} 个特征在 {dt.date()} 整列 NaN → fillna(0) 常数列"
        f"（示例: {fresh[:8]}）；若非稀疏因子请检查因子名 / 面板覆盖区间"
    )


def build_cross_section_from_store(
    store: RollingPoolPanelStore,
    forward_return: pd.DataFrame,
    date,
    feature_names: list[str],
    *,
    active_set: set[str] | None = None,
    strict: bool | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """
    按 ``feature_names`` 从 store 拼截面。

    ``feature_names`` 应为当期 pool_t（+ always_on）；历史训练日也用同一组列，
    取面板真实值（``active_set=None``）。

    ``active_set``：可选；若提供，不在集合内的列放 NaN（随后 fillna(0)），
    且 ``has_data`` 只认真实非 NaN。pool_t 语义下调用方应传 None，
    避免把「历史日未入池」再 mask 成另一套特征。

    ``strict``：None 时取 ``store.strict``（默认 True）。严格模式下，
    「面板拿不到 / 该日不在面板 index」直接 raise，禁止整列 NaN→fillna(0)
    静默当成常数特征；真实存在但当日全 NaN 的稀疏因子仍放行（合法）。
    """
    dt = pd.Timestamp(date)
    if not feature_names:
        return None, None
    if dt not in forward_return.index:
        return None, None
    if strict is None:
        strict = bool(getattr(store, "strict", True))

    rows: dict[str, pd.Series] = {}
    for name in feature_names:
        panel = store.get(name)
        if panel is None or panel.empty:
            if strict:
                raise MissingFactorPanelError(
                    f"rolling_pool: 特征 {name!r} 无可用面板（date={dt.date()}）；"
                    "禁止整列 NaN→0 静默当特征"
                )
            continue
        if dt not in panel.index:
            if strict:
                raise MissingFactorPanelError(
                    f"rolling_pool: 特征 {name!r} 面板缺少日期 {dt.date()}"
                    "（面板未与 prices index 对齐？）"
                )
            continue
        if active_set is not None and name not in active_set:
            rows[name] = pd.Series(np.nan, index=panel.columns, dtype=np.float32)
        else:
            rows[name] = panel.loc[dt]

    if not rows:
        return None, None

    X_raw = pd.DataFrame(rows)
    X_raw = X_raw.reindex(columns=feature_names)
    if active_set is None:
        _report_all_nan_columns(store, dt, X_raw, strict=strict)
    has_data = X_raw.notna().any(axis=1)
    X = X_raw.fillna(0.0)
    X = X.loc[has_data]
    y = forward_return.loc[dt].reindex(X.index).dropna()
    X = X.loc[X.index.intersection(y.index)]
    if X.empty:
        return None, None
    return X, y.loc[X.index]
