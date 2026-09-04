"""
FactorCorrProvider — 截面相关第二道去重的可插拔数据源。

设计
----
全库 ~500 因子逐日算截面相关过重。协议只要求对「本轮候选 ∪ 当前池」
的小集合给出两两 ``|corr|``。

默认实现 ``CachedPanelCorrProvider``：
  - 优先读 ``data/processed/factor_panels/`` 磁盘缓存（``factors.factor_cache``）
  - 对请求名流式加载 → 降采样切片 → 缓存切片（非整面板常驻）
  - 仅用 ``date <= asof`` 的采样日，保持因果

若某因子无缓存 / 面板不可用：该因子在截面相关中视为「不可比」，
不因第二道被剔除（第一道 IC 序列相关仍生效）。

降级
----
``NullFactorCorrProvider`` 始终返回空矩阵 → ``dedup_by_factor_cs_corr``
原样放行，并在 schedule 元数据里写 ``cs_corr_enabled=false``。
CLI 默认尝试 Cached；``--no-cs-corr`` 强制降级。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class FactorCorrProvider(Protocol):
    """小集合因子截面 |corr| 矩阵提供者。"""

    def pairwise_abs_corr(
        self,
        names: list[str],
        asof: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        返回 ``names × names`` 的平均截面 |Spearman corr|。

        - 对角为 1；缺失对为 NaN
        - 空 DataFrame 表示本轮无法做截面去重（调用方应跳过第二道）
        """
        ...

    @property
    def label(self) -> str:
        """人类可读实现名，写入元数据。"""
        ...


class NullFactorCorrProvider:
    """占位：第二道截面去重关闭。"""

    @property
    def label(self) -> str:
        return "null"

    def pairwise_abs_corr(
        self,
        names: list[str],
        asof: pd.Timestamp,
    ) -> pd.DataFrame:
        del names, asof
        return pd.DataFrame()


class CachedPanelCorrProvider:
    """
    从因子面板磁盘缓存加载小集合，算因果截面 Spearman |corr| 均值。

    Parameters
    ----------
    sample_step : 交易日抽样步长（降内存）
    max_sample_dates : 决策日前最多使用多少个采样截面
    min_stocks : 单日有效股票数下限
    """

    def __init__(
        self,
        *,
        sample_step: int = 20,
        max_sample_dates: int = 26,
        min_stocks: int = 30,
    ) -> None:
        self.sample_step = int(sample_step)
        self.max_sample_dates = int(max_sample_dates)
        self.min_stocks = int(min_stocks)
        # name -> (DatetimeIndex of sample rows, DataFrame float32 date×code)
        self._slices: dict[str, tuple[pd.DatetimeIndex, pd.DataFrame]] = {}
        self._missing: set[str] = set()
        self._name_to_parquet: dict[str, "Path"] | None = None
        self.n_loaded = 0
        self.n_missing = 0

    @property
    def label(self) -> str:
        return (
            f"cached_panel(step={self.sample_step},"
            f"max_dates={self.max_sample_dates})"
        )

    def _ensure_index(self) -> None:
        if self._name_to_parquet is not None:
            return
        from pathlib import Path

        from factors.factor_cache import FACTOR_CACHE_DIR, _cache_paths, _load_meta

        mapping: dict[str, Path] = {}
        if FACTOR_CACHE_DIR.exists():
            for meta_path in FACTOR_CACHE_DIR.glob("factor_panel_*.meta.json"):
                meta = _load_meta(meta_path)
                if not meta or "name" not in meta:
                    continue
                name = str(meta["name"])
                pq, _ = _cache_paths(name)
                if pq.exists():
                    mapping[name] = pq
        self._name_to_parquet = mapping

    def available_names(self) -> set[str]:
        self._ensure_index()
        assert self._name_to_parquet is not None
        return set(self._name_to_parquet)

    def _load_slice(self, name: str) -> tuple[pd.DatetimeIndex, pd.DataFrame] | None:
        if name in self._slices:
            return self._slices[name]
        if name in self._missing:
            return None
        self._ensure_index()
        assert self._name_to_parquet is not None
        pq = self._name_to_parquet.get(name)
        if pq is None:
            self._missing.add(name)
            self.n_missing += 1
            return None
        try:
            from factors.factor_cache import _load_panel

            panel = _load_panel(pq)
        except Exception:
            self._missing.add(name)
            self.n_missing += 1
            return None
        if panel is None or panel.empty:
            self._missing.add(name)
            self.n_missing += 1
            return None
        idx = panel.index[:: self.sample_step]
        sub = panel.loc[idx].astype(np.float32, copy=False)
        del panel
        self._slices[name] = (pd.DatetimeIndex(sub.index), sub)
        self.n_loaded += 1
        return self._slices[name]

    def pairwise_abs_corr(
        self,
        names: list[str],
        asof: pd.Timestamp,
    ) -> pd.DataFrame:
        asof = pd.Timestamp(asof)
        uniq = list(dict.fromkeys(names))
        if len(uniq) < 2:
            return pd.DataFrame()

        loaded: dict[str, pd.DataFrame] = {}
        for n in uniq:
            got = self._load_slice(n)
            if got is None:
                continue
            _idx, sub = got
            # causal: only dates <= asof
            mask = sub.index <= asof
            sub_c = sub.loc[mask]
            if sub_c.empty:
                continue
            if self.max_sample_dates > 0:
                sub_c = sub_c.tail(self.max_sample_dates)
            loaded[n] = sub_c

        usable = list(loaded.keys())
        if len(usable) < 2:
            return pd.DataFrame()

        # Align sample dates to intersection-ish: use union, skip sparse days
        date_sets = [set(df.index) for df in loaded.values()]
        # Prefer dates covered by majority of loaded factors
        from collections import Counter

        cnt: Counter = Counter()
        for ds in date_sets:
            cnt.update(ds)
        thr = max(2, int(np.ceil(0.5 * len(usable))))
        sample_dates = sorted(d for d, c in cnt.items() if c >= thr)
        if self.max_sample_dates > 0:
            sample_dates = sample_dates[-self.max_sample_dates :]
        if not sample_dates:
            return pd.DataFrame()

        corr_list: list[pd.DataFrame] = []
        for dt in sample_dates:
            row = {}
            for n, df in loaded.items():
                if dt in df.index:
                    row[n] = df.loc[dt]
            if len(row) < 2:
                continue
            # pairwise：先截面 rank 再 Pearson（等价 Spearman，更快）
            cs = pd.DataFrame(row)
            if cs.shape[0] < self.min_stocks:
                continue
            ranked = cs.rank(axis=0, method="average")
            c = ranked.corr(method="pearson", min_periods=self.min_stocks)
            if c is not None and not c.empty:
                corr_list.append(c.abs())

        if not corr_list:
            return pd.DataFrame()

        stacked = pd.concat(corr_list)
        mean_abs = stacked.groupby(level=0).mean()
        # Reindex to full uniq order (missing names → NaN row/col)
        mean_abs = mean_abs.reindex(index=uniq, columns=uniq)
        vals = np.array(mean_abs.to_numpy(dtype=float, copy=True), copy=True)
        n = vals.shape[0]
        for i in range(n):
            vals[i, i] = 1.0
        return pd.DataFrame(vals, index=uniq, columns=uniq)

    def release_cache(self) -> None:
        self._slices.clear()
