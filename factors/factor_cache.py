"""
factors/factor_cache.py — 因子面板磁盘缓存层

为 ``factors.factor.compute_single_factor`` 提供透明的 parquet 落盘缓存，
消除 IC 分析管线（Stage 2 IC / Stage 3 turnover / Stage 5 Barra /
Stage 7 corr 去重 / Stage 8 Gram-Schmidt）对同一因子面板的 4-5 遍重算。

设计要点
--------
- 缓存键：因子名 ``name``。文件名经 md5 sanitize 成 ASCII 安全串
  ``factor_panel_<hex12>.parquet``，附 sidecar ``factor_panel_<hex12>.meta.json``
  记录原 name + 输入数据指纹，便于审计与失效判定。
- 失效策略：sidecar ``.meta.json`` 记录输入 bundle 的指纹（prices/clean_ret/
  financial 等的 shape + index 首尾 + financial 列集 + financial 末报告期），
  加载时与当前输入比对，任一不匹配即重算覆盖。
- 内存：缓存读出的 panel 一律 ``astype(np.float32)``，与
  ``research.ic.ic_series._to_float32_panel`` 同口径；调用方用完即释，
  不在缓存层常驻。
- 调试：环境变量 ``FACTOR_CACHE_DISABLE=1`` 完全跳过缓存（强制重算）。
- 版本：``_FACTOR_LIB_VERSION`` 常量，改因子数学时 bump 即让全量缓存失效。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import PROCESSED_DIR

# 改动因子数学 / 标准化逻辑时 bump，让全量缓存自动失效
_FACTOR_LIB_VERSION = "3"  # OpenSourceAP Batch-3（价量残差/会计近似）

FACTOR_CACHE_DIR: Path = PROCESSED_DIR / "factor_panels"

# 压缩：优先 zstd（pyarrow 24+ 默认支持），写盘失败时 _save_panel 内回退 snappy
_COMPRESSION_CANDIDATES = ("zstd", "snappy")


def _cache_disabled() -> bool:
    """环境变量 FACTOR_CACHE_DISABLE=1 时跳过缓存（强制重算）。"""
    return os.getenv("FACTOR_CACHE_DISABLE", "0") == "1"


def _sanitize_name(name: str) -> str:
    """因子名（可能含中文/特殊字符）→ ASCII 安全文件名片段。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _cache_paths(name: str) -> tuple[Path, Path]:
    hex12 = _sanitize_name(name)
    base = FACTOR_CACHE_DIR / f"factor_panel_{hex12}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


# ══════════════════════════════════════════════════════════════════════════════
# 输入数据指纹
# ══════════════════════════════════════════════════════════════════════════════

def _index_ends(obj: pd.Index | pd.Series | pd.DataFrame) -> tuple[str | None, str | None]:
    """取 index 首尾的稳定字符串表示。"""
    try:
        idx = obj.index if not isinstance(obj, pd.Index) else obj
        first, last = idx[0], idx[-1]
        first_s = first.isoformat() if hasattr(first, "isoformat") else str(first)
        last_s = last.isoformat() if hasattr(last, "isoformat") else str(last)
        return first_s, last_s
    except Exception:
        return None, None


def _df_signature(df: pd.DataFrame | None) -> dict | None:
    """DataFrame 的轻量指纹：shape + index 首尾 + 列数。

    不对全量数据做 hash（太慢），但足以感知数据追加 / 重采样 / 列变动。
    """
    if df is None:
        return None
    first_s, last_s = _index_ends(df)
    return {
        "shape": list(df.shape),
        "index_first": first_s,
        "index_last": last_s,
    }


def _normalize_industry_id(industry_map: Any) -> pd.Series | None:
    """把 IC 的 DataFrame / ML 的 Series 统一成一维行业标识（优先 sw_l2）。"""
    if industry_map is None:
        return None
    if isinstance(industry_map, pd.Series):
        return industry_map
    if isinstance(industry_map, pd.DataFrame):
        if industry_map.empty:
            return pd.Series(dtype=object)
        if "sw_l2" in industry_map.columns:
            return industry_map["sw_l2"]
        return industry_map.iloc[:, 0]
    return None


def _industry_map_signature(industry_map: Any) -> dict | None:
    """industry_map 指纹：只看稳定的一维标识，避免 IC/ML 形态分叉。

    IC 路径常传 ``industry_map.parquet`` 整表 DataFrame（shape ``[N, 3]``），
    ML / ``run.py`` 常先取 ``sw_l2`` Series（shape ``[N]``）。二者语义相同，
    若用 ``_df_signature`` 直接签会因 shape 维数不同导致面板缓存假 MISS。
    """
    if industry_map is None:
        return None
    s = _normalize_industry_id(industry_map)
    if s is None:
        return {"type": type(industry_map).__name__}
    first_s, last_s = _index_ends(s)
    return {
        "shape": [int(len(s))],
        "index_first": first_s,
        "index_last": last_s,
    }


def _financial_signature(fin: pd.DataFrame | None) -> dict | None:
    """财务长表指纹：shape + 列集 hash + 末报告期。"""
    if fin is None or fin.empty:
        return None
    cols = sorted(fin.columns.tolist())
    cols_hash = hashlib.md5("|".join(cols).encode("utf-8")).hexdigest()[:12]
    last_date = None
    if "trade_date" in fin.columns:
        try:
            last_date = str(pd.to_datetime(fin["trade_date"]).max().date())
        except Exception:
            last_date = None
    return {
        "shape": list(fin.shape),
        "cols_hash": cols_hash,
        "last_trade_date": last_date,
    }


def _masks_signature(masks: dict | None) -> dict | None:
    """masks dict 指纹：键集 + 每个值的 shape。"""
    if masks is None:
        return None
    out = {}
    for k, v in sorted(masks.items()):
        if isinstance(v, pd.DataFrame):
            out[k] = {"shape": list(v.shape)}
        elif v is None:
            out[k] = None
        else:
            out[k] = {"type": type(v).__name__}
    return out


# 缓存指纹需采集的输入字段（与 compute_single_factor / iter_factor_registry 形参对齐）
# industry_map 单独走 _industry_map_signature（见下），不在此列表。
_SIG_DF_FIELDS = (
    "prices", "prices_raw", "volume", "amount", "open_", "high", "low",
    "clean_ret", "market_prices", "margin", "moneyflow",
    "northbound", "institution", "circ_mv", "total_mv",
)
_SIG_ALL_INPUT_FIELDS = _SIG_DF_FIELDS + ("industry_map",)


def build_input_signature(kwargs: dict) -> dict:
    """从 compute_single_factor 的 kwargs 构建输入数据指纹。"""
    sig: dict[str, Any] = {}
    for f in _SIG_DF_FIELDS:
        sig[f] = _df_signature(kwargs.get(f))
    sig["industry_map"] = _industry_map_signature(kwargs.get("industry_map"))
    sig["financial"] = _financial_signature(kwargs.get("financial"))
    sig["masks"] = _masks_signature(kwargs.get("masks"))
    sig["walk_forward_hmm"] = bool(kwargs.get("walk_forward_hmm", False))
    sig["include_regime"] = bool(kwargs.get("include_regime", False))
    sig["factor_lib_version"] = _FACTOR_LIB_VERSION
    return sig


def _signature_matches(meta: dict, current_sig: dict) -> bool:
    """比对 sidecar meta 与当前输入指纹；任一字段不匹配即视为失效。"""
    if meta is None:
        return False
    if meta.get("factor_lib_version") != current_sig["factor_lib_version"]:
        return False
    if meta.get("walk_forward_hmm") != current_sig["walk_forward_hmm"]:
        return False
    if meta.get("include_regime") != current_sig["include_regime"]:
        return False
    for f in _SIG_ALL_INPUT_FIELDS:
        if meta.get(f) != current_sig[f]:
            return False
    if meta.get("financial") != current_sig["financial"]:
        return False
    if meta.get("masks") != current_sig["masks"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 落盘 / 读盘
# ══════════════════════════════════════════════════════════════════════════════

def _save_panel(name: str, panel: pd.DataFrame, signature: dict) -> None:
    parquet_path, meta_path = _cache_paths(name)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再 rename，避免并发 / 中断产生半写文件
    tmp_pq = parquet_path.with_suffix(".parquet.tmp")
    tmp_meta = meta_path.with_suffix(".meta.json.tmp")
    try:
        saved = False
        last_err: Exception | None = None
        for comp in _COMPRESSION_CANDIDATES:
            try:
                panel.to_parquet(tmp_pq, compression=comp)
                used_comp = comp
                saved = True
                break
            except Exception as e:
                last_err = e
                continue
        if not saved:
            raise last_err or RuntimeError("parquet 写盘失败")
        meta = {
            "name": name,
            "signature": signature,
            "factor_lib_version": _FACTOR_LIB_VERSION,
            "walk_forward_hmm": signature["walk_forward_hmm"],
            "include_regime": signature["include_regime"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "compression": used_comp,
            "panel_shape": list(panel.shape),
        }
        # 把各字段指纹也平铺进 meta，便于 _signature_matches 直接读
        for f in _SIG_ALL_INPUT_FIELDS:
            meta[f] = signature[f]
        meta["financial"] = signature["financial"]
        meta["masks"] = signature["masks"]
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_pq, parquet_path)
        os.replace(tmp_meta, meta_path)
    finally:
        for tmp in (tmp_pq, tmp_meta):
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def _load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"因子缓存 meta 读取失败 ({meta_path.name}): {e}")
        return None


def _load_panel(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path).astype(np.float32, copy=False)


# ══════════════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════════════

def compute_single_factor_cached(
    name: str,
    compute_fn: Callable[[str], pd.DataFrame | None],
    signature: dict,
) -> pd.DataFrame | None:
    """
    带磁盘缓存的 ``compute_single_factor`` 包装。

    参数
    ----
    name        : 因子名
    compute_fn  : 缓存未命中时调用的真实计算闭包，签名 ``f(name) -> panel | None``
    signature   : ``build_input_signature(kwargs)`` 的结果（调用方预算好，
                  避免同一批因子重复构建指纹）

    返回 DataFrame（float32，date×code）或 None（因子数据源缺失）。
    """
    if _cache_disabled():
        return compute_fn(name)

    parquet_path, meta_path = _cache_paths(name)
    meta = _load_meta(meta_path)
    if meta is not None and _signature_matches(meta, signature) and parquet_path.exists():
        try:
            panel = _load_panel(parquet_path)
            logger.debug(f"因子缓存命中: {name} ({parquet_path.name})")
            return panel
        except Exception as e:
            logger.warning(f"因子缓存读取失败，将重算: {name} ({e})")

    # 缓存未命中 / 失效 → 重算
    t0 = time.perf_counter()
    panel = compute_fn(name)
    if panel is None:
        return None
    # 统一 float32 后落盘
    panel = panel.astype(np.float32, copy=False)
    try:
        _save_panel(name, panel, signature)
        dt = time.perf_counter() - t0
        logger.debug(f"因子缓存落盘: {name} shape={panel.shape} ({dt:.1f}s)")
    except Exception as e:
        # 落盘失败不应阻断计算结果返回
        logger.warning(f"因子缓存落盘失败 ({name}): {e}")
    return panel


def clear_factor_cache(names: list | set | None = None) -> int:
    """
    清除因子面板磁盘缓存。

    参数
    ----
    names : 指定因子名子集；None 表示清全部。

    返回删除的缓存条目数。
    """
    if not FACTOR_CACHE_DIR.exists():
        return 0
    removed = 0
    if names is None:
        for f in FACTOR_CACHE_DIR.glob("factor_panel_*"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    for name in names:
        parquet_path, meta_path = _cache_paths(name)
        for p in (parquet_path, meta_path):
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def list_cached_factors() -> list[str]:
    """返回当前缓存目录中所有可恢复的因子名（读 sidecar meta）。"""
    if not FACTOR_CACHE_DIR.exists():
        return []
    names: list[str] = []
    for meta_path in FACTOR_CACHE_DIR.glob("factor_panel_*.meta.json"):
        meta = _load_meta(meta_path)
        if meta and "name" in meta:
            names.append(meta["name"])
    return names


def probe_factor_cache(
    names: list[str] | set[str],
    signature: dict,
) -> tuple[list[str], list[str]]:
    """按当前输入指纹探测因子面板缓存，返回 (hits, misses)。

    供 IC CLI 在算 IC 前打印 HIT/MISS；universe mask 不进入 signature，
    换宇宙不应改变本结果。
    """
    if _cache_disabled():
        return [], list(names)
    hits: list[str] = []
    misses: list[str] = []
    for name in names:
        parquet_path, meta_path = _cache_paths(name)
        meta = _load_meta(meta_path)
        if (
            meta is not None
            and _signature_matches(meta, signature)
            and parquet_path.exists()
        ):
            hits.append(name)
        else:
            misses.append(name)
    return hits, misses


# ══════════════════════════════════════════════════════════════════════════════
# 数据集级 registry 缓存（strategies/ml.py 用）
# ── 与上面单因子面板缓存互补：这里缓存「整批因子 registry dict」，
#    按 (hold_period, rebalance_freq, factor_whitelist, start, end) 键，
#    使 ML 训练管线可 --skip-factor-build 直接载入已构建的因子集。
# ══════════════════════════════════════════════════════════════════════════════
import pickle as _pickle

_REGISTRY_CACHE_DIR: Path = PROCESSED_DIR / "registry_cache"


def factor_cache_path(
    hold_period: int,
    rebalance_freq: str,
    factor_whitelist: list | None,
    start,
    end,
    include_regime: bool = False,
) -> Path:
    """整 registry 缓存路径：参数组合 hash → 单个 .pkl 文件。

    include_regime 进 key（历史兼容）；市场/HMM 广播已退役，默认 False。
    """
    wl = sorted(factor_whitelist) if factor_whitelist else None
    key = f"h{hold_period}|{rebalance_freq}|{start}|{end}|wl={wl}|reg={int(include_regime)}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    _REGISTRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _REGISTRY_CACHE_DIR / f"registry_{h}.pkl"


def cache_exists(path: Path) -> bool:
    return Path(path).exists()


def load_factor_panel(path: Path) -> dict:
    """从盘载入整 registry dict[str, DataFrame]。"""
    with open(path, "rb") as f:
        return _pickle.load(f)


def save_factor_panel(
    path: Path,
    registry: dict,
    hold_period: int | None = None,
    rebalance_freq: str | None = None,
    factor_whitelist: list | None = None,
    start=None,
    end=None,
    include_regime: bool = False,
) -> None:
    """落盘整 registry dict。元数据作为 sidecar .meta.json 便于审计。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        _pickle.dump(registry, f, protocol=_pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    meta = {
        "hold_period": hold_period,
        "rebalance_freq": rebalance_freq,
        "factor_whitelist": sorted(factor_whitelist) if factor_whitelist else None,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "n_factors": len(registry),
        "include_regime": include_regime,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
