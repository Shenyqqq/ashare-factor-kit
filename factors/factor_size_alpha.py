"""
factors/factor_size_alpha.py — 市值相关 alpha（正常因子池，非仓位体制）

显式 Size / 风格对齐信号，供 IC 筛选与 YAML 白名单选用。
**不做** ``--regime-cs`` / ``市场*`` 注入；也不自动塞进生产白名单。

方向约定（项目统一「越高越好」）
--------------------------------
- ``对数市值`` / ``市值分位``：小市值高分（A 股小盘溢价口径；取负 / 反分位）。
- ``市值风格对齐_{w}d``：``z_size × (−SMB)``——小盘体制抬升时小票得分高。

中性化
------
本族与 Barra_Size 高度共线；``--feature-neutralize`` 时必须豁免
（见 ``SIZE_ALPHA_FACTOR_NAMES`` / ``factors.special_factors`` size pack），
否则残差≈噪声。也可经 ``--special-factors size`` 绕过 IC YAML 强制注入训练。

数据
----
优先 ``circ_mv``（流通市值宽表），其次 ``total_mv``，再次 financial
``total_mv``/``total_assets``；缺失时该因子跳过。面板可由调用方传入，
或自动读 ``data/raw/circ_mv.parquet`` / ``total_mv.parquet``
（东财 ``download_stock_value_em`` 主路径；缺则回退 ``*_computed``）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import RAW_DIR
from factors.factor import _normalize, cross_sectional_zscore


SIZE_ALPHA_FACTOR_NAMES = frozenset({
    "对数市值",
    "市值分位",
    "市值风格对齐_20d",
    "市值风格对齐_60d",
})


def _load_mv_parquet(prices: pd.DataFrame, fname: str) -> pd.DataFrame | None:
    from data.mv_panels import resolve_mv_path

    kind = "circ_mv" if "circ" in fname else "total_mv"
    path = resolve_mv_path(kind, allow_computed_fallback=True)
    if path is None:
        return None
    try:
        mv = pd.read_parquet(path)
        mv.index = pd.to_datetime(mv.index)
        return mv.reindex(index=prices.index, columns=prices.columns)
    except Exception as e:
        logger.warning(f"size_alpha: 读取 {path.name} 失败: {e}")
        return None


def resolve_mcap_panel(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """流通/总市值宽表（元）；优先 circ_mv → total_mv → financial。"""
    if circ_mv is not None and not circ_mv.empty:
        return circ_mv.reindex(index=prices.index, columns=prices.columns)
    loaded = _load_mv_parquet(prices, "circ_mv.parquet")
    if loaded is not None:
        return loaded
    if total_mv is not None and not total_mv.empty:
        return total_mv.reindex(index=prices.index, columns=prices.columns)
    loaded = _load_mv_parquet(prices, "total_mv.parquet")
    if loaded is not None:
        return loaded
    if financial is None or financial.empty:
        return None
    from factors.barra_risk import _pivot_ffill
    for col in ("total_mv", "total_assets"):
        if col not in financial.columns:
            continue
        try:
            mv = _pivot_ffill(financial, col, prices.index)
            return mv.reindex(columns=prices.columns)
        except Exception as e:
            logger.warning(f"size_alpha: financial.{col} 透视失败: {e}")
    return None


def _log_size_panel(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    mv = resolve_mcap_panel(prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial)
    if mv is None:
        return None
    return np.log(mv.replace(0, np.nan).abs())


def _stock_trailing_ret(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None,
    window: int,
) -> pd.DataFrame:
    """窗口累计收益；优先 clean_ret（涨跌停日 NaN）。"""
    min_p = max(5, window // 2)
    if clean_ret is not None:
        ret = clean_ret.reindex(index=prices.index, columns=prices.columns)
        log1p = np.log1p(ret.clip(lower=-0.999999))
        cum = log1p.rolling(window, min_periods=min_p).sum()
        return np.expm1(cum)
    return prices.pct_change(window)


def _sleeve_spread(
    trail_ret: pd.DataFrame,
    char: pd.DataFrame,
    bottom_pct: float = 0.3,
    top_pct: float = 0.3,
) -> pd.Series:
    """bottom 组均收益 − top 组均收益（SMB：小 − 大）。"""
    common = trail_ret.columns.intersection(char.columns)
    r = trail_ret[common]
    c = char.reindex(index=r.index, columns=common)
    ranks = c.rank(axis=1, pct=True, method="average")
    bottom = r.where(ranks <= bottom_pct)
    top = r.where(ranks >= (1.0 - top_pct))
    return bottom.mean(axis=1) - top.mean(axis=1)


def factor_log_mcap(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    对数市值（取负）：``−log(市值)`` → 小市值高分。

    与财务因子 ``规模``（−log 总资产）互补；本因子用日频市值面板。
    """
    log_mv = _log_size_panel(prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial)
    if log_mv is None:
        return None
    return _normalize(-log_mv)


def factor_mcap_percentile(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    市值分位（取反）：截面 pct-rank(市值) 后 ``1 − rank`` → 小市值高分。

    相对 ``对数市值`` 对极端大/小盘更稳健（秩变换）。
    """
    mv = resolve_mcap_panel(prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial)
    if mv is None:
        return None
    rank = mv.rank(axis=1, pct=True, method="average")
    return _normalize(1.0 - rank)


def factor_size_style_align(
    prices: pd.DataFrame,
    financial: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    window: int = 20,
) -> pd.DataFrame | None:
    """
    市值风格对齐_{window}d = ``z_size × (−SMB)``。

    ``z_size`` = CS-z(log 市值)，越大盘越高；
    ``SMB`` = R_small − R_large（市值最低 vs 最高 30% trailing 均收益）。
    小盘强 + 小市值 → 高分。
    """
    log_mv = _log_size_panel(prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial)
    if log_mv is None:
        return None
    trail = _stock_trailing_ret(prices, clean_ret, window)
    smb = _sleeve_spread(trail, log_mv, bottom_pct=0.3, top_pct=0.3)
    size_z = cross_sectional_zscore(log_mv)
    raw = size_z.mul((-smb).reindex(size_z.index), axis=0)
    return _normalize(raw)


def mcap_data_available(
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> bool:
    """轻量可用性检查（不读全表；parquet 仅看文件是否存在）。"""
    if circ_mv is not None and not circ_mv.empty:
        return True
    if total_mv is not None and not total_mv.empty:
        return True
    from data.mv_panels import resolve_mv_path

    if resolve_mv_path("circ_mv") is not None or resolve_mv_path("total_mv") is not None:
        return True
    if financial is not None and not financial.empty:
        cols = set(financial.columns)
        if "total_mv" in cols or "total_assets" in cols:
            return True
    return False


def get_size_alpha_factors(
    prices: pd.DataFrame,
    financial: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    factor_names: list | set | None = None,
) -> dict[str, pd.DataFrame]:
    """计算市值 alpha 子集；返回 {name: panel}。"""
    want = set(factor_names) if factor_names is not None else set(SIZE_ALPHA_FACTOR_NAMES)
    want &= SIZE_ALPHA_FACTOR_NAMES
    out: dict[str, pd.DataFrame] = {}

    def _put(name: str, panel: pd.DataFrame | None) -> None:
        if panel is not None and name in want:
            out[name] = panel

    if "对数市值" in want:
        _put("对数市值", factor_log_mcap(
            prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
        ))
    if "市值分位" in want:
        _put("市值分位", factor_mcap_percentile(
            prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
        ))
    if "市值风格对齐_20d" in want:
        _put("市值风格对齐_20d", factor_size_style_align(
            prices, financial=financial, circ_mv=circ_mv, total_mv=total_mv,
            clean_ret=clean_ret, window=20,
        ))
    if "市值风格对齐_60d" in want:
        _put("市值风格对齐_60d", factor_size_style_align(
            prices, financial=financial, circ_mv=circ_mv, total_mv=total_mv,
            clean_ret=clean_ret, window=60,
        ))
    if out:
        logger.info(f"市值 alpha 因子就绪: {len(out)} 个 → {sorted(out)}")
    return out
