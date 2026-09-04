"""
factors/sparse_factors.py — 语义稀疏因子名集合与工具

稀疏因子（sparse factors）：截面上多数股票无信号（大量 NaN），或事件驱动
（仅公告日/上榜日有值）。不适合与稠密量价/财务因子同一套严格 IC 门槛混筛，
也不应盲注入 dynamic（ICIR 动态加权）轨道。

语义池来源（与 ``factors.factor`` / ``factor_smallcap`` / ``factor_limit`` 对齐）：
  龙虎榜*、涨跌停*、开板*、解禁*、高管*、大宗*、业绩预告* 等。

IC 筛选：见 ``research.ic.selection`` 稀疏轨道（同向 IC 胜率 + 触发日截面胜率，
均按 ``sign(mean_IC)`` 对齐；无 t/NW-t/FDR 硬门槛；IC/ICIR 为软参考）。
训练注入：经 ``factors.special_factors`` 的 ``sparse`` pack（``--special-factors sparse``），
仅建议给 ridge 等线性模型；注入时做方差对齐（variance alignment）。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from factors.factor import EVENT_OVERLAY_FACTOR_NAMES

# 涨跌停 / 开板（factor_limit）
_SPARSE_LIMIT = frozenset({
    "涨停强度_20d",
    "跌停弱势_20d",
    "连板数",
    "涨跌停净强度_20d",
    "涨跌停状态",
    "开板反转_5d",
})

# 龙虎榜 / 解禁 / 高管 / 大宗（factor_smallcap 子集；股东户数等较稠密，不入池）
_SPARSE_SMALLCAP = frozenset({
    "龙虎榜上榜次数_20d",
    "龙虎榜净买额_20d",
    "龙虎榜连续上榜",
    "未来60日解禁市值占比",
    "未来30日解禁次数",
    "高管净增持额_60d",
    "高管增持次数_60d",
    "高管减持次数_60d",
    "增减持比_60d",
    "大宗交易折价率_20d",
    "大宗交易频次_20d",
})

# A 股特色稀疏增量（评级/研报/机构席位/回购/大宗质量/解禁等；两融截面不入池）
_SPARSE_ASHARE = frozenset({
    "评级上修_20d",
    "研报EPS上修次数_20d",
    "研报预期差",
    "龙虎榜机构净买入_20d",
    "龙虎榜机构买入强度_20d",
    "股份回购强度_60d",
    "大宗折价席位质量_20d",
    "板块资金流拥挤_5d",
    "龙虎榜净买占比_20d",
    "目标价上行空间",
    "研报覆盖热度_20d",
    "回购完成进度_60d",
    "解禁流动性压力_60d",
    "大宗机构接盘_20d",
    "评级下调规避_20d",
    # 两融日频截面（融资买入/净买入/融券卖出/余额市值比）走稠密轨，不入 sparse
    "龙虎榜涨幅上榜_20d",
    "龙虎榜换手上榜_20d",
    "龙虎榜跌幅上榜规避_20d",
    "解禁定增压力_60d",
    "解禁激励压力_60d",
    "转债转股稀释_60d",
    "激励行权稀释_60d",
    "限售上市供给_60d",
    "研报EPS斜率",
    "研报EPS分歧度",
    "大宗卖方机构抛压_20d",
    "大宗折溢价波动_20d",
})

# 事件 overlay（默认不进 get_factor_names 截面枚举）
_SPARSE_EVENT = frozenset(EVENT_OVERLAY_FACTOR_NAMES)

SPARSE_FACTOR_NAMES: frozenset[str] = frozenset(
    _SPARSE_LIMIT | _SPARSE_SMALLCAP | _SPARSE_ASHARE | _SPARSE_EVENT
)

# 主类别（互斥轨道标签）+ 警示标签（可叠加，不剔除）
CAT_DENSE = "普通因子"
CAT_SPARSE = "稀疏因子"
CAT_EMERGING = "新兴因子"
CAT_DECAYED = "衰减因子"
CAT_REVERSAL = "风格逆转"
# 向后兼容别名（旧「衰减但仍有效」）
CAT_DECAYED_OK = CAT_DECAYED


def is_sparse_factor(name: str) -> bool:
    return name in SPARSE_FACTOR_NAMES


def filter_sparse_names(names: Iterable[str]) -> list[str]:
    return [n for n in names if n in SPARSE_FACTOR_NAMES]


def partition_sparse(
    names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """拆成 (dense_names, sparse_names)，保持输入顺序。"""
    dense: list[str] = []
    sparse: list[str] = []
    for n in names:
        if n in SPARSE_FACTOR_NAMES:
            sparse.append(n)
        else:
            dense.append(n)
    return dense, sparse


def variance_align_panel(
    panel: pd.DataFrame,
    target_std: float = 1.0,
    min_obs: int = 30,
) -> pd.DataFrame:
    """方差对齐（variance alignment）：缩放面板使非 NaN 样本标准差 ≈ ``target_std``。

    公式
    ----
    令 ``σ = std({x_ij | x_ij 有限})``（总体标准差，ddof=0），则::

        x' = x * (target_std / σ)

    稠密因子经截面 winsorize + z-score 后，有值单元格方差约 1；稀疏因子常因
    大量 NaN / 原始事件尺度导致堆叠后列方差 ≪ 1，Ridge 的 L2 惩罚会系统性压掉
    其系数。本函数在注入 ridge 前把稀疏列拉到与稠密因子同量级。

    ``σ`` 过小或有效样本 ``< min_obs`` 时原样返回（避免除零放大噪声）。
    """
    if panel is None or panel.empty or target_std <= 0:
        return panel
    arr = panel.to_numpy(dtype=np.float64, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size < min_obs:
        return panel
    std = float(finite.std(ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        return panel
    scale = target_std / std
    return panel * scale


def compute_sparse_factors(
    prices: pd.DataFrame,
    factor_names: set[str] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """计算稀疏因子面板子集（供 special-factors sparse pack）。"""
    from factors.factor import compute_single_factor, get_event_overlay_factors

    want = set(SPARSE_FACTOR_NAMES) if factor_names is None else (
        set(factor_names) & set(SPARSE_FACTOR_NAMES)
    )
    if not want:
        return {}

    out: dict[str, pd.DataFrame] = {}
    event_want = want & _SPARSE_EVENT
    if event_want:
        out.update(get_event_overlay_factors(prices, factor_names=event_want))

    rest = want - set(out.keys())
    for name in sorted(rest):
        try:
            panel = compute_single_factor(
                name,
                prices=prices,
                financial=kwargs.get("financial"),
                prices_raw=kwargs.get("prices_raw"),
                volume=kwargs.get("volume"),
                amount=kwargs.get("amount"),
                open_=kwargs.get("open_"),
                high=kwargs.get("high"),
                low=kwargs.get("low"),
                clean_ret=kwargs.get("clean_ret"),
                masks=kwargs.get("masks"),
                market_prices=kwargs.get("market_prices"),
                industry_map=kwargs.get("industry_map"),
                margin=kwargs.get("margin"),
                moneyflow=kwargs.get("moneyflow"),
                northbound=kwargs.get("northbound"),
                institution=kwargs.get("institution"),
                circ_mv=kwargs.get("circ_mv"),
                total_mv=kwargs.get("total_mv"),
            )
        except Exception as e:
            logger.warning(f"sparse 因子 {name} 计算失败，跳过: {e}")
            continue
        if panel is not None and not getattr(panel, "empty", True):
            out[name] = panel
        else:
            logger.warning(f"sparse 因子 {name} 返回空面板，跳过")
    missing = want - set(out.keys())
    if missing:
        logger.warning(f"sparse 因子未产出: {sorted(missing)}")
    return out
