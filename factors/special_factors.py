"""
factors/special_factors.py — 统一「特殊因子」注入

特殊因子：可计算、可注册，但**不要求**出现在 IC 筛选后的 YAML 白名单里；
在白名单过滤之后 post-merge 进 ML 数据集（与历史 event overlay 同口径）。
默认 ``skip_neutralize=True``（``--feature-neutralize`` 时豁免 Barra+行业残差化）。

**dynamic 模式禁止注入**（``build_factor_dataset(deny_special_inject=True)`` /
``run.py --mode dynamic`` 会忽略并 warning）。

IC 筛选路径不变：不会自动把 specials 写进 YAML；本机制仅用于训练 / 打分注入。

Packs
-----
- ``event``：``EVENT_OVERLAY_FACTOR_NAMES`` / ``get_event_overlay_factors``
- ``size``：``SIZE_ALPHA_FACTOR_NAMES`` / ``get_size_alpha_factors``
- ``sparse``：语义稀疏池（龙虎榜/涨跌停/开板/解禁/高管/大宗/业绩预告等），
  IC 稀疏轨道入选名见 ``selected_factors_h*.json`` 的 ``factors_sparse``；
  注入时做方差对齐（供 ridge L2 不至于系统性压死稀疏列）。树模型无需额外处理。

CLI：``--special-factors event,size,sparse``（别名 ``--inject-factors``）；
``--event-overlay`` 为 deprecated alias → ``event``。
详见 ``docs/SPECIAL_FACTORS.md``。
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from loguru import logger

from config.settings import SPARSE_VARIANCE_ALIGN_STD
from factors.factor import EVENT_OVERLAY_FACTOR_NAMES, get_event_overlay_factors
from factors.factor_size_alpha import SIZE_ALPHA_FACTOR_NAMES, get_size_alpha_factors
from factors.sparse_factors import (
    SPARSE_FACTOR_NAMES,
    compute_sparse_factors,
    variance_align_panel,
)


ComputeFn = Callable[..., dict[str, pd.DataFrame]]


@dataclass(frozen=True)
class SpecialFactorPack:
    """一个可注入的特殊因子包。"""

    name: str
    factor_names: frozenset[str]
    compute: ComputeFn
    aliases: frozenset[str] = field(default_factory=frozenset)
    skip_neutralize: bool = True
    description: str = ""
    # sparse pack：注入后对面板做方差对齐（ridge）；其他 pack 默认 False
    variance_align: bool = False


def _compute_event(
    prices: pd.DataFrame,
    factor_names: set[str] | None = None,
    **_kwargs,
) -> dict[str, pd.DataFrame]:
    return get_event_overlay_factors(prices, factor_names=factor_names)


def _compute_size(
    prices: pd.DataFrame,
    factor_names: set[str] | None = None,
    financial: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    **_kwargs,
) -> dict[str, pd.DataFrame]:
    return get_size_alpha_factors(
        prices,
        financial=financial,
        circ_mv=circ_mv,
        total_mv=total_mv,
        clean_ret=clean_ret,
        factor_names=factor_names,
    )


def _compute_sparse(
    prices: pd.DataFrame,
    factor_names: set[str] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    return compute_sparse_factors(prices, factor_names=factor_names, **kwargs)


SPECIAL_FACTOR_PACKS: dict[str, SpecialFactorPack] = {
    "event": SpecialFactorPack(
        name="event",
        factor_names=frozenset(EVENT_OVERLAY_FACTOR_NAMES),
        compute=_compute_event,
        aliases=frozenset({
            "event", "events", "event_overlay", "overlay", "yjyg",
        }),
        skip_neutralize=True,
        description="事件 overlay（业绩预告等稀疏信号）",
    ),
    "size": SpecialFactorPack(
        name="size",
        factor_names=frozenset(SIZE_ALPHA_FACTOR_NAMES),
        compute=_compute_size,
        aliases=frozenset({
            "size", "size_alpha", "mcap", "市值", "市值alpha",
        }),
        skip_neutralize=True,
        description="市值 alpha（对数市值 / 分位 / 风格对齐）",
    ),
    "sparse": SpecialFactorPack(
        name="sparse",
        factor_names=frozenset(SPARSE_FACTOR_NAMES),
        compute=_compute_sparse,
        aliases=frozenset({
            "sparse", "sparse_factors", "稀疏", "稀疏因子",
        }),
        skip_neutralize=True,
        variance_align=True,
        description="语义稀疏因子池（龙虎榜/涨跌停/开板/解禁/高管/大宗/业绩预告等）",
    ),
}

# alias / 因子名 → pack canonical name
_ALIAS_TO_PACK: dict[str, str] = {}
_FACTOR_TO_PACK: dict[str, str] = {}
for _pack in SPECIAL_FACTOR_PACKS.values():
    for _a in _pack.aliases:
        _ALIAS_TO_PACK[_a.lower()] = _pack.name
    _ALIAS_TO_PACK[_pack.name.lower()] = _pack.name
    for _fname in _pack.factor_names:
        _FACTOR_TO_PACK[_fname] = _pack.name
        _ALIAS_TO_PACK[_fname.lower()] = _pack.name


@dataclass(frozen=True)
class SpecialFactorRequest:
    """解析后的特殊因子注入请求。"""

    packs: tuple[str, ...] = ()
    # None = 注入各 pack 全部因子；否则仅注入交集内的名字
    names: frozenset[str] | None = None

    def __bool__(self) -> bool:
        return bool(self.packs)

    def tag_suffix(self) -> str:
        """输出目录 tag 后缀，如 ``_event`` / ``_event_size``。"""
        if not self.packs:
            return ""
        return "_" + "_".join(self.packs)

    def factor_names_for_pack(self, pack_name: str) -> frozenset[str]:
        pack = SPECIAL_FACTOR_PACKS[pack_name]
        if self.names is None:
            return pack.factor_names
        return frozenset(pack.factor_names & self.names)

    def all_factor_names(self) -> frozenset[str]:
        out: set[str] = set()
        for p in self.packs:
            out |= self.factor_names_for_pack(p)
        return frozenset(out)


def list_pack_names() -> list[str]:
    return sorted(SPECIAL_FACTOR_PACKS)


def all_skip_neutralize_names() -> frozenset[str]:
    """所有 pack 中 ``skip_neutralize=True`` 的因子名（含未注入时 YAML 内的同名因子）。"""
    names: set[str] = set()
    for pack in SPECIAL_FACTOR_PACKS.values():
        if pack.skip_neutralize:
            names |= pack.factor_names
    return frozenset(names)


def _tokenize(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        raw = [t.strip() for t in spec.replace(";", ",").split(",")]
        return [t for t in raw if t]
    out: list[str] = []
    for item in spec:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if "," in s or ";" in s:
            out.extend(_tokenize(s))
        else:
            out.append(s)
    return out


def load_sparse_names_from_ic_json(
    path: str | Path,
) -> list[str]:
    """从 IC ``selected_factors_h*.json`` 读取 ``factors_sparse`` 列表。"""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    names = data.get("factors_sparse") or []
    return [n for n in names if n in SPARSE_FACTOR_NAMES]


def resolve_special_factors(
    spec: str | Sequence[str] | None = None,
    *,
    event_overlay: bool = False,
    warn_deprecated: bool = True,
    sparse_from_ic: str | Path | None = None,
) -> SpecialFactorRequest:
    """解析 CLI / API 规格为 ``SpecialFactorRequest``。

    接受：
    - pack 名 / 别名：``event``、``size``、``sparse``、``event_overlay``、``市值`` …
    - 包内具体因子名：``对数市值`` → 启用 ``size`` 且仅注入该名
      （若同请求里还有 pack 级 token，该 pack 仍全量注入）
    - ``event_overlay=True``（deprecated）→ 并入 ``event`` pack（全量）
    - ``sparse_from_ic``：IC JSON 路径；启用 ``sparse`` pack 且仅注入 JSON 中
      ``factors_sparse`` 名单（生产推荐，避免全量语义池）
    """
    tokens = _tokenize(spec)
    pack_order: list[str] = []
    seen_packs: set[str] = set()
    pack_level: set[str] = set()
    explicit_names: set[str] = set()
    unknown: list[str] = []

    def _add_pack(pname: str) -> None:
        if pname not in seen_packs:
            seen_packs.add(pname)
            pack_order.append(pname)

    for tok in tokens:
        if tok in _FACTOR_TO_PACK:
            pname = _FACTOR_TO_PACK[tok]
            _add_pack(pname)
            explicit_names.add(tok)
            continue
        key = tok.lower()
        if key in _ALIAS_TO_PACK:
            pname = _ALIAS_TO_PACK[key]
            _add_pack(pname)
            pack_level.add(pname)
            continue
        unknown.append(tok)

    if unknown:
        known = ", ".join(list_pack_names())
        raise ValueError(
            f"未知 special-factors 项: {unknown}；"
            f"可选 pack: {known}，或 pack 内因子名"
            f"（如 {sorted(SIZE_ALPHA_FACTOR_NAMES)[:2]}…）"
        )

    if event_overlay:
        if warn_deprecated:
            warnings.warn(
                "--event-overlay / event_overlay=True 已弃用，请改用 "
                "--special-factors event（或 --inject-factors event）。"
                "当前已自动映射为 event pack。",
                DeprecationWarning,
                stacklevel=2,
            )
        _add_pack("event")
        pack_level.add("event")

    if sparse_from_ic is not None:
        ic_names = load_sparse_names_from_ic_json(sparse_from_ic)
        if not ic_names:
            logger.warning(
                f"sparse_from_ic={sparse_from_ic} 无 factors_sparse，跳过 sparse 注入"
            )
        else:
            _add_pack("sparse")
            explicit_names.update(ic_names)
            # IC 白名单优先：勿让 pack-level ``sparse`` 再并入全量语义池
            pack_level.discard("sparse")
            logger.info(
                f"sparse_from_ic={sparse_from_ic}: 将注入 factors_sparse "
                f"{len(ic_names)} 个 → {ic_names}"
            )

    if not explicit_names:
        return SpecialFactorRequest(packs=tuple(pack_order), names=None)

    names: set[str] = set(explicit_names)
    for p in pack_level:
        names |= SPECIAL_FACTOR_PACKS[p].factor_names
    return SpecialFactorRequest(packs=tuple(pack_order), names=frozenset(names))


def inject_special_factors(
    registry: dict,
    request: SpecialFactorRequest | None,
    *,
    prices: pd.DataFrame,
    financial: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    variance_align_std: float | None = None,
    **compute_kwargs,
) -> list[str]:
    """白名单过滤之后，把特殊因子面板 merge 进 ``registry``。返回新注入的名字。

    ``sparse`` pack 默认做方差对齐：``x' = x * (target_std / std(x))``，
    使稀疏列训练方差接近稠密因子截面 z-score 后的单位方差，避免 Ridge L2
    系统性压死稀疏系数。树模型无额外处理（对齐亦不伤害分裂）。
    """
    if not request:
        return []

    target_std = (
        SPARSE_VARIANCE_ALIGN_STD if variance_align_std is None else variance_align_std
    )
    merged: list[str] = []
    for pack_name in request.packs:
        pack = SPECIAL_FACTOR_PACKS[pack_name]
        want = set(request.factor_names_for_pack(pack_name))
        if not want:
            continue
        try:
            panels = pack.compute(
                prices=prices,
                factor_names=want,
                financial=financial,
                circ_mv=circ_mv,
                total_mv=total_mv,
                clean_ret=clean_ret,
                **compute_kwargs,
            )
        except Exception as e:
            logger.warning(f"special_factors[{pack_name}] 计算失败: {e}")
            continue

        pack_merged: list[str] = []
        already: list[str] = []
        for name, panel in panels.items():
            if panel is None:
                continue
            if name not in want:
                continue
            if name in registry:
                already.append(name)
                continue
            if pack.variance_align:
                panel = variance_align_panel(panel, target_std=target_std)
            registry[name] = panel
            pack_merged.append(name)
            merged.append(name)

        missing = sorted(want - set(pack_merged) - set(already))
        if pack_merged:
            align_note = (
                f"，variance_align→std≈{target_std}" if pack.variance_align else ""
            )
            logger.info(
                f"special_factors[{pack_name}]: 注入 {len(pack_merged)}/{len(want)} 个因子"
                f"（skip_neutralize={pack.skip_neutralize}{align_note}）→ {pack_merged}；"
                f"registry 共 {len(registry)} 个特征"
            )
        else:
            logger.warning(
                f"special_factors[{pack_name}]=on 但未注入任何因子"
                f"（期望 {sorted(want)}；可能已在 registry 或数据缺失）"
            )
        if already:
            logger.info(
                f"special_factors[{pack_name}]: {len(already)} 个已在 registry，跳过 → {already}"
            )
        if missing:
            logger.warning(
                f"special_factors[{pack_name}]: {len(missing)} 个未产出"
                f"（不在 compute 结果中，常见原因：缺 masks/数据）→ {missing}"
            )
    return merged


def should_skip_neutralize(name: str) -> bool:
    """``feature_neutralize`` 时是否豁免该特征（Barra_ / 各 special pack）。"""
    if name.startswith("Barra_"):
        return True
    return name in all_skip_neutralize_names()
