"""
factors/factor_alpha158.py — Qlib Alpha158 量价特征集

来源：Qlib Alpha158（公式见 .tmp/alpha158/README.md；脚本依赖 phandas，本模块纯 pandas）
命名：A158_{NAME} 或 A158_{NAME}{N}（滚动窗 N∈{5,10,20,30,60}）

约定：
  - 日收益相关处优先用 clean_ret（CORD/CNT*/SUM*/WVMA 等）
  - VWAP 缺省时用典型价 (H+L+C)/3，并在日志标明
  - 截面标准化由 get_alpha158_factors 出口统一 _normalize
  - 因子方向保持 Qlib 原式（IC 后若需取反再在本文件改）

内存注意
--------
- **禁止** import / registry 构建时预计算全表。
- 默认按 ``factor_names`` 白名单**只算请求的因子**；同窗共享中间量惰性缓存，
  算完一批后缓存可丢（调用方 ``del`` / 分批调用）。
- 可选 ``compute_alpha158_batch``：一次算多列供 IC 分批；registry 仍走单次白名单。
- 单面板约 date×code；全量 153 列同时驻留会 OOM（32GB 机器勿整包物化）。

共 153 个：kbar 9 + price 4 + rolling 28×5（源库 rolling 脚本为 28 类；
README 写「29×5」与目录不一致，以脚本为准）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from loguru import logger

from factors.alpha_ops import (
    compute_vwap,
    corr,
    delay,
    maximum,
    minimum,
    safe_div,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_quantile,
    ts_resi,
    ts_rsquare,
    ts_slope,
    ts_std,
    ts_sum,
)
from factors.factor import _normalize

_WINDOWS = (5, 10, 20, 30, 60)
_EPS = 1e-12
_ROLL_BASES = (
    "ROC", "MA", "BETA", "RSQR", "RESI",
    "STD", "MAX", "MIN", "QTLU", "QTLD", "RSV",
    "IMAX", "IMIN", "IMXD",
    "CORR", "CORD", "CNTP", "CNTN", "CNTD",
    "SUMP", "SUMN", "SUMD",
    "VMA", "VSTD",
    "WVMA", "VSUMP", "VSUMN", "VSUMD",
)
_KBAR_NAMES = (
    "A158_KMID", "A158_KLEN", "A158_KMID2", "A158_KUP", "A158_KUP2",
    "A158_KLOW", "A158_KLOW2", "A158_KSFT", "A158_KSFT2",
    "A158_OPEN0", "A158_HIGH0", "A158_LOW0", "A158_VWAP0",
)

ALPHA158_NAMES: tuple[str, ...] = _KBAR_NAMES + tuple(
    f"A158_{base}{n}" for base in _ROLL_BASES for n in _WINDOWS
)

_ROLL_NAME_RE = re.compile(
    r"^A158_(" + "|".join(_ROLL_BASES) + r")(\d+)$"
)


def _parse_roll_name(name: str) -> tuple[str, int] | None:
    m = _ROLL_NAME_RE.match(name)
    if not m:
        return None
    base, n_s = m.group(1), m.group(2)
    n = int(n_s)
    if n not in _WINDOWS:
        return None
    return base, n


class _A158Ctx:
    """
    惰性中间量上下文：同一次 get_alpha158_factors / batch 调用内共享，
    不跨调用缓存，避免全局大矩阵驻留。
    """

    __slots__ = (
        "close", "open_", "high", "low", "volume", "vwap", "clean_ret", "_c",
    )

    def __init__(self, close, open_, high, low, volume, vwap, clean_ret):
        self.close = close
        self.open_ = open_
        self.high = high
        self.low = low
        self.volume = volume
        self.vwap = vwap
        self.clean_ret = clean_ret
        self._c: dict = {}

    def _get(self, key, factory):
        if key not in self._c:
            self._c[key] = factory()
        return self._c[key]

    def ret(self):
        return self._get("ret", lambda: (
            self.clean_ret if self.clean_ret is not None else self.close.pct_change()
        ))

    def log_vol(self):
        return self._get("log_vol", lambda: np.log(self.volume.replace(0, np.nan) + 1.0))

    def vol_chg(self):
        return self._get("vol_chg", lambda: self.volume / delay(self.volume, 1) - 1.0)

    def log_vol_chg(self):
        return self._get(
            "log_vol_chg",
            lambda: np.log(self.volume / delay(self.volume, 1).replace(0, np.nan) + 1.0),
        )

    def up(self):
        return self._get("up", lambda: (self.ret() > 0).astype(np.float64))

    def down(self):
        return self._get("down", lambda: (self.ret() < 0).astype(np.float64))

    def ret_pos(self):
        return self._get("ret_pos", lambda: self.ret().clip(lower=0))

    def ret_neg(self):
        return self._get("ret_neg", lambda: (-self.ret()).clip(lower=0))

    def abs_ret(self):
        return self._get("abs_ret", lambda: self.ret().abs())

    def abs_ret_vol(self):
        return self._get("abs_ret_vol", lambda: self.abs_ret() * self.volume)

    def vol_pos(self):
        return self._get("vol_pos", lambda: self.vol_chg().clip(lower=0))

    def vol_neg(self):
        return self._get("vol_neg", lambda: (-self.vol_chg()).clip(lower=0))

    def abs_vol_chg(self):
        return self._get("abs_vol_chg", lambda: self.vol_chg().abs())

    def clear(self):
        self._c.clear()


def _compute_kbar(name: str, ctx: _A158Ctx) -> pd.DataFrame | None:
    close, open_, high, low = ctx.close, ctx.open_, ctx.high, ctx.low
    if name == "A158_KMID":
        return (close - open_) / open_
    if name == "A158_KLEN":
        return (high - low) / open_
    if name == "A158_KMID2":
        return safe_div(close - open_, high - low, _EPS)
    if name == "A158_KUP":
        return (high - maximum(open_, close)) / open_
    if name == "A158_KUP2":
        return safe_div(high - maximum(open_, close), high - low, _EPS)
    if name == "A158_KLOW":
        return (minimum(open_, close) - low) / open_
    if name == "A158_KLOW2":
        return safe_div(minimum(open_, close) - low, high - low, _EPS)
    if name == "A158_KSFT":
        return (2.0 * close - high - low) / open_
    if name == "A158_KSFT2":
        return safe_div(2.0 * close - high - low, high - low, _EPS)
    if name == "A158_OPEN0":
        return open_ / close
    if name == "A158_HIGH0":
        return high / close
    if name == "A158_LOW0":
        return low / close
    if name == "A158_VWAP0":
        if ctx.vwap is None:
            return None
        return ctx.vwap / close
    return None


def _compute_rolling(base: str, n: int, ctx: _A158Ctx) -> pd.DataFrame | None:
    close, high, low, volume = ctx.close, ctx.high, ctx.low, ctx.volume
    # 同窗共享：hi_n/lo_n/imax/imin 等按需缓存
    cache_key_hi = ("hi", n)
    cache_key_lo = ("lo", n)

    if base == "ROC":
        return delay(close, n) / close
    if base == "MA":
        return ts_mean(close, n) / close
    if base == "BETA":
        return ts_slope(close, n) / close
    if base == "RSQR":
        return ts_rsquare(close, n)
    if base == "RESI":
        return ts_resi(close, n) / close
    if base == "STD":
        return ts_std(close, n) / close
    if base == "MAX":
        return ts_max(high, n) / close
    if base == "MIN":
        return ts_min(low, n) / close
    if base == "QTLU":
        return ts_quantile(close, n, 0.8) / close
    if base == "QTLD":
        return ts_quantile(close, n, 0.2) / close
    if base == "RSV":
        hi_n = ctx._get(cache_key_hi, lambda: ts_max(high, n))
        lo_n = ctx._get(cache_key_lo, lambda: ts_min(low, n))
        return safe_div(close - lo_n, hi_n - lo_n, _EPS)
    if base == "IMAX":
        return ctx._get(("imax", n), lambda: ts_argmax(high, n) / n)
    if base == "IMIN":
        return ctx._get(("imin", n), lambda: ts_argmin(low, n) / n)
    if base == "IMXD":
        imax = ctx._get(("imax", n), lambda: ts_argmax(high, n) / n)
        imin = ctx._get(("imin", n), lambda: ts_argmin(low, n) / n)
        return imax - imin
    if base == "CORR":
        return corr(close, ctx.log_vol(), n)
    if base == "CORD":
        return corr(1.0 + ctx.ret(), ctx.log_vol_chg(), n)
    if base == "CNTP":
        return ts_mean(ctx.up(), n)
    if base == "CNTN":
        return ts_mean(ctx.down(), n)
    if base == "CNTD":
        return ts_mean(ctx.up(), n) - ts_mean(ctx.down(), n)
    if base == "SUMP":
        sum_abs = ts_sum(ctx.abs_ret(), n)
        return safe_div(ts_sum(ctx.ret_pos(), n), sum_abs, _EPS)
    if base == "SUMN":
        sum_abs = ts_sum(ctx.abs_ret(), n)
        return safe_div(ts_sum(ctx.ret_neg(), n), sum_abs, _EPS)
    if base == "SUMD":
        sum_abs = ts_sum(ctx.abs_ret(), n)
        sp = safe_div(ts_sum(ctx.ret_pos(), n), sum_abs, _EPS)
        sn = safe_div(ts_sum(ctx.ret_neg(), n), sum_abs, _EPS)
        return sp - sn
    if base == "VMA":
        return ts_mean(volume, n) / volume.replace(0, np.nan)
    if base == "VSTD":
        return ts_std(volume, n) / volume.replace(0, np.nan)
    if base == "WVMA":
        arv = ctx.abs_ret_vol()
        return safe_div(ts_std(arv, n), ts_mean(arv, n), _EPS)
    if base == "VSUMP":
        return safe_div(ts_sum(ctx.vol_pos(), n), ts_sum(ctx.abs_vol_chg(), n), _EPS)
    if base == "VSUMN":
        return safe_div(ts_sum(ctx.vol_neg(), n), ts_sum(ctx.abs_vol_chg(), n), _EPS)
    if base == "VSUMD":
        den = ts_sum(ctx.abs_vol_chg(), n)
        sp = safe_div(ts_sum(ctx.vol_pos(), n), den, _EPS)
        sn = safe_div(ts_sum(ctx.vol_neg(), n), den, _EPS)
        return sp - sn
    return None


def _compute_one(name: str, ctx: _A158Ctx) -> pd.DataFrame | None:
    if name in _KBAR_NAMES:
        return _compute_kbar(name, ctx)
    parsed = _parse_roll_name(name)
    if parsed is None:
        return None
    base, n = parsed
    return _compute_rolling(base, n, ctx)


def iter_alpha158_factors(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
):
    """
    流式产出 Alpha158 (name, normalized_panel)。

    逐因子计算并 yield，不累积全量面板；同批共享 ``_A158Ctx`` 惰性中间量，
    每 10 个因子清一次缓存，避免 153 张全市场面板同时驻留导致 OOM。
    """
    import gc

    if any(x is None for x in (open_, high, low, volume)):
        logger.warning("Alpha158 跳过：缺少 open/high/low/volume")
        return

    if factor_names is None:
        wanted_list = list(ALPHA158_NAMES)
    else:
        wanted_set = set(factor_names)
        wanted_list = [n for n in ALPHA158_NAMES if n in wanted_set]

    if not wanted_list:
        return

    vwap, vwap_note = compute_vwap(amount, volume, high, low, prices)
    if vwap is not None and any(n == "A158_VWAP0" for n in wanted_list):
        if vwap_note != "amount/volume":
            logger.info(f"Alpha158 VWAP 近似: {vwap_note}")

    ctx = _A158Ctx(prices, open_, high, low, volume, vwap, clean_ret)
    n_out = 0
    try:
        for name in wanted_list:
            try:
                panel = _compute_one(name, ctx)
                if panel is None:
                    continue
                if not isinstance(panel, pd.DataFrame):
                    panel = pd.DataFrame(panel, index=prices.index, columns=prices.columns)
                if panel.isna().all(axis=None):
                    continue
                out = _normalize(panel)
                del panel
                yield name, out
                n_out += 1
                if n_out % 10 == 0:
                    ctx.clear()
                    gc.collect()
            except Exception as e:
                logger.warning(f"Alpha158 {name} 计算失败: {e}")
    finally:
        ctx.clear()

    subset = "" if factor_names is None else f" (白名单 {len(wanted_list)} 个)"
    logger.info(f"Alpha158 因子: 流式完成 {n_out}/{len(ALPHA158_NAMES)} 个{subset}")


def get_alpha158_factors(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
) -> dict:
    """
    返回 Alpha158 因子字典（已截面标准化）。

    必须：prices(close) + open/high/low/volume。
    ``factor_names`` 非空时只算白名单。全量 dict 会 OOM；内存敏感请用
    ``iter_alpha158_factors`` 或 ``compute_alpha158_batch``（小批）。
    """
    return dict(
        iter_alpha158_factors(
            prices=prices, open_=open_, high=high, low=low,
            volume=volume, amount=amount, clean_ret=clean_ret,
            factor_names=factor_names,
        )
    )


def compute_alpha158_batch(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
) -> dict:
    """
    一批 Alpha158（同批共享惰性中间量）。IC 建议每批 ≤20 个名字。
    """
    if factor_names is None:
        logger.warning(
            "compute_alpha158_batch 未指定 factor_names，将流式全量再收集为 dict（易 OOM）"
        )
    return get_alpha158_factors(
        prices=prices, open_=open_, high=high, low=low,
        volume=volume, amount=amount, clean_ret=clean_ret,
        factor_names=factor_names,
    )