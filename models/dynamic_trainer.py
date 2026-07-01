"""
models/dynamic_trainer.py — 因子动态加权策略（Factor Timing）

核心思路：
  不用 ML 预测个股收益，而是每个调仓日实时计算每个因子的近期 IC，
  以 ICIR 作为权重合成综合得分。自动跟踪因子有效性的时序变化，
  天然适应 A 股的风格轮动，不存在训练窗口 regime shift 的问题。

与 ML 方法的对比：
  ML：  权重 = 训练窗口内回归系数（固化，预测期才用）
  本方法：权重 = 过去 K 期每个因子的 ICIR（每期实时更新）

与 walk-forward 因子筛选的关系：
  WF 筛选：binary（IC 达标/不达标 → 用/不用）
  本方法：continuous（IC 高的多用、负 IC 减权/反向；权重取 signed ICIR，clip 至 ±2）

使用：
  trainer = DynamicFactorTrainer(lookback=6, method='icir')
  score_df = trainer.fit_predict(dataset)   # 与 WalkForwardTrainer 接口相同
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DYNAMIC_MAX_WORKERS

# 市场状态/HMM 特征是截面常数（所有股票同值），截面 IC 无意义，跳过
_REGIME_PREFIXES = ("市场", "HMM_")
# Barra 风格因子：用于残差化，不作为 alpha 因子评分
_BARRA_PREFIX = "Barra_"

# ICIR clip 上下限：防止小样本方差爆炸导致权重失控
_ICIR_CLIP = 2.0
_MIN_IC_STOCKS = 20


def _is_regime_feature(name: str) -> bool:
    return any(name.startswith(p) for p in _REGIME_PREFIXES)


def _is_barra_feature(name: str) -> bool:
    return name.startswith(_BARRA_PREFIX)


def _alpha_factor_names(feature_names: list) -> list:
    return [
        n for n in feature_names
        if not _is_regime_feature(n) and not _is_barra_feature(n)
    ]


def _vectorized_cross_ic(
    X: pd.DataFrame,
    y: pd.Series,
    min_stocks: int = _MIN_IC_STOCKS,
) -> pd.Series:
    """单期全因子截面 Spearman IC（rank + corrwith，每列 pairwise 有效样本）"""
    y_clean = y.dropna()
    if len(y_clean) < min_stocks:
        return pd.Series(np.nan, index=X.columns)
    X_sub = X.reindex(y_clean.index)
    ics = X_sub.rank().corrwith(y_clean.rank())
    valid_n = X_sub.notna().sum()
    ics = ics.where(valid_n >= min_stocks, np.nan)
    return ics


def _aggregate_ic_matrix(
    past_ic: pd.DataFrame,
    method: str,
    ic_decay: float,
    min_lookback: int,
) -> pd.Series:
    """对 (lookback × 因子) IC 矩阵按列聚合为权重（未 clip）"""
    valid = past_ic.notna().sum(axis=0)
    if method == "icir":
        mu = past_ic.mean(axis=0, skipna=True)
        std = past_ic.std(axis=0, skipna=True)
        weights = mu / (std + 1e-8)
    elif method == "decay":
        n = past_ic.shape[0]
        if n == 0:
            return pd.Series(dtype=float)
        decay_w = np.array(
            [(1 - ic_decay) ** (n - 1 - i) for i in range(n)],
            dtype=float,
        )
        decay_w /= decay_w.sum()
        weights = past_ic.fillna(0.0).values.T @ decay_w
        weights = pd.Series(weights, index=past_ic.columns)
    else:  # 'ic'
        weights = past_ic.mean(axis=0, skipna=True)

    return weights.where(valid >= min_lookback, np.nan)


def _composite_score(X_curr: pd.DataFrame, filtered: dict) -> pd.Series | None:
    """按因子权重合成截面 z-score 得分"""
    composite = pd.Series(0.0, index=X_curr.index)
    total_abs_weight = 0.0
    for fname, w in filtered.items():
        if fname not in X_curr.columns:
            continue
        f_vals = X_curr[fname].dropna()
        if len(f_vals) < 10:
            continue
        f_std = f_vals.std()
        if f_std < 1e-10:
            continue
        f_z = ((f_vals - f_vals.mean()) / f_std).clip(-3, 3)
        composite = composite.add(w * f_z, fill_value=0)
        total_abs_weight += abs(w)

    if total_abs_weight <= 0:
        return None
    return composite / total_abs_weight


def _process_rebalance_date(args):
    """单调仓日：ICIR 权重 + 合成得分（供线程池调用）"""
    (
        i,
        date,
        rebalance_dates,
        factor_names,
        ic_df,
        section_cache,
        lookback,
        min_lookback,
        method,
        ic_decay,
        min_weight,
    ) = args

    if i < min_lookback:
        return None

    past_dates = rebalance_dates[max(0, i - lookback): i]
    past_ic = ic_df.reindex(past_dates)
    if past_ic.empty:
        return None

    raw_w = _aggregate_ic_matrix(past_ic, method, ic_decay, min_lookback)
    raw_weights = {
        fname: float(np.clip(w, -_ICIR_CLIP, _ICIR_CLIP))
        for fname, w in raw_w.items()
        if fname in factor_names and np.isfinite(w)
    }
    if not raw_weights:
        return date, None, None, "no_weights"

    filtered = {k: v for k, v in raw_weights.items() if abs(v) >= min_weight}
    if not filtered:
        filtered = raw_weights

    if date not in section_cache:
        return date, None, None, "no_section"

    X_curr, _ = section_cache[date]
    composite = _composite_score(X_curr, filtered)
    if composite is None:
        return date, None, filtered, "no_composite"

    return date, composite, filtered, None


@dataclass
class DynamicFactorResult:
    """DynamicFactorTrainer.fit_predict() 的返回值"""
    score_df: pd.DataFrame           # (调仓日, 股票) 综合得分
    factor_weights: pd.DataFrame         # (调仓日, 因子名) 实时权重，供诊断用
    factor_weights_history: pd.DataFrame # (调仓日, 因子名) 当期权重时序（基于 ICIR）


class DynamicFactorTrainer:
    """
    因子动态加权训练器。

    参数
    ----
    lookback : int
        用于估计因子有效性的历史调仓期数（默认 6 期）。
        月频 = 过去 6 个月；周频 = 过去 6 周。
    min_lookback : int
        开始预测所需的最少历史期数（默认 3 期）。
    method : str
        权重计算方式：
          'icir'  — IC 均值 / IC 标准差（推荐，更稳定）
          'ic'    — 过去 K 期 IC 均值（简单）
          'decay' — 指数衰减加权 IC 均值（近期 IC 权重更高）
    ic_decay : float
        method='decay' 时的衰减因子，每往前一期乘以 (1 - ic_decay)。
    min_weight : float
        |权重| 低于此阈值的因子置零（过滤噪声因子）。
    max_workers : int
        调仓日并行度；默认读 config.settings.DYNAMIC_MAX_WORKERS（上限同该值）。
        单线程基线约 ~4.5GB；ThreadPoolExecutor 共享 section_cache/ic_df，
        通常不会完整复制基线，但并行 >4 在 32GB 机器上仍可能 OOM。
        与 ML _walk-forward 并发时建议 DYNAMIC_MAX_WORKERS=1。
    """

    def __init__(
        self,
        lookback: int = 6,
        min_lookback: int = 3,
        method: str = "icir",
        ic_decay: float = 0.2,
        min_weight: float = 0.1,
        max_workers: int | None = None,
    ):
        self.lookback = lookback
        self.min_lookback = min_lookback
        self.method = method
        self.ic_decay = ic_decay
        self.min_weight = min_weight
        _requested = DYNAMIC_MAX_WORKERS if max_workers is None else max_workers
        self.max_workers = max(1, min(_requested, DYNAMIC_MAX_WORKERS))

        self.factor_weights_log: dict = {}
        self.factor_ic_log: dict = {}

    def fit_predict(self, dataset) -> pd.DataFrame:
        """
        对每个调仓日，用过去 lookback 期的 IC 估计因子权重，
        合成当期截面得分。

        返回：score_df，shape=(调仓日, 股票)，与 WalkForwardTrainer 接口相同。
        """
        rebalance_dates = dataset.rebalance_dates
        total = len(rebalance_dates)
        factor_names = _alpha_factor_names(dataset.feature_names)

        # ── 预缓存截面（每调仓日仅构建一次，替代 O(因子×lookback×日期) 次 loc）──
        logger.info("DynamicFactorTrainer: 预缓存截面数据...")
        section_cache: dict = {}
        for date in rebalance_dates:
            X, y = dataset.get_cross_section(date)
            if X is not None and len(X) > 0:
                section_cache[date] = (X.astype(np.float32), y.astype(np.float32))

        # ── 预计算全日期×全因子 IC 矩阵 ─────────────────────────────────────
        logger.info("DynamicFactorTrainer: 预计算截面 IC 矩阵...")
        ic_rows = {}
        for date, (X, y) in section_cache.items():
            cols = [c for c in factor_names if c in X.columns]
            if not cols:
                continue
            ic_rows[date] = _vectorized_cross_ic(X[cols], y)
        ic_df = pd.DataFrame(ic_rows).T.astype(np.float32)
        ic_df.index.name = "date"

        n_workers = max(1, min(self.max_workers, total))
        logger.info(
            f"DynamicFactorTrainer: {total} 调仓日, {len(factor_names)} 因子, "
            f"lookback={self.lookback}, 并行={n_workers} "
            f"(DYNAMIC_MAX_WORKERS={DYNAMIC_MAX_WORKERS})"
        )

        job_args = [
            (
                i,
                date,
                rebalance_dates,
                factor_names,
                ic_df,
                section_cache,
                self.lookback,
                self.min_lookback,
                self.method,
                self.ic_decay,
                self.min_weight,
            )
            for i, date in enumerate(rebalance_dates)
        ]

        scores = {}
        weights_log = {}
        log_step = max(1, min(50, total // 10))
        done = 0

        def _collect(result):
            if result is None:
                return
            date, composite, filtered, err = result
            if err == "no_weights":
                logger.warning(f"{date}: 无有效因子权重，跳过")
                return
            if composite is None:
                return
            scores[date] = composite
            weights_log[date] = filtered

        if n_workers <= 1:
            for args in job_args:
                result = _process_rebalance_date(args)
                done += 1
                if done % log_step == 0 or done == total:
                    logger.info(f"DynamicFactorTrainer: {done}/{total} 调仓日")
                _collect(result)
        else:
            # 并行度 capped by DYNAMIC_MAX_WORKERS（默认 4）；32GB + 其他任务时用 1 最安全
            # done 计数在主线程 as_completed 循环里递增，避免多线程 race condition
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_process_rebalance_date, a) for a in job_args]
                for fut in as_completed(futures):
                    done += 1
                    if done % log_step == 0 or done == total:
                        logger.info(f"DynamicFactorTrainer: {done}/{total} 调仓日")
                    _collect(fut.result())

        if not scores:
            raise RuntimeError("DynamicFactorTrainer: 未能产生任何预测，检查数据")

        score_df = pd.DataFrame(scores).T
        score_df.index.name = "date"

        self.factor_weights_log = weights_log
        self.factor_weights_history = pd.DataFrame(weights_log).T.fillna(0)

        n_predictions = len(scores)
        logger.info(
            f"DynamicFactorTrainer 完成: {n_predictions} 个预测日, "
            f"lookback={self.lookback}, method={self.method}"
        )

        if weights_log:
            last_date = max(weights_log)
            top_w = sorted(
                weights_log[last_date].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
            logger.info(
                f"最新一期({last_date.date()})因子权重 Top10:\n"
                + "\n".join(f"  {k:30s}: {v:+.3f}" for k, v in top_w)
            )

        return score_df

    def print_weight_evolution(self, top_n: int = 8, last_n: int = 12):
        """打印因子权重的时序变化（诊断风格轮动）"""
        if not self.factor_weights_log:
            print("尚未运行 fit_predict()")
            return
        df = pd.DataFrame(self.factor_weights_log).T.fillna(0)
        recent = df.tail(last_n)
        top_factors = recent.abs().mean().nlargest(top_n).index.tolist()
        print(f"\n{'─'*70}")
        print(f"因子权重演变（最近 {last_n} 期，Top {top_n} 因子）")
        print(f"{'─'*70}")
        print(recent[top_factors].round(3).to_string())
        print(f"{'─'*70}\n")
