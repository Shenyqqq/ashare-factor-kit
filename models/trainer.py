"""
models/trainer.py — Walk-Forward 训练器（v2 模块化实现，已合并 v1）

WalkForwardTrainer 在每个调仓日用历史数据滚动训练，
输出样本外预测分矩阵（与回测引擎直接对接）。

实现要点（继承自原 trainer_v2 / models/wf/）：
  • Window-specific validation（每个训练窗口对应自己的验证段，无 lookahead）
  • Purged training + embargo（AFML Ch. 7，防止 forward-return 标签泄漏）
  • IC-weighted window/model selection（wf_selection）
  • Z-score 加权 ensemble（非盲 rank average）
  • Cross-sectional label standardization（label_mode）
  • Train / val / pred IC 诊断 CSV + 特征重要性导出
  • save_metrics() 写 model_metrics_<tag>.json

支持模型：ridge | lgbm | xgb | cat | rf | mlp
Ensemble：多窗口 × 多模型 → IC 加权 Z-score 平均

历史：原 v1 单体实现已删除；本文件保留 v1 的共享基础设施
（MLDataset / build_ml_dataset / 超参数 / resolve_train_windows 等）
供 wf 子包与 industry_trainer / dynamic_trainer / analyzer 复用。
"""
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from joblib import Parallel, delayed

import lightgbm as lgb
import xgboost as xgb_lib
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, TRAIN_MAX_WORKERS, TRAIN_N_JOBS
from utils.rebalance_dates import get_rebalance_dates

# Hyperparams live in models/wf/params.py to avoid circular imports with wf/models.py
from models.wf.params import (
    RIDGE_PARAMS, RIDGE_CV_ALPHAS,
    LGBM_PARAMS, XGB_PARAMS, CAT_PARAMS, RF_PARAMS, MLP_PARAMS,
)
from models.wf.splits import (
    get_window_splits,
    purge_train_indices,
    embargo_train_end,
    hold_period_to_embargo_periods,
)
from models.wf.labels import (
    transform_labels,
    precompute_label_controls,
    normalize_sample_weights_by_universe,
    compute_return_overlap_weights,
    triple_barrier_label,
)
from models.wf.ensemble import (
    combine_model_scores,
    select_window_weights,
    dynamic_model_weights,
)
from models.wf.models import fit_model, predict_model, extract_feature_importance
from models.wf.metrics import (
    spearman_ic,
    compute_drift_flags,
    export_diagnostics,
    export_feature_importance,
    append_feature_importance_rows,
)
from models.wf.persistence import save_fold_model, save_models_manifest

import json

N_CORES = os.cpu_count() or 4

# ── 超参数（集中配置）────────────────────────────────────────────────────────

TRAIN_WINDOWS_MONTHS = [6, 12]   # 日历月；WalkForwardTrainer 构造时转为调仓期数
VAL_WINDOW_MONTHS    = 6         # 日历月
TIME_DECAY           = 0.015
MIN_STOCKS_PER_DATE  = 30
REBALANCE_FREQ       = "ME"
MODEL_TYPES          = ["lgbm", "xgb"]   # 默认稳妥组合；cat 为可选模型（ordered boosting 抗过拟合），ridge/rf/mlp 可按需启用


def months_to_rebalance_periods(months: int, rebalance_freq: str) -> int:
    """将日历月数转为调仓期数（Walk-Forward 内部仍按 period 索引回溯）。"""
    fu = rebalance_freq.upper()
    if "2W" in fu:
        return max(1, round(months * 26 / 12))  # biweekly
    if fu.startswith("W") or fu.endswith("D"):  # weekly or N-day
        return max(1, round(months * 52 / 12))  # ~4.33 weeks/month
    return months  # ME monthly: 1 period ≈ 1 month


def resolve_train_windows(
    train_windows: list,
    val_window: int,
    rebalance_freq: str,
    units: str = "months",
) -> tuple[list, int]:
    """
    解析 Walk-Forward 训练/验证窗口长度（调仓期数）。

    units="months"（默认）：train_windows / val_window 为日历月，按 rebalance_freq 换算。
    units="periods"：直接当作调仓期数（历史 bug 行为：h5 下 6,12 即 6/12 周）。
    """
    if units == "periods":
        return list(train_windows), val_window
    if units != "months":
        raise ValueError(f"未知 train_window_units: {units}，可选 months | periods")
    return (
        [months_to_rebalance_periods(w, rebalance_freq) for w in train_windows],
        months_to_rebalance_periods(val_window, rebalance_freq),
    )


WF_SELECTION_DEFAULT = "ic_weighted"
LABEL_MODE_DEFAULT = "cs_zscore"
ENSEMBLE_METHOD_DEFAULT = "zscore"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class MLDataset:
    """
    多因子截面数据集。
    factor_panel:   {因子名: DataFrame(index=日期, columns=股票)}
    forward_return: DataFrame(index=日期, columns=股票)
    """
    factor_panel:    dict
    forward_return:  pd.DataFrame
    rebalance_dates: list
    feature_names:   list

    def get_cross_section(self, date):
        rows = {name: df.loc[date] for name, df in self.factor_panel.items()
                if date in df.index}
        if not rows:
            return None, None
        # 用 0 填充 NaN（z-score 空间里 0 = 截面中性），避免稀疏因子剔除大量股票
        X = pd.DataFrame(rows).fillna(0)
        # 至少要有一个因子有真实值的股票才保留
        has_data = pd.DataFrame(rows).notna().any(axis=1)
        X = X.loc[has_data]
        if date not in self.forward_return.index:
            return None, None
        y = self.forward_return.loc[date].reindex(X.index).dropna()
        X = X.loc[X.index.intersection(y.index)]
        return X, y.loc[X.index]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_ml_dataset(factor_dict, forward_return, rebalance_freq=REBALANCE_FREQ):
    if not factor_dict:
        raise ValueError("factor_dict 为空，没有任何因子被计算")
    all_dates = sorted(
        set.intersection(*[set(df.index) for df in factor_dict.values()])
        & set(forward_return.index)
    )
    if not all_dates:
        ranges = {n: f"[{df.index.min()}, {df.index.max()}]" if len(df) > 0 else "EMPTY"
                  for n, df in factor_dict.items()}
        raise ValueError(
            f"all_dates 为空：各因子日期无公共交集。\n"
            f"forward_return: [{forward_return.index.min()}, {forward_return.index.max()}]\n"
            f"各因子范围: {ranges}"
        )
    all_dates_idx = pd.DatetimeIndex(all_dates)
    rebalance_dates = get_rebalance_dates(all_dates_idx, rebalance_freq).tolist()
    logger.info(f"数据集: {len(factor_dict)}个因子, {len(rebalance_dates)}个调仓日")
    return MLDataset(
        factor_panel=factor_dict,
        forward_return=forward_return.astype(np.float32),
        rebalance_dates=rebalance_dates,
        feature_names=list(factor_dict.keys()),
    )


def to_rank(arr):
    if len(arr) == 0:
        return arr
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks / max(len(arr) - 1, 1)


def rank_average(rank_list):
    return np.mean(rank_list, axis=0)


# ── 单折结果 ──────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    window: int
    model_type: str
    pred_scores: np.ndarray
    val_ic: float
    train_ic: float
    model: object = None
    importance: dict | None = None


def _run_fold_job(job):
    """模块级单折训练函数（可被 joblib 多进程 pickle）。

    job 结构：
        (window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
         X_va, y_va, y_va_raw, X_pred, y_true, n_jobs,
         objective, feature_names, pred_date, device,
         group_tr, group_va)

    其中 ``group_tr`` / ``group_va`` 为每个调仓日的样本数 list[int]，
    仅 ``objective='rank'`` 时使用（用于 LGBMRanker.group / XGBRanker.qid /
    CatBoostRanker.group_id）；其它模式下为 None，被忽略。

    返回 FoldResult 或 None（训练异常时）。
    所有依赖（objective / feature_names / pred_date / device）通过 job 显式传入，
    避免 closure 引用 self 导致无法 pickle。
    """
    (window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
     X_va, y_va, y_va_raw, X_pred, y_true, n_jobs,
     objective, feature_names, pred_date, device,
     group_tr, group_va) = job
    try:
        model = fit_model(
            model_type, X_tr, y_tr, w_tr, X_va, y_va,
            n_jobs=n_jobs, objective=objective, device=device,
            group_tr=group_tr, group_va=group_va,
        )
        pred = predict_model(model, X_pred)
        val_ic = spearman_ic(model.predict(X_va), y_va_raw if y_va_raw is not None else y_va)
        train_ic = spearman_ic(model.predict(X_tr), y_tr_raw if y_tr_raw is not None else y_tr)
        imp = extract_feature_importance(
            model, model_type, feature_names,
            X_va=X_va, y_va=(y_va_raw if y_va_raw is not None else y_va),
        )
        return FoldResult(window, model_type, pred, val_ic, train_ic, model, imp)
    except Exception as e:
        date_str = pred_date.date() if pred_date is not None else "?"
        logger.warning(f"训练失败({model_type}, w={window}, {date_str}): {e}")
        return None


# ── Walk-Forward 训练器 ───────────────────────────────────────────────────────

class WalkForwardTrainer:
    """
    Walk-forward trainer with purged splits, IC-weighted ensembling, diagnostics.

    每个调仓日：多窗口 × 多模型 → IC 加权 Z-score 平均 → 最终得分。

    调用：
        trainer = WalkForwardTrainer(model_types=["lgbm", "xgb"])
        score_df = trainer.fit_predict(dataset)

    兼容别名 ``WalkForwardTrainerV2`` 保留向后引用。
    """

    def __init__(
        self,
        train_windows=TRAIN_WINDOWS_MONTHS,
        val_window=VAL_WINDOW_MONTHS,
        model_types=MODEL_TYPES,
        artifact_dir=None,
        rebalance_freq=REBALANCE_FREQ,
        train_window_units: str = "months",
        hold_period: int = 20,
        wf_selection: str = WF_SELECTION_DEFAULT,
        label_mode: str | dict = LABEL_MODE_DEFAULT,
        ensemble_method: str = ENSEMBLE_METHOD_DEFAULT,
        window_specific_val: bool = True,
        purge_train: bool = True,
        embargo: bool = True,
        save_models: bool = False,
        objective: str = "regression",
        output_rank: bool = False,
        tag: str | None = None,
        barra_factors: dict[str, pd.DataFrame] | None = None,
        industry_map: pd.Series | None = None,
        device: str = "cpu",
        prices: pd.DataFrame | None = None,
        open_prices: pd.DataFrame | None = None,
        triple_barrier_params: dict | None = None,
    ):
        # 默认 train_windows/val_window 为日历月；units=periods 时直接当调仓期数
        self.train_windows, self.val_window = resolve_train_windows(
            train_windows, val_window, rebalance_freq, train_window_units,
        )
        self.min_train_window = min(self.train_windows)
        self.model_types = list(model_types)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.rebalance_freq = rebalance_freq
        self.hold_period = hold_period
        self.wf_selection = wf_selection
        self.label_mode = label_mode
        self.ensemble_method = ensemble_method
        self.window_specific_val = window_specific_val
        self.purge_train = purge_train
        self.embargo = embargo
        self.save_models = save_models
        self.objective = objective
        # rank objective 默认配合 cs_rank 标签（截面排名 0-1，适配 LambdaRank
        # 的 label_gain）。仅在用户未显式覆盖 label_mode（仍是默认 cs_zscore）时
        # 自动切换；用户自定义 label_mode（含 dict 形式）时尊重其选择。
        if (
            objective == "rank"
            and isinstance(label_mode, str)
            and label_mode == LABEL_MODE_DEFAULT
        ):
            self.label_mode = "cs_rank"
            logger.info(
                "rank objective 检测到默认 label_mode='cs_zscore'，自动切换为 'cs_rank' "
                "（LambdaRank 标签增益适配）；如需保留 cs_zscore 请显式传入 label_mode。"
            )
        self.output_rank = output_rank
        self.tag = tag
        # Barra 残差化标签所需控制变量（仅 label_mode='barra_residual' 时使用）
        self.barra_factors = barra_factors
        self.industry_map = industry_map
        self.device = device
        self._label_controls: dict | None = None
        # Triple-barrier 标签（AFML §3，路径依赖 + 波动率自适应）
        # 仅 label_mode='triple_barrier' 时使用：预计算为面板 DataFrame
        # (index=signal_dates, columns=stocks)，_stack_cached 按当期行取用。
        self.prices = prices
        self.open_prices = open_prices
        self.triple_barrier_params = triple_barrier_params or {}
        self._barrier_labels: pd.DataFrame | None = None

        self.score_df: pd.DataFrame | None = None
        self.ic_series: pd.Series | None = None
        self.models: dict = {}
        self.model_ic: dict = {m: {} for m in self.model_types}
        self._diagnostics: list[dict] = []
        self._drift_history: list[dict] = []
        self._rolling_val_ic: dict[str, list[float]] = {m: [] for m in self.model_types}
        self._feature_importance_rows: list[dict] = []
        self._dataset: MLDataset | None = None

        logger.info(
            f"Walk-Forward 窗口: train={self.train_windows}, val={self.val_window} 期 "
            f"(units={train_window_units}, freq={rebalance_freq}); "
            f"wf_selection={wf_selection}, label_mode={self.label_mode}, "
            f"ensemble={ensemble_method}, purge={purge_train}, embargo={embargo}, "
            f"objective={self.objective}"
        )

    def _resolve_label_mode(self, model_type: str) -> str:
        if isinstance(self.label_mode, dict):
            return self.label_mode.get(model_type, "raw" if model_type == "ridge" else "cs_zscore")
        return self.label_mode

    def _needs_barra_controls(self) -> bool:
        """判断是否有任意模型分支使用了 barra_residual 标签。"""
        modes = (
            set(self.label_mode.values()) if isinstance(self.label_mode, dict)
            else {self.label_mode}
        )
        return "barra_residual" in modes

    def _needs_triple_barrier(self) -> bool:
        """判断是否有任意模型分支使用了 triple_barrier 标签。"""
        modes = (
            set(self.label_mode.values()) if isinstance(self.label_mode, dict)
            else {self.label_mode}
        )
        return "triple_barrier" in modes

    def fit_predict(self, dataset: MLDataset) -> pd.DataFrame:
        self._dataset = dataset
        dates = dataset.rebalance_dates
        n_dates = len(dates)
        min_history = max(self.train_windows) + self.val_window + (
            max(self.train_windows) - self.min_train_window if self.window_specific_val else 0
        )
        predict_start = min_history

        embargo_periods = hold_period_to_embargo_periods(self.hold_period, dates) if self.embargo else 0

        # 预构建 date→pos 映射，避免 purge_train_indices 内反复 list.index() (O(N)→O(1))
        date_to_pos = {d: i for i, d in enumerate(dates)}

        logger.info("预缓存截面数据...")
        section_cache: dict = {}
        for date in dates:
            X, y = dataset.get_cross_section(date)
            if X is not None and len(X) >= MIN_STOCKS_PER_DATE:
                section_cache[date] = (
                    X.values.astype(np.float32),
                    y.values.astype(np.float32),
                    X.index,
                )

        # 预计算 Barra 残差化标签控制变量（仅 label_mode='barra_residual' 时）
        if self._needs_barra_controls():
            logger.info("预计算 Barra 残差化控制矩阵...")
            self._label_controls = precompute_label_controls(
                self.barra_factors, self.industry_map, dates,
            )
            logger.info(
                f"Barra 残差化控制矩阵就绪: {len(self._label_controls)}/{len(dates)} 个调仓日"
            )

        # 预计算 triple-barrier 标签面板（AFML §3）
        # 一次性计算所有调仓日的路径依赖标签，供 _stack_cached 按当期行取用。
        if self._needs_triple_barrier():
            if self.prices is None or self.open_prices is None:
                raise ValueError(
                    "label_mode='triple_barrier' 需要在构造 WalkForwardTrainer 时传入 "
                    "prices 与 open_prices（日频收盘/开盘价 DataFrame）。"
                )
            tb_kwargs = {
                "hold_period": self.hold_period,
                "vol_window": int(self.triple_barrier_params.get("vol_window", 20)),
                "upper_mult": float(self.triple_barrier_params.get("upper_mult", 2.0)),
                "lower_mult": float(self.triple_barrier_params.get("lower_mult", 1.5)),
                "label_type": self.triple_barrier_params.get("label_type", "sign"),
            }
            logger.info(
                f"预计算 triple-barrier 标签面板: "
                f"{len(dates)} 调仓日 × hold={tb_kwargs['hold_period']} "
                f"σ_win={tb_kwargs['vol_window']} "
                f"up={tb_kwargs['upper_mult']}σ dn={tb_kwargs['lower_mult']}σ "
                f"label_type={tb_kwargs['label_type']}"
            )
            self._barrier_labels = triple_barrier_label(
                self.prices, self.open_prices, dates, **tb_kwargs,
            )
            if not self._barrier_labels.empty:
                valid_ratio = self._barrier_labels.notna().mean().mean()
                logger.info(
                    f"triple-barrier 面板就绪: "
                    f"{self._barrier_labels.shape[0]} 日 × "
                    f"{self._barrier_labels.shape[1]} 股，"
                    f"标签有效率={valid_ratio:.2%}"
                )

        def _stack_cached(date_list, label_mode: str = "raw"):
            X_list, y_list, y_raw_list, w_list = [], [], [], []
            dates_used, stocks_per_date = [], []
            use_barra = label_mode == "barra_residual"
            use_tb = label_mode == "triple_barrier"
            for i, d in enumerate(date_list):
                if d not in section_cache:
                    continue
                X_np, y_np, stock_idx = section_cache[d]
                # triple-barrier：用预计算面板的当期行替换 forward_return 作为 y
                if use_tb:
                    if self._barrier_labels is None or d not in self._barrier_labels.index:
                        continue
                    barrier_row = self._barrier_labels.loc[d].reindex(stock_idx)
                    y_np_tb = barrier_row.values.astype(np.float32)
                    valid_mask = ~np.isnan(y_np_tb)
                    if valid_mask.sum() < MIN_STOCKS_PER_DATE:
                        continue
                    X_np = X_np[valid_mask]
                    y_np = y_np_tb[valid_mask]
                    stock_idx = stock_idx[valid_mask]
                barra_df = None
                ind_dummies = None
                if use_barra and self._label_controls and d in self._label_controls:
                    b_full, i_full = self._label_controls[d]
                    # 对齐到当期截面的股票索引
                    barra_df = b_full.reindex(stock_idx).fillna(0.0)
                    ind_dummies = i_full.reindex(stock_idx).fillna(0.0)
                y_t = transform_labels(
                    y_np, label_mode,
                    barra_factors=barra_df, industry_dummies=ind_dummies,
                )
                decay = np.exp(TIME_DECAY * i)
                X_list.append(X_np)
                y_list.append(y_t)
                y_raw_list.append(y_np)
                w_list.extend([decay] * len(y_t))
                dates_used.append(d)
                stocks_per_date.append(int(len(y_t)))
            if not X_list:
                return None, None, None, None, None
            w_arr = np.array(w_list)
            # P1-2: 按 universe 大小归一化样本权重，避免晚期调仓日（股票多）
            # 总权重远大于早期，导致训练样本被晚期主导。
            w_arr = normalize_sample_weights_by_universe(
                w_arr, dates_used, stocks_per_date,
            )
            # AFML §4: 相邻训练样本 forward_return 标签时间重叠时降权。
            # 默认配置（调仓间隔≈hold_period）下 overlap≈0 → 权重≈1.0；
            # override 配置（--backtest-freq 更频繁调仓）下 overlap>0 → 降权。
            # 注意用 dates_used（实际入栈的调仓日），而非 date_list（可能含
            # section_cache 未覆盖的日期）。
            overlap_w = compute_return_overlap_weights(
                dates_used, self.hold_period,
            )
            if len(overlap_w) == len(stocks_per_date):
                overlap_expanded = np.repeat(overlap_w, stocks_per_date)
                if len(overlap_expanded) == len(w_arr):
                    w_arr = (w_arr * overlap_expanded).astype(w_arr.dtype)
            return (
                np.vstack(X_list),
                np.concatenate(y_list),
                w_arr,
                np.concatenate(y_raw_list),
                # group_sizes：每个调仓日的样本数 list[int]，供 rank objective 的
                # LGBMRanker.group / XGBRanker.qid / CatBoostRanker.group_id 使用。
                list(stocks_per_date),
            )

        # ── 并行度：默认串行各 (window, model) 任务，单模型多线程 ─────────────
        n_combos = len(self.train_windows) * len(self.model_types)
        n_workers = max(1, min(n_combos, TRAIN_MAX_WORKERS))
        cores_per_job = 1 if n_workers > 1 else (TRAIN_N_JOBS if TRAIN_N_JOBS > 0 else N_CORES)

        logger.info(
            f"Walk-Forward: 预测{max(0, n_dates - predict_start)}个调仓日, "
            f"模型={self.model_types}, "
            f"并行={n_workers}任务×{cores_per_job}核/任务 "
            f"(TRAIN_MAX_WORKERS={TRAIN_MAX_WORKERS})"
        )

        all_scores = {}
        models_lock = threading.Lock()
        n_predict = max(0, n_dates - predict_start)
        saved_model_entries = []

        for idx in range(predict_start, n_dates):
            pred_date = dates[idx]
            if pred_date not in section_cache:
                continue
            X_pred_np, y_true_np, pred_index = section_cache[pred_date]

            jobs = []

            for window in self.train_windows:
                ts, te, vs, ve = get_window_splits(
                    idx, window, self.val_window, n_dates,
                    min_train_window=self.min_train_window,
                    window_specific_val=self.window_specific_val,
                )
                train_dates = dates[ts:te]
                val_dates = dates[vs:ve]

                if self.embargo and te > 0:
                    eff_te = embargo_train_end(te, embargo_periods)
                    train_dates = dates[ts:eff_te]

                if self.purge_train:
                    train_dates = purge_train_indices(
                        train_dates, val_dates, pred_date, dates, self.hold_period,
                        date_pos_map=date_to_pos,
                    )

                # P1-3: 长窗口 floor 用 max(8, window // 3) - embargo，
                # 避免周频长窗口（如 52 期）下小样本污染集成。
                min_train_dates = max(8, window // 3) - (embargo_periods if self.embargo else 0)
                if len(train_dates) < min_train_dates or len(val_dates) < 2:
                    continue

                for model_type in self.model_types:
                    lm = self._resolve_label_mode(model_type)
                    stacked_tr = _stack_cached(train_dates, lm)
                    stacked_va = _stack_cached(val_dates, lm)
                    if stacked_tr[0] is None or stacked_va[0] is None:
                        continue
                    X_tr, y_tr, w_tr, y_tr_raw, group_tr = stacked_tr
                    X_va, y_va, _, y_va_raw, group_va = stacked_va
                    jobs.append((
                        window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
                        X_va, y_va, y_va_raw, X_pred_np, y_true_np, cores_per_job,
                        # 多进程 pickle 所需的额外上下文（_run_fold_job 为模块级函数）
                        self.objective, dataset.feature_names, pred_date, self.device,
                        # rank objective 所需的逐期 group 大小（非 rank 模式被忽略）
                        group_tr, group_va,
                    ))

            if not jobs:
                continue

            # n_workers<=1：串行执行（保留 lgbm/xgb 单模型多线程 cores_per_job 语义，
            #   避免进程 fork 与 pickle 开销，模型对象也无需跨进程序列化）
            # n_workers>1：joblib 多进程（绕过 GIL，纯 Python 训练循环 2-4× 加速）
            #   _run_fold_job 为模块级函数，所有依赖通过 job tuple 显式传入以兼容 pickle
            results = []
            if n_workers <= 1:
                for j in jobs:
                    r = _run_fold_job(j)
                    if r is not None:
                        results.append(r)
            else:
                # prefer="processes" 强制进程池；timeout 防止单个 job 卡死整个流水线
                raw = Parallel(
                    n_jobs=n_workers,
                    prefer="processes",
                    timeout=3600,
                )(
                    delayed(_run_fold_job)(j) for j in jobs
                )
                for r in raw:
                    if r is not None:
                        results.append(r)

            if not results:
                continue

            # Group by model_type: combine windows with IC weights
            model_final_scores = {}
            diag_row = {
                "date": pred_date,
                "pred_ic": np.nan,
            }

            dyn_w = dynamic_model_weights(self._rolling_val_ic, self.model_types)

            for model_type in self.model_types:
                folds = [r for r in results if r.model_type == model_type]
                if not folds:
                    continue

                scores_list = [f.pred_scores for f in folds]
                val_ics = [f.val_ic for f in folds]
                train_ics = [f.train_ic for f in folds]

                if self.wf_selection == "best_model":
                    w = select_window_weights(val_ics, "best_window")
                else:
                    w = select_window_weights(val_ics, self.wf_selection)

                combined = combine_model_scores(
                    scores_list, w,
                    method=self.ensemble_method,
                    output_rank=False,
                )
                model_final_scores[model_type] = combined

                for i, f in enumerate(folds):
                    diag_row[f"train_ic_{model_type}_w{f.window}"] = train_ics[i]
                    diag_row[f"val_ic_{model_type}_w{f.window}"] = val_ics[i]
                    with models_lock:
                        self.models[(f.window, model_type)] = f.model
                    if self.save_models and self.artifact_dir is not None:
                        p = save_fold_model(
                            f.model, model_type, f.window, pred_date,
                            self.artifact_dir / "models",
                        )
                        saved_model_entries.append({
                            "path": str(p), "model": model_type,
                            "window": f.window, "date": str(pred_date),
                        })
                    imp = f.importance if f.importance is not None else {}
                    append_feature_importance_rows(
                        self._feature_importance_rows,
                        model_type, f.window, pred_date, imp,
                    )

                self._rolling_val_ic[model_type].append(float(np.nanmean(val_ics)))

            if not model_final_scores:
                continue

            # Combine models (dynamic IC weights when multiple models)
            if len(model_final_scores) > 1:
                m_types = list(model_final_scores.keys())
                m_weights = np.array([dyn_w.get(m, 1.0 / len(m_types)) for m in m_types])
                m_weights = m_weights / m_weights.sum()
                final = combine_model_scores(
                    [model_final_scores[m] for m in m_types],
                    m_weights,
                    method=self.ensemble_method,
                    output_rank=self.output_rank,
                )
            else:
                final = combine_model_scores(
                    list(model_final_scores.values()),
                    method=self.ensemble_method,
                    output_rank=self.output_rank,
                )

            pred_ic = spearman_ic(final, y_true_np)
            diag_row["pred_ic"] = pred_ic
            drift = compute_drift_flags(pred_ic, self._drift_history)
            diag_row.update(drift)
            # P1-5: drift 历史改用 pred_ic，而非恒为 0 的 cs_zscore score_mean
            self._drift_history.append({"pred_ic": pred_ic})
            self._diagnostics.append(diag_row)

            all_scores[pred_date] = pd.Series(final, index=pred_index)

            for m, s in model_final_scores.items():
                ic = spearman_ic(s, y_true_np)
                self.model_ic[m][pred_date] = ic

            done = idx - predict_start + 1
            if done % 6 == 0 or done == n_predict:
                logger.info(f"进度 {done}/{n_predict}: {pred_date.date()}, IC={pred_ic:.4f}")

        self.score_df = pd.DataFrame(all_scores).T
        if not self.score_df.empty:
            self.score_df.index = pd.to_datetime(self.score_df.index)
        self.score_df.index.name = "date"

        if self.score_df.empty:
            logger.warning("无有效预测日，请检查 train_windows / val_window 是否过短")
            self.ic_series = pd.Series(dtype=float)
            return self.score_df

        ic_dict = {}
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = self.score_df.loc[date].dropna()
            y = y.reindex(s.index).dropna()
            s = s.loc[y.index]
            if len(s) >= MIN_STOCKS_PER_DATE:
                ic_dict[date] = spearman_ic(s.values, y.values)
        self.ic_series = pd.Series(ic_dict)

        for m in self.model_types:
            self.model_ic[m] = pd.Series(self.model_ic[m])

        ic_mean = self.ic_series.mean()
        std = self.ic_series.std()
        icir = ic_mean / std if std > 0 else np.nan
        logger.info(
            f"完成: IC均值={ic_mean:.4f}, ICIR={icir:.4f}, "
            f"胜率={(self.ic_series > 0).mean():.1%}"
        )

        out = self.artifact_dir or PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)
        # tag 加入文件名，避免多个 trainer（如分行业子训练器）产物互相覆盖
        tag_for_file = self.tag or "wf"
        self.score_df.to_parquet(out / f"ml_factor_scores_{tag_for_file}.parquet")
        self.ic_series.to_csv(out / "ic_series.csv", header=True)

        tag = self.tag or "wf"
        export_diagnostics(self._diagnostics, out / f"training_diagnostics_{tag}.csv")
        export_feature_importance(
            self._feature_importance_rows,
            out / f"feature_importance_{tag}.csv",
        )
        if self.save_models and saved_model_entries:
            manifest_metadata = {
                "feature_names": list(dataset.feature_names),
                "rebalance_freq": self.rebalance_freq,
                "hold_period": self.hold_period,
                "label_mode": self.label_mode,
                "device": self.device,
                "params": {
                    "ridge": RIDGE_PARAMS,
                    "ridge_cv_alphas": RIDGE_CV_ALPHAS,
                    "lgbm": LGBM_PARAMS,
                    "xgb": XGB_PARAMS,
                    "cat": CAT_PARAMS,
                    "rf": RF_PARAMS,
                    "mlp": MLP_PARAMS,
                },
            }
            save_models_manifest(
                saved_model_entries, out / "models",
                metadata=manifest_metadata,
            )

        return self.score_df

    def ic_summary(self):
        ic = self.ic_series.dropna()
        std = ic.std()
        return pd.Series({
            "IC均值":   round(ic.mean(), 4),
            "IC标准差": round(std, 4),
            "ICIR":     round(ic.mean() / std, 4) if std > 0 else np.nan,
            "IC>0胜率": round((ic > 0).mean(), 4),
        })

    def save_metrics(self, tag: str, output_dir=None) -> None:
        ic = self.ic_series.dropna()
        std = ic.std()
        out = Path(output_dir) if output_dir else (self.artifact_dir or PROCESSED_DIR)
        out.mkdir(parents=True, exist_ok=True)
        metrics = {
            "tag": tag,
            "trainer_engine": "v2",
            "IC均值": round(ic.mean(), 4),
            "IC标准差": round(std, 4),
            "ICIR": round(ic.mean() / std, 4) if std > 0 else None,
            "IC>0胜率": round((ic > 0).mean(), 4),
            "预测期数": len(ic),
            "wf_selection": self.wf_selection,
            "ensemble_method": self.ensemble_method,
            "label_mode": self.label_mode,
        }
        with open(out / f"model_metrics_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


# 向后兼容别名（原 trainer_v2.WalkForwardTrainerV2）
WalkForwardTrainerV2 = WalkForwardTrainer


# ── Regime-Conditional 训练器 ─────────────────────────────────────────────────
#
# 设计说明：
#   不修改 WalkForwardTrainer 的 __init__ / fit_predict 签名（与并行 worker 解耦）。
#   RegimeConditionalTrainer 作为子类，仅在 fit_predict 内部插入「按 regime 过滤
#   训练日期」的逻辑。子类与父类共享 MLDataset / splits / labels / models / ensemble
#   等全部基础设施，输出格式与 WalkForwardTrainer 完全一致。
#
#   差异点（相对于 WalkForwardTrainer.fit_predict）：
#     1. 进入 fit_predict 后预构建 regime 对齐查找表（ffill 到所有 rebalance_dates）
#     2. 在 per-idx 循环中，train_dates 完成 embargo + purge 之后，按当前 regime
#        过滤；若同 regime 样本数 < min_train_dates_regime，则 fallback 到全量
#        train_dates（避免小样本训练崩溃）
#     3. val_dates 同样按当前 regime 过滤（若不足 2 期则 fallback）
# =============================================================================


class RegimeConditionalTrainer(WalkForwardTrainer):
    """
    Walk-Forward 训练器 + Regime-conditional 训练数据过滤。

    每个 pred_date 的训练样本只取「与当前同 regime」的历史调仓日；
    同 regime 样本不足时回退到全量训练样本（保持稳健性）。

    Parameters
    ----------
    regime_states : pd.Series, optional
        index=date, values={'bull','bear','neutral'}（或任意 regime 标签）。
        None 时退化为标准 WalkForwardTrainer。
    min_train_dates_regime : int
        同 regime 训练样本下限，不足则 fallback 到全量 train_dates。
    min_val_dates_regime : int
        同 regime 验证样本下限，不足则 fallback 到全量 val_dates。

    其余参数与 WalkForwardTrainer 一致。
    """

    def __init__(
        self,
        *args,
        regime_states: pd.Series | None = None,
        min_train_dates_regime: int = 6,
        min_val_dates_regime: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if regime_states is not None and not isinstance(regime_states, pd.Series):
            regime_states = pd.Series(regime_states)
        self.regime_states = regime_states
        self.min_train_dates_regime = int(min_train_dates_regime)
        self.min_val_dates_regime = int(min_val_dates_regime)
        self._regime_aligned: pd.Series | None = None
        logger.info(
            f"RegimeConditionalTrainer: regime="
            f"{'enabled' if regime_states is not None else 'disabled (fallback to base)'}, "
            f"min_train_regime={self.min_train_dates_regime}, "
            f"min_val_regime={self.min_val_dates_regime}"
        )

    def _build_regime_lookup(self, dates: list) -> pd.Series | None:
        """构建 date→regime 的 ffill 对齐查找表。"""
        if self.regime_states is None:
            return None
        rs = self.regime_states
        if not isinstance(rs.index, pd.DatetimeIndex):
            rs.index = pd.to_datetime(rs.index)
        idx = pd.DatetimeIndex(sorted(set(dates) | set(rs.index)))
        aligned = rs.reindex(idx).ffill()
        return aligned

    def _regime_at(self, date, aligned: pd.Series) -> str | None:
        if aligned is None:
            return None
        if date in aligned.index:
            v = aligned.loc[date]
            return v if not pd.isna(v) else None
        prior = aligned[aligned.index <= date]
        if len(prior) == 0:
            return None
        v = prior.iloc[-1]
        return v if not pd.isna(v) else None

    def _filter_by_regime(
        self,
        date_list: list,
        target_regime: str | None,
        aligned: pd.Series,
    ) -> list:
        """返回 date_list 中 regime == target_regime 的子集。"""
        if target_regime is None:
            return list(date_list)
        kept = []
        for d in date_list:
            if self._regime_at(d, aligned) == target_regime:
                kept.append(d)
        return kept

    def fit_predict(self, dataset: MLDataset) -> pd.DataFrame:
        """
        Regime-conditional walk-forward 训练。

        与 WalkForwardTrainer.fit_predict 的唯一差异：在 per-idx 循环中，
        对 train_dates / val_dates 按当前 regime 过滤；同 regime 样本不足
        时 fallback 到全量。
        """
        if self.regime_states is None:
            logger.info("regime_states=None，退化为标准 WalkForwardTrainer")
            return super().fit_predict(dataset)

        self._dataset = dataset
        dates = dataset.rebalance_dates
        n_dates = len(dates)
        min_history = max(self.train_windows) + self.val_window + (
            max(self.train_windows) - self.min_train_window if self.window_specific_val else 0
        )
        predict_start = min_history

        embargo_periods = hold_period_to_embargo_periods(self.hold_period, dates) if self.embargo else 0

        date_to_pos = {d: i for i, d in enumerate(dates)}

        # ── regime 对齐查找表 ─────────────────────────────────────────────────
        regime_aligned = self._build_regime_lookup(dates)
        if regime_aligned is None or regime_aligned.dropna().empty:
            logger.warning("regime_states 全空，退化为标准 WalkForwardTrainer")
            return super().fit_predict(dataset)
        n_regimes = regime_aligned.dropna().nunique()
        logger.info(
            f"Regime-conditional: {n_regimes} 个 regime, "
            f"distribution={regime_aligned.dropna().value_counts().to_dict()}"
        )

        logger.info("预缓存截面数据...")
        section_cache: dict = {}
        for date in dates:
            X, y = dataset.get_cross_section(date)
            if X is not None and len(X) >= MIN_STOCKS_PER_DATE:
                section_cache[date] = (
                    X.values.astype(np.float32),
                    y.values.astype(np.float32),
                    X.index,
                )

        if self._needs_barra_controls():
            logger.info("预计算 Barra 残差化控制矩阵...")
            self._label_controls = precompute_label_controls(
                self.barra_factors, self.industry_map, dates,
            )

        def _stack_cached(date_list, label_mode: str = "raw"):
            X_list, y_list, y_raw_list, w_list = [], [], [], []
            dates_used, stocks_per_date = [], []
            use_barra = label_mode == "barra_residual"
            for i, d in enumerate(date_list):
                if d not in section_cache:
                    continue
                X_np, y_np, stock_idx = section_cache[d]
                barra_df = None
                ind_dummies = None
                if use_barra and self._label_controls and d in self._label_controls:
                    b_full, i_full = self._label_controls[d]
                    barra_df = b_full.reindex(stock_idx).fillna(0.0)
                    ind_dummies = i_full.reindex(stock_idx).fillna(0.0)
                y_t = transform_labels(
                    y_np, label_mode,
                    barra_factors=barra_df, industry_dummies=ind_dummies,
                )
                decay = np.exp(TIME_DECAY * i)
                X_list.append(X_np)
                y_list.append(y_t)
                y_raw_list.append(y_np)
                w_list.extend([decay] * len(y_t))
                dates_used.append(d)
                stocks_per_date.append(int(len(y_t)))
            if not X_list:
                return None, None, None, None, None
            w_arr = np.array(w_list)
            w_arr = normalize_sample_weights_by_universe(
                w_arr, dates_used, stocks_per_date,
            )
            overlap_w = compute_return_overlap_weights(
                dates_used, self.hold_period,
            )
            if len(overlap_w) == len(stocks_per_date):
                overlap_expanded = np.repeat(overlap_w, stocks_per_date)
                if len(overlap_expanded) == len(w_arr):
                    w_arr = (w_arr * overlap_expanded).astype(w_arr.dtype)
            return (
                np.vstack(X_list),
                np.concatenate(y_list),
                w_arr,
                np.concatenate(y_raw_list),
                # group_sizes：每个调仓日的样本数 list[int]，供 rank objective 的
                # LGBMRanker.group / XGBRanker.qid / CatBoostRanker.group_id 使用。
                list(stocks_per_date),
            )

        n_combos = len(self.train_windows) * len(self.model_types)
        n_workers = max(1, min(n_combos, TRAIN_MAX_WORKERS))
        cores_per_job = 1 if n_workers > 1 else (TRAIN_N_JOBS if TRAIN_N_JOBS > 0 else N_CORES)

        logger.info(
            f"Regime-WF: 预测{max(0, n_dates - predict_start)}个调仓日, "
            f"模型={self.model_types}, "
            f"并行={n_workers}任务×{cores_per_job}核/任务"
        )

        all_scores = {}
        models_lock = threading.Lock()
        n_predict = max(0, n_dates - predict_start)
        saved_model_entries = []
        regime_fallback_count = 0

        for idx in range(predict_start, n_dates):
            pred_date = dates[idx]
            if pred_date not in section_cache:
                continue
            X_pred_np, y_true_np, pred_index = section_cache[pred_date]

            current_regime = self._regime_at(pred_date, regime_aligned)

            jobs = []

            for window in self.train_windows:
                ts, te, vs, ve = get_window_splits(
                    idx, window, self.val_window, n_dates,
                    min_train_window=self.min_train_window,
                    window_specific_val=self.window_specific_val,
                )
                train_dates = dates[ts:te]
                val_dates = dates[vs:ve]

                if self.embargo and te > 0:
                    eff_te = embargo_train_end(te, embargo_periods)
                    train_dates = dates[ts:eff_te]

                if self.purge_train:
                    train_dates = purge_train_indices(
                        train_dates, val_dates, pred_date, dates, self.hold_period,
                        date_pos_map=date_to_pos,
                    )

                # ── Regime-conditional 过滤（与父类的唯一差异）──────────────────
                if current_regime is not None:
                    same_train = self._filter_by_regime(
                        train_dates, current_regime, regime_aligned,
                    )
                    if len(same_train) >= self.min_train_dates_regime:
                        train_dates = same_train
                    else:
                        regime_fallback_count += 1
                    same_val = self._filter_by_regime(
                        val_dates, current_regime, regime_aligned,
                    )
                    if len(same_val) >= self.min_val_dates_regime:
                        val_dates = same_val
                    # val 不足时不 fallback，下面 min_train_dates 检查会跳过本折
                # ──────────────────────────────────────────────────────────────

                min_train_dates = max(8, window // 3) - (embargo_periods if self.embargo else 0)
                if len(train_dates) < min_train_dates or len(val_dates) < 2:
                    continue

                for model_type in self.model_types:
                    lm = self._resolve_label_mode(model_type)
                    stacked_tr = _stack_cached(train_dates, lm)
                    stacked_va = _stack_cached(val_dates, lm)
                    if stacked_tr[0] is None or stacked_va[0] is None:
                        continue
                    X_tr, y_tr, w_tr, y_tr_raw, group_tr = stacked_tr
                    X_va, y_va, _, y_va_raw, group_va = stacked_va
                    jobs.append((
                        window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
                        X_va, y_va, y_va_raw, X_pred_np, y_true_np, cores_per_job,
                        self.objective, dataset.feature_names, pred_date, self.device,
                        group_tr, group_va,
                    ))

            if not jobs:
                continue

            results = []
            if n_workers <= 1:
                for j in jobs:
                    r = _run_fold_job(j)
                    if r is not None:
                        results.append(r)
            else:
                raw = Parallel(
                    n_jobs=n_workers,
                    prefer="processes",
                    timeout=3600,
                )(
                    delayed(_run_fold_job)(j) for j in jobs
                )
                for r in raw:
                    if r is not None:
                        results.append(r)

            if not results:
                continue

            model_final_scores = {}
            diag_row = {
                "date": pred_date,
                "pred_ic": np.nan,
                "regime": current_regime,
            }

            dyn_w = dynamic_model_weights(self._rolling_val_ic, self.model_types)

            for model_type in self.model_types:
                folds = [r for r in results if r.model_type == model_type]
                if not folds:
                    continue

                scores_list = [f.pred_scores for f in folds]
                val_ics = [f.val_ic for f in folds]
                train_ics = [f.train_ic for f in folds]

                if self.wf_selection == "best_model":
                    w = select_window_weights(val_ics, "best_window")
                else:
                    w = select_window_weights(val_ics, self.wf_selection)

                combined = combine_model_scores(
                    scores_list, w,
                    method=self.ensemble_method,
                    output_rank=False,
                )
                model_final_scores[model_type] = combined

                for i, f in enumerate(folds):
                    diag_row[f"train_ic_{model_type}_w{f.window}"] = train_ics[i]
                    diag_row[f"val_ic_{model_type}_w{f.window}"] = val_ics[i]
                    with models_lock:
                        self.models[(f.window, model_type)] = f.model
                    if self.save_models and self.artifact_dir is not None:
                        p = save_fold_model(
                            f.model, model_type, f.window, pred_date,
                            self.artifact_dir / "models",
                        )
                        saved_model_entries.append({
                            "path": str(p), "model": model_type,
                            "window": f.window, "date": str(pred_date),
                        })
                    imp = f.importance if f.importance is not None else {}
                    append_feature_importance_rows(
                        self._feature_importance_rows,
                        model_type, f.window, pred_date, imp,
                    )

                self._rolling_val_ic[model_type].append(float(np.nanmean(val_ics)))

            if not model_final_scores:
                continue

            if len(model_final_scores) > 1:
                m_types = list(model_final_scores.keys())
                m_weights = np.array([dyn_w.get(m, 1.0 / len(m_types)) for m in m_types])
                m_weights = m_weights / m_weights.sum()
                final = combine_model_scores(
                    [model_final_scores[m] for m in m_types],
                    m_weights,
                    method=self.ensemble_method,
                    output_rank=self.output_rank,
                )
            else:
                final = combine_model_scores(
                    list(model_final_scores.values()),
                    method=self.ensemble_method,
                    output_rank=self.output_rank,
                )

            pred_ic = spearman_ic(final, y_true_np)
            diag_row["pred_ic"] = pred_ic
            drift = compute_drift_flags(pred_ic, self._drift_history)
            diag_row.update(drift)
            self._drift_history.append({"pred_ic": pred_ic})
            self._diagnostics.append(diag_row)

            all_scores[pred_date] = pd.Series(final, index=pred_index)

            for m, s in model_final_scores.items():
                ic = spearman_ic(s, y_true_np)
                self.model_ic[m][pred_date] = ic

            done = idx - predict_start + 1
            if done % 6 == 0 or done == n_predict:
                logger.info(
                    f"进度 {done}/{n_predict}: {pred_date.date()}, "
                    f"regime={current_regime}, IC={pred_ic:.4f}"
                )

        if regime_fallback_count > 0:
            logger.info(
                f"Regime fallback 触发 {regime_fallback_count} 次"
                f"（同 regime 样本不足，回退到全量训练）"
            )

        self.score_df = pd.DataFrame(all_scores).T
        if not self.score_df.empty:
            self.score_df.index = pd.to_datetime(self.score_df.index)
        self.score_df.index.name = "date"

        if self.score_df.empty:
            logger.warning("无有效预测日，请检查 train_windows / val_window 是否过短")
            self.ic_series = pd.Series(dtype=float)
            return self.score_df

        ic_dict = {}
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = self.score_df.loc[date].dropna()
            y = y.reindex(s.index).dropna()
            s = s.loc[y.index]
            if len(s) >= MIN_STOCKS_PER_DATE:
                ic_dict[date] = spearman_ic(s.values, y.values)
        self.ic_series = pd.Series(ic_dict)

        for m in self.model_types:
            self.model_ic[m] = pd.Series(self.model_ic[m])

        ic_mean = self.ic_series.mean()
        std = self.ic_series.std()
        icir = ic_mean / std if std > 0 else np.nan
        logger.info(
            f"Regime-conditional 完成: IC均值={ic_mean:.4f}, ICIR={icir:.4f}, "
            f"胜率={(self.ic_series > 0).mean():.1%}"
        )

        out = self.artifact_dir or PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)
        tag_for_file = self.tag or "wf"
        self.score_df.to_parquet(out / f"ml_factor_scores_{tag_for_file}.parquet")
        self.ic_series.to_csv(out / "ic_series.csv", header=True)

        tag = self.tag or "wf"
        export_diagnostics(self._diagnostics, out / f"training_diagnostics_{tag}.csv")
        export_feature_importance(
            self._feature_importance_rows,
            out / f"feature_importance_{tag}.csv",
        )
        if self.save_models and saved_model_entries:
            manifest_metadata = {
                "feature_names": list(dataset.feature_names),
                "rebalance_freq": self.rebalance_freq,
                "hold_period": self.hold_period,
                "label_mode": self.label_mode,
                "device": self.device,
                "regime_conditional": True,
                "params": {
                    "ridge": RIDGE_PARAMS,
                    "ridge_cv_alphas": RIDGE_CV_ALPHAS,
                    "lgbm": LGBM_PARAMS,
                    "xgb": XGB_PARAMS,
                    "cat": CAT_PARAMS,
                    "rf": RF_PARAMS,
                    "mlp": MLP_PARAMS,
                },
            }
            save_models_manifest(
                saved_model_entries, out / "models",
                metadata=manifest_metadata,
            )

        return self.score_df


__all__ = [
    "WalkForwardTrainer",
    "WalkForwardTrainerV2",
    "RegimeConditionalTrainer",
    "MLDataset",
    "FoldResult",
    "build_ml_dataset",
    "resolve_train_windows",
    "months_to_rebalance_periods",
    "to_rank",
    "rank_average",
    "spearman_ic",
    "WF_SELECTION_DEFAULT",
    "LABEL_MODE_DEFAULT",
    "ENSEMBLE_METHOD_DEFAULT",
    "TRAIN_WINDOWS_MONTHS",
    "VAL_WINDOW_MONTHS",
    "MODEL_TYPES",
    "REBALANCE_FREQ",
    "TIME_DECAY",
    "MIN_STOCKS_PER_DATE",
    "RIDGE_PARAMS",
    "RIDGE_CV_ALPHAS",
    "LGBM_PARAMS",
    "XGB_PARAMS",
    "CAT_PARAMS",
    "RF_PARAMS",
    "MLP_PARAMS",
]
