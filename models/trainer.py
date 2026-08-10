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
try:
    import catboost as cb  # 可选模型；未安装时 cat 模式不可用，lgbm/xgb/ridge 不受影响
except ModuleNotFoundError:
    cb = None
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
    long_bias_sample_weights,
)
from models.wf.ensemble import (
    combine_model_scores,
    select_window_weights,
    dynamic_model_weights,
)
from models.wf.models import fit_model, predict_model, extract_feature_importance, _sanitize_X
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
VAL_WINDOW_MONTHS    = 6         # 日历月；两窗共用近期 val（h5 周频 ≈ round(6*52/12)=26 期）
TIME_DECAY           = 0.015
MIN_STOCKS_PER_DATE  = 30
REBALANCE_FREQ       = "ME"
MODEL_TYPES          = ["lgbm", "xgb"]   # 默认稳妥组合；cat 为可选模型（ordered boosting 抗过拟合），ridge/rf/mlp 可按需启用


def months_to_rebalance_periods(months: int, rebalance_freq: str) -> int:
    """将日历月数转为调仓期数（Walk-Forward 内部仍按 period 索引回溯）。

    ``months=0`` 原样返回 0（无独立 val；勿被周频 ``max(1, …)`` 抬成 1 期）。
    """
    if months == 0:
        return 0
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
    active_factors: 可选，调仓日 → 当期 rolling-pool 因子名列表（诊断 / 审计）
    lazy_rolling_pool: 若 True，因子面板经 ``lazy_store`` 按需加载（不常驻并集 U）
    lazy_store: RollingPoolPanelStore | None
    always_on_features: 非池因子（Barra/special 等）每期强制纳入
    """
    factor_panel:    dict
    forward_return:  pd.DataFrame
    rebalance_dates: list
    feature_names:   list
    active_factors:  dict | None = None
    lazy_rolling_pool: bool = False
    lazy_store: object | None = None
    always_on_features: list | None = None

    def get_cross_section(self, date, feature_names: list | None = None):
        """
        取调仓日截面。

        feature_names
            指定列集合。lazy rolling-pool 下应为当期预测日的 pool_t
            （+ always_on）；历史 train/val 日也传同一组列，取真实值。
            None 时：
            - lazy rolling-pool → 该 ``date`` 自身的 pool_t（+ always_on）
            - 否则 → 全部 ``factor_panel`` / dataset.feature_names
        """
        dt = pd.Timestamp(date)
        if self.lazy_rolling_pool and self.lazy_store is not None:
            from research.rolling_pool.lazy import (
                build_cross_section_from_store,
                pool_features_for_date,
            )
            if feature_names is None:
                feature_names = pool_features_for_date(
                    self.active_factors, dt,
                    always_on=self.always_on_features,
                )
            # pool_t 语义：列集由调用方固定；不对「历史日是否入池」二次 mask
            return build_cross_section_from_store(
                self.lazy_store,
                self.forward_return,
                dt,
                list(feature_names),
                active_set=None,
            )

        if feature_names is not None:
            rows = {}
            for name in feature_names:
                df = self.factor_panel.get(name)
                if df is not None and dt in df.index:
                    rows[name] = df.loc[dt]
            # 显式列集（pool_t）→ 用真实值；勿按历史日池成员再 mask
            # 仅当未指定 feature_names、走全 U 宽表时，非 lazy 路径才按日 mask
            # （见下方 else 分支后的 active_factors 处理——此处已有列集则跳过）
        else:
            rows = {name: df.loc[dt] for name, df in self.factor_panel.items()
                    if dt in df.index}
            if self.active_factors is not None:
                active = set(self.active_factors.get(dt, []))
                if self.always_on_features:
                    active |= set(self.always_on_features)
                # legacy 非 lazy 全 U：非当日池列 → NaN（再 fillna 0）
                for name in list(rows):
                    if name not in active:
                        rows[name] = pd.Series(
                            np.nan, index=rows[name].index, dtype=np.float32,
                        )
        if not rows:
            return None, None
        # 用 0 填充 NaN（z-score 空间里 0 = 截面中性），避免稀疏因子剔除大量股票
        X_raw = pd.DataFrame(rows)
        if feature_names is not None:
            X_raw = X_raw.reindex(columns=list(feature_names))
        has_data = X_raw.notna().any(axis=1)
        X = X_raw.fillna(0)
        X = X.loc[has_data]
        if dt not in self.forward_return.index:
            return None, None
        y = self.forward_return.loc[dt].reindex(X.index).dropna()
        X = X.loc[X.index.intersection(y.index)]
        return X, y.loc[X.index]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_ml_dataset(
    factor_dict,
    forward_return,
    rebalance_freq=REBALANCE_FREQ,
    active_factors: dict | None = None,
    *,
    lazy_rolling_pool: bool = False,
    lazy_store=None,
    always_on_features: list | None = None,
    feature_names: list | None = None,
    rebalance_dates: list | None = None,
):
    if lazy_rolling_pool:
        if lazy_store is None:
            raise ValueError("lazy_rolling_pool=True 需要 lazy_store")
        if rebalance_dates is None:
            all_dates_idx = pd.DatetimeIndex(forward_return.index)
            rebalance_dates = get_rebalance_dates(all_dates_idx, rebalance_freq).tolist()
        else:
            rebalance_dates = [pd.Timestamp(d) for d in rebalance_dates]
        names = list(feature_names) if feature_names is not None else []
        logger.info(
            f"数据集(lazy rolling-pool): |U|元数据={len(names)}个因子, "
            f"{len(rebalance_dates)}个调仓日（面板按需加载）"
        )
    else:
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
        names = list(factor_dict.keys())
        logger.info(f"数据集: {len(factor_dict)}个因子, {len(rebalance_dates)}个调仓日")

    if active_factors:
        # 仅保留实际调仓日；缺失日补空
        aligned = {}
        for d in rebalance_dates:
            dt = pd.Timestamp(d)
            aligned[dt] = list(active_factors.get(dt, active_factors.get(d, [])))
        n_active = [len(v) for v in aligned.values() if v]
        if n_active:
            logger.info(
                f"rolling_pool active_factors: "
                f"min/median/max="
                f"{min(n_active)}/{int(np.median(n_active))}/{max(n_active)}"
            )
        active_factors = aligned
    return MLDataset(
        factor_panel=factor_dict if factor_dict is not None else {},
        forward_return=forward_return.astype(np.float32),
        rebalance_dates=rebalance_dates,
        feature_names=names,
        active_factors=active_factors,
        lazy_rolling_pool=lazy_rolling_pool,
        lazy_store=lazy_store,
        always_on_features=list(always_on_features) if always_on_features else None,
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


def _lazy_cache_budget_hint(store) -> str:
    """sticky LRU 常驻上限的内存估算提示（float32 面板 T×N）。"""
    prices = getattr(store, "prices", None)
    cap = getattr(store, "max_cached", None)
    if prices is None or not cap:
        return ""
    per_gb = float(prices.shape[0]) * float(prices.shape[1]) * 4 / (1024 ** 3)
    total = per_gb * int(cap)
    hint = f", 常驻上限≈{total:.1f} GB（{per_gb * 1024:.0f} MB/面板）"
    if total > 8.0:
        hint += "；如内存吃紧请调小 --rolling-pool-max-cached"
    return hint


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


def is_retrain_step(
    pred_offset: int,
    retrain_every: int,
    *,
    has_cached_models: bool,
) -> bool:
    """是否在本预测步拟合新模型。

    ``retrain_every<=1`` → 每期重训（旧行为）。
    否则仅在 ``pred_offset % retrain_every == 0``（相对首个预测步）重训；
    尚无缓存模型时强制拟合，避免 skip 空窗。
    """
    every = int(retrain_every)
    if every <= 1:
        return True
    if not has_cached_models:
        return True
    return int(pred_offset) % every == 0


def _run_fold_job(job):
    """模块级单折训练函数（可被 joblib 多进程 pickle）。

    job 结构：
        (window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
         X_va, y_va, y_va_raw, X_pred, y_true, n_jobs,
         objective, feature_names, pred_date, device,
         group_tr, group_va, ridge_drop_regime)

    其中 ``group_tr`` / ``group_va`` 为每个调仓日的样本数 list[int]，
    仅 ``objective='rank'`` 时使用（用于 LGBMRanker.group / XGBRanker.qid /
    CatBoostRanker.group_id）；其它模式下为 None，被忽略。

    ``ridge_drop_regime`` 仅 ridge 模型生效：True 时 fit_model 会按
    ``feature_names`` 剔除 市场/HMM_ regime 列后再拟合（用于 ablation）。
    其它模型忽略。

    返回 FoldResult 或 None（训练异常时）。
    所有依赖（objective / feature_names / pred_date / device / ridge_drop_regime）
    通过 job 显式传入，避免 closure 引用 self 导致无法 pickle。
    """
    (window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
     X_va, y_va, y_va_raw, X_pred, y_true, n_jobs,
     objective, feature_names, pred_date, device,
     group_tr, group_va, ridge_drop_regime) = job
    # 统一在 fit/predict 前清洗 inf：树模型（lgbm/xgb/cat）原生支持 NaN 分裂，
    # 但 XGBoost 对 inf 直接抛 "Input data contains inf"；即便上游 build_factor_dataset
    # 已做 inf→NaN 源头清洗，这里作为防御性兜底，确保任意模型都不会因 inf 失败。
    # _sanitize_X 内部按 model_type 差异化：树模型仅 inf→NaN 保留 NaN 信号，
    # sklearn 模型额外用列均值填实 NaN。
    X_tr = _sanitize_X(X_tr, model_type)
    has_val = X_va is not None and len(X_va) > 0
    X_va = _sanitize_X(X_va, model_type) if has_val else None
    X_pred = _sanitize_X(X_pred, model_type)
    try:
        model = fit_model(
            model_type, X_tr, y_tr, w_tr,
            X_va if has_val else None,
            y_va if has_val else None,
            n_jobs=n_jobs, objective=objective, device=device,
            group_tr=group_tr,
            group_va=group_va if has_val else None,
            feature_names=feature_names, ridge_drop_regime=ridge_drop_regime,
        )
        pred = predict_model(model, X_pred, model_type)
        # 用 predict_model 而非裸 model.predict：ridge_drop_regime=True 时
        # predict_model 会按 _ridge_kept_mask 对齐 X_va/X_tr 列（剔除 regime 列），
        # 否则 RidgeCV 因特征数不匹配抛错（"X has 85 features, expecting 67"）。
        if has_val:
            val_ic = spearman_ic(
                predict_model(model, X_va, model_type),
                y_va_raw if y_va_raw is not None else y_va,
            )
        else:
            val_ic = float("nan")
        train_ic = spearman_ic(
            predict_model(model, X_tr, model_type),
            y_tr_raw if y_tr_raw is not None else y_tr,
        )
        imp = extract_feature_importance(
            model, model_type, feature_names,
            X_va=X_va if has_val else None,
            y_va=(y_va_raw if y_va_raw is not None else y_va) if has_val else None,
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
        window_specific_val: bool = True,  # True=共用近期 val；False=旧错位布局(deprecated)
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
        ridge_drop_regime: bool = False,
        enable_shap: bool = False,
        shap_top: int = 20,
        shap_max_samples: int = 500,
        shap_max_dates: int = 12,
        long_weight_top: float | None = None,
        long_weight_ratio: float = 0.25,
        long_weight_curve: str = "smooth",
        softlong_floor_slope: float = 0.25,
        retrain_every: int = 1,
    ):
        # 默认 train_windows/val_window 为日历月；units=periods 时直接当调仓期数
        self.train_windows, self.val_window = resolve_train_windows(
            train_windows, val_window, rebalance_freq, train_window_units,
        )
        if self.val_window < 0:
            raise ValueError(f"val_window 不能为负: {self.val_window}")
        self.retrain_every = int(retrain_every)
        if self.retrain_every < 1:
            raise ValueError(f"retrain_every 须 >= 1，收到 {retrain_every}")
        # 多窗无 val：ic_weighted / best_* 依赖 val IC，禁止静默用 NaN 加权
        if (
            self.val_window == 0
            and len(self.train_windows) > 1
            and wf_selection != "average"
        ):
            raise ValueError(
                "多窗且 val_window=0 时无法计算 val IC 加权；"
                "请使用 --wf-selection average（等权），或设置 --val-window > 0"
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
        # 多头偏置样本权重：标签保持 cs_rank/cs_zscore 连续；仅抬高 top 区 loss 权重。
        # long_weight_top=None/≤0 → 关闭；ridge 已支持 sample_weight（YetiRank 忽略）。
        self.long_weight_top = (
            None if long_weight_top is None or float(long_weight_top) <= 0
            else float(long_weight_top)
        )
        self.long_weight_ratio = float(long_weight_ratio)
        self.long_weight_curve = str(long_weight_curve)
        self.softlong_floor_slope = float(softlong_floor_slope)
        # SHAP（默认关）：仅在最近 shap_max_dates 个预测日、每折最多
        # shap_max_samples 行上算 Tree/Linear SHAP，避免拖慢全量 WF。
        self.enable_shap = bool(enable_shap)
        self.shap_top = int(shap_top)
        self.shap_max_samples = int(shap_max_samples)
        self.shap_max_dates = int(shap_max_dates)
        # rank objective 默认配合 cs_rank 百分位标签；fit 时经 prepare_rank_labels
        # 转为截面细整数秩（0..n-1）供 LGBM/XGB/Cat Ranker 共用。仅在用户未显式
        # 覆盖 label_mode（仍是默认 cs_zscore）时自动切换。
        if (
            objective == "rank"
            and isinstance(label_mode, str)
            and label_mode == LABEL_MODE_DEFAULT
        ):
            self.label_mode = "cs_rank"
            logger.info(
                "rank objective 检测到默认 label_mode='cs_zscore'，自动切换为 'cs_rank' "
                "（fit 时再转为细整数秩）；如需保留 cs_zscore 请显式传入 label_mode。"
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
        # ridge_drop_regime：仅 ridge 模型生效，剔除 市场/HMM_ regime 列（TS-zscore
        # 语义）后拟合，用于 ablation；默认 False 保持向后兼容。详见
        # models/wf/models.py::fit_model。
        self.ridge_drop_regime = bool(ridge_drop_regime)
        if self.ridge_drop_regime:
            logger.info(
                "ridge_drop_regime=True：ridge 拟合时剔除 市场/HMM_ regime 列"
                "（仅 ridge 模型生效；其它模型忽略）"
            )

        self.score_df: pd.DataFrame | None = None
        self.ic_series: pd.Series | None = None
        self.models: dict = {}
        self.model_ic: dict = {m: {} for m in self.model_types}
        self._diagnostics: list[dict] = []
        self._drift_history: list[dict] = []
        self._rolling_val_ic: dict[str, list[float]] = {m: [] for m in self.model_types}
        self._feature_importance_rows: list[dict] = []
        self._shap_rows: list[dict] = []
        self._dataset: MLDataset | None = None

        logger.info(
            f"Walk-Forward 窗口: train={self.train_windows}, val={self.val_window} 期 "
            f"(units={train_window_units}, freq={rebalance_freq}); "
            f"wf_selection={wf_selection}, label_mode={self.label_mode}, "
            f"ensemble={ensemble_method}, purge={purge_train}, embargo={embargo}, "
            f"objective={self.objective}, retrain_every={self.retrain_every}"
            + (
                f", shap=on(top={self.shap_top}, max_samples={self.shap_max_samples}, "
                f"max_dates={self.shap_max_dates})"
                if self.enable_shap else ", shap=off"
            )
        )
        if self.retrain_every > 1:
            logger.info(
                f"retrain_every={self.retrain_every}: 预测日复用最近重训日拟合的模型；"
                f"重训日 train 窗相对该日 purge/embargo，中间期只 predict"
            )
        _modes = (
            set(self.label_mode.values()) if isinstance(self.label_mode, dict)
            else {self.label_mode}
        )
        if "top40_cs_zscore" in _modes:
            logger.info(
                "label_mode=top40_cs_zscore 公式: "
                "z=cs_zscore(fwd_ret) 全截面；keep=rank_pct(fwd_ret)>=0.60（前40%）；"
                "y'=where(keep, z, 0)（阈下置常数 0，非区内 re-zscore）"
            )
        if "cs_rank_softlong" in _modes:
            logger.info(
                "label_mode=cs_rank_softlong 公式: "
                f"r=cs_rank(fwd_ret)；τ=1-top_frac；"
                f"r≥τ → (r-τ)/(1-τ)；r<τ → floor_slope*(r-τ)/τ "
                f"(floor_slope={self.softlong_floor_slope})"
            )
        if self.long_weight_top is not None:
            logger.info(
                "long-bias sample_weight: "
                f"top_frac={self.long_weight_top}, bottom_weight={self.long_weight_ratio}, "
                f"curve={self.long_weight_curve} "
                "（标签本身不变；w 乘到既有 time-decay / universe / overlap 权重上）"
            )

    def _section_base_weights(self, y_raw: np.ndarray, decay: float) -> np.ndarray:
        """单截面基础样本权重 = time-decay × 可选 long-bias。"""
        w = np.full(len(y_raw), float(decay), dtype=np.float64)
        if self.long_weight_top is not None:
            w = w * long_bias_sample_weights(
                y_raw,
                top_frac=self.long_weight_top,
                bottom_weight=self.long_weight_ratio,
                curve=self.long_weight_curve,
            )
        return w

    def _transform_y(
        self,
        y_np: np.ndarray,
        label_mode: str,
        barra_df=None,
        ind_dummies=None,
    ) -> np.ndarray:
        """transform_labels 薄封装：透传 softlong 的 floor_slope。"""
        return transform_labels(
            y_np, label_mode,
            barra_factors=barra_df, industry_dummies=ind_dummies,
            floor_slope=self.softlong_floor_slope,
        )

    def _should_compute_shap(self, done: int, n_predict: int) -> bool:
        """仅最近 shap_max_dates 个预测日计算 SHAP（0/负数 = 全部日期）。"""
        if not self.enable_shap:
            return False
        if self.shap_max_dates is None or self.shap_max_dates <= 0:
            return True
        return done > max(0, n_predict - self.shap_max_dates)

    def _record_fold_shap(
        self,
        fold: FoldResult,
        X_pred: np.ndarray,
        pred_date,
        window_weight: float,
        feature_names: list[str],
    ) -> None:
        """单折：用该折模型 + 预测日截面特征算 SHAP（防泄漏）。"""
        from models.wf.shap_analysis import (
            append_shap_rows,
            compute_fold_shap_summary,
        )
        summary, method = compute_fold_shap_summary(
            fold.model, fold.model_type, X_pred, feature_names,
            max_samples=self.shap_max_samples,
            random_state=42,
        )
        append_shap_rows(
            self._shap_rows, summary,
            model_type=fold.model_type,
            window=fold.window,
            pred_date=pred_date,
            method=method,
            weight=float(window_weight) if np.isfinite(window_weight) else 0.0,
        )

    def _finalize_shap_export(self) -> None:
        if not self.enable_shap:
            return
        from models.wf.shap_analysis import export_shap_artifacts

        out = self.artifact_dir or PROCESSED_DIR
        tag = self.tag or "wf"
        # 模型级权重：用各模型样本外 IC 均值（>0）归一；否则等权
        mw: dict[str, float] = {}
        for m in self.model_types:
            s = self.model_ic.get(m)
            if s is None:
                continue
            if isinstance(s, dict):
                s = pd.Series(s)
            if len(s) == 0:
                continue
            v = float(np.nanmean(np.clip(np.asarray(s, dtype=float), 0, None)))
            mw[m] = v if v > 0 else 0.0
        if mw and sum(mw.values()) <= 0:
            mw = {m: 1.0 for m in mw}
        export_shap_artifacts(
            self._shap_rows, out, tag,
            top_n=self.shap_top,
            model_weights=mw or None,
            meta_extra=self._shap_meta_extra(),
        )

    def _shap_meta_extra(self) -> dict:
        ds = self._dataset
        lazy = bool(getattr(ds, "lazy_rolling_pool", False))
        meta = {
            "mode": "walk_forward_pred_fold",
            "max_samples": self.shap_max_samples,
            "max_dates": self.shap_max_dates,
            "lazy_rolling_pool": lazy,
        }
        names = list(ds.feature_names) if ds else []
        if lazy:
            # 每期列集 = 当期 pool_t；并集仅为元数据，不能当作固定 feature_names
            meta["feature_names"] = None
            meta["feature_names_union_metadata"] = names
            meta["feature_names_note"] = (
                "lazy rolling-pool：SHAP 按折记录，列集为该折当期 pool_t(+always_on)"
            )
        else:
            meta["feature_names"] = names
        return meta

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

    def _fit_predict_lazy(self, dataset: MLDataset) -> pd.DataFrame:
        """
        rolling-pool lazy 路径：禁止预缓存固定列宽截面。

        每个预测/调仓日 t：特征列 = schedule 当日 pool_t（+ always_on）；
        该期 train / val / pred 全部只用 pool_t（历史截面取同列真实值）。
        禁止对窗内多调仓日的池取并集。运行时仅 ``ensure(pool_t)``。
        """
        from research.rolling_pool.lazy import log_rss, pool_features_for_date

        self._dataset = dataset
        dates = dataset.rebalance_dates
        n_dates = len(dates)
        # 共用近期 val：min_history = max(W)+V；旧错位布局才额外加 (maxW-minW)
        min_history = max(self.train_windows) + self.val_window + (
            max(self.train_windows) - self.min_train_window
            if not self.window_specific_val else 0
        )
        predict_start = min_history
        embargo_periods = (
            hold_period_to_embargo_periods(self.hold_period, dates) if self.embargo else 0
        )
        date_to_pos = {d: i for i, d in enumerate(dates)}
        store = dataset.lazy_store
        always_on = list(dataset.always_on_features or [])

        log_rss("wf_lazy_start")
        # sticky LRU：每期只 ensure(pool_t)，但**不**每期清空。池换手仅 ~20%，
        # 跨期保留能让大部分因子直接命中内存，避免重复读盘 / 重复残差化。
        # 常驻规模仍由 store.max_cached（--rolling-pool-max-cached）封顶，
        # 超限由 LRU 淘汰最久未用者；特征语义不变（每期仍是 pool_t + always_on）。
        logger.info(
            "Walk-Forward lazy rolling-pool: 每期 ensure(pool_t)，sticky LRU 跨期保留 "
            f"(max_cached={getattr(store, 'max_cached', '?')}"
            f"{_lazy_cache_budget_hint(store)})"
        )

        if self._needs_barra_controls():
            logger.info("预计算 Barra 残差化控制矩阵...")
            self._label_controls = precompute_label_controls(
                self.barra_factors, self.industry_map, dates,
            )

        if self._needs_triple_barrier():
            if self.prices is None or self.open_prices is None:
                raise ValueError(
                    "label_mode='triple_barrier' 需要 prices 与 open_prices"
                )
            tb_kwargs = {
                "hold_period": self.hold_period,
                "vol_window": int(self.triple_barrier_params.get("vol_window", 20)),
                "upper_mult": float(self.triple_barrier_params.get("upper_mult", 2.0)),
                "lower_mult": float(self.triple_barrier_params.get("lower_mult", 1.5)),
                "label_type": self.triple_barrier_params.get("label_type", "sign"),
            }
            self._barrier_labels = triple_barrier_label(
                self.prices, self.open_prices, dates, **tb_kwargs,
            )

        def _stack_dates(date_list, feature_names, label_mode: str = "raw"):
            X_list, y_list, y_raw_list, w_list = [], [], [], []
            dates_used, stocks_per_date = [], []
            use_barra = label_mode == "barra_residual"
            use_tb = label_mode == "triple_barrier"
            for i, d in enumerate(date_list):
                X, y = dataset.get_cross_section(d, feature_names=feature_names)
                if X is None or y is None or len(X) < MIN_STOCKS_PER_DATE:
                    continue
                X_np = X.to_numpy(dtype=np.float32, copy=False)
                y_np = y.to_numpy(dtype=np.float32, copy=False)
                stock_idx = X.index
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
                    barra_df = b_full.reindex(stock_idx).fillna(0.0)
                    ind_dummies = i_full.reindex(stock_idx).fillna(0.0)
                y_t = self._transform_y(
                    y_np, label_mode, barra_df=barra_df, ind_dummies=ind_dummies,
                )
                decay = np.exp(TIME_DECAY * i)
                X_list.append(X_np)
                y_list.append(y_t)
                y_raw_list.append(y_np)
                w_list.append(self._section_base_weights(y_np, decay))
                dates_used.append(d)
                stocks_per_date.append(int(len(y_t)))
            if not X_list:
                return None, None, None, None, None
            w_arr = np.concatenate(w_list)
            w_arr = normalize_sample_weights_by_universe(
                w_arr, dates_used, stocks_per_date,
            )
            overlap_w = compute_return_overlap_weights(dates_used, self.hold_period)
            if len(overlap_w) == len(stocks_per_date):
                overlap_expanded = np.repeat(overlap_w, stocks_per_date)
                if len(overlap_expanded) == len(w_arr):
                    w_arr = (w_arr * overlap_expanded).astype(w_arr.dtype)
            return (
                np.vstack(X_list),
                np.concatenate(y_list),
                w_arr,
                np.concatenate(y_raw_list),
                list(stocks_per_date),
            )

        n_combos = len(self.train_windows) * len(self.model_types)
        n_workers = max(1, min(n_combos, TRAIN_MAX_WORKERS))
        # lazy 路径：截面按折重建，多进程 pickle store 成本高 → 强制串行任务
        if n_workers > 1:
            logger.info(
                f"lazy rolling-pool: 将并行任务 {n_workers}→1"
                f"（避免跨进程复制 panel store）"
            )
            n_workers = 1
        cores_per_job = TRAIN_N_JOBS if TRAIN_N_JOBS > 0 else N_CORES

        logger.info(
            f"Walk-Forward lazy: 预测{max(0, n_dates - predict_start)}个调仓日, "
            f"模型={self.model_types}, 并行={n_workers}任务×{cores_per_job}核/任务, "
            f"retrain_every={self.retrain_every}"
        )

        all_scores = {}
        models_lock = threading.Lock()
        n_predict = max(0, n_dates - predict_start)
        saved_model_entries = []
        peak_rss = log_rss("wf_lazy_loop_enter")
        peak_pool_cols = 0
        last_pool_cols: list[str] = []
        cached_folds: dict[tuple[int, str], FoldResult] = {}
        cached_feature_names: list[str] | None = None
        last_retrain_date = None
        n_fit_steps = 0
        n_reuse_steps = 0

        for idx in range(predict_start, n_dates):
            pred_date = dates[idx]
            # 当期 pool_t：本预测日 train/val/pred 共用；禁止窗内并集
            cols = pool_features_for_date(
                dataset.active_factors, pred_date, always_on=always_on,
            )
            if not cols:
                continue
            peak_pool_cols = max(peak_pool_cols, len(cols))
            last_pool_cols = list(cols)
            if store is not None:
                store.ensure(cols)

            pred_offset = idx - predict_start
            cols_match = (
                cached_feature_names is not None
                and list(cols) == list(cached_feature_names)
            )
            need_retrain = is_retrain_step(
                pred_offset, self.retrain_every, has_cached_models=bool(cached_folds),
            )
            # rolling-pool 列集变化时不可复用旧模型（特征维不一致）
            if not cols_match:
                need_retrain = True

            X_pred_df, y_true_s = dataset.get_cross_section(
                pred_date, feature_names=cols,
            )
            if X_pred_df is None or len(X_pred_df) < MIN_STOCKS_PER_DATE:
                continue
            X_pred_np = X_pred_df.to_numpy(dtype=np.float32, copy=False)
            y_true_np = y_true_s.to_numpy(dtype=np.float32, copy=False)
            pred_index = X_pred_df.index

            results: list[FoldResult] = []
            jobs = []
            fold_metas = []

            if not need_retrain:
                for (window, model_type), cached in cached_folds.items():
                    if cached.model is None:
                        continue
                    try:
                        pred = predict_model(cached.model, X_pred_np, model_type)
                    except Exception as e:
                        logger.warning(
                            f"lazy 复用预测失败({model_type}, w={window}, "
                            f"{pred_date.date()}): {e}"
                        )
                        continue
                    results.append(FoldResult(
                        window, model_type, pred,
                        cached.val_ic, cached.train_ic,
                        cached.model, cached.importance,
                    ))
                    fold_metas.append({
                        "feature_names": cols,
                        "pred_index": pred_index,
                        "y_true_np": y_true_np,
                    })
                if results:
                    n_reuse_steps += 1
                    if n_reuse_steps <= 3 or n_reuse_steps % self.retrain_every == 0:
                        logger.info(
                            f"retrain_every={self.retrain_every}: lazy skip fit @ "
                            f"{pred_date.date()}, 复用模型自 {last_retrain_date.date()}"
                        )
                else:
                    need_retrain = True

            if need_retrain:
                jobs = []
                fold_metas = []

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

                    min_train_dates = max(8, window // 3) - (
                        embargo_periods if self.embargo else 0
                    )
                    no_val = self.val_window == 0
                    if len(train_dates) < min_train_dates:
                        continue
                    if not no_val and len(val_dates) < 2:
                        continue

                    for model_type in self.model_types:
                        lm = self._resolve_label_mode(model_type)
                        stacked_tr = _stack_dates(train_dates, cols, lm)
                        if stacked_tr[0] is None:
                            continue
                        if no_val:
                            X_va = y_va = y_va_raw = group_va = None
                        else:
                            stacked_va = _stack_dates(val_dates, cols, lm)
                            if stacked_va[0] is None:
                                continue
                            X_va, y_va, _, y_va_raw, group_va = stacked_va
                        X_tr, y_tr, w_tr, y_tr_raw, group_tr = stacked_tr
                        jobs.append((
                            window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
                            X_va, y_va, y_va_raw, X_pred_np, y_true_np, cores_per_job,
                            self.objective, cols, pred_date, self.device,
                            group_tr, group_va,
                            self.ridge_drop_regime,
                        ))
                        fold_metas.append({
                            "feature_names": cols,
                            "pred_index": pred_index,
                            "y_true_np": y_true_np,
                        })

                if not jobs:
                    continue

                results = []
                result_metas = []
                for j, meta in zip(jobs, fold_metas):
                    r = _run_fold_job(j)
                    if r is not None:
                        results.append(r)
                        result_metas.append(meta)

                if not results:
                    continue

                fold_metas = result_metas
                cached_folds = {(f.window, f.model_type): f for f in results}
                cached_feature_names = list(cols)
                last_retrain_date = pred_date
                n_fit_steps += 1
                if self.retrain_every > 1:
                    logger.info(
                        f"retrain_every={self.retrain_every}: lazy fit @ "
                        f"{pred_date.date()} (step {n_fit_steps}, offset={pred_offset})"
                    )

            if not results:
                continue

            # 同 pred_date：各 window 共用 pool_t 列集；取第一折 pred_index 对齐股票
            pred_index = fold_metas[0]["pred_index"] if fold_metas else pred_index
            y_true_np = fold_metas[0]["y_true_np"] if fold_metas else y_true_np

            model_final_scores = {}
            # n_pool_features：当期真实特征列数（lazy 下不是固定 |U|）
            diag_row = {
                "date": pred_date,
                "pred_ic": np.nan,
                "n_pool_features": len(cols),
                "reused_model": int(not need_retrain),
                "fit_date": last_retrain_date if last_retrain_date is not None else pred_date,
            }
            dyn_w = dynamic_model_weights(self._rolling_val_ic, self.model_types)

            for model_type in self.model_types:
                folds = [r for r in results if r.model_type == model_type]
                if not folds:
                    continue
                # 对齐到共同 pred_index 长度（各折 X_pred 同行）
                scores_list = []
                ok_folds = []
                for f in folds:
                    if len(f.pred_scores) != len(pred_index):
                        # 不同 window 若股票集不一致则按 index 重对齐（少见）
                        logger.warning(
                            f"lazy fold score len mismatch: {len(f.pred_scores)} "
                            f"vs {len(pred_index)} ({model_type} w={f.window})"
                        )
                        continue
                    scores_list.append(f.pred_scores)
                    ok_folds.append(f)
                if not ok_folds:
                    continue
                folds = ok_folds
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

                done_preview = idx - predict_start + 1
                do_shap = need_retrain and self._should_compute_shap(
                    done_preview, n_predict,
                )

                for i, f in enumerate(folds):
                    diag_row[f"train_ic_{model_type}_w{f.window}"] = train_ics[i]
                    diag_row[f"val_ic_{model_type}_w{f.window}"] = val_ics[i]
                    with models_lock:
                        self.models[(f.window, model_type)] = f.model
                    if need_retrain and self.save_models and self.artifact_dir is not None:
                        p = save_fold_model(
                            f.model, model_type, f.window, pred_date,
                            self.artifact_dir / "models",
                        )
                        saved_model_entries.append({
                            "path": str(p), "model": model_type,
                            "window": f.window, "date": str(pred_date),
                        })
                    if need_retrain:
                        imp = f.importance if f.importance is not None else {}
                        append_feature_importance_rows(
                            self._feature_importance_rows,
                            model_type, f.window, pred_date, imp,
                        )
                    if do_shap and f.model is not None:
                        ww = float(w[i]) if i < len(w) else 0.0
                        # 找该 fold 对应的 X_pred / feature_names
                        feat_names = cols
                        X_for_shap = X_pred_np
                        if jobs:
                            for j, meta in zip(jobs, fold_metas):
                                if j[0] == f.window and j[1] == model_type:
                                    feat_names = meta["feature_names"]
                                    X_for_shap = j[9]
                                    break
                        self._record_fold_shap(
                            f, X_for_shap, pred_date,
                            window_weight=ww,
                            feature_names=feat_names,
                        )

                self._rolling_val_ic[model_type].append(float(np.nanmean(val_ics)))

            if not model_final_scores:
                continue

            if len(model_final_scores) > 1:
                m_types = list(model_final_scores.keys())
                m_weights = np.array(
                    [dyn_w.get(m, 1.0 / len(m_types)) for m in m_types]
                )
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
                self.model_ic[m][pred_date] = spearman_ic(s, y_true_np)

            # 期后不释放（sticky LRU）：池换手仅 ~20%，跨期命中省下大量读盘 /
            # 残差化；常驻规模由 store.max_cached 封顶，超限走 LRU 淘汰。
            # 需要压内存时调小 --rolling-pool-max-cached（≈|pool_t| 即退回旧行为）。
            done = idx - predict_start + 1
            if done % 6 == 0 or done == n_predict:
                rss = log_rss(f"wf_lazy {done}/{n_predict}")
                if np.isfinite(rss):
                    peak_rss = max(peak_rss, rss) if np.isfinite(peak_rss) else rss
                store_s = store.stats_summary() if store is not None else ""
                logger.info(
                    f"进度 {done}/{n_predict}: {pred_date.date()}, IC={pred_ic:.4f}, "
                    f"peak_pool_cols={peak_pool_cols}, {store_s}"
                )

        self.score_df = pd.DataFrame(all_scores).T
        if not self.score_df.empty:
            self.score_df.index = pd.to_datetime(self.score_df.index)
        self.score_df.index.name = "date"

        if store is not None:
            logger.info(
                f"lazy store 汇总: {store.stats_summary()}, "
                f"peak_pool_cols={peak_pool_cols}"
            )
        log_rss("wf_lazy_done")
        if np.isfinite(peak_rss):
            logger.info(f"[mem] wf_lazy peak RSS observed≈{peak_rss:.2f} GB")

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
            f"胜率={(self.ic_series > 0).mean():.1%}, "
            f"fit={n_fit_steps}, reuse={n_reuse_steps} "
            f"(retrain_every={self.retrain_every})"
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
        self._finalize_shap_export()
        if self.save_models and saved_model_entries:
            # lazy rolling-pool 下没有全局固定列集：每期特征 = 当期 pool_t(+always_on)。
            # 不把并集 |U| 谎报成 feature_names；只给出当期/最后一期与并集元数据。
            manifest_metadata = {
                "feature_names": None,
                "feature_names_note": (
                    "lazy rolling-pool：无全局固定列集；每期特征列 = 当期 "
                    "pool_t(+always_on)，见各 entry 的 date 与 schedule"
                ),
                "feature_names_union_metadata": list(dataset.feature_names),
                "feature_names_last_pool": list(last_pool_cols),
                "always_on_features": list(always_on),
                "peak_pool_cols": int(peak_pool_cols),
                "rebalance_freq": self.rebalance_freq,
                "hold_period": self.hold_period,
                "label_mode": self.label_mode,
                "device": self.device,
                "retrain_every": self.retrain_every,
                "params": {
                    "ridge": RIDGE_PARAMS,
                    "ridge_cv_alphas": RIDGE_CV_ALPHAS,
                    "lgbm": LGBM_PARAMS,
                    "xgb": XGB_PARAMS,
                    "cat": CAT_PARAMS,
                    "rf": RF_PARAMS,
                    "mlp": MLP_PARAMS,
                },
                "lazy_rolling_pool": True,
            }
            save_models_manifest(
                saved_model_entries, out / "models",
                metadata=manifest_metadata,
            )
        return self.score_df

    def fit_predict(self, dataset: MLDataset) -> pd.DataFrame:
        if getattr(dataset, "lazy_rolling_pool", False):
            return self._fit_predict_lazy(dataset)
        self._dataset = dataset
        dates = dataset.rebalance_dates
        n_dates = len(dates)
        min_history = max(self.train_windows) + self.val_window + (
            max(self.train_windows) - self.min_train_window
            if not self.window_specific_val else 0
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
                y_t = self._transform_y(
                    y_np, label_mode, barra_df=barra_df, ind_dummies=ind_dummies,
                )
                decay = np.exp(TIME_DECAY * i)
                X_list.append(X_np)
                y_list.append(y_t)
                y_raw_list.append(y_np)
                w_list.append(self._section_base_weights(y_np, decay))
                dates_used.append(d)
                stocks_per_date.append(int(len(y_t)))
            if not X_list:
                return None, None, None, None, None
            w_arr = np.concatenate(w_list)
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
            f"(TRAIN_MAX_WORKERS={TRAIN_MAX_WORKERS}, retrain_every={self.retrain_every})"
        )

        all_scores = {}
        models_lock = threading.Lock()
        n_predict = max(0, n_dates - predict_start)
        saved_model_entries = []
        # retrain_every>1：缓存最近一次重训折结果，中间预测日只 predict
        cached_folds: dict[tuple[int, str], FoldResult] = {}
        last_retrain_date = None
        n_fit_steps = 0
        n_reuse_steps = 0

        for idx in range(predict_start, n_dates):
            pred_date = dates[idx]
            if pred_date not in section_cache:
                continue
            X_pred_np, y_true_np, pred_index = section_cache[pred_date]
            pred_offset = idx - predict_start
            need_retrain = is_retrain_step(
                pred_offset, self.retrain_every, has_cached_models=bool(cached_folds),
            )

            results: list[FoldResult] = []
            if not need_retrain:
                # 复用最近重训日模型：train 窗/purge/embargo 已相对重训日完成，无泄漏
                for (window, model_type), cached in cached_folds.items():
                    if cached.model is None:
                        continue
                    try:
                        pred = predict_model(cached.model, X_pred_np, model_type)
                    except Exception as e:
                        logger.warning(
                            f"复用预测失败({model_type}, w={window}, "
                            f"{pred_date.date()}): {e}"
                        )
                        continue
                    results.append(FoldResult(
                        window, model_type, pred,
                        cached.val_ic, cached.train_ic,
                        cached.model, cached.importance,
                    ))
                if results:
                    n_reuse_steps += 1
                    if n_reuse_steps <= 3 or n_reuse_steps % self.retrain_every == 0:
                        logger.info(
                            f"retrain_every={self.retrain_every}: skip fit @ "
                            f"{pred_date.date()}, 复用模型自 {last_retrain_date.date()}"
                        )
                else:
                    # 缓存失效 → 回退本折重训
                    need_retrain = True

            if need_retrain:
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
                    no_val = self.val_window == 0
                    if len(train_dates) < min_train_dates:
                        continue
                    if not no_val and len(val_dates) < 2:
                        continue

                    for model_type in self.model_types:
                        lm = self._resolve_label_mode(model_type)
                        stacked_tr = _stack_cached(train_dates, lm)
                        if stacked_tr[0] is None:
                            continue
                        if no_val:
                            X_va = y_va = y_va_raw = group_va = None
                        else:
                            stacked_va = _stack_cached(val_dates, lm)
                            if stacked_va[0] is None:
                                continue
                            X_va, y_va, _, y_va_raw, group_va = stacked_va
                        X_tr, y_tr, w_tr, y_tr_raw, group_tr = stacked_tr
                        jobs.append((
                            window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
                            X_va, y_va, y_va_raw, X_pred_np, y_true_np, cores_per_job,
                            # 多进程 pickle 所需的额外上下文（_run_fold_job 为模块级函数）
                            self.objective, dataset.feature_names, pred_date, self.device,
                            # rank objective 所需的逐期 group 大小（非 rank 模式被忽略）
                            group_tr, group_va,
                            # ridge ablation：剔除 regime 列（仅 ridge 生效）
                            self.ridge_drop_regime,
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

                cached_folds = {
                    (f.window, f.model_type): f for f in results
                }
                last_retrain_date = pred_date
                n_fit_steps += 1
                if self.retrain_every > 1:
                    logger.info(
                        f"retrain_every={self.retrain_every}: fit @ {pred_date.date()} "
                        f"(step {n_fit_steps}, offset={pred_offset})"
                    )

            if not results:
                continue

            # Group by model_type: combine windows with IC weights
            model_final_scores = {}
            diag_row = {
                "date": pred_date,
                "pred_ic": np.nan,
                "reused_model": int(not need_retrain),
                "fit_date": last_retrain_date if last_retrain_date is not None else pred_date,
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

                done_preview = idx - predict_start + 1
                do_shap = need_retrain and self._should_compute_shap(
                    done_preview, n_predict,
                )

                for i, f in enumerate(folds):
                    diag_row[f"train_ic_{model_type}_w{f.window}"] = train_ics[i]
                    diag_row[f"val_ic_{model_type}_w{f.window}"] = val_ics[i]
                    with models_lock:
                        self.models[(f.window, model_type)] = f.model
                    if need_retrain and self.save_models and self.artifact_dir is not None:
                        p = save_fold_model(
                            f.model, model_type, f.window, pred_date,
                            self.artifact_dir / "models",
                        )
                        saved_model_entries.append({
                            "path": str(p), "model": model_type,
                            "window": f.window, "date": str(pred_date),
                        })
                    if need_retrain:
                        imp = f.importance if f.importance is not None else {}
                        append_feature_importance_rows(
                            self._feature_importance_rows,
                            model_type, f.window, pred_date, imp,
                        )
                    if do_shap and f.model is not None:
                        # 折内只记窗口 IC 权重；模型间权重在 _finalize_shap_export 施加
                        ww = float(w[i]) if i < len(w) else 0.0
                        self._record_fold_shap(
                            f, X_pred_np, pred_date,
                            window_weight=ww,
                            feature_names=dataset.feature_names,
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
            f"胜率={(self.ic_series > 0).mean():.1%}, "
            f"fit={n_fit_steps}, reuse={n_reuse_steps} "
            f"(retrain_every={self.retrain_every})"
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
        self._finalize_shap_export()
        if self.save_models and saved_model_entries:
            manifest_metadata = {
                "feature_names": list(dataset.feature_names),
                "rebalance_freq": self.rebalance_freq,
                "hold_period": self.hold_period,
                "label_mode": self.label_mode,
                "device": self.device,
                "retrain_every": self.retrain_every,
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
            "retrain_every": self.retrain_every,
            "long_weight_top": self.long_weight_top,
            "long_weight_ratio": self.long_weight_ratio,
            "long_weight_curve": self.long_weight_curve,
            "softlong_floor_slope": self.softlong_floor_slope,
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

        if getattr(dataset, "lazy_rolling_pool", False):
            logger.warning(
                "regime_conditional + rolling_pool_lazy：暂无 regime 过滤的 lazy 实现，"
                "退化为 WalkForwardTrainer._fit_predict_lazy（不按 regime 筛训练日）"
            )
            return self._fit_predict_lazy(dataset)

        if self.retrain_every > 1:
            logger.warning(
                f"RegimeConditionalTrainer 暂不支持 retrain_every={self.retrain_every}，"
                "按每期重训执行"
            )

        self._dataset = dataset
        dates = dataset.rebalance_dates
        n_dates = len(dates)
        min_history = max(self.train_windows) + self.val_window + (
            max(self.train_windows) - self.min_train_window
            if not self.window_specific_val else 0
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
                y_t = self._transform_y(
                    y_np, label_mode, barra_df=barra_df, ind_dummies=ind_dummies,
                )
                decay = np.exp(TIME_DECAY * i)
                X_list.append(X_np)
                y_list.append(y_t)
                y_raw_list.append(y_np)
                w_list.append(self._section_base_weights(y_np, decay))
                dates_used.append(d)
                stocks_per_date.append(int(len(y_t)))
            if not X_list:
                return None, None, None, None, None
            w_arr = np.concatenate(w_list)
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
                no_val = self.val_window == 0
                if len(train_dates) < min_train_dates:
                    continue
                if not no_val and len(val_dates) < 2:
                    continue

                for model_type in self.model_types:
                    lm = self._resolve_label_mode(model_type)
                    stacked_tr = _stack_cached(train_dates, lm)
                    if stacked_tr[0] is None:
                        continue
                    if no_val:
                        X_va = y_va = y_va_raw = group_va = None
                    else:
                        stacked_va = _stack_cached(val_dates, lm)
                        if stacked_va[0] is None:
                            continue
                        X_va, y_va, _, y_va_raw, group_va = stacked_va
                    X_tr, y_tr, w_tr, y_tr_raw, group_tr = stacked_tr
                    jobs.append((
                        window, model_type, X_tr, y_tr, w_tr, y_tr_raw,
                        X_va, y_va, y_va_raw, X_pred_np, y_true_np, cores_per_job,
                        self.objective, dataset.feature_names, pred_date, self.device,
                        group_tr, group_va,
                        # ridge ablation：剔除 regime 列（仅 ridge 生效）
                        self.ridge_drop_regime,
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

                done_preview = idx - predict_start + 1
                do_shap = self._should_compute_shap(done_preview, n_predict)

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
                    if do_shap and f.model is not None:
                        ww = float(w[i]) if i < len(w) else 0.0
                        self._record_fold_shap(
                            f, X_pred_np, pred_date,
                            window_weight=ww,
                            feature_names=dataset.feature_names,
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
        self._finalize_shap_export()
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
