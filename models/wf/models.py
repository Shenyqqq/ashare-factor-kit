"""
Model wrappers for walk-forward training (ridge / lgbm / xgb / cat / rf / mlp).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb_lib
import catboost as cb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

from models.wf.params import (
    RIDGE_PARAMS, RIDGE_CV_ALPHAS,
    LGBM_PARAMS, XGB_PARAMS, CAT_PARAMS, RF_PARAMS, MLP_PARAMS,
    get_model_params,
)


def build_model(
    model_type: str,
    n_jobs: int = -1,
    objective: str = "regression",
    device: str = "cpu",
):
    """
    Build a sklearn / GBDT model.

    ``objective='rank'``: Learning-to-Rank 目标（LambdaRank 族）。

      - **lgbm**: ``lgb.LGBMRanker``（LambdaRank），fit 时需 ``group`` 参数
        （每个 group = 一个调仓日的所有股票截面）。
      - **xgb**: ``xgboost.XGBRanker`` + ``objective='rank:pairwise'``，fit 时需
        ``qid``（每个样本对应的调仓日 query id，需按 qid 升序排列——_stack_cached
        已按调仓日顺序堆叠，天然满足）。
      - **cat**: ``catboost.CatBoostRanker``，fit 时需 ``group_id``（按调仓日顺序
        排列的逐样本 group id，CatBoost 要求 group_id 单调非降）。
      - **ridge / rf / mlp**: 不支持 rank，fallback 到 regression 并发警告日志
        （调用方 fit_model 仍按 regression 路径走）。

    ``device='gpu'``: LGBM 加 ``device_type='gpu'``（需 ``lightgbm[gpu]``），
    XGB 加 ``tree_method='hist', device='cuda'``（XGBoost ≥ 2.0）；
    ridge / cat / rf / mlp 不受 device 影响（cat 自身有 task_type 参数，未在此处接入）。
    """
    is_rank = objective == "rank"

    # rank objective 对不支持 rank 的模型 fallback 到 regression
    if is_rank and model_type in {"ridge", "rf", "mlp"}:
        warnings.warn(
            f"模型 '{model_type}' 不支持 rank objective，回退到 regression；"
            f"如需 ranking 请使用 lgbm/xgb/cat。",
            stacklevel=2,
        )
        is_rank = False  # 实际按 regression 构建

    if model_type == "ridge":
        # RidgeCV 内置 alpha 选择（cv=None 用 Leave-One-Out / GCV），零成本超参搜索。
        # RidgeCV 支持 sample_weight（sklearn ≥ 0.22），fit_model 透传不受影响。
        return RidgeCV(alphas=RIDGE_CV_ALPHAS, cv=None)
    if model_type == "lgbm":
        extra = {}
        if device == "gpu":
            extra["device_type"] = "gpu"
        params = {**get_model_params("lgbm"), "n_jobs": n_jobs, **extra}
        if is_rank:
            # LGBMRanker 默认 objective='lambdarank'，metric='ndcg'；
            # label_gain 默认 (2^rel - 1)，cs_rank 标签 ∈ [0,1] 时近似 NDCG 增益。
            return lgb.LGBMRanker(**params)
        return lgb.LGBMRegressor(**params)
    if model_type == "xgb":
        extra = {}
        if device == "gpu":
            # XGBoost ≥ 2.0：device='cuda' 配合 tree_method='hist' 走 GPU hist
            extra["tree_method"] = "hist"
            extra["device"] = "cuda"
        params = {**get_model_params("xgb"), "n_jobs": n_jobs, **extra}
        if is_rank:
            # XGBRanker 内部强制 objective='rank:pairwise'；
            # 重新覆盖 params 中的 objective 字段（XGBRegressor 默认无 objective=rank）。
            params.pop("objective", None)
            return xgb_lib.XGBRanker(**params)
        return xgb_lib.XGBRegressor(**params)
    if model_type == "cat":
        params = {**CAT_PARAMS, "thread_count": n_jobs}
        if is_rank:
            # CatBoostRanker 默认 loss_function='QueryRMSE'；
            # CatBoost 要求 group_id 单调非降（_stack_cached 按调仓日顺序堆叠即满足）。
            return cb.CatBoostRanker(**params)
        return cb.CatBoostRegressor(**params)
    if model_type == "rf":
        return RandomForestRegressor(**{**get_model_params("rf"), "n_jobs": n_jobs})
    if model_type == "mlp":
        # Pipeline 自动 StandardScaler，MLP 必须缩放特征
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**MLP_PARAMS)),
        ])
    raise ValueError(f"未知模型: {model_type}，支持 ridge/lgbm/xgb/cat/rf/mlp")


def _group_sizes_to_qid(group_sizes: list[int]) -> np.ndarray:
    """把 list[int]（每个 group 的样本数）展开成逐样本 query id 数组。

    例：[3, 2] → [0, 0, 0, 1, 1]；XGBoost / CatBoost 的 ranker 需要此格式，
    且要求 qid / group_id 单调非降——_stack_cached 按调仓日顺序堆叠天然满足。
    """
    if not group_sizes:
        return np.array([], dtype=np.int64)
    return np.repeat(np.arange(len(group_sizes), dtype=np.int64), group_sizes)


def fit_model(
    model_type: str,
    X_tr, y_tr, w_tr,
    X_va, y_va,
    n_jobs: int = -1,
    objective: str = "regression",
    device: str = "cpu",
    group_tr: list[int] | None = None,
    group_va: list[int] | None = None,
):
    """Train one model.

    rank objective 时，``group_tr`` / ``group_va`` 为每个调仓日的样本数列表
    （list[int]，长度 = 调仓日数）；非 rank 模式下被忽略。
    不支持 rank 的模型（ridge/rf/mlp）即使 objective='rank' 也按 regression 走
    （build_model 已发警告并构建为 regressor）。
    """
    model = build_model(model_type, n_jobs, objective, device=device)
    is_ranker = (
        objective == "rank"
        and model_type in {"lgbm", "xgb", "cat"}
        and group_tr is not None
    )

    if model_type == "ridge":
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == "lgbm":
        if is_ranker:
            # LGBMRanker: group=每期样本数，eval_group 同结构
            # LightGBM LambdaRank 要求 label 为非负整数（relevance grade）；
            # cs_rank 标签 ∈ [0,1] 浮点需离散化为 N_REL bins（默认 5 级：0..4），
            # 默认 label_gain=(2^rel - 1) 在 5 级下增益为 [0,1,3,7,15]，符合 NDCG。
            N_REL = 5
            y_tr_int = np.digitize(np.asarray(y_tr), np.linspace(0, 1, N_REL + 1)[1:-1])
            y_va_int = np.digitize(np.asarray(y_va), np.linspace(0, 1, N_REL + 1)[1:-1])
            eval_set = [(X_va, y_va_int)] if X_va is not None and len(X_va) else None
            eval_group = [group_va] if group_va is not None else None
            fit_kwargs = dict(
                group=group_tr,
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            if eval_set is not None:
                fit_kwargs["eval_set"] = eval_set
            if eval_group is not None:
                fit_kwargs["eval_group"] = eval_group
            model.fit(X_tr, y_tr_int, sample_weight=w_tr, **fit_kwargs)
        else:
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
    elif model_type == "xgb":
        if is_ranker:
            # XGBRanker: 需 qid（逐样本 query id），eval_set+eval_qid 同结构
            # XGBoost rank objective 要求 label 为非负整数 relevance grade；
            # cs_rank ∈ [0,1] 浮点需离散化为 N_REL 级（默认 5：0..4）。
            # 注意：sample_weight 要求长度 = group 数（每 group 一个权重），
            # 而非逐样本；从逐样本 w_tr 按 group 聚合（取均值，组内 decay 一致）。
            N_REL = 5
            edges = np.linspace(0, 1, N_REL + 1)[1:-1]
            y_tr_int = np.digitize(np.asarray(y_tr), edges)
            qid_tr = _group_sizes_to_qid(group_tr)
            w_group_tr = None
            if w_tr is not None and len(w_tr) == len(qid_tr):
                w_arr = np.asarray(w_tr, dtype=np.float64)
                w_group_tr = np.array(
                    [w_arr[qid_tr == g].mean() for g in range(len(group_tr))],
                    dtype=np.float64,
                )
            fit_kwargs = dict(qid=qid_tr, verbose=False)
            if w_group_tr is not None:
                fit_kwargs["sample_weight"] = w_group_tr
            if X_va is not None and len(X_va) and group_va is not None:
                qid_va = _group_sizes_to_qid(group_va)
                y_va_int = np.digitize(np.asarray(y_va), edges)
                fit_kwargs["eval_set"] = [(X_va, y_va_int)]
                fit_kwargs["eval_qid"] = [qid_va]
            model.fit(X_tr, y_tr_int, **fit_kwargs)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "cat":
        if is_ranker:
            # CatBoostRanker: group_id=逐样本 group id（单调非降）
            # eval_set 必须是带 group_id 的 catboost.Pool（CatBoostRanker.fit
            # 无 eval_group_id 参数，eval_set 中的 group_id 通过 Pool 注入）。
            group_id_tr = _group_sizes_to_qid(group_tr)
            fit_kwargs = dict(group_id=group_id_tr)
            if X_va is not None and len(X_va) and group_va is not None:
                group_id_va = _group_sizes_to_qid(group_va)
                eval_pool = cb.Pool(X_va, y_va, group_id=group_id_va)
                fit_kwargs["eval_set"] = eval_pool
                fit_kwargs["early_stopping_rounds"] = 30
            model.fit(X_tr, y_tr, sample_weight=w_tr, **fit_kwargs)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
    elif model_type == "rf":
        # RF 支持 sample_weight，接受时间衰减权重
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == "mlp":
        # MLPRegressor 不支持 sample_weight；内置 early_stopping 已防过拟合
        model.fit(X_tr, y_tr)
    return model


def predict_model(model, X_pred) -> np.ndarray:
    return model.predict(X_pred)


def extract_feature_importance(
    model,
    model_type: str,
    feature_names: list[str],
    *,
    X_va=None,
    y_va=None,
) -> dict[str, float]:
    """Feature importance or |coef| for export.

    MLP 分支需要 ``X_va`` / ``y_va`` 以计算 permutation importance；
    未提供时返回空字典（permutation 必须在验证集上评估，避免训练集偏差）。
    """
    if model_type == "ridge":
        coef = model.coef_
        return {f: float(abs(c)) for f, c in zip(feature_names, coef)}
    if model_type == "lgbm":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "xgb":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "cat":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "rf":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "mlp":
        # MLP 无内置 feature_importances_；用 permutation importance 在验证集上估计
        # （比 scaler.scale_ 更准确——scale_ 只反映输入尺度，与模型对特征的依赖无关）
        if X_va is None or y_va is None:
            return {}
        try:
            result = permutation_importance(
                model, X_va, y_va, n_repeats=5, random_state=42, n_jobs=1,
            )
            return {f: float(v) for f, v in zip(feature_names, result.importances_mean)}
        except Exception:
            return {}
    return {}
