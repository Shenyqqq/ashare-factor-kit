"""
Model wrappers for walk-forward training (ridge / lgbm / xgb / cat / rf / mlp).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb_lib
try:
    import catboost as cb  # 可选；仅 "cat" 模型需要，未安装时 cat 不可用
except ModuleNotFoundError:
    cb = None
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
    LGBM_RANK_EVAL_AT,
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
        # Ridge 直接在 CS-zscored 特征上训练（上游 factors/factor.py::_normalize
        # 已做截面 winsorize+zscore，clip ±3σ，每调仓日 mean≈0 std≈1）。
        # 旧实现叠了一层 StandardScaler（在堆叠后的 X_tr 上做 pooled 标准化），
        # 既冗余又把不同调仓日的截面统计量 pool 到一起，模糊了 per-date CS 语义；
        # 已移除。regime 特征是 TS-zscore（语义不同），通过 ``ridge_drop_regime``
        # 标志可由调用方剔除（用于 ablation；不在 models.py 默认开启）。
        # RidgeCV 自动选 alpha；sample_weight 通过 fit_model 路由（无 Pipeline 时
        # 直接传 sample_weight=，无需 ridge__sample_weight 元数据路由）。
        return RidgeCV(alphas=RIDGE_CV_ALPHAS, cv=None)
    if model_type == "lgbm":
        extra = {}
        if device == "gpu":
            extra["device_type"] = "gpu"
        params = {**get_model_params("lgbm"), "n_jobs": n_jobs, **extra}
        if is_rank:
            # LGBMRanker 默认 objective='lambdarank'，metric='ndcg'。
            # 训练标签为截面细整数秩（见 prepare_rank_labels）；eval_at 看 top-K。
            # label_gain 在 fit_model 按 max label 设线性增益（避免默认 2^rel-1 溢出）。
            params = {**params, "eval_at": list(LGBM_RANK_EVAL_AT)}
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
            # 细秩 max label 常 >31，默认指数 NDCG gain 会报错 → 关掉。
            params.pop("objective", None)
            params["ndcg_exp_gain"] = False
            return xgb_lib.XGBRanker(**params)
        return xgb_lib.XGBRegressor(**params)
    if model_type == "cat":
        params = {**CAT_PARAMS, "thread_count": n_jobs}
        if is_rank:
            # CatBoostRanker 默认 loss 多为 YetiRank（pairwise）；
            # group_id 须单调非降（_stack_cached 按调仓日顺序堆叠即满足）。
            # pairwise loss 不支持 per-object sample_weight，见 fit_model。
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


def prepare_rank_labels(
    y,
    group_sizes: list[int] | None = None,
) -> np.ndarray:
    """将标签转为 LTR 细整数秩（非负 int），供 LGBM / XGB / CatBoost Ranker 共用。

    每个 query（调仓日截面）内映射为密集秩 ``0 .. n_g-1``（``y`` 越大 relevance 越高）。
    当 ``y`` 已是 ``cs_rank`` 百分位（``[0, 1]``，来自 ``cross_sectional_rank``）时，
    等价于 ``round(y * (n_g - 1))``；档位数随截面大小变化，不再固定 5 档 digitize。

    Parameters
    ----------
    y :
        堆叠后的训练/验证标签（通常为 per-date ``cs_rank``，来自未 winsor
        的 raw forward_return）。
    group_sizes :
        各调仓日样本数；``None`` 时把整段 ``y`` 视为单个截面。

    Returns
    -------
    np.ndarray
        ``dtype=int32``，长度与 ``y`` 相同。
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = int(y_arr.size)
    if group_sizes is None:
        group_sizes = [n] if n else []
    if int(sum(group_sizes)) != n:
        raise ValueError(
            f"sum(group_sizes)={sum(group_sizes)} != len(y)={n}"
        )
    out = np.empty(n, dtype=np.int32)
    start = 0
    for g in group_sizes:
        end = start + g
        if g <= 0:
            continue
        section = y_arr[start:end]
        # 截面密集秩 0..n-1（y 越大 relevance 越高）。
        # 当 section 为 exact cs_rank 百分位时，等价于 round(y * (n-1))。
        order = np.argsort(section, kind="mergesort")
        rel = np.empty(g, dtype=np.int32)
        rel[order] = np.arange(g, dtype=np.int32)
        out[start:end] = rel
        start = end
    return out


_SKLEARN_MODELS = {"ridge", "rf", "mlp"}


def _sanitize_X(X, model_type: str):
    """
    清洗特征矩阵，按模型类型差异化处理 inf/NaN。

    - 树模型（lgbm/xgb/cat）：仅 inf→NaN，保留 NaN 作为分裂信号
      （LightGBM/XGBoost 原生支持 NaN；inf 在部分版本会报错，统一转 NaN）。
    - sklearn 模型（ridge/rf/mlp）：inf→NaN，再按列用训练统计量填充。
      Ridge/RF/MLP 不接受 NaN/inf，必须填实。用 0 填充（截面标准化后 0 即均值，
      且 sklearn pipeline 内的 StandardScaler 已做中心化）。
    """
    if X is None:
        return X
    X = np.asarray(X, dtype=np.float64)
    # inf → nan（统一，对所有模型安全）
    if not np.all(np.isfinite(X)):
        X = np.where(np.isfinite(X), X, np.nan)
    if model_type in _SKLEARN_MODELS:
        # sklearn 模型不能有 NaN：用列均值填充，列全 NaN 则填 0
        if X.ndim == 2:
            col_mean = np.nanmean(X, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            inds = np.where(np.isnan(X))
            if inds[0].size > 0:
                X = X.copy()
                X[inds] = np.take(col_mean, inds[1])
        else:
            m = np.nanmean(X)
            X = np.where(np.isnan(X), m if np.isfinite(m) else 0.0, X)
    return X


def _regime_column_mask(feature_names: list[str] | None) -> np.ndarray | None:
    """根据 feature_names 返回 regime 列的 boolean mask（True = regime 列）。

    regime 特征指市场状态 / HMM 概率特征（与 factor.py::get_factor_names 中
    `市场` / `HMM_` 前缀一致，亦与 strategies/ml.py::_excluded 同口径）。
    feature_names 为 None 时返回 None（调用方按全列处理）。
    """
    if feature_names is None:
        return None
    return np.array(
        [n.startswith("市场") or n.startswith("HMM_") for n in feature_names],
        dtype=bool,
    )


def _drop_regime_cols(X, feature_names: list[str] | None,
                      ridge_drop_regime: bool) -> tuple:
    """若 ridge_drop_regime=True 且能识别 regime 列，返回去掉 regime 列后的 X
    与对应的列 mask（用于 predict 时对齐）。否则原样返回 (X, None_mask)。

    返回 (X_new, kept_mask) — kept_mask 为保留列的 boolean 索引（None 表示未裁剪）。
    """
    if not ridge_drop_regime:
        return X, None
    regime_mask = _regime_column_mask(feature_names)
    if regime_mask is None:
        return X, None
    kept = ~regime_mask
    if kept.all():
        return X, None
    X = np.asarray(X)
    if X.ndim != 2:
        return X, None
    return X[:, kept], kept


def fit_model(
    model_type: str,
    X_tr, y_tr, w_tr,
    X_va, y_va,
    n_jobs: int = -1,
    objective: str = "regression",
    device: str = "cpu",
    group_tr: list[int] | None = None,
    group_va: list[int] | None = None,
    feature_names: list[str] | None = None,
    ridge_drop_regime: bool = False,
):
    """Train one model.

    rank objective 时，``group_tr`` / ``group_va`` 为每个调仓日的样本数列表
    （list[int]，长度 = 调仓日数）；非 rank 模式下被忽略。
    不支持 rank 的模型（ridge/rf/mlp）即使 objective='rank' 也按 regression 走
    （build_model 已发警告并构建为 regressor）。

    ``feature_names`` + ``ridge_drop_regime``：仅 ridge 使用。当
    ``ridge_drop_regime=True`` 时，按 ``feature_names`` 识别并剔除市场 / HMM
    regime 列（用于 ablation：regime 是 TS-zscore 语义，与 CS-zscore 的 alpha
    特征混在同一 Ridge 上可能引入口径冲突；此 flag 让 ridge 仅在 alpha 特征上
    拟合）。默认 False，保持向后兼容。其它模型忽略这两个参数。

    ``X_va`` / ``y_va`` 为 None 或空时（val_window=0）：关闭树模型 early
    stopping，用固定 ``n_estimators`` / ``iterations`` 训满。
    """
    model = build_model(model_type, n_jobs, objective, device=device)
    is_ranker = (
        objective == "rank"
        and model_type in {"lgbm", "xgb", "cat"}
        and group_tr is not None
    )
    has_val = (
        X_va is not None and y_va is not None
        and len(X_va) > 0 and len(y_va) > 0
    )

    if model_type == "ridge":
        # regime 列裁剪（ablation 用，默认 False 不裁剪）
        X_tr_r, kept_mask = _drop_regime_cols(X_tr, feature_names, ridge_drop_regime)
        model.fit(_sanitize_X(X_tr_r, "ridge"), y_tr, sample_weight=w_tr)
        # 把 kept_mask 挂到 model 上，predict_model 据此对齐 X_pred
        model._ridge_kept_mask = kept_mask
    elif model_type == "lgbm":
        if is_ranker:
            # LGBMRanker: group=每期样本数，eval_group 同结构。
            # LambdaRank 要求非负整数 relevance；三模型共用 prepare_rank_labels 细秩。
            # 默认 label_gain=2^rel-1 在细秩下会溢出 → 线性增益。
            y_tr_int = prepare_rank_labels(y_tr, group_tr)
            y_va_int = (
                prepare_rank_labels(y_va, group_va)
                if has_val and group_va is not None
                else None
            )
            max_label = int(y_tr_int.max()) if y_tr_int.size else 0
            if y_va_int is not None and y_va_int.size:
                max_label = max(max_label, int(y_va_int.max()))
            model.set_params(label_gain=list(range(max_label + 1)))
            callbacks = [lgb.log_evaluation(-1)]
            fit_kwargs = dict(group=group_tr)
            if y_va_int is not None:
                callbacks.insert(0, lgb.early_stopping(30, verbose=False))
                fit_kwargs["eval_set"] = [(X_va, y_va_int)]
                fit_kwargs["eval_group"] = [group_va]
            fit_kwargs["callbacks"] = callbacks
            model.fit(X_tr, y_tr_int, sample_weight=w_tr, **fit_kwargs)
        elif has_val:
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
        else:
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                callbacks=[lgb.log_evaluation(-1)],
            )
    elif model_type == "xgb":
        if not has_val:
            # XGB 2.x：构造器里 early_stopping_rounds 在无 eval_set 时会报错
            model.set_params(early_stopping_rounds=None)
        if is_ranker:
            # XGBRanker: qid + 与 LGBM/Cat 同一套细整数秩。
            # sample_weight 长度 = group 数（非逐样本）；按 group 对 w_tr 取均值。
            y_tr_int = prepare_rank_labels(y_tr, group_tr)
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
            if has_val and group_va is not None:
                qid_va = _group_sizes_to_qid(group_va)
                y_va_int = prepare_rank_labels(y_va, group_va)
                fit_kwargs["eval_set"] = [(X_va, y_va_int)]
                fit_kwargs["eval_qid"] = [qid_va]
            model.fit(X_tr, y_tr_int, **fit_kwargs)
        elif has_val:
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    elif model_type == "cat":
        if is_ranker:
            # CatBoostRanker: group_id 单调非降；标签与 LGBM/XGB 同一细整数秩
            # （不再直接喂连续 cs_rank）。eval_set 须为带 group_id 的 Pool。
            # YetiRank 等 pairwise loss 拒绝 object weights → 不传 sample_weight
            # （否则 fit 报 "Pairwise losses don't support object weights"）。
            y_tr_int = prepare_rank_labels(y_tr, group_tr)
            group_id_tr = _group_sizes_to_qid(group_tr)
            fit_kwargs = dict(group_id=group_id_tr)
            if has_val and group_va is not None:
                group_id_va = _group_sizes_to_qid(group_va)
                y_va_int = prepare_rank_labels(y_va, group_va)
                eval_pool = cb.Pool(X_va, y_va_int, group_id=group_id_va)
                fit_kwargs["eval_set"] = eval_pool
                fit_kwargs["early_stopping_rounds"] = 30
            model.fit(X_tr, y_tr_int, **fit_kwargs)
        elif has_val:
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == "rf":
        # RF 支持 sample_weight，接受时间衰减权重
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == "mlp":
        # MLPRegressor 不支持 sample_weight；内置 early_stopping 从 train 切 validation_fraction
        model.fit(X_tr, y_tr)
    return model


def predict_model(model, X_pred, model_type: str | None = None) -> np.ndarray:
    if model_type:
        X_pred = _sanitize_X(X_pred, model_type)
    # ridge + ridge_drop_regime：fit 时存的 _ridge_kept_mask 用于对齐 X_pred 列
    if model_type == "ridge":
        kept_mask = getattr(model, "_ridge_kept_mask", None)
        if kept_mask is not None:
            Xp = np.asarray(X_pred)
            if Xp.ndim == 2 and kept_mask.shape[0] == Xp.shape[1]:
                X_pred = Xp[:, kept_mask]
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
        # ridge 已从 Pipeline 改为裸 RidgeCV（无 StandardScaler），直接取 coef_。
        # ridge_drop_regime=True 时，coef_ 仅对应 kept_mask 保留的列；被剔除的
        # regime 列 importance 记 0（未参与拟合）。
        coef = model.coef_
        kept_mask = getattr(model, "_ridge_kept_mask", None)
        if kept_mask is None or len(kept_mask) != len(feature_names):
            return {f: float(abs(c)) for f, c in zip(feature_names, coef)}
        out = {f: 0.0 for f in feature_names}
        kept_names = [f for f, k in zip(feature_names, kept_mask) if k]
        for f, c in zip(kept_names, coef):
            out[f] = float(abs(c))
        return out
    if model_type == "lgbm":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "xgb":
        imp = model.feature_importances_
        return {f: float(v) for f, v in zip(feature_names, imp)}
    if model_type == "cat":
        # CatBoostRegressor.feature_importances_ 正常；CatBoostRanker 常返回
        # 0-d 空数组 → 改用 PredictionValuesChange（无需再传训练 Pool）。
        imp = np.asarray(getattr(model, "feature_importances_", []), dtype=float).ravel()
        if imp.size != len(feature_names):
            try:
                imp = np.asarray(
                    model.get_feature_importance(type="PredictionValuesChange"),
                    dtype=float,
                ).ravel()
            except Exception:
                return {}
        if imp.size != len(feature_names):
            return {}
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
