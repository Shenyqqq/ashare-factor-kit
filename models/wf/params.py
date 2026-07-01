"""
Shared model hyperparameters for walk-forward training.

Lives under models/wf/ to avoid circular imports between models/trainer.py
and models/wf/models.py. Both modules import from here.
"""
from __future__ import annotations

# cholesky: dense X, small p (~40 factors); faster than auto/svd at walk-forward scale
# 注：build_model 中 ridge 分支已改用 RidgeCV 自动选 alpha，RIDGE_PARAMS 仅作
#     向后兼容/manifest 导出保留；RIDGE_CV_ALPHAS 为 RidgeCV 的候选 alpha 网格。
RIDGE_PARAMS = dict(alpha=10.0, solver="cholesky")

# RidgeCV 候选 alpha 网格（对数跨度覆盖弱/强正则）；cv=None 走 efficient LOO (GCV)
RIDGE_CV_ALPHAS = [0.1, 1.0, 10.0, 100.0]

LGBM_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    num_leaves=31,           # 与 max_depth=5 匹配，2^5=32
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20,    # 时间衰减权重下等效样本数约12-20
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
    reg_alpha=0.1, reg_lambda=1.0,
    early_stopping_rounds=30,   # XGBoost 2.x 在构造器里传，不在 fit() 里传
    random_state=42, verbosity=0, n_jobs=-1,
)

CAT_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    l2_leaf_reg=3, subsample=0.8,
    rsm=0.8,           # 特征随机采样（等价于 colsample_bytree），默认1.0导致过拟合
    random_strength=1, # 分裂点随机扰动，增强泛化
    random_seed=42, verbose=False, thread_count=-1,
)

# Random Forest：Bagging族代表，与GBDT天然正交（误差类型不同，预测相关性~0.65）
RF_PARAMS = dict(
    n_estimators=60,
    max_depth=6,            # 限深防过拟合；截面数据树不宜太深
    max_features=0.6,      # 等价于 colsample_bytree，60% 特征随机采样
    min_samples_leaf=20,   # 对应 min_child_samples，防止叶节点样本过少
    random_state=42,
)

# MLP：神经网络族代表，学习因子交互效应；不支持 sample_weight，用内置 early_stopping
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),  # 两层小网络，防止对 ~10万样本过拟合
    activation="relu",
    learning_rate_init=0.001,
    max_iter=200,                 # 配合 early_stopping，足够收敛
    alpha=0.01,                   # L2 正则，防过拟合
    early_stopping=True,          # 内置验证集早停，无需外部传 eval_set
    n_iter_no_change=20,
    validation_fraction=0.1,
    tol=1e-4,
    random_state=42,
)


# ── 调参后超参读取（追加：优先 tuned_params.json，否则用上方默认值）──────────
def _load_tuned_params_safe() -> dict | None:
    """读取 config/tuned_params.json，不存在/为空/解析失败返回 None。

    在 params.py 内独立实现 JSON 读取，避免与 tuning.py 形成循环导入
    （tuning.py 反向 import params 中的常量与 get_model_params）。
    """
    try:
        import json
        from pathlib import Path
        p = Path("config/tuned_params.json")
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if data else None
    except Exception:
        return None


def get_model_params(model_type: str) -> dict:
    """获取模型超参：优先用 tuned_params.json，否则用本文件默认值。

    tuned_params.json 通常只含 lgbm / xgb / rf 的子集（由 --tune 写入）；
    未调参的模型（ridge / cat / mlp）回退到下方默认值。
    返回的 dict 为副本，调用方可安全 mutate（如注入 n_jobs）。
    """
    tuned = _load_tuned_params_safe()
    if tuned and model_type in tuned and isinstance(tuned[model_type], dict):
        return dict(tuned[model_type])
    defaults = {
        "ridge": RIDGE_PARAMS, "lgbm": LGBM_PARAMS, "xgb": XGB_PARAMS,
        "cat": CAT_PARAMS, "rf": RF_PARAMS, "mlp": MLP_PARAMS,
    }
    base = defaults.get(model_type, {})
    return dict(base)
