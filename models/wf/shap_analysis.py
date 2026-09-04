"""
models/wf/shap_analysis.py — Walk-Forward SHAP 因子贡献分析

与 ``feature_importance_*.csv``（树分裂增益 / |coef|）和 Clustered FI
（AFML Ch.6 MDA）并存：

- SHAP：实例级边际贡献，再对样本取 mean(|SHAP|)，回答「预测时谁在推分」
- 内置 FI：训练期分裂增益或线性 |coef|，不保证与预测贡献一致
- Clustered FI：相关因子聚类后的置换重要性，避免共线分裂

防泄漏约定：只用**该折已训练模型** + **该折验证/预测截面特征**（默认 pred），
不回头用训练集算贡献。

树模型：``shap.TreeExplainer``；ridge：``LinearExplainer``，失败则
``coef × (X - mean)`` fallback；mlp 等不支持时跳过并记入 meta。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

TREE_MODEL_TYPES = frozenset({"lgbm", "xgb", "cat", "rf"})
LINEAR_MODEL_TYPES = frozenset({"ridge"})
UNSUPPORTED_MODEL_TYPES = frozenset({"mlp"})

DEFAULT_SHAP_MAX_SAMPLES = 500
DEFAULT_SHAP_MAX_DATES = 12
DEFAULT_SHAP_TOP = 20


def subsample_rows(
    X: np.ndarray | pd.DataFrame,
    max_samples: int = DEFAULT_SHAP_MAX_SAMPLES,
    random_state: int = 42,
) -> np.ndarray:
    """随机子采样行，控制 SHAP 内存。"""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape={arr.shape}")
    n = arr.shape[0]
    if n <= max_samples or max_samples <= 0:
        return arr
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return arr[idx]


def _ensure_2d_shap(shap_values) -> np.ndarray:
    """统一 TreeExplainer / LinearExplainer 输出为 (n_samples, n_features)。"""
    if isinstance(shap_values, list):
        # 多输出时取第一输出
        shap_values = shap_values[0]
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        # (n, n_features, n_outputs) → 取第 0 输出
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    return arr


def explain_tree_model(model, X: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    return _ensure_2d_shap(explainer.shap_values(X))


def explain_ridge_model(
    model,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, str]:
    """
    Ridge SHAP：优先 LinearExplainer；失败则 coef×中心化特征。

    Returns
    -------
    shap_values, method
        method ∈ {"linear_explainer", "coef_times_centered"}
    """
    coef = np.asarray(getattr(model, "coef_", None), dtype=float).ravel()
    kept_mask = getattr(model, "_ridge_kept_mask", None)
    if kept_mask is not None and len(kept_mask) == len(feature_names):
        X_use = X[:, np.asarray(kept_mask, dtype=bool)]
    else:
        X_use = X
        kept_mask = None

    if coef.size != X_use.shape[1]:
        raise ValueError(
            f"ridge coef size {coef.size} != X cols {X_use.shape[1]}"
        )

    try:
        import shap

        # Independent masker：特征已截面标准化时合理
        masker = shap.maskers.Independent(X_use, max_samples=min(100, len(X_use)))
        explainer = shap.LinearExplainer(model, masker)
        sv = _ensure_2d_shap(explainer.shap_values(X_use))
        method = "linear_explainer"
    except Exception as e:
        logger.debug(f"ridge LinearExplainer 失败，fallback coef×centered: {e}")
        Xc = X_use - np.nanmean(X_use, axis=0, keepdims=True)
        sv = Xc * coef.reshape(1, -1)
        method = "coef_times_centered"

    if kept_mask is not None:
        full = np.zeros((sv.shape[0], len(feature_names)), dtype=float)
        full[:, np.asarray(kept_mask, dtype=bool)] = sv
        return full, method
    return sv, method


def compute_shap_values(
    model,
    model_type: str,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    *,
    max_samples: int = DEFAULT_SHAP_MAX_SAMPLES,
    random_state: int = 42,
) -> tuple[np.ndarray | None, str]:
    """
    对单模型计算 SHAP 矩阵。

    Returns
    -------
    shap_values or None, method_or_skip_reason
    """
    mt = (model_type or "").lower()
    if mt in UNSUPPORTED_MODEL_TYPES:
        return None, f"unsupported:{mt}"
    X_sub = subsample_rows(X, max_samples=max_samples, random_state=random_state)
    if X_sub.shape[0] < 2:
        return None, "too_few_rows"
    if X_sub.shape[1] != len(feature_names):
        return None, (
            f"feature_dim_mismatch:{X_sub.shape[1]}!={len(feature_names)}"
        )

    try:
        if mt in TREE_MODEL_TYPES:
            return explain_tree_model(model, X_sub), "tree_explainer"
        if mt in LINEAR_MODEL_TYPES:
            return explain_ridge_model(model, X_sub, feature_names)
        # 未知类型：尝试 TreeExplainer
        try:
            return explain_tree_model(model, X_sub), "tree_explainer_fallback"
        except Exception:
            return None, f"unsupported:{mt}"
    except Exception as e:
        logger.warning(f"SHAP 计算失败 ({mt}): {e}")
        return None, f"error:{type(e).__name__}"


def summarize_shap_matrix(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """单批 SHAP → mean_|SHAP| / mean_SHAP / share。"""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    total = float(mean_abs.sum())
    share = mean_abs / total if total > 0 else np.zeros_like(mean_abs)
    return pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": mean_abs.astype(float),
        "mean_shap": mean_signed.astype(float),
        "share": share.astype(float),
    })


def compute_fold_shap_summary(
    model,
    model_type: str,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    *,
    max_samples: int = DEFAULT_SHAP_MAX_SAMPLES,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """单折 SHAP 汇总表 + method 标签。"""
    sv, method = compute_shap_values(
        model, model_type, X, feature_names,
        max_samples=max_samples, random_state=random_state,
    )
    if sv is None:
        return pd.DataFrame(), method
    return summarize_shap_matrix(sv, feature_names), method


def append_shap_rows(
    rows: list[dict],
    summary: pd.DataFrame,
    *,
    model_type: str,
    window: int,
    pred_date,
    method: str,
    weight: float = 1.0,
) -> None:
    """将单折汇总追加为长表行（含 fold weight，供后续加权聚合）。"""
    if summary is None or summary.empty:
        return
    for _, r in summary.iterrows():
        rows.append({
            "date": pred_date,
            "model": model_type,
            "window": int(window),
            "feature": r["feature"],
            "mean_abs_shap": float(r["mean_abs_shap"]),
            "mean_shap": float(r["mean_shap"]),
            "share": float(r["share"]),
            "method": method,
            "weight": float(weight),
        })


def aggregate_shap_rows(
    rows: list[dict],
    *,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """
    跨折加权聚合 mean(|SHAP|)。

    weight 默认为窗口 IC 权重（调用方写入）；同一 feature 上
    ``weighted_mean = sum(w * v) / sum(w)``。
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_keys = by or ["feature"]
    w = df["weight"].to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    def _wavg(col: str) -> pd.Series:
        v = df[col].to_numpy(dtype=float)
        num = df.assign(_n=w * v).groupby(group_keys, sort=False)["_n"].sum()
        den = df.assign(_w=w).groupby(group_keys, sort=False)["_w"].sum()
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)

    out = pd.DataFrame({
        "mean_abs_shap": _wavg("mean_abs_shap"),
        "mean_shap": _wavg("mean_shap"),
    }).reset_index()
    total = float(out["mean_abs_shap"].sum())
    out["share"] = (
        out["mean_abs_shap"] / total if total > 0 else 0.0
    )
    # 方法：取众数（同 feature 下）
    method_mode = (
        df.groupby(group_keys)["method"]
        .agg(lambda s: s.value_counts().index[0] if len(s) else "")
        .reset_index(name="method")
    )
    out = out.merge(method_mode, on=group_keys, how="left")
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def print_shap_top(summary: pd.DataFrame, top_n: int = DEFAULT_SHAP_TOP, title: str = "SHAP") -> None:
    if summary is None or summary.empty:
        logger.warning(f"{title}: 无可用结果")
        return
    n = max(1, int(top_n))
    head = summary.head(n)
    lines = [f"{title} Top-{min(n, len(head))} (按 mean_|SHAP|)："]
    for i, r in enumerate(head.itertuples(index=False), 1):
        feat = getattr(r, "feature", "?")
        mabs = getattr(r, "mean_abs_shap", float("nan"))
        share = getattr(r, "share", float("nan"))
        msign = getattr(r, "mean_shap", float("nan"))
        direction = "+" if msign > 0 else ("-" if msign < 0 else "0")
        lines.append(
            f"  {i:2d}. {feat}: mean_|SHAP|={mabs:.6f} "
            f"share={share:.1%} dir={direction}"
        )
    logger.info("\n".join(lines))


def export_shap_artifacts(
    rows: list[dict],
    output_dir: Path | str,
    tag: str,
    *,
    top_n: int = DEFAULT_SHAP_TOP,
    model_weights: dict[str, float] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    写出：
      - shap_fold_<tag>.csv          折级长表
      - shap_by_model_<tag>.csv      按模型聚合
      - shap_summary_<tag>.csv       ensemble / 全模型加权总表
      - shap_summary_<tag>.json      含 Top-N + meta
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not rows:
        logger.warning("SHAP: 无折级记录，跳过导出")
        return paths

    fold_path = out / f"shap_fold_{tag}.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False, encoding="utf-8-sig")
    paths["fold"] = fold_path

    by_model = aggregate_shap_rows(rows, by=["model", "feature"])
    if not by_model.empty:
        p = out / f"shap_by_model_{tag}.csv"
        by_model.to_csv(p, index=False, encoding="utf-8-sig")
        paths["by_model"] = p

    # Ensemble：先按模型聚合 feature，再按 model_weights 二次加权
    if model_weights and by_model is not None and not by_model.empty:
        mw = {k: float(v) for k, v in model_weights.items() if float(v) > 0}
        if mw:
            s = sum(mw.values())
            mw = {k: v / s for k, v in mw.items()}
            pieces = []
            for m, w in mw.items():
                sub = by_model[by_model["model"] == m].copy()
                if sub.empty:
                    continue
                sub["mean_abs_shap"] *= w
                sub["mean_shap"] *= w
                pieces.append(sub[["feature", "mean_abs_shap", "mean_shap"]])
            if pieces:
                ens = (
                    pd.concat(pieces, ignore_index=True)
                    .groupby("feature", as_index=False)
                    .sum()
                )
                total = float(ens["mean_abs_shap"].sum())
                ens["share"] = ens["mean_abs_shap"] / total if total > 0 else 0.0
                ens["method"] = "ensemble_weighted"
                ens = ens.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                summary = ens
            else:
                summary = aggregate_shap_rows(rows, by=["feature"])
        else:
            summary = aggregate_shap_rows(rows, by=["feature"])
    else:
        summary = aggregate_shap_rows(rows, by=["feature"])

    if not summary.empty:
        sp = out / f"shap_summary_{tag}.csv"
        summary.to_csv(sp, index=False, encoding="utf-8-sig")
        paths["summary"] = sp

        top = summary.head(max(1, int(top_n)))
        meta = {
            "tag": tag,
            "n_fold_rows": len(rows),
            "n_features": int(summary.shape[0]),
            "top_n": int(top_n),
            "model_weights": model_weights or {},
            "top_features": top.to_dict(orient="records"),
            "note": (
                "mean_abs_shap = 跨折加权平均的 mean(|SHAP|)；"
                "share = 占全部特征 mean_abs_shap 之和的比例；"
                "mean_shap 符号表示平均推高(+)/压低(-)预测分的方向。"
                "与 feature_importance / clustered FI 口径不同，可并存。"
            ),
        }
        if meta_extra:
            meta.update(meta_extra)
        jp = out / f"shap_summary_{tag}.json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        paths["json"] = jp
        print_shap_top(summary, top_n=top_n, title=f"SHAP[{tag}]")

    return paths


def compute_trainer_shap_recent(
    trainer,
    *,
    max_samples: int = DEFAULT_SHAP_MAX_SAMPLES,
    max_dates: int = DEFAULT_SHAP_MAX_DATES,
    top_n: int = DEFAULT_SHAP_TOP,
    random_state: int = 42,
    export: bool = True,
) -> pd.DataFrame:
    """
    训练结束后用**内存中残留模型**（通常为最后一折）+ 最近 OOS 截面做 SHAP。

    注意：``trainer.models`` 按 (window, model_type) 覆盖写入，仅代表末折；
    完整 WF 汇总请在训练期开启 ``enable_shap``（折内累计）。
    本函数作 ``--report`` / 事后分析的轻量路径。
    """
    dataset = getattr(trainer, "_dataset", None)
    models = getattr(trainer, "models", None) or {}
    if dataset is None or not models:
        logger.warning("SHAP recent: trainer 无 dataset/models")
        return pd.DataFrame()

    feature_names = list(dataset.feature_names)
    dates = list(getattr(trainer, "score_df", pd.DataFrame()).index)
    if not dates:
        dates = list(dataset.rebalance_dates)
    if max_dates and max_dates > 0:
        dates = dates[-max_dates:]

    rows: list[dict] = []
    methods: set[str] = set()
    for d in dates:
        X, _ = dataset.get_cross_section(d)
        if X is None or X.empty:
            continue
        X = X.reindex(columns=feature_names).fillna(0)
        for (window, model_type), model in models.items():
            summary, method = compute_fold_shap_summary(
                model, model_type, X.values, feature_names,
                max_samples=max_samples,
                random_state=random_state,
            )
            methods.add(method)
            append_shap_rows(
                rows, summary,
                model_type=model_type, window=window, pred_date=d,
                method=method, weight=1.0,
            )

    tag = getattr(trainer, "tag", None) or "wf"
    out_dir = getattr(trainer, "artifact_dir", None)
    model_types = list(getattr(trainer, "model_types", []))
    mw = {m: 1.0 / len(model_types) for m in model_types} if model_types else None
    summary = aggregate_shap_rows(rows, by=["feature"])
    if export and out_dir is not None:
        export_shap_artifacts(
            rows, out_dir, tag,
            top_n=top_n,
            model_weights=mw,
            meta_extra={
                "mode": "recent_oos_last_models",
                "methods": sorted(methods),
                "max_samples": max_samples,
                "max_dates": max_dates,
                "warning": (
                    "models 仅为训练末折残留；完整 walk-forward SHAP 请用 --shap"
                ),
            },
        )
    elif not summary.empty:
        print_shap_top(summary, top_n=top_n, title=f"SHAP[{tag}]")
    return summary
