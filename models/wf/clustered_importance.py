"""
models/wf/clustered_importance.py — Clustered Feature Importance (AFML Ch6)

步骤：
1. 对特征矩阵做相关性聚类（hierarchical clustering on correlation distance）
2. 按簇（cluster）而非单个特征做 permutation importance
3. 同簇特征一起 shuffle，评估对 IC 的影响
4. 簇内重要性按单特征 MDA 再分配

避免相关因子重要性分裂导致误删：两个高度相关的因子在普通 MDA / SHAP 下
各分一半重要性，都看起来不重要，从而同时被剔除。Clustered MDA 先把它们
聚到同簇按簇评估，整簇重要性得到保留，再在簇内做单特征 MDA 再分配。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# ── IC 评分 ────────────────────────────────────────────────────────────────
def _spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC；样本不足或常数列返回 NaN。"""
    if len(pred) < 5:
        return float("nan")
    p = pd.Series(pred)
    a = pd.Series(actual)
    if p.nunique() < 2 or a.nunique() < 2:
        return float("nan")
    return float(p.rank().corr(a.rank()))


def _default_score(model, X: pd.DataFrame, y: pd.Series) -> float:
    pred = model.predict(X.values if hasattr(X, "values") else np.asarray(X))
    return _spearman_ic(pred, np.asarray(y))


# ── 1. 相关性聚类 ────────────────────────────────────────────────────────────
def cluster_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.7,
) -> dict[str, list[str]]:
    """
    对特征做层次聚类（基于 |correlation| 距离）。

    距离定义：d = 1 - |corr|，故 |corr| >= correlation_threshold 的特征
    会被 fcluster 划入同簇。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵（行=样本，列=特征）。
    correlation_threshold : float
        聚类阈值；|corr| 高于该值的特征并入同簇。

    Returns
    -------
    dict[str, list[str]]
        ``{cluster_name: [feature1, feature2, ...]}``，cluster_name 形如
        ``"C01"``、``"C02"``，按簇大小降序、簇内按特征名排序。
    """
    if X.shape[1] < 2:
        # 单特征或空：唯一簇
        return {"C01": list(X.columns)}

    corr = X.corr().abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    # 距离 = 1 - |corr| ∈ [0, 1]
    dist = 1.0 - np.clip(corr, 0.0, 1.0)
    # condensed 上三角向量（linkage 要求）
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    # criterion='distance' + t = 1 - threshold：簇内最大距离 <= t
    t = 1.0 - correlation_threshold
    labels = hierarchy.fcluster(Z, t=t, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for feat, lab in zip(X.columns, labels):
        clusters.setdefault(int(lab), []).append(str(feat))

    # 排序：簇大小降序，簇内按特征名升序，重命名为 C01, C02, ...
    ordered = sorted(clusters.values(), key=lambda fs: (-len(fs), sorted(fs)))
    return {f"C{i+1:02d}": sorted(fs) for i, fs in enumerate(ordered)}


# ── 2. 簇级 MDA ──────────────────────────────────────────────────────────────
def clustered_mda(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    clusters: dict[str, list[str]],
    n_repeats: int = 10,
    scoring: str | Callable = "ic",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Clustered MDA：对每个簇整体 shuffle，评估 IC 下降量。

    importance = 原始 IC - shuffle 后 IC （>0 表示该簇有正向贡献）

    Parameters
    ----------
    model : fitted model
        任何带 ``predict(X)`` 接口的模型（与 ``scoring`` 配合）。
    X, y : pd.DataFrame, pd.Series
        评估集（推荐 OOS 验证集）。
    clusters : dict
        ``cluster_features`` 的返回值。
    n_repeats : int
        每簇 shuffle 重复次数，最终取均值。
    scoring : str | Callable
        ``"ic"`` = Spearman IC（默认）；可传自定义函数 ``f(model, X, y) -> float``。
    random_state : int

    Returns
    -------
    pd.DataFrame
        列：``cluster_name``、``importance`` (= IC 下降量)、``n_features``。
        按 importance 降序。
    """
    score_fn = _default_score if scoring == "ic" else scoring
    rng = np.random.default_rng(random_state)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)
    base_score = score_fn(model, pd.DataFrame(X_arr, columns=X.columns), pd.Series(y_arr))
    if not np.isfinite(base_score):
        base_score = 0.0

    rows = []
    for cname, feats in clusters.items():
        col_idx = [X.columns.get_loc(f) for f in feats if f in X.columns]
        if not col_idx:
            rows.append({"cluster_name": cname, "importance": 0.0, "n_features": 0})
            continue

        drops = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            # 整簇一起 shuffle 行索引
            perm = rng.permutation(X_arr.shape[0])
            X_perm[:, col_idx] = X_arr[perm, :][:, col_idx]
            s = score_fn(model, pd.DataFrame(X_perm, columns=X.columns), pd.Series(y_arr))
            drops.append(base_score - (s if np.isfinite(s) else 0.0))

        rows.append({
            "cluster_name": cname,
            "importance": float(np.mean(drops)),
            "n_features": len(col_idx),
        })

    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ── 3. 簇内单特征 MDA（再分配）──────────────────────────────────────────────
def intra_cluster_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cluster_features_list: list[str],
    n_repeats: int = 5,
    scoring: str | Callable = "ic",
    random_state: int = 42,
) -> dict[str, float]:
    """
    簇内单特征 permutation importance（在簇内再分配整簇重要性）。

    对簇内每个特征单独 shuffle 计算 IC 下降量。簇内只有一个特征时直接返回该特征
    等于整簇重要性（由上层 re-scale）。

    Returns
    -------
    dict[str, float]
        ``{feature_name: intra_importance}``，单特征 IC 下降量（>=0 通常占多数）。
    """
    score_fn = _default_score if scoring == "ic" else scoring
    rng = np.random.default_rng(random_state)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)
    base_score = score_fn(model, pd.DataFrame(X_arr, columns=X.columns), pd.Series(y_arr))
    if not np.isfinite(base_score):
        base_score = 0.0

    out: dict[str, float] = {}
    for feat in cluster_features_list:
        if feat not in X.columns:
            out[feat] = 0.0
            continue
        col_idx = X.columns.get_loc(feat)
        drops = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            perm = rng.permutation(X_arr.shape[0])
            X_perm[:, col_idx] = X_arr[perm, col_idx]
            s = score_fn(model, pd.DataFrame(X_perm, columns=X.columns), pd.Series(y_arr))
            drops.append(base_score - (s if np.isfinite(s) else 0.0))
        out[feat] = float(np.mean(drops))
    return out


# ── 4. 一键组合 ──────────────────────────────────────────────────────────────
def compute_clustered_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.7,
    n_repeats_cluster: int = 10,
    n_repeats_intra: int = 5,
    scoring: str | Callable = "ic",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    一键：cluster_features → clustered_mda → intra_cluster_importance。

    Returns
    -------
    pd.DataFrame
        列：
        - ``feature``：特征名
        - ``cluster``：所属簇名
        - ``cluster_importance``：整簇 MDA 重要性（IC 下降量）
        - ``intra_importance``：簇内单特征 MDA 重要性
        - ``total_importance``：按簇内占比再分配后的单特征总重要性
          ``= cluster_importance * intra / sum(intra_in_cluster)``

        按 total_importance 降序排列。
    """
    clusters = cluster_features(X, correlation_threshold=correlation_threshold)
    cluster_df = clustered_mda(
        model, X, y, clusters,
        n_repeats=n_repeats_cluster, scoring=scoring, random_state=random_state,
    )
    cluster_imp = dict(zip(cluster_df["cluster_name"], cluster_df["importance"]))

    rows = []
    for cname, feats in clusters.items():
        intra = intra_cluster_importance(
            model, X, y, feats,
            n_repeats=n_repeats_intra, scoring=scoring,
            random_state=random_state,
        )
        total_intra = sum(max(v, 0.0) for v in intra.values())
        c_imp = max(cluster_imp.get(cname, 0.0), 0.0)
        for feat in feats:
            intra_v = max(intra.get(feat, 0.0), 0.0)
            share = (intra_v / total_intra) if total_intra > 0 else (1.0 / len(feats))
            rows.append({
                "feature": feat,
                "cluster": cname,
                "cluster_importance": c_imp,
                "intra_importance": intra_v,
                "total_importance": c_imp * share,
            })

    out = pd.DataFrame(rows).sort_values("total_importance", ascending=False).reset_index(drop=True)
    return out
