"""
Training diagnostics: IC, drift detection, feature importance export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def spearman_ic(pred, actual) -> float:
    if len(pred) < 5:
        return np.nan
    return pd.Series(pred).rank().corr(pd.Series(actual).rank())


def compute_drift_flags(
    pred_ic: float,
    history: list[dict],
    z_threshold: float = 3.0,
    min_history: int = 5,
) -> dict:
    """
    Per predict date: drift detection based on预测 IC 的滚动 z-score (P1-5)。

    早期实现用 cs_zscore 化后的 score_mean 做 drift，但截面 z-score 的
    mean 恒为 0，drift 检测无效。改为对每期预测 IC（pred_ic）做滚动
    z-score：若 |z| > threshold 则 flagged。

    Parameters
    ----------
    pred_ic : float
        当期预测 IC（Spearman correlation between pred scores and actual）。
    history : list[dict]
        历史记录，每条至少含 ``{"pred_ic": float}``。
    z_threshold : float
        滚动 z-score 触发阈值。
    min_history : int
        触发 drift 检测所需的最小历史样本数。
    """
    ic = float(pred_ic) if pred_ic is not None and np.isfinite(pred_ic) else np.nan

    hist_ics = [h["pred_ic"] for h in history
                if np.isfinite(h.get("pred_ic", np.nan))]
    drift_z = np.nan
    flagged = False
    if len(hist_ics) >= min_history:
        mu_h = float(np.mean(hist_ics))
        sig_h = float(np.std(hist_ics))
        if sig_h > 1e-12 and np.isfinite(ic):
            drift_z = (ic - mu_h) / sig_h
            flagged = abs(drift_z) > z_threshold

    return {
        "pred_ic": ic,
        "drift_z": drift_z,
        "drift_flagged": flagged,
    }


def diagnostics_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def export_diagnostics(records: list[dict], path: Path) -> None:
    df = diagnostics_to_dataframe(records)
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")


def export_feature_importance(
    importance_rows: list[dict],
    path: Path,
) -> None:
    if not importance_rows:
        return
    df = pd.DataFrame(importance_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def append_feature_importance_rows(
    rows: list,
    model_type: str,
    window: int,
    pred_date,
    importance: dict[str, float],
) -> None:
    for feat, val in importance.items():
        rows.append({
            "date": pred_date,
            "model": model_type,
            "window": window,
            "feature": feat,
            "importance": val,
        })


def export_clustered_importance(
    model,
    X,
    y,
    output_path: Path,
    correlation_threshold: float = 0.7,
    n_repeats_cluster: int = 10,
    n_repeats_intra: int = 5,
    scoring="ic",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    AFML Ch6 Clustered Feature Importance 导出（追加接口，不影响现有函数）。

    调用 ``models.wf.clustered_importance.compute_clustered_importance`` 计算
    簇级 + 簇内再分配的 importance，并写 CSV。

    Returns
    -------
    pd.DataFrame
        compute_clustered_importance 的结果（便于调用方进一步使用）。
    """
    from models.wf.clustered_importance import compute_clustered_importance

    df = compute_clustered_importance(
        model, X, y,
        correlation_threshold=correlation_threshold,
        n_repeats_cluster=n_repeats_cluster,
        n_repeats_intra=n_repeats_intra,
        scoring=scoring,
        random_state=random_state,
    )
    if not df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
