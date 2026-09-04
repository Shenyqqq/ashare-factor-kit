"""
Optional model persistence per walk-forward fold (Issue ⑧).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd


def save_fold_model(
    model,
    model_type: str,
    window: int,
    pred_date,
    out_dir: Path,
) -> Path:
    """Save a single fold model; returns path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_type}_w{window}_{pd_date_str(pred_date)}"
    path = out_dir / f"{stem}.joblib"
    joblib.dump(model, path)
    return path


def save_lgbm_native(model, path: Path) -> None:
    """Optional LightGBM native format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def pd_date_str(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _git_commit(cwd: Path | None = None) -> str:
    """获取当前仓库 HEAD commit hash，失败时返回 'unknown'。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_models_manifest(
    entries: list[dict],
    out_dir: Path,
    metadata: dict | None = None,
    *,
    feature_neutralize: bool | None = None,
    neut_controls: str | None = None,
) -> None:
    """
    写模型 manifest JSON。

    Parameters
    ----------
    entries : list[dict]
        每个折叠模型的记录（含 path / model / window / date）。
    out_dir : Path
        manifest 写入目录。
    metadata : dict | None
        训练级共享元信息，会合并进每条 entry，建议包含：
        ``feature_names`` / ``rebalance_freq`` / ``hold_period`` /
        ``label_mode`` / ``params`` / ``git_commit``。
    feature_neutralize : bool | None
        训练时是否做了 Barra+行业残差化。显式传入时合并进 metadata
        （若 metadata 已含同名键则以 metadata 为准，避免双重覆盖）。
        live.daily_update 据此与 CLI ``--feature-neutralize`` 校验一致性。
    neut_controls : str | None
        残差化控制变量集合（``barra`` | ``size_industry``）。写入后
        live.daily_update 按同口径中性化；缺字段时 live 默认 ``barra``。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    if "feature_neutralize" not in meta and feature_neutralize is not None:
        meta["feature_neutralize"] = bool(feature_neutralize)
    if "neut_controls" not in meta and neut_controls is not None:
        meta["neut_controls"] = str(neut_controls)
    # 若未显式提供 git_commit，则自动获取一次
    if "git_commit" not in meta:
        meta["git_commit"] = _git_commit(out_dir.resolve() if hasattr(out_dir, "resolve") else None)

    enriched = []
    for e in entries:
        rec = dict(e)
        for k, v in meta.items():
            rec.setdefault(k, v)
        enriched.append(rec)

    manifest = out_dir / "models_manifest.json"
    manifest.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
