"""Stage-1 universe cache for two-stage pipelines.

Caches only the S1 score panel + daily Top-frac pool mask (and meta).
Does **not** store in-pool features X or labels y — stage 2 recomputes those
from the current factor-config / registry on the cached stock×date universe.

Layout (under ``results/<tag>/stage1_cache/`` or a custom path)::

    s1_scores.parquet      # date × stock scores (higher = better)
    s1_pool_mask.parquet   # bool mask, True = in S1 top pool that day
    meta.json              # horizon, factor-config hash/path, flags, pool_frac, …

Changing the stage-1 model or YAML requires a new cache (see ``factor_config_hash``
and related meta keys). Stage-2 factor swaps reuse the universe only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from models.wf.two_stage import top_frac_index

CACHE_SCORES_NAME = "s1_scores.parquet"
CACHE_MASK_NAME = "s1_pool_mask.parquet"
CACHE_META_NAME = "meta.json"


def hash_file(path: str | Path | None) -> str | None:
    """SHA256 prefix (16 hex) of a file; None if path missing/empty."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def build_pool_mask(s1_scores: pd.DataFrame, pool_frac: float) -> pd.DataFrame:
    """Boolean panel: True where name is in S1 top ``pool_frac`` that day."""
    if s1_scores is None or s1_scores.empty:
        return pd.DataFrame()
    mask = pd.DataFrame(False, index=s1_scores.index, columns=s1_scores.columns)
    for d in s1_scores.index:
        pool = top_frac_index(s1_scores.loc[d], pool_frac)
        if len(pool):
            mask.loc[d, pool] = True
    return mask


def default_stage1_cache_dir(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "stage1_cache"


def save_stage1_cache(
    cache_dir: str | Path,
    s1_scores: pd.DataFrame,
    *,
    pool_frac: float,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write scores + pool mask + meta. Returns the cache directory."""
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    if s1_scores is None or s1_scores.empty:
        raise ValueError("s1_scores is empty; cannot save stage1 cache")

    scores = s1_scores.copy()
    scores.index = pd.to_datetime(scores.index)
    scores.index.name = "date"
    pool_mask = build_pool_mask(scores, pool_frac)

    scores.to_parquet(out / CACHE_SCORES_NAME)
    pool_mask.to_parquet(out / CACHE_MASK_NAME)

    rebalance_dates = [pd.Timestamp(d).isoformat() for d in scores.index]
    payload: dict[str, Any] = {
        "pool_frac": float(pool_frac),
        "n_dates": int(len(scores)),
        "n_stocks": int(scores.shape[1]),
        "rebalance_dates": rebalance_dates,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contents": {
            "s1_scores": CACHE_SCORES_NAME,
            "s1_pool_mask": CACHE_MASK_NAME,
            "note": "Universe only — no in-pool X/y; recompute features for stage 2.",
        },
    }
    if meta:
        # meta wins on overlapping keys except we always keep contents/rebalance.
        for k, v in meta.items():
            if k in ("contents", "rebalance_dates", "n_dates", "n_stocks"):
                continue
            payload[k] = v
    payload["pool_frac"] = float(pool_frac)

    with open(out / CACHE_META_NAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Stage1 cache saved → {out} "
        f"(dates={payload['n_dates']}, stocks={payload['n_stocks']}, "
        f"pool_frac={pool_frac})"
    )
    return out


def load_stage1_cache(
    cache_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load ``(s1_scores, s1_pool_mask, meta)`` from a cache directory."""
    root = Path(cache_dir)
    scores_path = root / CACHE_SCORES_NAME
    mask_path = root / CACHE_MASK_NAME
    meta_path = root / CACHE_META_NAME
    if not scores_path.is_file():
        raise FileNotFoundError(f"missing {scores_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing {mask_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")

    s1_scores = pd.read_parquet(scores_path)
    s1_scores.index = pd.to_datetime(s1_scores.index)
    s1_scores.index.name = "date"

    pool_mask = pd.read_parquet(mask_path)
    pool_mask.index = pd.to_datetime(pool_mask.index)
    pool_mask.index.name = "date"
    # Ensure boolean dtype
    pool_mask = pool_mask.astype(bool)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(
        f"Stage1 cache loaded ← {root} "
        f"(dates={len(s1_scores)}, pool_frac={meta.get('pool_frac')})"
    )
    return s1_scores, pool_mask, meta


def pool_index_from_mask(mask_row: pd.Series) -> pd.Index:
    """Names with True in a single-date pool mask row."""
    m = mask_row.fillna(False).astype(bool)
    return m.index[m]
