"""Two-stage ridge refinement on S1 top-frac pools (PIT-safe).

Design
------
1. Stage 1 produces full-universe scores ``s1_scores`` (existing WF ridge/ensemble).
2. For each predict date:
   - Current pool = top ``pool_frac`` names by S1 (score only — no forward returns).
     Default ``pool_frac=0.2`` (Top20%). Under full-market quintiles this ≈ **Q5**,
     so the in-pool equal-weight benchmark should track single-stage **Q5** under the
     same tradable universe and cost assumptions (use as a sanity / bug check).
   - Stage-2 training rows = historical cross-sections filtered to names that were
     in the S1 top-frac pool *at those historical dates* (membership is PIT-safe).
    - **Per historical day** (not pooled across days): within that day's S1 pool,
     apply cross-sectional winsorize → cs_zscore to both **labels** and **features**,
     then concatenate days for Ridge MSE. Predict day uses the same in-pool
     winsor→zscore on features before scoring.
   - Training window ends with an embargo so forward-return labels do not overlap
     the predict date (same spirit as AFML Ch.7 purge/embargo).
   - Fit a RidgeCV on the stacked in-pool rows; predict S2 on the current pool.
3. Outside-pool names are filled with ``-inf`` (dates with any in-pool score) so
   Top-N stays in-pool while full-universe ``qcut`` / benchmark do **not** silently
   shrink to the pool (NaN→dropna would otherwise make Q1–Q5 = within-pool
   quintiles and "benchmark" = pool EW). Warm-up dates with no S2 fit stay all-NaN
   and are skipped by the backtest.

Feature order when ``--feature-neutralize`` is on
-------------------------------------------------
S1 features may already be globally Barra+industry residualized; stage 2 then
re-applies **in-pool** winsor→zscore on that residualized subset (does not undo
neutralization).

Stage-1 cache
-------------
Universe (scores + pool mask + meta) can be persisted via
``models.wf.stage1_cache`` — not X/y. Stage 2 may reload the mask and recompute
features from a different factor whitelist.

Documented for ablation via ``--two-stage``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from models.wf.labels import cross_sectional_zscore
from models.wf.models import fit_model, predict_model
from models.wf.splits import hold_period_to_embargo_periods

# Match global FWD_RETURN_WINSOR / factor winsor default (1% / 99%).
_DEFAULT_WINSOR = (0.01, 0.99)
# Default S1 pool = Top20% ≈ market Q5 under equal-frequency quintiles.
DEFAULT_STAGE2_POOL_FRAC = 0.2


def top_frac_index(scores: pd.Series, frac: float) -> pd.Index:
    """Return index of top ``frac`` names by score (higher = better)."""
    s = scores.dropna()
    if s.empty:
        return pd.Index([])
    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"pool_frac must be in (0, 1], got {frac}")
    n = max(1, int(np.ceil(len(s) * frac)))
    return s.nlargest(n).index


def _winsorize_1d(
    x: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    """Cross-sectional winsorize on a single day's pool vector."""
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    if int(mask.sum()) < 3:
        return x
    lo = float(np.nanquantile(x[mask], lower))
    hi = float(np.nanquantile(x[mask], upper))
    out = x.copy()
    out[mask] = np.clip(x[mask], lo, hi)
    return out


def pool_cs_winsor_zscore(
    values: pd.Series | np.ndarray,
    *,
    lower: float = _DEFAULT_WINSOR[0],
    upper: float = _DEFAULT_WINSOR[1],
) -> np.ndarray:
    """In-pool cross-section: winsorize then z-score (1d).

    Formula (per day, within S1 pool only)::

        y' = winsor_{lo,hi}(y)
        y'' = (y' - mean(y')) / std(y')
    """
    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(values, dtype=np.float64)
    arr = _winsorize_1d(arr, lower=lower, upper=upper)
    return cross_sectional_zscore(arr).astype(np.float32)


def pool_cs_winsor_zscore_frame(
    X: pd.DataFrame,
    *,
    lower: float = _DEFAULT_WINSOR[0],
    upper: float = _DEFAULT_WINSOR[1],
) -> pd.DataFrame:
    """Column-wise in-pool winsorize → z-score for one cross-section."""
    if X.empty:
        return X.astype(np.float32)
    out = pd.DataFrame(index=X.index, columns=X.columns, dtype=np.float32)
    for c in X.columns:
        out[c] = pool_cs_winsor_zscore(X[c], lower=lower, upper=upper)
    return out


def _pool_for_date(
    s1_row: pd.Series,
    pool_frac: float,
    pool_mask: pd.DataFrame | None,
    date,
) -> pd.Index:
    """Resolve pool membership from cache mask (if present) or live Top-frac."""
    if pool_mask is not None and date in pool_mask.index:
        m = pool_mask.loc[date].fillna(False).astype(bool)
        return m.index[m]
    return top_frac_index(s1_row, pool_frac)


def _stack_pool_rows(
    dataset: Any,
    dates: list,
    s1_scores: pd.DataFrame,
    pool_frac: float,
    feature_names: list[str],
    *,
    pool_mask: pd.DataFrame | None = None,
    winsor: tuple[float, float] = _DEFAULT_WINSOR,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stack X/y for historical dates, keeping only S1 top-frac names each day.

    Each day is transformed **separately** (winsor→zscore within that day's pool)
    before concatenation — never z-score across the multi-day stack.
    """
    lo, hi = winsor
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for d in dates:
        if d not in s1_scores.index:
            continue
        X, y = dataset.get_cross_section(d)
        if X is None or y is None or len(X) == 0:
            continue
        pool = _pool_for_date(s1_scores.loc[d], pool_frac, pool_mask, d)
        common = X.index.intersection(pool).intersection(y.index)
        if len(common) < 5:
            continue
        X_p = pool_cs_winsor_zscore_frame(
            X.loc[common, feature_names], lower=lo, upper=hi,
        )
        y_p = pool_cs_winsor_zscore(y.loc[common], lower=lo, upper=hi)
        X_list.append(X_p.values.astype(np.float32))
        y_list.append(y_p.astype(np.float32))
    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


def apply_two_stage_ridge(
    dataset: Any,
    s1_scores: pd.DataFrame,
    *,
    hold_period: int,
    pool_frac: float = DEFAULT_STAGE2_POOL_FRAC,
    lookback_periods: int | None = None,
    min_train_samples: int = 200,
    min_pool_size: int = 30,
    winsor: tuple[float, float] = _DEFAULT_WINSOR,
    pool_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refine S1 scores with a rolling in-pool ridge (stage 2).

    Parameters
    ----------
    dataset
        ``MLDataset`` (needs ``get_cross_section``, ``feature_names``,
        ``rebalance_dates``). Stage-2 features come from this dataset (may differ
        from the stage-1 factor set when loading an S1 universe cache).
    s1_scores
        Stage-1 score panel (index=predict dates, columns=stocks).
    hold_period
        Label horizon in trading days; used for embargo length.
    pool_frac
        Fraction of universe kept by S1 each day (default 0.2 = top 20% ≈ Q5).
        Ignored for membership when ``pool_mask`` is provided.
    lookback_periods
        Max historical rebalance periods used for stage-2 training.
        Default: ``max(24, 2 * embargo_periods)``.
    min_train_samples
        Skip stage-2 fit when stacked rows are below this.
    min_pool_size
        Skip predict day when current S1 pool is smaller than this.
    winsor
        In-pool cross-sectional winsor bounds ``(lower, upper)`` applied to
        both labels and features before cs_zscore (default 1%/99%).
    pool_mask
        Optional boolean panel from ``stage1_cache``; when set, pool membership
        follows the cached universe instead of recomputing Top-frac.

    Returns
    -------
    pd.DataFrame
        S2 score panel aligned to ``s1_scores``. In-pool names have finite S2
        scores; out-of-pool names are ``-inf`` on fitted dates (see module
        docstring). Unfitted warm-up dates remain all-NaN.

    Label / feature transform (per day, within S1 pool)
    --------------------------------------------------
    ::

        x'' = zscore(winsor(x));  y'' = zscore(winsor(y))
        Ridge MSE on stacked (x'', y''); predict with same x'' on pred day.
    """
    if s1_scores is None or s1_scores.empty:
        raise ValueError("s1_scores is empty; cannot run two-stage ridge")

    feature_names = list(dataset.feature_names)
    dates_all = list(dataset.rebalance_dates)
    date_to_pos = {d: i for i, d in enumerate(dates_all)}
    embargo = hold_period_to_embargo_periods(hold_period, dates_all)
    if lookback_periods is None:
        lookback_periods = max(24, 2 * max(1, embargo))
    lo, hi = winsor

    pred_dates = [d for d in s1_scores.index if d in date_to_pos]
    pred_dates = sorted(pd.to_datetime(pred_dates))
    s2_rows: dict = {}
    n_fit = 0
    n_skip = 0
    s1_s2_spearman: list[float] = []
    src = "cache_mask" if pool_mask is not None else "top_frac"

    logger.info(
        f"Two-stage ridge: pool_frac={pool_frac} (pool={src}), "
        f"lookback={lookback_periods} periods, "
        f"embargo={embargo} periods, in-pool winsor→zscore [{lo:.0%},{hi:.0%}], "
        f"pred_dates={len(pred_dates)}"
    )

    for pred_date in pred_dates:
        idx = date_to_pos[pred_date]
        # Train end excludes embargo periods before predict (label non-overlap).
        train_end = max(0, idx - max(1, embargo))
        train_start = max(0, train_end - lookback_periods)
        hist_dates = dates_all[train_start:train_end]
        # Only dates that already have S1 scores (OOS S1) — avoids using
        # in-sample S1 membership that was never produced for early warm-up.
        hist_dates = [d for d in hist_dates if d in s1_scores.index]
        if len(hist_dates) < 4:
            n_skip += 1
            continue

        s1_today = s1_scores.loc[pred_date]
        pool = _pool_for_date(s1_today, pool_frac, pool_mask, pred_date)
        if len(pool) < int(min_pool_size):
            n_skip += 1
            continue

        X_tr, y_tr = _stack_pool_rows(
            dataset, hist_dates, s1_scores, pool_frac, feature_names,
            pool_mask=pool_mask, winsor=winsor,
        )
        if X_tr is None or len(X_tr) < min_train_samples:
            n_skip += 1
            continue

        # Tiny holdout from the end of the train stack for RidgeCV / early checks.
        n_va = max(30, min(len(X_tr) // 5, 500))
        X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
        X_fit, y_fit = X_tr[:-n_va], y_tr[:-n_va]
        if len(X_fit) < min_train_samples // 2:
            X_fit, y_fit = X_tr, y_tr
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]

        w = np.ones(len(y_fit), dtype=np.float32)
        model = fit_model(
            "ridge", X_fit, y_fit, w, X_va, y_va,
            n_jobs=1, objective="regression",
            feature_names=feature_names,
        )

        X_pred, _ = dataset.get_cross_section(pred_date)
        if X_pred is None:
            n_skip += 1
            continue
        common = X_pred.index.intersection(pool)
        if len(common) < min_pool_size:
            n_skip += 1
            continue
        # Inference: same in-pool winsor→zscore on features as training.
        X_p = pool_cs_winsor_zscore_frame(
            X_pred.loc[common, feature_names], lower=lo, upper=hi,
        ).values.astype(np.float32)
        pred = predict_model(model, X_p, model_type="ridge")
        s2_today = pd.Series(pred, index=common, dtype=float)
        s2_rows[pred_date] = s2_today
        n_fit += 1

        # In-pool rank agreement with S1 (diagnostic; not used for scoring).
        s1_pool = s1_today.reindex(common).astype(float)
        if s1_pool.notna().sum() >= 30 and s2_today.notna().sum() >= 30:
            rho = s1_pool.corr(s2_today, method="spearman")
            if np.isfinite(rho):
                s1_s2_spearman.append(float(rho))

        if n_fit % 20 == 0:
            logger.info(
                f"Two-stage progress {n_fit}/{len(pred_dates)}: {pred_date.date()}, "
                f"pool={len(common)}, train_rows={len(X_tr)}"
            )

    if not s2_rows:
        logger.warning("Two-stage ridge produced no scores; returning empty frame")
        return pd.DataFrame(index=s1_scores.index, columns=s1_scores.columns, dtype=float)

    s2 = pd.DataFrame(s2_rows).T
    s2.index = pd.to_datetime(s2.index)
    s2.index.name = "date"
    # Align to S1; NaN = out of pool or unfitted date.
    s2 = s2.reindex(index=s1_scores.index, columns=s1_scores.columns)

    # Fitted dates: out-of-pool → -inf so dropna/qcut see the full universe.
    # All-NaN warm-up rows stay NaN (backtest skips them).
    fitted_mask = s2.notna().any(axis=1)
    if fitted_mask.any():
        s2.loc[fitted_mask] = s2.loc[fitted_mask].fillna(-np.inf)

    rho_mean = float(np.mean(s1_s2_spearman)) if s1_s2_spearman else float("nan")
    logger.info(
        f"Two-stage done: fitted={n_fit}, skipped={n_skip}, "
        f"coverage={fitted_mask.mean():.1%} dates with scores, "
        f"in-pool Spearman(S1,S2) mean={rho_mean:.3f} "
        f"(~0 ⇒ S2 reorders away from S1 head; Top-N ≠ S1 Q5)"
    )
    return s2
