"""
Ensemble combination: z-score averaging, IC-weighted window/model selection.
"""
from __future__ import annotations

import numpy as np


def cross_sectional_zscore_scores(scores: np.ndarray) -> np.ndarray:
    """Z-score a 1-D score vector cross-sectionally."""
    if len(scores) < 2:
        return np.zeros_like(scores, dtype=float)
    mu, sigma = np.nanmean(scores), np.nanstd(scores)
    if sigma < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - mu) / sigma


def to_rank(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks / max(len(arr) - 1, 1)


def softmax_weights(values: list[float], temperature: float = 1.0) -> np.ndarray:
    """Softmax over values; NaN → 0 weight."""
    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), -np.inf, arr)
    arr = arr / max(temperature, 1e-6)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    exp = np.where(np.isfinite(exp), exp, 0.0)
    s = exp.sum()
    if s <= 0:
        n = len(values)
        return np.ones(n) / n if n else np.array([])
    return exp / s


def ic_weighted_weights(val_ics: list[float], method: str = "ic_weighted") -> np.ndarray:
    """
    Compute combination weights from validation ICs.

    ``average``: equal 1/N (v1 compat).
    ``best_window`` / ``best_model``: one-hot on argmax val IC.
    ``ic_weighted``: max(0, IC) normalized, or softmax if all <= 0.
    """
    n = len(val_ics)
    if n == 0:
        return np.array([])
    if method == "average":
        return np.ones(n) / n
    if method in ("best_window", "best_model"):
        w = np.zeros(n)
        best = int(np.nanargmax(val_ics))
        w[best] = 1.0
        return w
    # ic_weighted (default)
    clipped = [max(0.0, ic) if np.isfinite(ic) else 0.0 for ic in val_ics]
    if sum(clipped) > 0:
        arr = np.array(clipped)
        return arr / arr.sum()
    return softmax_weights(val_ics, temperature=0.5)


def select_window_weights(
    val_ics: list[float],
    wf_selection: str = "ic_weighted",
) -> np.ndarray:
    """Alias for window-level IC weighting."""
    return ic_weighted_weights(val_ics, method=wf_selection)


def combine_model_scores(
    score_matrix: list[np.ndarray],
    weights: np.ndarray | None = None,
    method: str = "zscore",
    output_rank: bool = False,
) -> np.ndarray:
    """
    Combine multiple model/window score vectors.

    ``zscore`` (default): CS z-score each vector → weighted average.
    ``rank``: rank each → weighted average (v1 compat).
    ``output_rank``: if True, convert final combined score to percentile rank.
    """
    if not score_matrix:
        return np.array([])
    n = len(score_matrix)
    if weights is None or len(weights) != n:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(n) / n
        else:
            weights = weights / weights.sum()

    transformed = []
    for scores in score_matrix:
        if method == "rank":
            transformed.append(to_rank(scores))
        else:
            transformed.append(cross_sectional_zscore_scores(scores))

    combined = np.zeros_like(score_matrix[0], dtype=float)
    for w, t in zip(weights, transformed):
        combined += w * t

    if output_rank:
        return to_rank(combined)
    return combined


def dynamic_model_weights(
    rolling_val_ic: dict[str, list[float]],
    model_types: list[str],
    temperature: float = 0.5,
) -> dict[str, float]:
    """
    Rolling validation IC → softmax weights per model type (Issue ⑦).
    """
    means = []
    for m in model_types:
        ics = rolling_val_ic.get(m, [])
        recent = [x for x in ics[-6:] if np.isfinite(x)]
        means.append(float(np.mean(recent)) if recent else 0.0)
    w = softmax_weights(means, temperature=temperature)
    return {m: float(w[i]) for i, m in enumerate(model_types)}
