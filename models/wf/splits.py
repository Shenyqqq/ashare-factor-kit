"""
Walk-forward split utilities with purging and embargo (AFML Ch. 7).

Purging: remove training observations whose label horizon overlaps the
validation or prediction interval, preventing leakage through overlapping
forward-return labels.

Embargo: drop training samples within ``embargo_periods`` rebalance steps
after the training window ends and before validation starts, so serial
correlation in labels does not bleed into val IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_window_splits(
    idx: int,
    window: int,
    val_window: int,
    n_dates: int,
    *,
    min_train_window: int | None = None,
    window_specific_val: bool = True,
) -> tuple[int, int, int, int]:
    """
    Return (train_start, train_end, val_start, val_end) slice indices.

    ``window_specific_val=True`` (default): shared recent validation ending
    at ``idx`` for every train window; only train length ``W`` differs::

        val_window=V, predict at idx:
          any W:  train [idx-V-W, idx-V),  val [idx-V, idx)

        val_window=0 (no independent val):
          any W:  train [idx-W, idx),  val []  (val_start == val_end == idx)
          Caller still applies purge/embargo against pred to avoid label leak.

    ``window_specific_val=False`` (deprecated / tests only): legacy offset
    layout where longer windows push val earlier::

        min_w=6, val_window=6, predict at idx:
          W=6:  train [idx-12, idx-6),  val [idx-6, idx)
          W=12: train [idx-24, idx-12), val [idx-12, idx-6)

    All intervals satisfy ``val_end <= idx`` (no lookahead at prediction).
    """
    if window_specific_val:
        # Shared recent val for all train windows (V=0 → train abuts pred).
        t = idx - val_window
    else:
        # Legacy: longer train windows shift val earlier (deprecated).
        min_w = min_train_window if min_train_window is not None else window
        offset = max(0, window - min_w)
        t = idx - val_window - offset

    train_end = t
    train_start = max(0, train_end - window)
    val_start = t
    val_end = t + val_window

    train_start = max(0, min(train_start, n_dates))
    train_end = max(train_start, min(train_end, n_dates))
    val_start = max(train_end, min(val_start, n_dates))
    val_end = max(val_start, min(val_end, n_dates))
    return train_start, train_end, val_start, val_end


def embargo_train_end(
    train_end: int,
    embargo_periods: int,
) -> int:
    """Return effective train_end after embargo (exclude last ``embargo_periods``)."""
    return max(0, train_end - max(0, embargo_periods))


def _label_overlap_periods(
    date_i: pd.Timestamp,
    date_j: pd.Timestamp,
    rebalance_dates: list,
    hold_period_days: int,
    date_pos_map: dict | None = None,
) -> bool:
    """True if forward-return label at date_i overlaps interval starting at date_j."""
    if date_i >= date_j:
        return False
    if date_pos_map is not None:
        pos_i = date_pos_map.get(date_i)
        pos_j = date_pos_map.get(date_j)
        if pos_i is None or pos_j is None:
            return False
    else:
        pos_i = rebalance_dates.index(date_i)
        pos_j = rebalance_dates.index(date_j)
    avg_gap = 5
    if pos_i + 1 < len(rebalance_dates):
        gap = (rebalance_dates[pos_i + 1] - date_i).days
        if gap > 0:
            avg_gap = gap
    label_span = max(1, int(np.ceil(hold_period_days / avg_gap)))
    return pos_i + label_span > pos_j


def purge_train_indices(
    train_dates: list,
    val_dates: list,
    pred_date,
    rebalance_dates: list,
    hold_period_days: int,
    date_pos_map: dict | None = None,
) -> list:
    """
    Purge training dates whose forward-return label window overlaps val or pred.

    Reference: Lopez de Prado, *Advances in Financial Machine Learning*, §7.4.

    ``date_pos_map``（可选）: 预构建的 ``{date: position}`` 字典，避免在循环内
    反复调用 ``list.index()``（O(N) → O(1)）。未传入时回退到 ``list.index()``。
    """
    if not train_dates:
        return train_dates
    if date_pos_map is None:
        date_pos_map = {d: i for i, d in enumerate(rebalance_dates)}
    val_start = val_dates[0] if val_dates else None
    kept = []
    for d in train_dates:
        if val_start is not None and _label_overlap_periods(
            d, val_start, rebalance_dates, hold_period_days, date_pos_map,
        ):
            continue
        if pred_date is not None and _label_overlap_periods(
            d, pred_date, rebalance_dates, hold_period_days, date_pos_map,
        ):
            continue
        kept.append(d)
    return kept


def hold_period_to_embargo_periods(
    hold_period_days: int,
    rebalance_dates: list,
) -> int:
    """Minimum embargo in rebalance periods (>= 1)."""
    if len(rebalance_dates) < 2:
        return max(1, hold_period_days // 5)
    gaps = [
        (rebalance_dates[i + 1] - rebalance_dates[i]).days
        for i in range(min(20, len(rebalance_dates) - 1))
    ]
    avg = max(1, int(np.median(gaps)))
    return max(1, int(np.ceil(hold_period_days / avg)))
