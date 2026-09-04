"""
research/ic — modular IC analysis (v2).

Entry point: ``python -m research.ic_analysis_v2``

ICIR convention: std uses ddof=0 (population) project-wide in this package.
"""
from research.ic.statistics import ic_stats, icir, newey_west_t, prepare_ic_for_stats
from research.ic.ic_series import compute_ic_series

__all__ = [
    "ic_stats",
    "icir",
    "newey_west_t",
    "prepare_ic_for_stats",
    "compute_ic_series",
]
