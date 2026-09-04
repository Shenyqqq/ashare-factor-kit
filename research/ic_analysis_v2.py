"""
research/ic_analysis_v2.py — modular IC analysis entry point.

Usage:
    python -m research.ic_analysis_v2 --period 5 --neut-controls size_industry --save
    python -m research.ic_analysis_v2 --sample 5 --period 20
    python -m research.ic_analysis_v2 --help-advanced

Defaults include FDR, t=2.5, corr-dedup (Gram-Schmidt opt-in via --gram-schmidt).
See docs/操作手册.md.
"""
from research.ic.cli import main

if __name__ == "__main__":
    main()
