"""
research/ic_analysis_v2.py — modular IC analysis entry point.

Usage:
    python -m research.ic_analysis_v2 --period 5 --barra --save
    python -m research.ic_analysis_v2 --sample 5 --period 20

See research/ic/ package modules for implementation details.
"""
from research.ic.cli import main

if __name__ == "__main__":
    main()
