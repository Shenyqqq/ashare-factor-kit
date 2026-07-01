# LGBM train-window sweep — h10

Batch: manual sweep (2026-07-01) | log: `sweep.log` | IC log: `ic_h10.log`

**IC screening:** 34 factors selected (Barra pure IC) → `config/factor_configs.yaml` key `h10`  
**ML rebalance:** `2W-FRI` (10-day hold) | **Env:** `TRAIN_N_JOBS=20`, `TRAIN_MAX_WORKERS=1`

**Completed with backtest:** 3/3 runs.

| windows (months) | IC mean | ICIR | IC>0% | monotonicity | top30 ann return | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3,6 | 0.0156 | 0.1289 | 58.8% | 0.2444 | 0.64% | OK |
| 6,12 | 0.0199 | 0.1733 | 60.2% | 0.3000 | 7.34% | OK |
| 12,24 | 0.0178 | 0.1653 | 59.0% | 0.2750 | 8.72% | OK |

## Notes

- Best **ICIR** and **top30 ann return** at **6,12** months (ICIR 0.173, top30 ~7.3% ann).
- **12,24** slightly higher top30 ann (~8.7%) but lower monotonicity than 6,12.
- **3,6** underperforms on all backtest metrics (weak monotonicity, ~0.6% top30 ann).
- Factor cache reused after first run (`--skip-factor-build` on runs 2–3).
- Monotonicity = mean yearly Spearman rank correlation Q1→Q5 vs returns (`backtest_*_annual.csv`).

## Code change

- `models/trainer.py`: minimal **2W-FRI** support for biweekly rebalance dates and train-window month→period conversion (h10 was falling back to monthly before this fix).
