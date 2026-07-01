# LGBM train-window sweep

Batch: `results/lgbm_window_sweep/run_sweep.ps1` | log: `sweep.log`

**Completed with backtest:** 5/6 runs.

| horizon | windows (months) | IC mean | ICIR | IC>0% | monotonicity | top30 ann return | status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 3,6 | 0.0161 | 0.1576 | 0.5512 | 0.225 | 0.0957 | OK |
| 5 | 6,12 | 0.0066 | 0.0671 | 0.4986 | 0.1375 | 0.1348 | OK |
| 5 | 12,24 | 0.0073 | 0.0809 | 0.505 | 0.2 | 0.1782 | OK |
| 20 | 3,6 | N/A | N/A | N/A |  |  | FAILED (no backtest) |
| 20 | 6,12 | 0.0172 | 0.1415 | 0.5488 | -0.45 | 0.1206 | OK |
| 20 | 12,24 | 0.0299 | 0.2481 | 0.6 | -0.1857 | 0.0842 | OK |

## Failures / notes

- Initial script used `--skip-factor-build` on run 4 before h20 factor cache existed (fixed: per-horizon skip).
- h20 + windows 3,6: Walk-Forward had 0 prediction dates; backtest failed (empty factor_scores).

## Script fix

`--skip-factor-build` applies after the first **successful** run per horizon (h5 / h20), not after run 1 globally.

Monotonicity and top30 ann return come from backtest CSVs (not in `model_metrics_*.json`).
