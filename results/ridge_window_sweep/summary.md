# Ridge train-window sweep

Batch: `results/ridge_window_sweep/run_sweep.ps1` | optimizations: `optimization_notes.md` | log: `sweep.log`

**Completed with backtest:** 6/6 runs.

| horizon | windows (months) | IC mean | ICIR | IC>0% | monotonicity | top30 ann return | wall min | status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 12,24 | 0.0593 | 0.5746 | 0.7294 | 0.5286 | 0.4146 | 2.0 | OK |
| 5 | 6,12 | 0.0566 | 0.5426 | 0.6901 | 0.675 | 0.4302 | 2.5 | OK |
| 10 | 12,24 | 0.0567 | 0.4885 | 0.7279 | 0.1143 | -0.1166 | 1.9 | OK |
| 10 | 6,12 | 0.055 | 0.4662 | 0.7222 | 0.0625 | -0.0956 | 1.7 | OK |
| 20 | 12,24 | 0.0631 | 0.4745 | 0.6857 | -0.2857 | 0.1612 | 0.6 | OK |
| 20 | 6,12 | 0.0527 | 0.3874 | 0.6951 | -0.15 | 0.1906 | 0.6 | OK |
