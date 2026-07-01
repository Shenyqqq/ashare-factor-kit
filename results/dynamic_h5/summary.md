# Dynamic h5 lookback sweep (W-FRI rebalance periods)

Generated: 2026-07-01

Commands:
```powershell
$env:TRAIN_N_JOBS='20'; $env:DYNAMIC_MAX_WORKERS='4'; $env:TRAIN_MAX_WORKERS='1'; $env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 5 --dynamic-lookback 26 --factor-config config/factor_configs.yaml --output-dir results/dynamic_h5/
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 5 --dynamic-lookback 52 --factor-config config/factor_configs.yaml --output-dir results/dynamic_h5/ --skip-factor-build
```

Tags: `dynamic_h5_lb26`, `dynamic_h5_lb52` (via `--dynamic-lookback`).

| Lookback | IC | ICIR | IC>0 | Monotonicity | Top30 ann | n_preds |
|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.0689 | 0.5953 | 72.2% | 0.600 | 4.60% | 392 |
| 52 | 0.0723 | 0.6106 | 72.6% | 1.000 | -2.96% | 379 |

## Notes

- Lookback is in **rebalance periods** (W-FRI for h5): 26 ≈ 6 months, 52 ≈ 12 months.
- `DynamicFactorTrainer` uses **signed ICIR** weights (`mu/std`, clip ±2); negative IC reduces/reverses factor contribution.
- lb52 has better Q1–Q5 monotonicity but weaker Top30 backtest vs lb26 on this sample.

## Artifacts

- **dynamic_h5_lb26**: `model_metrics_dynamic_h5_lb26.json`, `backtest_dynamic_h5_lb26_nav.csv`, `backtest_dynamic_h5_lb26.png`
- **dynamic_h5_lb52**: `model_metrics_dynamic_h5_lb52.json`, `backtest_dynamic_h5_lb52_nav.csv`, `backtest_dynamic_h5_lb52.png`
