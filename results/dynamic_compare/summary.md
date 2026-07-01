# Dynamic h5/h20 lookback 6 vs 12 comparison

Generated: 2026-07-01

Signed ICIR: see `models/dynamic_trainer.py` (`weights = mu/std`, clip ±2; no `abs` on ICIR weights).

## Commands

```powershell
$env:TRAIN_N_JOBS='20'; $env:DYNAMIC_MAX_WORKERS='4'; $env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 5 --dynamic-lookback 6 --factor-config config/factor_configs.yaml --output-dir results/dynamic_compare/
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 5 --dynamic-lookback 12 --factor-config config/factor_configs.yaml --output-dir results/dynamic_compare/ --skip-factor-build
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 20 --dynamic-lookback 6 --factor-config config/factor_configs.yaml --output-dir results/dynamic_compare/ --skip-factor-build
.venv\Scripts\python.exe -u run.py --skip-download --mode dynamic --horizon 20 --dynamic-lookback 12 --factor-config config/factor_configs.yaml --output-dir results/dynamic_compare/ --skip-factor-build
```

Batch log: `logs/dynamic_compare_batch.log` (all four runs rc=0).

| horizon | lookback | IC | ICIR | IC>0% | monotonicity | top30 ann | n_preds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 6 | 0.0532 | 0.4416 | 66.9% | 0.000 | 2.00% | 402 |
| 5 | 12 | 0.0633 | 0.5285 | 72.9% | -0.011 | 1.52% | 399 |
| 20 | 6 | 0.0702 | 0.4357 | 73.0% | -0.044 | 2.20% | 63 |
| 20 | 12 | 0.0948 | 0.6453 | 75.0% | -0.044 | 2.09% | 60 |

Monotonicity and top30 ann from backtest annual/nav CSVs (same method as `results/dynamic_h5/summary.md`).

## vs previous h5 lookbacks (`results/dynamic_h5/`)

| lookback (h5 periods) | IC | ICIR | IC>0% | monotonicity | top30 ann | n_preds |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.0532 | 0.4416 | 66.9% | 0.000 | 2.00% | 402 |
| 12 | 0.0633 | 0.5285 | 72.9% | -0.011 | 1.52% | 399 |
| 26 (~6 mo) | 0.0689 | 0.5953 | 72.2% | 0.600 | 4.60% | 392 |
| 52 (~12 mo) | 0.0723 | 0.6106 | 72.6% | 1.000 | -2.96% | 379 |

- Shorter lookbacks (6–12) trail lb26/lb52 on IC/ICIR but keep modest positive Top30 (~1.5–2%); lb26 still best Top30 on this sample.
- lb52 improves Q1–Q5 monotonicity to 1.0 but Top30 turns negative; lb6/lb12 show weak or slightly negative monotonicity with smaller IC edge.
- h20: lb12 clearly wins on IC/ICIR (0.095 / 0.645 vs 0.070 / 0.436); backtest metrics similar (~2.1% Top30). Fewer n_preds reflects min lookback + ME rebalance count.

## Artifacts

Tags: `dynamic_h5_lb6`, `dynamic_h5_lb12`, `dynamic_h20_lb6`, `dynamic_h20_lb12` under `results/dynamic_compare/`.
