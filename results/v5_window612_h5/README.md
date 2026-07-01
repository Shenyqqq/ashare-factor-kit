# v5_window612_h5 — 2026-06-30 batch

Training windows: **[6, 12]** months (`TRAIN_WINDOWS_MONTHS` in `models/trainer.py`).

Horizon: **5** (weekly rebalance). Factor whitelist: `config/factor_configs.yaml`.

## prior_partial/

Incomplete ablation from an earlier run (before output-dir convention):

- ensemble / ridge / lgbm h5 & h20 metrics
- matching `factor_scores_*.parquet` moved from `data/processed/`
- cat h5/h20 metrics copied from `results/v1_ablation/`

## Current batch (h5 all models) — **completed 2026-06-30 22:14**

All outputs for this wave in this folder via `--output-dir results/v5_window612_h5/`.

| Mode | IC均值 | ICIR | IC>0胜率 |
|------|--------|------|----------|
| ridge | 0.0165 | 0.11 | 60% |
| mlp | 0.0157 | 0.12 | 51% |
| rf | 0.0001 | ~0 | 52% |
| cat | -0.0027 | -0.02 | 47% |
| lgbm | -0.0122 | -0.09 | 43% |
| ensemble | -0.0224 | -0.18 | 40% |
| xgb | -0.0233 | -0.19 | 34% |
| dynamic | n/a | n/a | ICIR factor timing (backtest only) |

Per run: `factor_scores_<tag>.parquet`, `model_metrics_<tag>.json`, `backtest_<tag>.png`, `holdings_top30_<tag>.csv`, nav/annual CSVs.

Log: `logs/h5_allmodels_w612.log` (PowerShell exit-code summary unreliable due to loguru stderr; all 8 runs produced outputs.)
