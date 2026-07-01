# Ridge training optimization review

Review date: 2026-07-01. Scope: `models/trainer.py` walk-forward path vs tree models (`lgbm`/`xgb`).

## Findings (no change needed)

| Area | Status |
|------|--------|
| **float32 matrices** | Already cached in `section_cache` (`X.values.astype(np.float32)`, `y.values.astype(np.float32)`). `forward_return` also cast to float32 in `build_ml_dataset`. |
| **Per-fold matrix copies** | `jobs` pass references to shared `X_va` / `X_pred_np`; only train stacks are rebuilt per window via `vstack`. Necessary. |
| **Validation stack once** | `X_va, y_va` computed once per rebalance date and reused across windows (optimization 2, existing). |
| **section_cache** | Same cache serves ridge and trees; ridge benefits equally from avoiding repeated `df.loc[date]`. |
| **Partial ridge state reuse** | Not applicable — walk-forward must refit each date/window; no warm-start API worth the complexity. |
| **Per-date batching** | Each date has different train window slice; batching folds would complicate time-decay weights with little gain for fast ridge fits. |
| **Ridge `n_jobs`** | Only `saga`/`sag` solvers use `n_jobs`; with ~40 features, `cholesky` is faster and single-threaded. `TRAIN_N_JOBS` remains relevant for tree models only. |
| **Tree-only overhead for ridge** | Early stopping / `eval_set` in `_fit_model` are tree-only; ridge path already skips them. Rank-averaging across windows is O(n_stocks) and negligible. |

## Redundant work (ridge-specific)

1. **Double standardization** — Registry factors are already `winsorize(1%) + cross_sectional_zscore(clip=3σ)` (`factors/factor.py`). The old ridge `Pipeline(StandardScaler → Ridge)` re-scaled the pooled train matrix every fold (~hundreds of dates × thousands of stocks). For linear ridge this is redundant and adds two full passes over `X_tr` per fit.

2. **Default solver `auto`** — For dense `(n_samples, ~40)` matrices sklearn often picks `svd`, which is slower than `cholesky` at this feature count.

## Changes applied (`models/trainer.py`)

| Change | Lines | Rationale |
|--------|-------|-----------|
| `RIDGE_PARAMS`: add `solver="cholesky"` | ~79 | Explicit fast dense solver for ~34–44 features; supports `sample_weight`. |
| `_build_model("ridge")`: return `Ridge(**RIDGE_PARAMS)` directly | ~238–240 | Remove `StandardScaler` pipeline; factors pre-normalized. |
| `_fit_model("ridge")`: `model.fit(X_tr, y_tr, sample_weight=w_tr)` | ~259–260 | Direct fit (no `model__sample_weight` pipeline kwargs). |

## Expected impact

- **Speed**: Moderate per-fold speedup (no scaler fit/transform; faster solver). Largest on long train windows (12,24 months) where `X_tr` has more rows.
- **Numerics**: Slight score/IC drift possible vs old pipeline+scaler path because L2 penalty now applies on pre-z-scored scale. Acceptable for ridge experiments; tree paths unchanged.

## Not implemented (low ROI / risky)

- Removing rank transform for single-model ridge (would break ensemble API contract).
- `float64` → `float32` inside sklearn (internal upcast; marginal).
- Caching `vstack` train matrices across adjacent walk-forward dates (memory heavy, complex invalidation).
