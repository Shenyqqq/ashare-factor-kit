# Ridge window sweep — deep analysis

Date: 2026-07-01. Scope: 6 runs in `results/ridge_window_sweep/` (h5/h10/h20 × w6-12 / w12-24). Read-only; no new training.

**Update (2026-07-01):** h10 rebalance calendar fixed — `utils/rebalance_dates.py` shared helper uses `resample(rebalance_freq).last()` everywhere (replaces broken `to_period("2W")`). h10 ridge rows in `summary.md` re-run post-fix; ML IC dates now align with backtest (~203 calendar / ~135–164 OOS biweekly vs ~434 weekly before).

Reference: `summary.md`, `optimization_notes.md`, code paths in `strategies/ml.py`, `models/trainer.py`, `backtest/quantile.py`, `run.py`.

---

## Executive summary

| Horizon | IC mean | Quintile monotonicity | Top30 ann | LS (Q5−Q1) | Verdict |
|--------|---------|----------------------|-----------|------------|---------|
| h5 w12-24 | 0.059 | **+0.53** (good) | **~41%** | +59% cum | **Plausibly real** — IC, quintiles, and Top30 align |
| h10 w12-24 | 0.054 | **+0.80** (good) | ~17% | +105% cum | Signal rank-orders OK; **Top30 underperforms Q5** (concentration / 2021 crash) |
| h20 w12-24 | 0.063 | **−0.29** (inverted) | ~16% | **−26% cum** | IC looks fine on paper; **backtest quintiles invert** — do not trust for production |

---

## 1. Is IC computed with horizon-matched forward return?

### Verdict: **Yes for the label definition; partially misaligned for h10/h20 rebalance calendars**

### Code path (training OOS IC)

1. **`run.py`** maps horizon → rebalance freq and passes `hold_period=horizon`:

```147:156:run.py
def _horizon_to_rebalance_freq(horizon: int) -> str:
    if horizon <= 3:
        return "3D"
    elif horizon <= 7:
        return "W-FRI"
    elif horizon <= 15:
        return "2W-FRI"
    else:
        return "ME"
```

2. **`strategies/ml.py`** builds N-day forward return (open t+1 → close t+N):

```24:37:strategies/ml.py
def _compute_forward_return(prices, hold_period, open_=None):
    if open_ is not None:
        buy_price = open_.shift(-1)
        sell_price = prices.shift(-hold_period)
        return (sell_price / buy_price.replace(0, float("nan")) - 1).astype("float32")
```

3. **`models/trainer.py`** IC at each prediction date `t` uses factor scores vs `forward_return.loc[t]`:

```154:157:models/trainer.py
        y = self.forward_return.loc[date].reindex(X.index).dropna()
        X = X.loc[X.index.intersection(y.index)]
        return X, y.loc[X.index]
```

```464:474:models/trainer.py
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            ...
            ic_dict[date] = spearman_ic(s.values, y.values)
```

So at signal date **t**, IC correlates scores with **close[t+N] / open[t+1] − 1** where **N = horizon**. The label is horizon-matched by construction.

### Caveats (why IC can still disagree with backtest)

| Issue | h5 | h10 | h20 |
|-------|----|-----|-----|
| ML rebalance dates vs backtest dates | Aligned (~303 vs 284) | **Fixed (2026-07-01)** — was ~366 IC / 173 BT; now ~136–162 IC ≈ 135–161 BT | Partial — 70 score dates, 47 backtest periods |
| `build_ml_dataset` vs `quantile` resample | `to_period("W")` ≈ `resample("W-FRI")` | **Fixed** — shared `get_rebalance_dates()` uses `resample("2W-FRI")` | `to_period("M")` last trading day vs `resample("ME")` — close but not identical |
| Return definition | Label: point-to-point; BT: daily compound + open exec day | Same mismatch, worse date offset | Label: 20 trading days; BT: month-end to month-end (~20–23 days), compounded |
| OOS sample size | 303 IC periods | 366 IC / 173 BT | **70 IC / 47 BT** — high variance |

**Rebalance date bug (h10) — FIXED 2026-07-01:** pandas `to_period("2W")` was weekly (~434 calendar dates); replaced with `resample("2W-FRI").last()` (~203 calendar, ~135–164 OOS). Post-fix ridge h10: IC slightly higher (0.055–0.057 vs 0.050–0.054) but Top30 backtest weaker (−9% to −12% ann vs +2% to +17% pre-fix) — prior h10 backtest metrics were inflated by calendar mismatch, not comparable apples-to-apples.

**IC analysis:** `research/ic_analysis.py` now uses `horizon_to_rebalance_freq(period)` → `2W-FRI` for period=10, aligned with `run.py`.

---

## 2. Why good IC across windows/horizons but h10/h20 backtest weaker?

### h5 — consistent (IC ↔ backtest)

From `backtest_ridge_h5_w12-24_nav.csv` (284 periods):

- Q1 → Q5 cumulative: **+23% → +93%** (monotonic)
- Top30: **+695%** cum (~41% ann in summary)
- Long-short: **+59%**
- Holdings: ~80% main-board (SH/SZ), ~13% ChiNext+STAR — not a microcap-only illusion

IC and backtest tell the same story. w12-24 slightly beats w6-12 on ICIR (0.57 vs 0.54) with similar Top30 ann.

### h10 — rank signal works; Top30 implementation lagging

Quintiles are **monotonic** (Q1 +6% → Q5 +116% cum; summary monotonicity **0.80**). Long-short **+105%**. So the cross-sectional rank information is real at the quintile level.

Top30 ann only **~17%** vs Q5 implied strength because:

1. **Concentration risk** — Top30 is 30 names; one bad month dominates. `backtest_ridge_h10_w12-24_annual.csv`: **2021 Top30 −22%** while Q5 **+28%**.
2. **Weekly IC vs biweekly trading** — 366 prediction dates, only 173 traded; IC averages over non-traded weeks too.
3. **Board mix** — slightly more STAR/BSE than h5 (6.6% / 5.0% vs 4.4% / 3.3%); late sample BSE rises to 9.7%.
4. **Still beats 沪深300** (+17% vs index) but not the h5 headline numbers.

h10 is **not** an IC failure; it is a **portfolio construction / frequency mismatch** story.

### h20 — IC positive, quintiles inverted

From `backtest_ridge_h20_w12-24_nav.csv` (only **47** periods):

- Q1 **+121%** cum vs Q5 **+71%** → monotonicity **−0.29**
- Long-short **−26%**
- Top30 **+134%** — above Q5 but driven by early bull years; flat 2023–2026

Yearly pattern (`backtest_ridge_h20_w12-24_annual.csv`): in **2020–2022 and 2024**, Q1 (lowest scores) beats Q5 (highest). IC by year is weak/negative in 2020 (−0.019) while 2021–2025 IC is +0.04–0.10 — IC and quintile PnL decouple.

Contributing factors:

1. **Tiny OOS backtest sample** — 47 monthly periods vs 303 weekly for h5; `min_history` for w12-24 + val burns ~3y before first prediction.
2. **Return definition drift** — IC label is exactly 20 trading days; backtest holds signal month-end → next month-end with daily compounding and variable length.
3. **Style crowding in Top30** — holdings **27% BSE, 15% STAR, 10% ChiNext** vs h5 **~13%** growth board combined. High-score names skew small/growth; these regimes reversed vs low-score value/main-board names in 2022–2023.
4. **Ridge linear blend** — 34–44 correlated z-scored factors; monthly refit may overweight factors that worked in training window but flip sign OOS (lgbm h20 also shows negative monotonicity in `lgbm_window_sweep`, so not ridge-only).

---

## 3. Possible mismatches checklist

| Check | Finding |
|-------|---------|
| ML rebalance freq vs backtest freq | **h5 OK**; **h10 broken** (weekly ML vs biweekly BT); **h20 approximate** |
| forward_return (open t+1 → close t+N) vs BT holding | Same economic intent; BT compounds daily path — small gap h5, larger h20 |
| IC horizon vs BT holding period | IC always N-day label at t; BT holds until **next rebalance date** (calendar bucket) |
| OOS period count | h5 303 / h10 366 IC (173 BT) / **h20 70 IC (47 BT)** |
| Costs | 3 bp one-way turnover penalty in `quantile.py`; not enough to explain h20 inversion |
| Universe filter | `MIN_MARKET_CAP=20e8` — wide cap range; Top30 still picks many small/growth names especially h20 |
| Score type | Walk-forward outputs **rank averages** [0,1]; qcut still valid |

---

## 4. Top30 holdings diversity (cap / board)

Holdings CSVs only contain codes (no cap column). Board prefix proxy:

| Tag | SH main | SZ main | ChiNext | STAR | BSE |
|-----|---------|---------|---------|------|-----|
| h5 w12-24 | 38.5% | 41.3% | 8.7% | 4.4% | 3.3% |
| h10 w12-24 | 38.0% | 36.5% | 9.6% | 6.6% | 5.0% |
| h20 w12-24 | 23.4% | 19.4% | 10.1% | 15.2% | **27.4%** |

- **h5:** Diverse main-board heavy; includes large liquified names (e.g. 601012, 601888 in early rows). Supports "signal may be real" — not purely sub-20e9 microcap churn.
- **h10:** Similar to h5 with modestly more growth board; Top30 pain is timing/concentration not universal microcap illusion.
- **h20:** Material **BSE/STAR/ChiNext** overweight vs h5 → higher beta to small-growth factor exposures; when those factors mean-revert, high-score bucket underperforms despite positive average IC.

Horizon lengthening **increases** growth/small tilt in Top30 — likely interaction with factor set (momentum/quality work weekly; monthly ridge loads different linear combo).

---

## 5. h20 negative monotonicity — root cause hypothesis

**Primary hypothesis:** Small-sample monthly backtest + **style exposure inversion** in the high-score quintile, compounded by **calendar/return-definition drift** between IC label and month-to-month BT — not a pure "IC is wrong" bug.

**Evidence:**

1. Cumulative nav: Q1 1.21× vs Q5 1.71× → inverted spread; LS 0.74× (`backtest_ridge_h20_w12-24_longshort.csv`).
2. Annual 2020–2024: Q1 beats Q5 in 4 of 5 years (`backtest_ridge_h20_w12-24_annual.csv`).
3. IC series 2020 mean **−0.019** (n=6) while later years **+0.04–0.10** — early OOS already wrong-sided.
4. Holdings board mix: BSE 27% vs h5 3%.
5. lgbm h20 w12-24 also shows monotonicity **−0.19** (`lgbm_window_sweep/summary.md`) — structural h20/monthly issue, not ridge-specific numerics (cholesky/scaler removal).
6. Only **47** traded months — monotonicity metric in `build_summary.py` averages yearly rank correlations; one bad year moves the needle heavily.

**Secondary hypothesis:** `to_period("M")` vs `resample("ME")` drops or shifts ~33% of score dates (23/70 not backtested), biasing BT to months with sufficient liquidity/stocks.

---

## 6. Is ridge "real" for h5 vs h20?

| | h5 | h20 |
|---|-----|-----|
| IC ICIR | 0.57 | 0.47 |
| Quintile monotonicity | +0.53 | **−0.29** |
| LS PnL | +59% | **−26%** |
| IC ↔ BT agreement | **Strong** | **Weak / opposite** |
| Production readiness | **Candidate baseline** (freeze w12-24, then TIME_DECAY sweep) | **No** — fix rebalance alignment, extend OOS, regime-slice IC before revisit |

**h5 ridge plausibly real:** Positive IC, monotonic quintiles, strong Top30, main-board diversified holdings, weekly ML/BT mostly aligned.

**h20 ridge not validated:** IC statistics look acceptable but portfolio sorts invert; likely mix of frequency bug, small N, and growth/small-cap factor crowding. Do not deploy on h20 until rebalance dates unified and regime/year IC validated.

---

## 7. Recommended next steps (no new runs in this analysis)

1. ~~Fix `build_ml_dataset` to use the same resample rule as `quantile._get_rebalance_dates` (especially `2W-FRI`).~~ **Done** — `utils/rebalance_dates.py`.
2. Freeze **ridge h5 w12-24**; TIME_DECAY + slim YAML on that baseline only.
3. For h10, compare Top30 vs Q5 equal-weight within top quintile before blaming the model.
4. For h20, defer; require ≥100 OOS months and positive yearly monotonicity in 3+ of last 5 years.
5. Add optional cap/ADV columns to holdings export for future liquidity audits.

---

## Appendix: key metrics (w12-24 windows)

| Tag | IC | ICIR | IC>0% | Pred periods | BT periods | Top30 ann | Monotonicity |
|-----|-----|------|-------|--------------|------------|-----------|--------------|
| ridge_h5_w12-24 | 0.059 | 0.57 | 73% | 303 | 284 | 0.41 | +0.53 |
| ridge_h10_w12-24 | 0.054 | 0.46 | 69% | 366 | 173 | 0.17 | +0.80 |
| ridge_h20_w12-24 | 0.063 | 0.47 | 69% | 70 | 47 | 0.16 | −0.29 |
