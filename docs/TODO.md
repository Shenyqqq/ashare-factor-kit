# Pending work

Consolidated checklist from recent ridge sweep / pipeline work. Ordered roughly by priority.

## Ridge baseline & tuning

- [ ] **Freeze ridge baseline** — h5, train windows `12,24` months (`ridge_h5_w12-24`: IC 0.059, Top30 ann ~41%, monotonicity OK)
- [ ] **TIME_DECAY sweep for ridge** — current `0.015`; grid e.g. `[0.005, 0.01, 0.015, 0.02, 0.03]` on frozen h5 baseline
- [ ] **Ridge-specific slim factor YAML** — subset of h5/h10/h20 whitelists tuned for linear model (drop collinear / low Barra-pure IC factors)
- [ ] **IC-by-year + regime slice validation** — cross-check OOS IC vs backtest by year and `research/output/market_regime_*.csv` buckets before trusting h10/h20

## Pipeline / code hygiene

- [ ] **Commit pending code changes** — ridge cholesky + no StandardScaler, `2W-FRI` rebalance path, `train_window_units` flag, sweep scripts, `docs/PIPELINE.md` updates
- [x] **Fix h10 rebalance date mismatch** — `utils/rebalance_dates.py` shared `resample(rebalance_freq).last()` replaces broken `to_period("2W")` in `build_ml_dataset`, `quantile`, and `ic_analysis` (2026-07-01)
- [x] **IC analysis v2** — modular `research/ic/` + `python -m research.ic_analysis_v2`: ddof=0 ICIR, IC clip/winsorize, Newey-West t, tradable-universe masks, min-stocks guard, rank method config, Barra `INDUSTRY_REFERENCE`, corr dedup max/p95, IC_after_cost proxy, stability metrics (2026-07-01)
- [x] **Walk-forward trainer v2 → 默认** — 模块化 `models/wf/` + `models/trainer.py`（已合并 v1，类名 `WalkForwardTrainer`，`WalkForwardTrainerV2` 为别名）：window-specific val, purged WF + embargo, IC-weighted ensemble, CS label z-score, diagnostics CSV，原 `trainer_v2.py` 已删除（2026-07-01）
- [x] **Wire driver.py to ic_analysis_v2** — driver 已切到 `research.ic_analysis_v2`，享受 NW t / 可交易池 / 扣成本 / 稳定性 (2026-07-02)
- [ ] **Dynamic re-verify after signed ICIR fix** — rerun `mode=dynamic` / `--blend-dynamic` once ICIR weight sign logic is confirmed

## AFML 方法论 (2026-07-02)

- [x] **Fractional Differencing** — `utils/fractional_diff.py`（AFML Ch.5，d=0.4），新增因子 `分数差分动量_20d`，保留长期记忆同时平稳
- [x] **PBO + Deflated Sharpe Ratio** — `research/pbo.py`（AFML Ch.13/15），接入 `logs/analyze_results.py`；当前 51 实验最优 dynamic_h20_lb12 的 DSR=0.0253，提示过拟合
- [x] **Clustered Feature Importance** — `models/wf/clustered_importance.py`（AFML Ch.6），接入 `models/wf/metrics.py` + `models/analyzer.py`，避免相关因子重要性分裂

## PIT 数据保护 (2026-07-02)

- [x] **M1 财务因子披露日对齐** — `utils/pit_align.py`，按法定披露窗口（Q1/Q3=+45 天、半年报=+75 天、年报=+120 天）做 PIT 对齐，修改 `factors/factor.py`/`barra_risk.py`/`factor_alpha.py`
- [x] **M4 退市股保留 + ST 时间序列** — `data/download_delisted.py`；股票池保留退市股，ST 状态时间序列化；改 `data/download.py`/`backtest/execution.py`/`backtest/quantile.py`/`run.py`/`research/ic/universe.py`/`load_data.py`/`cli.py`
- [x] **M2 行业 PIT 时间序列** — `data/industry/download_industry.py` 重写产出 `industry_map_panel.parquet`；`research/ic/barra.py` PIT 改造；`tests/test_industry_pit.py`
- [x] **PIT 审计报告** — `docs/PIT_AUDIT.md`（审计报告，2026-07-01）

## IC 分析模块优化 (2026-07-02，来自 docs/IC_ANALYSIS_REVIEW.md)

- [x] **P0-1 driver 切 v2** — `logs/driver.py` 改用 `research.ic_analysis_v2`
- [x] **P0-2 ML 特征中性化层** — `models/wf/labels.py::residualize_panel`；`strategies/ml.py` + `run.py` 接入 `--feature-neutralize`，修复 IC/ML 口径不一致
- [x] **P1-1 BH-FDR 校正** — `research/ic/statistics.py::benjamini_hochberg`；`selection.py::use_fdr`
- [x] **P1-2/3 Barra clean_ret + 行业中性化** — `factors/barra_risk.py` 的 beta/res_vol/momentum 用 clean_ret，Barra 因子自身做行业去均值
- [x] **P1-4 行业 PIT 时间序列** — `data/industry/download_industry.py` 重写产出 `industry_map_panel.parquet`；`research/ic/barra.py` PIT 改造；`tests/test_industry_pit.py`
- [x] **P1-5 rolling ICIR** — `research/ic/statistics.py` 新增 `IC滚动ICIR`
- [x] **P1-6 IC 衰减表增强** — `research/ic/decay_corr.py` 加 ICIR/t/half-life 列
- [x] **P1-7 JSON 元数据补全** — `research/ic/io.py` 加 universe_size/ic_series_length/sample_period/config_snapshot 等
- [x] **P1-8 IC_MIN_LISTING_DAYS 默认 252** — `config/settings.py`
- [x] **P1-9 t 阈值默认 2.5 + CLI** — `research/ic/selection.py` + `research/ic/cli.py`
- [x] **run.py kwarg 冲突修复** — ml_run 调用从 extra_kwargs 弹出 industry_map
- [x] **AKShare 接口修复** — `data/download.py` 修复 `_fetch_stock_list_with_metadata` 重复列名 + SZ symbol + 退市接口名

## 交易成本 (2026-07-02)

- [x] **bid-ask spread 成本** — `backtest/execution.py::bid_ask_spread_bps`；`config/settings.py::BID_ASK_SPREAD_BPS=10.0`；`run.py::--bid-ask-spread`

## Backtest & universe (lower urgency)

- [x] **Quantile backtest v2 → 默认** — 模块化 buy-and-hold 引擎合并到 `backtest/quantile.py`（execution/portfolio/return_engine/turnover/benchmark/report 子模块不变）；原 `quantile_v2.py` 已删除；`run.py --backtest-engine` 保留为 deprecated no-op（2026-07-01）
- [ ] **ADV ≥ 2500w liquidity filter** — apply consistently in train cross-section and backtest execution (user: not urgent)
- [ ] **Small-cap universe v2 experiment** — optional track; current MIN_MARKET_CAP=20e8 still allows wide cap range in Top30
- [ ] **IC-driven gross exposure scaling in backtest** — scale Top30 weight by rolling OOS IC / ICIR (research item)

## Future tracks

- [ ] **Industry model** — `IndustryWalkForwardTrainer` production criteria + YAML per horizon
- [ ] **Barra portfolio opt** — replace equal-weight Top30 with style/industry constrained optimizer（用户提到延后；截面因子中性化已做、IC 显著性已做）
- [ ] **Paid Tushare factors** — evaluate incremental IC after Barra orthogonalization
- [ ] **v1 general model definition criteria** — document when a model (ridge/lgbm/ensemble) is "production-ready": min OOS periods, monotonicity threshold, regime stability, holdings liquidity

## 已完成项汇总（2026-07-02）

- ✅ AFML 方法论：Fractional Differencing、PBO + DSR、Clustered FI
- ✅ PIT 数据保护：财务披露日对齐、退市股保留、行业 PIT 时间序列、PIT 审计报告
- ✅ IC v2 上线 driver：Newey-West、可交易池、BH-FDR、rolling ICIR、IC 衰减增强、JSON 元数据补全、t 阈值 2.5、`IC_MIN_LISTING_DAYS=252`
- ✅ ML 特征中性化（`--feature-neutralize`），IC/ML 口径一致
- ✅ Barra 因子修正：clean_ret + 行业去均值
- ✅ 交易成本：bid-ask spread
- ✅ AKShare 接口修复
- ✅ run.py kwarg 冲突修复
