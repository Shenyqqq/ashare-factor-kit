# AGENTS.md

Agent 在本仓库工作时的精简上下文指南。与 [README.md](README.md) 互补（README 对外介绍；命令教程见 [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) / [docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md)；本文是 agent 上手速查）。

## 项目简介

A 股多因子量化选股框架，定位 **辅助人工选股，不做全自动交易**。
端到端流水线：数据下载 → 因子计算 → IC 筛选 → ML / 动态加权 → 分组回测 → 候选股输出。
模型输出得分排名，人工二次筛选后手动操作。

## 架构概览

数据流：`AKShare → data/raw/*.parquet（含退市股）→ clean_ret + masks（内存）→ PIT 对齐（财务/行业）→ IC v2 筛选（Newey-West/BH-FDR/可交易池）→ factor_configs.yaml → WalkForwardTrainer / DynamicFactorTrainer → factor_scores → backtest/quantile.py → results/<tag>/`。

| 顶层目录 | 用途 |
|----------|------|
| `run.py` | 主入口 CLI；日常见 `--help`，高级/deprecated 见 `--help-advanced` / [docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md) |
| `config/` | `settings.py`（全局参数，含 `IC_MIN_LISTING_DAYS=252`、`BID_ASK_SPREAD_BPS=10.0`）、`factor_configs.yaml`（因子白名单） |
| `data/` | 下载 / 清洗 / 行业 / 财务事件；`raw/` `universe/` `processed/`；`download_delisted.py` 保留退市股；`industry/download_industry.py` 产出 PIT `industry_map_panel.parquet` |
| `factors/` | 因子实现 + `get_factor_registry()` 注册中心；含分数差分动量；财务因子经 PIT 对齐；`special_factors.py`（event/size 等绕过 IC YAML 的注入包） |
| `models/` | 训练器（已合并 v2，含 `wf/` 子包）；`trainer.py` `dynamic_trainer.py` `industry_trainer.py` `analyzer.py`（含 clustered FI） |
| `strategies/` | `linear.py` / `ml.py`（ML 调度，含 `--feature-neutralize`）/ `market_state.py` |
| `backtest/` | 模块化引擎：`quantile.py` + `execution/portfolio/return_engine/turnover/benchmark/report`；含 bid-ask spread 成本 |
| `research/` | IC 分析：`ic_analysis_v2.py` + `ic/` 子包（v2，driver 默认）+ `ic_analysis.py`（v1 后备）；`pbo.py`（PBO+DSR） |
| `utils/` | `rebalance_dates.py` / `fractional_diff.py`（AFML Ch.5）/ `pit_align.py`（PIT 披露日对齐） |
| `tests/` | `test_wf_splits.py` / `test_industry_pit.py`（行业 PIT 单测） |
| `logs/driver.py` | 自动化编排：IC v2 → YAML → 批量实验 |
| `results/<tag>/` | 实验产物（gitignore） |

## 关键设计决策速查

| 主题 | 约定 |
|------|------|
| **因子方向** | 因子函数内部已取反，输出「越高越好」；`FACTOR_WEIGHTS` 仅 linear 模式用 |
| **标准化** | 截面 winsorize(1%) → cross_sectional_zscore(clip=3σ)；市场/HMM 特征用时序滚动 z-score，不做截面标准化 |
| **clean_ret** | 量价因子必须用 `clean_ret`（涨跌停日 return=NaN），禁用 `prices.pct_change()`；Barra_Beta/Barra_ResVol 也已切换 |
| **rebalance_dates** | `groupby(period).last()` 取周期末实际交易日，不用日历 `ME`/`W-FRI` 虚拟日期 |
| **forward_return** | `close[t+N] / open[t+1] - 1`（有 open_ 时），信号日收盘后次日开盘买入 |
| **Walk-Forward split** | purged training + embargo（AFML Ch.7）；默认两窗共用近期 val `[idx-V, idx)`、train `[idx-V-W, idx-V)`（`VAL_WINDOW_MONTHS=6`）；`val_window=0` → train `[idx-W, idx)`、无 val（仍 purge/embargo；多窗须 `wf_selection=average`）；`hold_period_to_embargo_periods()` 自动换算 |
| **集成方式** | 多训练窗口 × 多模型 → IC 加权 Z-score 平均（非盲 rank average） |
| **Clustered FI** | `models/wf/clustered_importance.py`（AFML Ch.6）按相关因子聚类算重要性，避免相关因子重要性分裂 |
| **SHAP** | `models/wf/shap_analysis.py`；CLI `--shap`（默认关）；折内 Tree/Linear SHAP → `results/<tag>/shap_*.csv`；与 FI/Clustered FI 并存 |
| **市场代理** | 中证全指（`data/raw/csi_all.parquet`）作市场 regime / 基准 |
| **Q1-Q5** | `pd.qcut` 升序，Q1=最低分，Q5=最高分；IC>0 为正向 |
| **分位多空分解** | `research/ic/quantile_decomp.py`：分组前按时序 pure IC 均值对齐（均值<0 则翻转 `resid_x`），在「越高越好」方向上算 Q5/Q1/`long_share`；「无效」仅分母≤0 等数值病态，不因负 IC 未翻转；pure IC 序列本身不取反；历史 CSV 需重跑 `--barra` 才更新 |
| **long_share 稠密门** | 默认 `long_share > 0.4`（`--min-long-share`，`0` 关闭），与 \|IC\|∧\|ICIR\|∧t/FDR 合取，在 corr-dedup 之前；须符号对齐列（或 `--long-share-csv`） |
| **成本** | `COMMISSION_RATE=0.0001`、`STAMP_DUTY=0.0005`、`SLIPPAGE_RATE=0.0`、`BID_ASK_SPREAD_BPS=10.0`（`--bid-ask-spread`） |
| **研究 vs 执行口径** | 默认 research（`TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY=False`，`FWD_RETURN_EXEC_MASK=False`）：IC/ML 信号日可交易池**保留涨跌停**、标签不做 execution mask；`clean_ret` 与回测 execution 不变。回测得分宇宙默认 `--bt-score-universe strict`（避免训练池膨胀泄漏进等权基准）。`--tradable-strict` / `--label-exec-mask` 恢复旧口径。IC ckpt 后缀 `_tmr_v2`。因子 **`涨跌停状态`**（1/2/3，limit_up 优先，截面 winsor+zscore） |
| **PIT 披露日对齐** | 财务默认法定披露窗口近似（Q1/Q3=+30、半年报=+60、年报=+90；非真实公告日）；长表有 `ann_date` 则优先；`utils/pit_align.py`；禁用报告期直接 `ffill` |
| **退市股 + ST 时间序列** | 股票池保留退市股（`data/download_delisted.py`）；ST：深交所精确历史 + 沪/北 `sh_bj_current_st_conservative_fallback` |
| **行业 PIT** | `industry_map_panel.parquet` 严格默认（`--barra` 缺失即报错）；仅 `--allow-static-industry` 可退化 |
| **IC v2 上线** | `logs/driver.py` 已切到 `research.ic_analysis_v2`；Newey-West HAC t、可交易池 mask、BH-FDR、rolling ICIR、扣成本 IC 均进入生产筛选 |
| **ML 特征中性化** | `--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化，与 IC 纯 IC 同口径 |
| **FDR 多重检验校正** | `research/ic/statistics.py::benjamini_hochberg` + `selection.py::use_fdr`（CLI 默认 ON，`--no-use-fdr` 关） |
| **过拟合检验** | `research/pbo.py` 算 PBO + Deflated Sharpe Ratio（AFML Ch.13/15）；当前 51 实验最优 DSR=0.0253，提示过拟合 |
| **分数差分** | `utils/fractional_diff.py`（AFML Ch.5，d=0.4 默认）→ 因子 `分数差分动量_20d`，保留长期记忆同时平稳 |
| **数据对齐** | 财务季报先 PIT 对齐再 `reindex(..., method="ffill")` 前向填充到日频 |
| **v2 为唯一实现** | 原 v1 单体已合并删除；`--trainer-engine` / `--backtest-engine` 为 deprecated no-op（`--help` 隐藏，仍接受） |
| **rolling-pool 特征** | 每期调仓日 t 用当日 `pool_t`（≤50）；该期 train/val/pred 共用同一组列；禁止窗内并集；lazy 运行时只 `ensure(pool_t)` |
| **rolling-pool sticky LRU** | WF 每期只 `ensure(pool_t)`，**不**每期 `release_except`；换手仅 ~20%，跨期命中省读盘/残差化。常驻由 `--rolling-pool-max-cached`（默认 160）封顶，压内存就调小到 ≈\|pool_t\| |
| **rolling-pool 因果性** | schedule 生成侧决策日 t 的池只用 **`index < t`** 的 IC（IC_t 需 t→t+h 收益，当日未实现）；消费侧 asof(`<=`) 因此安全，**不要**在消费侧再 shift |
| **neut 缓存键** | `factor_panel_neut_*`（急切 `build_factor_dataset` 与 rolling-pool lazy 共用 `research/rolling_pool/neut_cache.py`）键 = 名 + `neut_v6` + hold_period + rebalance_freq + 宇宙指纹 + 调仓日历指纹 + Barra/行业/WLS 指纹；**跨 horizon / 调仓频率绝不可共用**；清缓存删 `data/processed/factor_panels/factor_panel_neut_*.parquet`（Barra bundle：`barra_bundle_*/`） |
| **Barra 因子口径**（2026-07-29 重写；市值源 2026-07-30 切东财） | Size=`log(流通市值)`（缺则总市值，**不再**用 total_assets）；NonlinSize=Size² 正交 Size；Liquidity=63/252 日**换手率**等权平均（非成交量）；Leverage=单一 DTOA；Growth=营收 YoY 50%+净利润 YoY 50%（各腿先 1% winsor+z-score 再合成）；Beta/ResVol=对**中证全指 close** 的半衰期 63 日加权回归（β + 252 日 HSIGMA）；Momentum=RSTR 240/20 半衰期 60 日加权 |
| **Barra 截面回归 = WLS** | IC 纯化（`research/ic/barra.py`）与 `residualize_panel`（`--feature-neutralize`）统一用 **WLS，权重=√市值**（`utils/wls.py`，`factors/barra_risk.py::barra_regression_weights`）；无市值面板才退化等权 OLS 并 warning |
| **市值/换手数据源** | **Size 主路径**：`download_stock_value_em` → `{circ_mv,total_mv}`（东财日频，元）；顺带落 `pe_ttm`/`pb`（估值因子尚未切主路径）。**换手**：`download_shares`+`compute_market_cap` → `turnover_rate`；自算市值落 `*_computed` 仅校验/兜底。volume=**手**，换手=`(vol×100)/circ_shares`；`download_market_cap` deprecated；须显式传入 `get_barra_factors` |
| **rolling-pool fail-fast** | `--rolling-pool-strict`（默认 ON）：schedule 因子缺面板 / `store.get` 失败直接 raise，禁止整列 NaN→`fillna(0)` 静默当常数特征 |
| **rolling-pool 产物 tag** | tag 追加 `_rp{schedule缩写}-{hash6}`，避免与同参数固定池实验共用 `results/<tag>/` |

## 常用命令

```bash
.venv\Scripts\activate

# 冒烟 / 日常训练（默认已含 feature-neutralize + bid-ask）
python run.py --sample 100
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb

# IC v2（默认 FDR / t=2.5 / corr-dedup；GS 需 --gram-schmidt；h5/h20 同参）
python -m research.ic_analysis_v2 --period 5 --barra --save
python -m research.ic_analysis_v2 --period 20 --barra --save

# 编排 / 定池 / 过拟合 / 退市股
python logs/driver.py --preset main
python -m research.rolling_pool --horizon 5
python -m research.pbo
python -m data.download_delisted
python logs/analyze_results.py
```

最短命令与隐藏高级开关见 [docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md)；`python run.py --help-advanced` / `python -m research.ic_analysis_v2 --help-advanced`。

`run.py` 模式：`linear` / `ridge|lgbm|xgb|cat|rf|mlp` / `ensemble`（默认 lgbm+xgb）/ `dynamic` / `industry` / `ensemble + --blend-dynamic`。
`--horizon`：5=周频 / 10=双周 / 20=月频（默认）/ 60=季频。

## 模块速查

- **`models/`**：已合并 v2，`WalkForwardTrainer` 复用 `wf/` 子包（splits/labels（含 `residualize_panel`）/models/metrics（含 clustered FI）/ensemble/persistence/`clustered_importance`/`shap_analysis`）；`DynamicFactorTrainer` 按 ICIR 实时加权；`IndustryWalkForwardTrainer` 分申万二级训练；`analyzer.py` 接入 clustered FI + SHAP。
- **`backtest/`**：模块化 buy-and-hold 引擎，`quantile.py` 为唯一回测入口；`execution/portfolio/return_engine/turnover/benchmark/report` 子模块；`execution.py` 含 `bid_ask_spread_bps` 成本。
- **`factors/`**：`get_factor_registry()` 注册中心，覆盖量价/财务/Alpha/技术/Alpha101/涨跌停/regime/分数差分动量；新增因子须在此注册后才能被管线使用。财务因子经 `utils/pit_align.py` PIT 对齐；`barra_risk.py` 的 beta/res_vol/momentum 用 `clean_ret`，Barra 因子自身做行业去均值；`barra_regression_weights()` 产出 √市值 WLS 权重面板，`market_return()` 从 OHLCV 指数表取 close（勿再 `.squeeze()`）。
- **`research/`**：IC 分析 v2（`ic_analysis_v2.py` + `ic/` 子包，driver 默认）+ v1（`ic_analysis.py`，后备）；`ic/statistics.py` 含 Newey-West/BH-FDR/rolling ICIR；`ic/decay_corr.py` 含 ICIR/t/half-life 列；`ic/io.py` JSON 元数据补全；`ic/barra.py` PIT 改造；`pbo.py` 算 PBO+DSR（接入 `logs/analyze_results.py`）。
- **`strategies/ml.py`**：ML / ensemble / dynamic / industry 调度入口，按 `--mode` 分发；`--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化（复用 `models/wf/labels.py::residualize_panel`）。
- **`utils/`**：`rebalance_dates.py`（调仓日）、`fractional_diff.py`（AFML Ch.5 分数差分）、`pit_align.py`（财务 PIT 对齐）、`wls.py`（截面加权最小二乘，Barra 口径 √市值权重；被 IC 纯化 / quantile 分解 / `residualize_panel` 共用）。
- **`data/`**：`download.py`（含 AKShare 接口修复：重复列名/SZ symbol/退市接口名）、`download_delisted.py`（退市股 OHLCV）、`industry/download_industry.py`（PIT 时间序列 `industry_map_panel.parquet`）。
- **`tests/`**：`test_wf_splits.py`、`test_industry_pit.py`（行业 PIT 单测）。

## 当前状态

- **Barra 风格因子重写 + WLS**（2026-07-29）：Size 从 `log(total_assets)` 改为 `log(流通市值)`；Liquidity 从 log 成交量改为 63/252 日换手率等权平均；Growth 改营收+净利 YoY 各半；Beta/ResVol 改为对中证全指 **close** 的半衰期 63 日加权回归（顺带修复 `market_prices.squeeze()` 在 5 列 OHLCV 上不降维、导致 rolling cov 走 pairwise 的旧 bug）；Momentum 改 RSTR 半衰期加权；IC 纯化与 `--feature-neutralize` 残差化统一改 **WLS（权重=√市值）**。**旧 neut 缓存与 barra_pure checkpoint 必须清除重跑**（见注意事项 13）。
- v1 / v2 已合并为 **v2 唯一实现**（trainer + backtest），原单体文件已删除。
- **IC v2 已上线 driver**（2026-07-02）：`logs/driver.py` 默认调 `research.ic_analysis_v2`，Newey-West HAC t、可交易池 mask、BH-FDR、rolling ICIR、扣成本 IC 均进入生产筛选；v1 `ic_analysis.py` 保留为后备。
- **PIT 数据保护已落地**（2026-07-02）：财务因子按法定披露窗口 PIT 对齐（`utils/pit_align.py`）；股票池保留退市股（`data/download_delisted.py`）；ST 状态时间序列化；行业映射 PIT 时间序列（`industry_map_panel.parquet`）。详见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)。
- **AFML 方法论接入**（2026-07-02）：Fractional Differencing（`utils/fractional_diff.py` + 因子 `分数差分动量_20d`）、PBO + Deflated Sharpe Ratio（`research/pbo.py`，接入 `logs/analyze_results.py`，最优 DSR=0.0253 提示过拟合）、Clustered Feature Importance（`models/wf/clustered_importance.py`，接入 `metrics.py` + `analyzer.py`）。
- **ML 特征中性化**（2026-07-02）：`--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化，修复 IC/ML 口径不一致。
- **Barra 因子修正**（2026-07-02）：beta/res_vol/momentum 用 `clean_ret`，Barra 因子自身做行业去均值。
- **IC v2 增强**（2026-07-02）：BH-FDR 校正、rolling ICIR、IC 衰减表（ICIR/t/half-life 列）、JSON 元数据补全（universe_size/ic_series_length/sample_period/config_snapshot）、`IC_MIN_LISTING_DAYS=252`、t 阈值默认 2.5 + CLI。
- **交易成本**（2026-07-02）：bid-ask spread 成本（`BID_ASK_SPREAD_BPS=10.0`，`--bid-ask-spread`）。
- **AKShare 接口修复**（2026-07-02）：`_fetch_stock_list_with_metadata` 重复列名 + SZ symbol + 退市接口名。
- 中证全指已接入作市场代理与基准。
- Barra 残差化标签（`label_mode='barra_residual'`）已实现，控制 9 风格 + 行业哑变量。
- Walk-Forward 已实现 purged split + embargo + 两窗共用近期 val（默认 V=6 月）+ IC 加权多窗口集成。
- 实验产物统一落 `results/<tag>/`，命名 `tag = {mode}_h{horizon}[_w{windows}][_m{models}][_blend][_v2]`。

## 注意事项（最容易踩坑）

1. **`clean_ret` 必须用**：涨跌停日 return=NaN，否则动量/波动率因子系统性失真；Barra_Beta/Barra_ResVol 也已切换。
2. **PIT 披露日对齐**：财务因子必须经 `utils/pit_align.py` 按法定披露窗口延迟可用日期，禁用报告期直接 `ffill`（详见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)）。
3. **退市股 + ST 时间序列**：股票池必须保留退市股（`data/download_delisted.py`），否则幸存者偏差；ST 状态按日期查询，不是静态集合。
4. **未来函数防护**：`forward_return` 用 `close[t+N]/open[t+1]-1` 而非 `close.pct_change(N)`；行业映射用 PIT `industry_map_panel.parquet`。
5. **涨跌停 masks 透传**：`factor_limit.py` 与回测均需 masks，构建数据集时不要丢失。
6. **`qcut` duplicates**：分组回测 Q1-Q5 用 `pd.qcut`，遇重复分位须 `duplicates='drop'`，否则抛错。
7. **IC/ML 口径一致**：开 `--feature-neutralize` 让 ML 特征做 Barra+行业残差化，与 IC 纯 IC 同口径；否则 ML 学到系统性风险敞口。
8. **过拟合警惕**：跑 `python -m research.pbo` 看 DSR/PBO；当前最优 DSR=0.0253 提示超参选择偏差，勿盲信单实验最优。
9. **内存与并行**：32GB 机器建议 `TRAIN_MAX_WORKERS=1`、`IC_MAX_WORKERS=1`、`DYNAMIC_MAX_WORKERS=1`（与 ML 并发时）；勿叠加 `driver --parallel-ic` 与高 `--workers`。
10. **rolling-pool 勿用窗内并集**：`--rolling-pool-schedule` 下 WF 特征列 = 预测日 `pool_t`，不是 train∪val∪pred 活跃因子并集；历史训练日也取 `pool_t` 列的真实值（不按历史日入池再 mask）。
11. **改 IC 窗口口径要重生成 schedule**：`research/rolling_pool/stats.py` 已切到「严格早于决策日」，旧 schedule parquet 是含当日（前视）口径，必须 `python -m research.rolling_pool` 重跑；warm-up 也从 `dates[window-1:]` 改成 `dates[window:]`。
12. **换 horizon 要清旧 neut 缓存**：缓存键已 bump 到 `neut_v6`（含 horizon+freq+日历/Barra 指纹；市值源切东财），旧 `factor_panel_neut_*.parquet` 不会再被命中，可直接删除回收磁盘；冷启动首次残差化耗时较长属正常。命中日志搜 `neut cache HIT` / `Barra cache HIT`。
13. **Barra 定义已变（2026-07-29）+ 市值源切东财（2026-07-30）→ 旧产物作废**：Size/NonlinSize/Liquidity/Leverage/Growth/Beta/ResVol/Momentum 重写 + WLS；市值主源改 `stock_value_em`。必须：① 删 `data/processed/factor_panels/factor_panel_neut_*.parquet` 与 `barra_bundle_*/`；② 删 `research/output/_checkpoints/barra_pure_h*.pkl` 并重跑 `research.ic_analysis_v2 --barra`；③ 由此产出的 `selected_factors_h*_barra_pure.json` / `factor_configs_h*_pure_ic.yaml` 与历史实验结论**不可跨口径比较**。
14. **Barra 必须传市值/换手面板**：`get_barra_factors(..., circ_mv=, total_mv=, turnover_rate=, amount=)`。不传则 Size 降级 log(total_assets)、Liquidity 降级 log 成交量、回归退化等权 OLS（均有 warning）。新增调用点别忘了透传。
15. **`prices_raw` 断更不再拖垮 Size 主路径**：Size 读东财 `circ_mv`/`total_mv`；自算 `*_computed` 仍依赖 `prices_raw × shares`。东财不可用时不复权可走新浪 `stock_zh_a_daily` / `python -m data.backfill_prices_raw_sina`。上线前用 `report_raw_hfq_coverage()` 验收 raw。
16. **股本禁止按 code 永跳过**：`download_shares` 按 `refresh_stale_days`（默认 30）增量刷新；全量刷新时中间落盘须保留未完成旧行（否则会截断 parquet）；启动前自动 `.bak`。
17. **行业 PIT 严格默认**：`--barra` 时无 `industry_map_panel.parquet` 直接报错；仅 `--allow-static-industry` 可退化（结果含 PIT 泄漏，不可与严格口径比）。
18. **volume 单位=手**：`volume.parquet` 为 AKShare 原口径（1 手=100 股）；换手率 = `(volume×100)/circ_shares`（`compute_market_cap.VOLUME_MUL=100`）。amount 校验见 `clean.py::check_amount_unit`。
19. **北向已停更**：约 2024-08-19 后披露停更；默认不加载进 IC/`run.py`；显式因子会 warning 并把停更后置 NaN。
20. **市值主链**：`python -m data.download_stock_value_em` → `total_mv`/`circ_mv`（及 `pe_ttm`/`pb`）；换手仍 `download_shares`+`compute_market_cap`（写 `turnover_rate` + `*_computed`）。小样本对照：`python -m data.validate_market_cap`。`download_market_cap` deprecated。
21. **财务 PIT**：主接口无可靠 first-ann_date（yjbb「最新公告日期」=修订日）；默认法定窗近似（Q1/Q3=+30、半年报=+60、年报=+90）并打 WARNING。长表若自带 `ann_date` 则优先使用。
22. **沪市 ST**：深交所有带日期简称变更；沪/北无公开带日期接口 → `source=sh_bj_current_st_conservative_fallback`（自 list_date 起保守标 ST），勿当精确历史。
23. **涨跌停三层语义（research v2，默认）**：① **`clean_ret`**（因子侧）涨跌停日 return=NaN，因子面板缓存可复用；② **IC/ML 可交易池**（`build_ic_tradability_mask`）信号日**不**因 `any_limit` 剔除，仍剔 ST/停牌/零成交/次新/退市（`TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY=False`）；③ **标签 `forward_return`** 训练/IC **不做** execution mask（`FWD_RETURN_EXEC_MASK=False`），保留截面 winsor；④ **回测 execution** 仍拦截买日一字涨停/卖日涨跌停。**⑤ 回测得分宇宙默认 strict**（`--bt-score-universe strict`）：`get_cross_section` 以标签非空为预测门控，research 训练池会膨胀 score 覆盖并抬高等权基准/Q；落盘分数保留训练覆盖，回测前 `mask_scores_for_backtest` 裁回 **strict 可交易 ∩ label-exec-mask 可用**（与旧 `--tradable-strict --label-exec-mask` 预测门控同口径）。`--bt-score-universe train` 才与训练池一致。旧 strict 口径：`--tradable-strict` + `--label-exec-mask` 或 `--tradable-limit-mode strict`。IC checkpoint 后缀 `_tmr_v2`（换口径须 `--fresh`）；JSON 元数据 `tradable_mode` / `label_exec_mask`。稀疏因子 **`涨跌停状态`**（1=跌停/2=中性/3=涨停，冲突 limit_up 优先，截面 winsor+zscore）。
