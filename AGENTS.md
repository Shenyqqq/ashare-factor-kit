# AGENTS.md

Agent 在本仓库工作时的精简上下文指南。与 [README.md](README.md) 互补（README 是用户文档，本文是 agent 上手速查）。

## 项目简介

A 股多因子量化选股框架，定位 **辅助人工选股，不做全自动交易**。
端到端流水线：数据下载 → 因子计算 → IC 筛选 → ML / 动态加权 → 分组回测 → 候选股输出。
模型输出得分排名，人工二次筛选后手动操作。

## 架构概览

数据流：`AKShare → data/raw/*.parquet（含退市股）→ clean_ret + masks（内存）→ PIT 对齐（财务/行业）→ IC v2 筛选（Newey-West/BH-FDR/可交易池）→ factor_configs.yaml → WalkForwardTrainer / DynamicFactorTrainer → factor_scores → backtest/quantile.py → results/<tag>/`。

| 顶层目录 | 用途 |
|----------|------|
| `run.py` | 主入口 CLI，串联 Step 1-4；含 `--feature-neutralize` `--bid-ask-spread` |
| `config/` | `settings.py`（全局参数，含 `IC_MIN_LISTING_DAYS=252`、`BID_ASK_SPREAD_BPS=10.0`）、`factor_configs.yaml`（因子白名单） |
| `data/` | 下载 / 清洗 / 行业 / 财务事件；`raw/` `universe/` `processed/`；`download_delisted.py` 保留退市股；`industry/download_industry.py` 产出 PIT `industry_map_panel.parquet` |
| `factors/` | 因子实现 + `get_factor_registry()` 注册中心；含分数差分动量；财务因子经 PIT 对齐 |
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
| **Walk-Forward split** | purged training + embargo（AFML Ch.7，防 forward-return 标签泄漏）；`hold_period_to_embargo_periods()` 自动换算 |
| **集成方式** | 多训练窗口 × 多模型 → IC 加权 Z-score 平均（非盲 rank average） |
| **Clustered FI** | `models/wf/clustered_importance.py`（AFML Ch.6）按相关因子聚类算重要性，避免相关因子重要性分裂 |
| **市场代理** | 中证全指（`data/raw/csi_all.parquet`）作市场 regime / 基准 |
| **Q1-Q5** | `pd.qcut` 升序，Q1=最低分，Q5=最高分；IC>0 为正向 |
| **成本** | `COMMISSION_RATE=0.0001`、`STAMP_DUTY=0.0005`、`SLIPPAGE_RATE=0.0`、`BID_ASK_SPREAD_BPS=10.0`（`--bid-ask-spread`） |
| **PIT 披露日对齐** | 财务因子按法定披露窗口（Q1/Q3=+45 天、半年报=+75 天、年报=+120 天）延迟可用日期，`utils/pit_align.py`；禁用报告期直接 `ffill` |
| **退市股 + ST 时间序列** | 股票池保留退市股（`data/download_delisted.py`）；ST 状态按日期查询（时间序列），非静态集合 |
| **行业 PIT** | `industry_map_panel.parquet`（date × code → sw_l2）按截面日期取当期行业，消除未来信息 |
| **IC v2 上线** | `logs/driver.py` 已切到 `research.ic_analysis_v2`；Newey-West HAC t、可交易池 mask、BH-FDR、rolling ICIR、扣成本 IC 均进入生产筛选 |
| **ML 特征中性化** | `--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化，与 IC 纯 IC 同口径 |
| **FDR 多重检验校正** | `research/ic/statistics.py::benjamini_hochberg` + `selection.py::use_fdr`，应对 30+ 因子多重 t 检验假阳性 |
| **过拟合检验** | `research/pbo.py` 算 PBO + Deflated Sharpe Ratio（AFML Ch.13/15）；当前 51 实验最优 DSR=0.0253，提示过拟合 |
| **分数差分** | `utils/fractional_diff.py`（AFML Ch.5，d=0.4 默认）→ 因子 `分数差分动量_20d`，保留长期记忆同时平稳 |
| **数据对齐** | 财务季报先 PIT 对齐再 `reindex(..., method="ffill")` 前向填充到日频 |
| **v2 为唯一实现** | 原 v1 单体（`trainer_v2.py`、`quantile_v2.py`、`engine.py`）已合并删除；`--trainer-engine` / `--backtest-engine` CLI 保留为 deprecated no-op |

## 常用命令

```bash
# 环境
.venv\Scripts\activate
pip install -r requirements.txt

# 快速测试（前 100 只股票）
python run.py --sample 100

# 主力流程（月频 ensemble，含特征中性化 + bid-ask spread）
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb \
  --feature-neutralize --bid-ask-spread 10

# 周频
python run.py --skip-download --mode ensemble --horizon 5 \
  --factor-config config/factor_configs.yaml

# IC 分析 + 因子筛选（Barra 残差）— v2 为 driver 默认
python -m research.ic_analysis_v2 --period 20 --barra --save --use-fdr --t-threshold 2.5
python -m research.ic_analysis --period 20 --barra --save   # v1 后备

# 过拟合检验（PBO + Deflated Sharpe Ratio）
python -m research.pbo

# 退市股下载（消除幸存者偏差）
python -m data.download_delisted

# 一键编排：IC v2 → YAML → ensemble + blend-dynamic，h5 + h20
python logs/driver.py --preset main
python logs/analyze_results.py
```

`run.py` 模式：`linear` / `ridge|lgbm|xgb|cat|rf|mlp` / `ensemble`（主力，默认 lgbm+xgb）/ `dynamic`（ICIR 加权，无 ML）/ `industry`（分行业 WF）/ `ensemble + --blend-dynamic`（best practice）。
`--horizon`：5=周频 / 10=双周 / 20=月频（默认）/ 60=季频。
`--objective regression|rank`：rank 启用 Learning-to-Rank（LGBMRanker/XGBRanker/CatBoostRanker），自动配 `cs_rank` 标签；ridge/rf/mlp 不支持 rank 时自动回退 regression。
cat（CatBoost）：可选模型，ordered boosting + 对称树抗过拟合，`--models lgbm,xgb,cat` 或 `--mode cat` 启用；Optuna 调参已覆盖（`--tune`）。

## 模块速查

- **`models/`**：已合并 v2，`WalkForwardTrainer` 复用 `wf/` 子包（splits/labels（含 `residualize_panel`）/models/metrics（含 clustered FI）/ensemble/persistence/`clustered_importance`）；`DynamicFactorTrainer` 按 ICIR 实时加权；`IndustryWalkForwardTrainer` 分申万二级训练；`analyzer.py` 接入 clustered FI。
- **`backtest/`**：模块化 buy-and-hold 引擎，`quantile.py` 为唯一回测入口；`execution/portfolio/return_engine/turnover/benchmark/report` 子模块；`execution.py` 含 `bid_ask_spread_bps` 成本。
- **`factors/`**：`get_factor_registry()` 注册中心，覆盖量价/财务/Alpha/技术/Alpha101/涨跌停/regime/分数差分动量；新增因子须在此注册后才能被管线使用。财务因子经 `utils/pit_align.py` PIT 对齐；`barra_risk.py` 的 beta/res_vol/momentum 用 `clean_ret`，Barra 因子自身做行业去均值。
- **`research/`**：IC 分析 v2（`ic_analysis_v2.py` + `ic/` 子包，driver 默认）+ v1（`ic_analysis.py`，后备）；`ic/statistics.py` 含 Newey-West/BH-FDR/rolling ICIR；`ic/decay_corr.py` 含 ICIR/t/half-life 列；`ic/io.py` JSON 元数据补全；`ic/barra.py` PIT 改造；`pbo.py` 算 PBO+DSR（接入 `logs/analyze_results.py`）。
- **`strategies/ml.py`**：ML / ensemble / dynamic / industry 调度入口，按 `--mode` 分发；`--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化（复用 `models/wf/labels.py::residualize_panel`）。
- **`utils/`**：`rebalance_dates.py`（调仓日）、`fractional_diff.py`（AFML Ch.5 分数差分）、`pit_align.py`（财务 PIT 对齐）。
- **`data/`**：`download.py`（含 AKShare 接口修复：重复列名/SZ symbol/退市接口名）、`download_delisted.py`（退市股 OHLCV）、`industry/download_industry.py`（PIT 时间序列 `industry_map_panel.parquet`）。
- **`tests/`**：`test_wf_splits.py`、`test_industry_pit.py`（行业 PIT 单测）。

## 当前状态

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
- Walk-Forward 已实现 purged split + embargo + window-specific validation + IC 加权多窗口集成。
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
