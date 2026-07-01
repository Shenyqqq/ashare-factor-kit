# A股多因子量化选股框架

个人使用的 A 股多因子量化选股系统。完整流程：**数据下载 → 因子计算 → IC 分析筛选 → ML / 动态加权 → 分组回测 → 候选股输出**。

> **定位**：辅助人工选股，不做全自动交易。模型输出得分排名，人工二次筛选后手动操作。

---

## 核心特性

- **端到端流水线**：从 AKShare 数据下载到 Q1–Q5 分组回测 + Top30 候选股导出，单入口 `run.py` 串联。
- **多策略模式**：线性基准、单模型 Walk-Forward（ridge/lgbm/xgb/cat/rf/mlp）、多模型 ensemble、分行业训练、ICIR 动态加权。
- **A 股特有处理**：涨跌停 `clean_ret`、次日开盘执行、一字涨停剔除、申万二级行业中性化。
- **因子体系完整**：基础量价/财务 + Alpha 扩展 + 技术因子 + WorldQuant Alpha101 精选 + 涨跌停信号 + 市场/HMM regime 特征 + 分数差分动量（Fractional Differencing）。
- **Barra 纯 IC 筛选**：截面 OLS 控制 9 风格因子 + 行业哑变量，残差 IC 三步自动筛选；Barra 因子自身做行业去均值 + 用 `clean_ret`。
- **Walk-Forward 训练**：多训练窗口 × 多模型 IC 加权 Z-score ensemble，purged split + embargo（AFML Ch. 7），时间衰减样本权重，样本外 IC 监控，诊断 CSV；**Clustered Feature Importance**（AFML Ch.6）避免相关因子重要性分裂。
- **AFML 量化方法论**：Fractional Differencing（Ch.5，保留长期记忆同时平稳）、PBO + Deflated Sharpe Ratio（Ch.13/15，过拟合检验，`research/pbo.py`）、Clustered Feature Importance（Ch.6）。
- **PIT 数据保护**：财务因子按法定披露窗口（Q1/Q3=+45 天、半年报=+75 天、年报=+120 天）做 PIT 对齐（`utils/pit_align.py`）；股票池保留退市股（`data/download_delisted.py`）；ST 状态改为时间序列；行业映射 PIT 时间序列（`industry_map_panel.parquet`）。审计见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)。
- **IC 分析 v2（已上线 driver）**：Newey-West HAC t、可交易池 mask、IC clip/winsorize、Benjamini-Hochberg FDR 多重检验校正、rolling ICIR、IC 衰减表（含 ICIR/t/half-life）、扣成本 IC、JSON 元数据补全（universe_size / ic_series_length / sample_period / config_snapshot）。审阅见 [docs/IC_ANALYSIS_REVIEW.md](docs/IC_ANALYSIS_REVIEW.md)。
- **ML 特征中性化**：`--feature-neutralize` 在 `build_factor_dataset` 出口对因子做 Barra + 行业残差化，修复 IC 筛选与 ML 训练口径不一致。
- **交易成本模型**：佣金 + 印花税 + 滑点 + **bid-ask spread**（`BID_ASK_SPREAD_BPS=10.0`，`--bid-ask-spread`）。
- **可复现实验管理**：`results/<tag>/` 命名约定 + 自动化 `logs/driver.py` 批量编排（已切到 IC v2）。
- **模块化引擎**：`models/wf/`（splits/labels/models/metrics/ensemble/persistence/clustered_importance）、`backtest/`（execution/portfolio/return_engine/turnover/benchmark/report）、`research/ic/`（IC 分析 v2 子包，driver 默认调用）。

---

## 快速开始

### 环境准备

```bash
# Windows / PowerShell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

依赖（详见 `requirements.txt`）：AKShare / Tushare、pandas / numpy / pyarrow、scikit-learn / xgboost / lightgbm / catboost / shap、statsmodels / hmmlearn、matplotlib / seaborn / plotly、cvxpy、loguru / PyYAML 等。

环境变量（复制 `.env.example` 为 `.env`）：

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | 备用数据源 token |
| `DATA_ROOT` | 数据目录（默认 `./data`，可指向外部硬盘） |

### 一键运行

```bash
# 首次：下载数据 + 快速测试（前 100 只股票）
python run.py --sample 100

# 主力流程（数据已有）：ML ensemble + 因子白名单 + 月频
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb

# 周频版本
python run.py --skip-download --mode ensemble --horizon 5 \
  --factor-config config/factor_configs.yaml
```

完整流水线细节见 [docs/PIPELINE.md](docs/PIPELINE.md)。

---

## 项目结构

```
quant_trading/
├── run.py                       # 主入口（CLI、Step 1-4 串联）
├── AGENTS.md                    # AI agent 上下文指南（权威约定，勿轻易改动）
├── README.md                    # 本文件
├── requirements.txt
├── .env.example
│
├── config/
│   ├── settings.py              # 全局参数（回测区间、成本、股票池、并行度、FACTOR_WEIGHTS）
│   ├── factor_configs.yaml      # 因子白名单（按 h5 / h20 分节）
│   └── encoding_bootstrap.py
│
├── data/
│   ├── download.py              # 主数据下载（OHLCV、财务、股票池；含 AKShare 接口修复）
│   ├── download_delisted.py     # 退市股下载（保留历史 OHLCV，消除幸存者偏差）
│   ├── clean.py                 # 价格/涨跌停/财务清洗（clean_ret + masks）
│   ├── download_margin.py       # 融资余额
│   ├── download_northbound.py   # 北向持股
│   ├── download_institution.py  # 机构持仓
│   ├── download_moneyflow.py    # 大单资金流
│   ├── industry/download_industry.py  # 申万二级行业映射（重写为 PIT 时间序列 industry_map_panel.parquet）
│   ├── events/                  # 业绩预告等（实验性）
│   ├── raw/                     # 原始宽表 parquet（index=日期, columns=股票）
│   ├── universe/                # 股票池（含 list_date/delist_date/name_history）
│   └── processed/               # Walk-Forward 中间缓存（非实验主产物）
│
├── factors/
│   ├── factor.py                # 基础量价/财务 + 市场 regime + get_factor_registry()
│   ├── factor_alpha.py          # 行业动量、特质波动、融资/北向/机构/大单
│   ├── factor_technical.py      # BIAS / PSY / ARBR / 换手率变体 / 行业相对强度
│   ├── factor_alpha101.py       # WorldQuant Alpha101 精选（10 个）
│   ├── factor_limit.py          # 涨跌停信号（需 masks）
│   ├── factor_event.py          # 业绩预告（未接入 registry，实验性）
│   ├── factor_cache.py          # 因子缓存
│   └── barra_risk.py            # 9 个 Barra 风格因子（仅 ic_analysis --barra）
│
├── models/
│   ├── trainer.py               # WalkForwardTrainer（模块化 WF，复用 wf/ 子包）
│   ├── wf/                      # WF 子模块：splits / labels（含 residualize_panel）/ models / metrics（含 clustered FI）/ ensemble / persistence / clustered_importance
│   ├── dynamic_trainer.py       # ICIR 动态加权（无 ML）
│   ├── industry_trainer.py      # 分行业 Walk-Forward
│   └── analyzer.py              # MLAnalyzer（--report 触发 IC / SHAP / clustered FI 可视化）
│
├── strategies/
│   ├── linear.py                # 线性加权基准
│   ├── ml.py                    # ML / ensemble / dynamic / industry 调度入口
│   └── market_state.py          # 市场状态独立工具
│
├── backtest/
│   ├── quantile.py              # Q1–Q5 分组回测（模块化 buy-and-hold 引擎）
│   ├── execution.py / portfolio.py / return_engine.py / turnover.py / benchmark.py / report.py
│   └── (engine.py / quantile_v2.py 已合并/移除)
│
├── research/
│   ├── ic_analysis.py           # IC 分析 + 因子筛选（v1，与 v2 并存，driver 已切 v2）
│   ├── ic_analysis_v2.py        # v2 入口（python -m research.ic_analysis_v2）— driver 默认调用
│   ├── ic/                      # v2 模块化 IC：statistics（Newey-West/BH-FDR/rolling ICIR）/ barra（PIT）/ forward_return / selection（use_fdr）/ universe / decay_corr（ICIR/t/half-life）/ cost / io（JSON 元数据补全）
│   ├── pbo.py                   # PBO + Deflated Sharpe Ratio（AFML Ch.13/15，过拟合检验）
│   ├── market_regime.py         # 市场 regime / HMM 状态
│   ├── output/                  # IC 汇总、筛选 JSON、regime CSV
│   └── notebooks/
│
├── utils/
│   ├── rebalance_dates.py       # 共享调仓日生成（resample(freq).last()，h10 修复）
│   ├── fractional_diff.py       # 分数差分（AFML Ch.5，d=0.4 默认）
│   └── pit_align.py             # PIT 披露日对齐（财务因子，法定披露窗口）
│
├── tests/
│   ├── test_wf_splits.py
│   └── test_industry_pit.py     # 行业 PIT 时间序列单测
│
├── logs/                        # 自动化脚本与实验日志（被 .gitignore）
│   ├── driver.py                # 主编排：IC → YAML → 批量实验
│   └── analyze_results.py       # 结果汇总
│
└── results/                     # 实验产物（按 tag / batch 分目录）
    └── README.md                # 实验产物命名约定说明
```

> `logs/` 与 `data/raw/` `data/processed/` `results/` 均被 `.gitignore` 忽略，仅本地存在。

---

## 数据流

```
AKShare
  ↓
data/download.py → data/raw/*.parquet（OHLCV、财务、股票池）
data/clean.py → clean_ret + 涨跌停 masks（内存，不落盘）

research/ic_analysis.py → research/output/selected_factors_h{5,20}.json
logs/driver.py → config/factor_configs.yaml（JSON 自动合并）

get_factor_registry() → strategies/ml.py → WalkForwardTrainer
  → models/dynamic_trainer.py（mode=dynamic）
  → models/industry_trainer.py（mode=industry）
  ↓
data/processed/ 仅保留中间缓存；实验产物在 results/<tag>/：
  factor_scores_<tag>.parquet, backtest_<tag>.png, holdings_top30_<tag>.csv
```

---

## 主要工作流

### 一键（推荐）

```bash
# IC → YAML → ensemble lgbm+xgb + blend-dynamic，h5 + h20
python logs/driver.py --preset main

# 跑完汇总
python logs/analyze_results.py
```

| 预设 | 内容 |
|------|------|
| `main` | ensemble lgbm+xgb + blend-dynamic，h5 + h20（当前最佳实践） |
| `ablation` | ridge/lgbm/xgb/cat/ensemble × h5/h20 消融 |
| `dynamic` | dynamic + 短窗口 ensemble w6-12 |
| `ic-full` | 仅 IC：h5/h10/h20，全样本 + 近 3 年 |

### 分步

```bash
# 1. IC 分析 + 因子筛选（Barra 纯 IC）— v2 已为 driver 默认，也可显式调用
python -m research.ic_analysis --period 5  --barra --save
python -m research.ic_analysis --period 20 --barra --save
python -m research.ic_analysis_v2 --period 20 --barra --save --use-fdr --t-threshold 2.5

# 2. JSON → YAML 同步
python logs/driver.py --ic-only --horizons 5,20

# 3. ML ensemble 训练 + 回测（含特征中性化 + bid-ask spread 成本）
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb \
  --feature-neutralize --bid-ask-spread 10

# 4. 过拟合检验（PBO + Deflated Sharpe Ratio）
python -m research.pbo
python logs/analyze_results.py   # 已接入 DSR/PBO 汇总

# 5. 退市股下载（消除幸存者偏差）
python -m data.download_delisted
```

> `ic_analysis --save` 只写 `research/output/selected_factors_h*.json`；`run.py --factor-config` 读 YAML。两者靠 `logs/driver.py` 的 `sync_factor_yaml()` 桥接，或手动编辑 YAML。

---

## run.py 模式速查

| `--mode` | 实现 | 说明 |
|----------|------|------|
| `linear` | `strategies/linear.py` | 线性加权基准，无需训练 |
| `ridge` / `lgbm` / `xgb` / `cat` / `rf` / `mlp` | `WalkForwardTrainer` | 单模型 Walk-Forward |
| `ensemble` | `WalkForwardTrainer`（多模型） | **主力策略**，默认 lgbm+xgb |
| `dynamic` | `DynamicFactorTrainer` | ICIR 动态加权，无 ML |
| `industry` | `IndustryWalkForwardTrainer` | 分申万二级行业独立训练 |
| ensemble + `--blend-dynamic` | WF + Dynamic rank 混合 | 当前 best practice |

| `--horizon` | 调仓频率 | 备注 |
|---|---|---|
| 3 | 3 日 | 研究用，T+1 下不建议实盘 |
| 5 | 周频（W-FRI） | 短线 |
| 10 | 双周（2W-FRI） | 中间频率 |
| 20 | 月频（默认） | 最稳定 |
| 60 | 季频 | 长周期 |

常用参数：`--skip-download` `--sample N` `--factor-config PATH` `--train-windows 6,12` `--models lgbm,xgb` `--blend-dynamic` `--holdings` `--report` `--output-dir results/<tag>/` `--wf-selection ic_weighted` `--label-mode cs_zscore` `--ensemble-method zscore` `--feature-neutralize`（ML 特征 Barra+行业残差化，与 IC 同口径）`--bid-ask-spread BPS`（bid-ask spread 成本，默认 0；`BID_ASK_SPREAD_BPS=10.0`）。

完整参数与命令示例见 [AGENTS.md](AGENTS.md) 与 [docs/PIPELINE.md](docs/PIPELINE.md)。

---

## 关键设计决策摘要

| 主题 | 约定 |
|------|------|
| **因子方向** | 因子函数内部已取反，输出「越高越好」。`FACTOR_WEIGHTS` 仅 linear 模式使用 |
| **标准化** | 截面 winsorize(1%) → cross_sectional_zscore(clip=3σ)；市场/HMM 特征用时序滚动 z-score |
| **clean_ret** | 量价因子必须用 `clean_ret`（涨跌停日 return=NaN），不用 `prices.pct_change()` |
| **rebalance_dates** | `groupby(period).last()` 取周期末实际交易日，不用日历 `ME`/`W-FRI` 虚拟日期（实现在 `utils/rebalance_dates.py`） |
| **forward_return** | `close[t+N] / open[t+1] - 1`（有 open_ 时），信号日收盘后次日开盘买入 |
| **Walk-Forward 起点** | 配置为月数（`TRAIN_WINDOWS_MONTHS=[6,12]`、`VAL_WINDOW_MONTHS=6`），构造时按调仓频率转为期数；h20 默认 min_history≈18 期，h5 周频≈78 期 |
| **ensemble 默认模型** | `MODEL_TYPES = ["lgbm", "xgb"]`（非四模型）；`TIME_DECAY = 0.015` |
| **因子白名单** | YAML 结构 `{ h5: {factors: [...]}, h20: {factors: [...]} }`；过滤时自动保留 `市场*` / `HMM_*` regime 特征 |
| **Barra 纯 IC** | 截面 OLS 控制 Barra 9 风格 + 行业哑变量，残差 IC；Barra 因子自身做行业去均值 + 用 `clean_ret`；剔除纯 IC<0.02 且 ICIR<0.3、\|t\|<2.5、corr>0.7 去重；可选 BH-FDR 校正 |
| **Q1–Q5** | `pd.qcut` 升序，Q1=最低分，Q5=最高分；IC>0 为正向 |
| **成本** | `COMMISSION_RATE=0.0001`、`STAMP_DUTY=0.0005`、`SLIPPAGE_RATE=0.0`、`BID_ASK_SPREAD_BPS=10.0`（可选，`--bid-ask-spread`）；分组回测内 `cost_bps≈3` |
| **数据对齐** | 财务季报按**法定披露窗口**做 PIT 对齐（Q1/Q3=+45 天、半年报=+75 天、年报=+120 天，`utils/pit_align.py`），再 `reindex(..., method="ffill")` 前向填充到日频 |
| **股票池** | 保留退市股（`data/download_delisted.py`），ST 状态时间序列化（按日期查询，非静态集合） |
| **行业 PIT** | `industry_map_panel.parquet`（date × code → sw_l2）按截面日期取当期行业，消除未来信息 |
| **IC v2 上线** | `logs/driver.py` 已切到 `research.ic_analysis_v2`；Newey-West HAC t、可交易池 mask、BH-FDR、rolling ICIR、扣成本 IC 均进入生产筛选 |
| **ML 特征中性化** | `--feature-neutralize` 在 `build_factor_dataset` 出口做 Barra+行业残差化，与 IC 纯 IC 同口径 |

> 详细约定与陷阱见 [AGENTS.md](AGENTS.md)「关键设计决策」与 [docs/PIPELINE.md](docs/PIPELINE.md)「常见陷阱与近期修复」。

---

## 配置说明

### `config/settings.py` 主要参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `BACKTEST_START` / `END` | 2018-01-01 / 今日 | 回测区间 |
| `N_STOCKS` | 30 | quantile 回测 Top30 持仓 |
| `MIN_MARKET_CAP` | 20e8 | 股票池过滤（最小市值） |
| `FACTOR_WEIGHTS` | 见文件 | linear 模式专用权重 |
| `IC_MAX_WORKERS` | 1 | IC 分析因子并行度（默认串行） |
| `BARRA_IC_WORKERS` | 1 | Barra 纯 IC 并行度 |
| `DYNAMIC_MAX_WORKERS` | 4 | Dynamic 并行度（与 ML 并发建议设 1） |
| `IC_MIN_LISTING_DAYS` | 252 | IC 分析最小上市天数过滤（剔除次新股噪声） |
| `BID_ASK_SPREAD_BPS` | 10.0 | bid-ask spread 成本（基点），`--bid-ask-spread` 启用 |

### ML 超参（`models/trainer.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `TRAIN_WINDOWS_MONTHS` | `[6, 12]` | 训练窗口（日历月，构造时转为调仓期数） |
| `VAL_WINDOW_MONTHS` | `6` | 验证窗口 |
| `TIME_DECAY` | `0.015` | 训练样本指数衰减权重 |
| `MODEL_TYPES` | `["lgbm", "xgb"]` | ensemble 子模型 |

并行/内存（32GB 机器）：`TRAIN_MAX_WORKERS=1`、`TRAIN_N_JOBS=4`，CLI 可 `--workers 2` 略加速；勿叠加 `driver_ic_parallel.sh` 多进程 + 高 `--workers`。

### `config/factor_configs.yaml` 结构

```yaml
h5:
  rebalance_freq: W-FRI
  factors: [动量_20d, 反转_5d, ...]   # 白名单
  excluded: [...]                      # driver.py 同步时写入的剔除说明
h20:
  rebalance_freq: ME
  factors: [...]
  excluded: [...]
```

---

## 实验产物

所有模型训练 / 回测输出落在 `results/<tag>/`，不落仓库根或 `data/processed/`。命名约定：

```
tag = {mode}_h{horizon}[_w{train_windows}][_m{models}][_blend][_v2]
```

例：`ensemble_h20_w6-12_mlgbm-xgb_blend`

| 文件 | 说明 |
|------|------|
| `factor_scores_<tag>.parquet` | 最终预测得分（调仓日 × 股票；linear 为日频） |
| `backtest_<tag>.png` | Q1–Q5 + Top30 + 基准 + 指数 四宫格图 |
| `backtest_<tag>_nav.csv` / `_annual.csv` / `_longshort.csv` | 净值 / 逐年收益 / 多空净值 |
| `holdings_top30_<tag>.csv` | 每期 Top30 候选股（`--holdings`） |
| `model_metrics_<tag>.json` | IC 均值、ICIR、胜率、预测期数（ML/industry） |
| `ic_series_<tag>.csv` | 逐期样本外 IC（ML/industry） |

历史批次：`results/v1_ablation/`、`v2_dynamic_short/`、`v4_normal_window/`、`v5_window612_h5/`、`dynamic_compare/`、`ridge_window_sweep/` 等。详见 [results/README.md](results/README.md)。

IC 分析输出在 `research/output/`（`ic_summary_h*.csv`、`ic_yearly_h*.csv`、`ic_barra_pure_h*.csv`、`selected_factors_h*.json`）。

---

## 常见问题 / 注意事项

- **量价因子必须用 `clean_ret`**：涨跌停日收益率置 NaN，否则波动率/动量系统性偏低。Barra_Beta/Barra_ResVol 也已切换到 `clean_ret`。
- **财务因子必须 PIT 对齐**：`utils/pit_align.py` 按法定披露窗口延迟可用日期，禁用报告期直接 `ffill`（详见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)）。
- **股票池必须保留退市股**：`data/download_delisted.py` 下载历史退市股 OHLCV，否则回测存在幸存者偏差；ST 状态按时间序列查询，不是静态集合。
- **IC v2 已上线 driver**：`logs/driver.py` 调 `research.ic_analysis_v2`，v1 `ic_analysis.py` 保留但不再被生产路径调用。Newey-West t / 可交易池 / BH-FDR / rolling ICIR 均已生效。
- **IC/ML 口径一致性**：开 `--feature-neutralize` 让 ML 特征做 Barra+行业残差化，与 IC 纯 IC 同口径；否则 ML 会学到系统性风险敞口而非真 alpha。
- **h5 与 h20 是两套独立策略**：白名单、有效性、调仓频率均不同，分开评估。
- **周频换手约 4 倍于月频**：确认扣费后净收益仍优再考虑实盘。
- **Q5 是候选池**，需结合基本面人工二次筛选，不要直接照搬。
- **新参数至少 3 个月模拟盘验证**后再考虑实盘。
- **过拟合检验**：跑 `python -m research.pbo` 看 DSR/PBO；当前 51 个实验中最优 dynamic_h20_lb12 的 DSR=0.0253，提示过拟合风险，需警惕超参选择偏差。
- **内存与并行**：32GB 机器建议 `IC_MAX_WORKERS=1`、`TRAIN_MAX_WORKERS=1`、`DYNAMIC_MAX_WORKERS=1`（与 ML 并发时）；勿叠加 `driver --parallel-ic` 与高 `--workers`。
- **IC 分析 v1/v2 双轨**：v1 `ic_analysis.py` 保留为后备，driver 已默认 v2；可按需显式 `python -m research.ic_analysis` 调 v1。
- **Walk-Forward 与回测已统一为模块化引擎**：`models/trainer.py` + `models/wf/`、`backtest/quantile.py` + `backtest/` 子模块。原 v1 单体实现（`models/trainer_v2.py`、`backtest/quantile_v2.py`）已合并删除；`--trainer-engine` / `--backtest-engine` CLI 参数保留为 deprecated no-op 以兼容旧脚本。

### 文档与代码一致性

- `strategies/ml.py` docstring 仍写默认四模型，以 `MODEL_TYPES=["lgbm","xgb"]` 为准。
- `industry_trainer.py` docstring 写默认 `[12,24,36]`，实际继承 `[6,12]`。
- `factor_event.py` 未接入 `get_factor_registry()`，仅实验性。
- `backtest/engine.py` 与 `backtest/quantile_v2.py` 已删除/合并；唯一回测路径为 `backtest/quantile.py`（模块化引擎）。
- Walk-Forward 与回测已统一为模块化引擎（`models/wf/`、`backtest/` 子模块），原 v1 单体实现于 2026-07-01 合并删除；IC v2 于 2026-07-02 接入 driver 为默认，v1 保留为后备。

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [AGENTS.md](AGENTS.md) | AI agent 上下文指南与项目权威约定（命令、架构、设计决策、注意事项） |
| [docs/PIPELINE.md](docs/PIPELINE.md) | 端到端流水线详解（数据 → IC → 训练 → 回测 → 输出，含 mermaid 图） |
| [docs/TODO.md](docs/TODO.md) | 待办与近期修复清单（含 P0/P1 完成情况） |
| [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md) | PIT 财务数据 + 幸存者偏差审计报告（2026-07-01） |
| [docs/IC_ANALYSIS_REVIEW.md](docs/IC_ANALYSIS_REVIEW.md) | IC 分析模块深度审阅报告（v1 vs v2，P0/P1/P2 清单） |
| [results/README.md](results/README.md) | 实验产物目录结构与命名约定 |
