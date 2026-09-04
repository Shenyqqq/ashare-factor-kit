# A 股多因子量化选股框架

面向研究与辅助选股的 A 股多因子流水线：**数据下载 → 因子计算 → IC 筛选 → ML / 动态加权 → 分组回测 → 按 fold 加载模型出候选股（live）**。

## 这是什么 / 不是什么

| 是 | 不是 |
|----|------|
| 可复现的研究框架与实验脚手架 | 全自动交易系统 / 券商下单接口 |
| 输出得分排名与 Top-N 候选池，供人工二次筛选 | 投资建议或收益承诺 |
| 强调 PIT、可交易口径、成本与过拟合检验 | 「一键复现完整历史曲线」的数据快照仓库 |

数据需自行通过 AKShare（等）下载；不同机器、不同下载时点与接口变更会导致结果不可逐点复现。过拟合风险真实存在，请用样本外与 PBO/DSR（概率过拟合 / 紧缩夏普，检验「挑出来的最优是否太好看」）等工具自检，勿盲信单次最优实验。

---

## 流水线说明

下面按真实执行顺序说明每一步在干什么（不是术语堆砌）。更细的陷阱与内部约定见 [docs/PIPELINE.md](docs/PIPELINE.md)、[AGENTS.md](AGENTS.md)。

### 1. 数据怎么下载

主入口是 `data/download.py`（AKShare）。日常拉两类行情：后复权 OHLCV（`*_hfq`，训练和量价因子用）和不复权 `prices_raw`（换手、市值校验用）。市值走东财日频 `python -m data.download_stock_value_em`，写入流通市值 `circ_mv` 和总市值 `total_mv`（顺带 `pe_ttm` / `pb`）；**Size 风格因子读的就是东财 `circ_mv`**，不是总资产。股本 `python -m data.download_shares` 再配合 `python -m data.compute_market_cap`，用「成交量（手）×100 / 流通股本」算换手率。

股票池刻意保留退市股（`python -m data.download_delisted`），避免只拿还活着的公司做研究。ST 用 `python -m data.download_st_history`：深交所有带日期的精确历史；沪市 / 北交所没有公开带日期接口，只能从上市日起保守标 ST。行业是时间序列面板（`python -m data.industry.download_industry` → `industry_map_panel.parquet`），不是一张永远不变的对照表。财务季报经 `utils/pit_align.py` 按法定披露窗口对齐后再用（季报约 +30 日、半年报 +60、年报 +90；表里若有公告日则优先），禁止按报告期直接往未来填。

日更不要全量扫股本；市值增量不要把「最近 N 天」当成 `--start` 去覆盖历史（会全量重拉或截断前向填充）。逐步命令与覆盖率检查见 [docs/LIVE_OPS_README.md](docs/LIVE_OPS_README.md)。研究全流程也可直接 `python run.py`（会顺带下载）；已有数据时加 `--skip-download`。

### 2. 因子计算

因子在 `factors/` 里实现，经 `get_factor_registry()` 注册后才能进管线。量价与部分风险因子必须用 `clean_ret`：涨跌停日的收益记为缺失，不当成普通涨跌。因子函数内部已经按「越高越好」取过方向，后面筛选和模型不再二次翻符号。

### 3. IC 凭什么筛选

生产筛选走 IC v2（`python -m research.ic_analysis_v2 --period 5 --barra --save`；月频把 `--period` 改成 20）。对每个因子算截面 Rank IC（因子排序与未来收益排序的相关）和 ICIR（IC 均值除以 IC 波动）。再叠加：Newey-West HAC t（考虑序列相关后的显著性）、BH-FDR（同时测很多因子时控制假发现率）、截面相关去重、可交易池掩码、滚动 ICIR、扣交易成本后的 IC。稠密因子还有 **long_share** 门（分组回测里多头贡献占比，默认须大于 0.4）；稀疏因子（事件类、经常大面积缺失的）走另一条轨，不和稠密因子抢同一套阈值。

约定：Q1 是最低分、Q5 是最高分；IC 为正表示高分那一组更好。通过的因子写入 `research/output/selected_factors_h*.json`，再落到 `config/factor_configs*.yaml` 白名单；之后训练只吃这份名单，而不是全市场随便堆特征。

### 4. ML / 加权是怎么加权的

`python run.py --mode ...` 不是同一种「动态加权」，要分开看：

| 模式 | 实际在加权什么 |
|------|----------------|
| **linear** | 配置里的 `FACTOR_WEIGHTS`，手工/静态权重。 |
| **单模型 ML**（`ridge` / `lgbm` / `xgb` / `cat` / `rf` / `mlp`） | Walk-Forward（按时间向前滚）训练该模型，用模型得分给股票排序。 |
| **ensemble** | 多个训练窗口 × 多个模型 → 按 IC 加权后做 Z-score 平均（**不是**把排名简单平均）。 |
| **dynamic** | 不训树模型，按滚动 ICIR 给因子实时加权。 |
| **industry** | 按申万二级行业分别训练。 |

当前旗舰是 **xgb + Size/行业中性化 + 训练窗 156 期 + 每 4 期重训一次**（`retrain_every=4`），不是 ensemble 盲平均。`--feature-neutralize` 会在特征出口做风格/行业的加权最小二乘残差化（WLS，权重大约是√市值）：可以是完整 Barra 风格，旗舰用的是 Size + 申万二级，与「纯 IC」同一套中性化口径。

### 5. 分组回测怎么回测

入口是 `backtest/quantile.py`（由 `run.py` 训练结束后调用）。信号在收盘后产生，**下一交易日开盘买入**，前瞻收益是 `收盘[t+N] / 开盘[t+1] - 1`。默认周五（W-FRI）调仓，持有到下一期信号。分组用 `pd.qcut` 切 Q1–Q5，另出 Top-N。成本默认：佣金 1bp、印花税 5bp、滑点 0、买卖价差 10bp。

回测成交仍拦截：买日一字涨停买不进、卖日涨跌停卖不出。回测用的得分宇宙默认 **strict**——只保留「严格可交易且标签可用」的股票，避免训练池更大、把等权基准抬高。研究和执行是分开的：默认信号日股票池**保留**涨跌停（当天不因涨跌停剔除），训练标签也不做成交掩码；不要把「研究能打分」理解成「回测一定能成交」。

### 6. 候选股输出（含 live）

模型只给得分排名和 Top-N 名单，**不做自动下单**。研究回测产物在 `results/<tag>/`。实盘是：先全量 Walk-Forward 加 `--save-models` 把每一折的模型存下来，再用 `python -m live.predict_from_wf_models` 按折加载对应模型，对最新信号日出候选股，供人工看 Top100 / Top30 后手动操作。详见下文「实盘 / Live」。

---

## 特色与特别实现

和「下载行情 → 算几个因子 → 回测」的传统脚本相比，本仓库把 **PIT（当时可观测信息）**、可交易性、多重检验、风格中性化、Walk-Forward 泄漏控制、以及研究/执行分口径做成了默认，而不是事后补丁。

### 传统做法 vs 本仓库

| 传统 A 股多因子里常见的坑 | 本仓库的默认做法 |
|---------------------------|------------------|
| 财务按报告期直接往未来填；行业用静态一张表；训练池丢掉退市 / ST → 幸存者偏差 | 法定披露窗口 PIT；行业时间序列；保留退市股；ST 按日查询 |
| `pct_change` 当动量，涨跌停日当正常收益 | `clean_ret`：涨跌停日收益为缺失 |
| `close.pct_change(N)` 当「未来收益」 | 次日开盘买：`close[t+N] / open[t+1] - 1` |
| 等权 OLS 中性化，用总资产当 Size | √市值 WLS；Size = log(流通市值)，源为东财 `circ_mv` |
| IC 只看均值，同时测很多因子也不校正 | HAC t + BH-FDR + long_share + 相关去重 |
| 随机切分样本，或训练/测试标签时间重叠 | purged Walk-Forward + embargo（重叠样本剔除，边界再留禁运带） |
| 回测股票池 = 训练池，等权基准被膨胀 | `--bt-score-universe strict` |
| 宇宙混进 B 股 | 剔除 B 股（代码 200 / 900）；北交所 92 开头保留，但行情可能断更 |

### 做成默认的工程点

- **PIT 与可交易性**：财务、行业、退市、ST 按上面表格处理，而不是「下载完就算历史已知」。详见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)。沪市 ST 没有精确历史接口，本地 `st_history` 过期会漏掉新戴帽，日更必须刷 `download_st_history`。
- **研究口径 vs 回测成交**：默认 research——信号日池保留涨跌停、标签不做成交掩码；回测仍拦买日一字涨停等。得分宇宙默认 strict。可用 `--tradable-strict` / `--label-exec-mask` 恢复旧的严格执行口径。
- **IC v2 与 Barra 纯化**：生产筛选即上一节的 HAC t / FDR / 去重 / long_share；`--feature-neutralize` 让 ML 特征与纯 IC 同口径，避免模型去学市值、行业等系统性敞口。
- **Walk-Forward 泄漏控制**：按时间向前滚；purged training + embargo；默认多窗共用近期验证集（`--val-window 0` 可关掉独立验证）。因子经 `get_factor_registry()` 统一注册，面板与残差化缓存带持有期 / 调仓频率指纹，换 horizon 不会误用旧缓存。
- **分位回测与成本**：Q1–Q5 + Top-N，佣金 / 印花 / 买卖价差进净值，而不是「不计成本的纸面多空」。
- **rolling-pool（滚动定池）**：每一期调仓日 t 的因子池，只用 **严格早于 t** 的 IC（当天 IC 依赖尚未实现的未来收益）。禁止把训练 / 验证 / 预测窗里出现过的因子并成一张大表。
- **live 与回测必须同口径**：生产路径是全量 Walk-Forward `--save-models`，live 按折加载当时那一期该用的模型（`live/predict_from_wf_models.py`）。`flagship_last_window` 是在最新窗口现训的快速备选，**折调度和中性化缓存都与回测不同**，不能拿来「验证」回测数字。
- **定位**：辅助人工选股，不是自动交易。

其它（可选）：AFML 分数差分动量、聚类特征重要性、可选 SHAP、事件类 special factors 注入。过拟合检验（PBO / DSR）用来泼冷水，不应当成「已经防过拟合」的卖点。

Agent 与开发约定见 [AGENTS.md](AGENTS.md)（内部速查，非对外教程）。

---

## 目录结构速览

```
quant_trading/
├── run.py                 # 主入口 CLI
├── ui/app.py              # 可选 Streamlit 简易面板（回测 run.py + IC 筛选）
├── config/                # settings.py、因子白名单 YAML
├── data/                  # 下载 / 清洗 / 行业 / 市值（raw 不进库）
├── factors/               # 因子实现 + registry + 面板缓存
├── models/                # Walk-Forward / dynamic / industry（含 wf/）
├── strategies/            # linear / ml 调度
├── backtest/              # 分组回测引擎与成本
├── research/              # IC v2、rolling_pool、pbo
├── utils/                 # rebalance / PIT / WLS / universe(cap-band)
├── live/                  # 实盘：按 fold 加载 WF 模型出候选股
├── tests/
├── docs/                  # 命令与流水线文档
├── logs/driver.py         # IC → YAML → 批量实验编排
└── results/<tag>/         # 本地实验产物（默认 gitignore）
```

---

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# 复制环境变量模板（可选：TUSHARE_TOKEN、DATA_ROOT）
copy .env.example .env

# 冒烟：下载 + 前 100 只股票跑通
python run.py --sample 100
```

日常训练、IC 筛选、cap-band、编排与高级开关见：

- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — 从数据到回测的推荐流程
- **[docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md)** — 最短命令与默认开关
- **[docs/PIPELINE.md](docs/PIPELINE.md)** — 端到端流水线与陷阱
- **[docs/LIVE_OPS_README.md](docs/LIVE_OPS_README.md)** — 旗舰 live：增量更新 → 全 WF 存模型 → 按 fold 出候选股
- `python run.py --help` / `--help-advanced`

研究侧最短命令（默认已含特征中性化与 10bp 买卖价差）：

```bash
python -m research.ic_analysis_v2 --period 5 --barra --save
python run.py --skip-download --mode xgb --horizon 5 \
  --factor-config config/factor_configs.yaml
```

### 图形界面（可选）

不想记命令时，可用本地 Streamlit 面板改常用参数并调用 `run.py`（回测）或 `research.ic_analysis_v2`（因子筛选）（**仍须**先装 Python / 依赖，并自行准备数据；全市场 / IC 全量很慢，建议先 sample 或短名单冒烟）。不做券商下单或 `logs/driver.py` 编排。

```bash
streamlit run ui/app.py
```

说明见 **[docs/UI.md](docs/UI.md)**。

---

## 实盘 / Live

定位：默认周五（W-FRI）收盘后出信号，**下一交易日开盘买**；人工看 Top100 / Top30；A 股 T+1。框架**不自动下单**。

旗舰名 `xgb_h5_sizeind_w156_nob`，配置 `config/flagship_xgb_h5_sizeind_w156_nob.yaml`。完整逐步命令、覆盖率数字与排错见 **[docs/LIVE_OPS_README.md](docs/LIVE_OPS_README.md)**（下文只给最短路径和必须检查项）。

### 生产主路径（与回测同口径）

先全量 Walk-Forward 训练并 `--save-models`，每期模型落到 `results/<tag>/models/`（只需在口径不变时跑一次；换中性化 / 持有期 / 调仓频率须重跑）。之后 live **按折加载**：`retrain_every=4` 时，例如 8 月 28 日的信号加载 8 月 14 日拟合的模型（中间三期复用，与回测调度一致），便于用同一份净值做实盘追踪。

```bash
# 1) 全 WF 存模型（口径变更或首次才需要；完整参数见 LIVE_OPS）
python run.py --skip-download --mode xgb --horizon 5 --save-models ...

# 2) 按 fold 出当日候选股
python -m live.predict_from_wf_models --as-of-date <信号日> \
  --model-dir results/xgb_h5_sizeind_w156_nob_wf_<日期> --top-n 100
```

### 快速备选（口径不同）

```bash
python -m live.flagship_last_window --no-download \
  --output-dir results/xgb_h5_sizeind_w156_nob_<日期>
```

在最新窗口现训一个模型再出分，快，但 Barra / 残差化缓存指纹与全量回测不一致，**不能用来核对回测 Top-N 或净值**。要严格同口径请走上一小节。

（通用增量出分入口 `python -m live.daily_update` 见 [docs/LIVE_DAILY.md](docs/LIVE_DAILY.md)，同样不是旗舰同口径路径。）

### 增量更新：最短顺序与必须检查

日常**不带** `--start`、不要 `--force-refresh` 全量扫股本。顺序：

```bash
python -m data.download
python -m data.download_stock_value_em
python -m data.download_st_history          # 沪市新戴帽靠这一步
python -m data.compute_market_cap           # 不要带 --start，否则换手会被截断
python -m data.download_shares              # 按过期天数增量，不要全量扫
```

上线前至少核对：

- **`circ_mv` 末日非空列数**：应是数千只；塌成几百只说明增量没跑完就被拼成稀疏行，须跑完再 assemble，不要对半成品 `--assemble-only`。
- **刷 ST**：`st_history` 过期会把新戴帽放进 Top-N。
- **换手不要被 lookback 截断**：`compute_market_cap` 带最近日期的 `--start` 会让 `turnover_rate` 只剩几十行。

---

## 许可与免责

本仓库以 [MIT License](LICENSE) 发布。

**免责声明**：本项目仅供学习与研究，不构成任何投资建议。证券投资有风险，据此操作造成的盈亏由使用者自行承担。
