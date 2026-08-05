# A 股多因子量化选股框架

面向研究与辅助选股的 A 股多因子流水线：**数据下载 → 因子计算 → IC 筛选 → ML / 动态加权 → 分组回测 → 候选股输出**。

## 这是什么 / 不是什么

| 是 | 不是 |
|----|------|
| 可复现的研究框架与实验脚手架 | 全自动交易系统 / 券商下单接口 |
| 输出得分排名与 Top-N 候选池，供人工二次筛选 | 投资建议或收益承诺 |
| 强调 PIT、可交易口径、成本与过拟合检验 | 「一键复现完整历史曲线」的数据快照仓库 |

数据需自行通过 AKShare（等）下载；不同机器、不同下载时点与接口变更会导致结果不可逐点复现。过拟合风险真实存在，请用样本外与 PBO/DSR（概率过拟合 / 紧缩夏普，检验「挑出来的最优是否太好看」）等工具自检，勿盲信单次最优实验。

---

## 特色与特别实现

- **PIT 数据保护**（Point-in-Time：按当时可观测信息对齐，避免用到尚未披露的数据）：财务因子按法定披露窗口近似对齐（Q1/Q3=+30、半年报=+60、年报=+90；有 `ann_date` 则优先）；行业映射用 `industry_map_panel.parquet`；股票池保留退市股；ST 按时间序列查询。详见 [docs/PIT_AUDIT.md](docs/PIT_AUDIT.md)。
- **`clean_ret`**（涨跌停日收益置 NaN 的收益序列）：量价与 Barra Beta / ResVol 等统一走该口径，避免限价日系统性失真。
- **研究口径 vs 严格执行**：默认 **research**——IC/ML 信号日可交易池保留涨跌停、标签不做 execution mask（不因买入/卖出日不可成交而改标签）；回测仍拦截买日一字涨停等。得分宇宙默认 **`strict`**（回测用分以严格可交易 ∩ 标签可用为准），避免训练池膨胀泄漏进基准。可用 `--tradable-strict` / `--label-exec-mask` 恢复旧口径。
- **IC v2 生产筛选**：IC / ICIR（因子与前瞻收益的截面相关及其信息比率）经 Newey-West HAC t、BH-FDR（多重检验下控制假发现率）、corr-dedup、可交易池 mask、rolling ICIR、扣成本 IC、稀疏因子轨与 `long_share` 稠密门；支持 `--cap-band` 市值缩域（如 `micro_30` / `micro_small_100`）。
- **Barra + WLS 纯化**：以 √市值加权的截面回归（WLS）控制风格与行业敞口，得到「纯」因子暴露；`--feature-neutralize` 让 ML 特征与纯 IC 同口径。
- **Walk-Forward**：按时间向前滚动训练与预测；**purged training + embargo**（剔除与预测标签重叠的训练样本，并在边界留禁运带，降低泄漏）；默认多窗共用近期 val；`--val-window 0` 可关闭独立验证（多窗须 `wf_selection=average`）；多窗×多模型按 IC 加权 Z-score 集成。
- **分位回测与成本**：`pd.qcut` 得 **Q1–Q5**（升序分位，Q1=最低分、Q5=最高分）+ Top-N；佣金 / 印花税 / 滑点 + bid-ask spread（默认 10bp）。
- **因子注册与面板缓存**：`get_factor_registry()` 统一注册；因子面板 / Barra 残差化缓存（`neut_v6`，含 horizon·频率指纹）。
- **其它**：rolling-pool 轮动定池、AFML 分数差分 / Clustered FI / 可选 SHAP、事件与 special factors 注入等。

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
- `python run.py --help` / `--help-advanced`

### 图形界面（可选）

不想记命令时，可用本地 Streamlit 面板改常用参数并调用 `run.py`（回测）或 `research.ic_analysis_v2`（因子筛选）（**仍须**先装 Python / 依赖，并自行准备数据；全市场 / IC 全量很慢，建议先 sample 或短名单冒烟）。不做券商下单或 `logs/driver.py` 编排。

```bash
streamlit run ui/app.py
```

说明见 **[docs/UI.md](docs/UI.md)**。

---

## 许可与免责

本仓库以 [MIT License](LICENSE) 发布。

**免责声明**：本项目仅供学习与研究，不构成任何投资建议。证券投资有风险，据此操作造成的盈亏由使用者自行承担。
