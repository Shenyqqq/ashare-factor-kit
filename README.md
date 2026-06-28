# A股多因子量化选股框架

个人使用的 A 股量化选股系统。完整流程：数据下载 → 因子计算 → ML 模型训练 → 回测验证 → 候选股输出。

定位：**辅助人工选股，不做全自动交易**。模型输出得分排名，人工二次筛选后手动操作，持仓集中。

---

## 快速开始

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 首次：下载数据（全量几小时，--sample 100 快速测试）
python run.py --sample 100

# 之后数据已有，直接跑策略+回测
python run.py --skip-download --mode ensemble --horizon 20
```

---

## 数据准备（一次性）

```bash
# 1. 下载全市场日K + 财务数据（主数据，首次需要几小时）
python run.py  # 含下载步骤

# 2. 下载业绩预告数据（2018-2025，已含约11万条记录）
python -m data.events.download_yjyg

# 3. 下载申万二级行业映射（分行业训练需要）
python -m data.industry.download_industry

# 4. 下载沪深300指数（市场状态识别需要，自动触发）
python -m strategies.market_state
```

---

## 主入口：run.py

```bash
python run.py [--mode MODE] [--horizon N] [--backtest MODE] [--skip-download] [--sample N] [--report]
```

### `--mode`：策略模式

| 模式 | 说明 | 速度 |
|------|------|------|
| `linear` | 线性加权因子，无需训练，作为基准 | 快 |
| `ridge` | Ridge 回归单模型 | 中 |
| `lgbm` | LightGBM 单模型 | 中 |
| `xgb` | XGBoost 单模型 | 中 |
| `cat` | CatBoost 单模型 | 中 |
| `ensemble` | Ridge+LightGBM+XGBoost+CatBoost Rank Averaging（推荐） | 慢 |
| `industry` | 分行业 ensemble，申万二级各行业独立训练（最强，最慢） | 很慢 |

### `--horizon`：持仓期（交易日）

| 值 | 调仓频率 | 适用场景 |
|----|---------|---------|
| `5` | 周频 | 短线，T+1 影响已可忽略 |
| `10` | 双周 | 中短线 |
| `20` | 月频（默认） | 最稳定，IC 最高 |
| `60` | 季频 | 低频，换手少 |

> A 股 T+1 制度：horizon < 3 的结果不具备实盘参考价值。

### `--backtest`：回测模式

| 模式 | 说明 |
|------|------|
| `quantile`（默认） | Q1-Q5 分组回测，验证因子排名能力，4图输出 |
| `topN` | top-30 等权持仓，看绝对收益 |

### 常用命令组合

```bash
# 基准对照（线性，月频，Q1-Q5）
python run.py --mode linear --horizon 20 --skip-download

# 主力策略（ML ensemble，月频）
python run.py --mode ensemble --horizon 20 --skip-download

# 周频版本
python run.py --mode ensemble --horizon 5 --skip-download

# 分行业版本（需要先下载行业数据）
python run.py --mode industry --horizon 20 --skip-download

# 训练后展示 IC / SHAP 分析报告
python run.py --mode ensemble --skip-download --report

# 快速调试（100只股票，跳过下载）
python run.py --mode ensemble --sample 100 --skip-download
```

---

## 因子分析：ic_analysis.py

诊断每个因子的预测能力，是调参和因子筛选的主要工具。

```bash
# 基础：全因子 IC 汇总 + 逐年分解（最常用）
python -m research.ic_analysis --period 20

# IC 衰减曲线（5/10/20/40/60 日，耗时较长）
python -m research.ic_analysis --period 20 --decay

# 因子相关矩阵（发现冗余因子）
python -m research.ic_analysis --period 20 --corr

# 申万二级行业中性化后的纯选股 IC（去掉行业 Beta）
python -m research.ic_analysis --period 20 --neutralize

# 输出图表 + 保存 CSV
python -m research.ic_analysis --period 20 --plot --save

# 组合使用
python -m research.ic_analysis --period 20 --decay --corr --plot --neutralize --save
```

输出说明：
- **全周期汇总**：绿=有效（|IC|>0.05），黄=弱信号（>0.03），红=无效
- **逐年 IC**：观察因子是否在衰减（连续下降→alpha 被套利，需换因子）
- **IC 衰减表**：找到每个因子最适合的持仓周期

---

## 各模块说明

### 因子层

**`factors/factor.py`** — 15 个 alpha 因子，`get_factor_registry()` 统一入口

| 类别 | 因子名 | 逻辑 |
|------|--------|------|
| 动量 | `momentum` | 过去 12 个月收益率 |
| 动量 | `reversal` | 过去 1 个月收益率（反转） |
| 动量 | `momentum_skip` | 12-1 月跳过最近 1 月的动量 |
| 动量 | `price_to_high` | 价格相对 52 周高点的位置 |
| 波动 | `volatility` | 60 日收益率标准差（低波动溢价） |
| 流动性 | `turnover` | 20 日平均换手率 |
| 流动性 | `amihud` | Amihud 非流动性比率（冲击成本） |
| 估值 | `value_pb` | 市净率倒数（低 PB = 高得分） |
| 估值 | `value_ep` | 市盈率倒数（低 PE = 高得分） |
| 质量 | `roe` | 净资产收益率 |
| 质量 | `roe_chg` | ROE 变化量（改善趋势） |
| 质量 | `gpm` | 毛利率 |
| 质量 | `gpm_chg` | 毛利率变化量 |
| 质量 | `accrual` | 应计项目比率（负向，低应计=高质量） |
| 规模 | `size` | 市值对数（小市值溢价） |
| 杠杆 | `leverage` | 负债率（低杠杆溢价） |

> 因子方向编码在 `config/settings.py` 的 `FACTOR_WEIGHTS` 符号里：负权重 = 方向取反，不是"减少影响"。

**`factors/factor_event.py`** — 业绩预告事件因子

- 预增/扭亏/首盈 → 正分；预减/首亏/预亏 → 负分
- 变动幅度对数放大，公告前股价已大幅上涨则惩罚（pre-drift 惩罚）
- 信号在公告日触发，持续 60 个交易日后衰减

### 模型层

**`models/trainer.py`** — `WalkForwardTrainer`

- Walk-Forward 滚动训练：每个调仓日用历史数据训练，预测当日
- 多训练窗口（24/36/60 月）× 多模型 → Rank Averaging ensemble
- 时间衰减权重（越近的训练样本权重越高）
- 自动输出样本外 IC 统计

**`models/industry_trainer.py`** — `IndustryWalkForwardTrainer`

- 对每个申万二级行业单独训练一套模型
- 银行用银行的因子权重，半导体用半导体的因子权重
- 小行业（<30 只）自动合并到对应一级行业桶
- 输出格式与普通 trainer 相同，可直接对接回测

**`models/analyzer.py`** — 训练后分析报告（`--report` 触发）

- 各模型 IC 对比、SHAP 因子重要性、分组净值

### 回测层

**`backtest/quantile.py`** — Q1-Q5 分组回测（主回测）

按因子得分将全市场分成 5 组，观察：
- Q5（高分）是否持续跑赢 Q1（低分）
- 分组收益是否单调递增（单调性越接近 1.0，选股能力越强）
- 多空组合（Q5-Q1）的绝对超额收益
- 逐年分组收益（发现策略在哪些年份失效）

**`backtest/engine.py`** — top-N 等权回测（备用）

固定选 top-30 等权持仓，看绝对收益曲线。适合和基准指数做简单对比。

### 策略层

**`strategies/market_state.py`** — 市场状态识别

基于沪深 300 多均线（MA20/MA60）将市场分为牛/震荡/熊三态，用途：
1. 作为 ML 模型特征（`encode_state()` → 1/0/-1）
2. 分状态 IC 分析（因子在牛市和熊市有效性可能截然不同）
3. 未来可扩展为高层路由（熊市降仓/切换防御因子）

```python
from strategies.market_state import get_market_state, ic_by_state
state = get_market_state()   # Series: 'bull' / 'neutral' / 'bear'
```

### 研究层

**`research/ic_analysis.py`** — 见上方"因子分析"章节

**`research/notebooks/factor_analysis.ipynb`** — 交互式探索（Jupyter）

---

## 项目结构

```
quant_trading/
├── run.py                          ← 主入口（策略模式 × 持仓期 × 回测模式）
├── config/
│   └── settings.py                 ← 所有参数（因子权重、持仓数、回测区间等）
├── data/
│   ├── download.py                 ← AKShare 拉取日K + 财务数据
│   ├── clean.py                    ← 数据清洗
│   ├── raw/                        ← 原始 parquet（prices_hfq / financial / yjyg 等）
│   ├── events/
│   │   └── download_yjyg.py        ← 业绩预告下载（18-25年，11万条）
│   └── industry/
│       └── download_industry.py    ← 申万二级行业成分股
├── factors/
│   ├── factor.py                   ← 15个 alpha 因子 + get_factor_registry()
│   └── factor_event.py             ← 业绩预告事件因子
├── models/
│   ├── trainer.py                  ← WalkForwardTrainer（全市场）
│   ├── industry_trainer.py         ← IndustryWalkForwardTrainer（分行业）
│   └── analyzer.py                 ← IC/SHAP/分组净值报告
├── strategies/
│   ├── linear.py                   ← 线性加权策略（基准）
│   ├── ml.py                       ← ML 策略入口
│   └── market_state.py             ← 市场状态识别（牛/震荡/熊）
├── backtest/
│   ├── quantile.py                 ← Q1-Q5 分组回测（主回测）
│   └── engine.py                   ← top-N 等权回测（备用）
└── research/
    ├── ic_analysis.py              ← 全功能因子 IC 诊断脚本
    └── notebooks/
        └── factor_analysis.ipynb   ← 交互式因子探索
```

---

## 配置参数（config/settings.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FACTOR_WEIGHTS` | 见文件 | 线性策略各因子权重，符号编码方向 |
| `N_STOCKS` | 30 | topN 回测模式的持仓数量 |
| `BACKTEST_START/END` | 2018/2024 | 回测时间范围 |
| `COMMISSION_RATE` | 0.0001 | 手续费率（双边各 0.01%） |
| `STAMP_DUTY` | 0.0005 | 印花税（卖出单边 0.05%） |

ML 训练超参数在 `models/trainer.py` 顶部集中配置（`TRAIN_WINDOWS_MONTHS`、`TIME_DECAY` 等）。

---

## 数据来源

- **行情数据**：AKShare（免费，全市场 A 股日 K，复权价）
- **财务数据**：AKShare（ROE/PB/PE/毛利率等，季报频率前向填充）
- **业绩预告**：AKShare `stock_yjyg_em`（东方财富口径）
- **行业分类**：AKShare 申万二级行业成分股
- **指数数据**：AKShare 沪深 300 日线

---

## 使用建议

1. **先跑 IC 分析**，确认因子在当前市场有效后再跑 ML 训练
2. **不同 horizon 分开看**：`--horizon 5` 和 `--horizon 20` 训练的是完全不同的模型参数
3. **Q5 是候选股池**，不是直接买入信号，结合基本面和市场状态人工二次筛选
4. **分行业模型（industry）** 适合行业轮动明显的时期；全市场 ensemble 更稳定
5. **模拟盘先验证**：新参数至少跑 3 个月模拟盘再考虑实盘
