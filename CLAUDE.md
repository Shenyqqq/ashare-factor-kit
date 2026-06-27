# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

A股多因子量化选股策略，完整流程：数据下载 → 因子计算 → 回测 → 可视化。

## 常用命令

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 快速测试跑通（100只股票，几分钟）
python run.py --sample 100

# 正式全量运行（首次，几小时）
python run.py

# 跳过下载，仅重跑因子+回测（数据已存在时）
python run.py --skip-download

# 单独下载数据
python -m data.download --start 2018-01-01 --end 2024-12-31 --sample 100

# 单独计算因子
python -m factors.compute

# 安装依赖
pip install -r requirements.txt
```

## 架构概览

### 数据流

```
AKShare / Tushare
      ↓
data/download.py → data/raw/prices_hfq.parquet
                 → data/raw/financial_indicators.parquet
                 → data/universe/stock_list.parquet
      ↓
factors/compute.py → data/processed/composite_factor.parquet
      ↓
backtest/engine.py → BacktestResult → backtest_result.png
```

### 关键设计决策

**因子方向编码在权重符号里**：`FACTOR_WEIGHTS` 中负权重不是"减少影响"，而是"方向取反"。例如 `"value_pb": -0.20` 表示低PB得高分（PB越低，因子得分越高）。修改权重时必须保持这个语义。

**因子标准化流程**：每个因子经过 winsorize（截面1%尾）→ cross_sectional_zscore（截面z-score，clip=3σ）。这是截面操作（按日期横向），不是时间序列操作。

**回测引擎无第三方依赖**：纯 pandas 实现，成本模型 = 手续费（双边 0.1%）+ 印花税（卖出 0.1%）+ 滑点（0.2%）。

**数据对齐**：财务数据（季报频率）通过 `reindex(..., method="ffill")` 前向填充到日频，与价格数据对齐。

### 配置

所有参数集中在 `config/settings.py`，不要在各模块里硬编码参数：

| 关键参数 | 默认值 | 说明 |
|---------|--------|------|
| `FACTOR_WEIGHTS` | 见文件 | 权重正负编码方向语义 |
| `N_STOCKS` | 30 | 选股数量 |
| `REBALANCE_FREQ` | `"ME"` | 调仓频率（pandas offset alias） |
| `BACKTEST_START/END` | 2018-2024 | 回测区间 |

环境变量（`.env` 文件）：`TUSHARE_TOKEN`、`DATA_ROOT`（可改为外部硬盘路径）。

### 扩展方向（README 规划中）

- `factors/ml_factor.py`：用 XGBoost 做非线性因子合成，替换线性加权
- `portfolio/optimizer.py`：风险平价替代等权
- `execution/broker.py`：对接券商 API

新增因子需在 `factors/compute.py` 的 `factor_map` 字典中注册，并在 `config/settings.py` 的 `FACTOR_WEIGHTS` 中添加权重。
