# A股量化交易项目

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 token（可选，不填只用 AKShare）
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN

# 3. 先用少量股票跑通流程（快，几分钟）
python run.py --sample 100

# 4. 正式跑全量（慢，几小时，只需跑一次）
python run.py

# 5. 之后数据已有，只重跑因子+回测
python run.py --skip-download
```

## 项目结构

```
quant_trading/
├── config/
│   └── settings.py          ← 所有参数改这里
├── data/
│   ├── download.py          ← 拉取原始数据
│   ├── raw/                 ← 原始 parquet 文件
│   └── processed/           ← 处理后的因子文件
├── factors/
│   └── compute.py           ← 因子计算（动量/价值/质量/规模）
├── backtest/
│   └── engine.py            ← 回测引擎 + 绩效分析
├── research/
│   └── notebooks/
│       └── factor_analysis.ipynb  ← 因子有效性探索
├── run.py                   ← 一键运行入口
└── requirements.txt
```

## 调参指南

所有参数集中在 `config/settings.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| N_STOCKS | 30 | 持仓股票数 |
| REBALANCE_FREQ | ME | 调仓频率（ME=月末，W-FRI=周五）|
| FACTOR_WEIGHTS | 见文件 | 各因子权重，正=越高越好，负=越低越好 |
| COMMISSION_RATE | 0.001 | 手续费率 |

## 下一步

1. 在 `research/notebooks/factor_analysis.ipynb` 里分析因子 IC，筛掉无效因子
2. 用 XGBoost 替换线性加权，做非线性因子合成（`factors/ml_factor.py`）
3. 加入组合优化（风险平价）替代等权（`portfolio/optimizer.py`）
4. 对接券商 API 自动化执行（`execution/broker.py`）
