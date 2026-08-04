# Getting Started

如果你会一点 Python、想亲手跑通「下载 → 选因子 → 训练 → 回测」，按下面顺序做即可。  
项目是什么、能做什么，先看根目录 [README.md](../README.md)；最短命令速查见 [CLI_QUICKSTART.md](CLI_QUICKSTART.md)；流水线细节见 [PIPELINE.md](PIPELINE.md)。

## 1. 环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
copy .env.example .env          # 按需填 TUSHARE_TOKEN / DATA_ROOT
```

数据默认落在 `data/raw/`、`data/universe/`（体积大，已 gitignore）。可用 `DATA_ROOT` 指到外部盘。

## 2. 冒烟

```bash
python run.py --sample 100
```

会尝试下载样本股票并跑通训练/回测。首次依赖网络与 AKShare 可用性；失败时检查接口与本地 parquet。

## 3. 准备完整数据（研究用）

按需执行（示例）：

```bash
python -m data.download_delisted          # 退市股，降低幸存者偏差
python -m data.download_stock_value_em    # 东财市值（Size / cap-band）
python -m data.download_shares            # 股本 → 换手
python -m data.compute_market_cap         # turnover_rate 等
python -m data.industry.download_industry # 行业 PIT 面板
```

主 OHLCV / 财务仍由 `run.py` 或 `data/download.py` 拉取。市值与质量校验见 `data/DATA_UPDATE.md`、`python -m data.validate_market_cap`。

## 4. IC 筛选 → 因子白名单

```bash
# Barra 纯 IC + 默认 FDR / t / corr-dedup；结果写 research/output/
python -m research.ic_analysis_v2 --period 5 --barra --save
python -m research.ic_analysis_v2 --period 20 --barra --save

# 市值缩域示例（小微盘）
python -m research.ic_analysis_v2 --period 5 --barra --save --cap-band micro_30

# JSON → config YAML（编排）
python logs/driver.py --ic-only --horizons 5,20
```

也可手写 / 选用 `config/factor_configs*.yaml`。

## 5. 训练与回测

```bash
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml

python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb

# 无独立 val（多窗须 average）；小微盘缩域
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs_h5_mcap30_20260804.yaml \
  --cap-band micro_30 --val-window 0 --wf-selection average
```

产物在 `results/<tag>/`（净值、metrics、可选 holdings）。默认已开 `--feature-neutralize` 与 bid-ask=10bp。

## 6. 一键编排与诊断

```bash
python logs/driver.py --preset main
python logs/analyze_results.py
python -m research.pbo                  # 过拟合检验（PBO + DSR）
python -m research.rolling_pool --horizon 5
```

## 7. 读文档地图

| 文档 | 用途 |
|------|------|
| [CLI_QUICKSTART.md](CLI_QUICKSTART.md) | 日常最短命令与默认开关 |
| [PIPELINE.md](PIPELINE.md) | 数据→IC→训练→回测全链路 |
| [PIT_AUDIT.md](PIT_AUDIT.md) | 财务 / 退市 / 行业 PIT 审计 |
| [操作手册.md](操作手册.md) | 无助手时的操作备忘 |
| [../AGENTS.md](../AGENTS.md) | 开发与 agent 权威约定 |

## 注意

- 完整历史曲线依赖你本机数据；仓库不附带 `data/raw`。
- 换 Barra / 中性化口径后须清 neut 缓存与 IC checkpoint（见 AGENTS.md 注意事项）。
- 默认 research 可交易口径 ≠ 回测 execution；解读 Q 组收益时注意口径。
