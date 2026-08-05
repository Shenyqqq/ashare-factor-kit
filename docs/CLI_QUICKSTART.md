# CLI Quickstart

新人只看本节即可开跑。完整流程见 [GETTING_STARTED.md](GETTING_STARTED.md)。  
全部参数：`python run.py --help` / `--help-advanced`，或 `python -m research.ic_analysis_v2 --help` / `--help-advanced`。

可选本地图形面板（包装常用 `run.py` 参数，仍需自备数据）：见 [UI.md](UI.md)，`streamlit run ui/app.py`。

## 日常最短命令

```bash
# 环境
.venv\Scripts\activate
pip install -r requirements.txt

# 冒烟（前 100 只）
python run.py --sample 100

# 训练 + 回测（默认已含 feature-neutralize / bid-ask=10bp / research tradable）
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml --models lgbm,xgb

# IC 筛选（默认已含 FDR / t=2.5 / corr-dedup；GS 需 --gram-schmidt）
python -m research.ic_analysis_v2 --period 5 --barra --save
python -m research.ic_analysis_v2 --period 20 --barra --save

# 市值缩域 IC / 训练（cap-band）
python -m research.ic_analysis_v2 --period 5 --barra --save --cap-band micro_30
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs_h5_mcap30_20260804.yaml \
  --cap-band micro_30

# 一键：IC → YAML → ensemble+blend（h5+h20）
python logs/driver.py --preset main
python logs/analyze_results.py

# 轮动定池 / 过拟合 / 退市股
python -m research.rolling_pool --horizon 5
python -m research.pbo
python -m data.download_delisted
```

## 默认已含（不必手写）

| 入口 | 默认 |
|------|------|
| `run.py` | `--feature-neutralize`、`BID_ASK_SPREAD_BPS=10`、research tradable、`--label-mode cs_zscore`、`--bt-score-universe strict` |
| `ic_analysis_v2` | `--use-fdr`、`--t-threshold 2.5`、`--corr-dedup`、decay/emerging/sparse 标注轨；`--gram-schmidt` 默认 OFF |

关闭示例：`--no-feature-neutralize`、`--no-use-fdr`、`--no-corr-dedup`。GS 显式开：`--gram-schmidt`。

## 高级开关（一行）

`run.py`：`--label-mode` / `--objective` / `--train-windows` / `--val-window`（默认 6 月共用近期 val；`0`=无独立 val，多窗须 `--wf-selection average`）/ `--cap-band`（`all` / `micro_30` / `micro_small_100` 等）/ `--sparse-from-ic` / `--rolling-pool-*` / `--position-regime` / `--portfolio-opt` / `--two-stage` / `--special-factors` / `--tradable-strict` / `--shap*` / `--tune` / `--multi-horizon` 等 → `--help-advanced`。

`ic_analysis_v2`：阈值门、`--universe` / `--cap-band`、decay/emerging/sparse 旋钮、`--resume`/`--fresh`、增量补录 `--only-new`/`--factors`（merge `ic_series`；`barra_pure` 指纹匹配时只补新区并 merge，仍重跑 selection）、workers → `--help-advanced`（详例见 `docs/操作手册.md`）。

## 变更说明（CLI 精简）

- 默认 `--help` 只列日常参数；高级与 deprecated 用 `help=SUPPRESS`，`--help-advanced` 可见。
- deprecated 仍接受（兼容旧脚本）：`--trainer-engine` / `--backtest-engine` / `--regime-cs` / `--ridge-drop-regime` / `--no-regime` / `--event-overlay`；IC 侧 `--decay-half-life-min` / `--decay-short-long-min` / `--emerging-recent-ic` / `--sparse-t-threshold`（no-op）。
- IC：`--use-fdr` 默认 ON；`--gram-schmidt` 默认 OFF（显式 `--gram-schmidt` 开启）。
- Cap-band：`micro_30` / `micro_lt30` 为流通市值 ∈(0,30 亿]、无 8 亿地板；`micro_small_100` 等见 `config/settings.py` 的 `CAP_BAND_PRESETS`。
