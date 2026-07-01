# 实验产物（results/）

所有模型训练 / 回测输出落在 `results/`，不落仓库根或 `data/processed/`。本目录被 `.gitignore` 忽略，仅本地存在。

## 目录布局

- `results/<tag>/` — 单次实验默认目录（`--output-dir` 省略时，`tag` 例如 `lgbm_h5`）
- `results/<batch>/` — 通过 `run.py --output-dir results/<batch>/` 指定的批次目录
- `research/output/` — IC 分析输出（不在本目录）

## 命名约定

```
tag = {mode}_h{horizon}[_w{train_windows}][_m{models}][_blend][_v2]
```

例：`ensemble_h20_w6-12_mlgbm-xgb_blend`

## 单次实验输出文件

| 文件 | 说明 |
|------|------|
| `factor_scores_<tag>.parquet` | 最终预测得分（调仓日 × 股票；linear 为日频） |
| `backtest_<tag>.png` | Q1–Q5 + Top30 + 基准 + 指数 四宫格图 |
| `backtest_<tag>_nav.csv` / `_annual.csv` / `_longshort.csv` | 净值 / 逐年收益 / 多空净值 |
| `holdings_top30_<tag>.csv` | 每期 Top30 候选股（`--holdings`） |
| `model_metrics_<tag>.json` | IC 均值、ICIR、胜率、预测期数（ML / industry） |
| `ic_series_<tag>.csv` | 逐期样本外 IC（ML / industry） |

## 历史批次目录

| 目录 | 说明 |
|------|------|
| `v1_ablation/` | 全模型消融 × h5/h20 |
| `v2_dynamic_short/` | dynamic + ensemble w6-12 |
| `v3_cat_fix/` | catboost 修复后重跑 |
| `v4_normal_window/` | ensemble lgbm+xgb blend，正常窗口 |
| `v5_window612_h5/` | 2026-06-30 批次：train windows [6,12]，h5 全模型消融（见目录内 README） |
| `v6/` | 后续批次 |
| `dynamic_compare/` | Dynamic lookback 6 vs 12 对照（h5/h20） |
| `dynamic_h5/` | Dynamic h5 lookback 26/52 |
| `ridge_window_sweep/` | Ridge 训练窗口扫描（h10 w6-12 / w12-24 等） |
| `lgbm_window_sweep/` / `lgbm_window_sweep_h10/` | LGBM 窗口扫描 |
| `ridge_h5_w6_p_v2/` | Ridge h5 + v2 trainer 引擎实验 |
| `repro_week_windows/` | 周频窗口复现实验 |
| `cat_h5/` / `cat_h20/` / `lgbm_h5/` / `lgbm_h20/` / `xgb_h5/` / `xgb_h20/` | 单模型单 horizon 实验 |
| `linear_h5/` | 线性基准 |
| `ic_analysis/` | IC 分析相关产物归档 |

## CLI 示例

```bash
# 单次实验指定输出目录
python run.py --skip-download --mode lgbm --horizon 5 \
  --factor-config config/factor_configs.yaml \
  --output-dir results/v5_window612_h5/

# 批量实验由 logs/driver.py 编排
python logs/driver.py --preset main
python logs/analyze_results.py        # 汇总 model_metrics / backtest
```
