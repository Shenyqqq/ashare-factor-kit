# 图形界面（可选）

面向「不想记 CLI」的本地试跑：用 Streamlit 改常用参数，生成并执行：

- **回测**：`python run.py ...`
- **因子筛选**：`python -m research.ic_analysis_v2 ...`

## 能做什么 / 不能做什么

| 能 | 不能 |
|----|------|
| 中文标签改 mode / horizon / 因子 YAML / cap-band 等 | 券商下单、账号、云端 SaaS |
| 预览命令 + 本机 subprocess 跑回测或 IC | ``logs/driver.py`` 批量编排 / 自动写 YAML |
| IC 常用开关（barra / save / FDR / long_share / resume…） | 「无本地数据一键出完整回测 / 全量 IC」 |
| 看日志尾部、停止进程 | 投资建议或自动下单 |

全市场训练与 **IC 全量**都很慢；请先 sample / 短名单冒烟。数据需自行下载，见 [GETTING_STARTED.md](GETTING_STARTED.md)。


## 启动

在仓库根目录：

```bash
# 已创建 .venv 并安装依赖后
pip install -r requirements.txt   # 含 streamlit
streamlit run ui/app.py
```

浏览器会打开本地页面。界面优先调用仓库 `.venv` 里的 Python。


## 回测页（映射到 ``run.py``）

| 界面项 | CLI |
|--------|-----|
| 模式 | `--mode` |
| 持仓期 | `--horizon` |
| 因子白名单 | `--factor-config`（扫描 `config/*.yaml`） |
| 市值带 | `--cap-band`（含 `all` / `micro_30` / `micro_small_100` 等） |
| 训练窗口 | `--train-windows` |
| 验证窗口 | `--val-window` |
| 特征中性化 | `--feature-neutralize` / `--no-feature-neutralize` |
| 标签 | `--label-mode` |
| 训练目标 | `--objective`（默认 regression；选 rank 会警示 YetiRank） |
| 买卖价差成本 | `--bid-ask-spread` |
| 快速试跑 | `--sample`（0=全市场） |
| 特殊因子 | `--special-factors` |
| 输出目录 | `--output-dir` |
| 跳过下载 | `--skip-download` |

完整 CLI 仍以 `python run.py --help` / `--help-advanced` 与 [CLI_QUICKSTART.md](CLI_QUICKSTART.md) 为准。


## 因子筛选页（映射到 ``ic_analysis_v2``）

| 界面项 | CLI / 说明 |
|--------|------------|
| 持仓期 | `--period`（界面暴露 5 / 20） |
| Barra 纯 IC | `--barra`（默认开） |
| 写出结果 | `--save`（默认开） |
| BH-FDR | 默认开（与 CLI 一致）；关 → `--no-use-fdr` |
| 稠密门 | `--min-long-share`（默认 0.4；**0=关闭**） |
| 市值带 | `--cap-band` |
| 因子范围 | 全部 / 手动 `--factors` / 从 shortlist·YAML·JSON·txt 展开 |
| 续跑 | `--resume`（有警示） |
| 清空重算 | `--fresh`（危险项警示） |
| 并行 | `--workers`（界面默认 **1**） |

### 输出落盘（诚实说明）

开 `--save` 后主产物在 `research/output/`：

- `selected_factors_h{period}.json`（cap-band 等会有后缀）
- `ic_summary_h{period}.csv`、`ic_barra_pure_*.csv` 等
- checkpoint：`research/output/_checkpoints/`（供 `--resume`）

本页**不**调用 `logs/driver.py`，因此**不会**自动把 JSON 同步进 `config/factor_configs.yaml`。需要白名单 YAML 时请另跑 driver 或手改；回测页再选对应 YAML。

IC 全量很慢；短名单 + `--resume` 较快。改筛选阈值后勿盲目 resume，不确定时用 `--fresh`。


## 限制（请读）

1. **必须自备环境与数据**：装 Python、装依赖、下载行情/财务等；本 UI 不替你完成这些。
2. **本机跑、本机看日志**：长任务请用「刷新日志」；可「停止」发 terminate。回测与 IC **共用**同一进程槽（同时只能跑一个任务）。
3. **不做投资建议**：输出是研究用得分/回测/因子名单，需人工二次筛选。
4. **高级开关未全暴露**：rolling-pool、SHAP、Gram-Schmidt、decay/emerging 旋钮、driver 编排等请直接用 CLI。
