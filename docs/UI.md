# 图形界面（可选）

面向「不想记 CLI」的本地试跑：用 Streamlit 改常用参数，生成并执行 `python run.py ...`。

## 能做什么 / 不能做什么

| 能 | 不能 |
|----|------|
| 中文标签改 mode / horizon / 因子 YAML / cap-band 等 | 券商下单、账号、云端 SaaS |
| 预览命令 + 本机 subprocess 跑 `run.py` | IC 全量筛选 / `logs/driver.py` 编排 |
| 看日志尾部 | 「无本地数据一键出完整回测」 |

全市场训练很慢；请先 `--sample` 冒烟。数据需自行下载，见 [GETTING_STARTED.md](GETTING_STARTED.md)。

## 启动

在仓库根目录：

```bash
# 已创建 .venv 并安装依赖后
pip install -r requirements.txt   # 含 streamlit
streamlit run ui/app.py
```

浏览器会打开本地页面。界面优先调用仓库 `.venv` 里的 Python。

## 界面暴露的参数（映射到 CLI）

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

## 限制（请读）

1. **必须自备环境与数据**：装 Python、装依赖、下载行情/财务等；本 UI 不替你完成这些。
2. **本机跑、本机看日志**：长任务请用「刷新日志」；可「停止」发 terminate。
3. **不做投资建议**：输出是研究用得分/回测，需人工二次筛选。
4. **高级开关未全暴露**：rolling-pool、SHAP、两阶段、position-regime 等请直接用 CLI。
