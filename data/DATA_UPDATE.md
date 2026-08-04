# 数据更新指南（Windows / `.venv` · 纯 AKShare）

本仓库日常数据更新以 **AKShare** 为准（`data.download`、股本/自算市值、中证全指等）。  
不要把 Tushare 当作主路径；`data.download_market_cap` 里若残留 Tushare 分支，**非本仓库日常流程**，可忽略。

仓库根目录执行；解释器用 `.\.venv\Scripts\python.exe`（或先 `.\.venv\Scripts\activate`）。

---

## 1. 日常最小命令组合（必跑）

```powershell
cd F:\PythonProject\quant_trading
$env:PYTHONIOENCODING = 'utf-8'

# ① OHLCV（后复权 + 不复权）+ 财务季报 —— 自带断点续传 / 增量补齐
.\.venv\Scripts\python.exe -m data.download --start 2018-01-01

# ② 东财日频市值（Size / WLS 主路径）—— 可增量 / resume / sample
#    全市场按票拉 stock_value_em，较慢；可后台跑，中断后续跑会 resume
.\.venv\Scripts\python.exe -m data.download_stock_value_em
# 调试：.\.venv\Scripts\python.exe -m data.download_stock_value_em --sample 20
# 指定：.\.venv\Scripts\python.exe -m data.download_stock_value_em --codes 600519,600000

# ②b 换手率（仍需）+ 自算市值校验面板（不覆盖东财主文件）
.\.venv\Scripts\python.exe -m data.download_shares
.\.venv\Scripts\python.exe -m data.compute_market_cap

# ②c（可选）东财主面板 vs 自算 / API 小样本校验
.\.venv\Scripts\python.exe -m data.validate_market_cap

# ③ 中证全指（市场代理 / 基准）—— 必须 force，否则读旧缓存不刷新
.\.venv\Scripts\python.exe -c "from strategies.market_state import download_csi_all; s=download_csi_all(force=True); print(s.index.min().date(), '->', s.index.max().date(), 'n=', len(s))"
```

验收（价格 / 市值 / 指数末日应对齐到最近交易日附近；财务报告期可更早）：

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
from pathlib import Path
for f in ['close_hfq.parquet','total_mv.parquet','circ_mv.parquet','csi_all.parquet','financial_indicators.parquet']:
    p=Path('data/raw')/f
    if not p.exists():
        print(f, 'MISSING'); continue
    df=pd.read_parquet(p)
    if 'trade_date' in df.columns:
        d=pd.to_datetime(df['trade_date']); print(f, d.max().date(), df.shape)
    elif 'date' in df.columns:
        d=pd.to_datetime(df['date']); print(f, d.max().date(), df.shape)
    else:
        print(f, pd.to_datetime(df.index).max().date(), df.shape)
"
```

---

## 2. 各扩展表何时跑

| 模块 | 命令 | 何时需要 |
|------|------|----------|
| 主链路 OHLCV+财务 | `python -m data.download` | **每次日常更新** |
| 东财日频市值 | `python -m data.download_stock_value_em` | **每次日常更新**（Size / WLS / 对数市值） |
| 股本 → 换手 + 自算校验 | `download_shares` + `compute_market_cap` | **每次日常更新**（`turnover_rate`；`*_computed` 对照） |
| 中证全指 | `download_csi_all(force=True)` | **每次日常更新** |
| 退市股 OHLCV | `python -m data.download_delisted` | 月更 / 消幸存者偏差审计 |
| 行业 PIT | `python -m data.industry.download_industry` | 行业变更后 / Barra 行业哑变量异常时 |
| ST 历史 | `python -m data.download_st_history` | 需要真实 ST 时间序列时 |
| 龙虎榜 / 解禁 / 北向 / 融资融券 / 资金流 等 | 对应 `data.download_*` | 只用到相关特殊因子时 |
| 质量报告 | `python -m data.download --quality-report` 或 `python -m data.quality_report` | 大更新后抽查 |

最小可用集：**OHLCV + 财务 + 市值（`total_mv`/`circ_mv`，由 `download_stock_value_em` 产出）+ 换手（`turnover_rate`）+ 中证全指**。  
有 PIT 行业面板（`industry_map_panel.parquet`）即可跑 Barra；不必每次重下行业。

---

## 3. 常见坑

1. **没有 `--update` 开关**  
   文档里曾写 `python -m data.download --update`，**CLI 不存在该参数**。直接跑 `python -m data.download` 即增量（已有股票只补新区间）。

2. **`csi_all` 默认不刷新**  
   `download_csi_all()` 在文件已存在且 `force=False` 时直接读缓存。日常必须 `force=True`，或先删 `data/raw/csi_all.parquet` 再拉。  
   东财失败时可能回退新浪；新浪历史常止于 ~2016，按 `start=2017-01-01` 过滤后可能为空——实现会**拒绝用空表覆盖缓存**；失败时清代理后重试东财。

3. **市值主路径 = 东财 `stock_value_em`**  
   - `python -m data.download_stock_value_em` → `total_mv` / `circ_mv`（及 `pe_ttm` / `pb`）  
     支持 `--sample` / `--codes` / `--refresh-stale-days` / `--force-refresh` / `--assemble-only`；单票缓存于 `data/raw/_cache/stock_value_em/`。  
   - 换手仍两步：`download_shares` → `compute_market_cap`（写 `turnover_rate` + `*_computed`，**默认不覆盖**东财主面板；应急才 `--promote-main`）。  
   **`download_market_cap` 已 deprecated**。对照：`python -m data.validate_market_cap`。

3b. **`prices_raw` 必须与 hfq 对齐**（自算校验 / 换手仍依赖）  
   `data.download` 不复权阶段会对照 `close_hfq` 末日强制补齐落后股票，结束后 `report_raw_hfq_coverage()`。  
   东财不可用时：`python -m data.backfill_prices_raw_sina`（新浪不复权收盘）。  
   Size 主路径已不依赖 `prices_raw × shares`；自算 `*_computed` 仍会。

3c. **volume 单位 = 手（×100 → 股）**  
   `volume.parquet` 为 AKShare `stock_zh_a_hist` 原口径；换手率 `(volume×100)/circ_shares`。

3d. **北向已停更（约 2024-08-19）**  
   默认不加载进 IC/`run.py`；勿用停更后区间做因子。

3e. **行业 PIT 严格**  
   跑 `--barra` 前需有 `industry_map_panel.parquet`；缺失默认报错（`--allow-static-industry` 仅 debug）。

4. **财务末日 ≠ 价格末日**  
   季报 `financial_indicators` 的报告期往往停在最近财报季（如 3/31、6/30），价格可到最近交易日——属正常。

5. **IC 序列末日会短于价格末日**  
   前向收益 `close[t+N]/open[t+1]-1` 需要未来 N 日，h5 约短 1 周。数据更到 T，IC 末日大约到 T−持仓期。

6. **代理 / 网络**  
   东财偶发连接失败时，可临时清空 `HTTP_PROXY`/`HTTPS_PROXY` 再跑指数与 AKShare。

7. **不要用 PowerShell `Tee-Object` 做唯一日志**  
   易 UTF-16/乱码；重活一次只跑一个（下载 / IC / WF）。

---

## 4. 更新后如何触发 IC 刷新

**无「只算新交易日」的真增量**；`--resume` 只跳过**已完成阶段的 checkpoint**。  
数据末日前进后，建议清空并重算 `ic_series` 及下游。

### 推荐（h5 生产口径，pure AND）

```powershell
Remove-Item research\output\_checkpoints\ic_series_h5.pkl -ErrorAction SilentlyContinue
Remove-Item research\output\_checkpoints\summary_h5.pkl -ErrorAction SilentlyContinue
Remove-Item research\output\_checkpoints\yearly_h5.pkl -ErrorAction SilentlyContinue
Remove-Item research\output\_checkpoints\barra_pure_h5.pkl -ErrorAction SilentlyContinue
Remove-Item research\output\_checkpoints\selection_h5.pkl -ErrorAction SilentlyContinue
Remove-Item research\output\_checkpoints\gramschmidt_h5.pkl -ErrorAction SilentlyContinue

$env:IC_MAX_WORKERS = '1'
$env:BARRA_IC_WORKERS = '2'

.\.venv\Scripts\python.exe -m research.ic_analysis_v2 `
  --period 5 --barra --save --use-fdr --t-threshold 2.5 --gram-schmidt `
  --workers 1 --barra-workers 2
```

- 默认稠密门：**pure** `|IC|≥0.015 ∧ |ICIR|≥0.30`（AND）。  
- 加速刷序列可加 `--no-quantile-decomp`；最终仍建议有可用 pure IC 序列。  
- 崩溃续跑：加 `--resume`（改阈值后不要误用旧 selection checkpoint）。  
- 月频：`--period 20`（checkpoint 后缀 `_h20`）。

验收：

```powershell
.\.venv\Scripts\python.exe -c "
import pickle, pandas as pd
with open('research/output/_checkpoints/ic_series_h5.pkl','rb') as f:
    n, d = pickle.load(f)
ends=[pd.Series(v).dropna().index.max() for v in d.values() if len(pd.Series(v).dropna())]
print('factors', len(d), 'IC max_end', max(ends).date())
"
```

产物：`research/output/ic_summary_h5.csv`、`ic_barra_pure_h5.csv`、`selected_factors_h5.json` 等。

---

## 5. 一句话流程

**`data.download` → `download_stock_value_em` → `download_shares` + `compute_market_cap` → `download_csi_all(force=True)` → 清 IC checkpoint → `ic_analysis_v2 --period 5 --barra --save --use-fdr ...` → 滚动近窗体检 / 回测。**
