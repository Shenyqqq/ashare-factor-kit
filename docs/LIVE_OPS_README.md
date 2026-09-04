# 旗舰 Live 运维手册（无 Agent 也能跑）

> 旗舰策略：`xgb_h5_sizeind_w156_nob`（xgb / horizon=5 / train=156 期 / val=0 / retrain_every=4 / time-decay 0 / cs_rank 不 winsor / feature-neutralize size_industry / dense 53 + sparse 11 / bid-ask 10bp / 无 B 股 / 北交所 92 保留）。
> 配置：`config/flagship_xgb_h5_sizeind_w156_nob.yaml`
>
> 生产路径：全 WF `--save-models` 存模型（§2.1）→ live 按 fold schedule 加载对应模型预测（§3.1），与回测严格同口径、方便实盘追踪。
>
> 本手册让你在没有 agent 时也能独立完成「增量更新数据 → 出 live 候选股 → 回测核验」。所有命令在仓库根 `F:\PythonProject\quant_trading` 下、激活 `.venv` 后执行。

---

## 0. 准备

```powershell
cd F:\PythonProject\quant_trading
.venv\Scripts\activate
# 单进程，避免内存爆炸 / AKShare 限流
$env:TRAIN_MAX_WORKERS="1"; $env:IC_MAX_WORKERS="1"; $env:DYNAMIC_MAX_WORKERS="1"
```

**数据末日 = 信号日**：A 股 T 日收盘后更新数据到 T 日，T 日即信号日，T+1 开盘买入（与 `forward_return = close[T+N]/open[T+1]-1` 一致）。今天周六（08-29），最新交易日是周五 08-28，所以信号日=08-28，买入日=周一 08-31。

---

## 1. 增量更新数据（按顺序）

### 1.1 OHLCV 后复权（主路径：东财）

```powershell
python -m data.download
```

- 增量拉取 close/open/high/low/volume/amount 的后复权（hfq）宽表到 `data/raw/*.parquet`。
- **东财接口挂了**走新浪兜底（全历史逐只慢，仅补缺失日）：

  ```powershell
  python -m data.backfill_prices_raw_sina
  ```

  注意：新浪是**不复权**日线（`prices_raw`），逐只全历史拉取很慢，只在东财恢复前补末日缺口用。东财恢复后回到 `python -m data.download` 主路径增量。

**验证覆盖率（一行）：**

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/raw/close_hfq.parquet'); print('close_hfq', d.shape, d.index.max().date(), '末日非空', int(d.iloc[-1].notna().sum()))"
```

期望：`rows≈2101 cols≈5808 末日=2026-08-28 末日非空≈5192`。末日非空 < 4000 → 东财增量没跑完，重跑。

### 1.2 东财市值（circ_mv / total_mv / pe_ttm / pb）—— 主路径

```powershell
python -m data.download_stock_value_em
```

- 增量拉 `circ_mv`/`total_mv`/`pe_ttm`/`pb`（东财日频，元）。含北交所 92。
- **不要把 `--start` 当历史起点覆盖全量**：`--start` 是 wide 表增量下界，传很早的日期会触发全量重拉（慢且易被限流）。日常增量**不带 `--start`**，让它按 `refresh_stale_days`（默认）增量刷新。
- 92 缺失可单独补：`python -m data.download_stock_value_em --bj-only`。

**验证：**

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/raw/circ_mv.parquet'); print('circ_mv', d.shape, d.index.max().date(), '末日非空', int(d.iloc[-1].notna().sum()))"
```

期望：`cols≈5569 末日非空≈5551`。**末日非空塌成几百只 → 增量没跑完被 assemble 写成稀疏行，重跑跑完再 assemble**（见 §4.1）。

### 1.3 ST 历史（沪市新戴帽靠这个）

```powershell
python -m data.download_st_history
```

- 产出 `data/raw/st_history.parquet`（长表，约 1800 段 / 800+ 只）。深交所有带日期精确历史；沪/北走保守 fallback（自 `list_date` 起标 ST）。
- **日更默认不刷这个**，要手动跑。沪市新戴帽股漏了会导致 Top100 出现 ST（见 §4.4）。

**验证：**

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/raw/st_history.parquet'); print('st_history', d.shape, '只数', d['code'].nunique())"
```

### 1.4 换手率（compute_market_cap）—— 注意 lookback 陷阱

```powershell
python -m data.compute_market_cap
```

- 读 `download_shares` 的股本 + `prices_raw` 算 `turnover_rate = (volume×100)/circ_shares`，顺带自算 `*_computed` 市值（校验/兜底）。
- **千万不要带 `--start` 当 lookback**：`--start` 是股本公告日下界，传最近 14 天会把 `ffill` 起点截到最近，导致 turnover 只剩几十行。**日常不带 `--start`**，全量 ffill。

**验证：**

```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/raw/turnover_rate.parquet'); print('turnover', d.shape, d.index.max().date(), '末日非空', int(d.iloc[-1].notna().sum()))"
```

期望：`rows≈2101 末日非空≈5044`。**只剩几十行 → 你带了 `--start`，去掉重算**（见 §4.3）。

### 1.5 股本（download_shares）—— 不要全量扫 5881 只

```powershell
python -m data.download_shares
```

- 按 `refresh_stale_days`（默认 30）增量刷新过期股本；**不要 `--force-refresh` 全量扫 5881 只**（极慢、易限流、中途 kill 会留半成品截断 parquet）。
- 启动前自动 `.bak`；全量刷新中间落盘须保留未完成旧行。

### 1.6 其它（龙虎榜 / 解禁 / 大宗）

非旗舰必需；旗舰 sparse 11 因子来自 `research/output/selected_factors_h5_sizeind_20260815.json`，已落盘。需要时单独跑对应 `data/download_*.py`，本手册不展开。

---

## 2. 跑回测 + 存模型（全 WF，生产主路径）

旗舰的**生产路径**是「全 WF `--save-models`」：从 2021 起完整 walk-forward 训练，每期 fold 模型留存到 `results/<tag>/models/`，live 出分时按 fold schedule 加载对应模型预测，**与回测严格同口径**（同一份 neut/Barra 缓存、同一批模型）。约 20-40 分钟（冷启动 neut 缓存更久）。

### 2.1 全 WF + --save-models（生产主路径，必跑一次）

```powershell
python run.py --skip-download --mode xgb --horizon 5 ^
  --factor-config config/factor_configs_h5_sizeind_20260815.yaml ^
  --special-factors sparse ^
  --sparse-from-ic research/output/selected_factors_h5_sizeind_20260815.json ^
  --train-windows 156 --train-window-units periods --val-window 0 ^
  --retrain-every 4 --time-decay 0 --feature-neutralize --neut-controls size_industry ^
  --bid-ask-spread 10 --save-models ^
  --output-dir results/xgb_h5_sizeind_w156_nob_wf_<日期>
```

- `--save-models`：每个 fit 日存一个 `xgb_w156_<fitdate>.joblib` 到 `results/<tag>/models/`，并写 `models_manifest.json`（含每个 fold 的 path/model/window/date/feature_names/feature_neutralize/neut_controls）。
- `retrain_every=4`：fit 日 = pred_offset ≡ 0 (mod 4) 的调仓日；中间 3 期复用上一个 fit 模型。`training_diagnostics_*.csv` 每行记 `date / reused_model / fit_date`，可核验调度。
- 产物：`backtest_*_nav.csv` / `_annual.csv` / `_risk_metrics.csv` / `.png` / `holdings_*` / `factor_scores_*.parquet` / `ic_series_*.csv` / `models/models_manifest.json` / `training_diagnostics_*.csv`。
- **当前全 WF 基线**（`results/xgb_h5_sizeind_w156_nob_wf_20260830/`，数据至 2026-08-28）：Top100 年化 23.3% / Sharpe 1.03 / MDD -24.3% / 超额 18.7%（等权基准 4.6%）；Q5 16.0% / Q1 -15.6%，单调性 1.00；IC均值 0.0787 / ICIR 0.6887 / 胜率 80.1%（286 期，fit=72 / reuse=214）；manifest 72 个 fold，最后 fit 日 **2026-08-14**（服务 08-28 信号日）。
- 注意：hfq 后复权会因分红除权**追溯调整历史价**，每年 8 月分红季后旧回测 NAV 会整体上移，属正常；跨口径比较须同一次复权基准。换口径（改 Barra 定义 / neut 控制变量 / horizon / 调仓频率）后须重跑全 WF，旧 manifest 不可复用。

### 2.2 last-window 续分回测（快速备选，口径略不同）

复用历史 WF 得分 + last-window 模型对新调仓日续分，避免从 2021 重训：

```powershell
python -m live.flagship_last_window --no-download --output-dir results/xgb_h5_sizeind_w156_nob_<日期>
```

- 训最新调仓日过去 156 期的一个 xgb，存 `models/`，再对数据末日出 Top100。
- **口径差异**：last-window 用 `live.daily_update` 的 warmup 窗（~450 日）重算 Barra+残差，Barra bundle 指纹与全量回测不同 → 残差化特征与全 WF 不完全一致（实测同日 Top100 重合仅 ~33%，spearman≈0.09）。**只适合快速看盘 / 临时出分，不可与全 WF 回测结论混用**。要严格同口径请用 §2.1 + §3.1。

### 2.3 拼接回测（历史得分 + last-window 续分）

把旧全 WF 得分与 last-window 新调仓日得分拼接，复用 `run.py` 回测口径出表。仓库根有一次性脚本 `backtest_stitch_0829.py`（参考其写法），产物落 `results/xgb_h5_sizeind_w156_nob_<日期>_bt/`。

---

## 3. 跑 Live（出当日候选股）

前提：数据已增量到 T 日（§1），且已跑过一次全 WF `--save-models`（§2.1，`results/<tag>/models/models_manifest.json` 存在）。

### 3.1 按 fold schedule 加载 WF 留存模型（生产主路径，与回测同口径）

```powershell
python -m live.predict_from_wf_models --as-of-date 2026-08-28 ^
  --model-dir results/xgb_h5_sizeind_w156_nob_wf_20260830 --top-n 100
```

- 复用 `run.py` 全量数据加载 + `strategies.ml.build_factor_dataset`，**命中回测的 `barra_bundle_*` 与 `factor_panel_neut_*` 缓存**（同一份 Barra 因子、同一份 Size+行业残差化特征），再加载 manifest 中服务信号日的 fold 模型 predict → 与回测严格同口径。
- 信号日截面 X 直接从 `dataset.factor_panel` 取（绕开 `get_cross_section` 对 `forward_return` 的 dropna —— 信号日标签未实现时它返回 None，但模型仍可对当日截面出分）。
- **fold schedule（retrain_every=4，W-FRI 周频）**：
  - fit 日 = pred_offset ≡ 0 (mod 4) 的调仓日；中间 3 期复用上一个 fit 模型。
  - 例：信号日 2026-08-28 的 pred_offset=18 → 复用 offset=16 的 fit 日 **2026-08-14** 模型（08-21、08-28 都复用 08-14，下一 fit 日是 09-11）。
  - 全 WF 产物 manifest 的"最新 fit 日"恰好就是服务当前信号日的 fold（数据末日前无更晚 fit），脚本取 `<= 信号日` 的最新 fit 日即正确。
- **同口径校验**：把 `--as-of-date` 设成回测已出分的日期（如 2026-08-21），对比 `candidates` 与回测 `ml_factor_scores_*.parquet` 该日 Top100 重合度。实测 08-21：**重合 99/100、spearman=1.0**（唯一差异为 strict mask 边界并列），确认 live 与回测同口径。
- 产物 `candidates_<信号日>.csv`，列：

  | 列 | 含义 |
  |----|------|
  | `signal_date` | 信号日（T，数据末日） |
  | `suggested_buy_date` | 建议买入日（T+1 开盘） |
  | `fit_date` | 用的哪个 fold 模型（=服务信号日的 fit 日） |
  | `rank` | 1..100（score 降序） |
  | `code` | 6 位代码 |
  | `name` | 股票名称 |
  | `sw_l2` | 申万二级行业 |
  | `circ_mv` | 流通市值（元） |
  | `circ_mv_yi` | 流通市值（亿元） |
  | `score` | 模型得分（越高越好） |

- 末尾日志打印 `Top100 中 B股=0 北交所92=0`，含 B 股会 `SystemExit` 报错。
- **当前 08-28 出分**（`results/xgb_h5_sizeind_w156_nob_wf_20260830/candidates_20260828.csv`）：fit_date=2026-08-14，买入日 2026-08-31；Top1 301267 华厦眼科 99.2亿 score=0.6400；score 区间 0.5825~0.64（非常数）；ST/B=0、北交所92=0；市值中位 58.0亿。

### 3.2 flagship_last_window 现训（快速备选，口径不同）

```powershell
python -m live.flagship_last_window --skip-train --no-download --output-dir results/xgb_h5_sizeind_w156_nob_<日期>
```

- 用 last-window 训出的单 fold 模型 + `live.daily_update` warmup 窗出分，快但 **Barra/neut 指纹与全 WF 回测不同**（见 §2.2 口径差异），仅作快速看盘备选，勿与全 WF 回测结论混用。要严格同口径请用 §3.1。

---

## 4. 常见数据缺失问题排查

### 4.1 circ_mv 末日塌成几百只
**原因**：东财增量没跑完就被 `--assemble-only` 写成稀疏行（部分股票末日没拉到）。
**处理**：重跑 `python -m data.download_stock_value_em`（不带 `--start`，跑完）再 `--assemble-only`；验证末日非空 ≈5500+。Size 因子读东财 `circ_mv`，覆盖不够会降级 `log(total_assets)` 并 warning。

### 4.2 prices_raw 停更
**原因**：日更默认不拉 raw（主路径用 hfq）；东财 raw 接口偶发不可用。
**处理**：走新浪兜底 `python -m data.backfill_prices_raw_sina`（全历史逐只慢，只补末日缺口）；东财恢复后回 `python -m data.download` 主路径增量。上线前用 `report_raw_hfq_coverage()` 验收 raw 覆盖。

### 4.3 turnover 截成几十行
**原因**：`compute_market_cap` 带了 `--start`（把 ffill 起点截到最近）。
**处理**：去掉 `--start`，`python -m data.compute_market_cap` 重算；验证 `rows≈2101`。

### 4.4 Top100 出现 ST
**原因**：`st_history.parquet` 过期 / 沪市新戴帽漏（沪/北无公开带日期接口，靠 fallback）。
**处理**：`python -m data.download_st_history` 刷新，再 `--skip-train` 重出分。沪市 ST 是保守近似，勿当精确历史。

### 4.5 大票异常变多
**原因**：① circ_mv 覆盖塌了（§4.1）；② Size 中性化被 `fillna(0)`（中性化失败 → Size 残差=原值 → 大票挤进 Top）；③ prices_raw 末日缺失导致可交易池错判。
**处理**：查 circ_mv 末日覆盖；看日志 `neut cache HIT/MISS` 与 `Size+PIT行业残差化` 是否完成；查 prices_raw 末日非空。

### 4.6 北交所 92 断更
**原因**：东财行情接口问题，**不是过滤问题**（`drop_excluded_universe_columns` 只剔 8 开头 BJ + B 股，保留 92）。
**处理**：`python -m data.download_stock_value_em --bj-only` 补 92；行情断更等接口恢复。

### 4.7 download_shares 全报 `No such keys(s): 'future.no_silent_downcasting'`
**原因**：`download_shares` 调 cninfo 股本接口时，akshare 内部 `pd.set_option('future.no_silent_downcasting', ...)` 在当前 pandas 版本已改名/移除，导致**逐只失败**（每只重试几次后跳过，日志一片 WARNING，全量扫 5884 只会卡几十分钟且无新增）。
**影响**：股本没更新，但 `turnover_rate` 由 `compute_market_cap` 用**已有 `share_change.parquet` + 当日 `prices_raw`** 重算，股本变动低频，几天内不影响。Size 主路径走东财 `circ_mv`，不依赖此步。
**处理**：① 不必等它跑完——`flagship_last_window` 的 `incremental_download` 把 `download_shares` 包在 try/except 里，失败只 warning 不中断；② 若卡住想跳过，Ctrl-C 终止后**手动补剩余步骤**再 `--no-download` 出分：

  ```powershell
  python -m data.download_st_history
  python -m data.compute_market_cap
  python -m live.flagship_last_window --no-download --output-dir results/xgb_h5_sizeind_w156_nob_<日期>
  ```

  ③ 长期修复：升级 akshare / pandas，或在 `data/download_shares.py::_fetch_one` 里 try/except 屏蔽该 option。

---

## 5. 怎么判断 live 出分是否可信

出分后看 `run.log` 末尾 + `candidates_*.csv`，逐项核对：

1. **circ_mv 覆盖**：日志 `circ_mv` 末日非空 ≈5500+；`candidates` 里 `circ_mv_yi` 不全 NaN。
2. **可交易池规模**：日志 `tradable_filter: ... 可交易格 / 2101 日`，末日可交易股应 4000+。
3. **Top100 市值结构 vs 历史**：`circ_mv_yi` 中位数 ~20-60 亿（size_industry 中性化后偏小中盘，随市场水位浮动）；若中位数 >100 亿 → Size 中性化可能失效（§4.5）。
4. **ST/B = 0**：日志 `Top100 中 B股=0 北交所92=0`（92 可为 0，B 必须 0）；含 B 股会 SystemExit。
5. **score 非常数**：分值有梯度（nunique≈100）；若全相等或全 NaN → 模型/特征异常。
6. **同口径校验**（§3.1）：把 `--as-of-date` 设成回测已出分日，对比 `candidates` 与 `ml_factor_scores_*.parquet` 该日 Top100 重合度，应 >90；实测 08-21 重合 99/100、spearman=1.0。
7. **IC**（全 WF 才有）：`ic_series_*.csv` mean IC ~0.079、ICIR ~0.69、胜率 ~80%（2026-08-30 全 WF 基线，286 期）。

---

## 6. 旗舰参数速查表

| 项 | 值 |
|----|----|
| 名 | `xgb_h5_sizeind_w156_nob` |
| YAML | `config/flagship_xgb_h5_sizeind_w156_nob.yaml` |
| mode | xgb |
| horizon | 5（周频） |
| hold_period | 5 |
| train_windows | 156（units=periods） |
| val_window | 0（无 val，仍 purge/embargo） |
| retrain_every | 4（每 4 期重训，中间复用） |
| time_decay | 0（窗内等权） |
| label_mode | cs_rank（不 winsor） |
| wf_selection | ic_weighted |
| feature_neutralize | True，neut_controls=size_industry（Size + PIT sw_l2，非 9 风格 Barra） |
| dense | `config/factor_configs_h5_sizeind_20260815.yaml`（53 个） |
| sparse | `research/output/selected_factors_h5_sizeind_20260815.json`（11 个，post-merge 注入） |
| bid_ask_spread | 10 bp |
| B 股 | 剔除（200/900） |
| 北交所 92 | 保留 |
| bt_score_universe | strict |
| 生产 live 入口 | `python -m live.predict_from_wf_models --as-of-date <T> --model-dir results/xgb_h5_sizeind_w156_nob_wf_<日期> --top-n 100`（按 fold 加载 WF 留存模型，与回测同口径） |
| 全 WF 存模型命令 | §2.1（`run.py ... --save-models --output-dir results/xgb_h5_sizeind_w156_nob_wf_<日期>`） |
| 快速备选 live | `python -m live.flagship_last_window --skip-train --no-download ...`（last-window 现训，口径不同） |
| 当前全 WF 基线 | `results/xgb_h5_sizeind_w156_nob_wf_20260830/`（Top100 23.3% / 超额 18.7%；manifest 72 fold，最后 fit 08-14） |

---

## 7. 不要做的事

- **不要全量扫 5881 只股本**（`--force-refresh`）：极慢、限流、中途 kill 留半成品。按 `refresh_stale_days` 增量。
- **不要把 `--start` 当 lookback / 历史起点**：`download_stock_value_em`、`compute_market_cap` 的 `--start` 是增量下界，传很早的日期会全量重拉，传最近日期会截断 ffill。日常不带 `--start`。
- **不要 kill 后 assemble 半成品**：`download_shares` 全量刷新中途 kill 会截断 parquet；要么跑完，要么用 `.bak` 回滚。
- **不要从 2021 重训全 WF 除非口径变更/要历史回测/要 --save-models 产物**：日常 live 用 §3.1 按 fold 加载已有 WF 留存模型即可（同口径、秒级出分）；只在换口径（Barra/neut/horizon/调仓频率）或 manifest 缺失时才重跑 §2.1 全 WF。
- **不要把 hold3 overlay 当旗舰**：旗舰是 horizon=5 周频；hold3 是另一套，勿混用参数。
- **不要改旗舰默认超参**：`train=156 / val=0 / retrain_every=4 / time_decay=0 / cs_rank` 是定稿，调参走 PBO 验证（`python -m research.pbo`）后再说。
