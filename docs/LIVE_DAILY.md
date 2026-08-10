# 实盘日更增量链路（live/daily_update）

每日增量出分，**辅助人工选股，非自动交易**。与全量研究回测路径（`run.py`）并存，不替换它。

## 是什么 / 不是什么

| 是 | 不是 |
|----|------|
| 每日只拉最近 N 日行情 append 到 `data/raw` | 历史回填（用 `data.download` 全量） |
| 因子面板只算新增交易日行 concat | 全量重算因子库（用 `run.py`） |
| 当日截面 WLS Barra+行业残差 | 整盘中性化重算 |
| 加载最近一次重训模型出 Top-N 候选 | 券商下单 / 自动调度 / 用户系统 |

## 前置条件

1. **已训模型**：先跑过 `python run.py --mode lgbm --horizon 5 --save-models ...`，产物在 `results/<tag>/models/models_manifest.json`（`--save-models` 必需，否则无 manifest）。
2. **数据已就绪**：`data/raw/` 有完整历史 OHLCV / 市值 / 换手 / 财务 / 行业（首次需全量下载）。
3. **激活环境**：`.venv\Scripts\activate`。

## 入口命令

```bash
# 日常：今天出分（自动增量下载 + 因子 append + 中性化 + 模型出分）
python -m live.daily_update --model-dir results/lgbm_h5_w6-12 --top-n 30

# 指定日期 + 跳过下载（仅用已有数据出分）
python -m live.daily_update --as-of-date 2026-08-08 --no-download \
    --model-dir results/lgbm_h5_w6-12 --output candidates.md

# 冒烟（小样本）
python -m live.daily_update --model-dir results/lgbm_h5_w6-12 --sample 50 --top-n 10
```

## CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--as-of-date` | 今天 | 当日（YYYY-MM-DD）；若该日无数据自动对齐到 ≤ 该日的最近交易日） |
| `--lookback-days` | 7 | 行情增量回看日历日数（≈5 交易日） |
| `--model-dir` | 必填 | 模型目录（指向 `results/<tag>/`，需含 `models/models_manifest.json`） |
| `--top-n` | 30 | Top-N 候选数 |
| `--cap-band` | all | 市值带（当前仅 `all`；训练时用 `--cap-band` 限定训练池更稳） |
| `--output` | `<model-dir>/candidates_<date>.csv` | 输出 `.csv` 或 `.md` |
| `--factor-config` | 无 | 因子白名单 YAML/JSON（manifest 无 `feature_names` 时兜底） |
| `--horizon` | 无 | 持仓期（仅用于选 factor_config 的 `h{N}` key，不参与计算） |
| `--warmup-days` | 450 | 热身窗日历日数（覆盖 Barra 252d EWM + RSTR 240/20 + 52 周新高） |
| `--no-feature-neutralize` | 关 | 训练未用 `--feature-neutralize` 时加此开关（用 raw 因子出分） |
| `--no-download` | 关 | 跳过行情增量下载 |
| `--sample` | 0 | 仅前 N 只股票（冒烟） |
| `--prefer-model` | 无 | 多模型 ensemble 时优先选某模型（如 `lgbm`） |
| `--prefer-window` | 无 | 优先选某训练窗口（如 `6`） |

## 流程

```
Step 1  行情增量下载（复用 data.download* / compute_market_cap）
        → data/raw 的 OHLCV/市值/换手 parquet append（写前 .bak）
Step 2  加载 + 清洗 raw 面板（clean_ohlc_aligned / clean_ohlcv / mask_post_delist）
        → 与 run.py::_load_data 同口径；切热身窗 [as_of-450d, as_of]
Step 3  因子面板 append
        → 热身窗上 _iter_factor_registry_raw 重算（绕过缓存避免签名失配）
        → 取 index > 已有末日的行 concat 到 factor_panel_<hash>.parquet
        → 写前 .bak；更新 .meta.json（标记 live_appended）
        → 失败回退全量重算并 warning
Step 4  当日中性化
        → 热身窗算 Barra 9 风格 + √市值 WLS 权重
        → residualize_panel(rebalance_dates=[as_of]) 只产当日 neut 行
        → should_skip_neutralize 的因子（Barra_*/special）取 raw 当日行
Step 5  模型出分
        → 读 models_manifest.json 取最近重训日 fold（可能多个 window×model）
        → 组装 X（neut 行 + 豁免因子 raw 行）→ predict
        → 多 fold IC 加权 Z-score 平均（与训练 ensemble 同口径）
        → strict 宇宙（build_ic_tradability_mask exclude_limit=True）
        → Top-N 降序输出 .csv/.md
```

## 增量 vs 全量预期提速（定性）

| 环节 | 全量 | 增量 | 备注 |
|------|------|------|------|
| 行情下载 | 全市场全历史 | 全市场最近 5 日 | download_ohlcv 本就是 per-stock date-append |
| 因子计算 | 全历史 × 全截面 | 热身窗 ~450 日 × 全截面 | EWM/rolling 只需近期；截面 zscore per-date 自洽 |
| 中性化 | 全调仓日 WLS | 单日 WLS | residualize_panel 本就是 per-date |
| 模型出分 | — | 单截面 predict | 模型已训，只 forward |

**整体**：全量 `run.py` 跑一次 ~小时级（含训练）；日更增量 **分钟级**（下载 + 因子热身窗 + 单日中性化 + predict）。提速主要来自因子计算只覆盖热身窗而非全历史。

## 已知限制

### 因子增量
- **EWM / 长窗因子**（Barra_Beta/ResVol/Momentum、分数差分动量、52周新高）：热身窗 450 日足够覆盖（EWM halflife 63 → 450 日前权重 ≈ 2^-7 可忽略）。若改 `--warmup-days` 太小，这些因子当日值会失真。
- **截面 winsorize/zscore**：per-date 自洽，增量与全量在当日截面上一致（只要当日股票池一致）。
- **PIT 财务因子**：依赖 `financial_indicators.parquet` 的报告期 + 法定披露窗；增量只追加新季报行，历史不重算。若新季报使某历史日披露窗变化（罕见），需全量重算。
- **rolling 全样本统计因子**（如 rolling rank across time）：本仓库核心因子无此类；若新增此类因子，会自动退化到热身窗内重算（非整盘），当日值可能略偏。
- **append 后指纹**：`factor_panel_<hash>.parquet` 的 `.meta.json` 输入指纹仍是旧值，下次全量 `run.py` 会因签名失配自动重算——这是预期行为（增量是捷径，不替代全量校准）。

### 模型定位
- **需 `--save-models`**：训练时必须加此 flag，否则无 `models/models_manifest.json`。
- **最近重训日**：按 manifest 中 `date` 字段降序取最近一日；`--retrain-every 4`（周频≈月度重训）下，模型约每月更新一次，期间日更复用同一批模型。
- **lazy rolling-pool**：manifest `feature_names=None` 时用 `feature_names_union_metadata` 兜底；列序可能与某期 pool_t 不完全一致，predict 时按 union 对齐（树模型对列序敏感，此为近似）。
- **feature_neutralize 一致性**：训练用 `--feature-neutralize` 时，日更默认也做中性化（`--no-feature-neutralize` 关闭）。**口径必须与训练一致**，否则特征分布漂移。manifest 暂未存 `feature_neutralize` 字段，需人工确保开关一致。

### strict 宇宙
- 与 `mask_scores_for_backtest(score_universe='strict')` 同口径：信号日剔涨跌停 + ST + 停牌 + 次新（`IC_MIN_LISTING_DAYS=252`）+ 退市。
- **不**做 label exec-mask 门控（实盘无未来收益，无法算 forward_return）。Top-N 在 strict 信号日可买宇宙内按得分降序。
- `--cap-band` 当前仅 `all`；市值带过滤请在训练时用 `--cap-band` 限定训练池，使模型本身只学该市值带的股票。
- **as_of 对齐**：`--as-of-date` 若该日无行情（如周末/节假日/数据未到），自动对齐到数据中 ≤ 该日的最近交易日（日志会打印「当日（对齐后）= YYYY-MM-DD」）。所有下游步骤（因子 append / 中性化 / 出分 / strict mask）统一用对齐后的 as_of。

### 中性化退化
- 若某因子当日残差全 NaN（`residualize_panel` 的 `valid < min_stocks`，常见于两融/事件类因子在数据未更新日），自动退化为该因子当日 raw 行（日志「N 个残差全 NaN 退化为 raw」）。树模型对 NaN/0 鲁棒，不影响出分；但若大量因子退化，提示数据源需增量更新。

### 行情增量
- `compute_market_cap`（turnover_rate）是**全量重算**（无增量接口），但输入已是增量，pivot+ffill 较快，作为兜底。
- `download_delisted` 不追加新日期（仅加新退市股）；已退市股的新日期行情由 `download_ohlcv` 覆盖。
- 北向已停更，不加载。

## 缓存与回退

- **因子面板**：`data/processed/factor_panels/factor_panel_<hash>.parquet` + `.meta.json`（标记 `live_appended`）。写前 `.bak`。
- **Barra bundle**：`barra_bundle_<hash>/`（热身窗算时自动缓存；下次日更命中即跳过）。
- **neut 行**：当日算当日用，不落盘（避免与训练 neut 缓存键冲突——训练键含全调仓日历指纹）。
- **append 失败回退**：单个因子 append 失败（schema/列漂移）→ warning + 用热身窗结果作该因子面板（冷启动兜底）。

## 不破坏研究路径

- 增量是新入口 `live/`，不改 `run.py` / `strategies/` / `models/` 任何现有逻辑。
- append 后的 `factor_panel_*.parquet` 会被下次全量 `run.py` 因签名失配自动重算覆盖（`_signature_matches` 检查 `index_last`）。
- 若担心 append 污染研究缓存，可删 `data/processed/factor_panels/factor_panel_*.parquet` 重跑全量恢复。

## 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| `未找到模型 manifest` | 训练时未加 `--save-models`；重训加上该 flag |
| 某因子当日整列 NaN | 热身窗太短（`--warmup-days` 调大）或数据源缺失 |
| Top-N 为空 | 当日全市场涨跌停/停牌（罕见）或 `--sample` 太小 |
| `feature_names` 不匹配 | lazy rolling-pool 模型；用 `--factor-config` 指定白名单或训练时非 lazy |
| predict 报错列数 | ensemble 多模型列序不一致；用 `--prefer-model` 选单模型 |
