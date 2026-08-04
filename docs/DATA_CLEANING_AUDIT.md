# 数据清洗流程审计报告

> 审计日期：2026-07-02（初版）  
> **状态更新：2026-07-29** — 数据架构审计 P0/P1/P2 已落地（见文末「2026-07-29 修订」）。  
> 审计范围：A 股多因子 ML 选股框架 `quant_trading` 的数据清洗全链路  
> 关联文档：[PIT_AUDIT.md](PIT_AUDIT.md)、[data/DATA_UPDATE.md](../data/DATA_UPDATE.md)、AGENTS.md 注意事项 15–22

---

## 一、当前清洗流程概览

### 1.1 数据流图（标注清洗节点）

```
AKShare API
   │
   ├─ stock_info_sh/sz_name_code + stock_info_sh/sz_delist
   │     └─► _normalize_meta_table                [清洗节点 A1：列名归一/去重列]
   │           └─► get_stock_list(include_delisted=True)
   │                 └─► filter_universe          [清洗节点 A2：剔除北交所 8开头]
   │                       └─► universe/stock_list.parquet (含 list_date/delist_date/is_st_current)
   │
   ├─ stock_zh_a_hist(adjust="hfq"/"")
   │     └─► raw/{open,close,high,low}_hfq.parquet + volume/amount.parquet + prices_raw.parquet
   │           │
   │           └─► clean_prices()                 [清洗节点 B1：去重/零负价/±100%/孤岛刺针/5天ffill]
   │                 └─► clean_ohlcv()            [清洗节点 B2：涨跌停 mask + clean_ret]
   │                       └─► (clean_ret, masks) 透传到 factors / ic / backtest
   │
   ├─ stock_financial_analysis_indicator
   │     └─► raw/financial_indicators.parquet (trade_date=报告期, code, roe, bvps, ...)
   │           └─► clean_financial()              [清洗节点 C1：去重/ROE±300%/bvps≤0/资产≤0/未来日期]
   │                 └─► pit_pivot_ffill()        [清洗节点 C2：PIT 法定披露窗口平移 + ffill]
   │                       └─► factors/factor.py 财务因子
   │
   ├─ stock_industry_clf_hist_sw
   │     └─► raw/industry_map_panel.parquet (code, effective_date, sw_l1, sw_l2, end_date)
   │           │   [清洗节点 D1：去重(code,effective_date) + end_date=下段-1day + dropna(start_date/symbol/industry_code)]
   │           └─► load_industry_as_of(date)      [清洗节点 D2：PIT as-of join]
   │
   ├─ stock_margin_detail_sse/szse / stock_individual_fund_flow / stock_hsgt_individual_em / stock_report_fund_hold
   │     └─► raw/{margin_balance, moneyflow_*, northbound_*, institution_holding}.parquet
   │         [清洗节点 E1：仅 pd.to_numeric(errors="coerce") + dropna；无独立清洗函数]
   │
   ├─ stock_zh_index_daily_em(sh000300 / sh000985)
   │     └─► raw/{csi300, csi_all}.parquet        [清洗节点 F1：仅按 start/end 日期切片，无异常值检测]
   │
   └─ stock_yjyg_em
         └─► raw/yjyg.parquet                     [清洗节点 G1：列名固定 + to_numeric + 去重(code,report_date,indicator)]
               └─► factor_event.py 用 announce_date 正确 PIT


因子出口统一处理                                  [清洗节点 H1：factors/factor.py:47-67]
   └─► _normalize = winsorize(1%) + cross_sectional_zscore(clip=3σ)
         └─► 因子方向函数内取反，输出"越高越好"
```

### 1.2 清洗步骤清单（按 A–F 分类）

| 类别 | 步骤 | 位置 | 状态 |
|------|------|------|------|
| **A 股票池** | 列名归一 + 去重列 | `data/download.py:106-142` | ✅ |
| A | 合并退市股 | `data/download.py:317-340`, `fetch_delisted_stocks:208-266` | ✅ |
| A | 剔除北交所（8 开头） | `data/download.py:343-372` | ✅ |
| A | 元数据 list_date/delist_date/is_st_current | `data/download.py:103, 150-205` | ✅ |
| A | ST 时间序列化 | `backtest/execution.py:219-270` | ⚠️ 保守实现，当前 ST 股在所有回测日均标记 ST |
| **B OHLCV** | 后复权 / 不复权双份存储 | `data/download.py:377-497` | ✅ |
| B | clean_prices 去重/零负价/±100%/孤岛刺针/5天ffill | `data/clean.py:25-67` | ✅ |
| B | 涨跌停 mask（主板 9.5% / 创科 19.5%） | `data/clean.py:72-137` | ✅ |
| B | clean_ret（涨跌停日 return=NaN） | `data/clean.py:140-176` | ✅ |
| B | 一字板识别（open==close） | `data/clean.py:115-120` | ✅ |
| B | 成交量/成交额清洗 | — | ❌ 缺失 |
| B | 停牌处理（>5 天长停牌） | `data/clean.py:60` ffill(limit=5) | ⚠️ 仅短停牌填充 |
| **C 财务** | clean_financial 去重/极值/未来日期 | `data/clean.py:181-228` | ⚠️ 仅 ROE/bvps/total_assets 三项 |
| C | PIT 法定披露窗口对齐 | `utils/pit_align.py:27-146` | ✅ |
| C | 财务其他指标异常值检测（eps/gpm/debt_ratio...） | — | ❌ 缺失 |
| C | 单位统一 | `data/download.py:547-549` to_numeric | ⚠️ 隐式 |
| **D 行业** | PIT 长表构建 + 去重 | `data/industry/download_industry.py:97-128` | ✅ |
| D | PIT as-of join | `data/industry/download_industry.py:271-306` | ✅ |
| D | 未分类股票处理 | `download_industry.py:108` dropna | ⚠️ 直接丢弃 |
| D | Fallback 静态 map（无 PIT） | `download_industry.py:186-240` | ⚠️ 退化路径 |
| **E 资金流/持仓** | to_numeric + dropna | 各 download_*.py | ⚠️ 最小清洗 |
| E | 机构持仓 PIT 对齐 | `factors/factor_alpha.py:453` pit_reindex_ffill | ✅ |
| E | 北向/资金流异常值检测 | — | ❌ 缺失 |
| E | 北向 2024 起停止披露的断点处理 | — | ❌ 缺失 |
| **F 因子预处理** | winsorize(1%) 截面缩尾 | `factors/factor.py:56-62` | ✅ |
| F | cross_sectional_zscore(clip=3σ) | `factors/factor.py:47-53` | ✅ |
| F | 因子方向统一（函数内取反） | `factors/factor.py` 各因子 | ✅ |
| F | 市场特征时序滚动 z-score | `strategies/market_state.py` | ✅ |
| F | Barra 因子 winsorize | `factors/barra_risk.py` | ❌ 仅 zscore 不 winsorize |

---

## 二、各清洗步骤详细评估

### A. 股票池构建

**当前实现**（`data/download.py:98-372`）：
- `_fetch_stock_list_with_metadata()` 调 SH/SZ 官方接口（主板A/B + 科创板 + 创业板 + B 股），统一列名 `code/name/list_date/delist_date`，强制去重列防 concat InvalidIndexError。
- `fetch_delisted_stocks()` 优先调 `stock_info_sh_delist`/`stock_info_sz_delist`，失败回退到 `_scan_delisted_from_raw()`（从已下载 parquet 反推曾出现过的代码）。
- `filter_universe()` 仅剔除 8 开头（北交所），保留退市/ST，元数据列缺失时补 NaT/False。
- ST 时间序列由 `backtest/execution.py:219-270 build_st_schedule()` 构造，但**当前实现是保守版**：当前 ST 股在所有回测日期均标记为 ST，未接入真实 ST 历史接口。

**正确性评估**：✅ 退市股 + list_date/delist_date 元数据已落地，幸存者偏差修复完成（PIT_AUDIT M4 已实施）。

**潜在问题**：
1. **ST 时间序列失真**（`execution.py:256-257`）：`pd.DataFrame(True, index=dates, columns=st_codes)` 对当前 ST 股在全部历史日期标 ST。摘帽股会被错误全程剔除；曾 ST 但当前摘帽的股票则完全不被剔除。注释里写了 hook 但未接入 `ak.stock_zh_a_st_em` 等真实历史接口。
2. **北交所剔除依赖代码前缀**：8 开头是新三板/北交所，但 AKShare `stock_zh_a_hist` 对 8 开头代码行为未在下载层显式过滤——`filter_universe` 在元数据层过滤，但若用户直接用 raw parquet 列就会漏过滤。
3. **B 股未剔除**：主板 B 股（沪 900xxx / 深 200xxx）以美元/港元交易，与 A 股因子不可比，当前未在 `filter_universe` 剔除。

### B. OHLCV 清洗

**当前实现**（`data/clean.py:25-176`）：
- `clean_prices()`：去重日期 → 零负价 NaN → 单日 |ret|>100% NaN → 孤岛刺针（价/前后两日均值比 >3 或 <1/3）NaN → `ffill(limit=5)` 短停牌填充。
- `make_limit_mask()`：主板 `threshold_main=0.095`，创/科创板 `threshold_star=0.195`，用 `high==close & ret≥thresh*0.98` 识别涨停（避开后复权累积误差），一字板 `open==close`。
- `clean_ohlcv()`：`clean_ret = close.pct_change()`，再把 `any_limit` 日置 NaN，返回 `(clean_ret, masks)`。

**正确性评估**：✅ 涨跌停清洗设计严谨，clean_ret 透传到所有量价因子（factor.py、factor_alpha101.py、barra_risk.py 均已切换，见 AGENTS.md 注意事项 1）。一字板识别正确。

**潜在问题**：
1. **forward_return 未用 clean_ret**（`research/ic/forward_return.py:7-20`）：`build_forward_return()` 用原始 `prices` 和 `open_` 计算 `close[t+N]/open[t+1]-1`，**涨跌停日收益仍进入标签**。这导致 IC/ML 标签与因子口径不一致——因子侧屏蔽了涨跌停日，但标签侧没有。一字板买入日（次日开盘=涨停价）实际无法买入，forward_return 却假设能买入。
2. **成交量/成交额无清洗**（`data/download.py:471`）：`df[ak_col].astype(float)` 直接转 float，未检测 0 成交量异常、未检成交量突增 100 倍等异常值。换手率/Amihud 因子直接消费这些数据。
3. **长停牌 >5 天不填充**（`data/clean.py:60`）：`ffill(limit=5)` 之后长停牌保留 NaN，rolling 因子用 `min_periods` 自动跳过，但 `factor_momentum` 用 `(1+clean_ret).rolling().apply(nanprod)` 时，长停牌股票的动量会被NaN稀释。回测侧 `return_engine.py:84 rets[susp]=0` 把停牌日收益置 0（NAV 平），与因子侧 NaN 处理口径不一致。
4. **复权方式**：后复权（hfq）用于回测和大部分因子，不复权（raw）用于 PB 计算——这是正确的设计（PB 用未复权价 / 每股净资产）。但 `factor_value_pb` 等因子的分子 bvps 是 PIT ffill 的报告期值，分母 prices_raw 是日频值，**两边日期对齐靠 ffill，未校验 bvps 的报告期与 prices_raw 的日期是否在同一个 PIT 窗口内**。
5. **±100% 阈值对创/科创板偏松**：`clean_prices:47` 用 `abs(daily_ret) <= 1.0`，但创/科创板涨跌停 20%，正常交易日不可能超过 ±20%（后复权累积下分红除权可能短暂超 20% 但远不到 100%）。阈值过松会放过数据错误。
6. **孤岛刺针 3 倍阈值对低价股不敏感**：3 元股票刺针到 9 元才触发，但 30 元股票刺针到 90 元同样触发——绝对比例阈值对高价股过严、对低价股过松。

### C. 财务数据清洗

**当前实现**（`data/clean.py:181-228` + `utils/pit_align.py`）：
- `clean_financial()`：去重 `(trade_date, code)` → ROE |x|>300% NaN → bvps≤0 NaN → total_assets≤0 NaN → 删 `trade_date > today` 的未来日期行。
- `pit_pivot_ffill()` / `pit_reindex_ffill()`：按报告期月份查 `_DISCLOSURE_WINDOWS`（Q1/Q3 +30，半年报 +60，年报 +90），把报告期平移到"可用日下界"再 pivot + ffill 到日频。

**正确性评估**：✅ PIT 对齐已落地（PIT_AUDIT M1 已实施），消除财务因子 look-ahead bias。

**潜在问题**：
1. **异常值检测仅覆盖 3 个指标**：eps / gross_profit_margin / operating_cashflow / debt_ratio / net_profit_growth / revenue_growth / net_profit_margin 这 7 个下载字段（`download.py:562-574`）均未做异常值检测。极端 debt_ratio=9999% 或 net_profit_growth=1e6% 会直接进入因子。
2. **单位统一隐式依赖 to_numeric**：`download.py:547-549` 用 `pd.to_numeric(errors="coerce")`，若 AKShare 某次接口返回字符串带单位（如 "1.2亿元"），会静默转 NaN 而非报错。
3. **法定披露窗口偏实务**：`pit_align.py` 用 +30/+60/+30/+90 天（年报较法定 +120 略松仍偏保守），实际公告日常早于或接近该下界；过松会引入 look-ahead，过紧损失时效。
4. **非季末月份归到最近过去季末**（`pit_align.py:54-57`）：若接口返回 2024-04-15 这种中间日期，会被归到 03-31 + 45 天。但 `_fetch_one_financial` 下载的是季报，理论上不会出现非季末日期，依赖接口契约。
5. **财报修订/重述未处理**：同一报告期若公司发布修订版，当前 `drop_duplicates(keep="last")` 保留最后一条，但未记录修订标记，无法回溯。
6. **未来日期检测用 `pd.Timestamp.today()`**（`clean.py:217`）：今天是 2026-07-02，但若回测 2018-2024 段，2025-2026 的报告期会被当作"未来"删除——这在增量更新场景下是对的，但在纯历史回测时若数据已含未来期会丢失。

### D. 行业数据清洗

**当前实现**（`data/industry/download_industry.py`）：
- `build_industry_panel()`：从 `stock_industry_clf_hist_sw()` 拿历史变更记录，dropna `(start_date, symbol, industry_code)`，按 `(code, effective_date)` 去重，`end_date = 下一段 effective_date - 1 day`，最后一段 NaT。
- 申万 2021 分类：6 位 industry_code，前 4 位 = sw_l2，前 2 位 = sw_l1。
- `load_industry_as_of(date)`：取 `effective_date <= date` 且 (`end_date` NaT 或 `≥ date`) 的记录，同 code 取 effective_date 最晚一条。

**正确性评估**：✅ PIT 行业面板已落地（PIT_AUDIT M2 已实施）。

**潜在问题**：
1. **未分类股票直接丢弃**（`download_industry.py:108`）：`dropna(subset=["start_date", "symbol", "industry_code"])` 把无行业记录的股票从面板中删除，下游 `load_industry_as_of` 查不到这些股票，Barra 行业哑变量会缺失它们——可能引入"行业未知"偏差。
2. **Fallback 路径无 PIT**（`download_industry.py:186-240`）：`stock_industry_clf_hist_sw` 失败时退化为 `_try_current_from_ths`，只产静态 `industry_map.parquet`，无 `industry_map_panel.parquet`。下游若依赖 PIT 面板会静默退化为静态行业，引入未来信息。
3. **申万分类版本切换未记录**：申万 2014/2021 两次大调整，`stock_industry_clf_hist_sw` 返回的是 2021 版回填历史，2014-2020 段实际用的是 2021 版分类，与当时市场认知的行业不一致。

### E. 资金流 / 持仓数据

**当前实现**：每个下载脚本各自做最小清洗：
- `download_margin.py:82`：`pd.to_numeric(combined["balance"], errors="coerce")` + 按日期 concat 去重。
- `download_moneyflow.py:91,100`：`pd.to_numeric(errors="coerce")` + 按代码 concat 去重。
- `download_northbound.py:74-75`：`pd.to_numeric(errors="coerce")` + 按代码 concat 去重。
- `download_institution.py:77`：`pd.to_numeric(errors="coerce")` + 按季报期 concat 去重。
- 机构持仓在因子侧用 `pit_reindex_ffill` 做 PIT 对齐（`factor_alpha.py:453`）。

**正确性评估**：⚠️ 仅最小清洗，无独立清洗函数。

**潜在问题**：
1. **无异常值检测**：融资余额突增 100 倍、北向持股量为负、大单净流入超市值等异常均未检测。
2. **北向资金 2024 起停止披露**：2024 年 8 月起沪深股通停止披露每日持股明细，`northbound_holding.parquet` 在 2024-08 后会有大段 NaN，因子 `factor_northbound_change` 在该段会失效，但未在因子层做自动屏蔽。
3. **资金流数据未在 run.py 清洗**（`run.py:137-140`）：`margin/moneyflow/northbound/institution` 直接 `_load_opt` 读 parquet 传入因子，未走 `clean_*` 系列。机构持仓靠 `pit_reindex_ffill` 在因子侧兜底，其他三个无 PIT 对齐——但它们是日频数据，本身无 look-ahead 风险，仅缺异常值检测。
4. **margin 按日期下载可能漏股**（`download_margin.py:62-63`）：单日两市接口若某只股票当日无融资余额（新上市未纳入两融标的），不会出现在记录里，宽表该格为 NaN——这本身正确（无融资余额 = 0 杠杆），但因子 `factor_margin_change` 用 `pct_change(window)` 时 NaN 会被当作缺失，可能误判为"融资余额下降"。

### F. 因子计算前预处理

**当前实现**（`factors/factor.py:47-67`）：
- `winsorize(df, pct=0.01)`：截面 1%/99% 分位 clip。
- `cross_sectional_zscore(df, clip=3.0)`：截面 z-score，裁剪 ±3σ。
- `_normalize = winsorize + zscore`：所有因子出口统一调用。
- 市场/HMM regime 特征用时序滚动 z-score，不做截面标准化（AGENTS.md 已规定）。
- 因子方向：函数内取反，输出"越高越好"，`FACTOR_WEIGHTS` 仅 linear 模式用。

**正确性评估**：✅ 截面标准化口径统一，与 IC 纯 IC 同口径（开 `--feature-neutralize` 后）。

**潜在问题**：
1. **Barra 因子不做 winsorize**（FACTORS_REVIEW.md:166）：`barra_risk.py` 的 9 个 Barra 因子只做截面 zscore，不做 winsorize。Barra 因子用作 IC 残差化控制变量，极值未缩尾可能影响 OLS 拟合稳定性。
2. **winsorize 截面样本量未守卫**：`factor.py:56-62` 对每行做 `quantile(0.01)`，若某截面有效股票 < 100，1% 分位等于单只股票值，winsorize 退化为无操作。`IC_MIN_LISTING_DAYS=252` 在 IC 侧守卫，但因子计算侧 `get_factor_registry` 未传 listing_dates 过滤。
3. **`_normalize` 对 NaN 不透明**：`scipy_zscore(row.dropna())` 后 `pd.Series(..., index=row.dropna().index)`，NaN 股票不在结果中——下游 `compute_composite_factor` 合成时若某因子全 NaN，等权平均会因索引不齐产生意外 NaN。

---

## 三、缺失清洗步骤清单（按重要性排序）

| # | 缺失项 | 影响 | 难度 | 修复建议 |
|---|--------|------|------|----------|
| 1 | **forward_return 未屏蔽涨跌停日** | 高 | 中 | `forward_return.py:7-20` 在涨跌停日把买入价/卖出价置 NaN，或用 clean_ret 复合 |
| 2 | **ST 真实历史时间序列未接入** | 高 | 中 | `execution.py:219-270` 接入 `ak.stock_zh_a_st_em` 构建 (date, code) → is_st 真实表 |
| 3 | **成交量/成交额无异常值检测** | 高 | 低 | 加 `clean_volume()`：0 成交、突增 100 倍、负值检测 |
| 4 | **财务其他 7 个指标无异常值检测** | 中 | 低 | `clean_financial` 扩展：debt_ratio∈[0,100]、gpm∈[-100,100]、growth 截断 |
| 5 | **Barra 因子不做 winsorize** | 中 | 低 | `barra_risk.py` 出口加 winsorize(1%) |
| 6 | **数据质量监控报告缺失** | 中 | 中 | 新增 `data/quality_report.py` 输出缺失率/异常率/覆盖率 |
| 7 | **未分类股票被 dropna 丢弃** | 中 | 低 | `download_industry.py:108` 改为填 "UNKNOWN" 而非删除 |
| 8 | **北向资金 2024-08 后断点未屏蔽** | 中 | 低 | `factor_northbound_change` 加日期守卫，2024-08 后返回 NaN |
| 9 | **资金流数据无独立清洗函数** | 中 | 中 | 抽象 `clean_aux_panel()` 覆盖 margin/moneyflow/northbound |
| 10 | **B 股未剔除** | 中 | 低 | `filter_universe` 加 `~code.startswith(("900","200"))` |
| 11 | **复权方式未在文档显式标注用途** | 低 | 低 | docs 标注：hfq→回测+动量，raw→PB，qfq→未使用 |
| 12 | **±100% 涨跌幅阈值对创/科创板过松** | 低 | 低 | `clean_prices:47` 改为按代码前缀分档（主板 ±20%，创科 ±30%） |
| 13 | **财报修订/重述未记录** | 低 | 高 | 引入修订标记字段，依赖接口能力 |
| 14 | **交易日历未独立校验** | 低 | 中 | 引入 `trading_calendar.parquet`，所有数据 reindex 到统一日历 |
| 15 | **多源数据交叉验证缺失** | 低 | 高 | AKShare + Tushare 双源校验，超出本次审计范围 |
| 16 | **数据快照版本管理缺失** | 低 | 中 | raw parquet 加日期后缀，每次下载保留历史版本 |

---

## 四、优先级 TODO 清单

### P0（必须，影响结果正确性）

#### P0-1：forward_return 屏蔽涨跌停日
- **位置**：`research/ic/forward_return.py:7-20`，`research/ic_analysis.py:77-95`
- **问题**：`build_forward_return()` 用原始 prices/open_，涨跌停日（尤其一字板次日开盘=涨停价）实际无法买入，但标签假设能买入。IC/ML 标签与因子口径（已屏蔽涨跌停）不一致。
- **改动量**：~20 行
- **难度**：中
- **预期收益**：消除涨跌停日标签噪声，IC 估计更准；ML 标签与因子同口径，模型不再学习"涨跌停日的虚假收益"。
- **修复建议**：传入 `masks`，把 `limit_up_open[signal_date+1]` 的买入价置 NaN；或用 `clean_ret` 复合计算 forward_return。

#### P0-2：ST 真实历史时间序列接入
- **位置**：`backtest/execution.py:219-270`（`build_st_schedule` 当前保守实现）
- **问题**：当前 ST 股在所有回测日均标 ST，摘帽股被错误全程剔除，曾 ST 现摘帽股完全不被剔除。
- **改动量**：~80 行（新增加载 + 解析）
- **难度**：中
- **预期收益**：消除 ST 过滤的时间维度偏差，回测股票池更准。
- **修复建议**：新增 `data/download_st_history.py` 调 `ak.stock_zh_a_st_em` 或 `stock_zh_a_sts_em`，构建 `(date, code) → is_st` 宽表，`build_st_schedule` 改为读取该表。

#### P0-3：成交量/成交额异常值检测
- **位置**：`data/clean.py`（新增 `clean_volume`），`run.py:132-136` 调用处
- **问题**：volume/amount 直接 `astype(float)` 未清洗，0 成交、负值、突增 100 倍均未检测，污染换手率/Amihud 因子。
- **改动量**：~30 行
- **难度**：低
- **预期收益**：换手率/Amihud 因子不再受数据错误驱动。
- **修复建议**：新增 `clean_volume(volume)`：负值 NaN、0 成交保留（停牌正常）、突增 100 倍日志告警。

### P1（建议，提升稳健性）

#### P1-1：财务其他指标异常值检测
- **位置**：`data/clean.py:181-228` `clean_financial`
- **问题**：仅 ROE/bvps/total_assets 检测，eps/gpm/debt_ratio/growth 等 7 项未检测。
- **改动量**：~25 行
- **难度**：低
- **预期收益**：财务因子抗极端值更稳。
- **修复建议**：加 `debt_ratio ∈ [0, 100]`、`gpm ∈ [-100, 100]`、`growth ∈ [-1000, 1000]`（百分比单位）截断。

#### P1-2：Barra 因子 winsorize
- **位置**：`factors/barra_risk.py` 出口
- **问题**：Barra 因子仅 zscore 不 winsorize，极值未缩尾影响 OLS 残差化稳定性。
- **改动量**：~10 行
- **难度**：低
- **预期收益**：Barra 残差化标签/IC 更稳。
- **修复建议**：Barra 因子出口统一 `winsorize(1%) + zscore`，与其他因子同口径。

#### P1-3：数据质量监控报告
- **位置**：新增 `data/quality_report.py`
- **问题**：当前无自动化数据质量报告，缺失率/异常率/覆盖率不可见。
- **改动量**：~150 行
- **难度**：中
- **预期收益**：每次下载后输出质量报告，及早发现数据源接口变更/字段缺失。
- **修复建议**：脚本读所有 raw parquet，输出每只股票的：日期覆盖率、NaN 率、零值率、极端值率、与上一版本的 diff 摘要。

#### P1-4：未分类股票保留为 "UNKNOWN"
- **位置**：`data/industry/download_industry.py:108`
- **问题**：`dropna(subset=["industry_code"])` 把无行业记录的股票从面板删除，下游 Barra 行业哑变量缺失它们。
- **改动量**：~5 行
- **难度**：低
- **预期收益**：避免"行业未知"股票被静默剔除。
- **修复建议**：`industry_code` 缺失填 `"UNKNOWN"`，保留股票记录。

#### P1-5：资金流数据独立清洗函数
- **位置**：新增 `data/clean.py::clean_aux_panel`
- **问题**：margin/moneyflow/northbound 仅 `to_numeric`，无异常值检测、无 PIT 对齐校验。
- **改动量**：~40 行 + run.py 调用
- **难度**：中
- **预期收益**：资金流因子抗异常值更稳。
- **修复建议**：抽象 `clean_aux_panel(df, name)`：负值 NaN、inf NaN、突增日志。

### P2（可选，提升效率/可监控性）

#### P2-1：B 股剔除
- **位置**：`data/download.py:343-372` `filter_universe`
- **改动量**：1 行；难度：低
- **建议**：加 `~code.startswith(("900", "200"))`。

#### P2-2：北向资金日期守卫
- **位置**：`factors/factor_alpha.py` `factor_northbound_change`
- **改动量**：~5 行；难度：低
- **建议**：2024-08-19 后返回 NaN 并日志告警。

#### P2-3：交易日历独立校验
- **位置**：新增 `data/trading_calendar.py`
- **改动量**：~80 行；难度：中
- **建议**：缓存 `trading_calendar.parquet`，所有数据 reindex 到统一日历，缺失日期日志告警。

#### P2-4：±100% 阈值分档
- **位置**：`data/clean.py:47`
- **改动量**：~8 行；难度：低
- **建议**：主板 ±20%，创/科 ±30%（留余量）。

#### P2-5：数据快照版本管理
- **位置**：`data/download.py` 各 `_save_wide`
- **改动量**：~30 行；难度：中
- **建议**：raw parquet 加日期后缀 `_YYYYMMDD`，保留历史版本供回溯。

---

## 五、数据质量监控建议

### 5.1 是否需要数据质量报告脚本

**强烈建议新增**。当前清洗只做"发现问题→置 NaN"，但**不输出问题摘要**。专业机构通常每次数据更新后跑质量报告，及早发现：
- AKShare 接口字段变更（如 `净资产收益率(%)` 改名）
- 某只股票数据突然全 NaN（接口限流/退市）
- 极端值数量异常飙升（数据源 bug）

### 5.2 关键监控指标

| 指标 | 计算方式 | 告警阈值 |
|------|----------|----------|
| 日期覆盖率 | 每股有效日期数 / 总交易日 | < 80% 告警 |
| NaN 率 | 每股 NaN 格子数 / 总格子 | > 20% 告警 |
| 零值率 | 每股 volume=0 占比 | > 10% 告警（可能停牌） |
| 极端收益率率 | 每股 \|ret\|>20% 占比 | > 1% 告警 |
| 财务字段缺失率 | 每字段 NaN 占比 | > 30% 告警 |
| 行业覆盖率 | 有行业记录的股票数 / 总股票 | < 95% 告警 |
| 退市股数量 | delist_date 非空股票数 | 与上版 diff>50 告警 |
| 接口返回列名 | 实际列名 vs 期望列名 | 不一致告警 |

### 5.3 实现建议

新增 `data/quality_report.py`，每次 `download_main` 后自动运行，输出 markdown 报告到 `data/quality_report_YYYYMMDD.md`，并在缺失率异常时 `logger.error` 中断流程（可选 `--strict` 模式）。

---

## 六、审计结论

### 整体评估

当前数据清洗流程**在 PIT 防穿越和幸存者偏差两个核心维度已达到专业水准**（PIT_AUDIT M1-M4 全部落地），clean_ret 设计严谨、Alpha101 全因子切换 clean_ret、行业 PIT 面板完整——这些是许多开源 A 股框架未做的工作。

主要短板集中在：
1. **标签侧与因子侧口径不一致**（P0-1：forward_return 未屏蔽涨跌停）——这是最影响结果正确性的缺口。
2. **ST 时间序列保守实现**（P0-2）——方向对但精度不足。
3. **辅助数据（volume/资金流）清洗薄弱**（P0-3, P1-5）——只做了主数据的清洗，辅助数据仅 to_numeric。
4. **无数据质量监控**（P1-3）——问题不可见。

### 建议优先级

- **立即修复**：P0-1（forward_return 屏蔽涨跌停）——影响所有 IC/ML 结果。
- **本季度修复**：P0-2（ST 历史）、P0-3（volume 清洗）、P1-1（财务异常值）、P1-2（Barra winsorize）。
- **下季度修复**：P1-3（质量报告）、P1-4（未分类股票）、P1-5（资金流清洗）。
- **长期规划**：P2 系列按需推进。

### 验证

- 报告中所有文件:行号引用均来自本次 Read 实际读取的源码内容。
- 严格只读审计，未修改任何代码文件。
- 本次审计与 [PIT_AUDIT.md](PIT_AUDIT.md) 互补：PIT_AUDIT 聚焦 PIT/幸存者偏差（已修复），本报告聚焦清洗步骤完整性与辅助数据清洗。

---

## 七、2026-07-29 数据架构审计修订（P0/P1/P2）

| 项 | 状态 | 说明 |
|----|------|------|
| P0 prices_raw vs hfq 覆盖 | ✅ | 根因：hfq/raw 分次增量、无对齐校验。`download_ohlcv(peer_last=)` + `report_raw_hfq_coverage()` |
| P0 download_shares 永跳过 | ✅ | 改为 `refresh_stale_days` 增量刷新 / `--force-refresh` |
| P0 compute_market_cap | ✅ | 补 raw + 刷新股本后重跑；验收覆盖率 |
| P1 财务 ann_date | ⚠️ 法定窗 | 主接口无可靠 first-ann_date；yjbb「最新」=修订日不接入；有 `ann_date` 列则用；日志标明近似 |
| P1 行业静态 fallback | ✅ | `--barra` 严格要求 panel；`--allow-static-industry` 才退化 |
| P1 沪市 ST | ⚠️ 标注 | `source=sh_bj_current_st_conservative_fallback` + list_date 收紧起点；无公开沪市带日期接口 |
| P1 stock_value_em 校验 | ✅ | `python -m data.validate_market_cap`（不替换主链） |
| P2 download_market_cap | ✅ | Deprecated + DeprecationWarning |
| P2 北向停更 | ✅ | 默认不加载；因子停更后置 NaN + WARNING |
| P2 volume×100 | ✅ | 文档/代码钉死：volume=手，`VOLUME_MUL=100` |
| P2 本文档过时结论 | ✅ | 本节修订；原「立即修复」清单部分已由其他任务落地，以 AGENTS.md 为准 |
