# PIT 财务数据 + 幸存者偏差 审计报告

> 审计日期：2026-07-01
> 审计范围：A 股多因子 ML 选股框架 `quant_trading`
> 审计方法：基于源码逐文件阅读，未运行回测验证

---

## 一、当前数据流程图（标注 PIT 风险点）

```
AKShare API
   │
   ├── ak.stock_info_a_code_name()                ──► 【幸存者偏差 S1】只返回当前在市股票
   │      │
   │      └─► filter_universe(~ST & ~退)          ──► 【幸存者偏差 S2】剔除当前 ST/退市
   │              │
   │              └─► universe/stock_list.parquet ──► 下游 stock_names / ST 过滤的唯一来源
   │
   ├── ak.stock_zh_a_hist(adjust="hfq"/"")        ──► prices_hfq / prices_raw / OHLCV  （无 PIT 风险）
   │
   ├── ak.stock_financial_analysis_indicator      ──► 【PIT 风险 F1】"日期"=报告期，无 ann_date
   │      │
   │      └─► raw/financial_indicators.parquet    (trade_date, code, roe, bvps, ...)
   │              │
   │              ├─► clean_financial()           ──► 仅删除 trade_date > today 的行（无 PIT 修复）
   │              │
   │              └─► factors/factor.py::_pivot_financial()
   │                     │
   │                     └─► pivot(trade_date) ─► reindex(prices.index, method="ffill")
   │                             │
   │                             └─► 【PIT 风险 F2】用报告期填充到该日之后所有日子
   │                                     ↓
   │                            factor_value_pb / _ep / quality_roe / _roe_chg /
   │                            quality_gpm / _gpm_chg / quality_accrual /
   │                            size / leverage                  ──► 【PIT 风险 F3】所有财务因子
   │
   ├── ak.stock_industry_clf_hist_sw()            ──► 【PIT 风险 I1】仅取 .last()，丢弃历史行业
   │              │
   │              └─► raw/industry_map.parquet (单行/股, 无时间维度)
   │
   └── ak.stock_yjyg_em (业绩预告)                ──► ✓ 正确使用 announce_date（唯一 PIT-correct 范例）
```

---

## 二、PIT 风险点清单

### F1：财务下载接口本身无公告日字段
- **位置**：`data/download.py:243-292` (`_fetch_one_financial`)
- **问题**：`ak.stock_financial_analysis_indicator(symbol=code)` 返回的 `日期` 列在 `data/download.py:304` 被映射为 `trade_date`，但该 `日期` 实际是**报告期**（如 2024-03-31），而非公告日。接口本身不提供 `ann_date`。
- **下游影响**：所有依赖 `financial_indicators.parquet` 的因子都继承此问题。

### F2：财务数据清洗未补 ann_date
- **位置**：`data/clean.py:181-228` (`clean_financial`)
- **问题**：清洗只处理去重 / 极值 / 未来日期 (`trade_date > today`)，未引入任何公告日概念，也未按"报告期 + 法定披露窗口"估算可用日期下界。

### F3：所有财务因子用 `reindex(method="ffill")` 直接对齐报告期到日频
| 文件:行号 | 函数 | 风险等级 |
|----------|------|---------|
| `factors/factor.py:69-73` | `_pivot_financial`（核心工具函数） | **高** |
| `factors/factor.py:198-209` | `factor_value_pb`（1/PB） | **高** |
| `factors/factor.py:212-226` | `factor_value_ep`（EP） | **高** |
| `factors/factor.py:229-236` | `factor_quality_roe` | **高** |
| `factors/factor.py:239-251` | `factor_quality_roe_chg` | **高** |
| `factors/factor.py:254-265` | `factor_quality_gpm` | **中**（毛利率列可能缺失） |
| `factors/factor.py:268-282` | `factor_quality_gpm_chg` | **中** |
| `factors/factor.py:285-305` | `factor_quality_accrual` | **高** |
| `factors/factor.py:308-316` | `factor_size` | **高** |
| `factors/factor.py:319-334` | `factor_leverage` | **高** |
| `factors/barra_risk.py:43-48` | `_pivot_ffill`（Barra 工具函数） | **高** |
| `factors/barra_risk.py:53-62` | `barra_size` | **高** |
| `factors/barra_risk.py:131-138` | `barra_value` | **高** |
| `factors/barra_risk.py:148-155` | `barra_leverage` | **高** |
| `factors/barra_risk.py:158-173` | `barra_growth` | **高** |
| `factors/factor_alpha.py:249-251` | `factor_institution_change`（机构持仓季报同样 ffill） | **中** |

### F4：行业数据用"当前行业"回填历史
- **位置**：`data/industry/download_industry.py:58-63`
- **代码**：
  ```python
  latest = (df.sort_values("start_date").groupby("symbol").last().reset_index()
            [["symbol", "industry_code"]])
  ```
- **问题**：`stock_industry_clf_hist_sw()` 本身返回带 `start_date` 的历史行业变更记录，但代码主动 `.last()` 只保留最新一条，丢弃时间维度。回测早期某股票会被分到当前行业而非彼时行业（行业切换发生在合并/重组/申万分类调整时会产生偏差）。
- **下游影响**：`factor_industry_momentum`（`factors/factor_alpha.py:29-89`）、`industry_trainer.py`、Barra 行业哑变量。

### ✓ PIT-correct 范例（无需修改）
- **位置**：`factors/factor_event.py:81-156`（`factor_yjyg`）
- **做法**：业绩预告数据带 `announce_date`，因子在 `announce_date` 当日放入信号，`ffill(limit=window)` 控制持续期。这是项目内**唯一**正确处理 PIT 的财务类因子。

### 非 PIT 风险的 ffill（澄清）
- `data/clean.py:60` `prices.ffill(limit=5)`：价格短缺口填充，模拟停牌复牌，无 PIT 风险。
- `models/trainer.py:882`、`strategies/market_state.py:206/256/308`：市场状态 regime 时序 ffill，属时序特征自身延迟发布，无 PIT 风险。
- `models/analyzer.py:126`：基准指数对齐 nav 索引，无 PIT 风险。
- `factors/factor_alpha101.py`：经 Grep 确认**不使用任何财务数据**，纯量价因子，无 PIT 风险。

### PIT 风险严重性估算
A 股法定披露窗口：
- 一季报：截止 4/30
- 半年报：截止 8/31
- 三季报：截止 10/31
- 年报：截止次年 4/30

以 2024Q1 为例：报告期 2024-03-31，实际公告日分布在 2024-04-01 ~ 2024-04-30。当前代码会把 2024Q1 数据 ffill 到 2024-04-01 ~ 2024-04-29 这段时间，即使用了**未公告**的数据。平均 look-ahead 约 **15-25 个交易日**（季报口径），年报口径最坏可达 **30-60 个交易日**。

---

## 三、幸存者偏差点清单

### S1：股票列表只含当前在市股票
- **位置**：`data/download.py:100-105` (`get_stock_list`)
- **代码**：
  ```python
  df = ak.stock_info_a_code_name()   # 仅返回当前在市 A 股
  ```
- **问题**：`stock_info_a_code_name` 接口返回的是**当前**仍上市的股票。历史上已退市、被吸收合并的股票不在列表中。回测从 2018-01-01 开始，但 2018 至今退市的股票（约 200+ 只）从回测股票池中完全消失。

### S2：filter_universe 进一步剔除 ST/退
- **位置**：`data/download.py:108-113` (`filter_universe`)
- **代码**：
  ```python
  mask = ~stock_list["name"].str.contains("ST|退", na=False)
  ```
- **问题**：当前名字含 "ST" 或 "退" 的股票全部被剔除。这等于把"**当前是 ST**"或"**当前处于退市整理期**"的股票从全部历史中抹除。
  - 历史上 ST 后又摘帽的股票会被错误剔除。
  - 退市股在退市前的全部历史数据被丢弃。

### S3：stock_list.parquet 是下游唯一的股票元数据来源
- **位置**：`data/download.py:399`（写入）；`data/download_northbound.py:112`、`data/download_moneyflow.py:135`、`research/ic/universe.py:10-20`（读取）
- **问题**：所有下游模块都从 `universe/stock_list.parquet` 获取股票列表，因此 S1/S2 的偏差被传播到北向、资金流、IC 分析、回测全链路。

### S4：ST 过滤无时间维度
- **位置**：`backtest/execution.py:169-174` (`infer_st_codes`)、`backtest/quantile.py:120`、`research/ic/universe.py:55-60`
- **代码**：
  ```python
  mask = stock_names.astype(str).str.contains("ST", case=False, na=False)
  return set(stock_names.index[mask].astype(str))
  ```
- **问题**：ST 集合是**当前**名字推断的静态集合，应用到所有历史日期。某股票 2020 年被 ST、2022 年摘帽，会被错误地从 2020、2021、2022 全部排除（或反向：当前未 ST 但历史曾被 ST 的股票不会被排除）。这是 ST 处理的 PIT 风险变种。

### S5：财务/资金流数据按当前股票池对齐
- **位置**：`data/download.py:401-422`、`data/download_northbound.py:112`、`data/download_moneyflow.py:135`
- **问题**：OHLCV / 财务 / 北向 / 资金流全部按当前 universe 下载，退市股在 raw parquet 中**根本没有列**。即使下游想恢复退市股，原始数据也已丢失。

---

## 四、修复建议清单

### 短期低成本修复

#### M1：财务因子按"法定披露截止日"做 PIT 对齐（**必须** / **中难度**）
- **做法**：在 `_pivot_financial` / `_pivot_ffill` 内部，把 pivot 索引由 `trade_date` 改为 `earliest_available_date = trade_date + 法定披露窗口`：
  - 03-31 报告 → 04-30 可用
  - 06-30 报告 → 08-31 可用
  - 09-30 报告 → 10-31 可用
  - 12-31 报告 → 次年 04-30 可用
- **优点**：纯保守估计（实际公告日 ≤ 法定截止日），不会引入新的 look-ahead；不需要修改下载接口。
- **缺点**：会用稍微"晚一些"的数据，损失少量最新季报的信息时效（但换得 PIT 正确性）。
- **影响范围**：`factors/factor.py:69-73`、`factors/barra_risk.py:43-48`，以及它们的 11 处调用点。

#### M2：行业数据保留历史记录（**建议** / **低难度**）
- **做法**：`data/industry/download_industry.py:58-63` 不再 `.last()`，改为保存全部 `(start_date, code, industry_code)` 记录；下游因子用 `as-of` join 按日期取当时行业。
- **影响范围**：`download_industry.py`、`factor_alpha.py::factor_industry_momentum`、`models/industry_trainer.py`。

#### M3：ST 状态时间序列化（**建议** / **中难度**）
- **做法**：下载 ST 历史变更记录（AKShare 有 `stock_zh_a_st_em` 等接口），构建 `st_state.parquet` (index=date, columns=stock, bool)；`infer_st_codes` 替换为按日期查询的 `is_st(date, code)`。
- **影响范围**：`backtest/execution.py`、`backtest/quantile.py:120`、`research/ic/universe.py:55-60`。

#### M4：保留退市股的 OHLCV（**必须** / **中难度**）
- **做法**：`get_stock_list` 改用包含历史退市股的接口（如 `ak.stock_info_sh_name_code` + `stock_info_sz_name_code` 取并集，或从 `stock_zh_a_hist` 反查曾出现过的代码）；`filter_universe` 不再按当前名字含 "退" 剔除，改为按"退市日期 > 回测当前日期"动态判断。
- **注意**：用户明确要求"不要修改数据下载接口"——M4 触及下载入口，建议先在报告中提出，待用户确认后再实施。

#### M5：股票元数据扩展（**建议** / **低难度**）
- **做法**：`stock_list.parquet` 增加 `list_date` / `delist_date` / `name_history` 列，供下游动态过滤。
- **依赖**：M4 提供完整股票列表。

### 长期重构

#### L1：建立 PIT 财务数据库（**建议** / **高难度**）
- **做法**：raw 层引入长表 schema `(code, report_date, ann_date, indicator_name, value)`，下载时优先用带 `ann_date` 的接口（如 Tushare `fina_indicator` 提供公告日；或 AKShare 的 `stock_financial_report_sina` 系列含 `更新日期`）。所有因子 pivot 改为以 `ann_date` 为可用时点。
- **优点**：完全消除 PIT 风险。
- **成本**：需要重新下载全量财务数据、重写 `_pivot_financial` 工具链、回归测试所有财务因子 IC。

#### L2：建立退市/ST 状态时间序列（**建议** / **高难度**）
- **做法**：维护 `universe/listing_history.parquet` (code, list_date, delist_date, name_changes) 和 `universe/st_history.parquet` (date, code, is_st)。回测/IC/选股全链路改为按日期查询。
- **成本**：需要新数据源 + 全链路改造。

#### L3：引入 PIT 校验测试（**可选** / **低难度**）
- **做法**：在 `tests/` 中加单测，构造一个"未来才公告"的财务记录，断言因子在公告日前对该股票输出 NaN。
- **优点**：防止未来回归。

---

## 五、本次审计的修复实施决策

**结论：本次只出报告，不改代码。**

理由（按用户给定的约束推导）：
1. **M1（必须项）** 涉及 `factors/factor.py:69-73`、`factors/barra_risk.py:43-48` 工具函数及其 11 处调用点的协同修改，不属于"纯增量、低风险、低成本"范畴。一旦改错 pivot 索引，所有财务因子 IC 会发生不可预期变化，需要完整回归测试。
2. **M4（必须项）** 涉及修改下载接口，用户明确要求"不要修改数据下载接口"。
3. **M2/M3** 单独实施意义有限——若财务因子仍带 PIT 风险，行业/ST 修复无法独立消除回测偏差。
4. **M5** 依赖 M4 提供完整股票列表，不可独立实施。

综合判断：本次审计的修复建议**没有满足"纯增量、低风险、低成本"且能独立生效**的项。按用户指示"如果不确定，只写报告不改代码"，本次仅输出本审计报告。

建议下一步由用户决定优先级后，单独发起 M1（财务 PIT 对齐）和 M2（行业历史化）两个改造任务，分别做完整回归测试。

---

## 六、验证

- `python -c "import data.download; import data.clean; print('imports OK')"` → ✓ `imports OK`
- 报告中所有文件:行号引用均来自本次 Read 实际读取的源码内容（见审计过程的工具调用记录）。
- `factors/factor_alpha101.py` 经 Grep 确认不含 `financial`/`trade_date`/`reindex.*method`，纯量价因子，未列入风险清单。

---

## 七、2026-07-29 状态更新

初版（2026-07-01）为只读审计。后续已落地：

- **M1 财务 PIT**：`utils/pit_align.py` 法定披露窗口（**近似**，非真实公告日；Q1/Q3=+30、半年报=+60、年报=+90）。AKShare 主接口仍无可靠 first-ann_date；`stock_yjbb_em`「最新公告日期」=修订日，**不接入**主链。长表若带 `ann_date` 则优先。
- **M2 行业 PIT**：`industry_map_panel.parquet`；IC `--barra` **严格要求** panel（`--allow-static-industry` 才允许静态退化）。
- **M4 退市股 + ST**：深交所精确 `sz_name_change`；沪/北 `sh_bj_current_st_conservative_fallback`（自 list_date 保守标 ST）。
- 详见 AGENTS.md 注意事项 15–22。
