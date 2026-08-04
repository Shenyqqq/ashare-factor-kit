# OpenSourceAP 因子覆盖与缺口分级

> **日期**：2026-08-02  
> **目的**：对照 Chen/Zimmermann **OpenSourceAP / CrossSection** 公开因子清单与本仓库注册实现，按数据成本与边际价值分级「未纳入 / 薄弱」缺口，指导下一步补因子而非再堆同质量价变体。  
> **范围**：聚焦 **OSAP 因子体系**（Acronym / Cat.Data / Cat.Economic）。A 股特色接口与 Tushare 产品线见 [ASHARE_FACTOR_DATA_GAPS.md](ASHARE_FACTOR_DATA_GAPS.md)，本文不重复粘贴该表。  
> **实现细节**（Batch 名单、Sign、IC 约定）：[OPENSOURCE_AP_FACTORS.md](OPENSOURCE_AP_FACTORS.md)。

---

## 0. 清单来源与核对口径

| 项 | 来源 / 版本范围 |
|----|----------------|
| 论文规模 | Chen & Zimmermann (2022), *Critical Finance Review*：「约 **319** firm-level characteristics」；[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626) / [openassetpricing.com](https://www.openassetpricing.com/data/)（数据页标注约 **October 2025** release） |
| 本仓库权威表 | `.tmp/OpenSourceAP/SignalDoc.csv`（本地快照）：**331** 行 = Predictor **212** + Placebo **114** + Drop **5** |
| Predictor 代码 | `.tmp/OpenSourceAP/Predictors/*.py`：**193** 个脚本（部分 ZZ* 多信号合一；与 SignalDoc Predictor 行非 1:1） |
| 上游仓库 | [OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) |

**核对方式（诚实边界）**：

- 已实现名单：与 `factors/factor_opensource_ap.py` 的 `OPENSOURCE_AP_FACTOR_NAMES` / `OPENSOURCE_AP_ACRONYM`（**30**）及 `get_factor_registry()` 挂钩逐一核对。
- 全库覆盖率：对 SignalDoc **212 Predictors** 按 `Cat.Data` 汇总；「近似覆盖」用主 registry 语义对照（如 `EP`↔`价值_EP`），**不是**对每一个 ticker/Acronym 做 Compustat 定义级复现审计。
- Placebo（114）默认不进生产补缺清单。

---

## 1. 覆盖概况

### 1.1 总览

| 口径 | 数字 | 说明 |
|------|------|------|
| OSAP 已注册（本模块） | **30** | Batch-1=6 + Batch-2+=16 + Batch-3=8；进全池 IC，**不**自动写 YAML |
| SignalDoc Predictors | **212** | 选型母集 |
| 严格覆盖率 | **30 / 212 ≈ 14.2%** | 仅计 `factor_opensource_ap` 注册名（含标注近似者） |
| 宽松覆盖率（+主池近似） | **≈ 60 / 212 ≈ 28%** | 另含动量/反转/EP/BM/杠杆/ROE/Amihud/回购等主 registry 近似 |
| Predictor 脚本文件 | 193 | 本地 `.tmp` |
| 论文全集（含变体/安慰剂语境） | ~319 | 与 SignalDoc 行集口径不同，勿直接相除 |

### 1.2 已注册 30 名 ↔ 经济类别（SignalDoc）

| 本仓库名 | Acronym | Cat.Data | Cat.Economic |
|----------|---------|----------|--------------|
| 资产增长 | AssetGrowth | Accounting | investment |
| 资产市值比 | AM | Accounting | valuation |
| 现金流市值比 | cfp | Accounting | valuation |
| 权益增长 | ChEQ | Accounting | investment |
| 盈利一致性 | EarningsConsistency | Accounting | earnings growth |
| 营收增长秩 | MeanRankRevGrowth | Accounting | sales growth |
| 盈利连增期数 | NumEarnIncrease | Accounting | earnings growth |
| 一年/五年股本扩张 | ShareIss1Y/5Y | Accounting | external financing |
| 综合股权融资 | CompEquIss | Accounting | external financing |
| 权益变化资产比 | DelEqu | Accounting | investment |
| 应计占比 | PctAcc | Accounting | accruals |
| 现金流价格波动 | VarCF | Accounting | cash flow risk |
| 毛利资产比 | GP | Accounting | profitability（**近似**） |
| 营收市值比 | SP | Accounting | valuation（**近似**） |
| 净债务市值比 | NetDebtPrice | Accounting | leverage（**近似**） |
| 综合债务融资 | CompositeDebtIssuance | Accounting | external financing（**近似**） |
| 应计资产比 | Accruals | Accounting | accruals（**CF 近似**，非 Sloan BS） |
| 经营利润权益比 | OperProf | Accounting | profitability（**近似**） |
| 外部融资资产比 | XFIN | Accounting | external financing（**近似**） |
| 资产周转变化 | ChAssetTurnover | Accounting | sales growth（**近似**） |
| 经营杠杆 | OPLeverage | Accounting | other（**近似**，缺 SGA） |
| 上市年龄 | FirmAge | Other | info proxy |
| 年龄动量 | FirmAgeMom | Price | momentum |
| 月最大收益 | MaxRet | Price | volatility |
| 收益偏度 | ReturnSkew | Price | risk |
| 协偏度 | Coskewness | Price | risk（日频 ACX） |
| 残差动量 | ResidualMomentum | Price | momentum（**CAPM 残差**） |
| 季节动量 | MomSeason | Price | other |
| 行业集中度 | Herf | Other | other（市值 HHI **近似**） |

IC 备忘（h5，2026-07-16）：OSAP 入选过线含 `月最大收益`、`综合股权融资`（详见 OPENSOURCE 文档）。Batch-3 尚未跑全量 IC。

### 1.3 按 Cat.Data：严格 vs 宽松

| Cat.Data | Predictors | 严格（OSAP 模块） | 宽松（+主池近似） | 判断 |
|----------|------------|-------------------|-------------------|------|
| Accounting | 99 | **22 (22%)** | ~35 (35%) | Batch-3 补 Accruals-CF/OperProf/XFIN 等近似；NOA/Cash/ChInv 仍卡三表 |
| Price | 45 | **6 (13%)** | ~20 (44%) | +协偏度/残差动量/季节动量；动量族主池仍大量近似 |
| Other | 12 | **2 (17%)** | ~2 | 专利/治理/供应链等多跳过 |
| Analyst | 18 | 0 | 0 | 付费一致预期主缺口 |
| Trading | 13 | 0 | ~5（Amihud/换手等） | 微观结构多数低 ROI |
| Options | 9 | 0 | 0 | A 股期权覆盖窄 |
| 13F | 8 | 0 | 0（机构持仓为 A 股另路） | 美股 13F 不适用 |
| Event | 8 | 0 | 部分事件用 yjyg/yjbb 另实现 | 美股公司事件多不适用 |

### 1.4 按经济主题（严格 OSAP，示意）

已有一定落点：`external financing` 4/12、`investment` 3/8、`valuation` 3/17、`earnings growth` 2/4。  
几乎空白且与「破同质化」相关：`investment alt` 0/10、`asset composition` 0/5、`accruals` 仅 PctAcc、`profitability` 仅近似 GP、`R&D` 0/5、`earnings forecast` 0/8、`liquidity`/`short sale` 等。

**数据现实**：`data/raw/financial_indicators.parquet` 仅约 10 个财务字段（`total_assets` / `eps` / `ocf_ps` / `bvps` / `debt_ratio` / 增长与利润率等），**无** ACT/CHE/INVT/PPE/DLTT 等 Compustat 级科目 → 大量 Accounting Predictors 卡在「免费三表未接」而非「逻辑不值得」。

---

## 2. 三类缺口

标签：

| 标签 | 含义 |
|------|------|
| `低成本可补` | 已有 raw / `clean_ret` / 市值，或 AKShare **免费三表**扩字段后可算 |
| `高成本有价值` | 需 Tushare VIP / 期权·融券·专利等难搞数据；与 OSAP 逻辑匹配且能补主池盲区 |
| `低重要性` | OSAP 有，但与现池高度同质、A 股不适用、或边际低于工程成本 |

### 2.1 `低成本可补`

| 因子或子类（代表 Acronym） | 逻辑 | 数据需求 | 与 registry 对照 |
|----------------------------|------|----------|------------------|
| **资产负债表投资/净经营资产** — NOA, dNoa, GrLTNOA, InvestPPEInv, ChInv, InvGrowth, DelCOA, ChNWC, ChNNCOA | 投资异象 / 净经营资产变化（HXZ 投资族） | 免费三表：AT, CHE, DLTT, ACT, INVT, PPE… + PIT | **仍跳过**（缺 CHE/INVT/DLTT…）；仅有 `资产增长` 近邻 |
| **应计精细化** — Accruals (Sloan BS), PctTotAcc, TotalAccruals, AbnormalAccruals* | 盈余质量；比 PctAcc 更贴原文 | ACT/CHE/LCT/DLC/DP 或应计分解 | ✅ `应计资产比`(Accruals **CF 近似**)；Sloan **BS 版仍跳过**；另有 PctAcc |
| **现金与有形性** — Cash, tang, OPLeverage | 资产结构 / 经营杠杆 | CHE, AT；有形资产科目 | ✅ `经营杠杆`(缺 SGA **近似**)；Cash/tang **仍跳过**（缺 CHE） |
| **外部融资现金流** — XFIN, NetDebtFinance, NetEquityFinance, DebtIssuance | 融资供给异象（股权已有 CompEquIss） | CF 表融资项或 Δ债务/Δ股本精细拆分 | ✅ `外部融资资产比`(XFIN **余额变化近似**)；严格 CF 融资项仍缺 |
| **盈利能力会计版** — OperProf, CBOperProf, roaq, CashProd, PS(Piotroski), OScore | 质量/综合财务打分 | 利润表+部分 BS；部分可用现有 ROE/毛利率**弱近似**后升级 | ✅ `经营利润权益比`(OperProf **近似**)；CBOperProf/PS/OScore/CashProd 仍缺 |
| **周转与销售质量** — ChAssetTurnover, GrSaleToGrInv, GrSaleToGrOverhead, OrderBacklog* | 效率与过度投资 | 营收、存货、开销；订单 backlog A 股稀疏 | ✅ `资产周转变化`(**近似** sales/AT)；GrSaleToGrInv 等仍缺 INVT |
| **价量残差族（Price，低工程）** — Coskewness, ResidualMomentum*, MomSeason / MomOffSeason（短窗） | 高阶矩 / 残差动量 / 季节动量 | 仅 `clean_ret` + 市场收益 | ✅ `协偏度` / `残差动量`(CAPM) / `季节动量`；MomOffSeason 未做 |
| **估值日频补强** — 用已有 `pe_ttm`/`pb` 对齐 BM/EP 日频刷新 | 估值时效 | 已有东财日频估值 parquet | `价值_EP`/`价值_PB` 偏季报；非新 OSAP 名，属同逻辑升级 |
| **雇佣** — hire | 用工增长≈投资 | 员工人数（部分免费财报/AKShare） | **仍跳过**（无员工人数面板） |

> 优先顺序建议：**先扩免费三表字段 → 再实现 NOA/ChInv/Accruals-BS/XFIN**；价量残差族可并行、不挡财务债。

### 2.2 `高成本有价值`

| 因子或子类（代表 Acronym） | 逻辑 | 数据需求 | 与 registry / A 股对照 |
|----------------------------|------|----------|------------------------|
| **分析师预期与修正** — FEPS, AnalystRevision, ForecastDispersion, ConsRecomm, ChangeInRecommendation, Up/DownRecomm, ChNAnalyst, EarningsForecastDisparity, fgr5yrLag, sfe, AOP/PredictedFE… | OSAP Analyst 整类（18）；预期差与修正是美股最稳截面之一 | Tushare 盈利预测 / 一致预期正式库（高积分）；免费研报为**伪一致** | 已有 `评级上修` / `研报EPS上修` / `研报预期差`（AKShare）；**缺** OSAP 级时间序列一致预期 |
| **卖空约束** — ShortInterest, IO_ShortInterest, 部分 short-sale 族 | 约束越强、负向信号越尖 | 融券余额/券源（明细已有两融，**券源深度**常付费） | `融资余额变化` 偏多头杠杆；卖空约束 OSAP 名未做 |
| **期权隐含风险** — SmileSlope, CPVolSpread, OptionVolume*, dVolCall/Put, RIVolSpread, skew1 | 期权风险溢价 / 偏度交易 | 个股期权 IV 曲面（覆盖窄、贵） | 无；仅当认真做期权卫星时值得 |
| **机构博弈细化** — Activism*, RIO_*, DelBreadth（13F 族） | 聪明钱持仓变化与注意力 | 美股 13F；A 股需基金/北向/机构持仓高质量面板 | `机构持仓变化` 另路；非 OSAP 13F 复现 |
| **信用与治理** — CredRatDG, Governance | 评级下调 / 治理质量 | 评级历史、治理评分商用库 | 无 OSAP 实现；可用 ST/违规免费事件弱替代 |
| **广告/组织资本/品牌** — BrandInvest, OrgCap, AdExp, GrAdExp | 无形资产投资异象 | 销售费用/广告精细科目 + 资本化假设 | 三表有销售费用时可**部分降级为低成本**；完整 OrgCap 仍重 |

与 [ASHARE_FACTOR_DATA_GAPS.md](ASHARE_FACTOR_DATA_GAPS.md) 的交叉点：OSAP **Analyst** ≈ 该文「正式盈利预测/一致预期」`收费·高重要性`；本文不展开 Tushare 积分档细节。

### 2.3 `低重要性`（可不急）

| 因子或子类 | 理由 | registry 对照 |
|------------|------|----------------|
| **标准动量/反转长清单** — Mom12m, Mom6m, Mom*Season*YrPlus, STreversal, MRreversal, LRreversal, IntMom… | 与现有 `动量_*` / `反转_*` / `分数差分动量` 高度同质；再注册 OSAP 名增益低 | 主池已覆盖核心期限结构 |
| **Size / Price** | 已有 `对数市值` / 价格信息寓于量价因子 | size special pack |
| **基础估值/杠杆/ROE** — EP, BM, BMdec, BookLeverage, Leverage, RoE, roaq | 主财务池已有 | `价值_*` / `杠杆` / `质量_ROE*` |
| **美股公司事件** — Spinoff, ExchSwitch, DivInit/Omit, RDIPO, AgeIPO… | 制度不同；A 股用解禁/增发/回购等特色事件更划算 | A 股事件/sparse 包 |
| **专利与罪恶产业** — PatentsRD, CitationsRD, sinAlgo | A 股专利引文面板弱；行业剔除可用申万 | 无 |
| **客户-供应商动量** — iomom_cust/supp, CustomerMomentum | 供应链图谱工程重、覆盖差 | 无 |
| **Trading 微观结构大部分** — zerotrade*, BidAskSpread, ProbInformedTrading… | 日频辅助选股边际低；Amihud/换手已近似流动性 | `Amihud_20d` / 换手族 |
| **Options 全类（无期权策略时）** | 标的覆盖与权限不匹配个人主策略 | — |
| **SignalDoc Placebo（114）** | 原文即非主预测或间接证据 | 不实现 |

---

## 3. 已覆盖占比（汇总）

| 维度 | 占比 |
|------|------|
| Predictors × OSAP 模块严格 | **14.2%**（30/212） |
| Predictors × 严格+主池语义近似 | **~28%**（~60/212） |
| Accounting 严格 | **22%**（22/99） |
| Price 严格 | **13%**（6/45）；宽松因动量族升至 ~44% |
| Analyst / Options / 13F / Event | 严格 **0%**（Analyst 有 A 股免费伪预期另路） |

「已覆盖」不等于「定义一致」：GP/SP/NetDebtPrice/Herf/CompositeDebtIssuance 等在实现中已标 **近似**。

---

## 4. 优先行动（5 条）

1. **扩免费财务三表科目（最大杠杆）**  
   下载并 PIT 对齐 AT 以外的 CHE/ACT/INVT/PPE/DLTT/融资现金流等，写入 raw；否则 NOA/ChInv/Accruals-BS 只能继续空转。与 A 股文档「三表明细仍浅」同一债，但交付物服务 **OSAP Accounting**。

2. **Batch-3 已落地（现有字段近似 + 价量）**  
   `应计资产比` / `经营利润权益比` / `外部融资资产比` / `资产周转变化` / `经营杠杆` / `协偏度` / `残差动量` / `季节动量` 已注册；**勿**硬塞生产 YAML，先小样本 IC。

3. **三表扩字段后补严格会计名**  
   CHE/INVT/ACT/DLTT/融资 CF 到位后再做 NOA/dNoa/ChInv/Cash/Sloan-BS；当前禁止假实现。

4. **Batch-3 冒烟 IC（workers=1）**  
   检验相对 `收益偏度`/`月最大收益`/动量族是否有独立 IC；非全量扫库。

5. **付费决策绑定 OSAP Analyst，而非再买一层量价**  
   若东财/AKShare 研报伪预期不够用，优先 Tushare **正式盈利预测/修正**以覆盖 FEPS/Revision/Dispersion 逻辑；期权 IV、13F 复现、专利库对当前「辅助人工选股」默认后置。细节见 [ASHARE_FACTOR_DATA_GAPS.md](ASHARE_FACTOR_DATA_GAPS.md) §4.2。

---

## 5. 文档分工

| 文档 | 职责 |
|------|------|
| [OPENSOURCE_AP_FACTORS.md](OPENSOURCE_AP_FACTORS.md) | 下载路径、已实现 Batch 表、Sign、IC/YAML 约定 |
| **本文** | 对 212 Predictors 的覆盖率 + 三类缺口 + 行动优先级 |
| [ASHARE_FACTOR_DATA_GAPS.md](ASHARE_FACTOR_DATA_GAPS.md) | A 股特色数据 × 免费/Tushare 分级（非 OSAP 清单） |
| [ASHARE_FACTOR_GAPS.md](ASHARE_FACTOR_GAPS.md) | A 股特色工程落地清单 |

更新原则：新增 OSAP 注册名或三表字段时，同步改 §1 覆盖数字与 §2 对应行；付费分析师数据就绪后把 Analyst 行从「高成本」迁到「已覆盖（A 股映射）」。
