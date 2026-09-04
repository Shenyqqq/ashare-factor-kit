# A 股特色因子缺口调研（AKShare / 仓库对照）

> **日期**：2026-07-30（初稿调研）→ **2026-07-30 落地实现**  
> **目的**：回答「过线且 long_share 好看的因子极少 → 是否只能找新因子？免费数据是否挖尽？」  
> **范围**：对照现有 `factors/` registry + special packs + `data/raw`；高/中优先级已实现下载 + 因子注册 + 单测。  
> **衔接**：[OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) Q4/Q8（特色优先于再扫同质量价）。  
> **互补**：[ASHARE_FACTOR_DATA_GAPS.md](ASHARE_FACTOR_DATA_GAPS.md) — THS 澄清、Tushare 清单映射、免费/收费重要性分级（勿与本文工程清单重复打架）。

---

## 0. 一句话结论

| 问题 | 结论 |
|------|------|
| 是否只能去「发明」新量价公式？ | **否。** 同质 Alpha101/158/191 + 动量反转已大量进池且衰减标签多；瓶颈更像**信息集合**，不是缺几个 WQ 变体。 |
| 免费 AKShare 日频量价是否挖尽？ | **大致挖尽**（OHLCV + 换手/市值/财务季报主链）。 |
| A 股特色是否挖尽？ | **未挖尽，但「已下载却未用满」与「接口能拉但难做成 PIT 日频面板」并存。** 真·一致预期修正时间序列仍偏**付费**。 |
| 对多头（Q5 / Top100）更有用？ | 优先 **预期/修正、残差资金流、营业部质量、回购**；慎堆 **跌停/开板反转、解禁压力、质押**（偏空头或反转）。 |
| **2026-07-30 落地** | 高+中优先级因子与下载脚本已进仓库；全市场 moneyflow / 研报 / institution 回补仍需本机长时间 resume 下载（东财易限流/代理）。 |

---

## 1. 仓库已有：registry / special / raw

### 1.1 因子注册（`get_factor_registry` 语义池）

| 池 | 代表因子 | 约数量级 |
|----|----------|----------|
| 量价 | 动量/反转/波动/换手/Amihud/分数差分动量 | ~18 |
| 财务 | PB/EP/ROE/应计/增长/杠杆/规模 | ~12 |
| Alpha2 | 行业动量、特质波动、融资余额变化、大单净流入、**大单残差净流入**、机构持仓变化 | 6 |
| 技术 | BIAS/PSY/AR/BR/换手加速度/行业相对强度 | ~8 |
| 涨跌停 | 涨停强度、跌停弱势、连板、开板反转等 | 5 |
| 小市值/事件稠密+稀疏 | 股东户数、龙虎榜、解禁、高管增减持、大宗、个股两融 | ~16 |
| **A 股特色稀疏（新增）** | 评级上修、研报EPS上修/预期差、龙虎榜机构净买入/强度、回购、大宗折价席位质量、板块资金流拥挤等 | 事件稀疏 |
| **A 股特色稠密（两融）** | `融资买入占成交额_5d`、`融资净买入_5d`、`融券卖出规避_5d`、`融资余额流通市值比` | 4（IC 稠密轨） |
| 市值 alpha | 对数市值、分位、风格对齐 | 4 |
| OpenSourceAP | 资产增长、应计、偏度、行业集中度等 | ~22 |
| WQ Alpha101 / Qlib Alpha158 / GTJA Alpha191 | 大量公式化量价 | 数百候选（白名单子集进 IC） |
| special `event` | `业绩预告_超预期`、`业绩快报超预期` | 2 |
| special `size` | 同上市值 alpha 包 | 4 |
| special `sparse` | 原稀疏池 + A 股特色稀疏增量（**不含**两融截面） | 事件稀疏 |

### 1.2 `data/raw` 特色表审计

| 文件 | 覆盖（约） | 与因子关系 | 缺口判断 |
|------|------------|------------|----------|
| `lhb_detail.parquet` | 长表 ~15 万行 | 上榜次数/净买/连续上榜 + **机构席位解析** | 营业部流水仍无；yybph 为快照 |
| `lhb_yybph.parquet` / `lhb_jgstatistic.parquet` | 快照（asof） | 辅助质量分；主路径用 interpretation | 非完整历史 |
| `block_trade.parquet` | 2018→2026 | 折价/频次 + **折价×席位质量** | dzjy_yybph 快照有轻微前视风险 |
| `dzjy_yybph.parquet` | 快照 | 买方胜率 | 增量 asof 拼接可继续改善 PIT |
| `institution_holding.parquet` | 2018Q1–2026Q2（34 季，~5.7k 列） | `机构持仓变化` | 已回补；无需再当「极薄」 |
| `moneyflow_large/superlarge` | **文件仍缺失**（cache 空） | `大单净流入_5d` / `大单残差净流入_5d` | **最大工程债**；东财限流/代理敏感；可先搁置或买 DC |
| `rank_forecast.parquet` | 2018-01→2026-07，~54 万行 / ~5.1k 股 | `评级上修_20d` 等 | 长样本已齐；日常增量即可 |
| `research_report.parquet` | 2017→2026，~14 万行 / ~4.5k 股（约八成股票） | `研报EPS上修次数_20d` / `研报预期差` 等 | 未覆盖股可 resume；逐股慢 |
| `repurchase.parquet` | 全市场截面含历史公告 | `股份回购强度_60d` | 已可拉 |
| `yjbb.parquet` | 按报告期 | `业绩快报超预期` | announce 可能为修订日 |
| `sector_fund_flow.parquet` / `concept_fund_flow.parquet` | 行业/概念 | `板块资金流拥挤_5d` | 东财 hist 常挂；THS 即时兜底 |
| `northbound_*` | ≥2024-08 停更 | 默认不注册 | 不做 |

---

## 2. 已实现交付清单（注册名 / 路径 / 下载命令）

### 2.1 因子注册名

| 优先级 | 注册名 | 模块 | 轨道 |
|--------|--------|------|------|
| 高 | `大单残差净流入_5d` | `factors/factor_ashare.py` | Alpha2 稠密（需 moneyflow） |
| 高 | `融资买入占成交额_5d` / `融资净买入_5d` / `融券卖出规避_5d` / `融资余额流通市值比` | 同上 | **稠密**（非 sparse；勿方差对齐注入） |
| 高 | `评级上修_20d` | 同上 | sparse |
| 高 | `研报EPS上修次数_20d` | 同上 | sparse |
| 高 | `研报预期差` | 同上 | sparse |
| 高 | `龙虎榜机构净买入_20d` | 同上（解析 `lhb_detail.interpretation`） | sparse |
| 高 | `龙虎榜机构买入强度_20d` | 同上 | sparse |
| 中 | `股份回购强度_60d` | 同上 | sparse |
| 中 | `机构持仓变化` | `factors/factor_alpha.py`（已有；加长下载） | Alpha2 |
| 中 | `大宗折价席位质量_20d` | `factor_ashare.py` | sparse |
| 中 | `业绩快报超预期` | `factor_ashare.py` | event overlay + sparse |
| 额外 | `板块资金流拥挤_5d` | `factor_ashare.py` | sparse（探索性） |

实现入口：`get_factor_registry()` / `get_ashare_factors()`；sparse pack：`--special-factors sparse`；event：`--special-factors event`。

### 2.2 下载命令（均可 resume / `--sample`）

```bash
# 高优先级数据
python -m data.download_moneyflow --sample 200          # 全市场去掉 --sample；支持 --force
python -m data.download_rank_forecast --start 2018-01-01 --sample 20
python -m data.download_research_report --sample 50
python -m data.download_lhb_seats                       # 营业部/机构席位快照

# 中优先级
python -m data.download_repurchase
python -m data.download_institution --start-year 2018   # 回补；--sample N 调试
python -m data.download_dzjy_yybph                      # 大宗营业部排行快照
python -m data.events.download_yjbb --start-year 2018 --sample 4

# 额外：板块/概念资金流
python -m data.download_sector_fund_flow --sample 5
python -m data.download_sector_fund_flow --no-hist      # 仅 THS 即时截面兜底
```

### 2.3 产物路径

| 脚本 | 输出 |
|------|------|
| `data/download_moneyflow.py` | `data/raw/moneyflow_large.parquet`, `moneyflow_superlarge.parquet` |
| `data/download_rank_forecast.py` | `data/raw/rank_forecast.parquet` |
| `data/download_research_report.py` | `data/raw/research_report.parquet` |
| `data/download_lhb_seats.py` | `data/raw/lhb_yybph.parquet`, `lhb_jgstatistic.parquet` |
| `data/download_repurchase.py` | `data/raw/repurchase.parquet` |
| `data/download_institution.py` | `data/raw/institution_holding.parquet` |
| `data/download_dzjy_yybph.py` | `data/raw/dzjy_yybph.parquet` |
| `data/events/download_yjbb.py` | `data/raw/yjbb.parquet` |
| `data/download_sector_fund_flow.py` | `data/raw/sector_fund_flow.parquet`, `concept_fund_flow.parquet` |

单测：`tests/test_ashare_factors.py`（合成数据，不打外网）。

---

## 3. AKShare 接口主题表（原调研，保留）

图例：**日频历史** = 能否稳定拼成 date×stock 面板；**免费稳定** = 东财/同花顺公开页，易限流/改版；**PIT** = 用公告日/发布日对齐难度。

| 主题 | 接口（代表） | 日频历史？ | 免费稳定？ | PIT 难度 | 仓库现状（落地后） |
|------|--------------|------------|------------|----------|-------------------|
| **资金流** | `stock_individual_fund_flow` | 按股拉，深度存疑 | 限流重 / 代理敏感 | 低 | 下载增强 resume；**残差因子已注册**；raw 待全市场填 |
| | `stock_sector_fund_flow_hist` / THS industry | 板块级 | 东财 hist 常挂 | 低 | 下载+探索性因子；THS 兜底 |
| **龙虎榜/营业部** | `stock_lhb_detail_em` | 事件日 | 尚可 | 低 | **机构解析因子已用 interpretation** |
| | `stock_lhb_yybph_em` / `jgstatistic_em` | 窗口快照 | 尚可 | 中 | 已下载落盘；非主 PIT 路径 |
| **分析师/预期** | `stock_rank_forecast_cninfo` | 按日 | 尚可 | 低–中 | **已接入** |
| | `stock_research_report_em` | 按股 | 全市场慢 | 中 | **已接入 MVP** |
| **回购** | `stock_repurchase_em` | 公告截面含历史 | 尚可 | 中 | **已接入** |
| **业绩** | `stock_yjbb_em` | 按报告期 | 尚可 | 中（修订日风险） | **已接入 surprise** |
| **大宗席位** | `stock_dzjy_yybph` | 快照 | 尚可 | 中 | **已接入质量加权** |
| **机构持仓** | `stock_report_fund_hold` | 季报 | 尚可 | 中（法定窗 PIT） | 下载支持 2018+ resume |

---

## 4. 仍存缺口 / 诚实风险

1. **全市场 moneyflow**：脚本就绪，但东财 `push2his` 在代理环境下易失败；需本机稳定网络长时间 `--sample`→全量 resume。历史深度仍可能只有近数月～年 → 标 `emerging` 做分段 IC。  
2. **研报全市场**：逐股慢；覆盖≠一致预期权重。  
3. **yybph / dzjy_yybph 快照**：用「当前胜率」回填历史大宗有轻微 look-ahead；主信号已优先「机构专用」买方（历史字段，PIT 安全）。  
4. **yjbb 公告日**：可能是修订日；与 yjyg 对齐做 surprise 时需知悉。  
5. **institution 2018+**：须显式跑 `download_institution --start-year 2018`；未回补前 IC 仍不可信。  
6. **板块拥挤**：行业名模糊匹配 + 静态 industry_map；严格 PIT 需 `industry_map_panel` 增强（未做）。  
7. **未做**：全量 IC、质押/舆情/ETF 穿透、付费一致预期。  
8. **北向**：停更，不做。

---

## 5. 可执行决策顺序（更新）

1. **修数据债**：全市场 `download_moneyflow`；`download_institution --start-year 2018`；`download_rank_forecast --start 2018-01-01`；研报可先 `--sample` 验证。  
2. **换信息，不换模型**：`大单残差净流入_5d` + `评级上修_20d` / 龙虎榜机构因子 → 单因子 IC / long_share（对照 baseline）。  
3. **席位质量**：`龙虎榜机构*` + `大宗折价席位质量_20d` → 看 Q5 而非 Q1。  
4. **评价层**：滚动 OOS + 近年分段；避免全样本 Sharpe 自欺。  
5. 仍不够再：**付费一致预期**或**微观降频**。

---

## 6. 与 roadmap 衔接

本文落实 roadmap **Q4（A 股特色紧要）** 与 **Q8#1–2（分析师修正、大单残差）** 的数据源级清单与**代码落地**；**不否定**「同信息集下再拧 ML 收益有限」，但明确：**下一步应先喂满 raw 再跑 IC，而不是再堆反转/Alpha 变体。**
