# OpenSourceAP CrossSection Predictors — 下载、分类与实现

来源：[OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection)（Chen / Zimmermann 复现库）  
SignalDoc 权威表：`.tmp/OpenSourceAP/SignalDoc.csv`（Acronym / Cat.Data / Sign / Predictability…）  
覆盖率与缺口分级（低成本 / 高成本有价值 / 低重要性）：[OSAP_FACTOR_COVERAGE.md](OSAP_FACTOR_COVERAGE.md)

## 下载路径

| 项 | 路径 |
|----|------|
| Predictors | `.tmp/OpenSourceAP/Predictors/` |
| SignalDoc | `.tmp/OpenSourceAP/SignalDoc.csv` |
| Browser HTML | `.tmp/OpenSourceAP/SignalDoc-Browser.html` |
| 文件清单 | `.tmp/OpenSourceAP/predictor_list.txt` |

- Predictor `.py` 文件数：**193**
- SignalDoc 行数：331（Predictor≈212 / Placebo≈114 / Drop≈5）

---

## 分类总览（面向本仓库选型）

状态图例：**✅** 已实现并注册 · **🟡** 需扩展 BS/CF 下载 · **🔴** 跳过（无 A 股数据 / 已被近似覆盖）

### 按 Cat.Data（Predictor，SignalDoc≈212）

| Cat.Data | 计数 | 策略 | 代表 |
|----------|------|------|------|
| Accounting | 99 | ✅ Batch-1/2/3 现有字段已落地；严格 BS 🟡 | ✅ Accruals-CF/OperProf/XFIN… · 🟡 NOA/ChInv/Cash… |
| Price | 45 | 多数已被仓库动量/反转/波动覆盖；缺口 ✅ | ✅ MaxRet/Skew/Coskew/ResMom/MomSeason |
| Other | 12 | 个案 | ✅ FirmAge / Herf · 🔴 PatentsRD/Governance… |
| Event | 8 | 多数 🔴（美股事件） | 🔴 DivInit/Spinoff/ExchSwitch… |
| Analyst | 18 | 🔴 | FEPS / ConsRecomm / AnalystRevision… |
| Trading | 13 | 🔴（微观结构） | Illiquidity / zerotrade* / BidAskSpread… |
| Options | 9 | 🔴 | SmileSlope / CPVolSpread… |
| 13F | 8 | 🔴 | Activism* / RIO_* / IO_ShortInterest |

### 按本仓库落地状态

| 状态 | 含义 | 本轮数量 |
|------|------|----------|
| ✅ implemented | 已注册入 `get_factor_registry()`，进 IC 全池；**不**自动写入生产 YAML | **30**（Batch-1=6 + Batch-2+=16 + Batch-3=8） |
| 🟡 expand BS/CF | 需 ACT/CHE/INVT/PPE/DLTT 等科目后再做 | NOA/Cash/ChInv/Sloan-BS 等仍缺 |
| 🔴 skip | Analyst/Options/13F/Trading/美股事件；或仓库已有近似不必重写 | 整类跳过 + EP/BM/Leverage 等近似 |

完整 Acronym / Sign 以 **SignalDoc.csv** 为准。

---

## 已实现因子（Batch-1 + Batch-2 + Batch-3）

实现模块：`factors/factor_opensource_ap.py`  
单测：`tests/test_opensource_ap_factors.py`  
**未**强制写入 `config/factor_configs.yaml`（须经 IC 筛选后再进白名单）。

### Batch-1（会计）

| English | 中文名 | Sign→处理 | 数据 |
|---------|--------|-----------|------|
| AssetGrowth | `资产增长` | -1→取负 | `total_assets` YoY |
| AM | `资产市值比` | +1 | AT / mcap |
| cfp | `现金流市值比` | +1 | ocf_ps / price |
| ChEQ | `权益增长` | -1→取负 | bvps YoY |
| EarningsConsistency | `盈利一致性` | +1 | eps |
| MeanRankRevGrowth | `营收增长秩` | +1 | revenue_growth 秩 |

### Batch-2+（现有字段，无新表下载）

| English | 中文名 | Sign→处理 | 数据 / 备注 |
|---------|--------|-----------|-------------|
| NumEarnIncrease | `盈利连增期数` | +1 | eps YoY 连增季度数（≤8） |
| ShareIss1Y | `一年股本扩张` | -1→取负 | `share_change` 总股本 |
| ShareIss5Y | `五年股本扩张` | -1→取负 | 同上 |
| CompEquIss | `综合股权融资` | -1→取负 | log(ΔME)−BH_ret(~60m) |
| FirmAge | `上市年龄` | -1→取负 | `list_date`（或缺则首次有效价） |
| FirmAgeMom | `年龄动量` | +1 | 最年轻五分位 × 6m mom |
| DelEqu | `权益变化资产比` | -1→取负 | Δ(bvps×shares)/avg(AT) |
| PctAcc | `应计占比` | -1→取负 | (eps−ocf_ps)/\|eps\|；并修 `质量_应计项目` eps fallback |
| VarCF | `现金流价格波动` | -1→取负 | ocf/P 滚动方差 |
| GP | `毛利资产比` | +1 | **近似** gpm×sales_est/AT |
| SP | `营收市值比` | +1 | **近似** (\|eps\|/\|npm\|)/price |
| NetDebtPrice | `净债务市值比` | -1→取负 | **近似** debt_ratio×AT/ME |
| CompositeDebtIssuance | `综合债务融资` | -1→取负 | **近似** log(债务₅ᵧ增长) |
| MaxRet | `月最大收益` | -1→取负 | 21d max(clean_ret) |
| ReturnSkew | `收益偏度` | -1→取负 | 21d skew(clean_ret) |
| Herf | `行业集中度` | -1→取负 | **近似** 行业 mcap HHI（非销售额） |

### Batch-3（现有字段 + 价量；2026-08）

| English | 中文名 | Sign→处理 | 数据 / 备注 |
|---------|--------|-----------|-------------|
| Accruals | `应计资产比` | -1→取负 | **CF 近似** (eps−ocf)×shares/avg(AT)；非 Sloan BS |
| OperProf | `经营利润权益比` | +1 | **近似** gpm×sales/(bvps×shares)；剔最小市值三分位 |
| XFIN | `外部融资资产比` | -1→取负 | **近似** (Δshares×P+Δdebt)/avg(AT) |
| ChAssetTurnover | `资产周转变化` | +1 | **近似** Δ(sales_est/AT) |
| OPLeverage | `经营杠杆` | +1 | **近似** (1−gpm)×sales/AT（缺 SGA） |
| Coskewness | `协偏度` | -1→取负 | 日频 CoskewACX：`clean_ret`+中证全指 |
| ResidualMomentum | `残差动量` | +1 | **CAPM 残差**近似（无免费 FF3） |
| MomSeason | `季节动量` | +1 | 月收益滞后 23/35/47/59 |

近似因子均在 docstring 标注「近似」，非 Compustat 原定义。

---

## 🟡 扩展 BS/CF 后高价值（未实现 / 跳过假实现）

| English | 缺字段（当前 `financial_indicators` 无） |
|---------|------------------------------------------|
| NOA / dNoa | CHE, DLTT, MIB, DC…（仅有 AT/bvps/debt_ratio） |
| DelCOA / ChInv / InvGrowth | ACT, CHE, INVT |
| InvestPPEInv | PPEGT, INVT |
| Cash / CashProd / tang | CHE（及有形资产科目） |
| CBOperProf / OScore / PS | 多 WC/损益科目 |
| NetDebtFinance / NetEquityFinance（严格 CF） | sstk/prstkc/dltis/dltr/dlcch；已有 XFIN **余额变化近似** |
| Accruals（Sloan BS 版） | ACT/CHE/LCT/DLC/DP；已有 CF 版 `应计资产比` |
| hire | 员工人数 |

## 🔴 明确跳过

| 类别 | 例子 |
|------|------|
| Analyst | AnalystRevision, ConsRecomm, FEPS, ForecastDispersion, CredRatDG… |
| Options | SmileSlope, CPVolSpread, OptionVolume1/2, dVolCall/Put… |
| 13F | Activism1/2, RIO_*, IO_ShortInterest, DelBreadth |
| Trading | Illiquidity, zerotrade*, BidAskSpread, DolVol, ShareVol… |
| Event（美股特有） | Spinoff, ExchSwitch, DivInit/Omit, RDIPO… |
| Other（专利/治理等） | PatentsRD, CitationsRD, Governance, sinAlgo, iomom_* |

## 已有近似（不必重写，标 🔴 重复实现）

| OpenSourceAP | 本仓库近似 |
|--------------|------------|
| Accruals / PctAcc | `质量_应计项目`（已修 eps fallback）+ 新 `应计占比` |
| EP | `价值_EP` |
| BM / BMdec | `价值_PB` |
| BookLeverage / Leverage | `杠杆` |
| roaq / RoE | `质量_ROE` |
| Size / Price | `对数市值` / 价量类已有 |
| Mom* / *reversal | 仓库动量/反转因子族 |

---

## IC / 配置产出约定

```powershell
# 全池 IC（含 OpenSourceAP 新名；workers=1）
python -m research.ic_analysis_v2 --period 5 --barra --save --use-fdr --t-threshold 2.5 --gram-schmidt --workers 1 --barra-workers 1 --fresh
```

| 产物 | 路径 |
|------|------|
| IC 摘要 | `research/output/ic_summary_h5.csv` |
| Barra pure IC | `research/output/ic_barra_pure_h5.csv` |
| 筛选 JSON（含 `factors` + `factors_orth`） | `research/output/selected_factors_h5.json` |
| 常规 top-N YAML | `config/factor_configs_h5_top31_pure_ic.yaml`（镜像 `research/output/`） |
| **GS dynamic YAML** | `config/factor_configs_h5_gs_dynamic.yaml`（`factors_orth`，供 dynamic） |
| 写配置脚本 | `research/output/_write_h5_configs_from_ic.py` |

2026-07-16 h5 全池 IC（`--barra --use-fdr --t-threshold 2.5 --gram-schmidt`，~1.9h）：筛选 **31** 因子；GS orth **8** 因子。OpenSourceAP 入选：`月最大收益`、`综合股权融资`（orth 含 `月最大收益`）。

勿在 IC 选出之前把新因子硬塞进旧生产 `factor_configs.yaml`。

## 不做的事

- 全量移植 193 个 predictor
- 未经验证写入生产 YAML
- 组合优化实验（本任务范围外）
