# factors/ 模块深度审阅报告

> 审阅日期：2026-07-02
> 审阅范围：`factors/*.py` 共 8 个文件
> 审阅性质：只读审阅，未修改任何代码
> 审阅目标：评估因子库完整性与实现质量，识别现代 A 股 alpha 因子缺口

---

## 1. 当前因子库概览

### 1.1 文件结构

| 文件 | 行数 | 职责 |
|------|------|------|
| `factor.py` | 760 | 主入口 `get_factor_registry()`；基础量价/财务/市场 regime |
| `factor_alpha.py` | 299 | 第二批 alpha：行业动量/IVOL/融资/大单/北向/机构 |
| `factor_alpha101.py` | 291 | WorldQuant Alpha101 精选 10 个 |
| `factor_technical.py` | 218 | BIAS/PSY/ARBR/换手率变体/行业相对强度 |
| `factor_limit.py` | 104 | 涨跌停信号 5 个 |
| `barra_risk.py` | 321 | Barra 9 风格（仅用于 IC 残差控制，不直接选股） |
| `factor_cache.py` | 100 | 因子面板磁盘缓存 |
| `factor_event.py` | 184 | 业绩预告因子（**未接入 registry**） |

### 1.2 因子总数与分布

注册表 `get_factor_registry()` 返回的因子总数（最大模式下，Barra 不计入）：

| 类别 | 因子数 | 因子名 |
|------|--------|--------|
| 动量/反转 | 11 | 动量_3/10/20/40/60/120d、动量_skip、反转_3/5/10/20d |
| 波动率 | 3 | 波动率_20/60d、特质波动率_60d |
| 流动性/换手 | 5 | 换手率_20d、Amihud_20d、振幅_20d、换手率加速度、换手率行业中性_20d |
| 价值 | 2 | 价值_PB、价值_EP |
| 质量 | 5 | 质量_ROE、ROE变化、毛利率、毛利率变化、应计项目 |
| 规模/杠杆 | 2 | 规模、杠杆 |
| 行业动量 | 2 | 行业动量_20d、行业相对强度_20d |
| 技术指标 | 5 | BIAS_5/20d、PSY_12d、AR_26d、BR_26d |
| 资金流 | 4 | 融资余额变化_20d、大单净流入_5d、北向持股变化_20d、机构持仓变化 |
| Alpha101 | 10 | WQ_001/002/006/007/012/028/034/053/061/101 |
| 涨跌停信号 | 5 | 涨停强度_20d、跌停弱势_20d、连板数、涨跌停净强度_20d、开板反转_5d |
| 分数差分 | 1 | 分数差分动量_20d |
| 52周新高 | 1 | 52周新高 |
| 市场 regime | 5 | 市场动量_20/60/120d、市场波动率_20d、市场MA偏离_60d、HMM_强势/弱势概率 |
| **合计 registry** | **≈56** | （实际可用条数随数据完整性变化） |
| Barra 风格（独立） | 9 | Size/NonlinSize/Beta/Momentum/ResVol/Value/Liquidity/Leverage/Growth |
| 业绩预告（未接入） | 1 | factor_yjyg |

### 1.3 数据流

```
data/raw/*.parquet
   ├── prices_hfq / open_hfq / high_hfq / low_hfq
   ├── prices_raw（PB/EP 用）
   ├── volume / amount
   ├── financial_indicators.parquet（PIT 对齐 via utils/pit_align）
   ├── csi_all.parquet（市场代理）
   ├── industry_map.parquet（静态，注意：未用 PIT panel）
   ├── margin_balance / moneyflow_large / northbound_holding / institution_holding
   └── yjyg.parquet（业绩预告，因子未接入）
        │
        ▼
data/clean.clean_ohlcv() → clean_ret + masks{limit_up, limit_down, limit_up_open, ...}
        │
        ▼
factors/get_factor_registry(prices, financial, ..., clean_ret, masks, ...)
        │
        ▼
{因子名: DataFrame(date×stock)} → IC v2 筛选 → factor_configs.yaml
        │
        ▼
strategies/ml.py → WalkForwardTrainer → factor_scores → backtest/quantile.py
```

### 1.4 生产白名单使用情况（`config/factor_configs.yaml`）

- **h5（周频）**：29 个因子入选
- **h10（双周）**：32 个因子入选
- **h20（月频）**：36 个因子入选
- 被排除因子多因纯 IC<0.02 或 ICIR<0.3（已记入 yaml `excluded` 注释）

---

## 2. 已实现因子清单与实现质量评估

### 2.1 量价/动量类（`factor.py`）

| 因子名 | 行号 | 数据依赖 | clean_ret | 实现质量 | 问题 |
|--------|------|----------|-----------|----------|------|
| 动量_Nd | 89-103 | prices, clean_ret | ✅ | 中 | `(1+ret).rolling.apply(np.nanprod)` 是 Python 级循环，性能慢；可改 log-return sum 向量化 |
| 反转_Nd | 106-118 | prices, clean_ret | ✅ | 中 | 同上，与动量复用同段逻辑可抽取公共函数 |
| 动量_skip | 121-139 | prices, clean_ret | ✅ | 中 | 同上性能问题 |
| 波动率_Nd | 142-150 | prices, clean_ret | ✅ | 高 | 向量化 std，OK |
| 换手率_20d | 153-159 | volume | ❌ | 低 | **未传 clean_ret**（注释认为成交量为信号），但未做流通股本归一化，混入规模效应；`rolling(window).mean()` 无 `min_periods`，前 19 日全 NaN |
| Amihud_20d | 162-173 | prices, amount, clean_ret | ✅ | 高 | OK；`amount.replace(0, NaN)` 处理停牌 |
| 振幅_20d | 176-190 | prices, high, low, masks | N/A | 高 | 屏蔽一字板，OK |
| 52周新高 | 193-200 | prices | N/A | 中 | `rolling(window).max()` 无 `min_periods`，前 51 日全 NaN；非真正 52 周新高（用 52*5=260 交易日近似） |
| 分数差分动量_20d | 203-230 | prices | N/A | 高 | AFML Ch5 实现，OK |

### 2.2 财务类（`factor.py`）

| 因子名 | 行号 | 数据依赖 | PIT 对齐 | 实现质量 | 问题 |
|--------|------|----------|----------|----------|------|
| 价值_PB | 237-248 | financial.bvps, prices_raw | ✅ via pit_pivot_ffill | 高 | OK |
| 价值_EP | 251-265 | financial.roe, bvps, prices_raw | ✅ | 中 | **`eps = roe * bvps` 是近似**，financial 实际有 `eps` 列，应直接用 |
| 质量_ROE | 268-275 | financial.roe | ✅ | 高 | OK |
| 质量_ROE变化 | 278-292 | financial.roe | ✅ via pit_reindex_ffill | 高 | OK |
| 质量_毛利率 | 295-306 | financial.gross_profit_margin | ✅ | 高 | OK；缺列时返回 None |
| 质量_毛利率变化 | 309-325 | financial.gross_profit_margin | ✅ | 高 | OK |
| 质量_应计项目 | 328-348 | financial.net_profit, operating_cashflow, total_assets | ✅ | 高 | OK；Sloan 应计标准实现 |
| 规模 | 351-359 | financial.total_assets | ✅ | 中 | **用 total_assets 代理市值**，与 Barra_Size 用 total_mv 不一致；建议改用流通市值 |
| 杠杆 | 362-377 | financial.total_assets, bvps | ✅ | **低** | **量纲错误**：`assets / bvps` 混了"总资产（元）"与"每股净资产（元/股）"单位；financial 已有 `debt_ratio` 列，应直接用 |

### 2.3 第二批 Alpha（`factor_alpha.py`）

| 因子名 | 行号 | 数据依赖 | clean_ret | 实现质量 | 问题 |
|--------|------|----------|-----------|----------|------|
| 行业动量_20d | 30-90 | prices, industry_map, clean_ret | ✅ | 中 | **PIT 风险**：`industry_map.loc[common, "sw_l2"]` 用静态映射，未用 `industry_map_panel.parquet`；行业归属变更时引入未来信息；双重循环 `for ind / for code` 性能慢，可向量化 groupby+transform |
| 特质波动率_60d | 97-162 | prices, market_prices, clean_ret | ✅ | **低** | **Python 逐日循环** `for end_idx in range(min_obs, len(common_dates))`，O(T) 循环每次切 (T,N) 矩阵，全样本极慢；可改用 rolling cov/var 向量化（参考 `barra_risk.py:144` 的 `barra_res_vol` 实现） |
| 融资余额变化_20d | 169-185 | margin | N/A | 高 | OK；pct_change 处理 inf→NaN |
| 大单净流入_5d | 192-207 | moneyflow | N/A | **低** | **未按成交额标准化**（注释承诺但代码没做），大市值股票天然大单额大，截面 zscore 仅部分缓解；应改 `moneyflow / amount` |
| 北向持股变化_20d | 214-229 | northbound | N/A | 中 | `pct_change(window)` 对 0→X 起跳产生 inf→NaN，新纳入港股通标的会缺失；建议改 `delta / 流通股本` |
| 机构持仓变化 | 236-254 | institution, prices | N/A | 高 | PIT 对齐 OK |

### 2.4 Alpha101（`factor_alpha101.py`）

| 因子名 | 行号 | 数据依赖 | clean_ret | 实现质量 | 问题 |
|--------|------|----------|-----------|----------|------|
| WQ_001 | 79-92 | close, volume | ❌ | 中 | **未传 clean_ret**，涨跌停日 pct_change 截断污染 argmax 信号 |
| WQ_002 | 95-106 | close, open, volume | ❌ | 中 | 同上 |
| WQ_006 | 109-118 | open, volume | ❌ | 中 | 同上 |
| WQ_007 | 121-136 | close, volume | ❌ | 中 | 同上；`adv20 < volume` 三元条件用 `.where` 实现，OK |
| WQ_012 | 139-147 | close, volume | ❌ | 中 | 同上 |
| WQ_028 | 150-163 | close, high, low, volume | ❌ | 中 | 同上 |
| WQ_034 | 166-181 | close | ❌ | 中 | 同上 |
| WQ_053 | 184-197 | close, high, low | ❌ | 中 | 同上 |
| WQ_061 | 200-217 | close, volume, amount | ❌ | 中 | 同上；VWAP = amount/volume 合理 |
| WQ_101 | 220-229 | close, open, high, low | ❌ | 中 | 同上 |

**共性问题**：`get_alpha101_factors()` 接口完全未接受 `clean_ret` 参数，全部用 `close.pct_change()`，与 AGENTS.md "量价因子必须用 clean_ret" 约定不一致。这是**全文件级 P0 问题**。

### 2.5 技术指标（`factor_technical.py`）

| 因子名 | 行号 | 数据依赖 | clean_ret | 实现质量 | 问题 |
|--------|------|----------|-----------|----------|------|
| BIAS_5/20d | 61-64 | prices | N/A | 高 | OK |
| PSY_12d | 75-79 | prices, clean_ret | ✅ | 高 | OK |
| AR_26d | 91-96 | open, high, low | N/A | 高 | OK |
| BR_26d | 99-106 | prices, high, low | N/A | 高 | OK |
| 换手率行业中性_20d | 118-125 | volume, industry_map | N/A | 中 | **同行业动量 PIT 风险**；`_industry_demean` 内部 `for grp_name` 循环可向量化 |
| 换手率加速度 | 137-143 | volume | N/A | 高 | OK |
| 行业相对强度_20d | 154-164 | prices, industry_map, clean_ret | ✅ | 中 | 同上 PIT 风险 |

### 2.6 涨跌停信号（`factor_limit.py`）

| 因子名 | 行号 | 数据依赖 | 实现质量 | 问题 |
|--------|------|----------|----------|------|
| 涨停强度_20d | 21-34 | masks | 高 | 排除一字板，OK |
| 跌停弱势_20d | 37-40 | masks | 高 | OK |
| 连板数 | 43-54 | masks | 中 | `for t in range(1, len(arr))` 是 Python 循环；可用 cumsum+groupby 向量化 |
| 涨跌停净强度_20d | 57-61 | masks | 高 | OK |
| 开板反转_5d | 64-85 | close, masks | 高 | shift(hold_days) 防未来函数，注释清晰，OK |

### 2.7 Barra 风格（`barra_risk.py`，已审过，简要）

9 个 Barra 因子做截面 zscore（不做 winsorize），用作 IC 残差化控制变量。`barra_beta`/`barra_res_vol`/`barra_momentum` 均已切到 clean_ret，行业中性化可选传入。`barra_nonlin_size` 仍有逐日 `for date in size_df.index` Python 循环（行 81-94），样本大时性能瓶颈。

### 2.8 业绩预告（`factor_event.py`，未接入）

| 因子名 | 行号 | 实现质量 | 问题 |
|--------|------|----------|------|
| factor_yjyg | 52-158 | 中 | **未在 `get_factor_registry()` 中注册**，全代码 base，已下载 yjyg.parquet 但生产管线用不到；pre_drift 计算用 `for code` 循环慢；事件信号 ffill 60 日后衰减合理 |

### 2.9 市场 regime（`factor.py:384-466`）

| 特征名 | 实现质量 | 问题 |
|--------|----------|------|
| 市场动量_20/60/120d | 高 | shift(1) 防前视，OK |
| 市场波动率_20d | 高 | OK |
| 市场MA偏离_60d | 高 | OK |
| HMM_强势/弱势概率 | 中 | `model.fit(returns)` 用全样本拟合（含未来），HMM 状态序列本身是 in-sample 后验概率 → **存在轻微 look-ahead bias**；应改为扩展窗口滚动拟合或在线更新 |

---

## 3. 缺失因子评估

对照现代 A 股量化因子体系，分 7 大类列出缺失因子。每条标注：
- **重要性**：高/中/低（对该项目 AI 牛市超额收益目标）
- **数据可得性**：✅ AKShare 现成 / ⚠️ 需另下载 / ❌ 需另源（Wind/聚宽/Tushare Pro）
- **实现难度**：低/中/高
- **预期 IC 贡献**：高/中/低（基于公开文献与 A 股实证）

### A. 基本面因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **Piotroski F-Score** | 高 | ✅ financial 现有列组合即可 | 低 | 中 | 9 项打分（ROE/现金流/杠杆/发行/毛利率/资产周转），A股小盘显著 |
| **Greenblatt Magic Formula** | 中 | ✅ 已有 ROE+PB | 低 | 中 | 资本回报率+收益率组合排名 |
| **PS/PCF/EV/EBITDA** | 中 | ⚠️ 需补市值、股本、EBITDA | 中 | 中 | 价值因子多元化，PS 在成长股有效 |
| **股息率** | 中 | ⚠️ 需分红数据 | 中 | 中 | A 股红利策略近年显著 |
| **回购因子** | 中 | ❌ 需另源 | 高 | 中 | A 股回购逐年增多 |
| **营收增长率/净利增长率** | **高** | ✅ financial 已有 `revenue_growth`、`net_profit_growth` 列 | **低** | **高** | **当前未使用！** 直接 pivot+ffill 即可，A 股成长因子显著 |
| **EPS 增长率** | 高 | ✅ financial 已有 `eps` 列 | 低 | 高 | 同上，差分即可 |
| **ROE 变化（已有）+ ROA 变化** | 中 | ✅ 已有 total_assets, net_profit | 低 | 中 | ROA = net_profit/total_assets |
| **现金流稳定性** | 中 | ✅ 已有 operating_cashflow | 中 | 中 | 滚动 8 季现金流 std/mean |
| **Sloan 应计异常（已有）+ 资产增长率** | 中 | ✅ total_assets | 低 | 中 | YoY 总资产增速，警示外延扩张 |
| **资产负债率变化** | 中 | ✅ financial 已有 `debt_ratio` | 低 | 中 | 去杠杆信号 |
| **利息保障倍数** | 中 | ❌ 财务费用/EBIT 缺 | 高 | 中 | 需补财务附注字段 |
| **营运资金周转** | 中 | ❌ 应收/存货/应付缺 | 高 | 中 | 需补资产负债表明细 |

### B. 量价/技术因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **下行波动率 / UPR** | 高 | ✅ prices | 低 | 中 | 半方差、upside potential ratio，比总波动更稳健 |
| **最大回撤 / 偏度 / 峰度** | 高 | ✅ prices | 低 | 中 | 收益分布高阶矩，A 股负偏度显著 |
| **波动率锥 / 分位波动率** | 中 | ✅ prices | 中 | 中 | 当前波动率在历史分位的位置 |
| **量价相关系数** | 中 | ✅ prices, volume | 低 | 中 | rolling corr(ret, volume)，量价配合 |
| **OBV / 量能累积** | 中 | ✅ prices, volume | 低 | 中 | On-Balance Volume |
| **资金流向比率** | 中 | ✅ moneyflow_large/superlarge 已下载 | 低 | 中 | 大单/超大单占成交额比，已下载数据未充分利用 |
| **零收益率天数占比** | 中 | ✅ prices | 低 | 低 | 流动性代理，A 股效应较弱 |
| **Amihud 改进（平方版）** | 低 | ✅ 已有 | 低 | 低 | 学术小改进 |
| **Kyle lambda / OFI 订单流不平衡** | 高 | ❌ 需高频 tick | 高 | 高 | 微观结构因子，A 股日频代理有限 |
| **GARCH 波动率预测** | 中 | ✅ prices | 高 | 中 | 需 arch 库，对 ML 价值边际 |
| **PEAD（盈余公告后漂移）** | **高** | ✅ yjyg 已下载 | 中 | **高** | 业绩预告因子已实现但**未接入**！P0 |
| **特质动量（residual momentum）** | 高 | ✅ prices, market_prices | 中 | 高 | 剔除 beta 后的动量，比裸动量更稳健 |
| **52 周动量（距 52 周高点距离已有，距 52 周低点缺失）** | 低 | ✅ prices | 低 | 低 | 已有"52周新高"，可补"距52周低点" |

### C. 资金流/持仓因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **北向资金个股净流入（金额）** | 高 | ✅ northbound_value 已下载未用 | 低 | 高 | **当前用持股量变化，金额维度数据已下载但未用** |
| **北向 V-shaped 信号** | 中 | ✅ 已有 | 中 | 中 | 持股先减后增的抄底信号 |
| **北向持股比例（绝对水平）** | 中 | ✅ 已有 | 低 | 中 | 外资偏好代理 |
| **融资买入额/占比** | 高 | ⚠️ 需补融资买入额 | 中 | 高 | 比融资余额变化更敏感 |
| **融券余额/融券卖出** | 中 | ⚠️ 需补融券数据 | 中 | 中 | 做空压力信号 |
| **龙虎榜上榜频率** | 高 | ⚠️ AKShare 可下 | 中 | 高 | A 股短线情绪核心信号 |
| **龙虎榜机构席位买卖** | 高 | ⚠️ AKShare 可下 | 中 | 高 | 机构席位净买入，事件+资金双信号 |
| **大宗交易折价率** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 大宗折价高 → 大股东减持信号 |
| **股东户数变化** | 高 | ⚠️ 需补定期报告股东数据 | 中 | 高 | 户数减少=筹码集中，A 股强信号 |
| **高管增减持** | 高 | ⚠️ AKShare 可下 | 中 | 高 | 内幕人信号，A 股显著 |
| **股权质押比例** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 高质押=爆仓风险 |
| **解禁事件** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 解禁前抛压 |

### D. 跨资产/宏观因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **市场情绪综合指标（融资情绪/换手率情绪）** | 高 | ✅ 已有 margin, volume | 中 | 高 | 已有原料，未做情绪合成 |
| **国债收益率期限结构（10Y-1Y 利差）** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 风险偏好代理 |
| **信用利差（AA-AAA）** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 信用周期信号 |
| **M2 同比 / 社融同比** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 流动性周期 |
| **PMI / 工业增加值** | 低 | ⚠️ AKShare 可下 | 中 | 低 | 月频宏观，与日频因子对齐难 |
| **商品-股票相关性（南华商品指数）** | 低 | ⚠️ AKShare 可下 | 中 | 低 | 周期股敏感 |
| **汇率影响（USDCNY）** | 低 | ⚠️ AKShare 可下 | 中 | 低 | 出口/外债股敏感 |

### E. 事件因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **业绩预告（已有，未接入）** | **高** | ✅ 已下载 | **低** | **高** | **P0：已实现，仅差注册到 `get_factor_registry()`** |
| **业绩超预期（实际 vs 预告）** | 高 | ⚠️ 需补实际业绩+预告 | 中 | 高 | PEAD 核心信号 |
| **业绩快报** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 比正式报告早 |
| **定增事件** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 折价定增+解禁期 |
| **回购公告** | 中 | ❌ 需另源 | 高 | 中 | A 股回购逐年增多 |
| **股权激励公告** | 中 | ⚠️ AKShare 可下 | 中 | 中 | 行权条件隐含成长指引 |
| **指数成分股调整（沪深300调入调出）** | 高 | ⚠️ AKShare 可下 | 中 | 高 | 调入前/调出后被动资金流，A 股显著 |
| **ST 摘帽/戴帽** | 中 | ✅ 已有 ST 时间序列 | 低 | 中 | 已有原料，未做事件因子 |

### F. Alpha101 / 学术因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **Alpha101 全集（101 个）** | 中 | ✅ 已有 OHLCV | 中 | 中 | 当前 10 个，可扩到 30-50 个，IC 筛选保留有效者 |
| **Hou-Xue-Zhang CHN-3 / CHN-4 因子模型** | 高 | ✅ 已有 ROE/inv/gross_profit | 中 | 高 | 中国版 Fama-French，A 股实证更适用；CHN-4 = Market+Size+Value+Investment；CHN-3 加 Profitability |
| **Fama-French 5 因子中国版** | 中 | ✅ 已有原料 | 中 | 中 | 与 CHN-3/4 重叠，择一 |
| **Q-factor model（Hou-Mo-Xue-Zhang）** | 中 | ⚠️ 需投资因子 | 中 | 中 | 学术前沿，A 股实证有效 |
| **短期反转（已有）+ 隔夜反转** | 高 | ✅ open, close | 低 | 高 | 隔夜收益 vs 日内收益分解，A 股隔夜反转显著（Lou-Polk-Skouras） |
| **尾盘收益占比** | 中 | ⚠️ 需分钟数据 | 高 | 中 | 14:30-15:00 收益占全天比，机构行为 |
| **停牌频率因子** | 中 | ✅ volume=0 可识别 | 低 | 中 | 停牌多=流动性差+不确定性高 |

### G. AI/另类因子

| 缺失因子 | 重要性 | 数据可得性 | 难度 | 预期 IC | 备注 |
|----------|--------|-----------|------|---------|------|
| **新闻/研报文本情绪** | 高 | ❌ 需 NLP+数据源 | 高 | 高 | 超出当前项目范围，但收益最高 |
| **基金重仓股变化** | 高 | ✅ institution 已有 | 低 | 高 | **已下载但只用了 diff，可补重仓集中度、新进/退出** |
| **基金重仓股拥挤度** | 中 | ✅ institution | 低 | 中 | 重仓比例过高=拥挤，反转信号 |
| **产业链关联（供应链/同业）** | 中 | ❌ 需另源 | 高 | 中 | 跨股票收益传染 |
| **研报关注度/分析师一致预期** | 高 | ❌ 需另源（Wind/Choice） | 高 | 高 | A 股分析师覆盖度因子显著 |
| **股权关联（集团/母子）** | 低 | ❌ 需另源 | 高 | 低 | 主题联动 |

---

## 4. 实现质量问题汇总

按 文件:行号 — 问题 — 修复建议 — 难度 列出。

### 4.1 正确性问题（P0）

| # | 位置 | 问题 | 修复建议 | 难度 |
|---|------|------|----------|------|
| P0-1 | `factor_alpha101.py:236-290` `get_alpha101_factors` | **全文件未传 clean_ret**，10 个 Alpha101 因子全部用 `close.pct_change()`，涨跌停日 return 截断污染 | 接口加 `clean_ret` 参数，`wq_alpha001/002/006/007/012/028/034/053/061/101` 内部 `ret = clean_ret if clean_ret is not None else close.pct_change()` | 中 |
| P0-2 | `factor_alpha.py:30-90` `factor_industry_momentum`、`factor_technical.py:154` `factor_industry_relative_strength`、`factor_technical.py:118` `factor_turnover_neutral` | **industry_map 用静态映射**，未用 PIT `industry_map_panel.parquet`（AGENTS.md 已声明 PIT 落地但因子层未消费） | 改用 PIT panel：按 date 取当期 sw_l2；`_industry_demean` 改为按行（date）从 panel 取行业 | 高 |
| P0-3 | `factor.py:362-377` `factor_leverage` | **量纲错误**：`assets / bvps` 混了"元"与"元/股"单位；financial 已有 `debt_ratio` 列 | 直接 `_pivot_financial(financial, "debt_ratio", prices)` 后取负 | 低 |
| P0-4 | `factor_event.py:52-158` `factor_yjyg` | **已实现未接入**，`get_factor_registry()` 未注册，业绩预告数据已下载但生产管线零使用 | 在 `get_factor_registry()` 加 `_EVENT_FACTOR_NAMES` 段，导入 `factor_yjyg` 并按 `factor_names` 过滤 | 低 |
| P0-5 | `factor.py:251-265` `factor_value_ep` | **EPS 用 `roe*bvps` 近似**，financial 实际有 `eps` 列（见 data 检查） | 改 `_pivot_financial(financial, "eps", prices) / price` | 低 |
| P0-6 | `factor.py:153-159` `factor_turnover` | **未做流通股本归一化**，raw volume 均值混入规模效应；`rolling(window).mean()` 无 `min_periods` | 用 `volume / 流通股本` 得真换手率；若缺股本数据，至少加 `min_periods=window//2` | 中 |
| P0-7 | `factor_alpha.py:192-207` `factor_moneyflow_large` | **未按成交额标准化**（注释承诺但代码没做），大市值股票大单额天然大 | 改 `(moneyflow / amount).rolling(window).mean()`，amount 传入因子 | 低 |
| P0-8 | `factor.py:384-415` `_fit_hmm_regime` | **HMM 全样本拟合**，`model.fit(returns)` 用全部历史（含未来）数据，后验概率为 in-sample | 改扩展窗口滚动拟合：每个 t 用 [0, t] 拟合，或在线更新；或退化为基础技术规则 | 高 |

### 4.2 性能问题（P1）

| # | 位置 | 问题 | 修复建议 | 难度 |
|---|------|------|----------|------|
| P1-1 | `factor_alpha.py:139-160` `factor_idiosyncratic_vol` | **Python 逐日循环** O(T)，每次切 (window, N) 矩阵做 OLS，全样本极慢 | 改 rolling cov/var 向量化（参考 `barra_risk.py:144` `barra_res_vol`），`ivol = total_vol * sqrt(1-R²)` | 中 |
| P1-2 | `factor.py:89-139` `factor_momentum/reversal/momentum_skip` | `(1+ret).rolling.apply(lambda x: np.nanprod(x)-1)` 是 Python 级 lambda，慢 | 改用 `clean_ret.rolling(window).sum()` 近似（短窗口 log-return 近似有效），或 `np.exp(np.log1p(ret).rolling(window).sum()) - 1` 全向量化 | 低 |
| P1-3 | `factor_alpha.py:67-86` `factor_industry_momentum` | 双重循环 `for ind / for code`，每个行业逐股票算"排除自身均值" | 向量化：`ind_sum - ret` 与 `ind_count - 1` 一次性算；或 `groupby(industry).transform(lambda x: (x.sum() - x) / (x.notna().sum() - 1))` | 中 |
| P1-4 | `barra_risk.py:81-94` `barra_nonlin_size` | 逐日 `for date in size_df.index` 做 OLS 正交化 | 可用 `np.linalg.lstsq` 批量化，或用 groupby 年度做（非线性规模变化慢） | 中 |
| P1-5 | `factor_limit.py:48-51` `factor_consecutive_limit_up` | Python 循环算连板数 | 向量化：`(arr == 0).cumsum()` 分组后再 cumsum | 低 |
| P1-6 | `factor_technical.py:43-50` `_industry_demean` | `for grp_name in ind.unique()` 循环 | 用 `panel.T.groupby(ind.values).transform('mean').T` 一次性算（`barra_risk.py:305` 已是此模式） | 低 |

### 4.3 健壮性问题（P2）

| # | 位置 | 问题 | 修复建议 | 难度 |
|---|------|------|----------|------|
| P2-1 | `factor.py:158, 198` rolling 无 `min_periods` | `volume.rolling(window).mean()` 与 `prices.rolling(window).max()` 前 N-1 日全 NaN | 加 `min_periods=window//2` | 低 |
| P2-2 | `factor.py:716` `compute_composite_factor` | 仅传 prices/financial/prices_raw，未传 clean_ret/volume/masks，线性基线动量未用 clean_ret | 补全参数透传 | 低 |
| P2-3 | `factor_alpha.py:214-229` `factor_northbound_change` | `pct_change` 对 0→X 起跳产生 inf→NaN，新纳入港股通标的缺失 | 改 `delta(holding) / 流通股本` | 中 |
| P2-4 | `factor_alpha.py:62` 假设 industry_map 是 DataFrame | 若传入 Series 会 AttributeError | 类型分支处理（参考 `factor_technical.py:202-205`） | 低 |
| P2-5 | `factor_alpha.py:169-185` `factor_margin_change` | 融资余额数据未做截面去新股偏差（次新股融资余额低、变化率极端） | 加 `IC_MIN_LISTING_DAYS=252` mask 透传 | 中 |
| P2-6 | `factor.py:193-200` `factor_price_to_high` | `52*5=260` 交易日近似 52 周，A 股每年约 244 交易日，近似偏长 | 改 244 或参数化 | 低 |
| P2-7 | `factor_alpha101.py:213` `wq_alpha061` `vwap = amount / volume` | 停牌日 volume=0→NaN，ffill 未处理 | 检查 NaN 比例，必要时 `vwap.rolling(5).mean()` 改 `min_periods` | 低 |
| P2-8 | `factor_event.py:106-122` `factor_yjyg` pre_drift | `for code` Python 循环算漂移 | 向量化：用 prices.pct_change(pre_drift_period).shift(1) 后按 announce_date 索引 | 中 |

### 4.4 文档/接口问题（P2）

| # | 位置 | 问题 | 修复建议 | 难度 |
|---|------|------|----------|------|
| P2-9 | `factor.py:14-15` docstring | 写"换手率（需要成交量数据）"但函数体是 raw volume 均值，不是真换手率 | 改文档或改实现（见 P0-6） | 低 |
| P2-10 | `factor_alpha.py:206-207` docstring | 写"用成交额做标准化"但代码没做 | 改代码（见 P0-7） | 低 |
| P2-11 | `factor.py:485-505` 因子名常量 | `factor_names` 白名单与实际 registry key 必须严格一致，目前手工维护易漂移 | 加单元测试断言常量集 == registry.keys() | 低 |

---

## 5. 优先级 TODO 清单

### P0 — 必须改（影响正确性）

| # | 任务 | 改动量 | 难度 | 预期收益 |
|---|------|--------|------|----------|
| P0-1 | Alpha101 接入 clean_ret：10 个因子全部改 | 中（~30 行） | 中 | 修复涨跌停污染，IC 估计无偏 |
| P0-2 | 行业因子改用 PIT `industry_map_panel.parquet`：3 个因子 | 大（需重构 `_industry_demean`） | 高 | 消除未来函数，A 股行业变更频繁 |
| P0-3 | `factor_leverage` 改用 `debt_ratio` 列 | 小（~5 行） | 低 | 修复量纲错误 |
| P0-4 | `factor_yjyg` 注册到 `get_factor_registry()` | 小（~15 行） | 低 | **零成本接入已实现因子**，PEAD 是 A 股强信号 |
| P0-5 | `factor_value_ep` 直接用 `eps` 列 | 小（~3 行） | 低 | 修复 EPS 近似误差 |
| P0-6 | `factor_turnover` 改真换手率（流通股本归一化） | 中（需补流通股本数据） | 中 | 修复规模混入，IC 估计无偏 |
| P0-7 | `factor_moneyflow_large` 按成交额标准化 | 小（~3 行） | 低 | 修复规模混入 |
| P0-8 | HMM 改扩展窗口滚动拟合 | 大 | 高 | 消除 in-sample 后验概率前视 |

### P1 — 建议改（补充关键因子，高 ROI）

| # | 任务 | 改动量 | 难度 | 预期收益 |
|---|------|--------|------|----------|
| P1-1 | **接入成长类因子**：营收增长率、净利增长率、EPS 增长率（financial 已有列，零下载成本） | 小 | 低 | **高** — A 股成长因子显著，3 个新因子 |
| P1-2 | **接入已下载的 northbound_value**：北向资金金额净流入（当前只用持股量） | 小 | 低 | 高 — 双维度北向信号 |
| P1-3 | **基金重仓股因子扩展**：重仓集中度、新进/退出、拥挤度（institution 已有） | 中 | 中 | 高 — 已有数据未充分利用 |
| P1-4 | **下行波动率 / 偏度 / 峰度**：3 个收益分布因子 | 小 | 低 | 中 — 比总波动更稳健 |
| P1-5 | **特质动量（residual momentum）**：剔除 beta 后动量 | 中 | 中 | 高 — 比裸动量更稳健 |
| P1-6 | **Piotroski F-Score**：9 项打分 | 中 | 中 | 中 — A 股小盘显著 |
| P1-7 | **PEAD 因子（实际 vs 预告差异）**：需补实际业绩对齐 | 大 | 中 | 高 — 业绩超预期强信号 |
| P1-8 | **隔夜反转 + 日内动量分解**：`(open[-1]/close[-1]-1)` vs `(close/open-1)` | 小 | 低 | 高 — Lou-Polk-Skouras A 股实证 |
| P1-9 | **龙虎榜因子**：上榜频率 + 机构席位买卖 | 大（需下载） | 中 | 高 — A 股短线情绪核心 |
| P1-10 | **高管增减持因子** | 大（需下载） | 中 | 高 — 内幕人信号 |
| P1-11 | **股东户数变化** | 大（需下载） | 中 | 高 — 筹码集中度 |
| P1-12 | **CHN-3 / CHN-4 中国因子模型**：Market+Size+Value+Investment+Profitability | 大 | 高 | 高 — 学术验证 A 股更适用 |
| P1-13 | **指数成分股调整事件因子** | 大（需下载） | 中 | 高 — 调入调出被动资金 |
| P1-14 | **Alpha101 扩展到 30-50 个**：现有 10 个，扩到主流有效集 | 中 | 中 | 中 — IC 筛选保留有效者 |
| P1-15 | **IVOL 向量化重构**（P1-1 性能） | 中 | 中 | 性能 10×+ |
| P1-16 | **动量/反转向量化**（P1-2 性能） | 小 | 低 | 性能 5×+ |

### P2 — 可选（补充次要因子或优化）

| # | 任务 | 改动量 | 难度 | 预期收益 |
|---|------|--------|------|----------|
| P2-1 | 龙虎榜 / 大宗交易 / 股权质押 / 解禁事件因子 | 大 | 中 | 中 |
| P2-2 | 融资买入额/融券余额（需补数据） | 中 | 中 | 中 |
| P2-3 | 量价相关、OBV、资金流向比率 | 小 | 低 | 中 |
| P2-4 | ROA、ROA 变化、现金流稳定性、资产增长率 | 小 | 低 | 中 |
| P2-5 | PS / PCF / 股息率 / EV-EBITDA（需补数据） | 中 | 中 | 中 |
| P2-6 | 宏观因子：国债利差、信用利差、M2、社融 | 中 | 中 | 中 |
| P2-7 | 情绪合成指标（融资情绪+换手率情绪） | 小 | 低 | 中 |
| P2-8 | ST 摘帽/戴帽事件因子（原料已有） | 小 | 低 | 中 |
| P2-9 | 停牌频率因子（volume=0 识别） | 小 | 低 | 中 |
| P2-10 | 文本情绪 / 分析师一致预期 | 极大 | 高 | 高但超范围 |
| P2-11 | 滚动 min_periods 修复（P2-1） | 小 | 低 | 健壮性 |
| P2-12 | 因子名常量单测断言（P2-11） | 小 | 低 | 防漂移 |

---

## 6. Top 10 应补充因子（按预期收益/实现成本排序）

| 排名 | 因子 | 类别 | 实现成本 | 预期收益 | 理由 |
|------|------|------|----------|----------|------|
| 1 | **factor_yjyg 接入 registry** | 事件 | 极低（已实现） | 高 | 代码已写好+数据已下载，仅差 15 行注册；PEAD 是 A 股强 alpha |
| 2 | **营收增长率 / 净利增长率 / EPS 增长率** | 基本面 | 极低（列已存在） | 高 | financial_indicators.parquet 已有 `revenue_growth`、`net_profit_growth`、`eps` 三列，pivot+ffill 即可，3 个新因子 |
| 3 | **factor_value_ep 用真 eps 列** | 基本面 | 极低（3 行改） | 中-高 | 修复 EPS 近似误差，PB/EP 价值因子组合更稳健 |
| 4 | **factor_leverage 改用 debt_ratio** | 基本面 | 极低（5 行改） | 中 | 修复量纲错误，杠杆因子 IC 估计无偏 |
| 5 | **北向资金金额净流入**（northbound_value 已下载未用） | 资金流 | 低 | 高 | 数据已下载零成本，金额维度比持股量变化更敏感 |
| 6 | **基金重仓股扩展**：重仓集中度、新进/退出、拥挤度 | 资金流 | 低-中 | 高 | institution_holding 已下载，当前仅用 diff，可挖 3-4 个新因子 |
| 7 | **Alpha101 接入 clean_ret** | 量价 | 中（~30 行） | 中-高 | 修复 10 个现有因子的涨跌停污染，IC 估计无偏 |
| 8 | **下行波动率 / 偏度 / 峰度**（3 个收益分布因子） | 量价 | 低 | 中 | 全向量化，A 股负偏度显著，比总波动率更稳健 |
| 9 | **隔夜反转 + 日内动量分解** | 量价 | 低 | 高 | Lou-Polk-Skouras A 股实证显著，open/close 已有 |
| 10 | **Piotroski F-Score** | 基本面 | 中 | 中 | 9 项打分（ROE/现金流/杠杆/发行/毛利率/资产周转），financial 现有列可组合 6-7 项 |

**Top 10 综合评估**：前 5 项几乎零成本，可在 1-2 天内落地，预期新增 5-8 个有效因子；第 6-10 项需 3-5 天，进一步覆盖资金流、量价分布、质量打分三大缺口。

---

## 7. 关键发现（审阅摘要）

### 7.1 优势

1. **基础架构扎实**：PIT 对齐、clean_ret、walk-forward、IC 筛选、Barra 残差化等现代量化方法论已落地，远超普通个人项目。
2. **涨跌停处理到位**：量价因子普遍传 clean_ret，振幅/连板/开板反转有专门 mask 逻辑，符合 A 股微观结构。
3. **市场 regime 工程化**：HMM 3 态 + 多周期动量 + MA 偏离 + 滚动 zscore，树模型感知宏观状态。
4. **Alpha101 精选有据**：基于 A 股实证论文（arxiv:2507.07107、FactorMiner）选 10 个，非盲目堆 101 个。
5. **分数差分动量**：AFML Ch5 已落地，体现对前沿方法的跟踪。

### 7.2 主要问题

1. **因子库广度不足**：约 56 个 registry 因子 + 9 个 Barra = 65 个，对比头部量化 200-500 个因子库体量明显偏小。**最关键缺口是基本面成长类（营收/净利增长率列已存在但未用）与事件类（业绩预告已实现但未接入）**——这两类零成本可补，是被忽视的低垂果实。
2. **资金流数据未充分挖掘**：`northbound_value.parquet`、`institution_holding.parquet`、`moneyflow_large.parquet` 等已下载但只用单一维度，可挖 5-8 个新因子。
3. **Alpha101 全文件未用 clean_ret**：与项目约定不一致，是 P0 级一致性问题。
4. **行业 PIT 未在因子层落地**：AGENTS.md 声明 PIT 已落地，但 `factor_industry_momentum`/`factor_industry_relative_strength`/`factor_turnover_neutral` 仍用静态 industry_map，存在未来信息风险。
5. **HMM 全样本拟合**：in-sample 后验概率，理论上轻微前视。
6. **杠杆因子量纲错误**、**EP 因子 EPS 近似**、**换手率未归一化**、**大单未按成交额标准化**：4 处实现 bug，均小改动可修。
7. **性能瓶颈**：`factor_idiosyncratic_vol` Python 逐日循环、动量 lambda 循环、行业动量双重循环，全样本运行可能数十分钟级。

### 7.3 推荐落地顺序

**第一周（P0 + Top 1-5）**：修复 6 处实现 bug + 接入 yjyg + 接入 3 个成长因子 + 北向金额 + 基金重仓扩展 → **预计因子数从 56 → 65+，IC 显著提升**

**第二周（Top 6-10）**：Alpha101 clean_ret + 3 个分布因子 + 隔夜反转 + F-Score → **因子数 → 72+**

**第三周起（P1 高 ROI）**：PEAD、特质动量、龙虎榜、高管增减持、股东户数、CHN-3/4、指数调整事件 → **因子数 → 90+**

**长期（P2）**：宏观因子、文本情绪、分析师预期、产业链关联等另类数据，需评估数据成本与项目范围。
