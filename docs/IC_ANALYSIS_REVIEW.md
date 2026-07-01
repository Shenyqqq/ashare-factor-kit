# IC 分析模块深度审阅报告

> 审阅范围：`research/ic_analysis.py`（v1）、`research/ic_analysis_v2.py` + `research/ic/` 子包（v2）、`factors/barra_risk.py`、`factors/factor.py`、`models/wf/labels.py`、`config/settings.py`、`logs/driver.py`
> 审阅日期：2026-07-01
> 模式：只读审阅，不改代码

---

## 1. 当前架构概览

### 1.1 v1 vs v2 对比

| 维度 | v1 (`ic_analysis.py`) | v2 (`research/ic/` 子包) |
|------|----------------------|--------------------------|
| 入口 | `python -m research.ic_analysis` | `python -m research.ic_analysis_v2` |
| 模块化 | 单文件 1200 行 | 14 个子模块（cli/load_data/forward_return/ic_series/statistics/barra/selection/industry/universe/decay_corr/cost/parallel/display/io） |
| **生产是否在用** | ✅ `logs/driver.py::run_ic` 第 137 行硬编码 `research.ic_analysis` | ❌ 未被 driver 调用 |
| t 统计量 | IID t（`ic.std()/sqrt(N)`，第 124-125 行） | IID t + **Newey-West HAC t**（`statistics.py::newey_west_t`，第 43-64 行） |
| 筛选用 t | IID t（`select_factors` 第 488 行） | 默认用 NW_t（`cli.py` 第 251 行 `nw_t_threshold=2.0 if use_nw_t`） |
| IC 稳定性 | ❌ 无 | ✅ `ic_stability_metrics`（rolling σ + 同向年份占比，`statistics.py` 第 99-114 行） |
| 可交易池 mask | ❌ 不剔除 ST/涨跌停/停牌 | ✅ `universe.py::build_ic_tradability_mask`（M4 ST 时间表） |
| IC 截面 winsorize/clip | ❌ 直接用原始 IC | ✅ `IC_CLIP=0.3` 或 `IC_WINSORIZE_PCT`（`statistics.py::prepare_ic_for_stats`） |
| IC rank method | 硬编码 `average` | 可配置 `IC_RANK_METHOD`（`ic_series.py:: _rank_panel`） |
| 扣成本 IC | ❌ 无 | ✅ `cost.py::estimate_ic_after_cost` + `selection.py::enrich_summary_with_cost` |
| 去冗余相关 | `mean` 聚合 | `max` / `p95` / `mean` 可选（默认 `max`，`selection.py::_aggregate_corr`） |
| Bonferroni 校正 | ❌ 无 | ❌ 无 |
| JSON 元数据 | horizon / lookback / factors / excluded / ic_stats(3 列) | + `engine="v2"` + NW_t + IC_after_cost 列 |
| 单元测试 | ❌ 无 | ✅ `test_smoke.py`（覆盖 icir ddof / win_rate / clip / NW / ic_stats keys） |

**关键结论**：v2 在统计稳健性上明显领先，但**生产 driver 仍在跑 v1**——v2 的 Newey-West、可交易池、扣成本、稳定性等改进全部未进入生产筛选链路。这是 P0 级配置问题。

### 1.2 数据流图

```
                        ┌──────────────────────────────────────────────┐
                        │ logs/driver.py::run_ic                        │
                        │   ↓ 调用 v1 (research.ic_analysis)            │
                        ├──────────────────────────────────────────────┤
  prices/financial/...  │   load → clean_ohlcv → get_factor_registry    │
  industry_map.parquet  │   ↓                                          │
                        │   build_forward_return(close[t+N]/open[t+1])  │
                        │   ↓                                          │
                        │   compute_ic_series (Spearman, 向量化 rank)   │
                        │   ↓                                          │
                        │   [可选 --barra]                             │
                        │     get_barra_factors(9 风格)                │
                        │     precompute_ctrl_matrices (调仓日缓存)     │
                        │     compute_pure_ic_fast (截面 OLS 残差 IC)   │
                        │   ↓                                          │
                        │   ic_stats → select_factors                  │
                        │     Step1: |IC|<0.02 且 ICIR<0.3 剔除         │
                        │     Step2: |t|<2 剔除 (v1: IID t)             │
                        │     Step3: |corr|>0.7 去重 (按 ICIR 贪心)     │
                        │   ↓                                          │
                        │   写 selected_factors_h{N}.json               │
                        └──────────────────────────────────────────────┘
                                          ↓
                        logs/driver.py::sync_factor_yaml
                                          ↓
                        config/factor_configs.yaml
                                          ↓
                        run.py --factor-config → get_factor_registry
                        build_ml_dataset (raw 因子, 无中性化)
                                          ↓
                        WalkForwardTrainer (label_mode=cs_zscore 默认)
                          └─ label 可选 barra_residual (残差化标签)
```

---

## 2. 截面因子中性化问题

### 2.1 当前实现评估

#### A. Barra 纯 IC 是否做对了？—— **基本正确，但有几处缺陷**

**正确点**：
- 控制变量构造合理：9 个 Barra 风格因子（Size/NonlinSize/Beta/Momentum/ResVol/Value/Liquidity/Leverage/Growth）+ 行业哑变量（drop_first 避免共线），见 `barra_risk.py::get_barra_factors` + `barra.py::_industry_dummies`。
- 残差化方法正确：逐截面 `f ~ [1, Barra_*, industry_dummies]` OLS 取残差，再与 forward_return 算 Spearman IC（`barra.py::compute_pure_ic_fast` 第 106-153 行）。
- 控制矩阵在调仓日预计算并缓存（`precompute_ctrl_matrices`），避免逐日 OLS 爆内存，效率合理。
- 残差 IC 仅用于筛选判定（`select_factors::effective_ic`），不污染原始 IC 报表。

**缺陷点**：

1. **【P1】Barra 因子自身未做行业中性化**。Barra 风格因子（如 Size、Liquidity、Leverage）有显著行业属性——金融业天然高杠杆、小盘股天然在创业板/科创。当前控制变量只对 alpha 因子做了一次残差化，但 Barra 控制变量之间未正交，导致残差仍可能含行业残余成分。
   - 文件：`factors/barra_risk.py` 第 187-241 行 `get_barra_factors`
   - 建议：在 `get_barra_factors` 出口对每个 Barra 因子再做一次行业去均值（`factor - 行业截面均值`），或用 Barra 标准两步法（先 Size/Beta 正交，再加行业）。

2. **【P1】Barra_Beta / Barra_ResVol 用 `prices.pct_change()` 而非 `clean_ret`**。
   - 文件：`barra_risk.py` 第 102、128 行
   - 影响：涨跌停日 return 被强制截断为 ±10%/±20%，会系统性高估个股 beta 与市场相关系数，污染 Barra_Beta 和 Barra_ResVol 的截面排名，进而污染纯 IC 残差。
   - 与 `factors/factor.py::factor_momentum` 的处理不一致——后者强制要求 `clean_ret`。
   - 建议：`barra_beta` / `barra_res_vol` 增加 `clean_ret` 参数并优先使用。

3. **【P1】行业映射是静态当前值，非 PIT 时间序列**。
   - 文件：`load_data.py` 第 94 行 `industry_map_df["sw_l2"]`；`barra.py::_industry_dummies` 第 26 行 `industry_map.reindex(stock_index)`
   - 影响：股票行业分类会随时间变化（如某股从机械重分类到新能源），用当前行业映射回填历史截面会引入未来信息。对纯 IC 影响有限（行业哑变量噪声），但对长样本（10 年+）的早期段是 PIT 泄漏。
   - 建议：构建时间序列 `industry_map_panel.parquet`（date × code → sw_l2），按截面日期取当期行业；若 akshare 不提供历史分类，至少用上市日 + 申万调整记录近似。

4. **【P2】行业用 sw_l2（申万二级，~130 个行业）而非 sw_l1（一级，31 个）**。
   - 文件：`load_data.py` 第 94 行；`ic_analysis.py` 第 956 行
   - 影响：二级行业哑变量更多（~129 列），调仓日截面 ~4000 股时控制变量矩阵 `(4000, 9+129)`，OLS 仍稳定；但样本不足的小行业（<30 股）哑变量近乎共线，可能引入噪声。
   - 用户问"是否用申万一级"——**当前是二级**。建议提供 `--industry-level l1|l2` 选项，l1 更稳健、l2 更精细；或对小行业合并到上级。

5. **【P2】`compute_pure_ic_fast` 用 `np.linalg.lstsq` 而非 QR 缓存**。
   - 文件：`barra.py` 第 144 行；TODO 注释在第 49-50 行已自承认
   - 影响：对每个 alpha 因子 × 每个调仓日都重做一次 `(N,K)` 的 QR 分解，~30 因子 × ~200 调仓日 = 6000 次 lstsq。控制矩阵 `ctrl_arr` 在调仓日内是共享的，本可预计算 `Q, R = qr(ctrl)` 后只做 `resid = y - Q@Q^T@y`。
   - 建议：在 `precompute_ctrl_matrices` 出口额外缓存 `Q` 矩阵，`compute_pure_ic_fast` 改用 `resid = f_v - Q@(Q.T@f_v)`，预期加速 3-5×。

#### B. 中性化是否对所有因子做？—— **是，但仅限 IC 筛选阶段**

- `ic_analysis.py` 第 1063-1095 行：`barra_names = [n for n in summary_df.index if n in registry]` —— 对 registry 中所有因子都计算纯 IC。
- v2 `barra.py::run_barra_pure_ic` 第 203 行同样：`barra_names = [n for n in summary_index if n in registry]`。
- **结论**：纯 IC 覆盖全因子，无误。

#### C. 【P0 关键问题】中性化只在 IC 分析阶段，ML 训练用的因子是未中性化的

这是审阅中**最重要的发现**。当前架构存在「IC 筛选口径」与「ML 训练口径」不一致：

| 阶段 | 因子值 | 标签值 |
|------|--------|--------|
| IC 筛选 | 计算纯 IC（剔除 Barra + 行业）作为筛选判据 | forward_return 原始 |
| ML 训练 | **原始因子**（`get_factor_registry` 出口只做 winsorize + zscore，无 Barra/行业中性化） | 默认 `cs_zscore`；可选 `barra_residual`（残差化标签） |

**问题链路**：
1. `factors/factor.py::get_factor_registry` 第 508-698 行：所有因子出口统一为 `_normalize = winsorize(1%) + cross_sectional_zscore(clip=3σ)`，**无任何 Barra/行业中性化**。
2. `strategies/ml.py::run` 第 80 行直接用 `get_factor_registry` 喂给 `build_ml_dataset`。
3. `models/wf/labels.py::transform_labels` 第 216-223 行：`barra_residual` 模式只对**标签 y** 做残差化，**对特征 X 完全不动**。
4. 默认 `label_mode="cs_zscore"`（`strategies/ml.py` 第 189 行），连标签残差化都没开。

**后果**：
- 一个 alpha 因子如果原始 IC=0.04 但纯 IC=0.015（即 60% 是 Size/Beta 敞口），IC 筛选可能保留它（纯 IC 仍 >0.02 阈值边缘），但 ML 拿到的是**含 60% 系统性敞口的原始因子**。模型会学到"这个因子有效"，实则大部分预测力来自 Size/Beta——而 Size/Beta 已被 Barra 残差化标签剔除（如果开了 `barra_residual`），导致特征与标签尺度不一致；如果没开（默认），则模型学到的就是"小市值 + 低 beta 有效"，纯粹是系统性风险溢价，不是真 alpha。
- 多个 alpha 因子之间如果都暴露 Size，它们在 ML 看来高度相关（伪相关），IC 筛选的 corr>0.7 去重检测的是原始因子相关而非纯 alpha 相关，可能漏判冗余。

#### D. 中性化应在哪个阶段做？

| 选项 | 优点 | 缺点 | 建议 |
|------|------|------|------|
| 因子计算阶段（`factors/factor.py` 出口） | IC 与 ML 口径一致；一次计算多处复用 | 增加因子计算成本；对所有下游都强制中性化，灵活性差 | ❌ 不建议作为默认 |
| IC 分析阶段（当前） | 诊断价值——能对比原始 IC vs 纯 IC，量化 alpha 含量 | 与 ML 口径不一致（当前问题） | ✅ 保留作为诊断 |
| ML 训练阶段（`build_ml_dataset` 出口） | IC 用原始/纯双口径，ML 用纯口径；可配置开关 | 需要新增特征中性化层 | ✅ **推荐** |
| 同时在 IC + ML 做中性化 | 最严谨 | 实现复杂度高 | 长期目标 |

**推荐方案**：在 `strategies/ml.py::build_factor_dataset` 出口新增可选 `feature_neutralize=True` 参数，调用 `factors/barra_risk.get_barra_factors` + 行业哑变量对每个因子做截面残差化（复用 `models/wf/labels.py::residual_return_label` 的 OLS 逻辑，把 y 换成因子值）。与 IC 阶段的纯 IC 用同一套控制变量，保证口径一致。

### 2.2 中性化修复建议（具体）

**修复 1（P0）：ML 特征中性化层**

文件：`strategies/ml.py` 新增函数 + `models/wf/labels.py` 抽取通用残差函数

```python
# models/wf/labels.py 新增（residual_return_label 已存在，抽取通用版）
def residualize_panel(
    factor_panel: pd.DataFrame,           # (date × stock) 单因子
    barra_factors: dict[str, pd.DataFrame],
    industry_map: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """逐截面 OLS 因子值 ~ [1, Barra_*, ind_dummies]，返回残差面板。"""
    # 复用 research/ic/barra.py::precompute_ctrl_matrices + compute_pure_ic_fast 的
    # 控制矩阵缓存逻辑，但输出残差面板而非 IC
    ...
```

```python
# strategies/ml.py 在 build_factor_dataset 出口
if feature_neutralize:
    from models.wf.labels import residualize_panel
    registry = {
        name: residualize_panel(f, barra_factors, industry_map, rebalance_dates)
        for name, f in registry.items()
    }
```

改动量：~80 行；难度：中；预期收益：**高**（修复 IC/ML 口径不一致，是当前架构最深的隐患）。

**修复 2（P1）：Barra 因子用 clean_ret**

文件：`factors/barra_risk.py` 第 99-137 行

```python
def barra_beta(prices, market_prices, window=252, clean_ret=None, mkt_clean_ret=None):
    stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
    mkt_ret = mkt_clean_ret if mkt_clean_ret is not None else market_prices.squeeze().pct_change()
    ...
```

`get_barra_factors` 透传 `clean_ret`。改动量：~15 行；难度：低；收益：中。

**修复 3（P1）：Barra 因子自身行业中性化**

文件：`factors/barra_risk.py::get_barra_factors` 出口

```python
if industry_map is not None:
    for name in factors:
        factors[name] = factors[name].sub(
            factors[name].groupby(industry_map, axis=1).transform('mean'),
            axis=0
        ).pipe(_cross_zscore)
```

改动量：~10 行；难度：低；收益：中。

---

## 3. IC 显著性检验问题

### 3.1 当前 t-stat 实现评估

| 实现 | t 公式 | 自相关调整 | 位置 |
|------|--------|-----------|------|
| v1 `ic_stats` | `mean / (std/sqrt(N))`，IID 假设 | ❌ 无 | `ic_analysis.py` 第 124-125 行 |
| v1 `select_factors` | 用 v1 `ic_stats` 的 IID t | ❌ 无 | `ic_analysis.py` 第 488 行 |
| v2 `ic_stats` | 同上 IID t（`t统计量`） + **NW HAC t（`NW_t统计量`）** | ✅ Newey-West | `statistics.py` 第 117-143 行 |
| v2 `select_factors` | 默认用 `NW_t统计量` | ✅ | `selection.py` 第 95-116 行 |

**v1 的问题**：IC 序列在月频调仓下有显著自相关（持仓期重叠 + 因子持续性），IID t 高估显著性，会导致弱因子被误判为有效保留下来。月频 h=20、调仓间隔约 20 日时持仓不重叠，自相关较弱；但周频 h=5、调仓间隔 5 日时如果用户用更长 hold_period（如 h=20 但周调仓），持仓重叠严重，IID t 严重失真。

**v2 已修复**：`newey_west_t` 用 statsmodels OLS(HAC) 或自实现 Bartlett kernel fallback，默认 lag = `floor(4*(n/100)^(2/9))`（`statistics.py` 第 39-40 行），是 Newey-1987 标准建议。**但生产 driver 仍调 v1，享受不到此修复**。

### 3.2 Newey-West 是否需要？—— **需要，v2 已做但未上线**

- 月频不重叠：lag=1 已足够，NW t ≈ IID t（影响小）。
- 周频 / 高频调仓：持仓重叠时 NW t 比 IID t 小 20-50%，是显著性判断的关键。
- **结论**：v2 实现 OK，只需把 driver 切到 v2（见 §5 P0-1）。

### 3.3 rolling ICIR / IC 衰减 / 符号稳定性 是否缺失

| 检验 | v1 | v2 | 评价 |
|------|----|----|------|
| rolling IC std | ❌ 仅在 `--plot` 时画图，无统计输出 | ✅ `ic_stability_metrics` 输出 `IC滚动标准差`（12 期窗口） | v2 已实现，但只算 std，未算 rolling ICIR |
| rolling ICIR | ❌ 无 | ❌ 无（只有滚动 std，未除以滚动 mean） | **缺失**——建议加 `rolling_icir = rolling_mean / rolling_std` |
| 同向年份占比 | ❌ 无 | ✅ `ic_stability_metrics` 输出 `同向年份占比` | v2 已实现 |
| IC 衰减分析 | ✅ `ic_decay_table`（5/10/20/40/60 日 IC 均值） | ✅ `decay_corr.py::ic_decay_table` | 已实现，但**只输出 IC 均值**，未输出各 lag 的 ICIR / t / 衰减比率 |
| IC 衰减统计 | ❌ 无 | ❌ 无 | **缺失**——建议 `ic_decay_table` 加 ICIR / t / half-life 列 |
| 符号稳定性（每年同号） | ❌ 无（仅逐年 IC 表，肉眼判断） | ✅ 同向年份占比 | v2 已实现 |
| IC 厚尾性 / 非正态 | ❌ 无 | ❌ 无 | **缺失**——建议加 IC 偏度 / 峰度 / bootstrap CI |

### 3.4 多重检验校正是否需要？—— **需要，当前完全没做**

- 当前对 ~30 个因子各自做 t 检验，阈值 `|t|>2` 即单因子 95% 置信。
- 30 个因子独立检验，整体 family-wise error rate ≈ `1 - 0.95^30 ≈ 78%`，即至少一个因子是假阳性的概率近 80%。
- **Bonferroni** 太保守（阈值变 `|t|>3.0` 单因子 99.9%），会误杀真 alpha。
- **推荐**：用 **Benjamini-Hochberg FDR** 控制（允许 5% 假发现率），比 Bonferroni 更适合因子筛选。

**缺失修复**（v2 加）：

```python
# research/ic/statistics.py 新增
def benjamini_hochberg(t_stats: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """返回每个因子是否在 FDR=alpha 下显著（bool 数组）。"""
    from scipy.stats import norm
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = p_values[order] * n / np.arange(1, n + 1)
    threshold = alpha
    significant = np.zeros(n, dtype=bool)
    for i, idx in enumerate(order):
        if adjusted[i] <= threshold:
            significant[idx] = True
    return significant
```

`select_factors` 增加一个 `use_fdr=True` 开关，对 NW_t 做 BH 校正后判定。改动量：~30 行；难度：低；收益：**高**（解决多因子多重检验假阳性）。

### 3.5 |t|<2 阈值是否合理？

- 单因子 95% 置信 = |t|>1.96 ≈ 2，**对探索性筛选偏松**。
- 学术标准（如 Hou et al. 2020 Replicating Anomalies）要求 |t|>3.0（Harvey-Liu 2016 推荐 |t|>3.0 应对数据窥探）。
- **建议**：v2 加 `--t-threshold` 参数，默认仍 2.0（保留现行行为），但 driver `--preset main` 改用 2.5-3.0。

### 3.6 IC 非正态性

- IC 序列在 A 股常呈厚尾（极端行情 IC 飙到 ±0.3-0.5），`ic_stats` 用 `mean/std` 假设近似正态，对小样本 N<50 不可靠。
- v2 已加 `IC_CLIP=0.3`（`config/settings.py` 第 54 行）+ `IC_WINSORIZE_PCT`，是合理的工程处理——但**只有 v2 用，v1 完全没 clip**。
- 建议：v2 增加 IC 偏度/峰度诊断列 + bootstrap 95% CI（重采样 1000 次算 IC 均值分位）。

---

## 4. 其他优化点

### 4.1 forward_return 公式

**实现**：`close[t+N] / open[t+1] - 1`（v1 第 87-91 行，v2 `forward_return.py` 第 16-19 行）。
- ✅ 公式正确：信号日 t 收盘后次日开盘买入，t+N 日收盘卖出，与回测 `backtest/execution.py` 一致。
- ⚠️ **未用 clean_ret**：forward_return 是"未来 N 日实际持有收益"，本来就应该用原始价格（涨停日卖不掉是真实摩擦，回测会处理）。但**没有应用涨跌停 mask 剔除买入日一字涨停的股票**——这类股票 open[t+1] = close[t+1] = 涨停价，买入信号无法成交，forward_return 却正常算出一个值，会污染 IC。
  - v2 的 `build_ic_tradability_mask` 在 IC 截面层剔除了涨跌停日（`universe.py` 第 64-68 行），但 v1 没有。
  - 建议：v1 也接 `IC_APPLY_TRADABLE` 配置（或直接把 driver 切到 v2）。

### 4.2 IC 计算用 Spearman 还是 Pearson

- ✅ 全程用 Spearman（`compute_ic_series` 用 `rank().corrwith(rank())`，`ic_analysis.py` 第 113-115 行；v2 `ic_series.py` 第 24-46 行）。
- ✅ NaN 处理：`na_option="keep"` + `valid_count >= MIN_IC_STOCKS=30` 才算该截面（v2 第 41、45 行；v1 无 min_stocks 检查）。
- ⚠️ **v1 缺 min_stocks 检查**：早期调仓日股票数 <30 时 IC 仍会被算出来并进入汇总，噪声大。v2 已修复。

### 4.3 股票池 PIT 同步

- ✅ v2 `universe.py::build_ic_tradability_mask` 第 70-79 行：M4 修复后 ST 用时间序列 `build_st_schedule`（按日期精确查询），不是静态 ST 集合。
- ❌ **行业映射仍是静态当前值**（见 §2.1 缺陷 3）——PIT 不一致。
- ❌ **`IC_MIN_LISTING_DAYS=0` 默认禁用上市天数过滤**（`settings.py` 第 59 行）——次新股高波动会污染 IC。建议改为 60-252。
- ❌ v1 完全没有可交易池 mask，ST/涨跌停/停牌股全部进入 IC 计算，**系统性高估因子有效性**（特别是流动性因子，停牌股流动性=0 会拿到极端高/低分）。

### 4.4 并行实现

- ✅ `_run_bounded_parallel` / `run_bounded_parallel` 用 ThreadPoolExecutor + 有界提交（`ic_analysis.py` 第 142-180 行；`parallel.py` 第 5-40 行），避免一次性 submit 全部因子爆内存。
- ⚠️ 用线程而非进程：numpy/scipy 在 rank/corr 上释放 GIL，线程并行有效；但 pandas 操作持有 GIL，加速有限。
- ⚠️ 默认 `IC_MAX_WORKERS=1` / `BARRA_IC_WORKERS=1`（32GB 机器串行）——保守但慢。建议 64GB 机器开 2-4。
- ❌ 无 joblib Memory 缓存：每次跑 IC 都重算因子 + 重算 forward_return。`factors/factor_cache.py` 已存在但只在 ML 流程用，IC 流程没用。建议 IC 也接 factor_cache。

### 4.5 输出 JSON 元数据

当前 v1 JSON（`ic_analysis.py` 第 1151-1158 行）：
```json
{
  "horizon": 20,
  "lookback_years": "full",
  "ic_start_date": "all",
  "factors": [...],
  "excluded": {...},
  "ic_stats": {"IC均值": {...}, "ICIR": {...}, "胜率": {...}}
}
```

**缺失**：
- `universe_size`：每个调仓日平均参与 IC 计算的股票数（验证样本量）
- `ic_series_length`：IC 序列长度 N（验证 t 检验自由度）
- `sample_period`：[first_date, last_date]（验证回测时段对齐）
- `barra_factors_used`：用了哪些 Barra 控制变量（可复现性）
- `industry_reference`：行业哑变量参照类（可复现性）
- `nw_lag`：Newey-West 用的 lag 数（v2 才有）
- `config_snapshot`：IC_CLIP / IC_CORR_METHOD / IC_RANK_METHOD 等关键配置（可复现性）

v2 `io.py` 第 39-48 行已加 `engine="v2"` 和 NW_t / IC_after_cost 列，但仍缺上述元数据。**建议**在 `save_results` 入口加 `meta` 段。

### 4.6 多 horizon IC 对比

- ❌ **无跨 horizon 对比**：当前 h5 和 h20 各跑一次，输出独立 JSON。无法回答"动量_20d 在 h5 还是 h20 更有效"。
- ✅ `ic_decay_table` 提供"同一因子 × 多持仓期"的 IC 均值表（`ic_analysis.py` 第 183-206 行；v2 `decay_corr.py`），但只在 `--decay` 时打印，不写入 JSON。
- 建议：driver 跑完 h5/h20/h10 后，加一个 `compare_horizons.py` 汇总脚本，输出"因子 × horizon"矩阵 + 推荐 horizon 标注。

### 4.7 v1/v2 双轨遗留

- v1 `ic_analysis.py` 仍有 1200 行活代码，且 driver 在用。
- v2 `research/ic/` 已模块化且统计更严谨，但未上线。
- **建议**：driver 切到 v2 后，v1 标记 `@deprecated`，6 个月后删除。AGENTS.md 已声明"v1/v2 双轨未迁移"，此审阅确认应启动迁移。

### 4.8 Barra 因子面板完整性检查缺失

- `get_barra_factors` 在 `barra_risk.py` 第 187-241 行：每个 Barra 因子独立计算，缺数据时跳过（如无 `pb` 列则跳过 Value）。
- 但 `precompute_ctrl_matrices` 第 77-81 行：只有当某个 Barra 因子在该日有数据才加入控制矩阵——**不同调仓日的控制变量集可能不同**（如早期段无 Value 列），导致纯 IC 的"纯度"随时间不一致。
- 建议：在 `run_barra_pure_ic` 入口加一致性检查，若某 Barra 因子在 <80% 调仓日有数据，整列剔除或报警。

### 4.9 `industry_map_df["sw_l2"]` 硬编码

- `load_data.py` 第 94 行、`ic_analysis.py` 第 955 行：硬编码取 `sw_l2` 列。
- 若 `industry_map.parquet` 含 `sw_l1` / `sw_l2` / `sw_l3` 多列，无法通过 CLI 切换。
- 建议：加 `--industry-level` 参数，对应 `INDUSTRY_LEVEL` 配置。

---

## 5. 优先级排序的 TODO 清单

### P0（必须改，影响结果正确性）

| # | 任务 | 文件:行号 | 改动量 | 难度 | 预期收益 |
|---|------|----------|--------|------|---------|
| P0-1 | **driver 切到 v2**：`logs/driver.py::run_ic` 第 137 行 `research.ic_analysis` → `research.ic_analysis_v2`；`run_ic_batch` 第 233 行同步 | `logs/driver.py:137,233` | 2 行 | 极低 | **极高**——立即享受 NW t / 可交易池 / 扣成本 / 稳定性 |
| P0-2 | **ML 特征中性化层**：新增 `residualize_panel`，在 `build_factor_dataset` 出口可选中性化因子（与 IC 纯 IC 同口径） | `models/wf/labels.py` 新增 + `strategies/ml.py:build_factor_dataset` 出口 | ~80 行 | 中 | **极高**——修复 IC 筛选与 ML 训练口径不一致（§2.1 C） |
| P0-3 | **v1 接可交易池 mask**：v1 `run()` 调 `build_ic_tradability_mask` 并传给 `compute_ic_series` | `ic_analysis.py:918,985` | ~15 行 | 低 | 高——避免高估流动性因子等（若不切 v2 则必须做） |

### P1（建议改，提升稳健性）

| # | 任务 | 文件:行号 | 改动量 | 难度 | 预期收益 |
|---|------|----------|--------|------|---------|
| P1-1 | **Benjamini-Hochberg FDR 校正**：v2 `select_factors` 增加 FDR 模式 | `research/ic/statistics.py` 新增 + `research/ic/selection.py:74` | ~30 行 | 低 | 高——应对多因子多重检验假阳性 |
| P1-2 | **Barra_Beta / Barra_ResVol 用 clean_ret** | `factors/barra_risk.py:99-137` | ~15 行 | 低 | 中——避免涨跌停污染 beta/残差波动 |
| P1-3 | **Barra 因子自身行业中性化** | `factors/barra_risk.py:239` 出口 | ~10 行 | 低 | 中 |
| P1-4 | **行业映射 PIT 时间序列**：构建 `industry_map_panel.parquet`，按截面日期取当期行业 | `data/industry/download_industry.py` + `research/ic/barra.py:_industry_dummies:26` | ~60 行 | 中 | 中——长样本早期段消除未来信息 |
| P1-5 | **rolling ICIR**：v2 `ic_stability_metrics` 增加 `rolling_icir = rolling_mean/rolling_std` | `research/ic/statistics.py:99` | ~10 行 | 低 | 中 |
| P1-6 | **IC 衰减表加 ICIR / t / half-life 列** | `research/ic/decay_corr.py:ic_decay_table:11` | ~25 行 | 低 | 中——判断因子最适合的持仓期 |
| P1-7 | **JSON 元数据补全**：universe_size / ic_series_length / sample_period / config_snapshot / nw_lag | `research/ic/io.py:save_results:12` | ~25 行 | 低 | 中——可复现性 |
| P1-8 | **`IC_MIN_LISTING_DAYS` 默认改 60-252** | `config/settings.py:59` | 1 行 | 极低 | 中——剔除次新股高波动噪声 |
| P1-9 | **t 阈值可配置 + 默认收紧到 2.5** | `research/ic/cli.py:297` + `research/ic/selection.py:80` | ~10 行 | 低 | 中 |

### P2（可选，提升效率/可读性）

| # | 任务 | 文件:行号 | 改动量 | 难度 | 预期收益 |
|---|------|----------|--------|------|---------|
| P2-1 | **QR 缓存**：`precompute_ctrl_matrices` 预算 Q，`compute_pure_ic_fast` 用 `resid = f - Q@(Q.T@f)` | `research/ic/barra.py:39,106` | ~40 行 | 中 | 中——纯 IC 提速 3-5× |
| P2-2 | **IC 流程接 factor_cache**：`research/ic/cli.py:run` 加 `use_factor_cache` 选项 | `research/ic/cli.py:99` | ~30 行 | 低 | 中——重复跑 IC 节省 50%+ 时间 |
| P2-3 | **IC 偏度/峰度/bootstrap CI** | `research/ic/statistics.py` 新增 | ~30 行 | 低 | 低——诊断非正态 |
| P2-4 | **多 horizon 对比汇总脚本** | `research/compare_horizons.py` 新建 | ~60 行 | 低 | 中——跨周期决策 |
| P2-5 | **`--industry-level l1\|l2\|l3` 选项** | `config/settings.py` + `research/ic/load_data.py:94` | ~20 行 | 低 | 低——灵活性 |
| P2-6 | **v1 标记 deprecated + 6 个月后删除** | `research/ic_analysis.py` 顶部 | 1 行 | 极低 | 中——减少双轨维护成本 |
| P2-7 | **Barra 面板完整性检查**：缺失 >20% 的 Barra 因子剔除或报警 | `research/ic/barra.py:run_barra_pure_ic:188` | ~15 行 | 低 | 低 |
| P2-8 | **进程并行替代线程并行**（64GB 机器） | `research/ic/parallel.py` | ~30 行 | 中 | 低-中 |

---

## 6. 审阅摘要 + 关键发现

### 摘要

IC 分析模块**架构方向正确**——v2 的模块化、Newey-West、可交易池、扣成本、稳定性诊断都是业内标准做法。**主要问题在于 v2 没有真正上线**：生产 `logs/driver.py` 仍硬编码调用 v1，所有 v2 改进沦为"测试代码"。最大的实质性缺陷是 **IC 筛选用纯 IC（Barra 残差化）作为判据，而 ML 训练喂的是未中性化的原始因子**，两套口径不一致会让模型学到系统性风险敞口而非真 alpha。

### 三大关键发现

1. **【P0】IC/ML 口径不一致**：IC 筛选判据用纯 IC（剔除 Barra + 行业），但 `strategies/ml.py::build_factor_dataset` 出口的因子未做任何中性化，`label_mode` 默认 `cs_zscore`（连标签残差化都没开）。建议新增 `residualize_panel` 在 ML 特征层做可选中性化，与 IC 同口径（§2.1 C、§5 P0-2）。

2. **【P0】v2 改进未进入生产**：v2 已实现 Newey-West HAC t、可交易池 mask、IC 稳定性、扣成本 IC——但 `logs/driver.py:137` 仍调 v1。两行改动即可让生产享受全部 v2 改进（§5 P0-1）。

3. **【P1】多重检验校正缺失**：30+ 因子各做 t 检验，整体假阳性率近 80%。v2 已有 NW t 但无 FDR/Bonferroni 校正。建议加 Benjamini-Hochberg FDR 模式（§3.4、§5 P1-1）。

### 次要发现

- Barra_Beta / Barra_ResVol 用 `prices.pct_change()` 而非 `clean_ret`，涨跌停日污染 beta 截面排名（§2.1 缺陷 2）。
- 行业映射是静态当前值，长样本早期段存在 PIT 泄漏（§2.1 缺陷 3）。
- 行业用 sw_l2（~130 类）而非 sw_l1，小行业哑变量近乎共线（§2.1 缺陷 4）。
- `IC_MIN_LISTING_DAYS=0` 默认禁用上市天数过滤，次新股噪声进 IC（§4.3、§5 P1-8）。
- v1 `compute_ic_series` 无 min_stocks 检查，早期段小样本 IC 仍进汇总（§4.2）。
- 输出 JSON 缺 universe_size / ic_series_length / sample_period / config_snapshot，可复现性不足（§4.5）。
- 无跨 horizon IC 对比汇总（§4.6）。
- Barra 控制矩阵未缓存 QR 分解，每个因子 × 每个调仓日重复 lstsq（§2.1 缺陷 5）。

### 建议执行顺序

1. **本周**：P0-1（driver 切 v2，2 行）→ 立即获得 80% 收益
2. **下周**：P0-2（ML 特征中性化层）+ P1-1（FDR 校正）+ P1-2（Barra 用 clean_ret）
3. **本月**：P1-4（行业 PIT）+ P1-5/P1-6（rolling ICIR / IC 衰减统计）+ P1-7（JSON 元数据）
4. **长期**：P2 全部 + v1 删除
