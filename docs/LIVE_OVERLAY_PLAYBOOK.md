# 实盘套壳 Playbook（内部备忘录）

> 日期：2026-08-04  
> 定位：把**现有**截面 ML 策略（非全自动交易）套上可落地的执行/宇宙/组合/敞口/人工菜单层。  
> 原则：只写仓库里能对上的旋钮与结果数字；激进 ≠ 加杠杆瞎冲，而是**可控地放大可兑现的超额形态**。

---

## 0. 先锚定：当前主线是什么

### 0.1 生产配方（截至 2026-08-04）

| 维度 | 现状（有日志/CSV 支撑） |
|------|-------------------------|
| 模型 | **ridge**（主对照）/ **cat**（树轨对照）；非 YetiRank |
| 标签 | `--label-mode cs_rank` |
| 特征 | 默认 `--feature-neutralize`（Barra+行业残差；sparse pack **豁免**残差化） |
| 期限 | `--horizon 5`，调仓 **W-FRI** |
| WF | train 默认 `6,12` 月；**val=2m 共用**（≈9 个 W-FRI 期）——`ridge_h5_barra_em_sharedval2m_20260804` 已落地；旧 `ridge_h5_barra_em_20260730_cs_rank` 日志为 `val=26` 期，**不可与 sharedval2m 直接比「同口径」** |
| 因子池 | dense YAML 白名单 + `--special-factors sparse`（约 26 个语义/事件因子） |
| 成本 | `BID_ASK_SPREAD_BPS=10` + 佣金/印花税；`SLIPPAGE=0`（资金量 ~200 万假设） |
| 训练可交易池 | research 默认（信号日**保留**涨跌停样本） |
| 回测得分宇宙 | 默认 `--bt-score-universe strict`（买日前剔涨跌停 ∩ label-exec 门控） |
| 持仓度量 | Top100 + Q1–Q5；`N_STOCKS=100` 是**技能度量**，不是实盘必买 100 |
| 产品定位 | **辅助人工选股**，不做全自动下单（`AGENTS.md`） |

### 0.2 代表性结果（扣 bid-ask 后，CSV）

| 实验目录 | Top100 ann / Sharpe / MDD | Q5 ann / Sharpe / MDD | 备注 |
|----------|---------------------------|------------------------|------|
| `ridge_h5_barra_em_20260730_cs_rank` | **13.5% / 0.55 / −31.1%** | 11.7% / 0.53 / −26.9% | 旧 val 窗；2025 Top100 +65.8%，2026 −15.6% |
| `ridge_h5_barra_em_sharedval2m_20260804` | **8.4% / 0.35 / −32.8%** | 10.8% / 0.49 / −27.8% | 现行 val=2m；**Top100 劣于 Q5**；2024 Top100 −21.0% |
| `cat_h5_nolongshare_cs_rank_tune_20260804` | **12.5% / 0.56 / −34.3%** | 10.0% / 0.49 / −27.9% | Top100 多数年份 > Q5；2024 −22.7%；**2026 中后期 score 塌缩** |
| `ridge_h5_2026alive_w13_val2_20260804` | 2026 cum Top100 **−20.9%**（诊断） | Q5 cum −13.7% | 仅 2026 活体窗；有 IC、多头未兑现超额 |

**换手（Top100）**：ridge/sharedval 均值约 **0.44**/周；cat 约 **0.46**/周。周频一手买卖≈40–50% 名义换手，实盘若跟满 Top100 成本与执行摩擦不可忽视。

### 0.3 mcap100 实验状态（未完成）

| 步骤 | 状态 |
|------|------|
| 短名单 `research/output/shortlist_mcap100_20260804/`（120 因子） | ✅ 完成 |
| IC ckpt：`*_cap_micro_small_100_tmr_v2.pkl`（barra_pure / ic_series / summary / yearly） | ✅ 有落盘 |
| `selected_factors_*mcap*` JSON / `config/factor_configs_*mcap*` YAML | ❌ 未见 |
| `results/*mcap*` / `*micro_small*` ML 回测目录 | ❌ 无 |

结论：**宇宙壳（`--cap-band micro_small_100`）的 IC 已跑，训练/回测未出**——下文「激进配方」把 mcap100 标为 *待验*，不可当已证实 alpha。

---

## 1. 现状诊断：为什么「想激进却激进不了」

### 1.1 框架本身在「压风格」

- `--feature-neutralize` 把 dense 特征做成 **Barra Size 中性**；仓位层 `size_style`（SMB）在 `position_regime` 里**只记录、不进敞口合成**（`docs/POSITION_REGIME.md`）。  
- 结果：模型擅长挖 **中性截面排序**，不擅长「小盘贝塔 + 高 beta」那种账面上好看的激进净值。  
- 实证侧：cat 最近一期持仓流通市值分位中位≈0.37（略偏小），但这是**排序后的残余倾斜**，不是显式 size 杠杆。

### 1.2 Top100 ≠ 更稳的激进；坏年更脆

| 事实 | 含义 |
|------|------|
| sharedval2m：Top100 Sharpe **0.35** < Q5 **0.49**；2024 Top100 −21% vs Q5 −12.6% | 把「更激进」等同于 Top100 会在弱市放大左尾 |
| 全样本 Top100 MDD **31–34%** | 个人账户若满仓跟 Top100，心理与资金曲线不允许「永远满敞口」 |
| 2025 大年（Top100 +60%↑）vs 2024/2026 差年 | 收益形态高度 **regime 依赖**；无 L3 敞口壳时，激进=赌小盘/题材年 |

### 1.3 2026：截面还有一点力，多头组合没兑现

`ridge_h5_2026alive_*`：`ic_2026 mean≈0.015, ICIR≈0.36`，但 Top100/Q5 均跑输等权基准。  
cat 诊断：`2026-06-18` 仅 6 个唯一分、最高分并列约 41 只；`2026-07-03` 约 75% 股票卡在最高分——**Top100 退化为代码序补齐**。激进加仓在这种日子=噪声。

### 1.4 一句话

> 当前主线是「Barra 中性 + 周频截面排序 + 人工筛」；  
> **激进不了**的主因不是缺一个更猛的模型，而是：(a) 风格被中性化；(b) TopN 在坏年比 Q5 更脆；(c) 无已验证的杠杆/宇宙壳；(d) 2026 信号偶发塌缩时加仓无意义。

---

## 2. 已有 / 缺 旋钮表

| 套壳能力 | 状态 | CLI / 模块 | 实盘可用性 |
|----------|------|------------|------------|
| 买日一字涨停拦截、卖日涨跌停、次日开盘 | **已有** | `backtest/execution` + `open_` | L0 直接复用逻辑 |
| bid-ask / 佣金 / 印花税 | **已有** | `--bid-ask-spread`，settings | 小资金默认 10bp 够用 |
| research vs 回测 strict 宇宙 | **已有** | `--bt-score-universe` / `--tradable-strict` | 实盘名单必须走 **strict 可下单** |
| 市值带宇宙 | **已有** | `--cap-band`（含 `micro_small_100`） | L1；mcap100 ML **未验** |
| TopN | **已有** | `--top-n`（默认 100） | L2；实盘建议 20–40 |
| 组合加权 | **已有** | `--portfolio-opt ew\|score\|rank\|invvol\|mv\|rp` + `--max-weight` | L2；小资金优先 ew/rank/invvol |
| 换手约束 | **已有** | `--turnover-limit` / `--rank-change-threshold` | L2；周频建议开 |
| special factors | **已有** | `--special-factors sparse\|size\|event` | 激进时可加 `size`（跳过中性化） |
| val 窗 | **已有** | `--val-window`（默认 6） | 训练口径，非实盘动作 |
| 仓位体制（0.3–1.0） | **已有** | `--position-regime` / `--force-exposure` | L3 **半成品**：历史实验 mean exposure≈0.64 |
| 杠杆 >1 / 1.5× | **缺** | `force_exposure` 硬限制 `[0,1]` | 需开发才谈 1.5 |
| Moreira 式 vol-target（∝ 1/σ²） | **缺** | 现有是离散计分，非连续 vol 缩放 | M |
| 单票 ADV 参与率上限 | **缺** | 无 | S–M（实盘必补） |
| 平行策略菜单 + 人工切换 UI/脚本 | **缺**（概念可手工） | 多 `results/<tag>` + holdings CSV | S（流程）/ M（半自动） |
| 全自动 meta（IC 加权切策略） | **缺且不建议先做** | — | L；PBO/DSR 已提示过拟合 |
| 券商 API 下单 | **缺（刻意）** | — | 产品定位外 |

---

## 3. 套壳分层架构（可落地）

```
人工菜单(L4) → 风险预算(L3) → 组合构造(L2) → 宇宙壳(L1) → 可交易执行(L0)
                     ↑
              截面分数 factor_scores（ridge/cat）
```

### L0 — 可交易执行

| | |
|--|--|
| **输入** | 信号日候选代码 + 次日 open / masks / ST / 停牌 |
| **输出** | 可下单名单；买不进一字涨停、卖不出涨跌停的剔除 |
| **现有** | 回测 execution；`mask_scores_for_backtest`；`--bt-score-universe strict` |
| **缺** | 券商路由（不在范围内）；实盘用 holdings CSV 人工下单即可 |
| **工作量** | 流程文档化 **S** |

**动作**：每周五收盘后出分 → 次一交易日开盘前人工确认 → 开盘价附近成交；涨停买不进则放弃，不追板。

### L1 — 宇宙壳

| | |
|--|--|
| **输入** | `circ_mv` / `amount` 日频面板 |
| **输出** | `eligible_mask`（PIT） |
| **现有** | `--cap-band`：`all` / `small` / `small_mid` / `small_mid_wide` / `mid` / `micro` / `micro_small_100`；另有 20d 成交额 ≥2000 万 |
| **缺** | mcap100 上 **IC→YAML→ML** 闭环；壳股/注册制后壳溢价变化的定期复检 |
| **工作量** | 跑完 mcap100 训练 **M**；日常切换 band **S** |

**映射证据**：Liu–Stambaugh–Yuan (2019) 强调中国最小 30% 含 **壳价值**，构造 size 因子时应剔除——与盲目 `micro_small_100`（lower=0）**张力很大**。激进宇宙≠无脑下沉微盘。

**推荐默认实盘宇宙**：`small_mid` 或 `small_mid_wide`（有壳股地板）；`micro_small_100` 仅作平行实验轨。

### L2 — 组合构造壳

| | |
|--|--|
| **输入** | 分数截面（strict 可交易） |
| **输出** | 持仓集合 + 权重 |
| **现有** | `--top-n`；`--portfolio-opt`；`--max-weight`；`--turnover-limit`；`--rank-change-threshold` |
| **缺** | ADV% 上限；行业/单主题硬顶（非 Barra QP） |
| **工作量** | ADV 过滤 **S–M**；行业帽 **M** |

**实盘默认建议（非回测 Top100）**：

```text
--top-n 30 --portfolio-opt ew --max-weight 0.08 \
--turnover-limit 0.35 --rank-change-threshold 0.15
```

含义：约 30 票等权、单票≤8%、周换手封顶、排名小幅波动不换仓。回测 Top100 继续作**技能仪表**，不直接当订单簿。

### L3 — 风险预算 / 杠杆壳

| | |
|--|--|
| **输入** | 中证全指 trend/vol/breadth（已实现）；可选策略自身 20d σ |
| **输出** | `target_exposure ∈ {0.5, 1.0}`（近期）；远期可扩到 `{0.5,1.0,1.5}` |
| **现有** | `--position-regime`：`exposure = 0.3 + 0.7 × score/3`；`--force-exposure` 人工覆盖；历史 mean≈**0.64** |
| **缺** | (1) 连续 vol-target（Moreira–Muir）；(2) exposure>1；(3) 用**策略净值**而非仅指数 vol；(4) 与 L4 菜单联动的状态机 |
| **工作量** | 把 `force_exposure` 扩到 >1 并加融资约束校验 **M**；真正 vol-target **M**；完整融资实盘 **L** |

**近期可执行规则（不写新代码也能做）**：

| 条件（信号日，PIT） | 动作 |
|--------------------|------|
| `position_regime` 合成 exposure ≤0.53 **或** 中证全指 20d σ > 1.2×252d 中位数 | 账户股票仓位 **0.5** |
| exposure ≥0.77 且 最近 4 周策略 IC/多空未翻脸 | 仓位 **1.0** |
| cat/ridge 出现「唯一分 < 100」或 max_tie>30% | **强制 0.5**，且本周只人工精选 ≤10 只 |
| 想上 1.5× | **禁止**，直到代码支持且单独回测通过 |

证据锚点：Moreira & Muir (2017) — 高波动时降风险可抬 Sharpe；你们已有离散版，连续版是增强不是前提。

### L4 — 人工菜单（平行策略）

| | |
|--|--|
| **输入** | 2–3 条平行配方的 holdings + 当年/近季诊断 |
| **输出** | 本周采用哪条轨 + 敞口档位 |
| **现有** | 多 `results/<tag>/holdings_*.csv`；`recent_top100_picks.md` 类诊断 |
| **缺** | 一页菜单脚本（打印：轨 A/B/C 本周 Top30、重合度、score 健康度、regime exposure） |
| **工作量** | 菜单脚本 **S**；自动 meta 切换 **L（反对先做）** |

**规则**：人做切换，模型做排序。禁止用滚动 IC 自动在 ridge/cat/mcap 间每周切换（样本内过拟合；仓库 PBO/DSR 已警告）。

---

## 4. 推荐 2–3 条平行实盘配方

资金假设：`INITIAL_CAPITAL≈200万`；人工二次筛选；周频。

### 配方 A — `CORE_NEUT`（默认主仓，稳健）

| 项 | 参数 |
|----|------|
| 模型 | `ridge` + `cs_rank` + feature-neutralize + sparse |
| 参考结果 | `ridge_h5_barra_em_sharedval2m_20260804`（训练口径）；技能看 Q5 |
| 宇宙 | `--cap-band all` 或 `small_mid_wide` |
| 组合 | Top**30** ew，`--turnover-limit 0.35` |
| 敞口 | `--position-regime` 或人工 0.5/1.0 |
| 预期形态 | 年化波动 ~18–22%；MDD 目标压到 **~20%**（靠 L3）；超额来自中性排序，不靠小盘年 |

### 配方 B — `TREE_TILT`（卫星 / 进攻对照）

| 项 | 参数 |
|----|------|
| 模型 | `cat` + `cs_rank` + neutralize + sparse（nolongshare YAML） |
| 参考结果 | `cat_h5_nolongshare_cs_rank_tune_20260804` |
| 宇宙 | `all`；组合 Top**25–30** |
| 硬门 | **score 健康检查**：`n_unique < 200` 或 `max_tie%>0.3` → 本周禁用本轨，回退 A |
| 预期形态 | 好年可打过 ridge Top；坏年/塌缩年左侧更狠（MDD 34% 量级）；**只作 0–40% 卫星** |

### 配方 C — `MICRO100_AGG`（激进轨，**待验**）

| 项 | 参数 |
|----|------|
| 模型 | 同 A，但 `--cap-band micro_small_100` + 专用 IC YAML（尚未产出） |
| 可选增强 | `--special-factors sparse,size`（显式放开市值信息） |
| 组合 | Top**20**，`--max-weight 0.10`，换手更严（0.25） |
| 敞口 | 默认 **0.5**；仅当 regime 满仓 **且** 近季未发生微盘踩踏时升到 1.0；**不做 1.5** |
| 预期形态 | 高波动、高换手敏感、壳/微盘左尾厚；可能在 2025 类年份爆发，在 2024/注册制壳溢价变化期失效 |
| 状态 | IC ckpt 有、ML **无** → 标注 **纸面激进，未获实盘准入** |

**哪条叫「激进」？** → **C**。在 C 未验通前，可执行的「相对激进」是：**B 作卫星 + L3 在 risk-on 用满 1.0**，而不是加杠杆或无脑微盘。

### 每周操作清单（硬）

1. 刷新数据 → 跑 A（必要）/ B（可选）出分。  
2. 打印 L3 exposure + score 健康度。  
3. 取 **strict** 可交易 ∩ Top30；人工剔除：即将解禁巨量、ST 边缘、你看不懂的纯题材。  
4. 按敞口档位下单；涨停买不进则现金留存，不挪到名单外追涨。  
5. 记录实际成交 vs 回测假设（开盘滑点），月度复盘一次。

---

## 5. 证据附录（调研 → 本仓库旋钮）

| # | 文献 / 来源 | 年份 | 结论一句 | 证据层级 | 接到本仓库 |
|---|-------------|------|----------|----------|------------|
| 1 | Liu, Stambaugh, Yuan — *Size and Value in China* (JFE) | 2019 | 中国 size/value 重要，但最小 30% 被壳价值污染，构造因子应剔除 | **顶刊实证** | L1：优先 `small`/`small_mid*`；`micro_small_100` 需单独证伪壳溢价 |
| 2 | Moreira & Muir — *Volatility-Managed Portfolios* (JF) | 2017 | 高波动降仓可抬多类因子 Sharpe | **顶刊实证** | L3：强化 `--position-regime` 的 vol 腿；远期做 ∝1/σ² |
| 3 | Moskowitz, Ooi, Pedersen — *Time Series Momentum* (JFE) | 2012 | 资产自身 1–12m 趋势可预测；极端市表现好 | **顶刊实证** | L3/L4：用指数 TSMOM 作**敞口开关**，不要灌回截面 X（旧 `regime-cs` 已退役） |
| 4 | Long et al. — *Beware of the crash risk* (Applied Economics) | 2019 | A 股尾部 beta 与收益关系异常，需警惕崩盘暴露 | **期刊实证** | L3 降仓 + L2 降集中度；反对无约束 TopN 加杠杆 |
| 5 | 壳/size 后续文献（如 Size effect in China, 2021） | 2021 | 剔最小 30% 后 size 溢价多被市场/价值解释；流动性与壳扭曲显著 | **期刊** | 与 #1 同向；注册制推进后微盘溢价不稳定 → C 轨必须滚动失效检验 |
| 6 | Baltussen 等 / 业界 crowding 文献；AEA「What Alleviates Crowding」 | 2021+ | 因子拥挤吃 capacity；多特征交易可对冲冲击 | **学术+业界** | L2 换手帽 + 多因子（已有）；资金小时 capacity 不是主矛盾，**周换手 0.45 才是** |
| 7 | A 股涨跌停「上游污染」类工作（如 2025 arXiv ML+mask） | 2025 | 涨跌停价进窗口会灌 IC、害实盘 Sharpe | **预印本** | 你们已有 `clean_ret` + execution mask；实盘坚持 **strict 名单**，勿用 research 训练池直接下单 |
| 8 | 本仓库 PBO/DSR（`research/pbo.py`） | 内部 | 历史最优 DSR≈0.025 → 超参选择偏差 | **内部实证** | **反对**全自动 meta 切轨 |

---

## 6. 明确反对

| 听着很激进 | 为什么不做 |
|------------|------------|
| 融资 1.5× 顶满 Top100 | 代码敞口≤1；Top100 MDD>30%；周换手~45%；与「辅助人工」定位冲突 |
| 把市场/HMM/`轮动_*` 灌回 ML X | 已退役；截面方差≈0 或与 neutralize 打架 |
| 无脑 `micro_small_100` 当主仓 | 与 LSY(2019) 壳价值结论冲突；ML 未出数 |
| 每周用滚动 IC 自动在 A/B/C 间切换 | meta 过拟合；DSR 已警告 |
| 用 YetiRank / rank objective「更激进」 | `cat_*_rank_tune` 已中止；现行是 regression+cs_rank |
| 关掉 feature-neutralize 追小盘年 | IC/ML 口径分裂；账面上的收益可能只是 Size 贝塔 |
| 实盘跟满回测 Top100 | `N_STOCKS=100` 是技能度量；执行与注意力不可扩展 |
| 在 score 大同分日加仓「博反弹」 | 2026-06/07 cat 证据：排序已坏 |

---

## 7. 开发优先级（若只做三件事）

1. **S**：`scripts/live_menu.py`（或等价）——读 A/B holdings + regime exposure + score 唯一值/并列率，打印本周菜单。  
2. **S–M**：实盘候选加 **ADV% 与金额** 过滤（例如单票下单 ≤5%×20d 均额）。  
3. **M**：`position_regime` 增加连续 vol-target；**另开**实验 tag 验证后再谈 exposure∈(1,1.5]。

mcap100：先出 `selected_factors` + YAML + 一条 ridge/cat 回测，再决定 C 是否进菜单。

---

## 8. 附录：主线复现命令（研究，非实盘下单）

```bash
# A 口径（shared val 2m）
python run.py --skip-download --mode ridge --horizon 5 \
  --label-mode cs_rank --feature-neutralize \
  --special-factors sparse --sparse-from-ic research/output/selected_factors_h5_barra_em_20260730.json \
  --factor-config config/factor_configs_h5_barra_em_20260730.yaml \
  --val-window 2 --bid-ask-spread 10 \
  --output-dir results/ridge_h5_barra_em_sharedval2m_20260804

# L3 套壳回测
python run.py ... --position-regime --top-n 30 --portfolio-opt ew \
  --turnover-limit 0.35 --output-dir results/ridge_h5_live_shell_A/

# L1 激进宇宙（仅研究）
python -m research.ic_analysis_v2 --period 5 --barra --save \
  --cap-band micro_small_100 --min-long-share 0
# → 再 YAML → run.py --cap-band micro_small_100 ...
```

---

*本文是内部操作备忘，不是收益承诺。数字均来自所列 `results/*/backtest_*_risk_metrics.csv` 与诊断 md；口径变更（Barra/市值源/bt-universe）后旧结论作废。*
