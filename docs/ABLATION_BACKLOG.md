# Ablation Backlog & Next Step

> **状态**：下文 ablation 均为 **backlog**，非当前 sprint。  
> **当前优先**：先做持仓策略层（见文首 NEXT），再回头做小而有意义的 ablation。

---

## Canonical IC screening（h5 / h20 必须同参）

生产 IC 白名单（`selected_factors_h{5,20}.json` → YAML）**必须**用同一套 CLI；
`corr-dedup` 默认 ON（剔除冗余 WQ/波动率克隆）。关闭：`--no-corr-dedup`。

```powershell
# h5
.\.venv\Scripts\python.exe -m research.ic_analysis_v2 --period 5 `
  --barra --save --use-fdr --t-threshold 2.5 --gram-schmidt

# h20（同参，只改 --period）
.\.venv\Scripts\python.exe -m research.ic_analysis_v2 --period 20 `
  --barra --save --use-fdr --t-threshold 2.5 --gram-schmidt
```

| 旋钮 | 生产默认 | 说明 |
|------|----------|------|
| `--barra` | ON（显式） | Barra 纯 IC 门槛 |
| `--use-fdr` | ON（显式） | BH-FDR 多重检验 |
| `--t-threshold 2.5` | 2.5 | NW_t / FDR 配套 |
| `--corr-dedup` | **ON（默认）** | 截面相关去重；`--no-corr-dedup` 关闭 |
| `--gram-schmidt` | ON（显式） | 正交精筛 → `factors_orth`（dynamic） |
| `--corr` | OFF | **仅打印**相关矩阵，不控制去重 |

`meta.config_snapshot` 会记录 `barra / use_fdr / t_threshold / corr_dedup / corr_threshold / gram_schmidt`。
历史坑（h20 pre-GS 64 vs h5 31）：
1. 全局 `dropna` 在 60+ 候选时截面塌缩 → corr 矩阵空 → 去重静默跳过；已改 pairwise `min_periods`。
2. pairwise 引入 NaN 后，`max([nan, …])` 聚合失效；已改有限值过滤 / `nanmax`。

---

## NEXT / 优先：持仓策略（Holdings Strategy）

**目标**：模型得分 → 可交易仓位 → 扣成本后 Sharpe / IR / maxDD。  
验证「当前 IC 能否撑起高 Sharpe 策略」，而非再堆模型 ablation。  
这是 **组合层**，不是新一轮模型训练。

### 策略旋钮（后续可落地）

| 旋钮 | 示例 |
|------|------|
| 持仓规模 | topN = 30 / 50 / 100 |
| 权重 | 等权 vs 得分加权（score-rank / softmax） |
| 调仓规则 | 周频全换；换手率上限（如 50%）；跌出 topK 才卖 |
| 风控 | 现金缓冲；regime 降仓/haircut；单票上限 |
| 评价 | 对齐日历；用现有 ridge/dynamic 得分；报 **策略 Sharpe**，不只看 Q5 |

### v0 可先实现的规格（3–5 条）

- [ ] **v0-A**：ridge 得分 → Top50 等权 → 周频全调仓 → 扣成本 NAV/Sharpe
- [ ] **v0-B**：Top100 等权 + 单期最大换手 50%（复用 `apply_turnover_control`）
- [ ] **v0-C**：ridge + dynamic 排名 blend → Top50 等权周频
- [ ] **v0-D**：Top50 等权 + 跌出 Top100 才卖（rank keep）
- [ ] **v0-E**：v0-A + 单票上限（如 5%）/ 可选 10–20% 现金缓冲

### 今天已有 vs 仍需设计

| 已有（`backtest/`） | 仍需 |
|--------------------|------|
| `select_top_n` + 等权；quantile 回测自带 **Top100** track | 现金缓冲作为一等公民；显式「策略产品」报告层 |
| `apply_turnover_control`（换手上限 + rank keep） | 权重层换手（相对上期 w 的 L1）软约束 |
| `--portfolio-opt`：ew/score/rank/mv/invvol/rp + `--max-weight`（见 [PORTFOLIO_OPT.md](PORTFOLIO_OPT.md)） | Barra 风险模型 QP；多策略一键对照表 |
| `--position-regime` 敞口缩放（见 [POSITION_REGIME.md](POSITION_REGIME.md)） | 与 opt 联用的 ablation 表（策略 Sharpe） |
| `risk_metrics`：年化/波动/Sharpe/maxDD（含 Top100） | 以策略 Sharpe 为主指标的对照表（非 Q5 叙事） |
| `export_holdings` | 多策略对照（EW vs score-wt、N、换手）一键跑 |

**今日可粗跑**：现有 `quantile` 回测已输出 Top100 等权 NAV + 风险指标，可先看 ridge/dynamic 的 Top100 Sharpe 作下限参考。  
**设计缺口**：偏交易的持仓策略（现金、部分调仓、cap、blend）尚未独立成「策略产品」层。

---

## Backlog：小而有意义的 Ablation

> 帮助理解系统，但对实盘交易优先级低于持仓策略。勾选表示「值得做」，非正在做。

### Features

- [x] ~~regime 特征 on/off（`市场*`/`HMM_*` 广播进 X）~~ — **已退役**；ML 默认不注入
- [x] ~~CS regime（`轮动_*`）on/off~~ — **已退役**；见 [POSITION_REGIME.md](POSITION_REGIME.md)
- [ ] **仓位体制 `--position-regime` on/off**（回测敞口缩放，非 ML 特征）：
  见 [POSITION_REGIME.md](POSITION_REGIME.md)。Ablation：默认（满仓）vs `--position-regime`；可选 `--force-exposure 0.5`
- [ ] **special factors** on / off（统一入口 `--special-factors` / `--inject-factors`；见 [SPECIAL_FACTORS.md](SPECIAL_FACTORS.md)）
  - [ ] `event` pack on / off（旧 `--event-overlay` 已 deprecated → `event`）
  - [ ] `size` pack on / off（绕过 IC YAML，强制注入市值 alpha）
  - [ ] `event,size` 组合 vs 单 pack
- [ ] `--feature-neutralize` on / off
- [ ] `label_mode=barra_residual`（**bug 已修，从未跑过有效 ablation**）

### Factors

- [ ] GS / orth 筛选 vs pure IC top50
- [ ] 因子数量：30 / 50 / …
- [ ] WorldQuant 冗余因子剔除（相关去重再严一档）
- [ ] special inject vs 把同名因子写进 YAML（口径对照）

### Models

- [ ] ridge vs dynamic lookback（3m / 6m / 12m）
- [ ] **两阶段 ridge**（`--two-stage --stage2-pool-frac 0.2 --top-n 100`）：
  S1 全市场 WF → 每期 S1 Top20%（≈Q5）池 → 历史日各自池内 `winsor→cs_zscore`（label+特征）
  再拼接 rolling ridge（S2）→ 池内 Top100。实现：`models/wf/two_stage.py`；
  S1 universe cache：`results/<tag>/stage1_cache/`（scores+mask+meta，不含 X/y）；
  池 EW 对照单段 Q5 验口径。Ablation：单阶段 ridge vs two-stage（同因子 YAML + feature-neutralize + event）。
- [ ] lgbm：MSE vs rank（rankloss / qcut 标签）
- [ ] Optuna 调参（冻结基线后再扫）
- [ ] ensemble / `--blend-dynamic`
- [ ] ~~cat / xgb~~ — **除非树模型 MSE 稳定打过 ridge，否则先跳过**

### Horizon / Windows

- [ ] horizon：h5 / h10 / h20
- [ ] train windows 长短对比
- [ ] dynamic lookback：3m / 6m / 12m

### Universe

- [ ] 全市场 vs `small_mid`
- [ ] 行业子集 / 市值分档

### Backtest plumbing（已完成，备忘）

qcut ties、`Top100` track、年化收益口径等回测基建已落地；ablation 时直接复用即可。
