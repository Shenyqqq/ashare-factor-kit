# 组合权重优化（Portfolio Opt v1）

在 **已选** 股票集合（Q1–Q5 / TopN）上分配权重；不负责选股。  
默认 `--portfolio-opt ew`，与历史等权路径 **bit-identical**。

## 方法

| CLI | 含义 | 需要 Σ/σ |
|-----|------|----------|
| `ew` | 等权 | 否 |
| `score` | 得分正比（候选内 shift 非负） | 否 |
| `rank` | 截面 rank 正比 | 否 |
| `invvol` | 逆波动率 `w ∝ 1/σ` | 是（对角） |
| `mv` | 均值-方差 lite：`μ`←得分 z-score，`Σ`←Ledoit-Wolf | 是 |
| `rp` | 简易风险平价（scipy）；失败回退 `invvol` | 是 |

约束（v1）：long-only；权重和 = 1；可选 `--max-weight` 单票上限。  
换手约束仍用既有 `--turnover-limit` / `--rank-change-threshold`（控持仓集合，非权重 L1）。

实现：`backtest/optimize.py` → `quantile` 调仓后传入 `simulate_period(..., weights=...)`。  
`benchmark` track 始终等权。

## 与仓位体制组合

```
optimize → 权重和 = 1（invested book）
apply_exposure → r_eff = exposure × r_invested
```

两者可同时开；**不要**把 `target_exposure` 再折进权重和（会双重缩放）。  
仓位体制见 [POSITION_REGIME.md](POSITION_REGIME.md)。

## CLI

```bash
# 默认等权（与旧实验可比）
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml

# 逆波动 + 单票 10%
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml \
  --portfolio-opt invvol --max-weight 0.1 --top-n 30

# 得分加权 + 仓位体制
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml \
  --portfolio-opt score --position-regime
```

非 `ew` 时 tag 后缀 `_opt{method}`。

## 注意事项（请先读）

1. **Σ 估计嘈杂**：周频 A 股 TopN（尤其 N≤50）样本短、停牌/涨跌停多，Ledoit-Wolf 只能缓解不能根治。
2. **MV 经典但脆弱**：μ 用得分代理、Σ 用近期收益，小资金/小 N 上常不如 `invvol` / `rank` 稳。
3. **小资金优先**：实盘辅助选股场景更推荐 `ew` / `rank` / `invvol`；`mv`/`rp` 作对照 ablation。
4. **非 Barra QP**：无风格/行业风险模型约束；完整风险模型另议。
5. **收益面板**：优先传入 `clean_ret`（涨跌停日 NaN）；缺失时退回 `prices.pct_change()`。
