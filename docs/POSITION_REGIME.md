# 仓位体制（Position Regime）

> **取代**旧文档 `REGIME_FEATURES_CS.md`（`轮动_*` 截面 ML 特征）与
> 旧 `市场*` / `HMM_*` 广播进 ML 因子矩阵 X 的路径。二者均已退役。

## 设计原则

| 层级 | 用途 | 实现 |
|------|------|------|
| **Alpha / 选股** | 截面排序「买谁」 | 因子矩阵 X（YAML 白名单）；行业相对强度等已覆盖板块差异 |
| **仓位 / 总敞口** | 市场级标量「买多少」 | `backtest/regime.py` → `target_exposure` → quantile 回测缩放 |

- 不做 CS「轮动_*」再灌进 X（行业趋势已有 alpha；市值风格保持 Barra 中性）。
- 不做广播 `市场*`/`HMM_*` 进 X（截面方差为 0，对 ridge/排序无信息）。
- Size 风格（SMB）可选记录，**默认不进敞口合成**（v0）。

## 信号（PIT：全部 `shift(1)`）

实现：`backtest/regime.py::compute_position_regime`。

| 列 | 公式 | risk-on 条件 |
|----|------|----------------|
| `mkt_trend` | 中证全指 `close/MA60 − 1` | `> 0` → `trend_on=1` |
| `mkt_vol` | `σ_20 / median_252(σ_20)` | `< 1` → `vol_ok=1` |
| `mkt_breadth` | 个股站上 MA20 占比 | `> 0.5` → `breadth_on=1` |
| `size_style` | 小盘 − 大盘 20d 收益（需 `circ_mv`） | **仅记录** |

## 合成规则（v0）

```
score = I(trend_on) + I(vol_ok) + I(breadth_on)   # breadth 缺失时分母为 2
target_exposure = e_min + (1 − e_min) × score / n_parts
```

默认 `e_min = 0.30`。可选 `--force-exposure 0.5` 覆盖合成（人工降仓钩子）。

## 回测应用

对 **非 benchmark** track（Q1–Q5、TopN）：

```
r_eff = exposure × r_invested      # 现金收益记 0
```

等价于：`NAV = exposure × NAV_stock + (1−exposure) × 1`。

flag **关闭**时不传入 regime，quantile 路径与旧实验 **bit-identical**。

产物（开启时）：

- `results/<tag>/position_exposure_<tag>.csv` — 调仓日敞口
- `results/<tag>/position_regime_<tag>.csv` — 日频信号全表
- tag 后缀 `_posreg`

## CLI

```bash
# 默认：无仓位体制（可比旧实验）
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml --feature-neutralize

# 启用仓位体制
python run.py --skip-download --mode ridge --horizon 5 \
  --factor-config config/factor_configs.yaml --feature-neutralize \
  --position-regime --output-dir results/ridge_h5_posreg/

# 强制半仓
python run.py ... --position-regime --force-exposure 0.5
```

别名：`--pos-regime` ≡ `--position-regime`。

可与 `--portfolio-opt` 叠加：先在 invested book 上分配权重（和=1），再按
`target_exposure` 缩放收益。见 [PORTFOLIO_OPT.md](PORTFOLIO_OPT.md)。

## 已退役 CLI（no-op + warning）

| 旧开关 | 现状 |
|--------|------|
| `--regime-cs` / `_regcs` | 已删除 `轮动_*` 注入 |
| `--no-regime` | ML 默认本就不注入市场/HMM，可省略 |
| `--ridge-drop-regime` / `_nodreg` | 无 regime 列可 drop |

研究用 HMM / 广播面板仍保留在 `factors/factor.py::_market_regime_features`，
**不再**经 `get_factor_registry` yield。
