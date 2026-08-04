# Special Factors（特殊因子注入）

统一入口：把**不必经 IC YAML 白名单**、且通常应**跳过 `--feature-neutralize`** 的因子，在白名单过滤之后 post-merge 进 **ML** 训练与打分。

**dynamic（ICIR 动态加权）轨道禁止注入**：`--mode dynamic` 会忽略 `--special-factors` 并 warning；`blend-dynamic` 的 dynamic 半边同样 `deny_special_inject=True`。

IC 筛选路径**不变**：不会自动把 specials 写进 `factor_configs.yaml`；本机制仅用于训练 / 打分注入。稀疏入选名单写在 IC JSON 的 `factors_sparse`。

## CLI

```bash
# 推荐：统一入口（--inject-factors 为别名）
python run.py --skip-download --mode ensemble --horizon 20 \
  --factor-config config/factor_configs.yaml \
  --special-factors event,size --feature-neutralize

# 稀疏包：建议 ridge + 从 IC JSON 取入选名（避免全量语义池）
python run.py --skip-download --mode ridge --horizon 5 \
  --special-factors sparse \
  --sparse-from-ic research/output/selected_factors_h5.json

# 仅事件 / 仅市值
python run.py --skip-download --mode ridge --horizon 5 \
  --special-factors event
python run.py --skip-download --mode ridge --horizon 5 \
  --inject-factors size

# dynamic 会忽略 special-factors（warning）
python run.py --skip-download --mode dynamic --horizon 5 \
  --special-factors sparse   # 不会注入

# 也可传 pack 内具体因子名（只注入这些）
python run.py --skip-download --mode ridge --horizon 5 \
  --special-factors 对数市值,市值分位

# 已弃用（仍可用，映射为 event，并 DeprecationWarning）
python run.py ... --event-overlay
```

输出目录 tag 后缀反映启用的 pack，例如 `_event`、`_size`、`_sparse`、`_event_size`。

## Packs

| Pack | 因子名 | 计算入口 | skip neutralize | 方差对齐 |
|------|--------|----------|-----------------|----------|
| `event` | `EVENT_OVERLAY_FACTOR_NAMES`（当前：`业绩预告_超预期`） | `get_event_overlay_factors` | 是 | 否 |
| `size` | `SIZE_ALPHA_FACTOR_NAMES` | `get_size_alpha_factors` | 是 | 否 |
| `sparse` | `SPARSE_FACTOR_NAMES`（龙虎榜/涨跌停/开板/解禁/高管/大宗/业绩预告等；**不含**两融日频截面） | `factors.sparse_factors.compute_sparse_factors` | 是 | **是**（ridge） |

实现：`factors/special_factors.py`、`factors/sparse_factors.py`；接入：`strategies/ml.py::build_factor_dataset`。

### 方差对齐公式（sparse pack）

稠密因子经截面 winsorize + z-score 后，有值单元格标准差约 1。稀疏因子堆叠后列方差常 ≪ 1，Ridge 的 L2 惩罚会系统性压掉其系数。注入时：

\[
x' = x \cdot \frac{\texttt{target\_std}}{\mathrm{std}(\{x_{ij}: x_{ij}\ \mathrm{finite}\})}
\]

默认 `target_std = 1.0`（`SPARSE_VARIANCE_ALIGN_STD`）。有效样本过少或 σ≈0 时跳过缩放。树模型无额外处理。

## 与 IC / YAML 的关系

- IC v2 多轨筛选：稠密（普通；衰减/风格逆转仅标注）写 `factors`；新兴写 `factors_emerging`（**观察用，默认不进 ML**）；稀疏写 `factors_sparse`；GS 写 `factors_orth`（dynamic）。
- **不**因 `--special-factors` 自动扩 YAML 白名单；也**没有**现成 `emerging` pack——若要试新兴，需自定义白名单或仿 sparse 注入，且 dynamic 会 `deny_special_inject`。
- `size` 因子也可出现在正常 registry / YAML；注入 pack 是为了**强制进入训练**而不依赖 YAML。
- `event` 默认不进 `get_factor_names()` 截面池，只能经 special inject。
- `sparse`：语义池集中在 `factors/sparse_factors.py`；生产推荐 `--sparse-from-ic research/output/selected_factors_h{p}.json`。
  稀疏轨硬门槛：同向 IC 胜率（默认 0.56）+ 触发日截面胜率 `payoff_hit`（默认 0.55），
  均按 `s=sign(mean_IC)` 对齐（负 IC 触发侧为 `f<0`）；日期与普通 IC 有效交易日对齐；
  **无 t/NW-t/FDR**。过线后做 corr-dedup（默认阈值 0.70，`--sparse-corr-threshold`；
  优先截面 Spearman，不稳定时 fallback IC 序列相关；保留更高 \|ICIR\|），结果写入 `factors_sparse`。
- 生产 IC 筛选（写 YAML 前）h5/h20 **必须同参**，见操作手册 §3：
  `--barra --save --use-fdr --t-threshold 2.5 --gram-schmidt --decay`
  （`corr-dedup` / `decay-gate` / `reversal-label` / `emerging` / `sparse-track` 默认 ON）。
  新兴近窗与衰减近窗在 `--barra` 下对齐 pure IC 序列；新兴近窗另做 NW-t BH-FDR
  （校正域=稠密全体被测因子）+ ICIR + lift，仅观察、不进 `factors`。

## Deprecated

| 旧标志 | 替代 |
|--------|------|
| `--event-overlay` | `--special-factors event` / `--inject-factors event` |
