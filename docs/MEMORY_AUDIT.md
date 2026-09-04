# 内存占用热点审计报告

> 只读审计 · 2026-07-02
> 数据规模：2059 交易日 × 5803 股票，67 因子（60 参与 IC），float64 原始面板

## 0. 量化基线（用于估算）

单面板（2059 × 5803）内存：

| dtype | 单面板大小 |
|---|---|
| float64 | 95.5 MB |
| float32 | 47.7 MB |
| bool    | 11.9 MB |

`data/raw/*.parquet` 实测为 **float64**（已确认 `prices_hfq` / `volume` 均为 float64），故 `load_ic_data()` / `_load_data()` 加载的原始 OHLCV+辅助面板单张约 95 MB。

---

## 1. 内存峰值环节排序（由高到低）

| 排名 | 阶段 | 估算峰值 | 关键占用 |
|---|---|---|---|
| 🥇 1 | **ML 训练 `fit_predict`**（h20 ensemble） | **~5.5–6 GB** | base bundle 1.5GB + Barra 0.43GB + `MLDataset.factor_panel` 2.86GB + section_cache 0.15GB + jobs/models 0.4GB |
| 🥈 2 | **IC 分析 `research/ic/cli.py:run()`**（含 `--barra`） | **~5–5.5 GB** | base bundle 1.5GB + registry 2.86GB + Barra 0.43GB + `date_ctrl` 0.3GB + per-factor 峰值 0.25GB |
| 🥉 3 | **`build_factor_dataset` + `feature_neutralize` 残差化循环** | **~5 GB** | base bundle + registry + Barra + 残差化 transients |
| 4 | `forward_return` 构造（`build_forward_return`） | **+0.3 GB 瞬时** | `prices.astype(float64)` 复制 95MB + `open_.astype(float64)` 95MB + `buy`/`sell`/`fwd` 各 95MB |
| 5 | 回测 `run_quantile_backtest` | **~0.5–1 GB** | `prices_a` 95MB + `open_a` 95MB + masks 80MB + 各 group 现金流路径 |

> 用户观测「IC 单进程 3.7GB / 系统 31.7GB」与本估算一致：3.7GB 是常驻集（bundle+registry+Barra 已加载，但 per-factor 排名拷贝峰值尚未触发）；31.7GB 系统总占用说明同时跑了多个进程（很可能 driver 在并行多 horizon / 多实验，或此前 ML 残留未释放）。

---

## 2. 内存热点 TOP 5（按"同时持有"贡献排序）

### 🔴 热点 #1：IC 阶段同时持有 60 个因子面板（2.86 GB）

**位置**：`research/ic/cli.py:134`

```134:135:research/ic/cli.py
    registry = {k: _to_float32_panel(v) for k, v in registry.items()}
    t0 = _log_phase(f"计算因子（{len(registry)}个）", t0)
```

- dict comprehension 构造新 dict，**原 registry（仍 float64，60 × 95.5 MB = 5.7 GB）与新 float32 registry（60 × 47.7 MB = 2.86 GB）短暂并存**，峰值可达 8.5 GB。
- 之后原 dict 被 GC 回收，常驻 2.86 GB。
- `registry` 之后被 `compute_ic_series` / `enrich_summary_with_cost` / `run_barra_pure_ic` / `select_factors` / `factor_corr_matrix` / `ic_decay_table` **持续引用，直到函数末尾才释放**。
- 即便 IC 计算是逐因子的（`run_bounded_parallel` 串行模式），也始终持有全部 60 个面板。

**估算**：常驻 2.86 GB，瞬时峰值（comprehension 切换瞬间）可达 8.5 GB。

---

### 🔴 热点 #2：base bundle 全程不释放（~1.5 GB）

**位置**：`research/ic/load_data.py:41-87`（IC 阶段）、`run.py:103-166`（ML 阶段）

`load_ic_data()` 一次性加载并持有：

| 字段 | 大小 |
|---|---|
| `prices` (float64) | 95 MB |
| `prices_raw` | 95 MB |
| `open_` (float64) | 95 MB |
| `high` / `low` (float64) | 191 MB |
| `volume` / `amount` (float64) | 191 MB |
| `margin` / `moneyflow` / `northbound` / `institution` | ≤ 4 × 95 = 380 MB |
| `clean_ret` (由 `clean_ohlcv` 产生) | 95 MB |
| `masks`（6 个 bool 面板） | ~70 MB |
| `industry_map_df` / `market_prices` / `financial` | 较小 |
| **合计** | **~1.4–1.6 GB** |

bundle 作为 `ICDataBundle` dataclass 字段，**从函数入口一直活到函数末尾**，即便后续步骤只需要 `prices` 和 `forward_return`。ML 阶段 `run.py:_load_data()` 同样如此。

---

### 🔴 热点 #3：`compute_ic_series` 单因子排名拷贝（~240 MB / 因子，串行叠加）

**位置**：`research/ic/ic_series.py:24-46`

```24:46:research/ic/ic_series.py
def compute_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
    min_stocks: int | None = None,
) -> pd.Series:
    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    f, r = apply_tradable_mask(factor, forward_return, tradable)
    common = f.index.intersection(r.index)
    f = _to_float32_panel(f.loc[common])
    r = _to_float32_panel(r.loc[common])

    valid_count = (f.notna() & r.notna()).sum(axis=1)
    f_ranked = _rank_panel(f)
    r_ranked = _rank_panel(r)
    ic = f_ranked.corrwith(r_ranked, axis=1)
    ic = ic.where(valid_count >= min_n)
    return ic.dropna()
```

- `apply_tradable_mask` 内部 `factor.reindex(...).where(t)` 创建一份新 float64 DataFrame → ~95 MB
- `_to_float32_panel(f.loc[common])` → 48 MB
- `_rank_panel` 调 `df.rank(axis=1, method=...)`：**pandas `rank()` 默认返回 float64**，所以 `f_ranked` = 95 MB，`r_ranked` = 95 MB
- `corrwith` 内部还会再算一次 rank（Spearman 等价于 rank 后 Pearson），但 pandas 会复用已 rank 的输入走 Pearson 路径——实测仍有中间 Series 分配
- `valid_count` Series：~17 MB

**单因子调用峰值 ≈ 48 + 95 + 95 + 17 ≈ 255 MB**，串行模式下与 2.86 GB registry 叠加；若 `IC_MAX_WORKERS>1` 用线程池，多线程同时各占 250 MB 且共享 GIL 但 **不共享 DataFrame 内存**（pandas 多线程修改/拷贝安全但读取竞争），线程数 × 250 MB 叠加。

---

### 🔴 热点 #4：Barra 残差化 IC（`run_barra_pure_ic`）≈ 0.73 GB

**位置**：`research/ic/barra.py:50-150, 203-281`

- `get_barra_factors()` 返回 **9 个 Barra 因子面板** × 47.7 MB = **429 MB**（其中 `Barra_Beta` 滚动 252 日、`Barra_ResVol` 滚动 60 日本身计算时还有 transient）。
- `precompute_ctrl_matrices` 给每个调仓日缓存 `(ctrl_arr, ctrl_idx, fwd_arr)`：
  - h20 月频调仓日 ≈ 96 个
  - 单日 `ctrl_arr` = (5803 股 × (9 Barra + ~130 行业 dummy)) × 4 bytes ≈ 3.25 MB
  - 96 × 3.25 ≈ **312 MB** 常驻到 `del date_ctrl`（line 279）
- `compute_pure_ic_fast` 对 60 个因子逐个跑 `np.linalg.lstsq`（每截面 96 次 OLS），`factor_arr` 还会 to_numpy 拷贝一次（每因子 48 MB）

**总计 Barra 路径 ≈ 429 + 312 + transients ≈ 0.75–1 GB**，叠加在 IC 已有 4 GB 之上。

---

### 🔴 热点 #5：ML `fit_predict` 的 `MLDataset.factor_panel` + `section_cache` + jobs

**位置**：`models/trainer.py:385-778`

#### 5a. `MLDataset.factor_panel` 全程持有 60 面板 = 2.86 GB

`build_ml_dataset` (`trainer.py:167-190`) 直接把 `factor_dict` 装进 `MLDataset.factor_panel`，**整个 fit_predict 期间不释放**。`get_cross_section` 每次都从这些面板按行 `.loc[date]` 切片。

#### 5b. `section_cache` 预缓存所有调仓日（~150 MB，小但累积）

```399:408:models/trainer.py
        logger.info("预缓存截面数据...")
        section_cache: dict = {}
        for date in dates:
            X, y = dataset.get_cross_section(date)
            if X is not None and len(X) >= MIN_STOCKS_PER_DATE:
                section_cache[date] = (
                    X.values.astype(np.float32),
                    y.values.astype(np.float32),
                    X.index,
                )
```

- 每日：X = (5803 × 60 × 4) = 1.4 MB + y 23KB + index 指针对象
- h20 月频 ~96 调仓日 → ~135 MB 常驻

#### 5c. `_stack_cached` 每次 vstack 训练矩阵（~16–32 MB / call）

```513:521:models/trainer.py
            return (
                np.vstack(X_list),
                np.concatenate(y_list),
                w_arr,
                np.concatenate(y_raw_list),
                list(stocks_per_date),
            )
```

- 窗口 12 月 → 12 dates × 5803 股 × 60 因子 × 4 = 16 MB（X_tr）
- 每 pred_date 调用 2 窗口 × 2 模型 = 4 次 stack → **同时存在 4 个 (X_tr, X_va) ≈ 64 MB**
- 每个被装进 job tuple，传给 `joblib.Parallel(prefer="processes")` → **pickle 序列化复制到子进程**，子进程再 deserialize 一份 → 单 job 内存翻倍到 ~40 MB
- 4 个 job × 2 份 ≈ 160 MB 瞬时

#### 5d. `self.models` 字典保留最近一折模型

`self.models[(window, model_type)] = f.model`（line 653）覆盖式写入，仅保留 4 个 LightGBM 模型（每个 ~1–5 MB），可控。

#### 5e. `_feature_importance_rows` / `_diagnostics 列表累积`

96 个 pred_date × 4 折 × 60 因子 importance 行 → ~23K 行 dict，约 10–20 MB，可忽略。

**ML fit_predict 总峰值 ≈ 1.5 (bundle) + 0.43 (Barra) + 2.86 (factor_panel) + 0.15 (section_cache) + 0.16 (jobs) + 0.05 (models) + LightGBM 内部 ~0.5 ≈ 5.6 GB**

---

## 3. 次要热点

### `forward_return` 构造的 float64 复制（~0.3 GB 瞬时）

`research/ic/forward_return.py:31-39`：

```31:39:research/ic/forward_return.py
    prices = prices.astype(np.float64)
    if open_ is not None:
        open_ = open_.astype(np.float64)

    if open_ is not None:
        buy = open_.shift(-1)
        sell = prices.shift(-period)
        fwd = sell / buy.replace(0.0, np.nan) - 1
```

- 强制 `astype(np.float64)` 即便输入已是 float64 也会**生成一份新拷贝**（pandas 行为）
- `buy` / `sell` / `fwd` 三个 95 MB 矩阵并存 → ~285 MB 瞬时
- 末尾 `.astype(np.float32)` 才降到 48 MB

### `residualize_panel`（feature_neutralize）per-factor 拷贝

`models/wf/labels.py:286-393`：对 60 个因子逐个：
- `out = pd.DataFrame(np.nan, ..., dtype=float32)` → 48 MB
- `f_arr = factor_panel.to_numpy(dtype=np.float64, copy=False)` → 95 MB（copy=False 但 dtype 不同会强制拷贝）
- 每截面 OLS 时 `f_v`/`X_v`/`A` 各几十 KB
- 单因子峰值 ~150 MB，串行叠加在 2.86 GB registry 之上

### `factor_corr_matrix` / `ic_decay_table`（IC 阶段可选分支）

`--decay` / `--corr` 开启时，会对 60 因子两两 corr（60×60 矩阵不大），但 `factor_corr_matrix` 内部若同时取出所有因子 stack 成长表 → (2059 × 5803 × 60) × 4 = 2.86 GB 额外瞬时。

---

## 4. 优化建议（按 收益 / 改动量 排序）

### 🟢 高收益 · 低改动

1. **IC 阶段流式释放因子**（解决热点 #1）
   - `cli.py:134` 改为：计算 IC 时逐因子 `compute_one(name, fac)` → 算完立即 `del fac`，仅保留 `all_ic[name]`（Series，~16 KB）
   - 保留 registry 仅用于 `enrich_summary_with_cost`（仅用调仓日切片）和 `select_factors`（用 corr 矩阵，可流式两两算）
   - **预期降内存：2.86 GB → ~0.5 GB**（仅持有当前因子 + 1 个 f32 拷贝）
   - 改动量：中等（需重构 select_factors / corr 的调用方式）

2. **base bundle 按需加载 + 显式 `del`**（解决热点 #2）
   - IC 阶段算完 forward_return 后立即 `del bundle.volume, bundle.amount, bundle.margin, ...`，仅保留 `prices` / `open_` / `masks`
   - 或在 `load_ic_data()` 增加参数 `lazy=True`，按字段 property 加载
   - **预期降内存：1.5 GB → ~0.4 GB**
   - 改动量：低（加几行 `del` + `gc.collect()`）

3. **`compute_ic_series` 内部改用 numpy rank + float32**（解决热点 #3）
   - `_rank_panel` 改为 `df.rank(...).astype(np.float32)`，或直接 numpy `rankdata` 沿 axis=1
   - 删除 `apply_tradable_mask` 中的 `.reindex().where()` 多余拷贝，改用布尔索引 in-place mask
   - **预期降内存：255 MB → ~100 MB / 因子**
   - 改动量：低

4. **`forward_return.py` 跳过冗余 astype**
   - 输入已是 float64 时跳过 `astype(np.float64)`
   - `buy` / `sell` 算完立即 `fwd = (sell.values / buy.values - 1).astype(np.float32)` 转 DataFrame，避免三份并存
   - **预期降内存：285 MB → 95 MB 瞬时**
   - 改动量：极低

### 🟡 中收益 · 中改动

5. **`MLDataset.factor_panel` 改为按需切片 + 内存映射**
   - 用 `pyarrow` parquet memory-mapped，或把 panel 拼成单个 (date, stock, factor) float32 三维 ndarray（2059 × 5803 × 60 × 4 = 2.86 GB 一次分配，但避免 60 个独立 DataFrame 索引开销）
   - `get_cross_section(date)` 改为从大 ndarray 切片
   - **预期降内存：2.86 GB → 2.86 GB（不变）但分配次数减少，GC 压力降低；若改为 parquet mmap 则常驻 ~0**
   - 改动量：高

6. **`section_cache` 改为 LRU 缓存（仅保留最近 N 个调仓日）**
   - walk-forward 训练时只用到 `[idx - max_window - val, idx]` 范围的截面
   - 改为 `functools.lru_cache(maxsize=24)` 或自实现 ring buffer
   - **预期降内存：135 MB → ~30 MB**
   - 改动量：低

7. **joblib 改用 `loky` + 共享内存（`SharedMemory` / `memmap`）**
   - 当前 `Parallel(prefer="processes")` 用 pickle 复制每个 job 的 X_tr/X_va
   - 改为 `joblib.dump` 到 memmap 文件，子进程 `load(..., mmap_mode='r')`
   - **预期降内存：~80 MB 翻倍开销 → 0**（共享物理页）
   - 改动量：中

8. **`precompute_ctrl_matrices` 改为 QR 缓存 + 行业 dummy 共享**
   - 当前每调仓日存独立 (ctrl_arr, fwd_arr)；可只存一次 `barra_df_full`（9 × 5803 × 96 = 19 MB）+ 行业 dummy 一次（130 × 5803 = 3 MB），OLS 时按日切片
   - **预期降内存：312 MB → ~25 MB**
   - 改动量：中

### 🔵 长期重构

9. **统一 dtype 策略：raw parquet 一次性转 float32 落盘**
   - 加载时 `pd.read_parquet(...).astype(np.float32)`，所有下游默认 float32
   - **预期降内存：bundle 1.5 GB → 0.75 GB；factor panel 2.86 GB → 1.43 GB**
   - 改动量：低（一行 astype），但需全链路测试精度损失

10. **因子计算改流式（generator 而非 dict）**
    - `get_factor_registry` 改为 `iter_factors()` yield (name, panel)，IC 阶段边算边释放
    - ML 阶段则需要先收集 name 列表，再决定保留哪些 → 需要 two-pass 或 metadata 预声明
    - **预期降内存：2.86 GB → 50 MB**（IC 阶段）
    - 改动量：高（接口破坏性变更）

11. **数据子集化（dim reduction）**
    - 在 `config/settings.py` 增加 `UNIVERSE_FILTER`（如剔除上市<2年、ST、市值<20亿），在 `load_ic_data` 入口裁剪 columns，5803 → ~3500
    - **预期降内存：所有面板降 ~40%**（1.5 GB → 0.9 GB，2.86 GB → 1.7 GB）
    - 改动量：低

---

## 5. 快速缓解方案（不改代码）

按效果排序：

| # | 措施 | 命令 / 配置 | 预期降内存 |
|---|---|---|---|
| 1 | **缩减因子白名单** | 编辑 `config/factor_configs.yaml` 把 h20 从 60 因子砍到 25–30 | -1.5 GB（registry & factor_panel 同降） |
| 2 | **不开 `--barra` 跑 IC** | `python -m research.ic_analysis_v2 --period 20 --save`（去掉 `--barra`） | -0.75 GB（跳过 9 Barra 面板 + date_ctrl） |
| 3 | **不开 `--feature-neutralize` 跑 ML** | `python run.py ... `（去掉 `--feature-neutralize`） | -0.43 GB Barra + 60×0.15 GB 残差化 transients |
| 4 | **缩短回测区间** | 设置 `BACKTEST_START=2020-01-01`（env 或改 settings.py） | 日期维度降 ~30%，全链路 -30% |
| 5 | **保持默认并行度** | 确认 env 未覆盖：`IC_MAX_WORKERS=1 BARRA_IC_WORKERS=1 TRAIN_MAX_WORKERS=1` | 防止线程池 × transients 翻倍 |
| 6 | **`--sample N` 调试** | `python run.py --sample 2000 ...` | 股票维度降 ~65% |
| 7 | **分阶段执行而非 driver 一把梭** | 手动跑 IC → 落 YAML → 单独跑 `run.py` → 单独跑回测；每阶段间重启 Python 释放 | 避免 bundle/registry 跨阶段累积 |
| 8 | **关闭其他应用** | 浏览器/IDE/其他 Python 进程 | 释放给量化流程 |
| 9 | **增加虚拟内存 / swap** | Windows 设大 pagefile（≥16 GB） | 防 OOM 杀进程，但不提速 |
| 10 | **不混跑 dynamic / blend-dynamic 与 ML** | 单独跑 `--mode dynamic` 或 `--mode ensemble`，不要同时开 `--blend-dynamic` | 避免额外 `build_factor_dataset` 二次构建 |

---

## 6. 长期重构方案（按优先级）

### P0：流式 IC 计算管线

**目标**：IC 阶段常驻内存从 5 GB 降到 < 1 GB。

- 改造 `get_factor_registry()` 为 `iter_factors()` generator
- `cli.py:run()` 改为：每拿到一个因子 → 算 IC → 算 Barra 纯 IC（如果需要）→ 累积 Series → `del panel`
- `select_factors` 中的 `factor_corr_matrix` 改为：边迭代边两两更新 corr 上三角
- `enrich_summary_with_cost` 只需调仓日切片 → 改为因子 generator 内 yield 调仓日子集

### P1：base bundle lazy + 早期释放

- `load_ic_data()` 改为 lazy property：第一次访问 `bundle.volume` 才读盘
- IC 阶段算完 forward_return 后显式 `del bundle.{volume,amount,margin,moneyflow,northbound,institution,prices_raw,high,low}`
- ML 阶段同理：`build_factor_dataset` 末尾 `del bundle_aux`

### P2：统一 float32 落盘

- 新增 `data/clean.py::downcast_panel(df)` 在清洗出口统一 `astype(np.float32)`
- 重写 raw parquet 为 float32（一次性脚本）
- 所有下游不再 `_to_float32_panel`（删除冗余拷贝）

### P3：joblib 共享内存

- `models/trainer.py:605-611` 的 `Parallel(prefer="processes")` 改为：
  ```python
  from joblib import Parallel, delayed, dump, load
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
      for i, j in enumerate(jobs):
          dump(j, f"{tmp}/job_{i}.pkl")
      memmaped = [load(f"{tmp}/job_{i}.pkl", mmap_mode="r") for i in range(len(jobs))]
      raw = Parallel(n_jobs=n_workers, prefer="processes")(delayed(_run_fold_job)(j) for j in memmaped)
  ```
- 子进程共享物理内存页，job tuple 不再 pickle 复制

### P4：MLDataset 改为 parquet memory-mapped

- `build_factor_dataset` 不再返回 dict[str, DataFrame]，而是把 60 个面板拼接成单个 multi-index parquet 落盘
- `MLDataset` 持有 `pyarrow.parquet.ParquetFile` 句柄，`get_cross_section(date)` 用 row-group 切片读取
- 常驻内存 < 100 MB，trade-off 是每次切片有 IO（可用 LRU 缓存最近 24 个截面）

### P5：UNIVERSE 预过滤

- 在 `load_ic_data()` 入口加 `universe_filter` 参数：剔除上市<2年、ST 当前、最近 60 日均成交额 < 5000万
- 5803 → ~3500，全链路降 40% 内存

---

## 7. 结论

**内存峰值阶段**：ML 训练 `fit_predict`（~5.5–6 GB）> IC 分析 `cli.py:run`（~5–5.5 GB）> build_factor_dataset+feature_neutralize（~5 GB）> 回测（~1 GB）。

**内存热点 TOP5**：
1. `research/ic/cli.py:134` — 同时持有 60 个 f32 因子面板 = 2.86 GB（dict comprehension 瞬时峰值 8.5 GB）
2. `research/ic/load_data.py:41` / `run.py:103` — base bundle 全程不释放 = ~1.5 GB
3. `research/ic/ic_series.py:24-46` — 单因子 rank 拷贝 ~255 MB（pandas rank 返回 float64）
4. `research/ic/barra.py:50,203` — 9 Barra 面板 0.43 GB + date_ctrl 0.31 GB
5. `models/trainer.py:385-778` — `MLDataset.factor_panel` 2.86 GB + section_cache 0.15 GB + joblib pickle 复制 0.16 GB

**快速缓解方案（不改代码，立即可用）**：
1. 缩减 `factor_configs.yaml` 因子白名单 60 → 30（-1.5 GB）
2. IC 阶段去掉 `--barra`、ML 阶段去掉 `--feature-neutralize`（-1.2 GB）
3. `BACKTEST_START=2020-01-01` 缩短回测区间（全链路 -30%）
4. 确认 env `IC_MAX_WORKERS=1 BARRA_IC_WORKERS=1 TRAIN_MAX_WORKERS=1`（防 transients × 并行翻倍）
5. 分阶段手动跑（IC → YAML → run.py → 回测），阶段间重启 Python 释放
6. `--sample 2000` 调试模式（-65%）

**长期最优解**：P0 流式 IC + P1 bundle lazy + P2 float32 落盘 三件套，可把 IC 阶段常驻从 5 GB 降到 < 1 GB，ML 阶段从 6 GB 降到 ~3 GB。
