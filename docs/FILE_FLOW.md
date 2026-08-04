# 文件流转审计（FILE_FLOW）

> 起因：2026-07-03 一次冒烟测试直接覆盖了真实的 149 因子 Barra 纯 IC 产物（50 分钟计算量），
> 且 `save_results` 在覆盖前**没有任何备份**。本文档固化所有产物的写/读路径与覆盖风险等级，
> 并记录流水线现已遵循的防覆盖规则，避免再次发生。

## 1. 产物流转总表

| 产物路径 | 写入方（file:function:line） | 读取方 | 覆盖风险 | 已备份保护 |
|---|---|---|---|---|
| `data/raw/*.parquet` | `data/download.py` | `clean` / `load_data` / `run.py` | 低 | 否（源头数据，按下载日期命名） |
| `data/processed/factor_cache/*.parquet` | `factors/factor_cache.py:compute_single_factor_cached` | `iter_factor_registry` | 中 | 否（hash-keyed，按内容哈希分桶） |
| `data/processed/factor_panel_*.parquet` | `run.py` / `strategies/ml.py` | ML 训练 | 中 | 否（hash-keyed，内容变即换文件） |
| `research/output/ic_summary_h{p}.csv` | `research/ic/io.py:save_results:27` | 人工查看 | **高** | ✅ 已实施（_archive/） |
| `research/output/ic_yearly_h{p}.csv` | `research/ic/io.py:save_results:28` | 人工查看 | **高** | ✅ 已实施 |
| `research/output/ic_industry_h{p}.csv` | `research/ic/io.py:save_results:31` | 人工查看 | **高** | ✅ 已实施 |
| `research/output/ic_barra_pure_h{p}.csv` | `research/ic/io.py:save_results:39` | 人工查看 | **高** | ✅ 已实施 |
| `research/output/selected_factors_h{p}.json` | `research/ic/io.py:save_results:60` | `logs/driver.py:sync_factor_yaml:85` | **高** | ✅ 已实施 |
| `research/output/_checkpoints/*.pkl` | `research/ic/cli.py:_save_ckpt` | `_load_ckpt`（resume） | **高** | 实施中（原子写、不主动清、完整性校验，由另一子代理负责） |
| `config/factor_configs.yaml` | `logs/driver.py:sync_factor_yaml:107` | `run.py:_load_factor_config` | 中 | 否（受 Git 版本控制兜底） |
| `results/<tag>/*` | `run.py` | 人工查看 | 低 | 否（tag 隔离，不同实验不冲突） |

## 2. 2026-07-03 破坏链（警示案例）

当日一次冒烟测试触发了一条完整的"覆写传播链"：

1. **冒烟测试启动** → `research/ic/cli.py` 的 checkpoint 被主动清空（旧实现"开跑即清"）。
2. **2 因子小测试完成** → `research/ic/io.py:save_results` **无条件覆盖**了
   `ic_summary_h20.csv` / `ic_yearly_h20.csv` / `ic_barra_pure_h20.csv` /
   `selected_factors_h20.json`，把真实的 149 因子产物替换成 2 因子垃圾。
3. **用户尝试 resume** → resume 读到的是 2 因子的"已完成"状态，污染继续传播。
4. **resume 触发的全量重跑被中断** → checkpoint 再次被清空，**真实的 149 因子结果彻底丢失**，
   且没有任何备份可恢复。

根因有二：
- `save_results` 覆盖前不备份（Task 1 已修复）。
- checkpoint 旧实现"开跑即清"且无完整性校验（另一子代理正在修复）。

## 3. 流水线现遵循的规则

### 3.1 `save_results` 覆盖前归档（已实施）

`research/ic/io.py::_archive_before_write` 在覆盖任何既有产物之前：

1. 若目标文件**不存在或为空（0 字节）**：跳过归档（防止把空垃圾文件归档传播）。
2. 否则用 `shutil.copy2`（保留元数据）复制到
   `OUTPUT_DIR / "_archive" / <stem>.<YYYYMMDD_HHMMSS>.<suffix>`，
   时间戳为本地时间 `%Y%m%d_%H%M%S`。
3. 归档后对**同一基名**保留最新 20 份，超出部分按 mtime 删最旧的。
4. 每归档一份打印一行：`  [archive] <name> → _archive/<archived_name>`。

适用对象：`ic_summary_h{p}.csv` / `ic_yearly_h{p}.csv` / `ic_industry_h{p}.csv` /
`ic_barra_pure_h{p}.csv` / `selected_factors_h{p}.json` 共 5 个产物。
函数签名、JSON 结构、`engine="v2"` / `factors` / `factors_orth` / `orthogonalization`
字段**均未改动**，仅在每次写入前插入归档步骤。

### 3.2 Checkpoint 可靠性（实施中，由另一子代理负责）

- **原子写**：先写临时文件再 `os.replace`，避免半写状态被 resume 当成完整。
- **不主动清**：开跑时不再 eager-clear；只有 `--fresh` 显式强制清空。
- **resume 完整性校验**：恢复前校验 checkpoint 字段完整，不完整则丢弃。
- **增量 `--only-new` / `--factors`**：保留 `ic_series_h{p}.pkl`，只补算缺失因子并 merge；
  清除 `summary/yearly/selection/gramschmidt` 下游 checkpoint 并重跑报告；
  **`barra_pure` 保留**：指纹（`BARRA_CACHE_VERSION`）匹配时只对新区做纯化并 merge，
  版本变 / `--fresh` / 无版本元数据（增量场景）→ 全量 pure。

> 本节由负责 `research/ic/cli.py` 的子代理实施，本文件作者**未触碰** `cli.py` / `ic_analysis.py`。

### 3.3 单一调用入口

`research/ic_analysis.py` 已收敛为 shim，仅委托 `research/ic/cli.py:main()`，
保证 IC 分析只有**一条代码路径**写入上述产物，不再有 v1/v2 双写歧义。

## 4. Barra 残差化冗余说明

当前存在一处冗余计算：

- **纯 IC 阶段**（`research/ic/barra.py`）：在内存里临时计算 Barra+行业残差化面板，
  算完即丢，**不落盘**。
- **ML 阶段**（`run.py --feature-neutralize`）：通过
  `models/wf/labels.py:residualize_panel` **再次**计算同样的 Barra+行业残差化面板。

两处口径现已对齐（同一残差化函数），但**重复计算**浪费算力。

**建议（未实施）**：在纯 IC 阶段把残差化后的"纯净因子面板"落盘一次
（如 `data/processed/factor_panel_pure_*.parquet`，hash-keyed），
ML 阶段直接读取复用，避免重复计算 + 保证口径严格一致。

## 5. 安全冒烟测试配方

测试 `save_results` 的归档逻辑而**不破坏真实产物**的几种姿势：

1. **换 horizon**：用 `--period 999`（或其他未用过的 period）跑，产物落
   `ic_summary_h999.csv` 等，绝不会触碰 h5/h10/h20 真实文件。
2. **临时 OUTPUT_DIR**：monkey-patch `research.ic.io.OUTPUT_DIR` 指向临时目录
   后再调 `save_results`，跑完即扔。
3. **`--raw-select` + 检查 `_archive/`**：用极小因子子集跑一次后，立即
   `ls research/output/_archive/` 确认归档副本已生成，再删 h999 文件与对应归档。

**严禁**：在生产 period（5/10/20/60）上跑 2-3 因子冒烟测试，
即使有归档兜底也应避免——归档是最后一道防线，不是常规回滚手段。
