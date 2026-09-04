"""Save IC analysis outputs (driver.py compatible JSON)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# 每个基名在 _archive/ 中保留的最大历史副本数
_ARCHIVE_CAP = 20


def _archive_before_write(path: Path) -> None:
    """覆盖前将既有文件归档到 OUTPUT_DIR/_archive/<name>.<YYYYMMDD_HHMMSS>.<ext>。

    - 不存在或空文件（0 字节）跳过归档（避免归档垃圾空文件，
      这是历史上 2 因子冒烟测试残留得以传播的根源）。
    - 归档使用 shutil.copy2 保留元数据。
    - 归档后对同名基名保留最新 _ARCHIVE_CAP 份，删除更旧的。
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    archive_dir = OUTPUT_DIR / "_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件名格式：<stem>.<ts>.<suffix>，例如 ic_summary_h20.csv.20260703_174000.csv
    archived_name = f"{path.stem}.{ts}{path.suffix}"
    archived_path = archive_dir / archived_name
    shutil.copy2(path, archived_path)
    print(f"  [archive] {path.name} → _archive/{archived_name}")

    # 同基名（按 stem 前缀匹配）保留最新 _ARCHIVE_CAP 份
    # 基名 = 原始文件名去掉时间戳后的部分；这里用 path.stem 作为基名匹配前缀
    prefix = path.stem + "."
    siblings = sorted(
        archive_dir.glob(f"{prefix}*{path.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in siblings[_ARCHIVE_CAP:]:
        try:
            stale.unlink()
        except OSError:
            pass


def save_results(
    period: int,
    summary_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    kept_factors: list,
    exclusion_reasons: dict,
    lookback_years: int = 0,
    lookback_date=None,
    ind_ic_df: pd.DataFrame | None = None,
    pure_ic_means: dict | None = None,
    meta: dict | None = None,
    orth_factors: list | None = None,
    sparse_factors: list | None = None,
    emerging_factors: list | None = None,
    categories: dict | None = None,
    labels: dict | None = None,
    quantile_df: pd.DataFrame | None = None,
    name_suffix: str = "",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 非空后缀用于小市值等子宇宙，避免覆盖全市场产物（如 _small_mcap_q30）
    suf = f"_{name_suffix}" if name_suffix else ""

    summary_path = OUTPUT_DIR / f"ic_summary_h{period}{suf}.csv"
    _archive_before_write(summary_path)
    summary_df.to_csv(summary_path, encoding="utf-8-sig")

    yearly_path = OUTPUT_DIR / f"ic_yearly_h{period}{suf}.csv"
    _archive_before_write(yearly_path)
    yearly_df.to_csv(yearly_path, encoding="utf-8-sig")

    if ind_ic_df is not None and not ind_ic_df.empty:
        ind_path = OUTPUT_DIR / f"ic_industry_h{period}{suf}.csv"
        _archive_before_write(ind_path)
        ind_ic_df.to_csv(ind_path, encoding="utf-8-sig")

    if pure_ic_means:
        pure_df = pd.DataFrame.from_dict(
            pure_ic_means, orient="index", columns=["纯因子IC均值"]
        )
        pure_df["原始IC均值"] = summary_df["IC均值"].reindex(pure_df.index)
        pure_df["保留率"] = pure_df["纯因子IC均值"] / pure_df["原始IC均值"]
        # 向后兼容：有分位分解时追加列，无则保持旧三列 schema
        if quantile_df is not None and not quantile_df.empty:
            q_cols = [
                c for c in (
                    "多头超额", "空头贡献", "long_share", "多空来源",
                    "Q5_mean", "Q1_mean", "spread", "n_days",
                )
                if c in quantile_df.columns
            ]
            pure_df = pure_df.join(quantile_df[q_cols], how="left")
        pure_path = OUTPUT_DIR / f"ic_barra_pure_h{period}{suf}.csv"
        _archive_before_write(pure_path)
        pure_df.to_csv(pure_path, encoding="utf-8-sig")

    if quantile_df is not None and not quantile_df.empty:
        q_path = OUTPUT_DIR / f"ic_quantile_ls_h{period}{suf}.csv"
        _archive_before_write(q_path)
        quantile_df.to_csv(q_path, encoding="utf-8-sig")

    ic_cols = [c for c in ["IC均值", "ICIR", "胜率", "NW_t统计量", "IC_after_cost"] if c in summary_df.columns]
    selection = {
        "horizon": period,
        "lookback_years": lookback_years if lookback_years > 0 else "full",
        "ic_start_date": str(lookback_date.date()) if lookback_date is not None else "all",
        "engine": "v2",
        "factors": kept_factors,
        "excluded": exclusion_reasons,
        "ic_stats": summary_df[ic_cols].to_dict(),
        "meta": meta or {},
    }
    # 双轨制：factors = ML 完整 pure-IC 集（pre-GS ~65），
    # factors_orth = dynamic Gram-Schmidt 正交集（≤30）；未跑 GS 时缺省（dynamic 回退用 factors）
    if orth_factors:
        selection["factors_orth"] = orth_factors
    # 稀疏轨道：不进 factors / factors_orth；供 --special-factors sparse 注入 ridge
    if sparse_factors:
        selection["factors_sparse"] = list(sparse_factors)
    # 新兴：仅观察名单，不进 factors / ML 默认白名单
    if emerging_factors:
        selection["factors_emerging"] = list(emerging_factors)
    # 主类别：普通 / 稀疏 / 新兴；警示标签：衰减 / 风格逆转（可叠加，不剔除）
    if categories:
        selection["categories"] = dict(categories)
    if labels:
        selection["labels"] = {k: list(v) for k, v in labels.items()}
    if categories or labels:
        by_cat: dict[str, list[str]] = {}
        for fname, cat in (categories or {}).items():
            by_cat.setdefault(cat, []).append(fname)
        for fname, labs in (labels or {}).items():
            for lab in labs:
                by_cat.setdefault(lab, []).append(fname)
        selection["categories_grouped"] = by_cat
    # 把 orthogonalization 元数据从 meta 提升为顶层字段，便于 driver / ML 直接读取
    if meta and isinstance(meta, dict) and meta.get("orthogonalization"):
        selection["orthogonalization"] = meta["orthogonalization"]
    json_path = OUTPUT_DIR / f"selected_factors_h{period}{suf}.json"
    _archive_before_write(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    # 可选 Markdown 分类摘要（与 JSON 同步）
    if categories or sparse_factors or emerging_factors or labels:
        md_path = OUTPUT_DIR / f"selected_factors_h{period}{suf}_categories.md"
        _archive_before_write(md_path)
        lines = [
            f"# IC 入选因子分类（h{period}）",
            "",
            "类别说明：",
            "- **普通因子**：稠密池全样本过线（IC/ICIR/t/FDR/corr）→ `factors`",
            "- **稀疏因子**：语义稀疏池独立轨道（同向IC胜率 + 触发日截面胜率，按 sign(mean_IC) 对齐），仅建议 ridge 注入",
            "- **新兴因子**：全样本未过 IC/ICIR 门 + 近窗 FDR∧ICIR∧lift；**仅观察**（`factors_emerging`），不进 ML 主池",
            "- **衰减因子**（警示标签，仍在 factors 池）："
            "R 塌缩 ∧ |ICIR_recent| 弱 ∧ |IC_recent| 弱（合取；近窗与 barra pure 对齐）",
            "- **风格逆转**（警示标签，可与衰减叠加）：近一季多数强 IC 与全样本符号相反",
            "",
        ]
        grouped = selection.get("categories_grouped") or {}
        for cat in ("普通因子", "稀疏因子", "新兴因子", "衰减因子", "风格逆转"):
            names = grouped.get(cat) or (
                list(sparse_factors) if cat == "稀疏因子" and sparse_factors else []
            )
            if cat == "新兴因子" and not names and emerging_factors:
                names = list(emerging_factors)

            # 去重保序
            seen = set()
            uniq = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    uniq.append(n)
            lines.append(f"## {cat}（{len(uniq)}）")
            for n in uniq:
                extra = ""
                if labels and n in labels and cat in ("普通因子", "稀疏因子", "新兴因子"):
                    extra = f"  [{', '.join(labels[n])}]"
                lines.append(f"- {n}{extra}")
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path
