"""Load pure-IC checkpoint and write rolling-pool schedule artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_CKPT = OUTPUT_DIR / "_checkpoints" / "barra_pure_h5.pkl"


def load_barra_pure_ic(
    ckpt: str | Path | None = None,
    *,
    ensure_weekly: bool = True,
) -> dict[str, pd.Series]:
    """
    加载 ``barra_pure_h{N}.pkl`` checkpoint 中的 pure IC 序列字典。

    checkpoint 结构（与 ``research.ic.cli._save_ckpt`` 一致）::
        (pure_ic_means, style_names, pure_ic_series, quantile_df)
    """
    path = Path(ckpt) if ckpt else DEFAULT_CKPT
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) >= 3 and isinstance(obj[2], dict):
        pure = obj[2]
    elif isinstance(obj, dict):
        # allow raw dict
        pure = obj
    else:
        raise TypeError(f"无法解析 checkpoint 结构: {type(obj)}")

    out: dict[str, pd.Series] = {}
    for k, v in pure.items():
        s = pd.Series(v).dropna().astype(float).sort_index()
        if s.empty:
            continue
        if ensure_weekly:
            s = _to_weekly(s)
        if len(s) > 0:
            out[str(k)] = s
    if not out:
        raise ValueError(f"checkpoint 无有效 IC 序列: {path}")
    return out


def _to_weekly(s: pd.Series) -> pd.Series:
    """若已是约周频则原样返回，否则 W-FRI 末值。"""
    s = s.dropna().sort_index().astype(float)
    if len(s) < 3:
        return s
    gaps = np.diff(s.index.values).astype("timedelta64[D]").astype(float)
    if np.nanmedian(gaps) >= 4.0:
        return s
    return s.resample("W-FRI").last().dropna()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if k != "period_df"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def write_schedule_outputs(
    schedule: pd.DataFrame,
    meta: dict,
    *,
    out_prefix: str | Path | None = None,
    horizon: int = 5,
) -> dict[str, Path]:
    """
    写出 parquet 长表 + meta json + 摘要 md。

    默认前缀：``research/output/rolling_pool_schedule_h{horizon}``
    """
    if out_prefix is None:
        out_prefix = OUTPUT_DIR / f"rolling_pool_schedule_h{horizon}"
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pq_path = prefix.with_suffix(".parquet")
    csv_path = prefix.with_suffix(".csv")
    meta_path = Path(str(prefix) + ".meta.json")
    md_path = Path(str(prefix) + "_summary.md")
    period_path = Path(str(prefix) + "_periods.csv")

    sched = schedule.copy()
    if not sched.empty:
        sched["date"] = pd.to_datetime(sched["date"])
    sched.to_parquet(pq_path, index=False)
    sched.to_csv(csv_path, index=False, encoding="utf-8-sig")

    period_df = meta.get("period_df")
    if isinstance(period_df, pd.DataFrame):
        period_df.to_csv(period_path, index=False, encoding="utf-8-sig")

    meta_path.write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_render_summary_md(meta, schedule), encoding="utf-8")

    return {
        "parquet": pq_path,
        "csv": csv_path,
        "meta": meta_path,
        "summary_md": md_path,
        "periods": period_path,
    }


def _render_summary_md(meta: dict, schedule: pd.DataFrame) -> str:
    params = meta.get("params", {})
    seg = meta.get("segment_2026") or {}
    smoke = meta.get("smoke_first5") or []
    degraded = bool(meta.get("cs_corr_degraded"))
    lines = [
        "# 轮动定池 schedule 摘要",
        "",
        "## 参数",
        "",
        f"- window={params.get('window')} 周（因果，**不含决策日**：只用 index < t 的 IC）",
        f"- 硬门: |mean|>{params.get('abs_mean_min')} 且 |ICIR|>{params.get('abs_icir_min')} "
        f"(min_periods={params.get('min_periods')}, ddof={params.get('ddof')})",
        f"- K_max={params.get('k_max')}（无下限）",
        f"- 强制换手: target_out = max(n_fail, floor({params.get('turnover_frac')} * n))",
        f"- 去重阈值: IC corr={params.get('ic_corr_thr')}, CS corr={params.get('cs_corr_thr')}",
        f"- 冷却期: {params.get('cooldown_periods')}",
        f"- 决策日: {meta.get('date_start')} → {meta.get('date_end')} "
        f"(n={meta.get('n_rebalance_dates')})",
        f"- 宇宙因子数: {meta.get('n_factors_universe')}",
        f"- cs_provider: `{meta.get('cs_provider')}`",
        f"- 因果性: ic_window_excludes_decision_date="
        f"{meta.get('ic_window_excludes_decision_date')}"
        "（IC_t 需 t→t+h 收益，故只用 index < t；消费侧 asof(<=) 无需再 shift）",
        "",
    ]
    if degraded:
        lines += [
            "## [WARN] 截面去重已降级",
            "",
            "第二道因子值截面相关 **未成功应用**（或整段跳过）。",
            f"- skip_reasons: {meta.get('cs_skip_reasons')}",
            "- 仅第一道 **IC 序列相关** 去重生效。",
            "- 若需启用第二道：确保 `data/processed/factor_panels/` 有候选因子缓存，"
            "且不要传 `--no-cs-corr`。",
            "",
        ]
    else:
        lines += [
            "## 截面去重",
            "",
            "第二道因子值截面相关 **已启用**（至少在部分决策日 applied）。",
            "",
        ]

    mean_n = meta.get("mean_n_pool")
    mean_n_s = f"{mean_n:.2f}" if mean_n is not None else "NA"
    lines += [
        "## 规模与换手",
        "",
        f"- 并集 |U| = **{meta.get('union_size')}**",
        f"- 平均池大小 = {mean_n_s}",
        f"- 平均换手（不含首期） = **{meta.get('mean_turnover_ex_init'):.4f}**",
        f"- 平均换手（含首期） = {meta.get('mean_turnover_all'):.4f}",
        "",
        "## 2026 段",
        "",
    ]
    if seg:
        lines += [
            f"- n_periods = {seg.get('n_periods')}",
            f"- mean_n_pool = {seg.get('mean_n_pool'):.2f}",
            f"- mean_turnover = {seg.get('mean_turnover'):.4f}",
            f"- mean_n_out = {seg.get('mean_n_out'):.2f}",
            f"- mean_n_fail = {seg.get('mean_n_fail'):.2f}",
            f"- mean_n_trim = {seg.get('mean_n_trim'):.2f}",
            "",
        ]
    else:
        lines += ["- （无 2026 决策日）", ""]

    lines += ["## 冒烟：前 5 个决策日", "", "| date | n_pool | n_fail | n_trim | n_out | turnover |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in smoke:
        lines.append(
            f"| {r.get('date')} | {r.get('n_pool')} | {r.get('n_fail')} | "
            f"{r.get('n_trim')} | {r.get('n_out')} | {r.get('turnover'):.3f} |"
        )
    lines.append("")

    if schedule is not None and not schedule.empty:
        by_date = schedule.groupby("date")["factor"].count()
        lines += [
            "## 池大小时间序列（描述）",
            "",
            f"- min/median/max = {int(by_date.min())} / {int(by_date.median())} / {int(by_date.max())}",
            "",
        ]
    return "\n".join(lines)
