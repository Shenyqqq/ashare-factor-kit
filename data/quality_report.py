"""
data/quality_report.py  —  数据质量监控报告

读 data/raw/ 下全部 parquet，输出每只股票 / 全市场的质量摘要：
  - 日期覆盖率（有效日期数 / 总交易日跨度）
  - NaN 率
  - 零值率（volume=0 占比，停牌代理）
  - 极端收益率率（|ret| > 20% 占比，后复权日收益率）
  - 财务字段缺失率（按财务面板列）
  - 行业覆盖率（industry_map_panel.parquet 中有效 sw_l2 占比）

输出 markdown 报告到 data/quality_report_YYYYMMDD.md。
异常时 logger.error 告警（不中断，除非 strict=True）。
如存在上一版本报告，输出与上一版的 diff 摘要（退市股数量、覆盖率变化等）。

用法:
    python -m data.quality_report                 # 生成报告
    python -m data.quality_report --strict        # 异常时 raise
    python -m data.quality_report --data-dir data/raw
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# 报告阈值（超过即视为异常，触发 logger.error）
THRESHOLDS = {
    "min_coverage":   0.80,   # 日期覆盖率下限
    "max_nan_rate":   0.30,   # NaN 率上限
    "max_zero_rate":  0.30,   # volume=0 占比上限
    "max_extreme_ret": 0.01,  # |ret|>20% 占比上限
}

# 财务面板列（缺失率统计）
FIN_COLS = [
    "roe", "bvps", "total_assets", "eps",
    "gross_profit_margin", "operating_cashflow",
    "debt_ratio", "net_profit_growth",
    "revenue_growth", "net_profit_margin",
]


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"读取 {path.name} 失败: {e}")
        return None


def _panel_quality(df: pd.DataFrame, name: str) -> dict:
    """计算宽表（index=日期, columns=股票）的基本质量指标。"""
    if df is None or df.empty:
        return {}
    n_cells = df.size
    n_nan = int(df.isna().sum().sum())
    # 日期覆盖率：有效（非全 NaN）列占比 + 平均每列有效日期占比
    col_valid_ratio = (df.notna().any()).mean()
    avg_date_cov = float(df.notna().mean().mean())
    return {
        "shape":         f"{df.shape[0]}×{df.shape[1]}",
        "n_cells":       n_cells,
        "nan_rate":      round(n_nan / max(n_cells, 1), 4),
        "col_coverage":  round(float(col_valid_ratio), 4),
        "date_coverage": round(avg_date_cov, 4),
    }


def _volume_zero_rate(vol: pd.DataFrame | None) -> float | None:
    if vol is None or vol.empty:
        return None
    n_zero = int((vol == 0).sum().sum())
    return round(n_zero / max(vol.size, 1), 4)


def _extreme_return_rate(prices: pd.DataFrame | None) -> float | None:
    if prices is None or prices.empty:
        return None
    ret = prices.pct_change()
    n_extreme = int((ret.abs() > 0.20).sum().sum())
    n_valid = int(ret.notna().sum().sum())
    return round(n_extreme / max(n_valid, 1), 4)


def _financial_missing(fin: pd.DataFrame | None) -> dict:
    if fin is None or fin.empty:
        return {}
    out = {}
    n = max(len(fin), 1)
    for col in FIN_COLS:
        if col in fin.columns:
            out[col] = round(float(fin[col].isna().mean()), 4)
    return out


def _industry_coverage(ind_path: Path) -> float | None:
    if not ind_path.exists():
        return None
    df = _safe_read_parquet(ind_path)
    if df is None or df.empty:
        return None
    # 期望含 sw_l2 列；若是宽表则取首个值列
    if "sw_l2" in df.columns:
        cov = float(df["sw_l2"].notna().mean())
    else:
        cov = float(df.notna().mean().mean())
    return round(cov, 4)


def _find_prev_report(data_dir: Path) -> Path | None:
    """找上一份历史报告（按日期排序，最新一份之外的最近一份）。"""
    if not data_dir.exists():
        return None
    reports = sorted(data_dir.glob("quality_report_*.md"))
    if len(reports) < 2:
        return None
    return reports[-2]  # 倒数第二份（最新是即将写的）


def _diff_prev_report(prev_path: Path, curr_summary: dict) -> str:
    """从上一份报告正则提取关键数字，生成 diff 摘要。"""
    try:
        text = prev_path.read_text(encoding="utf-8")
    except Exception:
        return ""

    def _extract(key: str) -> str | None:
        m = re.search(rf"{key}[:：]\s*([0-9.\-]+)", text)
        return m.group(1) if m else None

    lines = ["", "## 与上一版本 diff 摘要", ""]
    mapping = {
        "股票数":   str(curr_summary.get("n_stocks", "-")),
        "日期跨度": str(curr_summary.get("n_dates", "-")),
        "退市股数": str(curr_summary.get("n_delisted", "-")),
    }
    for k, v in mapping.items():
        old = _extract(k)
        if old is not None and old != v:
            lines.append(f"- {k}: {old} → {v}")
    if len(lines) <= 2:
        lines.append("- 关键指标无变化")
    return "\n".join(lines) + "\n"


def generate_quality_report(
    data_dir: str = "data/raw",
    strict: bool = False,
) -> str:
    """
    生成数据质量报告。

    Parameters
    ----------
    data_dir : str
        raw parquet 目录，默认 'data/raw'。
    strict : bool
        True=发现阈值越界时 raise；False=仅 logger.error。

    Returns
    -------
    str
        报告文件路径。
    """
    raw = Path(data_dir)
    if not raw.exists():
        msg = f"数据目录不存在: {raw}"
        logger.error(msg)
        if strict:
            raise FileNotFoundError(msg)
        return ""

    today_str = datetime.today().strftime("%Y%m%d")
    out_path = raw.parent / f"quality_report_{today_str}.md"

    prices = _safe_read_parquet(raw / "prices_hfq.parquet")
    if prices is None:
        prices = _safe_read_parquet(raw / "close_hfq.parquet")
    prices_raw = _safe_read_parquet(raw / "prices_raw.parquet")
    volume = _safe_read_parquet(raw / "volume.parquet")
    amount = _safe_read_parquet(raw / "amount.parquet")
    open_ = _safe_read_parquet(raw / "open_hfq.parquet")
    high = _safe_read_parquet(raw / "high_hfq.parquet")
    low = _safe_read_parquet(raw / "low_hfq.parquet")
    fin = _safe_read_parquet(raw / "financial_indicators.parquet")

    # 行业面板：优先 PIT 时间序列版本（落在 data/raw/）
    ind_panel_path = raw / "industry_map_panel.parquet"
    if not ind_panel_path.exists():
        ind_panel_path = raw.parent / "industry" / "industry_map_panel.parquet"
    if not ind_panel_path.exists():
        ind_panel_path = raw / "industry_map.parquet"
    ind_cov = _industry_coverage(ind_panel_path)

    # 退市股数量：从 stock_list.parquet 读
    n_delisted = "-"
    sl_path = raw.parent / "universe" / "stock_list.parquet"
    if sl_path.exists():
        try:
            sl = pd.read_parquet(sl_path)
            if "delist_date" in sl.columns:
                n_delisted = int(sl["delist_date"].notna().sum())
        except Exception as e:
            logger.warning(f"读取 stock_list 失败: {e}")

    n_stocks = prices.shape[1] if prices is not None else 0
    n_dates = prices.shape[0] if prices is not None else 0

    prices_q = _panel_quality(prices, "prices_hfq")
    vol_q = _panel_quality(volume, "volume")
    amt_q = _panel_quality(amount, "amount")
    vol_zero = _volume_zero_rate(volume)
    extreme_ret = _extreme_return_rate(prices)
    fin_miss = _financial_missing(fin)

    # 阈值检查
    violations: list[str] = []
    if prices_q:
        if prices_q.get("date_coverage", 1) < THRESHOLDS["min_coverage"]:
            violations.append(
                f"prices 日期覆盖率 {prices_q['date_coverage']} < "
                f"{THRESHOLDS['min_coverage']}"
            )
        if prices_q.get("nan_rate", 0) > THRESHOLDS["max_nan_rate"]:
            violations.append(
                f"prices NaN 率 {prices_q['nan_rate']} > "
                f"{THRESHOLDS['max_nan_rate']}"
            )
    if vol_zero is not None and vol_zero > THRESHOLDS["max_zero_rate"]:
        violations.append(
            f"volume 0 值率 {vol_zero} > {THRESHOLDS['max_zero_rate']}"
        )
    if extreme_ret is not None and extreme_ret > THRESHOLDS["max_extreme_ret"]:
        violations.append(
            f"|ret|>20% 占比 {extreme_ret} > {THRESHOLDS['max_extreme_ret']}"
        )

    # ── 拼装 markdown ──────────────────────────────────────────────────────
    md = []
    md.append(f"# 数据质量报告 {today_str}\n")
    md.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"数据目录: `{raw}`\n")

    md.append("## 全市场概览\n")
    md.append(f"- 股票数: {n_stocks}")
    md.append(f"- 日期跨度: {n_dates} 个交易日")
    md.append(f"- 退市股数: {n_delisted}")
    md.append(f"- 行业覆盖率: {ind_cov}")
    md.append("")

    md.append("## 价格面板\n")
    if prices_q:
        for k, v in prices_q.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- 无 prices_hfq.parquet")
    md.append("")

    md.append("## 成交量/成交额\n")
    if vol_q:
        md.append("### volume")
        for k, v in vol_q.items():
            md.append(f"- {k}: {v}")
        md.append(f"- zero_rate: {vol_zero}")
    else:
        md.append("- 无 volume.parquet")
    md.append("")
    if amt_q:
        md.append("### amount")
        for k, v in amt_q.items():
            md.append(f"- {k}: {v}")
    md.append("")

    md.append("## 收益率质量\n")
    md.append(f"- |ret|>20% 占比: {extreme_ret}")
    md.append("")

    md.append("## 财务面板缺失率\n")
    if fin_miss:
        for k, v in fin_miss.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- 无 financial_indicators.parquet 或无目标列")
    md.append("")

    md.append("## 其他 OHLCV\n")
    for name, p in [("open_hfq", "open_hfq.parquet"),
                    ("high_hfq", "high_hfq.parquet"),
                    ("low_hfq", "low_hfq.parquet"),
                    ("prices_raw", "prices_raw.parquet")]:
        df = _safe_read_parquet(raw / p)
        q = _panel_quality(df, name)
        if q:
            md.append(f"### {name}")
            for k, v in q.items():
                md.append(f"- {k}: {v}")
            md.append("")

    if violations:
        md.append("## 阈值越界告警\n")
        for v in violations:
            md.append(f"- ❌ {v}")
        md.append("")
        for v in violations:
            logger.error(f"[quality] {v}")
        if strict:
            raise RuntimeError(f"数据质量阈值越界: {violations}")
    else:
        md.append("## 阈值检查\n")
        md.append("- ✅ 所有指标在阈值范围内")
        md.append("")

    # 与上一版本 diff
    prev = _find_prev_report(raw.parent)
    if prev is not None:
        summary = {
            "n_stocks": n_stocks,
            "n_dates": n_dates,
            "n_delisted": n_delisted,
        }
        md.append(_diff_prev_report(prev, summary))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"数据质量报告已生成: {out_path}")
    return str(out_path)


def main(data_dir: str = "data/raw", strict: bool = False):
    generate_quality_report(data_dir=data_dir, strict=strict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--strict", action="store_true",
                        help="阈值越界时 raise 而非仅 logger.error")
    args = parser.parse_args()
    main(data_dir=args.data_dir, strict=args.strict)
