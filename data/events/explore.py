"""
data/events/explore.py  —  探索 AKShare 能拉到哪些事件类数据

直接运行：python -m data.events.explore

会逐个尝试各类接口，打印：
  - 数据形状
  - 时间范围（最早/最晚）
  - 列名
  - 前3行样本
并把结果汇总保存到 data/events/explore_result.txt
"""
import sys
import io
from pathlib import Path
from datetime import datetime

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUTPUT = Path(__file__).parent / "explore_result.txt"
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(msg)


def try_fetch(name: str, fn, **kwargs):
    log(f"\n{'='*60}")
    log(f"【{name}】")
    try:
        df = fn(**kwargs)
        if df is None or (hasattr(df, "__len__") and len(df) == 0):
            log("  ⚠️  返回空数据")
            return None

        log(f"  shape : {df.shape}")
        log(f"  列名  : {list(df.columns)}")

        # 自动找日期列，推断时间范围
        date_cols = [c for c in df.columns
                     if any(k in str(c) for k in ["日期", "时间", "date", "Date", "公告", "披露"])]
        for dc in date_cols:
            try:
                dates = pd.to_datetime(df[dc], errors="coerce").dropna()
                if len(dates):
                    log(f"  [{dc}] 范围: {dates.min().date()} ~ {dates.max().date()}  ({len(dates)}条)")
            except Exception:
                pass

        log(f"\n  前3行:")
        with pd.option_context("display.max_columns", 10,
                               "display.width", 120,
                               "display.max_colwidth", 30):
            for line in df.head(3).to_string(index=False).splitlines():
                log(f"    {line}")
        return df

    except Exception as e:
        log(f"  ❌ 失败: {e}")
        return None


def main():
    log(f"AKShare 事件数据探索  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"AKShare 版本: {ak.__version__}")

    results = {}

    # ── 1. 业绩预告（同花顺，官方公告口径）────────────────────────────────────
    results["业绩预告_同花顺"] = try_fetch(
        "业绩预告（同花顺，官方公告）",
        ak.stock_profit_forecast_ths,
    )

    # ── 2. 股票回购 ───────────────────────────────────────────────────────────
    results["回购_东财"] = try_fetch(
        "股票回购（东方财富）",
        ak.stock_repurchase_em,
    )

    # ── 3. 高管增减持（东财）─────────────────────────────────────────────────
    results["高管增减持_东财"] = try_fetch(
        "高管增减持-按人员（东方财富）",
        ak.stock_hold_management_person_em,
    )

    results["高管增减持_明细"] = try_fetch(
        "高管增减持-明细（东方财富）",
        ak.stock_hold_management_detail_em,
    )

    # ── 4. 股东增减持（上交所/深交所/北交所官方披露）────────────────────────
    results["股东增减持_上交所"] = try_fetch(
        "股东增减持（上交所）",
        ak.stock_share_hold_change_sse,
    )

    results["股东增减持_深交所"] = try_fetch(
        "股东增减持（深交所）",
        ak.stock_share_hold_change_szse,
    )

    # ── 5. 股东增减持（同花顺）──────────────────────────────────────────────
    results["股东增减持_同花顺"] = try_fetch(
        "股东增减持（同花顺）",
        ak.stock_shareholder_change_ths,
    )

    # ── 6. 流通股东持仓变化（东财，适合追踪机构进出）───────────────────────
    results["流通股东变化_东财"] = try_fetch(
        "流通股东持仓变化（东方财富）",
        ak.stock_gdfx_free_holding_change_em,
    )

    # ── 7. 限售股解禁（摘要）─────────────────────────────────────────────────
    results["限售解禁"] = try_fetch(
        "限售股解禁摘要（东方财富）",
        ak.stock_restricted_release_summary_em,
    )

    # ── 8. 定向增发（巨潮）───────────────────────────────────────────────────
    results["定向增发_巨潮"] = try_fetch(
        "定向增发（巨潮）",
        ak.stock_allotment_cninfo,
    )

    # ── 9. 分红历史（巨潮）───────────────────────────────────────────────────
    results["分红_巨潮"] = try_fetch(
        "分红历史（巨潮，000001示例）",
        ak.stock_dividend_cninfo,
        symbol="000001",
    )

    # ── 10. 公告列表（巨潮，单只股票）───────────────────────────────────────
    results["公告列表_巨潮"] = try_fetch(
        "公告列表（巨潮，000001）",
        ak.stock_notice_report,
        symbol="000001",
        date="20240101",
    )

    # ── 11. 个股公告（巨潮，单只股票）──────────────────────────────────────
    results["个股公告_巨潮"] = try_fetch(
        "个股公告（巨潮，000001）",
        ak.stock_individual_notice_report,
        symbol="000001",
    )

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    log(f"\n\n{'='*60}")
    log("汇总")
    log(f"{'='*60}")
    for name, df in results.items():
        if df is not None:
            log(f"  ✅ {name:<16}  {df.shape[0]}行 × {df.shape[1]}列")
        else:
            log(f"  ❌ {name}")

    # 保存结果
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(_log_lines), encoding="utf-8")
    print(f"\n结果已保存至 {OUTPUT}")


if __name__ == "__main__":
    main()
