"""
data/download_all.py  —  下载脚本统一索引（轻量入口，不合并实现）

为何不做大爆炸合并
    各 ``download_*.py`` 的接口/限流/增量续传/产物 schema 差异大
    （按日 / 按股 / 按月 / 截面快照并存）。强行合并会破坏既有
    ``python -m data.download_xxx`` 路径与 resume 语义。
    本模块只做**可发现性**：``--list`` 打印目录；``--run`` 按名转发。

用法
    python -m data.download_all --list
    python -m data.download_all --list --group ashare
    python -m data.download_all --run repurchase
    python -m data.download_all --run moneyflow_ths -- --smoke
"""
from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadEntry:
    key: str
    module: str
    outputs: str
    group: str
    note: str = ""


# 索引：key → 可 ``python -m <module>`` 的脚本（保持原路径不变）
CATALOG: tuple[DownloadEntry, ...] = (
    DownloadEntry(
        "ohlcv", "data.download",
        "prices_hfq / volume / amount / …",
        "core", "主链 OHLCV（含质量报告可选）",
    ),
    DownloadEntry(
        "delisted", "data.download_delisted",
        "并入 prices_*（退市股）",
        "core", "消除幸存者偏差",
    ),
    DownloadEntry(
        "shares", "data.download_shares",
        "circ_shares_em / total_shares_em",
        "core",
    ),
    DownloadEntry(
        "stock_value_em", "data.download_stock_value_em",
        "circ_mv / total_mv / pe_ttm / pb",
        "core", "市值主路径（东财）",
    ),
    DownloadEntry(
        "market_cap", "data.download_market_cap",
        "（deprecated）",
        "core", "已弃用；换手仍走 shares+compute_market_cap",
    ),
    DownloadEntry(
        "st_history", "data.download_st_history",
        "st_history.parquet",
        "core",
    ),
    DownloadEntry(
        "industry", "data.industry.download_industry",
        "industry_map_panel.parquet",
        "core", "行业 PIT",
    ),
    DownloadEntry(
        "margin", "data.download_margin",
        "margin_detail / margin_balance",
        "ashare", "两融长表 + 余额宽表",
    ),
    DownloadEntry(
        "moneyflow", "data.download_moneyflow",
        "moneyflow_large / moneyflow_superlarge",
        "ashare", "东财大单；限流敏感，全市场未就绪勿依赖",
    ),
    DownloadEntry(
        "moneyflow_ths", "data.download_moneyflow_ths",
        "moneyflow_ths_individual_* / industry / concept",
        "ashare", "THS 截面多窗 + 日归档（非东财大单）",
    ),
    DownloadEntry(
        "sector_fund_flow", "data.download_sector_fund_flow",
        "sector_fund_flow / concept_fund_flow",
        "ashare",
    ),
    DownloadEntry(
        "lhb", "data.download_lhb",
        "lhb_detail.parquet",
        "ashare",
    ),
    DownloadEntry(
        "lhb_seats", "data.download_lhb_seats",
        "lhb_yybph / lhb_jgstatistic",
        "ashare", "营业部/机构席位快照",
    ),
    DownloadEntry(
        "lockup", "data.download_lockup",
        "lockup_release.parquet",
        "ashare",
    ),
    DownloadEntry(
        "holder_trade", "data.download_holder_trade",
        "holder_trade / block_trade",
        "ashare", "高管增减持 + 可选大宗",
    ),
    DownloadEntry(
        "shareholder", "data.download_shareholder",
        "shareholder_count.parquet",
        "ashare",
    ),
    DownloadEntry(
        "institution", "data.download_institution",
        "institution_holding.parquet",
        "ashare", "建议 --start-year 2018 回补",
    ),
    DownloadEntry(
        "rank_forecast", "data.download_rank_forecast",
        "rank_forecast.parquet",
        "ashare", "巨潮评级变动",
    ),
    DownloadEntry(
        "research_report", "data.download_research_report",
        "research_report.parquet",
        "ashare", "东财研报（逐股慢）",
    ),
    DownloadEntry(
        "repurchase", "data.download_repurchase",
        "repurchase.parquet",
        "ashare",
    ),
    DownloadEntry(
        "dzjy_yybph", "data.download_dzjy_yybph",
        "dzjy_yybph.parquet",
        "ashare", "大宗营业部排行快照",
    ),
    DownloadEntry(
        "northbound", "data.download_northbound",
        "northbound_*",
        "ashare", "约 2024-08 后停更；默认不进 IC",
    ),
    DownloadEntry(
        "yjbb", "data.events.download_yjbb",
        "yjbb.parquet",
        "events", "业绩快报/正式稿",
    ),
    DownloadEntry(
        "yjyg", "data.events.download_yjyg",
        "yjyg.parquet",
        "events", "业绩预告",
    ),
)


def _by_key() -> dict[str, DownloadEntry]:
    return {e.key: e for e in CATALOG}


def list_catalog(group: str | None = None) -> None:
    rows = [e for e in CATALOG if group is None or e.group == group]
    if not rows:
        print(f"(无匹配 group={group!r})")
        return
    groups = sorted({e.group for e in rows})
    print("下载脚本索引（python -m data.download_all --list）")
    print("说明：不合并实现；各脚本仍用 python -m <module>\n")
    for g in groups:
        print(f"[{g}]")
        for e in rows:
            if e.group != g:
                continue
            note = f"  # {e.note}" if e.note else ""
            print(f"  {e.key:<18}  python -m {e.module}")
            print(f"  {'':18}  → {e.outputs}{note}")
        print()
    print("转发示例: python -m data.download_all --run moneyflow_ths -- --smoke")


def run_entry(key: str, passthrough: list[str]) -> int:
    catalog = _by_key()
    if key not in catalog:
        print(f"未知 key={key!r}。可用: {', '.join(catalog)}")
        return 2
    entry = catalog[key]
    # 模拟 ``python -m module …``：改写 argv 后 run_module
    sys.argv = [entry.module] + list(passthrough)
    runpy.run_module(entry.module, run_name="__main__", alter_sys=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="下载脚本统一索引（--list）/ 按 key 转发（--run）",
    )
    p.add_argument("--list", action="store_true", help="打印目录")
    p.add_argument(
        "--group",
        choices=("core", "ashare", "events"),
        default=None,
        help="仅列出某一分组",
    )
    p.add_argument("--run", metavar="KEY", help="按索引 key 转发到对应模块")
    args, passthrough = p.parse_known_args(argv)

    if args.list or (not args.run and not args.list):
        # 无参数时默认 --list，便于发现
        list_catalog(args.group)
        if not args.run:
            return 0

    if args.run:
        # 允许 ``--run x -- --smoke`` 或 ``--run x --smoke``
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]
        return run_entry(args.run, passthrough)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
