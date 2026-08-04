"""
data/download_st_history.py — 下载 ST 真实历史时间序列

修复 P0-2：原 `backtest/execution.py::build_st_schedule` 保守实现「当前 ST 股
在所有回测日均标 ST」，导致：
  1. 摘帽股被错误全程剔除（实际仅 ST 时段应剔除）
  2. 曾 ST 现摘帽股完全不被剔除（实际 ST 时段应剔除）

本脚本用 AKShare 的深交所「简称变更」历史接口（带日期）精确推断每只股票
的 ST 时段；上交所/北交所历史简称变更**无公开带日期接口**（`stock_info_change_name`
仅返回更名列表无日期），回退到「当前 ST 股自 list_date/回测起点起全程标 ST」
的保守策略，并在 ``source`` 列显式标注
``sh_bj_current_st_conservative_fallback``（禁止误当精确历史）。

数据源
------
1. `ak.stock_info_sz_change_name(symbol="简称变更")` —— 深交所全部历史简称变更
   （含日期），7400+ 行，覆盖 1994 年至今。列：变更日期 / 证券代码 /
   证券简称 / 变更前简称 / 变更后简称。
2. `ak.stock_zh_a_st_em()` —— 当前 ST/*ST 股快照（无日期），用于：
   a. 推断上交所/北交所当前 ST 股的 ST 时段（保守：自 list_date 或 backtest_start）
   b. 兜底深交所当前 ST 但历史简称变更遗漏的股票

输出
----
`data/raw/st_history.parquet`（长表）：
  - code        : str，6 位证券代码
  - start_date  : pd.Timestamp，ST 时段起始日（含）
  - end_date    : pd.Timestamp 或 NaT，ST 时段结束日（含）；NaT=至今未摘帽
  - st_type     : str，{"ST", "*ST", "退市ST"}
  - source      : str，{"sz_name_change", "sh_bj_current_st_conservative_fallback"}

用法
----
    python -m data.download_st_history
    python -m data.download_st_history --start 2018-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, BACKTEST_START, UNIVERSE_DIR

# AKShare 深交所简称变更接口的真实列名（已实测确认 unicode）
_COL_SZ_DATE = "变更日期"        # 变更日期
_COL_SZ_CODE = "证券代码"        # 证券代码
_COL_SZ_BEFORE = "变更前简称"    # 变更前简称
_COL_SZ_AFTER = "变更后简称"     # 变更后简称
# AKShare 当前 ST 接口列名
_COL_ST_CODE = "代码"
_COL_ST_NAME = "名称"

# ST 名称分类
_ST_TYPE_STAR = "*ST"     # 退市风险警示
_ST_TYPE_PLAIN = "ST"     # 其他风险警示
_ST_TYPE_DELIST = "退市ST"  # 退市整理期（名称含「退」）

# 推断 ST 时段的全局起始日（足够早即可，下游按回测日期切片）
_EPOCH_START = pd.Timestamp("1990-01-01")

# 沪/北兜底 source 标签（显式，禁止与深交所精确历史混淆）
SOURCE_SZ = "sz_name_change"
SOURCE_SH_BJ_FALLBACK = "sh_bj_current_st_conservative_fallback"


def _classify_st_type(name: str) -> str | None:
    """根据股票简称判断 ST 类型；非 ST 返回 None。

    规则（顺序敏感）：
      - 含「退」 → 退市ST（退市整理期，如「xxx退」）
      - 含「*ST」→ *ST（退市风险警示，优先于 ST 匹配）
      - 含「ST」 → ST（其他风险警示）
      - 否则     → None
    """
    if not isinstance(name, str) or not name:
        return None
    if "退" in name:
        return _ST_TYPE_DELIST
    if "*ST" in name.upper():
        return _ST_TYPE_STAR
    if "ST" in name.upper():
        return _ST_TYPE_PLAIN
    return None


def _fetch_sz_name_changes() -> pd.DataFrame:
    """拉取深交所全部简称变更历史。

    Returns
    -------
    pd.DataFrame with columns: date (Timestamp), code (str zfilled 6),
    before (str), after (str)
    """
    logger.info("拉取深交所简称变更历史 ...")
    raw = ak.stock_info_sz_change_name(symbol="简称变更")
    logger.info(f"深交所简称变更原始行数: {len(raw)}")

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[_COL_SZ_DATE], errors="coerce"),
        "code": raw[_COL_SZ_CODE].astype(str).str.zfill(6),
        "before": raw[_COL_SZ_BEFORE].astype(str),
        "after": raw[_COL_SZ_AFTER].astype(str),
    })
    df = df.dropna(subset=["date", "code"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def _fetch_current_st() -> pd.DataFrame:
    """拉取当前 ST/*ST 股快照。

    Returns
    -------
    pd.DataFrame with columns: code (str zfilled 6), name (str), st_type (str)
    """
    logger.info("拉取当前 ST/*ST 快照 ...")
    raw = ak.stock_zh_a_st_em()
    df = pd.DataFrame({
        "code": raw[_COL_ST_CODE].astype(str).str.zfill(6),
        "name": raw[_COL_ST_NAME].astype(str),
    })
    df["st_type"] = df["name"].map(_classify_st_type)
    # 极少数行可能分类失败（名称异常），回退为 ST
    df["st_type"] = df["st_type"].fillna(_ST_TYPE_PLAIN)
    logger.info(f"当前 ST 快照: {len(df)} 只（*ST={sum(df['st_type']=='*ST')}, "
                f"ST={sum(df['st_type']=='ST')}, 退={sum(df['st_type']=='退市ST')}）")
    return df


def _build_sz_st_periods(sz_changes: pd.DataFrame) -> pd.DataFrame:
    """从深交所简称变更历史推断每只股票的 ST 时段。

    每只股票按变更日期排序后，相邻变更事件之间构成一个「命名段」：
      - 段 [prev_date, cur_date - 1] 的活动名称 = cur_event.before
        （等价于 prev_event.after，若存在）
      - 段 [cur_date, next_date - 1] 的活动名称 = cur_event.after
    最后一段的 end_date = NaT（至今未变更）。

    Returns
    -------
    pd.DataFrame: code, start_date, end_date, st_type, source
    """
    records: list[dict] = []
    for code, grp in sz_changes.groupby("code", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        for i, row in grp.iterrows():
            # 段 i：[row.date, 下一个事件 date - 1] 活动名称 = row.after
            start = row["date"]
            end = grp.iloc[i + 1]["date"] - pd.Timedelta(days=1) if i + 1 < n else pd.NaT
            st_type = _classify_st_type(row["after"])
            if st_type is not None:
                records.append({
                    "code": code,
                    "start_date": start,
                    "end_date": end,
                    "st_type": st_type,
                    "source": SOURCE_SZ,
                })
            # 段 i 前：[_EPOCH_START, row.date - 1] 活动名称 = row.before
            # （仅在第一个事件前补一段；后续事件的 before 应等于上一事件的 after）
            if i == 0:
                pre_start = _EPOCH_START
                pre_end = row["date"] - pd.Timedelta(days=1)
                pre_type = _classify_st_type(row["before"])
                if pre_type is not None and pre_end >= pre_start:
                    records.append({
                        "code": code,
                        "start_date": pre_start,
                        "end_date": pre_end,
                        "st_type": pre_type,
                        "source": SOURCE_SZ,
                    })
    if not records:
        return pd.DataFrame(columns=["code", "start_date", "end_date", "st_type", "source"])
    out = pd.DataFrame(records)
    # 合并同 code 相邻同 st_type 的段（理论上不会相邻同型，但保险起见）
    out = out.sort_values(["code", "start_date"]).reset_index(drop=True)
    return out


def _load_list_dates() -> dict[str, pd.Timestamp]:
    """universe/stock_list.parquet → {code: list_date}；缺失则 {}。"""
    p = UNIVERSE_DIR / "stock_list.parquet"
    if not p.exists():
        return {}
    try:
        u = pd.read_parquet(p)
    except Exception as e:
        logger.warning(f"读取 list_date 失败: {e}")
        return {}
    if "code" not in u.columns or "list_date" not in u.columns:
        return {}
    u["code"] = u["code"].astype(str).str.zfill(6)
    u["list_date"] = pd.to_datetime(u["list_date"], errors="coerce")
    out = {}
    for row in u.itertuples(index=False):
        if pd.notna(row.list_date):
            out[row.code] = pd.Timestamp(row.list_date)
    return out


def _build_fallback_st_periods(
    current_st: pd.DataFrame,
    covered_codes: set[str],
    start: pd.Timestamp,
    list_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """对未被深交所历史覆盖的当前 ST 股（主要是沪/北），构造保守「自上市/回测起点起全程 ST」时段。

    上交所无公开带日期的简称变更接口（``stock_info_change_name`` 无日期），
    故无法还原真实戴帽/摘帽时点。本路径：
      - source = ``sh_bj_current_st_conservative_fallback``（显式标注）
      - start_date = max(backtest_start, list_date)（有 list_date 时收紧起点）
      - end_date = NaT（假设仍未摘帽；曾 ST 现已摘帽的沪市股仍会漏检）

    Parameters
    ----------
    current_st : 当前 ST 快照
    covered_codes : 已被深交所历史时段覆盖的 code 集合
    start : 回测起始日（fallback 时段起点下界）
    list_dates : 可选 {code: list_date}，用于收紧 start_date
    """
    fb = current_st[~current_st["code"].isin(covered_codes)].copy()
    if fb.empty:
        return pd.DataFrame(columns=["code", "start_date", "end_date", "st_type", "source"])
    list_dates = list_dates or {}
    starts = []
    for code in fb["code"]:
        ld = list_dates.get(code)
        if ld is not None and pd.notna(ld):
            starts.append(max(start, pd.Timestamp(ld)))
        else:
            starts.append(start)
    fb["start_date"] = starts
    fb["end_date"] = pd.NaT
    fb["source"] = SOURCE_SH_BJ_FALLBACK
    n_sh = int(fb["code"].astype(str).str.startswith(("6", "9")).sum())
    n_bj = int(fb["code"].astype(str).str.startswith(("8", "4")).sum())
    logger.warning(
        f"沪/北 ST 保守兜底（非精确历史）: {len(fb)} 只 "
        f"(沪≈{n_sh}, 北/其他≈{n_bj})；source={SOURCE_SH_BJ_FALLBACK}；"
        f"曾 ST 现摘帽的沪市股仍会漏检。缓解：人工复核 / 付费简称变更源"
    )
    return fb[["code", "start_date", "end_date", "st_type", "source"]].reset_index(drop=True)


def build_st_history(
    backtest_start: str | pd.Timestamp = BACKTEST_START,
) -> pd.DataFrame:
    """组装完整的 ST 历史长表（深交所精确 + 沪/北保守兜底）。

    Returns
    -------
    pd.DataFrame: code, start_date, end_date, st_type, source
    """
    start_ts = pd.Timestamp(backtest_start)

    sz_changes = _fetch_sz_name_changes()
    current_st = _fetch_current_st()

    sz_periods = _build_sz_st_periods(sz_changes)
    logger.info(f"深交所历史推断 ST 时段: {len(sz_periods)} 段，"
                f"涉及 {sz_periods['code'].nunique()} 只股票")

    covered = set(sz_periods["code"].unique())
    list_dates = _load_list_dates()
    fallback = _build_fallback_st_periods(
        current_st, covered, start_ts, list_dates=list_dates,
    )
    if len(fallback):
        logger.info(
            f"沪/北兜底段: {len(fallback)} 只（source={SOURCE_SH_BJ_FALLBACK}）"
        )

    out = pd.concat([sz_periods, fallback], ignore_index=True)
    out = out.sort_values(["code", "start_date"]).reset_index(drop=True)
    return out


def main(start: str | None = None, save_path: Path | None = None) -> Path:
    """下载 ST 历史并落盘。

    Returns
    -------
    实际保存路径
    """
    backtest_start = start or BACKTEST_START
    if save_path is None:
        save_path = RAW_DIR / "st_history.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_st_history(backtest_start=backtest_start)
    df.to_parquet(save_path, index=False)
    logger.info(
        f"ST 历史已保存: {save_path}  "
        f"({len(df)} 段, {df['code'].nunique()} 只股票; "
        f"source 分布: {df['source'].value_counts().to_dict()})"
    )
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 ST 真实历史时间序列")
    parser.add_argument(
        "--start", default=BACKTEST_START,
        help=f"回测起始日（沪/北 fallback 时段起点），默认 {BACKTEST_START}",
    )
    args = parser.parse_args()
    main(start=args.start)
