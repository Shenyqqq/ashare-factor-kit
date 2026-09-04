"""
data/download_market_cap.py  —  【DEPRECATED】遗留市值下载（勿作日常主链）

.. deprecated:: 2026-07-29
    日常市值请用::

        python -m data.download_shares
        python -m data.compute_market_cap

    产出 ``total_mv`` / ``circ_mv`` / ``turnover_rate``（PIT 安全、与 prices_raw 对齐）。
    本模块保留仅供对照/应急；入口会打 DeprecationWarning。

存储文件（宽表：index=trade_date, columns=股票代码，单位=元）:
    data/raw/total_mv.parquet       — 总市值
    data/raw/circ_mv.parquet        — 流通市值（仅 Tushare 模式可用）
    data/raw/turnover_rate.parquet  — 换手率 %（仅 Tushare 模式可用）

两条路径：
  1. 主路径（推荐）：Tushare pro.daily_basic(trade_date=...)
     - 按交易日全市场一次性返回 5000+ 股
     - 字段：total_mv（万元）/ circ_mv（万元）/ turnover_rate（%）等
     - 单位换算：万元 × 1e4 = 元；turnover_rate 不换算
     - 需要 2000 积分；token 缺失或权限不足会抛 TushareException
  2. 降级路径：AKShare stock_zh_valuation_baidu(symbol, indicator='总市值', period='全部')
     - 单股返回上市至最新交易日的日频总市值（单位=亿元）
     - 单位换算：亿元 × 1e8 = 元
     - 仅 total_mv 可用；circ_mv / turnover_rate 不可用（文件不生成）
     - 单次调用 ~0.7s，5000 股串行 ~60 min，并发 4 线程 ~20 min
     - 适合无 Tushare token 的备用方案

特性：
  - 自动选择路径：token 存在且权限足够 → Tushare；否则降级 AKShare
  - 增量下载 + 断点续传：已存在的交易日（Tushare）/已下载的股票（AKShare）跳过
  - 失败重试、并发控制（Tushare 每分钟 ≤200 次，AKShare 4 线程并发）
  - 涨跌停/退市股通过 stock_list.parquet 自动获取（与主下载流程一致）

用法（不推荐）:
    python -m data.download_market_cap --start 2018-01-01 --end 2026-07-01
"""
import argparse
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR, TUSHARE_TOKEN

_DEPRECATION_MSG = (
    "data.download_market_cap 已废弃：请改用 "
    "`python -m data.download_shares` + `python -m data.compute_market_cap`。"
    "本模块仅应急/对照，勿覆盖生产 circ_mv/turnover_rate。"
)


# ── 字段定义 ──────────────────────────────────────────────────────────────────

# 输出文件名 → Tushare daily_basic 列名（遗留分支；日常请用 download_shares + compute_market_cap）
TUSHARE_FIELDS = {
    "total_mv":      "total_mv",       # 万元 → 元
    "circ_mv":       "circ_mv",        # 万元 → 元
    "turnover_rate": "turnover_rate",  # %，原值保留
}

# 单位换算因子（Tushare 返回的万元 → 元）
TUSHARE_UNIT_FACTOR = {
    "total_mv":      1e4,
    "circ_mv":       1e4,
    "turnover_rate": 1.0,
}

# AKShare baidu 估值接口仅支持 total_mv（单位=亿元 → 元）
AKSHARE_UNIT_FACTOR = 1e8

# Tushare 遗留分支用；本仓库日常市值走 download_shares + compute_market_cap
TUSHARE_RATE_LIMIT_PER_MIN = 150
# AKShare 并发线程数（baidu 估值接口对并发不严格，4 线程实测稳定）
AKSHARE_WORKERS = 4


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _load_stock_list() -> list[str]:
    """从 universe/stock_list.parquet 读取股票代码列表（含退市股）。

    fallback：调 data.download.get_stock_list 现场拉取。
    """
    path = UNIVERSE_DIR / "stock_list.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "code" in df.columns:
            codes = df["code"].astype(str).str.zfill(6).tolist()
            logger.info(f"从 {path.name} 读取 {len(codes)} 只股票")
            return codes
    # fallback：现场拉
    logger.info(f"{path} 不存在，现场拉取股票列表")
    from data.download import get_stock_list, filter_universe
    sl = get_stock_list(include_delisted=True)
    sl = filter_universe(sl)
    codes = sl["code"].astype(str).str.zfill(6).tolist()
    return codes


def _get_trade_dates(start: str, end: str) -> list[pd.Timestamp]:
    """获取 [start, end] 区间的实际交易日（用 AKShare 新浪交易日历）。"""
    import akshare as ak
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"])
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        trade_dates = sorted(dates[(dates >= s) & (dates <= e)].tolist())
        logger.info(f"交易日历: {len(trade_dates)} 个交易日 ({trade_dates[0].date()} → {trade_dates[-1].date()})")
        return trade_dates
    except Exception as e:
        logger.warning(f"交易日历获取失败，回退到工作日: {e}")
        return pd.bdate_range(start, end).tolist()


def _normalize_tushare_code(ts_code: str) -> str:
    """Tushare '000001.SZ' → '000001'"""
    return str(ts_code).split(".")[0].zfill(6)


def _save_wide(panel: dict[str, pd.Series], path: Path) -> pd.DataFrame:
    """dict{code: Series(index=date)} → 宽表 parquet (index=date, columns=code)"""
    if not panel:
        df = pd.DataFrame()
        df.to_parquet(path)
        return df
    df = pd.DataFrame(panel)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(path)
    return df


# ── 主路径：Tushare daily_basic ────────────────────────────────────────────────

def _init_tushare() -> "object":
    """初始化 Tushare pro_api，token 缺失抛 ValueError。"""
    if not TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN 未配置（.env 缺失或为空）")
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def _test_tushare_daily_basic(pro) -> bool:
    """试一个交易日确认权限与 schema，True=可用。（遗留分支；日常勿用）"""
    try:
        df = pro.daily_basic(trade_date="20240105")
        if df is None or df.empty:
            logger.warning("Tushare daily_basic 返回空，可能权限不足")
            return False
        cols = set(df.columns)
        required = {"ts_code", "trade_date", "total_mv", "circ_mv", "turnover_rate"}
        missing = required - cols
        if missing:
            logger.warning(f"Tushare daily_basic 缺少字段 {missing}")
            return False
        logger.info(f"Tushare daily_basic 权限 OK，schema 测试 shape={df.shape}")
        return True
    except Exception as e:
        logger.warning(f"Tushare daily_basic 调用失败: {type(e).__name__}: {e}")
        return False


def download_via_tushare(
    start: str,
    end: str,
    out_dir: Path = RAW_DIR,
    rate_limit_per_min: int = TUSHARE_RATE_LIMIT_PER_MIN,
) -> dict[str, Path]:
    """按交易日循环调 Tushare daily_basic，合并成宽表 parquet。

    增量续传：已有数据中存在的 trade_date 跳过。
    返回 {字段名: 输出路径}。
    """
    pro = _init_tushare()
    if not _test_tushare_daily_basic(pro):
        raise RuntimeError("Tushare daily_basic 不可用（积分/权限不足）")

    out_paths = {field: out_dir / f"{field}.parquet" for field in TUSHARE_FIELDS}
    # 加载已有数据（断点续传）
    existing_panels: dict[str, dict[str, pd.Series]] = {}
    done_dates: set[pd.Timestamp] = set()
    for field, path in out_paths.items():
        if path.exists():
            df = pd.read_parquet(path)
            existing_panels[field] = {c: df[c].dropna() for c in df.columns}
            done_dates.update(pd.to_datetime(df.index))
        else:
            existing_panels[field] = {}
    if done_dates:
        logger.info(f"已有数据覆盖 {len(done_dates)} 个交易日，将增量跳过")

    trade_dates = _get_trade_dates(start, end)
    pending = [d for d in trade_dates if d not in done_dates]
    logger.info(f"待下载交易日: {len(pending)} / {len(trade_dates)}")

    if not pending:
        logger.info("所有交易日已是最新，跳过")
        return out_paths

    # 简易限速（遗留分支）
    min_interval = 60.0 / max(rate_limit_per_min, 1)
    lock = threading.Lock()
    last_save = time.time()
    save_every = 50

    for i, dt in enumerate(pending, 1):
        td_str = dt.strftime("%Y%m%d")
        time.sleep(max(0.0, min_interval - 0.05))
        try:
            df = pro.daily_basic(trade_date=td_str)
        except Exception as e:
            logger.warning(f"Tushare daily_basic({td_str}) 失败: {e}，跳过")
            continue
        if df is None or df.empty:
            logger.debug(f"{td_str} 返回空（可能非交易日）")
            continue
        df["code"] = df["ts_code"].apply(_normalize_tushare_code)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for field, ts_col in TUSHARE_FIELDS.items():
            factor = TUSHARE_UNIT_FACTOR[field]
            sub = df[["trade_date", "code", ts_col]].dropna(subset=[ts_col])
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                code = row["code"]
                val = float(row[ts_col]) * factor
                s = existing_panels[field].get(code)
                if s is None:
                    existing_panels[field][code] = pd.Series({row["trade_date"]: val})
                else:
                    s.loc[row["trade_date"]] = val
                    existing_panels[field][code] = s.sort_index()

        if i % 10 == 0:
            logger.info(f"Tushare 进度 {i}/{len(pending)} ({td_str})")
        with lock:
            if time.time() - last_save > save_every * min_interval * 0.5 and i % save_every == 0:
                for field, path in out_paths.items():
                    _save_wide(existing_panels[field], path)
                last_save = time.time()
                logger.info(f"  → 中途落盘 {i}/{len(pending)}")

    # 最终落盘
    for field, path in out_paths.items():
        df = _save_wide(existing_panels[field], path)
        logger.info(f"  {path.name}: shape={df.shape}")
    return out_paths


# ── 降级路径：AKShare baidu 估值（仅 total_mv）────────────────────────────────

def _fetch_one_baidu(code: str, period: str = "近十年") -> pd.Series | None:
    """单股调 baidu 估值返回 Series(index=date, name=code)，单位=元。

    period 选择（百度接口对总行数有 ~1100 上限，period 越短密度越高）：
      - "近一年": 365 行，1 日间隔（日频，仅最近 1 年）
      - "近三年": 1096 行，1 日间隔（日频，最近 3 年）
      - "近五年": 913 行，2 日间隔
      - "近十年": 731 行，5 日间隔（周频采样，覆盖 2016-2026）
      - "全部":   606 行，上市至今稀疏采样（不推荐，密度最差）

    本期默认 "近十年"：覆盖 2018-2026 目标区间且密度最佳。
    后续 clean/universe 阶段可用 ffill(limit=7) 把 5 日采样扩到日频。
    """
    import akshare as ak
    try:
        df = ak.stock_zh_valuation_baidu(
            symbol=code, indicator="总市值", period=period
        )
    except Exception as e:
        return None, e
    if df is None or df.empty:
        return None, None
    # 列：date, value（单位=亿元）
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values * AKSHARE_UNIT_FACTOR,
        index=pd.to_datetime(df["date"]),
        name=code,
    ).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s, None


def download_via_akshare(
    start: str,
    end: str,
    out_dir: Path = RAW_DIR,
    codes: list[str] = None,
    workers: int = AKSHARE_WORKERS,
    save_every: int = 50,
    period: str = "近十年",
) -> dict[str, Path]:
    """单股循环调 baidu 估值，仅下载 total_mv。

    增量续传：已存在于 total_mv.parquet 的 code 跳过。
    返回 {字段名: 输出路径}；circ_mv / turnover_rate 不生成。
    """
    if codes is None:
        codes = _load_stock_list()
    out_path = out_dir / "total_mv.parquet"

    # 加载已有
    panel: dict[str, pd.Series] = {}
    if out_path.exists():
        df = pd.read_parquet(out_path)
        panel = {c: df[c].dropna() for c in df.columns if c in df.columns}
    done_codes = set(panel.keys())
    logger.info(f"AKShare 路径: 已下载 {len(done_codes)} 只，待下载 {len(codes) - len(done_codes)} 只")

    # 仅过滤掉已完成的 code
    pending = [c for c in codes if c not in done_codes]
    if not pending:
        logger.info("所有股票已下载，跳过")
        _save_wide(panel, out_path)
        return {"total_mv": out_path}

    s_start = pd.Timestamp(start)
    s_end = pd.Timestamp(end)

    lock = threading.Lock()
    failed: list[tuple[str, str]] = []
    done_count = 0
    success = 0

    def _fetch(code: str):
        s, err = _fetch_one_baidu(code)
        time.sleep(0.05)
        return code, s, err

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, c): c for c in pending}
        for fut in as_completed(futures):
            code, s, err = fut.result()
            with lock:
                done_count += 1
                if err is not None:
                    failed.append((code, str(err)[:80]))
                elif s is None or s.empty:
                    failed.append((code, "empty"))
                else:
                    # 截到 [start, end]
                    s = s[(s.index >= s_start) & (s.index <= s_end)]
                    if not s.empty:
                        panel[code] = s
                        success += 1
                if done_count % 10 == 0 or done_count == len(pending):
                    logger.info(
                        f"AKShare 进度 {done_count}/{len(pending)} "
                        f"成功={success} 失败={len(failed)}"
                    )
                if done_count % save_every == 0:
                    _save_wide(panel, out_path)

    _save_wide(panel, out_path)
    if failed:
        logger.warning(f"AKShare 失败 {len(failed)} 只，前 5: {failed[:5]}")
    logger.info(f"AKShare 完成: total_mv.parquet shape={pd.read_parquet(out_path).shape}")
    return {"total_mv": out_path}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(
    start: str,
    end: str,
    source: str = "akshare",
    sample: int = 0,
    period: str = "近十年",
):
    """【DEPRECATED】日常市值请用 ``download_shares`` + ``compute_market_cap``。"""
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    logger.error(_DEPRECATION_MSG)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== [DEPRECATED] 下载日频市值数据 ({start} → {end}) source={source} ===")
    logger.info(
        "提示：完整日频市值请优先 "
        "`python -m data.download_shares` + `python -m data.compute_market_cap`"
    )

    if source == "tushare":
        logger.warning("Tushare 为遗留分支，非本仓库日常路径")
        paths = download_via_tushare(start, end)
        logger.info(f"Tushare 路径完成: {list(paths.keys())}")
        return

    # 默认 / akshare / auto → AKShare（仅 total_mv）
    logger.warning("=== AKShare baidu 路径（仅 total_mv；circ_mv/turnover 请走 compute_market_cap）===")
    codes = _load_stock_list() if sample == 0 else _load_stock_list()[:sample]
    if sample:
        logger.info(f"调试模式：仅下载前 {sample} 只股票")
    download_via_akshare(start, end, codes=codes, period=period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--source",
        choices=["auto", "tushare", "akshare"],
        default="akshare",
        help="默认 akshare；日常完整市值请用 download_shares+compute_market_cap",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--period", default="近十年",
                        choices=["近一年", "近三年", "近五年", "近十年", "全部"],
                        help="AKShare baidu 估值采样区间（影响数据密度，默认近十年=5日采样）")
    args = parser.parse_args()
    main(args.start, args.end, source=args.source, sample=args.sample)
