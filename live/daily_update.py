"""live/daily_update.py — 实盘日更增量链路（辅助人工选股，非自动交易）

每日增量四步：
  1. 行情增量：复用 data.download.download_ohlcv / download_stock_value_em /
     download_shares / compute_market_cap，只拉最近 N 日 append 到 data/raw 的
     OHLCV / 市值 / 换手等 parquet，保留旧行，启动前 .bak。
  2. 因子面板 append：对模型用到的每个因子，在「热身窗」（最近 ~450 日）上重算，
     只取新增交易日行 concat 到 data/processed/factor_panels/factor_panel_*.parquet；
     指纹策略保留——append 失败回退全量重算并 warning。
  3. 当日中性化：对新截面做 WLS Barra+行业残差（复用 residualize_panel），
     只产出当日 neut 行，不整盘重算。
  4. 用已训模型出分：加载 results/<tag>/models/models_manifest.json 中最近一次
     重训模型，对当日截面 predict，输出 Top-N 候选清单（strict 宇宙 + 涨跌停/停牌 mask）。

入口:
    python -m live.daily_update --model-dir results/lgbm_h5_... --top-n 30

与现有研究口径一致：clean_ret、PIT、tradable strict、barra WLS（√市值）。
不破坏研究回测路径：增量是新入口，不替换现有全量 run.py。
已知限制见 docs/操作手册.md §10。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR, IC_MIN_LISTING_DAYS
from factors.factor_cache import (
    FACTOR_CACHE_DIR,
    _cache_paths,
    _load_meta,
    _load_panel,
    _save_panel,
)

DEFAULT_WARMUP_DAYS = 450
DEFAULT_LOOKBACK_DAYS = 7


# 申万 2021 二级：panel 存 6 位行业码的前 4 位。候选清单给人看时映射成名称。
_SW_L2_NAMES = {
    "1101": "种植业", "1102": "渔业", "1103": "林业Ⅱ", "1104": "饲料",
    "1105": "农产品加工", "1107": "养殖业", "1108": "动物保健Ⅱ", "1109": "农业综合Ⅱ",
    "2202": "化学原料", "2203": "化学制品", "2204": "化学纤维", "2205": "塑料",
    "2206": "橡胶", "2208": "农化制品", "2209": "非金属材料Ⅱ",
    "2303": "冶钢原料", "2304": "普钢", "2305": "特钢Ⅱ",
    "2402": "金属新材料", "2403": "工业金属", "2404": "贵金属", "2405": "小金属",
    "2406": "能源金属",
    "2701": "半导体", "2702": "元件", "2703": "光学光电子", "2704": "其他电子Ⅱ",
    "2705": "消费电子", "2706": "电子化学品Ⅱ",
    "2802": "汽车零部件", "2803": "汽车服务", "2804": "摩托车及其他",
    "2805": "乘用车", "2806": "商用车",
    "3301": "白色家电", "3302": "黑色家电", "3303": "小家电", "3304": "厨卫电器",
    "3305": "照明设备Ⅱ", "3306": "家电零部件Ⅱ", "3307": "其他家电Ⅱ",
    "3404": "食品加工", "3405": "白酒Ⅱ", "3406": "非白酒", "3407": "饮料乳品",
    "3408": "休闲食品", "3409": "调味发酵品Ⅱ",
    "3501": "纺织制造", "3502": "服装家纺", "3503": "饰品",
    "3601": "造纸", "3602": "包装印刷", "3603": "家居用品", "3605": "文娱用品",
    "3701": "化学制药", "3702": "中药Ⅱ", "3703": "生物制品", "3704": "医药商业",
    "3705": "医疗器械", "3706": "医疗服务",
    "4101": "电力", "4103": "燃气Ⅱ",
    "4208": "物流", "4209": "铁路公路", "4210": "航空机场", "4211": "航运港口",
    "4301": "房地产开发", "4303": "房地产服务",
    "4502": "贸易Ⅱ", "4503": "一般零售", "4504": "专业连锁Ⅱ",
    "4506": "互联网电商", "4507": "旅游零售Ⅱ",
    "4606": "体育Ⅱ", "4607": "本地生活服务Ⅱ", "4608": "专业服务",
    "4609": "酒店餐饮", "4610": "旅游及景区", "4611": "教育",
    "4802": "国有大型银行Ⅱ", "4803": "股份制银行Ⅱ", "4804": "城商行Ⅱ",
    "4805": "农商行Ⅱ", "4806": "其他银行Ⅱ",
    "4901": "证券Ⅱ", "4902": "保险Ⅱ", "4903": "多元金融",
    "5101": "综合Ⅱ",
    "6101": "水泥", "6102": "玻璃玻纤", "6103": "装修建材",
    "6201": "房屋建设Ⅱ", "6202": "装修装饰Ⅱ", "6203": "基础建设",
    "6204": "专业工程", "6206": "工程咨询服务Ⅱ",
    "6301": "电机Ⅱ", "6303": "其他电源设备Ⅱ", "6305": "光伏设备",
    "6306": "风电设备", "6307": "电池", "6308": "电网设备",
    "6401": "通用设备", "6402": "专用设备", "6405": "轨交设备Ⅱ",
    "6406": "工程机械", "6407": "自动化设备",
    "6501": "航天装备Ⅱ", "6502": "航空装备Ⅱ", "6503": "地面兵装Ⅱ",
    "6504": "航海装备Ⅱ", "6505": "军工电子Ⅱ",
    "7101": "计算机设备", "7103": "IT服务Ⅱ", "7104": "软件开发",
    "7204": "游戏Ⅱ", "7205": "广告营销", "7206": "影视院线", "7207": "数字媒体",
    "7208": "社交Ⅱ", "7209": "出版", "7210": "电视广播Ⅱ",
    "7301": "通信服务", "7302": "通信设备",
    "7401": "煤炭开采", "7402": "焦炭Ⅱ",
    "7501": "油气开采Ⅱ", "7502": "油服工程", "7503": "炼化及贸易",
    "7601": "环境治理", "7602": "环保设备Ⅱ",
    "7701": "个护用品", "7702": "化妆品", "7703": "医疗美容",
}

CANDIDATE_COLS = [
    "as_of_date", "rank", "code", "name", "sw_l2", "circ_mv", "circ_mv_yi", "score",
]


def _backup_parquet(path: Path) -> None:
    """写前 .bak 备份（与 download_shares 习惯一致）。"""
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
    except OSError as e:
        logger.warning(f".bak 备份失败（继续）: {path.name} -> {e}")


def _log_circ_mv_shape(tag: str) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """记录 circ_mv 行数/首末日，防止 lookback 切片覆盖全历史。"""
    path = RAW_DIR / "circ_mv.parquet"
    if not path.exists():
        logger.warning(f"circ_mv [{tag}]: 文件不存在")
        return 0, None, None
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    n0, n1 = idx.min(), idx.max()
    logger.info(
        f"circ_mv [{tag}]: shape={df.shape}  {n0.date()} → {n1.date()}"
    )
    return int(df.shape[0]), n0, n1


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 行情增量下载
# ══════════════════════════════════════════════════════════════════════════════

def incremental_download(as_of, lookback_days, sample=0):
    """增量拉取最近 lookback_days 日行情/市值/换手，append 到 data/raw。

    复用现有下载函数（本身已是 per-stock date-append 增量）。
    """
    from data.download import download_ohlcv, filter_universe, get_stock_list

    as_of = pd.Timestamp(as_of)
    start = (as_of - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    logger.info(f"[Step 1] 行情增量下载 {start} ~ {end}")

    stock_list = filter_universe(get_stock_list(include_delisted=True))
    codes = list(stock_list["code"].astype(str).str.zfill(6))
    if sample and sample > 0:
        codes = codes[:sample]
        logger.info(f"--sample {sample}: 仅前 {len(codes)} 只")

    for fname in ("close_hfq", "open_hfq", "high_hfq", "low_hfq",
                  "volume", "amount", "prices_raw"):
        _backup_parquet(RAW_DIR / f"{fname}.parquet")

    try:
        download_ohlcv(codes, start=start, end=end, adjust="hfq")
    except Exception as e:
        logger.error(f"OHLCV 增量下载失败: {e}")
        raise

    # 日更原先只拉 hfq，prices_raw 会停在旧末日（2026-07 曾因此停更到 7/29）。
    try:
        from data.download import _last_valid_by_code

        peer_last = _last_valid_by_code(RAW_DIR / "close_hfq.parquet")
        download_ohlcv(
            codes, start=start, end=end, adjust="", peer_last=peer_last,
        )
    except Exception as e:
        logger.warning(f"prices_raw 增量下载失败（自算市值/换手将塌缩）: {e}")

    # 与主下载流程（data.download:807-808）一致：close_hfq → prices_hfq 向后兼容。
    # load_clean_panels 读 prices_hfq.parquet；不补这一步则 hfq 增量不生效。
    try:
        close_hfq_path = RAW_DIR / "close_hfq.parquet"
        prices_hfq_path = RAW_DIR / "prices_hfq.parquet"
        if close_hfq_path.exists():
            _backup_parquet(prices_hfq_path)
            import pandas as _pd
            ch = _pd.read_parquet(close_hfq_path)
            ch.sort_index().to_parquet(prices_hfq_path)
            logger.info(f"[Step 1] close_hfq → prices_hfq 同步: {ch.shape}")
    except Exception as e:
        logger.warning(f"close_hfq → prices_hfq 同步失败: {e}")

    try:
        from data.download_stock_value_em import download_stock_value_em
        # start 是 wide 面板日期下界（默认 2018-01-01），不是 lookback 窗口。
        # 把 as_of-lookback 传进去会在 assemble 时切片后整表覆盖，砍掉历史市值。
        n_before, _, _ = _log_circ_mv_shape("download 前")
        download_stock_value_em(codes, refresh_stale_days=5)
        n_after, _, _ = _log_circ_mv_shape("download 后")
        if n_before > 0 and n_after < n_before:
            logger.error(
                f"circ_mv 行数从 {n_before} 砍到 {n_after}，历史被截断！"
                "assemble 必须 outer merge，禁止用 lookback 当 start 覆盖全表"
            )
    except Exception as e:
        logger.warning(f"stock_value_em 增量下载失败（市值因子将退化）: {e}")

    try:
        from data.download_shares import download_shares
        # start 是股本长表历史下界（默认 2015），不是 lookback。
        # 传入 as_of-lookback 会 _apply_start_filter 砍掉历史变动记录。
        download_shares(codes, start="2015-01-01", refresh_stale_days=30)
    except Exception as e:
        logger.warning(f"download_shares 增量下载失败（换手率将退化）: {e}")

    # ST 时间序列：incremental 原先不刷新，沪市新 ST（如 600530）会漏出 live 宇宙。
    try:
        from data.download_st_history import main as download_st_history
        logger.info("[Step 1] 刷新 ST 历史 st_history.parquet")
        download_st_history()
    except Exception as e:
        logger.warning(f"download_st_history 失败（ST mask 将用旧表）: {e}")

    try:
        from data.compute_market_cap import compute_market_cap
        _backup_parquet(RAW_DIR / "turnover_rate.parquet")
        # 禁止把 lookback 当 start：会把 turnover_rate 整表覆盖成最近 N 日（曾砍成 23 行）。
        compute_market_cap(
            shares_path=RAW_DIR / "share_change.parquet",
            prices_path=RAW_DIR / "prices_raw.parquet",
            volume_path=RAW_DIR / "volume.parquet",
            start="2018-01-01",
            end=None,
            sample_codes=codes if sample else None,
        )
    except Exception as e:
        logger.warning(f"compute_market_cap（turnover_rate）失败: {e}")

    # ── 财务季报（季频；resume by code，已 4 月内新鲜的跳过）──
    try:
        from data.download import download_financial_indicators
        _fin_path = RAW_DIR / "financial_indicators.parquet"
        _backup_parquet(_fin_path)
        download_financial_indicators(codes, start_year=start[:4], out_path=_fin_path)
    except Exception as e:
        logger.warning(f"financial_indicators 增量下载失败: {e}")

    # ── 两融（日频；margin_detail 长表 + margin_balance 宽表，按交易日 resume）──
    try:
        from data.download_margin import main as _margin_main
        for _fn in ("margin_balance.parquet", "margin_detail.parquet"):
            _backup_parquet(RAW_DIR / _fn)
        _margin_main(start, end, sample=0)
    except Exception as e:
        logger.warning(f"margin 增量下载失败（融资因子将退化）: {e}")

    # ── 大单/超大单净流入：已弃用（akshare 资金流数据不足，因子不可用）──
    # 不再增量下载 moneyflow_large/superlarge；因子侧已标记弃用，勿进入 IC/生产池。
    # 详见 factors/factor_alpha.py::load_moneyflow。

    # ── 机构持仓（季频；resume by 季报期）──
    try:
        from data.download_institution import download_institution
        _backup_parquet(RAW_DIR / "institution_holding.parquet")
        download_institution(start_year=int(start[:4]))
    except Exception as e:
        logger.warning(f"institution 增量下载失败: {e}")

    # ── 龙虎榜（日频；month-chunks resume）──
    try:
        from data.download_lhb import main as _lhb_main
        _backup_parquet(RAW_DIR / "lhb_detail.parquet")
        _lhb_main(start, end, sample=0)
    except Exception as e:
        logger.warning(f"lhb 增量下载失败: {e}")

    # ── 大宗交易 + 高管增减持（日频；month-chunks / 全量合并）──
    try:
        from data.download_holder_trade import download_block_trade, download_holder_trade
        for _fn in ("block_trade.parquet", "holder_trade.parquet"):
            _backup_parquet(RAW_DIR / _fn)
        download_block_trade(start, end, sample=0)
        download_holder_trade(start, end)
    except Exception as e:
        logger.warning(f"block_trade/holder_trade 增量下载失败: {e}")

    # ── 解禁（日频；month-chunks resume）──
    try:
        from data.download_lockup import main as _lockup_main
        _backup_parquet(RAW_DIR / "lockup_release.parquet")
        _lockup_main(start, end, sample=0)
    except Exception as e:
        logger.warning(f"lockup 增量下载失败: {e}")

    # ── 回购（事件；全量快照合并去重）──
    try:
        from data.download_repurchase import download_repurchase
        _backup_parquet(RAW_DIR / "repurchase.parquet")
        download_repurchase()
    except Exception as e:
        logger.warning(f"repurchase 增量下载失败: {e}")

    # ── 评级变动（日频；by announce_date resume）──
    try:
        from data.download_rank_forecast import download_rank_forecast
        _backup_parquet(RAW_DIR / "rank_forecast.parquet")
        download_rank_forecast(start=start, end=end)
    except Exception as e:
        logger.warning(f"rank_forecast 增量下载失败: {e}")

    # ── 研报列表（per-stock resume；逐股慢，仅拉未覆盖股，不重刷已有股）──
    try:
        from data.download_research_report import download_research_report
        _backup_parquet(RAW_DIR / "research_report.parquet")
        download_research_report(codes=codes, force=False)
    except Exception as e:
        logger.warning(f"research_report 增量下载失败: {e}")

    # ── 行业资金流（per-sector resume）──
    try:
        from data.download_sector_fund_flow import download_sector_fund_flow
        _backup_parquet(RAW_DIR / "sector_fund_flow.parquet")
        download_sector_fund_flow(use_hist=True)
    except Exception as e:
        logger.warning(f"sector_fund_flow 增量下载失败: {e}")

    # ── 业绩快报/正式稿（季频；resume by report_date）──
    try:
        from data.events.download_yjbb import download_yjbb
        _backup_parquet(RAW_DIR / "yjbb.parquet")
        download_yjbb(start_year=int(start[:4]))
    except Exception as e:
        logger.warning(f"yjbb 增量下载失败: {e}")

    # ── 业绩预告（季频；resume by report_date）──
    try:
        from data.events.download_yjyg import download as _download_yjyg
        _backup_parquet(RAW_DIR / "yjyg.parquet")
        _download_yjyg()
    except Exception as e:
        logger.warning(f"yjyg 增量下载失败: {e}")

    # ── 大宗营业部排行（快照；大宗折价席位质量因子依赖）──
    try:
        from data.download_dzjy_yybph import download_dzjy_yybph
        _backup_parquet(RAW_DIR / "dzjy_yybph.parquet")
        download_dzjy_yybph()
    except Exception as e:
        logger.warning(f"dzjy_yybph 增量下载失败: {e}")

    # ── 股东户数（季频；per-stock resume）──
    try:
        from data.download_shareholder import main as _shareholder_main
        _backup_parquet(RAW_DIR / "shareholder_count.parquet")
        _shareholder_main(start=start, sample=0)
    except Exception as e:
        logger.warning(f"shareholder 增量下载失败: {e}")

    # ── 指数增量更新（沪深300 / 创业板指；缓存末日 < end 则从 last+1 拉到 end）──
    try:
        import akshare as _ak
        _INDEX_MAP = {"沪深300": "000300", "创业板指": "399006"}
        for _nm, _code in _INDEX_MAP.items():
            _cache = RAW_DIR / f"index_{_code}.parquet"
            try:
                if _cache.exists():
                    _s = pd.read_parquet(_cache).squeeze()
                    if isinstance(_s, pd.DataFrame):
                        _s = _s.iloc[:, 0]
                    _last = pd.Timestamp(_s.index.max())
                    if _last >= pd.Timestamp(end):
                        continue
                    _inc_start = (_last + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    _inc_end = pd.Timestamp(end).strftime("%Y%m%d")
                    _df = _ak.index_zh_a_hist(
                        symbol=_code, period="daily",
                        start_date=_inc_start, end_date=_inc_end,
                    )
                    _df["日期"] = pd.to_datetime(_df["日期"])
                    _inc = _df.set_index("日期")["收盘"].rename(_nm)
                    _merged = pd.concat([_s, _inc])
                    _merged = _merged[~_merged.index.duplicated(keep="last")].sort_index()
                    _merged.name = _nm
                    _backup_parquet(_cache)
                    _merged.to_frame().to_parquet(_cache)
                    logger.info(f"指数 {_nm}({_code}) 增量 +{len(_inc)} 行 → 末日 {_merged.index.max().date()}")
                else:
                    _df = _ak.index_zh_a_hist(
                        symbol=_code, period="daily",
                        start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                    )
                    _df["日期"] = pd.to_datetime(_df["日期"])
                    _s = _df.set_index("日期")["收盘"].rename(_nm)
                    _s.to_frame().to_parquet(_cache)
                    logger.info(f"指数 {_nm}({_code}) 首次下载 shape={_s.shape}")
            except Exception as e:
                logger.warning(f"指数 {_nm}({_code}) 增量更新失败: {e}，沿用旧缓存")
    except Exception as e:
        logger.warning(f"指数增量更新整体失败: {e}")

    # ── 中证全指（csi_all；复用 strategies.market_state.download_csi_all，force 全量快照）──
    try:
        from strategies.market_state import download_csi_all
        _backup_parquet(RAW_DIR / "csi_all.parquet")
        download_csi_all(force=True)
    except Exception as e:
        logger.warning(f"csi_all 增量更新失败: {e}，沿用旧缓存")

    logger.info("[Step 1] 行情增量下载完成")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 加载 + 清洗 raw 面板（与 run.py._load_data 同口径，简化版）
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_sparse_prices_raw(prices_raw, min_frac: float = 0.80):
    """截掉新浪补数中的稀疏尾（索引已到末日、但只有先补到的代码段有值）。

    取最后一个覆盖率 >= min_frac 的交易日；Size 仍走完整 circ_mv，hfq 不截。
    """
    if prices_raw is None or not isinstance(prices_raw, pd.DataFrame) or prices_raw.empty:
        return prices_raw
    cov = prices_raw.notna().sum(axis=1)
    n_col = max(int(prices_raw.shape[1]), 1)
    ok = cov[cov >= min_frac * n_col]
    if ok.empty:
        cut = cov.idxmax()
        logger.warning(
            f"prices_raw 无 >= {min_frac:.0%} 覆盖日，退到最大覆盖 "
            f"{pd.Timestamp(cut).date()} n={int(cov.loc[cut])}/{n_col}"
        )
    else:
        cut = ok.index.max()
    cut = pd.Timestamp(cut)
    tail = pd.Timestamp(prices_raw.index.max())
    if tail > cut:
        logger.warning(
            f"prices_raw 截到最后高覆盖日 {cut.date()} "
            f"n={int(cov.loc[cut])}/{n_col}；丢弃稀疏尾 {tail.date()} "
            f"n={int(cov.loc[tail])}/{n_col}（避免 600xxx 先补、其余 NaN 的截面偏）"
        )
        return prices_raw.loc[prices_raw.index <= cut]
    logger.info(
        f"prices_raw 末日 {tail.date()} 覆盖 {int(cov.loc[tail])}/{n_col}，无需截尾"
    )
    return prices_raw


def _load_opt(fname):
    p = RAW_DIR / fname
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_clean_panels(sample=0):
    """加载并清洗 raw 面板，返回与 factors.factor registry 同口径的 kwargs dict。

    与 run.py::_load_data 同口径：clean_ohlc_aligned / clean_ohlcv / clean_* /
    mask_post_delist / 退市股保留。北向停更不加载。
    sample>0 时截取前 N 只股票（冒烟，与 run.py --sample 同口径）。
    """
    from data.clean import (
        clean_prices, clean_financial, clean_ohlcv, clean_ohlc_aligned,
        clean_volume, clean_amount, clean_aux_panel, clean_market_cap,
        mask_post_delist, validate_amount_units,
    )

    logger.info("[Step 2a] 加载 + 清洗 raw 面板")
    _close_raw = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    _open_raw = _load_opt("open_hfq.parquet")
    _high_raw = _load_opt("high_hfq.parquet")
    _low_raw = _load_opt("low_hfq.parquet")
    prices, open_, high, low = clean_ohlc_aligned(_close_raw, _open_raw, _high_raw, _low_raw)

    prices_raw_path = RAW_DIR / "prices_raw.parquet"
    snap = os.environ.get("LIVE_PRICES_RAW_PATH")
    if snap:
        snap_p = Path(snap)
        if snap_p.exists():
            prices_raw_path = snap_p
            logger.info(f"prices_raw 改读快照 {prices_raw_path}")
    prices_raw = (
        clean_prices(pd.read_parquet(prices_raw_path), label="prices_raw")
        if prices_raw_path.exists() else None
    )
    prices_raw = _truncate_sparse_prices_raw(prices_raw)
    fin_path = RAW_DIR / "financial_indicators.parquet"
    financial = clean_financial(pd.read_parquet(fin_path)) if fin_path.exists() else None

    _vol_raw = _load_opt("volume.parquet")
    volume = clean_volume(_vol_raw, name="volume") if _vol_raw is not None else None
    _amt_raw = _load_opt("amount.parquet")
    amount = clean_amount(_amt_raw, name="amount") if _amt_raw is not None else None
    if amount is not None and volume is not None:
        try:
            validate_amount_units(amount, volume, prices)
        except Exception as e:
            logger.warning(f"amount 量纲校验失败（继续）: {e}")

    try:
        from research.ic.universe import load_delist_dates
        _delist = load_delist_dates()
    except Exception:
        _delist = None
    if _delist:
        prices = mask_post_delist(prices, _delist)
        if prices_raw is not None:
            prices_raw = mask_post_delist(prices_raw, _delist)
        open_ = mask_post_delist(open_, _delist)
        high = mask_post_delist(high, _delist)
        low = mask_post_delist(low, _delist)
        if volume is not None:
            volume = mask_post_delist(volume, _delist)
        if amount is not None:
            amount = mask_post_delist(amount, _delist)

    _margin_raw = _load_opt("margin_balance.parquet")
    margin = clean_aux_panel(_margin_raw, name="margin") if _margin_raw is not None else None
    # moneyflow 已弃用：akshare 东财大单全市场拉不稳、单票历史常仅近数月，
    # 因子不可用。不再加载 moneyflow_large，强制 None 以跳过相关因子计算。
    if (_load_opt("moneyflow_large.parquet") is not None):
        logger.warning("moneyflow_large 已弃用（akshare 资金流数据不足），跳过加载；大单净流入/残差因子不计算。")
    moneyflow = None
    institution = _load_opt("institution_holding.parquet")

    from data.mv_panels import load_mv_raw
    _total_mv_raw = load_mv_raw("total_mv")
    total_mv = clean_market_cap(_total_mv_raw, name="total_mv") if _total_mv_raw is not None else None
    _circ_mv_raw = load_mv_raw("circ_mv")
    circ_mv = clean_market_cap(_circ_mv_raw, name="circ_mv") if _circ_mv_raw is not None else None
    _turnover_raw = _load_opt("turnover_rate.parquet")
    turnover_rate = clean_aux_panel(_turnover_raw, name="turnover_rate") if _turnover_raw is not None else None

    market_prices = _load_opt("csi_all.parquet")
    if market_prices is None:
        market_prices = _load_opt("csi300.parquet")
    industry_map = _load_opt("industry_map.parquet")

    logger.info("[Step 2b] 涨跌停清洗（生成 clean_ret）")
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    from data.download import drop_excluded_universe_columns

    prices = drop_excluded_universe_columns(prices, name="prices")
    prices_raw = drop_excluded_universe_columns(prices_raw)
    open_ = drop_excluded_universe_columns(open_)
    high = drop_excluded_universe_columns(high)
    low = drop_excluded_universe_columns(low)
    volume = drop_excluded_universe_columns(volume)
    amount = drop_excluded_universe_columns(amount)
    clean_ret = drop_excluded_universe_columns(clean_ret)
    margin = drop_excluded_universe_columns(margin)
    institution = drop_excluded_universe_columns(institution)
    total_mv = drop_excluded_universe_columns(total_mv)
    circ_mv = drop_excluded_universe_columns(circ_mv)
    turnover_rate = drop_excluded_universe_columns(turnover_rate)
    if masks:
        masks = {
            k: (drop_excluded_universe_columns(v) if isinstance(v, pd.DataFrame) else v)
            for k, v in masks.items()
        }

    panels = dict(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=None, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv, turnover_rate=turnover_rate,
    )
    if sample and sample > 0:
        panels = _apply_sample(panels, sample)
    return panels


def _apply_sample(panels, sample):
    """截取前 N 只股票（与 run.py --sample 同口径，冒烟用）。"""
    prices = panels["prices"]
    codes = list(prices.columns[:sample])
    def _col(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.columns.isin(codes).any():
            keep = [c for c in codes if c in df.columns]
            return df.loc[:, keep]
        return df
    out = dict(panels)
    for k in ("prices", "prices_raw", "open_", "high", "low", "volume",
              "amount", "clean_ret", "margin", "moneyflow", "institution",
              "total_mv", "circ_mv", "turnover_rate"):
        if k in out:
            out[k] = _col(out[k])
    if out.get("masks"):
        out["masks"] = {mk: _col(mv) for mk, mv in out["masks"].items()}
    logger.info(f"--sample {sample}: 截取 {len(codes)} 只股票")
    return out


def slice_warmup(panels, as_of, warmup_days):
    """把所有面板切到 [as_of - warmup_days, as_of] 热身窗，供增量因子计算。

    截面列（股票）保持全量——winsorize/zscore 是 per-date 的，需保留当日全截面。
    长表（financial/industry）不按日期截。
    """
    prices = panels["prices"]
    end = pd.Timestamp(as_of)
    if end not in prices.index:
        le = prices.index[prices.index <= end]
        if len(le) == 0:
            raise ValueError(f"as_of={as_of} 早于数据首日 {prices.index[0]}")
        end = le[-1]
    start = end - pd.Timedelta(days=warmup_days)
    idx = prices.index[(prices.index >= start) & (prices.index <= end)]
    if len(idx) < 20:
        raise ValueError(
            f"热身窗太短：{len(idx)} 日（warmup_days={warmup_days}, as_of={as_of}）"
        )

    def _sl(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.index.equals(prices.index):
            return df.loc[idx]
        return df

    out = {}
    for k, v in panels.items():
        if k == "masks" and v is not None:
            out[k] = {mk: _sl(mv) for mk, mv in v.items()}
        else:
            out[k] = _sl(v)
    out["_warmup_idx"] = idx
    out["_as_of"] = end
    logger.info(f"热身窗: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)} 日)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 因子面板 append（热身窗重算 → 取新行 concat）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_factors_warmup(names, warmup_panels):
    """在热身窗输入上重算指定因子（绕过磁盘缓存，避免签名失配）。

    直接调 _iter_factor_registry_raw（无缓存计算核），与 compute_single_factor
    同口径。返回 {name: panel}，panel 已 reindex 到热身窗 prices.index。
    """
    from factors.factor import _iter_factor_registry_raw, _filter_none_emit

    kwargs = dict(
        prices=warmup_panels["prices"], financial=warmup_panels["financial"],
        prices_raw=warmup_panels["prices_raw"], volume=warmup_panels["volume"],
        amount=warmup_panels["amount"], open_=warmup_panels["open_"],
        high=warmup_panels["high"], low=warmup_panels["low"],
        clean_ret=warmup_panels["clean_ret"], masks=warmup_panels["masks"],
        market_prices=warmup_panels["market_prices"],
        industry_map=warmup_panels["industry_map"],
        margin=warmup_panels["margin"], moneyflow=warmup_panels["moneyflow"],
        northbound=warmup_panels["northbound"],
        institution=warmup_panels["institution"],
        circ_mv=warmup_panels["circ_mv"], total_mv=warmup_panels["total_mv"],
        factor_names=set(names), include_regime=False,
    )
    out = {}
    for n, panel in _filter_none_emit(_iter_factor_registry_raw(**kwargs)):
        out[n] = panel
    missing = [n for n in names if n not in out]
    if missing:
        logger.warning(
            f"热身窗重算缺 {len(missing)} 个因子: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return out


def append_factor_panels(names, warmup_panels, as_of):
    """对每个因子：加载已有 factor_panel_<hash>.parquet，热身窗重算，取新行 concat。

    - 优先 append：已有面板 + 新行（index > 已有末日）concat，dedup keep='last'，sort
    - 失败回退：全量重算（get_factor_registry 全输入）并 warning
    - 写前 .bak；写后更新 .meta.json（标记 live_appended）
    - 返回 {name: full_panel}（含历史 + 新行），供中性化 / 打分用

    注：append 后 .meta.json 的输入指纹仍是旧值，下次全量研究管线会因签名
    失配自动重算——这是预期行为（append 是捷径，不替代全量校准）。
    """
    as_of = pd.Timestamp(as_of)
    fresh = _compute_factors_warmup(names, warmup_panels)
    result = {}

    for name in names:
        if name not in fresh:
            continue
        fresh_panel = fresh[name]
        pq_path, meta_path = _cache_paths(name)

        if pq_path.exists():
            try:
                existing = _load_panel(pq_path)
                last_existing = existing.index[-1]
                new_rows = fresh_panel.loc[fresh_panel.index > last_existing]
                if new_rows.empty:
                    # 已有面板含 as_of（上一份坏 Size / 稀疏 raw 可能写过）→ 用热身窗覆盖当日
                    if as_of in existing.index and as_of in fresh_panel.index:
                        existing = existing.copy()
                        common = existing.columns.intersection(fresh_panel.columns)
                        existing.loc[as_of, common] = fresh_panel.loc[as_of, common]
                        extra = fresh_panel.columns.difference(existing.columns)
                        if len(extra):
                            existing = existing.reindex(
                                columns=existing.columns.union(extra)
                            )
                            existing.loc[as_of, extra] = fresh_panel.loc[as_of, extra]
                        logger.info(
                            f"[append] {name}: 已有至 {last_existing.date()}，"
                            f"热身窗覆盖 {as_of.date()}（内存，不重写磁盘）"
                        )
                    else:
                        logger.info(f"[append] {name}: 无新行（已有至 {last_existing.date()}）")
                    result[name] = existing
                    continue
                # 列对齐：union(existing, new)，缺失填 NaN
                all_cols = existing.columns.union(new_rows.columns)
                existing_a = existing.reindex(columns=all_cols)
                new_a = new_rows.reindex(columns=all_cols)
                merged = pd.concat([existing_a, new_a])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged = merged.astype(np.float32, copy=False)
                _backup_parquet(pq_path)
                _save_panel_live(name, merged, meta_path, as_of, appended=len(new_rows))
                result[name] = merged
                logger.info(
                    f"[append] {name}: +{len(new_rows)} 行 "
                    f"({last_existing.date()} -> {merged.index[-1].date()})"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"[append] {name}: append 失败 ({e})，回退全量重算"
                )
        # 无已有面板 或 append 失败 → 用热身窗结果作为该因子面板（冷启动 / 兜底）
        _backup_parquet(pq_path)
        _save_panel_live(name, fresh_panel, meta_path, as_of, appended=len(fresh_panel))
        result[name] = fresh_panel
        logger.info(
            f"[append] {name}: 冷启动/兜底，写热身窗 {len(fresh_panel)} 行"
        )
    return result


def _save_panel_live(name, panel, meta_path, as_of, appended=0):
    """落盘因子面板 + 更新 .meta.json（标记 live_appended，保留旧指纹）。"""
    from factors.factor_cache import _cache_paths, _save_panel
    pq_path, _ = _cache_paths(name)
    # 复用 _save_panel 写 parquet（原子 tmp+rename），但 meta 用旧指纹 + live 标记
    old_meta = _load_meta(meta_path)
    sig = old_meta.get("signature") if old_meta else None
    # 直接调 _save_panel 会用 sig 重建 meta；若 sig=None 则写空指纹 meta
    if sig is not None:
        _save_panel(name, panel, sig)
    else:
        # 冷启动：写一个最小 meta（无指纹，下次研究管线会重算）
        panel.astype(np.float32, copy=False).to_parquet(pq_path)
    # 追加 live 标记
    meta = _load_meta(meta_path) or {}
    meta["live_appended"] = True
    meta["live_last_as_of"] = pd.Timestamp(as_of).isoformat()
    meta["live_last_appended_rows"] = int(appended)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 当日中性化（WLS Barra+行业残差，只产当日 neut 行）
# ══════════════════════════════════════════════════════════════════════════════

def compute_barra_warmup(warmup_panels):
    """在热身窗上算 Barra 9 风格因子 + WLS 权重面板（√市值）。

    复用 factors.barra_risk.get_barra_factors / barra_regression_weights。
    返回 (barra_factors: dict, neut_weights: DataFrame|None)。
    """
    from factors.barra_risk import get_barra_factors, barra_regression_weights

    prices = warmup_panels["prices"]
    ind_map_arg = None
    if warmup_panels["industry_map"] is not None:
        im = warmup_panels["industry_map"]
        if isinstance(im, pd.DataFrame) and "sw_l2" in im.columns:
            ind_map_arg = im["sw_l2"]
        else:
            ind_map_arg = im
    barra = get_barra_factors(
        prices=prices,
        financial=warmup_panels["financial"],
        market_prices=warmup_panels["market_prices"],
        volume=warmup_panels["volume"],
        clean_ret=warmup_panels["clean_ret"],
        industry_map=ind_map_arg,
        prices_raw=warmup_panels["prices_raw"],
        circ_mv=warmup_panels["circ_mv"],
        total_mv=warmup_panels["total_mv"],
        turnover_rate=warmup_panels["turnover_rate"],
        amount=warmup_panels["amount"],
    )
    weights = barra_regression_weights(
        prices, circ_mv=warmup_panels["circ_mv"], total_mv=warmup_panels["total_mv"],
    )
    logger.info(f"[Step 4] Barra 因子就绪: {len(barra)} 个；WLS 权重={'有' if weights is not None else '无(等权OLS)'}")
    size = barra.get("Barra_Size")
    asof = warmup_panels.get("_as_of")
    if size is not None and asof is not None:
        asof = pd.Timestamp(asof)
        if asof in size.index:
            n_sz = int(pd.to_numeric(size.loc[asof], errors="coerce").notna().sum())
            n_fill0 = int((pd.to_numeric(size.loc[asof], errors="coerce").fillna(0) == 0).sum())
            logger.info(
                f"Barra_Size 当日 {asof.date()} 非空={n_sz}/{size.shape[1]} "
                f"（零值/fillna候选={n_fill0}；大量 0 说明 Size 塌缩）"
            )
            if n_sz < 1000:
                logger.error(
                    f"Barra_Size 当日覆盖过低 {n_sz}，中性化会 fillna(0) 退化"
                )
        else:
            logger.error(f"Barra_Size 无当日 {asof.date()}，中性化会 fillna(0)")
    elif size is None:
        logger.error("Barra_Size 未算出，size_industry 中性化会失败/fillna(0)")
    return barra, weights


def neutralize_as_of(factor_panels, barra, weights, industry_map, as_of, universe=None,
                     prices=None, neut_controls="barra"):
    """对 as_of 截面做 WLS 残差，返回 {name: neut_row_series}。

    - 复用 models.wf.labels.residualize_panel（与训练同口径）
    - rebalance_dates=[as_of]：只算当日一行
    - should_skip_neutralize 的因子（Barra_*/special）原样返回 raw 行
    - 输出每项是 pd.Series(index=universe)（当日截面残差 + re-zscore）
    - universe: 统一股票宇宙（prices.columns）；各因子面板列集可能不同
      （如融资因子仅覆盖两融标的），统一 reindex 到 universe 避免空 DataFrame。
    - prices: 用于 live neut 缓存宇宙指纹；None 时退化为 universe 构造指纹
      （缓存命中率可能略降，但不影响正确性）。
    - neut_controls: ``barra``（9 风格+行业）或 ``size_industry``（Size+行业）。
      须与训练 manifest 一致；写入 live_neut 缓存键。
    - live neut 缓存：键 = name + live_neut_v1 + as_of + 宇宙指纹 + ctrl_sig；
      默认 barra 不加 nc:（与旧 live 文件兼容），size_industry 才加 ``|nc:size_industry``。
      命中直接用，未命中算并落盘（data/processed/factor_panels/live_neut_<hash>.parquet）。
    """
    from models.wf.labels import (
        NEUT_CONTROLS_SIZE,
        NEUT_CONTROLS_SIZE_INDUSTRY,
        residualize_panel,
        select_neut_control_factors,
        normalize_neut_controls,
    )
    from factors.special_factors import should_skip_neutralize
    from research.rolling_pool.neut_cache import (
        barra_bundle_sig, live_neut_cache_path,
        try_load_live_neut, save_live_neut,
    )

    as_of = pd.Timestamp(as_of)
    neut_mode = normalize_neut_controls(neut_controls)
    barra = select_neut_control_factors(barra, neut_mode)
    if neut_mode == NEUT_CONTROLS_SIZE_INDUSTRY:
        logger.info(
            "feature_neutralize size_industry: Size+PIT行业 WLS，未用 9 风格"
        )
    elif neut_mode == NEUT_CONTROLS_SIZE:
        logger.info("feature_neutralize size: 仅 Size WLS，无行业")
    if universe is None:
        # 取所有面板列的并集
        cols = []
        for p in factor_panels.values():
            if p is not None:
                cols.extend(list(p.columns))
        universe = pd.Index(sorted(set(cols)))
    universe = universe.astype(str).str.zfill(6)

    # live neut 缓存控制变量指纹（Barra + 行业 + WLS 权重）
    ind_series = None
    if industry_map is not None:
        if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns:
            ind_series = industry_map["sw_l2"]
        else:
            ind_series = industry_map
    industry_panel = None
    if neut_mode != NEUT_CONTROLS_SIZE:
        try:
            from research.ic.load_data import load_industry_panel
            industry_panel = load_industry_panel(required=False)
        except Exception as e:
            logger.warning(f"live neutralize: 加载 industry_map_panel 失败: {e}")
    ctrl_sig = barra_bundle_sig(
        barra, industry_map=ind_series, weight_panel=weights,
        industry_panel=industry_panel,
    )
    # 宇宙指纹用 prices（若提供）以与训练 neut 缓存口径一致
    prices_for_sig = prices if prices is not None else _universe_to_prices_sig(universe)

    def _zscore_keep_nan(series):
        """截面 zscore 但保留 NaN（不 dropna），与 cross_sectional_zscore 在
        非全 NaN 时等价；全 NaN 时返回全 NaN（index 不丢）。"""
        s = series.astype(np.float64)
        finite = np.isfinite(s.values)
        if finite.sum() < 2:
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        mu = s.values[finite].mean()
        sd = s.values[finite].std()
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        z = np.clip((s.values - mu) / sd, -3.0, 3.0)
        z[~finite] = np.nan
        return pd.Series(z, index=s.index, dtype=np.float32)

    out = {}
    n_skip = 0
    n_neut = 0
    n_allnan = 0
    n_cache_hit = 0
    for name, panel in factor_panels.items():
        if should_skip_neutralize(name):
            if as_of in panel.index:
                out[name] = panel.loc[as_of].reindex(universe)
            else:
                out[name] = pd.Series(np.nan, index=universe, dtype=np.float32)
            n_skip += 1
            continue
        # 先查 live neut 缓存
        cache_path = live_neut_cache_path(
            name, as_of, prices_for_sig, ctrl_sig=ctrl_sig,
            neut_controls=neut_mode,
        )
        cached = try_load_live_neut(
            cache_path, as_of=as_of, universe=universe, name=name,
        )
        if cached is not None:
            out[name] = cached
            n_cache_hit += 1
            n_neut += 1
            continue
        resid = residualize_panel(
            panel, barra, ind_series, pd.DatetimeIndex([as_of]),
            weight_panel=weights,
            industry_panel=industry_panel,
        )
        if as_of in resid.index:
            row = resid.loc[as_of]
        else:
            row = pd.Series(np.nan, index=panel.columns, dtype=np.float32)
        row = row.reindex(universe)
        if not np.isfinite(row.values).any():
            # 残差全 NaN（min_stocks 未达 / 数据未更新）→ 退化用 raw 因子行
            if as_of in panel.index:
                row = panel.loc[as_of].reindex(universe)
            n_allnan += 1
        zrow = _zscore_keep_nan(row)
        out[name] = zrow
        # 落盘 live neut 缓存（单日行）
        save_live_neut(cache_path, zrow, as_of=as_of, name=name)
        n_neut += 1
    logger.info(
        f"[Step 4] 当日中性化: {n_neut} 残差 + {n_skip} 豁免"
        + (f"（{n_allnan} 个残差全 NaN 退化为 raw）" if n_allnan else "")
        + (f"（{n_cache_hit} 个 live neut 缓存命中）" if n_cache_hit else "")
    )
    return out


def _universe_to_prices_sig(universe: pd.Index) -> pd.DataFrame:
    """用 universe 构造一个最小 prices DataFrame 供 universe_sig 取指纹。

    live neut 缓存键需要 universe_sig(prices)；当调用方未传 prices 时用此兜底，
    生成一个 1×N 的 DataFrame（首尾列名与 universe 一致即可稳定指纹）。
    """
    cols = universe.astype(str).str.zfill(6)
    if len(cols) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {c: [0.0] for c in cols},
        index=pd.DatetimeIndex([pd.Timestamp("2000-01-01")]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 加载已训模型 + predict + strict 宇宙 + Top-N 输出
# ══════════════════════════════════════════════════════════════════════════════

def load_latest_models(model_dir, prefer_model=None, prefer_window=None):
    """读 results/<tag>/models/models_manifest.json，返回最近重训日的模型 entries。

    - 按 date 降序取最近重训日（同一日可能有多个 window×model）
    - prefer_model/prefer_window: 优先选指定模型/窗口（None=全部）
    - 返回 (list[dict], manifest_feature_neutralize, manifest_neut_controls)
      - manifest_feature_neutralize: manifest 中记录的训练 feature_neutralize；
        缺失键时返回 None（调用方按 False 默认 + warning 处理）
      - manifest_neut_controls: ``barra`` | ``size_industry``；缺失键时默认
        ``barra`` 并 warning
    - 需要 --save-models 产物；无 manifest 报错并提示
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "models" / "models_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到模型 manifest: {manifest_path}\n"
            f"请用 --save-models 重训，或指定正确的 --model-dir（指向 results/<tag>/）"
        )
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"manifest 为空: {manifest_path}")
    # 提取训练时记录的 feature_neutralize（每条 entry 都应含；取首条即可）
    manifest_fn = entries[0].get("feature_neutralize")
    if manifest_fn is None:
        logger.warning(
            "manifest 无 feature_neutralize 字段（旧训练产物），按 False 默认处理；"
            "建议用更新后的 trainer 重训以写入该字段"
        )
    from models.wf.labels import normalize_neut_controls
    manifest_nc = normalize_neut_controls(
        entries[0].get("neut_controls"), missing_warn=True,
    )
    # 过滤
    cand = entries
    if prefer_model:
        cand = [e for e in cand if e.get("model") == prefer_model]
    if prefer_window:
        cand = [e for e in cand if int(e.get("window", -1)) == int(prefer_window)]
    if not cand:
        raise ValueError(
            f"manifest 无匹配模型 (prefer_model={prefer_model}, "
            f"prefer_window={prefer_window})"
        )
    # 最近重训日
    latest_date = max(e["date"] for e in cand)
    batch = [e for e in cand if e["date"] == latest_date]
    logger.info(
        f"[Step 5] 加载模型: {len(batch)} 个 fold（重训日 {latest_date}，"
        f"models={[e['model'] + '_w' + str(e['window']) for e in batch]}，"
        f"feature_neutralize={manifest_fn}, neut_controls={manifest_nc})"
    )
    return batch, manifest_fn, manifest_nc


def build_feature_matrix(feature_names, neut_rows, raw_panels, as_of, universe=None):
    """组装当日特征矩阵 X (n_stocks × n_features)。

    - feature_names: 模型训练时的特征列序（来自 manifest）
    - neut_rows: {name: Series} 当日中性化后的截面（对非豁免因子）
    - raw_panels: {name: DataFrame} 全量因子面板（对豁免因子取当日 raw 行）
    - as_of: 当日
    - universe: 统一股票宇宙；None 时取 neut_rows/raw_panels 列并集
    返回 (X: DataFrame[index=stock, columns=feature_names], stock_index)

    保证 X 非空（至少 universe 行数），NaN 保留（树模型原生支持 NaN 分裂）。
    """
    as_of = pd.Timestamp(as_of)
    if universe is None:
        cols = []
        for s in neut_rows.values():
            if s is not None:
                cols.extend(list(s.index))
        for p in raw_panels.values():
            if p is not None:
                cols.extend(list(p.columns))
        universe = pd.Index(sorted(set(cols))) if cols else pd.Index(["__empty__"])
    universe = universe.astype(str).str.zfill(6)

    cols = {}
    for fn in feature_names:
        if fn in neut_rows:
            cols[fn] = neut_rows[fn].reindex(universe)
        elif fn in raw_panels:
            panel = raw_panels[fn]
            if as_of in panel.index:
                cols[fn] = panel.loc[as_of].reindex(universe)
            else:
                cols[fn] = pd.Series(np.nan, index=universe, dtype=np.float32)
        else:
            logger.warning(f"特征 {fn} 缺面板/neut 行 → 整列 NaN")
            cols[fn] = pd.Series(np.nan, index=universe, dtype=np.float32)
    X = pd.DataFrame(cols, index=universe)
    X.index = X.index.astype(str).str.zfill(6)
    return X


def predict_scores(model_entries, X):
    """对当日截面 X 出分，多 fold IC 加权 Z-score 平均（与训练 ensemble 同口径）。

    返回 pd.Series(index=stock, values=score)。单模型直接 predict。
    """
    import joblib
    from models.wf.models import predict_model

    # 对齐 X 列序到 manifest feature_names
    feature_names = model_entries[0].get("feature_names")
    if feature_names:
        X = X.reindex(columns=feature_names)
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    # inf -> NaN -> 0（与训练 get_cross_section 同口径 fillna(0)）
    X_np = np.where(np.isfinite(X_np), X_np, 0.0)

    fold_scores = []
    fold_ics = []
    for e in model_entries:
        model = joblib.load(e["path"])
        pred = predict_model(model, X_np, e["model"])
        s = pd.Series(pred, index=X.index, dtype=np.float32)
        fold_scores.append(s)
        # IC 权重：训练时记录的 fold IC（若有）；无则等权
        ic = e.get("fold_ic") or e.get("pred_ic")
        fold_ics.append(ic if ic is not None and np.isfinite(ic) else 0.0)

    if len(fold_scores) == 1:
        return fold_scores[0]
    # IC 加权 Z-score 平均（与 ensemble.combine_model_scores 同语义）
    zscores = []
    weights = []
    for s, ic in zip(fold_scores, fold_ics):
        std = s.std()
        if std == 0 or not np.isfinite(std):
            continue
        zscores.append((s - s.mean()) / std)
        weights.append(max(ic, 0.0))  # 负 IC 折半/丢弃与训练一致：这里取 max(0)
    if not zscores:
        return fold_scores[0]
    w = np.array(weights, dtype=np.float64)
    if w.sum() == 0:
        w = np.ones(len(zscores))
    w = w / w.sum()
    zarr = np.zeros_like(zscores[0].values, dtype=np.float64)
    for zi, wi in zip(zscores, w):
        zarr += wi * zi.values
    return pd.Series(zarr, index=X.index, dtype=np.float32)


def strict_universe_mask(prices, volume, masks, as_of):
    """当日 strict 可买宇宙（与 mask_scores_for_backtest strict 同口径）。

    信号日剔涨跌停 + ST + 停牌 + 次新 + 退市。返回 bool Series(index=stock)。
    """
    from research.ic.universe import (
        build_ic_tradability_mask, load_stock_names, load_is_st_current,
        load_listing_dates, load_delist_dates, load_st_history,
    )

    as_of = pd.Timestamp(as_of)
    # 取单日面板（build_ic_tradability_mask 需 DataFrame）
    if as_of in prices.index:
        p1 = prices.loc[[as_of]]
        v1 = volume.loc[[as_of]] if volume is not None and as_of in volume.index else None
    else:
        le = prices.index[prices.index <= as_of]
        if len(le) == 0:
            return pd.Series(False, index=prices.columns)
        d = le[-1]
        p1 = prices.loc[[d]]
        v1 = volume.loc[[d]] if volume is not None and d in volume.index else None
    m1 = None
    if masks is not None:
        m1 = {k: (v.loc[[as_of]] if as_of in v.index else v.iloc[0:0])
              for k, v in masks.items() if v is not None}

    sn = load_stock_names()
    ist = load_is_st_current()
    ld = load_listing_dates()
    dd = load_delist_dates()
    sth = load_st_history()
    tradable = build_ic_tradability_mask(
        p1, volume=v1, masks=m1,
        stock_names=sn, listing_dates=ld, delist_dates=dd,
        is_st_current=ist, st_history=sth,
        exclude_limit_on_signal=True,  # strict
        min_listing_days=IC_MIN_LISTING_DAYS,
    )
    row = tradable.iloc[0] if len(tradable) else pd.Series(False, index=prices.columns)
    return row.reindex(prices.columns).fillna(False)


def _pretty_sw_l2(val):
    """4 位申万二级代码 → 中文名；未知则保留原值。"""
    if val is None:
        return val
    try:
        if pd.isna(val):
            return val
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return np.nan
    return _SW_L2_NAMES.get(s[:4], s)


def _sw_l2_as_of(codes, as_of, sw_l2=None):
    """PIT 申万二级（as_of 当日或 asof≤）。无 panel 时退化静态 map 并 warning。"""
    codes = pd.Index(codes).astype(str).str.zfill(6)
    if sw_l2 is not None:
        s = sw_l2.copy()
        s.index = s.index.astype(str).str.zfill(6)
        return s.reindex(codes)

    from data.industry.download_industry import (
        PANEL_PATH, OUT_PATH, load_industry_panel, load_industry_as_of,
        load_industry_map,
    )
    if PANEL_PATH.exists():
        panel = load_industry_panel()
        if panel is not None and not panel.empty:
            s = load_industry_as_of(panel, as_of, level="sw_l2")
            s.index = s.index.astype(str).str.zfill(6)
            return s.reindex(codes)
        logger.warning("industry_map_panel.parquet 为空，退化静态 industry_map.parquet")
    else:
        logger.warning(
            "industry_map_panel.parquet 缺失，退化静态 industry_map.parquet（无 PIT）"
        )
    if OUT_PATH.exists():
        imap = load_industry_map()
        imap.index = imap.index.astype(str).str.zfill(6)
        col = "sw_l2" if "sw_l2" in imap.columns else imap.columns[0]
        return imap[col].reindex(codes)
    return pd.Series(index=codes, dtype=object)


def _circ_mv_as_of(codes, as_of, circ_mv=None):
    """东财流通市值（元）：≤as_of 最后有效值（按股票 ffill）。

    增量日若只拼出稀疏新行（多数 NaN），取末日行会整列空；
    须沿时间前向填充后再取 asof。
    """
    codes = pd.Index(codes).astype(str).str.zfill(6)
    if circ_mv is None:
        path = RAW_DIR / "circ_mv.parquet"
        if not path.exists():
            logger.warning("circ_mv.parquet 缺失，市值列为空")
            return pd.Series(index=codes, dtype=float)
        circ_mv = pd.read_parquet(path)
    if circ_mv is None or circ_mv.empty:
        return pd.Series(index=codes, dtype=float)
    df = circ_mv.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df.columns = df.columns.astype(str).str.zfill(6)
    le = df.index[df.index <= pd.Timestamp(as_of)]
    if len(le) == 0:
        logger.warning(f"circ_mv 无 <= {pd.Timestamp(as_of).date()} 的日期")
        return pd.Series(index=codes, dtype=float)
    sub = df.loc[le].sort_index()
    row = sub.ffill().iloc[-1]
    d = sub.index[-1]
    n_ok = int(pd.to_numeric(row.reindex(codes), errors="coerce").notna().sum())
    logger.info(
        f"circ_mv asof {d.date()}（请求 {pd.Timestamp(as_of).date()}）"
        f" 候选命中 {n_ok}/{len(codes)}"
    )
    return row.reindex(codes)


def _enrich_candidate_cols(top, as_of, circ_mv=None, sw_l2=None):
    """给候选表加 sw_l2 / circ_mv（元）/ circ_mv_yi（亿元）。"""
    top = top.copy()
    top["code"] = top["code"].astype(str).str.zfill(6)
    sw = _sw_l2_as_of(top["code"], as_of, sw_l2=sw_l2)
    top["sw_l2"] = [ _pretty_sw_l2(sw.get(c)) for c in top["code"] ]
    mv = _circ_mv_as_of(top["code"], as_of, circ_mv=circ_mv)
    top["circ_mv"] = [mv.get(c) for c in top["code"]]
    top["circ_mv_yi"] = pd.to_numeric(top["circ_mv"], errors="coerce") / 1e8
    ordered = [c for c in CANDIDATE_COLS if c in top.columns]
    extra = [c for c in top.columns if c not in ordered]
    return top[ordered + extra]


def output_topn(scores, mask, top_n, cap_band, as_of, output_path, stock_names=None,
                circ_mv=None, sw_l2=None):
    """输出 Top-N 候选清单（strict 宇宙内按得分降序）。

    - scores: pd.Series(stock -> score)
    - mask: bool Series(stock -> 可买)
    - cap_band: 可选市值带过滤（'all' 关闭）；当前仅 'all'，其它值 warning
    - output_path: .csv 或 .md（同时写两种格式）
    - circ_mv: 东财流通市值宽表（元）；None 则读 data/raw/circ_mv.parquet
    - sw_l2: 可选 code→申万二级 Series（测试注入）；None 则 PIT panel asof
    返回 Top-N DataFrame。
    """
    as_of = pd.Timestamp(as_of)
    s = scores.reindex(mask.index).where(mask)
    s = s.dropna()
    s = s.sort_values(ascending=False)
    top = s.head(top_n).reset_index()
    top.columns = ["code", "score"]
    top["code"] = top["code"].astype(str).str.zfill(6)
    if stock_names is not None:
        names = stock_names.rename("name")
        names.index = names.index.astype(str).str.zfill(6)
        names.index.name = "code"
        top = top.merge(names.reset_index(), on="code", how="left")
    top.insert(0, "as_of_date", as_of.strftime("%Y-%m-%d"))
    top.insert(len(top.columns), "rank", range(1, len(top) + 1))
    if cap_band and cap_band != "all":
        logger.warning(
            f"cap_band={cap_band} 在 live 暂未实现市值带过滤（用 'all'）；"
            f"可在训练时用 --cap-band 限定训练池以间接约束"
        )
    top = _enrich_candidate_cols(top, as_of, circ_mv=circ_mv, sw_l2=sw_l2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".md":
        csv_path = out.with_suffix(".csv")
        md_path = out
    else:
        csv_path = out if suffix else out.with_suffix(".csv")
        md_path = csv_path.with_suffix(".md")
    top.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_md(top, md_path, as_of)
    logger.info(f"[Step 5] Top-{len(top)} 候选清单 -> {csv_path} / {md_path}")
    return top


def _fmt_md_cell(val, kind="str"):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if kind == "score":
        return f"{float(val):.4f}"
    if kind == "yi":
        return f"{float(val):.2f}"
    if kind == "yuan":
        return f"{float(val):.0f}"
    return str(val)
    return str(val)


def _write_md(top, out, as_of):
    """写 Markdown 候选清单（含申万二级 / 流通市值）。"""
    lines = [
        f"# 实盘候选清单 {as_of.strftime('%Y-%m-%d')}",
        "",
        f"共 {len(top)} 只（strict 可买宇宙，按模型得分降序）。",
        "",
        "流通市值 `circ_mv` 单位：元（东财）；`circ_mv_yi` 单位：亿元（= circ_mv / 1e8）。",
        "`sw_l2` 为申万 2021 二级行业（PIT as-of 当日，无 panel 则静态映射）。",
        "",
        "| rank | code | name | sw_l2 | circ_mv(元) | circ_mv_yi(亿元) | score |",
        "|------|------|------|-------|-------------|------------------|-------|",
    ]
    for _, r in top.iterrows():
        lines.append(
            "| {rank} | {code} | {name} | {sw} | {mv} | {yi} | {score} |".format(
                rank=int(r["rank"]),
                code=r["code"],
                name=_fmt_md_cell(r.get("name", "")),
                sw=_fmt_md_cell(r.get("sw_l2", "")),
                mv=_fmt_md_cell(r.get("circ_mv"), "yuan"),
                yi=_fmt_md_cell(r.get("circ_mv_yi"), "yi"),
                score=_fmt_md_cell(r.get("score"), "score"),
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 特征名解析 + 主编排
# ══════════════════════════════════════════════════════════════════════════════

def resolve_feature_names(model_entries, factor_config, horizon=None):
    """确定要计算哪些因子。

    优先级：manifest.feature_names > factor_config YAML/JSON > Barra_* 补集 > 全量。
    返回 (feature_names: list, source: str)
    """
    fn = model_entries[0].get("feature_names")
    if fn:
        return list(fn), "manifest"
    # lazy rolling-pool：用并集元数据兜底
    union = model_entries[0].get("feature_names_union_metadata")
    if union:
        return list(union), "manifest_union"
    if factor_config:
        wl = _load_factor_whitelist(factor_config, horizon)
        if wl:
            return wl, "factor_config"
    return None, "all"


def _load_factor_whitelist(config_path, horizon=None):
    """从 YAML/JSON 读取因子白名单（与 run.py::_load_factor_config 同口径，简化）。

    YAML: { h5: {factors: [...]} }（按 horizon 选 key，无 horizon 则取第一个）
    JSON: { horizon: 5, factors: [...] }
    """
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"因子配置文件不存在: {config_path}")
        return None
    import json
    if p.suffix in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            return None
        if horizon is not None:
            key = f"h{horizon}"
            if key in cfg:
                return cfg[key].get("factors")
        # 无 horizon：取第一个含 factors 的 section
        for v in cfg.values():
            if isinstance(v, dict) and v.get("factors"):
                return v["factors"]
        return None
    if p.suffix == ".json":
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg.get("factors")
    logger.warning(f"不支持的配置格式: {p.suffix}")
    return None


def main(as_of=None, lookback_days=DEFAULT_LOOKBACK_DAYS,
         model_dir=None, top_n=30, cap_band="all", output=None,
         factor_config=None, horizon=None, warmup_days=DEFAULT_WARMUP_DAYS,
         feature_neutralize=True, no_download=False, sample=0,
         prefer_model=None, prefer_window=None):
    """实盘日更增量主流程。"""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    logger.info(f"=== live.daily_update as_of={as_of.date()} ===")

    if model_dir is None:
        raise ValueError("必须指定 --model-dir（指向 results/<tag>/）")

    # Step 1: 行情增量下载
    if not no_download:
        incremental_download(as_of, lookback_days, sample=sample)
    else:
        logger.info("[Step 1] --no-download: 跳过下载")

    # Step 2: 加载 + 清洗 raw 面板
    panels = load_clean_panels(sample=sample)

    # Step 5a: 加载模型 manifest（先于因子计算，以确定 feature_names）
    model_entries, manifest_fn, manifest_nc = load_latest_models(
        model_dir, prefer_model=prefer_model, prefer_window=prefer_window,
    )
    # feature_neutralize 一致性校验：manifest 记录 vs CLI --no-feature-neutralize
    # 推荐自动覆盖 + warning（避免误用 raw 因子给残差化训练的模型出分，反之亦然）
    if manifest_fn is not None and bool(manifest_fn) != bool(feature_neutralize):
        logger.warning(
            f"feature_neutralize 不一致：manifest={manifest_fn} vs CLI={feature_neutralize}；"
            f"自动覆盖为 manifest 值 ({manifest_fn}) 以匹配训练口径"
        )
        feature_neutralize = bool(manifest_fn)
    feature_names, fn_source = resolve_feature_names(model_entries, factor_config, horizon)
    if feature_names is None:
        # 全量：枚举所有可计算因子名
        from factors.factor import get_factor_names
        feature_names = get_factor_names(
            prices=panels["prices"], financial=panels["financial"],
            prices_raw=panels["prices_raw"], volume=panels["volume"],
            amount=panels["amount"], open_=panels["open_"],
            high=panels["high"], low=panels["low"],
            clean_ret=panels["clean_ret"], masks=panels["masks"],
            market_prices=panels["market_prices"],
            industry_map=panels["industry_map"],
            circ_mv=panels["circ_mv"], total_mv=panels["total_mv"],
        )
        fn_source = "all"
    logger.info(f"特征集: {len(feature_names)} 个（来源={fn_source}）")

    # Step 2b: 切热身窗（as_of 对齐到数据中 <= as_of 的最近交易日）
    warmup = slice_warmup(panels, as_of, warmup_days)
    as_of = warmup["_as_of"]  # 用对齐后的实际交易日作为当日
    logger.info(f"当日（对齐后）= {as_of.date()}")

    # Step 3: 因子面板 append（热身窗重算 + concat）
    factor_panels = append_factor_panels(feature_names, warmup, as_of)

    # Step 4: 当日中性化（若训练用了 feature_neutralize）
    neut_rows = None
    universe = panels["prices"].columns.astype(str).str.zfill(6)
    if feature_neutralize:
        barra, weights = compute_barra_warmup(warmup)
        neut_rows = neutralize_as_of(
            factor_panels, barra, weights, panels["industry_map"], as_of,
            universe=universe,
            prices=panels["prices"],
            neut_controls=manifest_nc,
        )

    # Step 5b: 组装特征矩阵 + predict
    X = build_feature_matrix(feature_names, neut_rows or {}, factor_panels, as_of, universe=universe)
    n_ok = X.notna().sum()
    all_nan = [c for c in X.columns if int(n_ok.get(c, 0)) == 0]
    low_cov = [
        f"{c}:{int(n_ok[c])}"
        for c in X.columns if 0 < int(n_ok.get(c, 0)) < 200
    ]
    logger.info(
        f"特征覆盖 as_of={as_of.date()} 全NaN={all_nan or '无'} "
        f"低覆盖(<200)={low_cov or '无'}"
    )
    scores = predict_scores(model_entries, X)

    # Step 5c: strict 宇宙 + Top-N 输出
    mask = strict_universe_mask(
        panels["prices"], panels["volume"], panels["masks"], as_of,
    )
    if output is None:
        out_dir = Path(model_dir)
        output = out_dir / f"candidates_{as_of.strftime('%Y%m%d')}.csv"
    from research.ic.universe import load_stock_names
    sn = load_stock_names()
    top = output_topn(
        scores, mask, top_n, cap_band, as_of, output, stock_names=sn,
        circ_mv=panels.get("circ_mv"),
    )

    # 控制台打印 Top-N
    logger.info(f"--- Top-{len(top)} 候选（{as_of.date()}）---")
    for _, r in top.iterrows():
        name = r.get("name", "")
        name = "" if pd.isna(name) else name
        sw = r.get("sw_l2", "")
        sw = "" if pd.isna(sw) else sw
        yi = r.get("circ_mv_yi", np.nan)
        yi_s = f"{float(yi):.1f}亿" if pd.notna(yi) else "NA"
        logger.info(
            f"  {int(r['rank']):>3}. {r['code']} {name:<8} {sw:<10} "
            f"{yi_s:<10} score={r['score']:.4f}"
        )
    return top


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="实盘日更增量链路：行情增量 → 因子 append → 当日中性化 → 模型出分 → Top-N",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m live.daily_update --model-dir results/lgbm_h5_w6-12 --top-n 30\n"
            "  python -m live.daily_update --as-of-date 2026-08-08 --no-download \\\n"
            "      --model-dir results/lgbm_h5_w6-12 --output candidates.md\n"
            "\n"
            "需先有 --save-models 训练产物（results/<tag>/models/models_manifest.json）。\n"
            "详见 docs/操作手册.md §10"
        ),
    )
    p.add_argument("--as-of-date", default=None, help="当日（YYYY-MM-DD，默认今天）")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help=f"行情增量回看日历日数（默认 {DEFAULT_LOOKBACK_DAYS}）")
    p.add_argument("--model-dir", required=True,
                   help="模型目录（指向 results/<tag>/，需含 models/models_manifest.json）")
    p.add_argument("--top-n", type=int, default=30, help="Top-N 候选数（默认 30）")
    p.add_argument("--cap-band", default="all", help="市值带（当前仅 'all'）")
    p.add_argument("--output", default=None, help="输出路径 .csv 或 .md（默认 <model-dir>/candidates_<date>.csv）")
    p.add_argument("--factor-config", default=None, help="因子白名单 YAML（manifest 无 feature_names 时兜底）")
    p.add_argument("--horizon", type=int, default=None, help="持仓期（仅提示，不参与计算）")
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                   help=f"热身窗日历日数（默认 {DEFAULT_WARMUP_DAYS}，覆盖 Barra 252d EWM）")
    p.add_argument("--no-feature-neutralize", dest="feature_neutralize",
                   action="store_false", default=True,
                   help="训练未用 --feature-neutralize 时加此开关（用 raw 因子出分）")
    p.add_argument("--no-download", action="store_true", help="跳过行情增量下载")
    p.add_argument("--sample", type=int, default=0, help="仅前 N 只股票（冒烟）")
    p.add_argument("--prefer-model", default=None, help="多模型 ensemble 时优先选某模型（如 lgbm）")
    p.add_argument("--prefer-window", type=int, default=None, help="优先选某训练窗口（如 6）")
    return p.parse_args(argv)


def _cli():
    try:
        from config.encoding_bootstrap import configure_loguru
        configure_loguru()
    except Exception:
        pass
    args = _parse_args()
    main(
        as_of=args.as_of_date,
        lookback_days=args.lookback_days,
        model_dir=args.model_dir,
        top_n=args.top_n,
        cap_band=args.cap_band,
        output=args.output,
        factor_config=args.factor_config,
        horizon=args.horizon,
        warmup_days=args.warmup_days,
        feature_neutralize=args.feature_neutralize,
        no_download=args.no_download,
        sample=args.sample,
        prefer_model=args.prefer_model,
        prefer_window=args.prefer_window,
    )


if __name__ == "__main__":
    _cli()
