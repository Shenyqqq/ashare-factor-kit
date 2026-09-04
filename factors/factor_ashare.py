"""
factors/factor_ashare.py  —  A 股特色增量因子

落实 docs/ASHARE_FACTOR_GAPS.md 高/中优先级，并补充「有数据未注册」增量：
  - 大单残差净流入（对收益+换手截面回归残差；依赖东财 moneyflow，未就绪则空）
  - 评级上修 / 下调规避 / 研报 EPS 预期差 / 覆盖热度 / 目标价上行空间
  - 龙虎榜机构席位质量 + 净买占比 + reason 分类
  - 股份回购强度 / 完成进度
  - 大宗折价×席位质量 / 机构接盘 / 卖方机构抛压 / 折溢价波动
  - 业绩快报/正式稿 surprise（vs 预告）
  - 板块资金流拥挤（探索性）
  - 融资买入占成交额、净买入、融券卖出规避、融资余额/流通市值
  - 解禁流动性压力 + release_type 定增/激励分化
  - 股本 change_reason 细分（转债/激励行权/限售上市）
  - 研报 EPS 多年度斜率 / 分歧度

日频 PE_TTM/PB：与已有 ``价值_EP``/``价值_PB``（季报 PIT eps/bvps÷不复权价）
经济含义重复，**不注册**（东财 pe_ttm/pb parquet 仅作市值下载副产品）。
ST 戴帽/摘帽：用户剔除 ST，不注册。

约定：
  - 输出「越高越好」；内部 _normalize（稀疏事件因子除外，走 event overlay）
  - PIT：公告类用 announce_date；龙虎榜/大宗/两融用交易日；
    解禁未来窗与 factor_smallcap 一致（交易所提前公告，合法前视）
  - 缺数据返回 NaN 面板，不抛错
  - **勿依赖** ``moneyflow_large``（东财大单全市场未就绪）
  - THS 个股净流入已认定不可用，不注册（下载脚本可保留）
"""
from __future__ import annotations

import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from factors.factor import _normalize


# ── 注册名集合 ──────────────────────────────────────────────────────────────
ASHARE_DENSE_FACTOR_NAMES: frozenset[str] = frozenset({
    "大单残差净流入_5d",
    # 两融日频截面（标的池内较稠密；走 IC 稠密轨，勿进 sparse 方差对齐）
    "融资买入占成交额_5d",
    "融资净买入_5d",
    "融券卖出规避_5d",
    "融资余额流通市值比",
})

ASHARE_SPARSE_FACTOR_NAMES: frozenset[str] = frozenset({
    "评级上修_20d",
    "研报EPS上修次数_20d",
    "研报预期差",
    "龙虎榜机构净买入_20d",
    "龙虎榜机构买入强度_20d",
    "股份回购强度_60d",
    "大宗折价席位质量_20d",
    "板块资金流拥挤_5d",
    # 2026-08 增量（有 raw / 文献口径；避免与 smallcap 同名）
    "龙虎榜净买占比_20d",
    "目标价上行空间",
    "研报覆盖热度_20d",
    "回购完成进度_60d",
    "解禁流动性压力_60d",
    "大宗机构接盘_20d",
    "评级下调规避_20d",
    # 2026-08 事件/公告薄加工（不含日频 PE/PB、不含 ST、不含两融截面）
    "龙虎榜涨幅上榜_20d",
    "龙虎榜换手上榜_20d",
    "龙虎榜跌幅上榜规避_20d",
    "解禁定增压力_60d",
    "解禁激励压力_60d",
    "转债转股稀释_60d",
    "激励行权稀释_60d",
    "限售上市供给_60d",
    "研报EPS斜率",
    "研报EPS分歧度",
    "大宗卖方机构抛压_20d",
    "大宗折溢价波动_20d",
})

ASHARE_EVENT_FACTOR_NAMES: frozenset[str] = frozenset({
    "业绩快报超预期",
})

ASHARE_FACTOR_NAMES: frozenset[str] = (
    ASHARE_DENSE_FACTOR_NAMES | ASHARE_SPARSE_FACTOR_NAMES | ASHARE_EVENT_FACTOR_NAMES
)


def _zero_pad(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6).str.strip()


def _empty_like(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)


def _pivot_event(
    long_df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    prices: pd.DataFrame,
    agg: str = "sum",
) -> pd.DataFrame:
    df = long_df.copy()
    df["code"] = _zero_pad(df["code"])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "code"])
    if value_col is None:
        df = df.assign(_evt=1.0)
        value_col = "_evt"
    panel = df.pivot_table(
        index=date_col, columns="code", values=value_col, aggfunc=agg,
    )
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index().reindex(index=prices.index, columns=prices.columns)


def _ffill_event(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """事件日有值，之后最多 window 个交易日前向填充，再清零。"""
    if panel is None or panel.empty:
        return panel
    filled = panel.ffill(limit=window)
    # 非事件且未在窗口内 → 保持 NaN（ffill 已处理）；再把全 NaN 行保留
    return filled


def _cs_residual(
    y: pd.DataFrame,
    *controls: pd.DataFrame,
    min_obs: int = 30,
) -> pd.DataFrame:
    """逐日截面 OLS：y ~ 1 + controls，返回残差面板。"""
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=np.float64)
    ctrl_list = [c.reindex_like(y) for c in controls]
    # 小宇宙自动放宽最小样本（不能超过股票数）
    n_col = max(1, int(y.shape[1]))
    min_obs = min(min_obs, n_col)
    min_obs = max(3, min_obs) if n_col >= 3 else max(2, n_col)
    for dt in y.index:
        yi = y.loc[dt]
        mats = [yi]
        for c in ctrl_list:
            mats.append(c.loc[dt])
        frame = pd.concat(mats, axis=1).dropna()
        if len(frame) < min_obs:
            continue
        yv = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        X = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        A = np.column_stack([np.ones(len(yv)), X])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, yv, rcond=None)
            resid = yv - A @ coef
            out.loc[dt, frame.index] = resid
        except Exception:
            continue
    return out


def _safe_normalize(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """_normalize 后强制对齐 prices 列（全 NaN 列可能被 zscore 丢掉）。"""
    if panel is None or panel.empty:
        return _empty_like(prices)
    try:
        out = _normalize(panel)
    except Exception:
        out = panel
    return out.reindex(index=prices.index, columns=prices.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 大单残差净流入 — 【已弃用 / DEPRECATED】
# 弃用：依赖东财 moneyflow_large，akshare 全市场拉不稳、单票历史短，因子不可用。
# 勿再进入 IC / 生产池。保留实现供 Tushare DC 接入后复活。
# 详见 docs/ASHARE_FACTOR_DATA_GAPS.md §1。
# ══════════════════════════════════════════════════════════════════════════════

def factor_moneyflow_residual(
    moneyflow: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    turnover: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    [DEPRECATED] 大单残差净流入_5d：先算净流入/成交额滚动均值，再对当日收益与换手截面回归取残差。

    弃用：akshare 资金流数据不足，因子不可用。勿再进入 IC / 生产池。

    开源证券口径近似：剔除「跟涨/跟换手」的被动资金流，保留主动残差。
    缺 amount/ret/turnover 时降级为标准化净流入（仍可用）。
    """
    if moneyflow is None or moneyflow.empty:
        if prices is not None:
            return _empty_like(prices)
        return pd.DataFrame()

    mf = moneyflow.copy()
    if amount is not None and not amount.empty:
        common = mf.columns.intersection(amount.columns)
        ratio = mf[common] / amount[common].replace(0, np.nan)
    else:
        ratio = mf
        logger.warning("大单残差：无 amount，跳过成交额标准化")

    y = ratio.rolling(window, min_periods=max(2, window // 2)).mean()

    controls = []
    if clean_ret is not None and not clean_ret.empty:
        controls.append(clean_ret.reindex_like(y))
    elif prices is not None:
        controls.append(prices.pct_change())
    if turnover is not None and not turnover.empty:
        controls.append(turnover.reindex_like(y))
    elif amount is not None and prices is not None:
        # 粗换手代理：amount / (price * 常数) 不可靠；用 amount 截面 z 作控制
        controls.append(amount.reindex_like(y))

    if controls:
        resid = _cs_residual(y, *controls)
    else:
        resid = y
        logger.warning("大单残差：无控制变量，退化为滚动净流入比")

    return _safe_normalize(resid, prices if prices is not None else resid)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 评级上修 / 研报预期差
# ══════════════════════════════════════════════════════════════════════════════

def _load_rank_forecast() -> pd.DataFrame:
    p = RAW_DIR / "rank_forecast.parquet"
    if not p.exists():
        logger.warning(f"评级变动文件不存在: {p}，请先 python -m data.download_rank_forecast")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    return df.dropna(subset=["announce_date", "code"])


def factor_rating_upgrade(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    评级上修_20d：过去 window 日「调高」次数（巨潮 rank_forecast）。
    PIT：announce_date。
    """
    df = _load_rank_forecast()
    if df.empty:
        return _empty_like(prices)
    chg = df["rating_change"].astype(str)
    up = df[chg.str.contains("调高|上调|上修", na=False)].copy()
    if up.empty:
        # 兜底：首次且评级为买入/增持
        mask = (
            df["is_first"].astype(str).str.contains("是|首次", na=False)
            & df["rating"].astype(str).str.contains("买入|增持|强烈推荐", na=False)
        )
        up = df[mask].copy()
    if up.empty:
        return _empty_like(prices)
    panel = _pivot_event(up, "announce_date", None, prices, agg="sum").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


def _load_research_report() -> pd.DataFrame:
    p = RAW_DIR / "research_report.parquet"
    if not p.exists():
        logger.warning(
            f"研报文件不存在: {p}，请先 python -m data.download_research_report"
        )
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    if "eps_forecast" in df.columns:
        df["eps_forecast"] = pd.to_numeric(df["eps_forecast"], errors="coerce")
    return df.dropna(subset=["announce_date", "code"])


def factor_research_eps_upgrade(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    研报EPS上修次数_20d：同一股票相邻研报（同机构或截面共识）EPS 上修次数。

    MVP：按 code+announce_date 取当日均值 EPS，相对该股此前最近一次均值若上升记 1。
    """
    df = _load_research_report()
    if df.empty or "eps_forecast" not in df.columns:
        return _empty_like(prices)
    daily = (
        df.dropna(subset=["eps_forecast"])
        .groupby(["code", "announce_date"], as_index=False)["eps_forecast"]
        .mean()
        .sort_values(["code", "announce_date"])
    )
    daily["prev"] = daily.groupby("code")["eps_forecast"].shift(1)
    daily["upgrade"] = (daily["eps_forecast"] > daily["prev"]).astype(float)
    daily.loc[daily["prev"].isna(), "upgrade"] = np.nan
    up = daily.dropna(subset=["upgrade"])
    up = up[up["upgrade"] > 0]
    if up.empty:
        return _empty_like(prices)
    panel = _pivot_event(up, "announce_date", "upgrade", prices, agg="sum").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


def factor_research_eps_surprise(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    研报预期差：最新研报共识 EPS 相对过去 lookback 日共识均值的偏离。
    事件日写入，ffill lookback 日。
    """
    df = _load_research_report()
    if df.empty or "eps_forecast" not in df.columns:
        return _empty_like(prices)
    daily = (
        df.dropna(subset=["eps_forecast"])
        .groupby(["code", "announce_date"], as_index=False)["eps_forecast"]
        .mean()
    )
    panel = _pivot_event(daily, "announce_date", "eps_forecast", prices, agg="last")
    # 共识轨迹：事件日有值，其余 NaN → ffill 后与滚动均值比
    traj = panel.ffill()
    mean = traj.rolling(lookback, min_periods=max(5, lookback // 4)).mean()
    surprise = traj - mean
    # 仅在有新研报的窗口保留信号：用事件 mask
    evt = panel.notna().astype(float)
    evt = evt.replace(0, np.nan).ffill(limit=lookback)
    surprise = surprise.where(evt.notna())
    return _safe_normalize(surprise, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 龙虎榜机构席位质量（interpretation）
# ══════════════════════════════════════════════════════════════════════════════

_INST_BUY_RE = re.compile(r"(\d+)\s*家机构买入")
_INST_SELL_RE = re.compile(r"(\d+)\s*家机构卖出")


def _parse_lhb_institution(interpretation: str) -> tuple[float, float]:
    """返回 (机构买入家数, 机构卖出家数)。"""
    s = str(interpretation) if interpretation is not None else ""
    buy = sell = 0.0
    m = _INST_BUY_RE.search(s)
    if m:
        buy = float(m.group(1))
    m = _INST_SELL_RE.search(s)
    if m:
        sell = float(m.group(1))
    return buy, sell


def _load_lhb_inst() -> pd.DataFrame:
    p = RAW_DIR / "lhb_detail.parquet"
    if not p.exists():
        logger.warning(f"龙虎榜文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["lhb_date"] = pd.to_datetime(df["lhb_date"], errors="coerce")
    df = df.dropna(subset=["lhb_date", "code"])
    buys, sells = [], []
    for s in df.get("interpretation", pd.Series(index=df.index, dtype=str)):
        b, e = _parse_lhb_institution(s)
        buys.append(b)
        sells.append(e)
    df["inst_buy_n"] = buys
    df["inst_sell_n"] = sells
    df["inst_net_n"] = df["inst_buy_n"] - df["inst_sell_n"]
    return df


def factor_lhb_inst_net_buy(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """龙虎榜机构净买入_20d：过去 window 日（机构买入家数 - 卖出家数）累计。"""
    df = _load_lhb_inst()
    if df.empty:
        return _empty_like(prices)
    daily = df.groupby(["lhb_date", "code"], as_index=False)["inst_net_n"].sum()
    panel = _pivot_event(daily, "lhb_date", "inst_net_n", prices, agg="sum").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


def factor_lhb_inst_buy_strength(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    龙虎榜机构买入强度_20d：机构买入家数 × sign(net_buy) 的滚动和。
    机构买入且龙虎榜净买额为正 → 更强多头席位信号。
    """
    df = _load_lhb_inst()
    if df.empty:
        return _empty_like(prices)
    net = pd.to_numeric(df.get("net_buy"), errors="coerce").fillna(0)
    df["strength"] = df["inst_buy_n"] * np.sign(net)
    # 机构卖出且净卖 → 负向
    df.loc[df["inst_sell_n"] > 0, "strength"] = (
        df.loc[df["inst_sell_n"] > 0, "strength"]
        - df.loc[df["inst_sell_n"] > 0, "inst_sell_n"]
    )
    daily = df.groupby(["lhb_date", "code"], as_index=False)["strength"].sum()
    panel = _pivot_event(daily, "lhb_date", "strength", prices, agg="sum").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 股份回购强度
# ══════════════════════════════════════════════════════════════════════════════

def factor_repurchase_intensity(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    股份回购强度_60d：公告日计划回购金额上限 / 流通市值（缺则用计划股本比例上限）。
    停止实施/否决记 0；PIT 用 announce_date。
    """
    p = RAW_DIR / "repurchase.parquet"
    if not p.exists():
        logger.warning(f"回购文件不存在: {p}，请先 python -m data.download_repurchase")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df = df.dropna(subset=["announce_date", "code"])
    progress = df.get("progress", pd.Series("", index=df.index)).astype(str)
    bad = progress.str.contains("停止|否决", na=False)
    df = df.loc[~bad].copy()

    intensity = pd.to_numeric(df.get("plan_pct_hi"), errors="coerce")
    # 若有金额 + 市值，优先金额/市值
    mv_path = RAW_DIR / "circ_mv.parquet"
    amt = pd.to_numeric(df.get("plan_amt_hi"), errors="coerce")
    if mv_path.exists() and amt.notna().any():
        mv = pd.read_parquet(mv_path)
        mv_vals = []
        for _, row in df.iterrows():
            code, dt = row["code"], row["announce_date"]
            if code in mv.columns:
                hist = mv[code].dropna()
                hist = hist[hist.index <= dt]
                mv_vals.append(float(hist.iloc[-1]) if len(hist) else np.nan)
            else:
                mv_vals.append(np.nan)
        mv_s = pd.Series(mv_vals, index=df.index)
        ratio = amt / mv_s.replace(0, np.nan)
        intensity = ratio.where(ratio.notna(), intensity / 100.0)
    else:
        intensity = intensity / 100.0

    df["intensity"] = intensity.fillna(0).clip(lower=0)
    panel = _pivot_event(df, "announce_date", "intensity", prices, agg="sum")
    panel = _ffill_event(panel, window)
    return _safe_normalize(panel, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 大宗折价 × 买方席位质量
# ══════════════════════════════════════════════════════════════════════════════

def factor_block_discount_seat_quality(
    prices: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    大宗折价席位质量_20d：折价率 × 买方质量分。

    质量分：买方含「机构专用」→ 1.0；否则若在 dzjy_yybph 近一月胜率榜
    则用 win_rate_5d/100（asof 需 ≤ 交易日，否则 0.5 中性）。
    输出 -discount × quality（折价越大且买方越好 → 越高）。
    """
    p = RAW_DIR / "block_trade.parquet"
    if not p.exists():
        logger.warning(f"大宗交易文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "code"])
    df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors="coerce")
    buyer = df.get("buyer_branch", pd.Series("", index=df.index)).astype(str)

    quality = pd.Series(0.5, index=df.index, dtype=float)
    quality[buyer.str.contains("机构专用", na=False)] = 1.0

    yyb_path = RAW_DIR / "dzjy_yybph.parquet"
    if yyb_path.exists():
        yyb = pd.read_parquet(yyb_path)
        # 取每个 asof 最近一月窗口
        yyb["asof_date"] = pd.to_datetime(yyb["asof_date"], errors="coerce")
        yyb = yyb[yyb.get("window", "近一月") == "近一月"] if "window" in yyb.columns else yyb
        if not yyb.empty and "branch" in yyb.columns:
            rate_col = "win_rate_5d" if "win_rate_5d" in yyb.columns else None
            if rate_col:
                # 简化：用最新快照（文档标明 look-ahead 风险）；机构专用已优待
                latest_asof = yyb["asof_date"].max()
                snap = yyb[yyb["asof_date"] == latest_asof].drop_duplicates("branch")
                rate_map = dict(zip(snap["branch"], snap[rate_col] / 100.0))
                for i, b in buyer.items():
                    if "机构专用" in b:
                        continue
                    if b in rate_map and np.isfinite(rate_map[b]):
                        quality.loc[i] = float(rate_map[b])

    # 折价为负；-discount * quality → 折价越大质量越好分数越高
    df["score"] = (-df["discount_rate"].fillna(0)) * quality
    daily = df.groupby(["trade_date", "code"], as_index=False)["score"].mean()
    panel = _pivot_event(daily, "trade_date", "score", prices, agg="mean")
    avg = panel.rolling(window, min_periods=1).mean()
    return _safe_normalize(avg, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 业绩快报/正式稿 surprise
# ══════════════════════════════════════════════════════════════════════════════

def factor_yjbb_surprise(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    业绩快报超预期：yjbb 净利润 YoY 相对同报告期 yjyg 预告变动中枢的超出部分。

    无匹配预告时退化为 net_profit_yoy 自身（标准化）。
    PIT：优先 announce_date（注意东财可能为修订日）。
    """
    yjbb_path = RAW_DIR / "yjbb.parquet"
    if not yjbb_path.exists():
        logger.warning(f"yjbb 不存在: {yjbb_path}，请先 python -m data.events.download_yjbb")
        return _empty_like(prices)
    yjbb = pd.read_parquet(yjbb_path)
    yjbb["code"] = _zero_pad(yjbb["code"])
    yjbb["announce_date"] = pd.to_datetime(yjbb["announce_date"], errors="coerce")
    yjbb["net_profit_yoy"] = pd.to_numeric(yjbb.get("net_profit_yoy"), errors="coerce")
    yjbb = yjbb.dropna(subset=["announce_date", "code"])

    surprise = yjbb["net_profit_yoy"].copy()
    yjyg_path = RAW_DIR / "yjyg.parquet"
    if yjyg_path.exists():
        yjyg = pd.read_parquet(yjyg_path)
        yjyg["code"] = _zero_pad(yjyg["code"])
        yjyg["change_pct"] = pd.to_numeric(yjyg.get("change_pct"), errors="coerce")
        # 同 code+report_date 预告变动
        if "report_date" in yjyg.columns and "report_date" in yjbb.columns:
            pred = (
                yjyg.dropna(subset=["change_pct"])
                .groupby(["code", "report_date"])["change_pct"]
                .median()
            )
            key = list(zip(yjbb["code"], yjbb["report_date"].astype(str)))
            pred_vals = [pred.get((c, r), np.nan) for c, r in key]
            surprise = yjbb["net_profit_yoy"] - pd.Series(pred_vals, index=yjbb.index)

    yjbb = yjbb.assign(surprise=surprise)
    panel = _pivot_event(yjbb, "announce_date", "surprise", prices, agg="last")
    panel = _ffill_event(panel, window)
    return _safe_normalize(panel, prices)


def factor_yjbb_surprise_raw(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """事件 overlay 版：保持原尺度，不做截面 z-score。"""
    yjbb_path = RAW_DIR / "yjbb.parquet"
    if not yjbb_path.exists():
        return _empty_like(prices)
    yjbb = pd.read_parquet(yjbb_path)
    yjbb["code"] = _zero_pad(yjbb["code"])
    yjbb["announce_date"] = pd.to_datetime(yjbb["announce_date"], errors="coerce")
    yjbb["net_profit_yoy"] = pd.to_numeric(yjbb.get("net_profit_yoy"), errors="coerce")
    yjbb = yjbb.dropna(subset=["announce_date", "code"])
    surprise = yjbb["net_profit_yoy"].copy()
    yjyg_path = RAW_DIR / "yjyg.parquet"
    if yjyg_path.exists() and "report_date" in yjbb.columns:
        yjyg = pd.read_parquet(yjyg_path)
        yjyg["code"] = _zero_pad(yjyg["code"])
        yjyg["change_pct"] = pd.to_numeric(yjyg.get("change_pct"), errors="coerce")
        if "report_date" in yjyg.columns:
            pred = (
                yjyg.dropna(subset=["change_pct"])
                .groupby(["code", "report_date"])["change_pct"]
                .median()
            )
            key = list(zip(yjbb["code"], yjbb["report_date"].astype(str)))
            pred_vals = [pred.get((c, str(r)), np.nan) for c, r in key]
            surprise = yjbb["net_profit_yoy"] - pd.Series(pred_vals, index=yjbb.index)
    yjbb = yjbb.assign(surprise=surprise)
    panel = _pivot_event(yjbb, "announce_date", "surprise", prices, agg="last")
    return _ffill_event(panel, window)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 板块资金流拥挤（探索性）
# ══════════════════════════════════════════════════════════════════════════════

def factor_sector_fund_crowding(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    板块资金流拥挤_5d：个股所属申万行业的主力净流入滚动均值（映射到股票）。

    行业名匹配依赖 industry_map 与 sector_fund_flow.sector 字符串近似；
    匹配失败则该股 NaN。适合 regime/过滤，不作主 IC。
    """
    sec_path = RAW_DIR / "sector_fund_flow.parquet"
    ind_path = RAW_DIR / "industry_map.parquet"
    if not sec_path.exists():
        logger.warning(f"板块资金流不存在: {sec_path}")
        return _empty_like(prices)
    if not ind_path.exists():
        logger.warning(f"行业映射不存在: {ind_path}")
        return _empty_like(prices)

    sec = pd.read_parquet(sec_path)
    sec["date"] = pd.to_datetime(sec["date"], errors="coerce")
    if "main_net" not in sec.columns:
        return _empty_like(prices)
    sec["main_net"] = pd.to_numeric(sec["main_net"], errors="coerce")
    # 行业宽表
    wide = sec.pivot_table(
        index="date", columns="sector", values="main_net", aggfunc="last"
    ).sort_index()
    wide = wide.reindex(prices.index).ffill()
    roll = wide.rolling(window, min_periods=1).mean()

    ind = pd.read_parquet(ind_path)
    # 常见列：code + sw_l1/sw_l2/industry
    code_col = "code" if "code" in ind.columns else ind.columns[0]
    ind[code_col] = _zero_pad(ind[code_col])
    name_col = None
    for c in ("sw_l1", "sw_l2", "industry", "行业", "name"):
        if c in ind.columns:
            name_col = c
            break
    if name_col is None:
        return _empty_like(prices)

    # 静态映射 code → 行业名（探索性；严格 PIT 应用 industry_map_panel）
    mapping = dict(zip(ind[code_col], ind[name_col].astype(str)))
    sectors = list(roll.columns.astype(str))
    out = _empty_like(prices)
    for code in prices.columns:
        ind_name = mapping.get(code)
        if not ind_name:
            continue
        # 模糊匹配：行业名包含或被包含
        match = None
        for s in sectors:
            if ind_name in s or s in ind_name:
                match = s
                break
        if match is None:
            continue
        out[code] = roll[match].to_numpy()
    return _safe_normalize(out, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 龙虎榜净买占比 / 评级下调 / 目标价 / 研报覆盖 / 回购进度 / 大宗机构 / 两融 / 解禁
# ══════════════════════════════════════════════════════════════════════════════

def _map_asof_to_trading_day(
    asof: pd.Series, trading_index: pd.DatetimeIndex,
) -> pd.Series:
    """将 asof 映射到 ≤ asof 的最近交易日（OHLCV 未更新到当日时仍可落点）。"""
    td = pd.DatetimeIndex(trading_index).sort_values()
    if len(td) == 0:
        return pd.Series(pd.NaT, index=asof.index)
    vals = pd.to_datetime(asof, errors="coerce")
    # searchsorted 右侧 → 插入点，减 1 得 ≤ asof 的位置
    pos = td.searchsorted(vals, side="right") - 1
    out = pd.Series(pd.NaT, index=asof.index, dtype="datetime64[ns]")
    ok = pos >= 0
    out.loc[ok] = td[pos[ok]]
    return out


def factor_lhb_net_buy_pct(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    龙虎榜净买占比_20d：过去 window 日 ``net_buy_pct``（净买额/成交额%）累计。

    相对绝对净买额，占比已按成交额标准化，减轻大票偏差。
    出处要点：中泰/华鑫等龙虎榜资金结构研究强调「净买额占成交」刻画异动强度。
    与 ``龙虎榜净买额_20d``（smallcap）并存、口径不同。
    """
    p = RAW_DIR / "lhb_detail.parquet"
    if not p.exists():
        logger.warning(f"龙虎榜文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["lhb_date"] = pd.to_datetime(df["lhb_date"], errors="coerce")
    df["net_buy_pct"] = pd.to_numeric(df.get("net_buy_pct"), errors="coerce")
    df = df.dropna(subset=["lhb_date", "code"])
    daily = df.groupby(["lhb_date", "code"], as_index=False)["net_buy_pct"].mean()
    panel = _pivot_event(daily, "lhb_date", "net_buy_pct", prices, agg="mean").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


def factor_rating_downgrade_avoid(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    评级下调规避_20d：过去 window 日「调低」次数取负（下调越少 = 越高）。

    分析师一致预期下修通常伴随负向 alpha（国金资金跟踪：预测下调板块弱势）。
    PIT：announce_date。与 ``评级上修_20d`` 互补。
    """
    df = _load_rank_forecast()
    if df.empty:
        return _empty_like(prices)
    chg = df["rating_change"].astype(str)
    down = df[chg.str.contains("调低|下调|下修", na=False)].copy()
    if down.empty:
        return _empty_like(prices)
    panel = _pivot_event(down, "announce_date", None, prices, agg="sum").fillna(0)
    # 取负：下调少 → 高分
    return _safe_normalize(-panel.rolling(window, min_periods=1).sum(), prices)


def factor_target_price_upside(
    prices: pd.DataFrame,
    prices_raw: pd.DataFrame | None = None,
    hold: int = 40,
) -> pd.DataFrame:
    """
    目标价上行空间：公告日共识目标价上限相对现价的升水 (target_high/close - 1)。

    卖方目标价隐含预期收益；升水越大 → 一致预期越乐观（伪一致预期代理）。
    PIT：announce_date；事件后 ffill ``hold`` 个交易日。
    价格优先不复权 ``prices_raw``（目标价通常按市价口径）。
    """
    df = _load_rank_forecast()
    if df.empty or "target_high" not in df.columns:
        return _empty_like(prices)
    df["target_high"] = pd.to_numeric(df["target_high"], errors="coerce")
    df = df.dropna(subset=["target_high"])
    if df.empty:
        return _empty_like(prices)
    px = prices_raw if prices_raw is not None and not prices_raw.empty else prices
    px = px.reindex(index=prices.index, columns=prices.columns)
    closes = []
    for _, row in df.iterrows():
        code, dt = row["code"], row["announce_date"]
        if code not in px.columns:
            closes.append(np.nan)
            continue
        hist = px[code].dropna()
        hist = hist[hist.index <= dt]
        closes.append(float(hist.iloc[-1]) if len(hist) else np.nan)
    df = df.assign(_close=closes)
    df["upside"] = df["target_high"] / df["_close"].replace(0, np.nan) - 1.0
    df = df.dropna(subset=["upside"])
    if df.empty:
        return _empty_like(prices)
    # 同日多机构：取中位数共识
    daily = df.groupby(["announce_date", "code"], as_index=False)["upside"].median()
    panel = _pivot_event(daily, "announce_date", "upside", prices, agg="last")
    panel = _ffill_event(panel, hold)
    return _safe_normalize(panel, prices)


def factor_research_coverage(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    研报覆盖热度_20d：过去 window 日研报篇数。

    分析师关注度上升常对应信息挖掘与机构关注（覆盖因子文献常见）。
    PIT：announce_date。越高越好（探索性；需防小盘炒作噪声）。
    """
    df = _load_research_report()
    if df.empty:
        return _empty_like(prices)
    panel = _pivot_event(df, "announce_date", None, prices, agg="sum").fillna(0)
    return _safe_normalize(panel.rolling(window, min_periods=1).sum(), prices)


def factor_margin_buy_to_amount(
    prices: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    融资买入占成交额_5d：滚动融资买入额 / 成交额。

    国金等「两融活跃度/融资买入占比」口径；相对裸融资买入额更可比。
    与 smallcap ``融资买入额_5d`` 并存。无 PIT 问题（T+1 披露，按交易日对齐）。
    """
    p = RAW_DIR / "margin_detail.parquet"
    if not p.exists():
        logger.warning(f"两融明细不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p, columns=["date", "code", "margin_buy_amount"])
    df["code"] = _zero_pad(df["code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["margin_buy_amount"] = pd.to_numeric(df["margin_buy_amount"], errors="coerce")
    df = df.dropna(subset=["date", "code", "margin_buy_amount"])
    buy = df.pivot_table(
        index="date", columns="code", values="margin_buy_amount", aggfunc="last",
    ).sort_index()
    buy = buy.reindex(index=prices.index, columns=prices.columns)
    if amount is None or amount.empty:
        ap = RAW_DIR / "amount.parquet"
        if ap.exists():
            amount = pd.read_parquet(ap)
        else:
            logger.warning("融资买入占成交额：无 amount，退化为滚动买入额")
            return _safe_normalize(buy.rolling(window, min_periods=2).sum(), prices)
    amount = amount.reindex(index=prices.index, columns=prices.columns)
    ratio = buy / amount.replace(0, np.nan)
    roll = ratio.rolling(window, min_periods=2).mean()
    return _safe_normalize(roll, prices)


def factor_repurchase_completion(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    回购完成进度_60d：完成率 × 计划规模代理。

    完成率 = done_amt / plan_amt_hi；再乘 plan_pct_hi/100（缺则用
    log1p(plan_amt)），避免「多数已完成=1」导致截面无变异、标准化后全 NaN。
    相对「计划强度」，更强调真实执行。PIT：announce_date；停止/否决剔除。
    """
    p = RAW_DIR / "repurchase.parquet"
    if not p.exists():
        logger.warning(f"回购文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df = df.dropna(subset=["announce_date", "code"])
    progress = df.get("progress", pd.Series("", index=df.index)).astype(str)
    df = df.loc[~progress.str.contains("停止|否决", na=False)].copy()
    done = pd.to_numeric(df.get("done_amt"), errors="coerce")
    plan = pd.to_numeric(df.get("plan_amt_hi"), errors="coerce")
    pct = pd.to_numeric(df.get("plan_pct_hi"), errors="coerce")
    completion = (done / plan.replace(0, np.nan)).clip(lower=0, upper=1.5)
    scale = (pct / 100.0).where(pct.notna(), np.log1p(plan.clip(lower=0)))
    df["score"] = (completion * scale).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["score"])
    if df.empty:
        return _empty_like(prices)
    df = df.assign(
        signal_date=_map_asof_to_trading_day(df["announce_date"], prices.index),
    ).dropna(subset=["signal_date"])
    panel = _pivot_event(df, "signal_date", "score", prices, agg="last")
    panel = _ffill_event(panel, window)
    return _safe_normalize(panel, prices)


def factor_lockup_adv_pressure(prices: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """
    解禁流动性压力_60d：未来 horizon 日解禁市值 / 近 20 日日均成交额，取负。

    与 ``未来60日解禁市值占比``（/流通市值）互补：ADV 分母刻画「消化能力」。
    前视合法性同 smallcap：解禁日交易所提前公告，T 日已知 [t,t+h) 计划。
    """
    p = RAW_DIR / "lockup_release.parquet"
    if not p.exists():
        logger.warning(f"解禁文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["actual_release_value"] = pd.to_numeric(
        df.get("actual_release_value"), errors="coerce",
    )
    df = df.dropna(subset=["release_date", "code"])
    panel = _pivot_event(
        df, "release_date", "actual_release_value", prices, agg="sum",
    ).fillna(0)
    # 前向滚动：翻转 + backward rolling（与 factor_smallcap 一致）
    flipped = panel[::-1]
    fwd = flipped.rolling(horizon, min_periods=1).sum()[::-1]
    ap = RAW_DIR / "amount.parquet"
    if not ap.exists():
        logger.warning("解禁流动性压力：无 amount.parquet")
        return _empty_like(prices)
    amt = pd.read_parquet(ap).reindex(index=prices.index, columns=prices.columns)
    adv20 = amt.rolling(20, min_periods=5).mean()
    pressure = fwd / adv20.replace(0, np.nan)
    return _safe_normalize(-pressure.replace([np.inf, -np.inf], np.nan), prices)


def factor_block_inst_takeover(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    大宗机构接盘_20d：买方含「机构专用」的成交额滚动和 / 流通市值。

    折价大宗+机构接盘更偏「换手」而非游资出货（与折价席位质量因子互补：
    本因子强调机构买方规模，不依赖 yybph 快照胜率）。
    """
    p = RAW_DIR / "block_trade.parquet"
    if not p.exists():
        logger.warning(f"大宗交易文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    buyer = df.get("buyer_branch", pd.Series("", index=df.index)).astype(str)
    df = df.loc[buyer.str.contains("机构专用", na=False)].copy()
    if df.empty:
        return _empty_like(prices)
    daily = df.groupby(["trade_date", "code"], as_index=False)["amount"].sum()
    panel = _pivot_event(daily, "trade_date", "amount", prices, agg="sum").fillna(0)
    roll = panel.rolling(window, min_periods=1).sum()
    mv_path = RAW_DIR / "circ_mv.parquet"
    if mv_path.exists():
        mv = pd.read_parquet(mv_path).reindex(index=prices.index, columns=prices.columns)
        roll = roll / mv.replace(0, np.nan)
    return _safe_normalize(roll, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 9. 两融偿还/净买入、融券卖出、融资/流通市值
# ══════════════════════════════════════════════════════════════════════════════

def _load_margin_detail_long() -> pd.DataFrame:
    p = RAW_DIR / "margin_detail.parquet"
    if not p.exists():
        logger.warning(f"两融明细不存在: {p}")
        return pd.DataFrame()
    cols = [
        "date", "code", "margin_balance", "margin_buy_amount", "margin_repay_amount",
        "short_sell_volume", "short_repay_volume", "short_balance_amount",
    ]
    df = pd.read_parquet(p)
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    df["code"] = _zero_pad(df["code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in keep:
        if c in ("date", "code"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "code"])


def _pivot_margin_col(
    df: pd.DataFrame, col: str, prices: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return _empty_like(prices)
    wide = df.pivot_table(
        index="date", columns="code", values=col, aggfunc="last",
    ).sort_index()
    return wide.reindex(index=prices.index, columns=prices.columns)


def factor_margin_net_buy(
    prices: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    融资净买入_5d：滚动 (融资买入额 - 融资偿还额) / 成交额。

    深市偿还列常缺 → repay 按 0 处理（此时退化为买入额口径，与
    ``融资买入占成交额_5d`` 接近；沪市有偿还时可刻画「去杠杆」）。
    无 PIT 问题（T+1 披露，按交易日对齐）。
    """
    df = _load_margin_detail_long()
    if df.empty:
        return _empty_like(prices)
    buy = _pivot_margin_col(df, "margin_buy_amount", prices)
    repay = _pivot_margin_col(df, "margin_repay_amount", prices)
    net = buy.fillna(0) - repay.fillna(0)
    net = net.where(buy.notna() | repay.notna())
    if amount is None or amount.empty:
        ap = RAW_DIR / "amount.parquet"
        amount = pd.read_parquet(ap) if ap.exists() else None
    if amount is not None and not amount.empty:
        amount = amount.reindex(index=prices.index, columns=prices.columns)
        net = net / amount.replace(0, np.nan)
    roll = net.rolling(window, min_periods=2).mean()
    return _safe_normalize(roll, prices)


def factor_short_sell_avoid(
    prices: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    融券卖出规避_5d：滚动融券卖出量（主动卖空）取负。

    融券卖出增加 = 看空压力 → 取负让「少卖空 = 高分」。
    有成交额时用卖出量/成交额近似强度（量单位为股/手混源，截面 z 仍可比）。
    """
    df = _load_margin_detail_long()
    if df.empty:
        return _empty_like(prices)
    short = _pivot_margin_col(df, "short_sell_volume", prices)
    if amount is None or amount.empty:
        ap = RAW_DIR / "amount.parquet"
        amount = pd.read_parquet(ap) if ap.exists() else None
    if amount is not None and not amount.empty:
        amount = amount.reindex(index=prices.index, columns=prices.columns)
        short = short / amount.replace(0, np.nan)
    roll = short.rolling(window, min_periods=2).mean()
    return _safe_normalize(-roll, prices)


def factor_margin_balance_to_float(prices: pd.DataFrame) -> pd.DataFrame:
    """
    融资余额流通市值比：margin_balance / circ_mv，取负（杠杆拥挤越高分越低）。

    与余额变化率/买入额因子互补：刻画存量杠杆拥挤而非边际流入。
    """
    df = _load_margin_detail_long()
    if df.empty:
        return _empty_like(prices)
    bal = _pivot_margin_col(df, "margin_balance", prices)
    mv_path = RAW_DIR / "circ_mv.parquet"
    if not mv_path.exists():
        logger.warning("融资余额流通市值比：无 circ_mv.parquet")
        return _empty_like(prices)
    mv = pd.read_parquet(mv_path).reindex(index=prices.index, columns=prices.columns)
    ratio = bal / mv.replace(0, np.nan)
    return _safe_normalize(-ratio.replace([np.inf, -np.inf], np.nan), prices)


# ══════════════════════════════════════════════════════════════════════════════
# 10. 龙虎榜 reason 分类
# ══════════════════════════════════════════════════════════════════════════════

def _lhb_reason_bucket(reason: object) -> str:
    """将上榜原因粗分为 up / down / turnover / other。"""
    s = str(reason) if reason is not None else ""
    if "换手" in s:
        return "turnover"
    if "跌" in s:
        return "down"
    if "涨" in s:
        return "up"
    return "other"


def _factor_lhb_reason_count(
    prices: pd.DataFrame,
    bucket: str,
    window: int = 20,
    negate: bool = False,
) -> pd.DataFrame:
    p = RAW_DIR / "lhb_detail.parquet"
    if not p.exists():
        logger.warning(f"龙虎榜文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p, columns=["code", "lhb_date", "reason"])
    df["code"] = _zero_pad(df["code"])
    df["lhb_date"] = pd.to_datetime(df["lhb_date"], errors="coerce")
    df = df.dropna(subset=["lhb_date", "code"])
    df["bucket"] = df["reason"].map(_lhb_reason_bucket)
    df = df.loc[df["bucket"] == bucket]
    if df.empty:
        return _empty_like(prices)
    # 同日同票多 reason 行：按日去重计数为 1
    daily = df.drop_duplicates(subset=["code", "lhb_date"]).assign(_evt=1.0)
    panel = _pivot_event(daily, "lhb_date", "_evt", prices, agg="sum").fillna(0)
    roll = panel.rolling(window, min_periods=1).sum()
    if negate:
        roll = -roll
    return _safe_normalize(roll, prices)


def factor_lhb_reason_up(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """龙虎榜涨幅上榜_20d：过去 window 日因涨幅偏离上榜的天数。"""
    return _factor_lhb_reason_count(prices, "up", window=window, negate=False)


def factor_lhb_reason_turnover(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """龙虎榜换手上榜_20d：过去 window 日因换手上榜的天数。"""
    return _factor_lhb_reason_count(prices, "turnover", window=window, negate=False)


def factor_lhb_reason_down_avoid(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """龙虎榜跌幅上榜规避_20d：跌幅类上榜天数取负。"""
    return _factor_lhb_reason_count(prices, "down", window=window, negate=True)


# ══════════════════════════════════════════════════════════════════════════════
# 11. 解禁 release_type 分化
# ══════════════════════════════════════════════════════════════════════════════

def _lockup_type_pressure(
    prices: pd.DataFrame,
    type_keywords: tuple[str, ...],
    horizon: int = 60,
) -> pd.DataFrame:
    """
    未来 horizon 日匹配 release_type 的解禁市值 / 流通市值，取负。
    前视合法性同现有解禁因子。
    """
    p = RAW_DIR / "lockup_release.parquet"
    if not p.exists():
        logger.warning(f"解禁文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["actual_release_value"] = pd.to_numeric(
        df.get("actual_release_value"), errors="coerce",
    )
    rt = df.get("release_type", pd.Series("", index=df.index)).astype(str)
    mask = False
    for kw in type_keywords:
        mask = mask | rt.str.contains(kw, na=False)
    df = df.loc[mask].dropna(subset=["release_date", "code"])
    if df.empty:
        return _empty_like(prices)
    panel = _pivot_event(
        df, "release_date", "actual_release_value", prices, agg="sum",
    ).fillna(0)
    fwd = panel[::-1].rolling(horizon, min_periods=1).sum()[::-1]
    mv_path = RAW_DIR / "circ_mv.parquet"
    if not mv_path.exists():
        return _safe_normalize(-fwd, prices)
    mv = pd.read_parquet(mv_path).reindex(index=prices.index, columns=prices.columns)
    ratio = fwd / mv.replace(0, np.nan)
    return _safe_normalize(-ratio.replace([np.inf, -np.inf], np.nan), prices)


def factor_lockup_placement_pressure(prices: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """解禁定增压力_60d：未来定增/定向增发类解禁市值占比取负。"""
    return _lockup_type_pressure(prices, ("定向增发",), horizon=horizon)


def factor_lockup_incentive_pressure(prices: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """解禁激励压力_60d：未来股权激励类解禁市值占比取负。"""
    return _lockup_type_pressure(prices, ("股权激励",), horizon=horizon)


# ══════════════════════════════════════════════════════════════════════════════
# 12. 股本 change_reason 细分
# ══════════════════════════════════════════════════════════════════════════════

def _share_change_reason_pressure(
    prices: pd.DataFrame,
    reason_keywords: tuple[str, ...],
    window: int = 60,
) -> pd.DataFrame:
    """
    过去 window 日匹配 change_reason 的事件次数取负（稀释/供给压力）。
    PIT：announce_date（股本变动公告日）。
    """
    p = RAW_DIR / "share_change.parquet"
    if not p.exists():
        # 兼容旧文件名
        p = RAW_DIR / "shares.parquet"
    if not p.exists():
        logger.warning(f"股本变动文件不存在: share_change/shares.parquet")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    if "change_reason" not in df.columns or "announce_date" not in df.columns:
        return _empty_like(prices)
    df["code"] = _zero_pad(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    reason = df["change_reason"].astype(str)
    # 剔除纯「定期报告」噪声行
    mask = False
    for kw in reason_keywords:
        mask = mask | reason.str.contains(kw, na=False)
    df = df.loc[mask].dropna(subset=["announce_date", "code"])
    if df.empty:
        return _empty_like(prices)
    daily = df.drop_duplicates(subset=["code", "announce_date"]).assign(_evt=1.0)
    panel = _pivot_event(daily, "announce_date", "_evt", prices, agg="sum").fillna(0)
    return _safe_normalize(-panel.rolling(window, min_periods=1).sum(), prices)


def factor_cb_conversion_dilution(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """转债转股稀释_60d：可转债转股公告次数取负。"""
    return _share_change_reason_pressure(prices, ("可转债转股",), window=window)


def factor_incentive_exercise_dilution(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """激励行权稀释_60d：期权行权/激励股份解禁等取负。"""
    return _share_change_reason_pressure(
        prices, ("期权行权", "激励股份解禁"), window=window,
    )


def factor_restricted_listing_supply(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """限售上市供给_60d：限售股份上市/股权分置受限股份上市次数取负。"""
    return _share_change_reason_pressure(
        prices, ("限售股份上市", "股权分置受限"), window=window,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 13. 研报 EPS 斜率 / 分歧度
# ══════════════════════════════════════════════════════════════════════════════

_EPS_YEAR_RE = re.compile(r"^eps_(\d{4})$")


def factor_research_eps_slope(prices: pd.DataFrame, hold: int = 40) -> pd.DataFrame:
    """
    研报EPS斜率：同研报多年度 eps_YYYY 的年化斜率（远年 - 近年）/ |近年|。

    增长预期结构越陡 → 越高。PIT：announce_date；事件后 ffill hold 日。
    历史覆盖偏薄（东财多年份列近年才齐），缺两年预测则 NaN。
    """
    df = _load_research_report()
    if df.empty:
        return _empty_like(prices)
    year_cols = sorted(
        ((int(m.group(1)), c) for c in df.columns if (m := _EPS_YEAR_RE.match(str(c)))),
        key=lambda x: x[0],
    )
    if len(year_cols) < 2:
        return _empty_like(prices)
    years = [y for y, _ in year_cols]
    cols = [c for _, c in year_cols]
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # 逐行取首尾有效点
    slopes = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = mat[i]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        idxs = np.flatnonzero(ok)
        i0, i1 = idxs[0], idxs[-1]
        e0, e1 = row[i0], row[i1]
        span = max(1, years[i1] - years[i0])
        base = abs(e0)
        if base < 1e-6:
            continue
        slopes[i] = ((e1 - e0) / span) / base
    df = df.assign(_slope=slopes).dropna(subset=["_slope"])
    if df.empty:
        return _empty_like(prices)
    daily = df.groupby(["announce_date", "code"], as_index=False)["_slope"].median()
    panel = _pivot_event(daily, "announce_date", "_slope", prices, agg="last")
    panel = _ffill_event(panel, hold)
    return _safe_normalize(panel, prices)


def factor_research_eps_dispersion(prices: pd.DataFrame, hold: int = 40) -> pd.DataFrame:
    """
    研报EPS分歧度：同日同票多机构 eps_forecast 截面标准差，取负。

    分歧大 → 不确定性溢价/未来收益偏弱（文献常见）→ 取负让共识更齐 = 高分。
    PIT：announce_date。单机构日无分歧 → NaN。
    """
    df = _load_research_report()
    if df.empty or "eps_forecast" not in df.columns:
        return _empty_like(prices)
    sub = df.dropna(subset=["eps_forecast"]).copy()
    if sub.empty:
        return _empty_like(prices)
    g = sub.groupby(["announce_date", "code"])["eps_forecast"]
    daily = g.agg(eps_std="std", n="count").reset_index()
    daily = daily.loc[daily["n"] >= 2]
    if daily.empty:
        return _empty_like(prices)
    panel = _pivot_event(daily, "announce_date", "eps_std", prices, agg="last")
    panel = _ffill_event(panel, hold)
    return _safe_normalize(-panel, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 14. 大宗卖方质量 / 折溢价波动
# ══════════════════════════════════════════════════════════════════════════════

def factor_block_seller_inst_pressure(
    prices: pd.DataFrame, window: int = 20,
) -> pd.DataFrame:
    """
    大宗卖方机构抛压_20d：卖方含「机构专用」的成交额/流通市值滚动和，取负。

    与 ``大宗机构接盘_20d``（买方机构）对称：机构卖出更偏知情抛压。
    """
    p = RAW_DIR / "block_trade.parquet"
    if not p.exists():
        logger.warning(f"大宗交易文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    seller = df.get("seller_branch", pd.Series("", index=df.index)).astype(str)
    df = df.loc[seller.str.contains("机构专用", na=False)].copy()
    if df.empty:
        return _empty_like(prices)
    daily = df.groupby(["trade_date", "code"], as_index=False)["amount"].sum()
    panel = _pivot_event(daily, "trade_date", "amount", prices, agg="sum").fillna(0)
    roll = panel.rolling(window, min_periods=1).sum()
    mv_path = RAW_DIR / "circ_mv.parquet"
    if mv_path.exists():
        mv = pd.read_parquet(mv_path).reindex(index=prices.index, columns=prices.columns)
        roll = roll / mv.replace(0, np.nan)
    return _safe_normalize(-roll, prices)


def factor_block_discount_vol(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    大宗折溢价波动_20d：过去 window 日 discount_rate 标准差取负。

    与均值折价率互补：折溢价不稳常伴随博弈/出货分歧。无成交日不计入。
    """
    p = RAW_DIR / "block_trade.parquet"
    if not p.exists():
        logger.warning(f"大宗交易文件不存在: {p}")
        return _empty_like(prices)
    df = pd.read_parquet(p)
    df["code"] = _zero_pad(df["code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["discount_rate"] = pd.to_numeric(df.get("discount_rate"), errors="coerce")
    df = df.dropna(subset=["trade_date", "code", "discount_rate"])
    if df.empty:
        return _empty_like(prices)
    daily = df.groupby(["trade_date", "code"], as_index=False)["discount_rate"].mean()
    panel = _pivot_event(daily, "trade_date", "discount_rate", prices, agg="mean")
    vol = panel.rolling(window, min_periods=2).std()
    return _safe_normalize(-vol, prices)


# ══════════════════════════════════════════════════════════════════════════════
# 批量入口
# ══════════════════════════════════════════════════════════════════════════════

def get_ashare_factors(
    prices: pd.DataFrame,
    moneyflow: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    turnover: pd.DataFrame | None = None,
    prices_raw: pd.DataFrame | None = None,
    factor_names: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """计算 A 股特色因子子集；缺数据时跳过并 warning。"""
    want = set(ASHARE_FACTOR_NAMES) if factor_names is None else (
        set(factor_names) & set(ASHARE_FACTOR_NAMES)
    )
    # event 版单独走 get_event_overlay；此处排除 raw overlay 名的双重计算策略：
    # 「业绩快报超预期」在 registry 走 normalize 版
    out: dict[str, pd.DataFrame] = {}

    def _put(name: str, fn, *args, **kwargs):
        if name not in want:
            return
        try:
            panel = fn(*args, **kwargs)
            if panel is not None and not getattr(panel, "empty", True):
                out[name] = panel
            else:
                logger.warning(f"A股因子 {name} 返回空面板")
        except Exception as e:
            logger.warning(f"A股因子 {name} 计算失败: {e}")

    _put(
        "大单残差净流入_5d",
        factor_moneyflow_residual,
        moneyflow if moneyflow is not None else pd.DataFrame(),
        amount=amount,
        clean_ret=clean_ret,
        turnover=turnover,
        prices=prices,
        window=5,
    )
    _put("评级上修_20d", factor_rating_upgrade, prices, 20)
    _put("研报EPS上修次数_20d", factor_research_eps_upgrade, prices, 20)
    _put("研报预期差", factor_research_eps_surprise, prices, 60)
    _put("龙虎榜机构净买入_20d", factor_lhb_inst_net_buy, prices, 20)
    _put("龙虎榜机构买入强度_20d", factor_lhb_inst_buy_strength, prices, 20)
    _put("股份回购强度_60d", factor_repurchase_intensity, prices, 60)
    _put("大宗折价席位质量_20d", factor_block_discount_seat_quality, prices, 20)
    _put("业绩快报超预期", factor_yjbb_surprise, prices, 60)
    _put("板块资金流拥挤_5d", factor_sector_fund_crowding, prices, 5)
    # 2026-08 增量
    _put("龙虎榜净买占比_20d", factor_lhb_net_buy_pct, prices, 20)
    _put(
        "目标价上行空间",
        factor_target_price_upside,
        prices,
        prices_raw=prices_raw,
        hold=40,
    )
    _put("研报覆盖热度_20d", factor_research_coverage, prices, 20)
    _put(
        "融资买入占成交额_5d",
        factor_margin_buy_to_amount,
        prices,
        amount=amount,
        window=5,
    )
    _put("回购完成进度_60d", factor_repurchase_completion, prices, 60)
    _put("解禁流动性压力_60d", factor_lockup_adv_pressure, prices, 60)
    _put("大宗机构接盘_20d", factor_block_inst_takeover, prices, 20)
    _put("评级下调规避_20d", factor_rating_downgrade_avoid, prices, 20)
    # 2026-08 有数据未注册
    _put("融资净买入_5d", factor_margin_net_buy, prices, amount=amount, window=5)
    _put("融券卖出规避_5d", factor_short_sell_avoid, prices, amount=amount, window=5)
    _put("融资余额流通市值比", factor_margin_balance_to_float, prices)
    _put("龙虎榜涨幅上榜_20d", factor_lhb_reason_up, prices, 20)
    _put("龙虎榜换手上榜_20d", factor_lhb_reason_turnover, prices, 20)
    _put("龙虎榜跌幅上榜规避_20d", factor_lhb_reason_down_avoid, prices, 20)
    _put("解禁定增压力_60d", factor_lockup_placement_pressure, prices, 60)
    _put("解禁激励压力_60d", factor_lockup_incentive_pressure, prices, 60)
    _put("转债转股稀释_60d", factor_cb_conversion_dilution, prices, 60)
    _put("激励行权稀释_60d", factor_incentive_exercise_dilution, prices, 60)
    _put("限售上市供给_60d", factor_restricted_listing_supply, prices, 60)
    _put("研报EPS斜率", factor_research_eps_slope, prices, 40)
    _put("研报EPS分歧度", factor_research_eps_dispersion, prices, 40)
    _put("大宗卖方机构抛压_20d", factor_block_seller_inst_pressure, prices, 20)
    _put("大宗折溢价波动_20d", factor_block_discount_vol, prices, 20)
    return out
