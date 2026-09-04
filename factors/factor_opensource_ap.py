"""
factors/factor_opensource_ap.py — OpenSourceAP CrossSection 会计/价量因子

A 股改写自 Chen/Zimmermann OpenSourceAP predictors。
财务字段经季报 pivot → 信号构造 → ``pit_reindex_ffill`` / ``_pivot_financial`` PIT 对齐。

方向约定（项目统一「越高越好」）
--------------------------------
| 中文名         | OpenSourceAP Acronym   | SignalDoc Sign | 本仓库处理        |
|----------------|------------------------|----------------|-------------------|
| 资产增长       | AssetGrowth            | -1             | 取负              |
| 资产市值比     | AM                     | +1             | 原向              |
| 现金流市值比   | cfp                    | +1             | 原向              |
| 权益增长       | ChEQ                   | -1             | 取负              |
| 盈利一致性     | EarningsConsistency    | +1             | 原向              |
| 营收增长秩     | MeanRankRevGrowth      | +1*            | 原向（见 Batch-1）|
| 盈利连增期数   | NumEarnIncrease        | +1             | 原向              |
| 一年股本扩张   | ShareIss1Y             | -1             | 取负              |
| 五年股本扩张   | ShareIss5Y             | -1             | 取负              |
| 综合股权融资   | CompEquIss             | -1             | 取负              |
| 上市年龄       | FirmAge                | -1             | 取负（年轻高分）  |
| 年龄动量       | FirmAgeMom             | +1             | 仅最年轻五分位    |
| 权益变化资产比 | DelEqu                 | -1             | 取负              |
| 应计占比       | PctAcc                 | -1             | 取负（eps+ocf_ps）|
| 现金流价格波动 | VarCF                  | -1             | 取负（ocf/P 波动）|
| 毛利资产比     | GP                     | +1             | **近似** 见 docstring |
| 营收市值比     | SP                     | +1             | **近似** 见 docstring |
| 净债务市值比   | NetDebtPrice           | -1             | **近似** 取负     |
| 综合债务融资   | CompositeDebtIssuance  | -1             | **近似** 取负     |
| 月最大收益     | MaxRet                 | -1             | 取负              |
| 收益偏度       | ReturnSkew             | -1             | 取负              |
| 行业集中度     | Herf                   | -1             | 取负（市值份额 HHI）|
| 应计资产比     | Accruals               | -1             | **CF 近似** 取负  |
| 经营利润权益比 | OperProf               | +1             | **近似**（缺 SGA/利息）|
| 外部融资资产比 | XFIN                   | -1             | **近似** 取负     |
| 资产周转变化   | ChAssetTurnover        | +1             | **近似**（sales/AT）|
| 经营杠杆       | OPLeverage             | +1             | **近似**（缺 SGA）|
| 协偏度         | Coskewness             | -1             | 日频 ACX 式取负   |
| 残差动量       | ResidualMomentum       | +1             | **CAPM 残差**近似 |
| 季节动量       | MomSeason              | +1             | 原向（年 2–5）    |

数据口径备注
------------
- ``operating_cashflow`` 为**每股**经营现金流。
- ChEQ / DelEqu 用 ``bvps``（±股本）作账面权益代理。
- GP/SP/NetDebtPrice/CompositeDebtIssuance/OperProf/XFIN/应计资产比/经营杠杆/资产周转变化
  为现有字段可算的**近似**，非 Compustat 原定义。
- Batch-3 严格缺口（缺 CHE/INVT/ACT/融资 CF 等）不假实现。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import RAW_DIR, UNIVERSE_DIR
from factors.factor import _normalize, _pivot_financial
from factors.factor_size_alpha import resolve_mcap_panel
from utils.pit_align import pit_reindex_ffill


OPENSOURCE_AP_FACTOR_NAMES = frozenset({
    # Batch-1
    "资产增长",
    "资产市值比",
    "现金流市值比",
    "权益增长",
    "盈利一致性",
    "营收增长秩",
    # Batch-2+
    "盈利连增期数",
    "一年股本扩张",
    "五年股本扩张",
    "综合股权融资",
    "上市年龄",
    "年龄动量",
    "权益变化资产比",
    "应计占比",
    "现金流价格波动",
    "毛利资产比",
    "营收市值比",
    "净债务市值比",
    "综合债务融资",
    "月最大收益",
    "收益偏度",
    "行业集中度",
    # Batch-3（现有字段 + 价量；严格 BS/CF 缺口见文档跳过项）
    "应计资产比",
    "经营利润权益比",
    "外部融资资产比",
    "资产周转变化",
    "经营杠杆",
    "协偏度",
    "残差动量",
    "季节动量",
})

OPENSOURCE_AP_ACRONYM = {
    "资产增长": "AssetGrowth",
    "资产市值比": "AM",
    "现金流市值比": "cfp",
    "权益增长": "ChEQ",
    "盈利一致性": "EarningsConsistency",
    "营收增长秩": "MeanRankRevGrowth",
    "盈利连增期数": "NumEarnIncrease",
    "一年股本扩张": "ShareIss1Y",
    "五年股本扩张": "ShareIss5Y",
    "综合股权融资": "CompEquIss",
    "上市年龄": "FirmAge",
    "年龄动量": "FirmAgeMom",
    "权益变化资产比": "DelEqu",
    "应计占比": "PctAcc",
    "现金流价格波动": "VarCF",
    "毛利资产比": "GP",
    "营收市值比": "SP",
    "净债务市值比": "NetDebtPrice",
    "综合债务融资": "CompositeDebtIssuance",
    "月最大收益": "MaxRet",
    "收益偏度": "ReturnSkew",
    "行业集中度": "Herf",
    "应计资产比": "Accruals",
    "经营利润权益比": "OperProf",
    "外部融资资产比": "XFIN",
    "资产周转变化": "ChAssetTurnover",
    "经营杠杆": "OPLeverage",
    "协偏度": "Coskewness",
    "残差动量": "ResidualMomentum",
    "季节动量": "MomSeason",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _quarterly_pivot(financial: pd.DataFrame, col: str) -> pd.DataFrame | None:
    """长表 → 季报宽表（index=报告期，未做 PIT）。"""
    if col not in financial.columns or "trade_date" not in financial.columns:
        return None
    if "code" not in financial.columns:
        return None
    piv = financial.pivot_table(index="trade_date", columns="code", values=col)
    piv.index = pd.to_datetime(piv.index)
    return piv.sort_index()


def _yoy_growth(level: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    """(x_t - x_{t-k}) / x_{t-k}，分母为 0 → NaN。"""
    lag = level.shift(periods)
    out = (level - lag) / lag.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _finalize(raw: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """对齐日历/股票池后 _normalize；全 NaN 时仍返回同 shape。"""
    panel = raw.reindex(index=prices.index, columns=prices.columns)
    if not panel.notna().any().any():
        return panel.astype(np.float32)
    out = _normalize(panel)
    return out.reindex(index=prices.index, columns=prices.columns)


def _pct_to_ratio(x: pd.DataFrame) -> pd.DataFrame:
    """若中位数 > 1.5 则视为百分数（如 15.2 → 0.152）。"""
    med = float(np.nanmedian(x.to_numpy(dtype=float)))
    if np.isfinite(med) and abs(med) > 1.5:
        return x / 100.0
    return x


def _load_shares_long() -> pd.DataFrame | None:
    path = RAW_DIR / "share_change.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not {"code", "announce_date", "total_shares"} <= set(df.columns):
        return None
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["total_shares"] = pd.to_numeric(df["total_shares"], errors="coerce")
    df = df.dropna(subset=["announce_date", "total_shares"])
    df = df[df["total_shares"] > 0]
    df = df.sort_values(["code", "announce_date"])
    df = df.drop_duplicates(subset=["code", "announce_date"], keep="last")
    return df


def _shares_panel(trade_dates: pd.DatetimeIndex, codes=None) -> pd.DataFrame | None:
    """日频总股本面板（announce_date PIT ffill）。"""
    long = _load_shares_long()
    if long is None or long.empty:
        return None
    from data.compute_market_cap import _build_shares_panel
    panel = _build_shares_panel(long, trade_dates, "total_shares")
    if codes is not None:
        panel = panel.reindex(columns=codes)
    return panel


def _load_listing_dates() -> dict[str, pd.Timestamp]:
    path = UNIVERSE_DIR / "stock_list.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "code" not in df.columns or "list_date" not in df.columns:
        return {}
    out: dict[str, pd.Timestamp] = {}
    for _, row in df[["code", "list_date"]].dropna().iterrows():
        code = str(row["code"]).zfill(6)
        ld = pd.Timestamp(row["list_date"])
        if pd.notna(ld):
            out[code] = ld
    return out


def _firm_age_days(
    prices: pd.DataFrame,
    listing_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """上市天数面板（date × code）；无 list_date 时用首次有效价日期。"""
    ld = listing_dates if listing_dates is not None else _load_listing_dates()
    idx = prices.index
    cols = prices.columns
    age = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    for c in cols:
        key = str(c)
        start = ld.get(key)
        if start is None and key.isdigit():
            start = ld.get(key.zfill(6))
        if start is None or pd.isna(start):
            s = prices[c].dropna()
            if s.empty:
                continue
            start = s.index[0]
        start = pd.Timestamp(start)
        days = (idx - start).days.astype(float)
        age[c] = np.where(days >= 0, days, np.nan)
    return age


def _sales_est_per_share(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    prices_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    用 eps / net_profit_margin 反推每股营收（近似）。

    sales_ps ≈ |eps| / |npm|（npm 为比率）；总额 sales ≈ sales_ps × shares。
    返回 (sales_ps_daily, None) 或失败时 (None, None)。
    """
    if "eps" not in financial.columns or "net_profit_margin" not in financial.columns:
        return None, None
    eps = _pivot_financial(financial, "eps", prices)
    npm = _pivot_financial(financial, "net_profit_margin", prices)
    npm = _pct_to_ratio(npm)
    sales_ps = eps.abs() / npm.abs().replace(0, np.nan)
    sales_ps = sales_ps.replace([np.inf, -np.inf], np.nan)
    # 保留 eps 符号信息无意义；营收取正
    sales_ps = sales_ps.where(eps.notna() & npm.notna())
    return sales_ps.reindex(index=prices.index, columns=prices.columns), None


# ── Batch-1 ──────────────────────────────────────────────────────────────────

def factor_asset_growth(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """资产增长（AssetGrowth）；Sign=-1 → 取负。"""
    if "total_assets" not in financial.columns:
        logger.warning("财务数据无 total_assets，资产增长 跳过")
        return None
    ta = _quarterly_pivot(financial, "total_assets")
    if ta is None or ta.empty:
        return None
    growth = _yoy_growth(ta, 4)
    daily = pit_reindex_ffill(growth, prices.index)
    return _finalize(-daily, prices)


def factor_assets_to_market(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """资产市值比（AM）；Sign=+1。"""
    if "total_assets" not in financial.columns:
        logger.warning("财务数据无 total_assets，资产市值比 跳过")
        return None
    mv = resolve_mcap_panel(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if mv is None:
        logger.warning("无市值面板，资产市值比 跳过")
        return None
    at = _pivot_financial(financial, "total_assets", prices)
    at = at.reindex(index=prices.index, columns=prices.columns)
    mv = mv.reindex(index=prices.index, columns=prices.columns)
    am = at / mv.replace(0, np.nan)
    am = am.replace([np.inf, -np.inf], np.nan)
    return _finalize(am, prices)


def factor_cfp(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    prices_raw: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """经营现金流/市值（cfp）≈ ocf_ps / price；Sign=+1。"""
    if "operating_cashflow" not in financial.columns:
        logger.warning("财务数据无 operating_cashflow，现金流市值比 跳过")
        return None
    px = prices_raw if prices_raw is not None else prices
    ocf = _pivot_financial(financial, "operating_cashflow", px)
    ocf = ocf.reindex(index=px.index, columns=px.columns)
    price = px.reindex(index=ocf.index, columns=ocf.columns)
    cfp = ocf / price.replace(0, np.nan)
    cfp = cfp.replace([np.inf, -np.inf], np.nan)
    return _finalize(cfp, prices)


def factor_cheq(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """账面权益增长（ChEQ）；bvps YoY；Sign=-1 → 取负。"""
    if "bvps" not in financial.columns:
        logger.warning("财务数据无 bvps，权益增长 跳过")
        return None
    bv = _quarterly_pivot(financial, "bvps")
    if bv is None or bv.empty:
        return None
    lag = bv.shift(4)
    cheq = pd.DataFrame(np.nan, index=bv.index, columns=bv.columns)
    valid = (bv > 0) & (lag > 0)
    cheq = cheq.where(~valid, bv / lag)
    daily = pit_reindex_ffill(cheq, prices.index)
    return _finalize(-daily, prices)


def _earnings_consistency_quarterly(eps: pd.DataFrame) -> pd.DataFrame:
    """季报面板 EarningsConsistency（Alwathainani 2009 简化版）。"""
    e0 = eps
    e4 = eps.shift(4)
    e8 = eps.shift(8)
    denom = 0.5 * (e4.abs() + e8.abs())
    egrowth = (e0 - e4) / denom.replace(0, np.nan)
    egrowth = egrowth.replace([np.inf, -np.inf], np.nan)

    lags = [egrowth.shift(4 * k) for k in range(5)]
    stack = np.stack([g.to_numpy(dtype=float) for g in lags], axis=0)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    summed = np.where(valid, stack, 0.0).sum(axis=0)
    cons_vals = np.divide(
        summed, counts, out=np.full_like(summed, np.nan, dtype=float), where=counts > 0,
    )
    cons = pd.DataFrame(cons_vals, index=eps.index, columns=eps.columns)

    ratio = (e0 / e4.replace(0, np.nan)).abs()
    eg4 = egrowth.shift(4)
    sign_flip = (
        ((egrowth > 0) & (eg4 < 0))
        | ((egrowth < 0) & ((eg4 > 0) | eg4.isna()))
    )
    exception = e0.isna() | e4.isna() | (ratio > 6) | sign_flip
    cons = cons.where(~exception)
    return cons


def factor_earnings_consistency(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """盈利一致性；Sign=+1。"""
    if "eps" not in financial.columns:
        logger.warning("财务数据无 eps，盈利一致性 跳过")
        return None
    eps = _quarterly_pivot(financial, "eps")
    if eps is None or eps.empty:
        return None
    cons = _earnings_consistency_quarterly(eps)
    daily = pit_reindex_ffill(cons, prices.index)
    return _finalize(daily, prices)


def _mean_rank_rev_growth_quarterly(rev_growth: pd.DataFrame) -> pd.DataFrame:
    """过去 5 年营收增长秩的加权平均（LSV 1994）。"""
    ranks = rev_growth.rank(axis=1, ascending=False, method="average")
    w = [5, 4, 3, 2, 1]
    shifts = [4, 8, 12, 16, 20]
    num = None
    for wi, sh in zip(w, shifts):
        term = ranks.shift(sh) * wi
        num = term if num is None else num + term
    return num / 15.0


def factor_mean_rank_rev_growth(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """营收增长秩；Sign=+1。"""
    if "revenue_growth" not in financial.columns:
        logger.warning("财务数据无 revenue_growth，营收增长秩 跳过")
        return None
    g = _quarterly_pivot(financial, "revenue_growth")
    if g is None or g.empty:
        return None
    raw = _mean_rank_rev_growth_quarterly(g)
    daily = pit_reindex_ffill(raw, prices.index)
    return _finalize(daily, prices)


# ── Batch-2+ ─────────────────────────────────────────────────────────────────

def _num_earn_increase_quarterly(eps: pd.DataFrame) -> pd.DataFrame:
    """
    NumEarnIncrease：连续 YoY 盈利增长季度数（最多 8）。

    chearn_t = eps_t - eps_{t-4}；缺失视为「正」（对齐源码）；
    nincr = 连续正增长长度，在首次非正处截断。
    """
    chearn = eps - eps.shift(4)
    # is_nonpos: 明确 ≤0（缺失不算 break，对齐源码「缺失当正」）
    is_ok = (chearn > 0) | chearn.isna()
    is_break = chearn <= 0

    nincr = pd.DataFrame(0.0, index=eps.index, columns=eps.columns)
    for k in range(1, 9):
        ok_all = is_ok.copy()
        for j in range(1, k):
            ok_all = ok_all & is_ok.shift(j)
        if k < 8:
            cond = ok_all & is_break.shift(k)
        else:
            # k=8：当前至 lag7 均为 ok，且 lag8 为 break（源码 l24<=0）
            cond = ok_all & is_break.shift(8)
        nincr = nincr.where(~cond.fillna(False), float(k))
    # 全程 8 期皆正且无 break：仍记 8（源码在 l24<=0 才置 8；若全正则可能仍为 0）
    # 补：若 0..7 全 ok 且无任一 break 于 1..8，置 8
    all8 = is_ok.copy()
    for j in range(1, 8):
        all8 = all8 & is_ok.shift(j)
    nincr = nincr.where(~(all8.fillna(False) & (nincr == 0)), 8.0)
    # 无 eps 处保持 NaN
    nincr = nincr.where(eps.notna())
    return nincr


def factor_num_earn_increase(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """盈利连增期数（NumEarnIncrease, Loh/Warachka）；Sign=+1。"""
    if "eps" not in financial.columns:
        logger.warning("财务数据无 eps，盈利连增期数 跳过")
        return None
    eps = _quarterly_pivot(financial, "eps")
    if eps is None or eps.empty:
        return None
    raw = _num_earn_increase_quarterly(eps)
    daily = pit_reindex_ffill(raw, prices.index)
    return _finalize(daily, prices)


def factor_share_iss(
    prices: pd.DataFrame,
    years: int = 1,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    股本扩张 ShareIss1Y / ShareIss5Y（Pontiff/Woodgate）。

    原定义用月频调整股本：1Y=(sh_{t-6m}-sh_{t-18m})/sh_{t-18m}；
    5Y=(sh_{t-5m}-sh_{t-65m})/sh_{t-65m}。
    A 股日频近似：用 ~21 交易日/月，Sign=-1 → 输出取负。
    """
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        logger.warning("无 share_change 股本面板，股本扩张 跳过")
        return None
    sh = sh.reindex(index=prices.index, columns=prices.columns)
    if years == 1:
        # 6m / 18m ≈ 126 / 378 交易日
        near, far = 126, 378
    else:
        # 5m / 65m ≈ 105 / 1365
        near, far = 105, 1365
    lag_near = sh.shift(near)
    lag_far = sh.shift(far)
    iss = (lag_near - lag_far) / lag_far.replace(0, np.nan)
    iss = iss.replace([np.inf, -np.inf], np.nan)
    return _finalize(-iss, prices)


def factor_comp_equ_iss(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    综合股权融资（CompEquIss, Daniel/Titman）：log(ME_t/ME_{t-60m}) - BH_ret_60m。
    Sign=-1 → 取负。
    """
    mv = resolve_mcap_panel(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if mv is None:
        logger.warning("无市值面板，综合股权融资 跳过")
        return None
    mv = mv.reindex(index=prices.index, columns=prices.columns)
    lag = 60 * 21  # ~60 个月交易日
    mv_lag = mv.shift(lag)
    log_me_g = np.log(mv / mv_lag.replace(0, np.nan))
    if clean_ret is not None:
        # 累计收益：(∏(1+r))-1 over lag days；用 logsumexpm1 近似
        r = clean_ret.reindex(index=prices.index, columns=prices.columns)
        # rolling product of (1+r) over lag — memory heavy; use cumsum of log1p
        log1p = np.log1p(r.fillna(0.0).clip(lower=-0.999999))
        # mask: only where clean_ret was finite
        valid = r.notna().astype(float)
        cum_log = log1p.cumsum()
        cum_n = valid.cumsum()
        bh_log = cum_log - cum_log.shift(lag)
        n_obs = cum_n - cum_n.shift(lag)
        bh = np.expm1(bh_log)
        bh = bh.where(n_obs >= lag * 0.8)
    else:
        # 退化：用价格比
        bh = prices / prices.shift(lag) - 1.0
    cei = log_me_g - bh
    cei = cei.replace([np.inf, -np.inf], np.nan)
    return _finalize(-cei, prices)


def factor_firm_age(
    prices: pd.DataFrame,
    listing_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame | None:
    """上市年龄（FirmAge）；天数；Sign=-1 → 年轻高分。"""
    age = _firm_age_days(prices, listing_dates)
    # 至少上市 1 天；用 log1p 缓和极端
    raw = np.log1p(age)
    return _finalize(-raw, prices)


def factor_firm_age_mom(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame | None:
    """
    年龄动量（FirmAgeMom, Zhang 2006）：最年轻五分位的 6 个月动量。
    Sign=+1；非年轻组置 NaN。
    """
    if clean_ret is None:
        logger.warning("无 clean_ret，年龄动量 跳过")
        return None
    age = _firm_age_days(prices, listing_dates)
    # 至少 ~12 个月历史
    age = age.where(age >= 252)
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    # 6 个月动量 ≈ t-21..t-126（跳过最近 1 月，对齐 l1..l5 月）
    log1p = np.log1p(r.fillna(0.0).clip(lower=-0.999999))
    valid = r.notna().astype(float)
    cum = log1p.cumsum()
    # product from lag 21 to lag 126: cum.shift(21) - cum.shift(126)
    mom = np.expm1(cum.shift(21) - cum.shift(126))
    n_obs = valid.cumsum().shift(21) - valid.cumsum().shift(126)
    mom = mom.where(n_obs >= 80)

    # 截面年龄五分位，仅保留最年轻组（q=1）
    def _young_mask(row: pd.Series) -> pd.Series:
        s = row.dropna()
        if len(s) < 30:
            return pd.Series(False, index=row.index)
        try:
            q = pd.qcut(s, 5, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(False, index=row.index)
        m = pd.Series(False, index=row.index)
        m.loc[q.index] = (q == 0)
        return m

    young = age.apply(_young_mask, axis=1)
    raw = mom.where(young)
    return _finalize(raw, prices)


def factor_del_equ(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    权益变化/资产（DelEqu, Richardson et al.）。

    ceq ≈ bvps × total_shares；Δceq / avg(AT)；Sign=-1 → 取负。
    无股本时退化为 bvps YoY 变化 / 1（弱代理，仍注册）。
    """
    if "bvps" not in financial.columns or "total_assets" not in financial.columns:
        logger.warning("财务数据缺 bvps/total_assets，权益变化资产比 跳过")
        return None
    bv = _quarterly_pivot(financial, "bvps")
    at = _quarterly_pivot(financial, "total_assets")
    if bv is None or at is None:
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is not None and not sh.empty:
        # 把日频股本对齐到季报日（PIT：报告期当日已知的最新股本）
        sh_q = sh.reindex(bv.index.union(sh.index)).sort_index().ffill()
        sh_q = sh_q.reindex(bv.index).reindex(columns=bv.columns)
        ceq = bv * sh_q
    else:
        # 无股本：用 bvps 水平差 / AT 不可比；改用 (bv_t-bv_{t-4})/avg 相对变化弱代理
        logger.warning("DelEqu：无股本面板，用 bvps 差分 / AT 弱代理")
        ceq = bv
    lag_ceq = ceq.shift(4)
    lag_at = at.shift(4)
    avg_at = 0.5 * (at + lag_at)
    delequ = (ceq - lag_ceq) / avg_at.replace(0, np.nan)
    delequ = delequ.replace([np.inf, -np.inf], np.nan)
    daily = pit_reindex_ffill(delequ, prices.index)
    return _finalize(-daily, prices)


def factor_pct_acc(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    应计占比（PctAcc, Hafzalla et al.）：(eps - ocf_ps) / |eps|。

    缺资产负债表 fallback；Sign=-1 → 取负（低应计高分）。
    同时作为破损 ``质量_应计项目``（依赖缺失的 net_profit）的替代。
    """
    if "eps" not in financial.columns or "operating_cashflow" not in financial.columns:
        logger.warning("财务数据缺 eps/operating_cashflow，应计占比 跳过")
        return None
    eps = _pivot_financial(financial, "eps", prices)
    ocf = _pivot_financial(financial, "operating_cashflow", prices)
    denom = eps.abs().where(eps.abs() >= 1e-6, 0.01)
    pct = (eps - ocf) / denom
    pct = pct.replace([np.inf, -np.inf], np.nan)
    return _finalize(-pct, prices)


def factor_var_cf(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    prices_raw: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    现金流/价格波动（VarCF）：ocf_ps/price 过去 ~60 月滚动方差（min 24 月）。
    Sign=-1 → 取负。
    """
    if "operating_cashflow" not in financial.columns:
        logger.warning("财务数据无 operating_cashflow，现金流价格波动 跳过")
        return None
    px = prices_raw if prices_raw is not None else prices
    ocf = _pivot_financial(financial, "operating_cashflow", px)
    cfp = ocf / px.reindex_like(ocf).replace(0, np.nan)
    cfp = cfp.replace([np.inf, -np.inf], np.nan)
    # 60 月 ≈ 1260 交易日；min 24 月 ≈ 504
    var = cfp.rolling(1260, min_periods=504).var()
    return _finalize(-var, prices)


def factor_gp_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    毛利资产比（GP）**近似**：GP ≈ gpm × sales_est / AT，
    sales_est = |eps|/|npm| × shares。

    缺营收/COGS 总额时的重建；非 Novy-Marx 原定义。Sign=+1。
    """
    need = {"gross_profit_margin", "eps", "net_profit_margin", "total_assets"}
    if not need <= set(financial.columns):
        logger.warning(f"财务数据缺 {need - set(financial.columns)}，毛利资产比 跳过")
        return None
    sales_ps, _ = _sales_est_per_share(financial, prices)
    if sales_ps is None:
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        logger.warning("毛利资产比：无股本，跳过")
        return None
    sh = sh.reindex(index=prices.index, columns=prices.columns)
    gpm = _pct_to_ratio(_pivot_financial(financial, "gross_profit_margin", prices))
    at = _pivot_financial(financial, "total_assets", prices)
    sales = sales_ps * sh
    gp = gpm * sales / at.replace(0, np.nan)
    gp = gp.replace([np.inf, -np.inf], np.nan)
    return _finalize(gp, prices)


def factor_sp_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    prices_raw: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    营收市值比（SP）**近似**：SP ≈ (|eps|/|npm|) / price = sales_ps / price。

    缺营收总额时的重建。Sign=+1。
    """
    if not {"eps", "net_profit_margin"} <= set(financial.columns):
        logger.warning("财务数据缺 eps/net_profit_margin，营收市值比 跳过")
        return None
    sales_ps, _ = _sales_est_per_share(financial, prices, prices_raw)
    if sales_ps is None:
        return None
    px = prices_raw if prices_raw is not None else prices
    px = px.reindex(index=prices.index, columns=prices.columns)
    sp = sales_ps / px.replace(0, np.nan)
    sp = sp.replace([np.inf, -np.inf], np.nan)
    return _finalize(sp, prices)


def factor_net_debt_price_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    净债务/市值（NetDebtPrice）**近似**：debt_ratio × AT / ME（不减现金、不含优先股）。

    Sign=-1 → 取负。非 Penman 原定义。
    """
    if not {"debt_ratio", "total_assets"} <= set(financial.columns):
        logger.warning("财务数据缺 debt_ratio/total_assets，净债务市值比 跳过")
        return None
    mv = resolve_mcap_panel(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if mv is None:
        return None
    dr = _pct_to_ratio(_pivot_financial(financial, "debt_ratio", prices))
    at = _pivot_financial(financial, "total_assets", prices)
    debt = dr * at
    ndp = debt / mv.reindex_like(debt).replace(0, np.nan)
    ndp = ndp.replace([np.inf, -np.inf], np.nan)
    return _finalize(-ndp, prices)


def factor_composite_debt_issuance_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    综合债务融资（CompositeDebtIssuance）**近似**：
    log( (debt_ratio×AT)_t / (debt_ratio×AT)_{t-20q} )。

    Sign=-1 → 取负。非 Lyandres 原定义（缺 DLTT+DLC）。
    """
    if not {"debt_ratio", "total_assets"} <= set(financial.columns):
        logger.warning("财务数据缺 debt_ratio/total_assets，综合债务融资 跳过")
        return None
    dr = _quarterly_pivot(financial, "debt_ratio")
    at = _quarterly_pivot(financial, "total_assets")
    if dr is None or at is None:
        return None
    dr = _pct_to_ratio(dr)
    debt = dr * at
    lag = debt.shift(20)  # 5 年 × 4 季
    cdi = np.log(debt / lag.replace(0, np.nan))
    cdi = cdi.replace([np.inf, -np.inf], np.nan)
    daily = pit_reindex_ffill(cdi, prices.index)
    return _finalize(-daily, prices)


def factor_max_ret(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """月最大收益（MaxRet）：过去 21 日 clean_ret 最大值；Sign=-1 → 取负。"""
    if clean_ret is None:
        logger.warning("无 clean_ret，月最大收益 跳过")
        return None
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    mx = r.rolling(21, min_periods=10).max()
    return _finalize(-mx, prices)


def factor_return_skew(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """收益偏度（ReturnSkew）：过去 21 日 skew（min 15）；Sign=-1 → 取负。"""
    if clean_ret is None:
        logger.warning("无 clean_ret，收益偏度 跳过")
        return None
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    sk = r.rolling(21, min_periods=15).skew()
    return _finalize(-sk, prices)


def factor_herf_mcap(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
    industry_map: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    行业集中度（Herf）**近似**：用流通市值份额算行业 HHI，再 36 月滚动均值。

    原定义用销售额；此处用 mcap。Sign=-1 → 取负。
    """
    mv = resolve_mcap_panel(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if mv is None:
        logger.warning("无市值，行业集中度 跳过")
        return None
    mv = mv.reindex(index=prices.index, columns=prices.columns)

    # 行业：优先静态 map；若有 PIT panel 在调用方传入 map 即可
    ind = industry_map
    if ind is None:
        try:
            from factors.factor_alpha import load_industry_map
            ind = load_industry_map()
        except Exception:
            ind = None
    if ind is None or ind.empty:
        logger.warning("无行业映射，行业集中度 跳过")
        return None

    if "sw_l2" in ind.columns:
        sw = ind["sw_l2"]
    else:
        # 可能 index=code
        sw = ind.iloc[:, 0] if ind.shape[1] else None
        if sw is None:
            return None
    sw = sw.reindex(mv.columns)

    # 按月抽样：每月最后交易日算行业 HHI（向量化），再 36 月滚动均值后 ffill
    ym = mv.index.to_period("M")
    month_ends = mv.groupby(ym).tail(1).index
    mv_m = mv.loc[month_ends]
    # industry code → int id for groupby
    sw_aligned = sw.reindex(mv_m.columns)
    valid_ind = sw_aligned.notna()
    ind_codes = sw_aligned.astype("category")
    # 对每个月：按行业汇总市值 → 份额²和
    hhi_m = pd.DataFrame(np.nan, index=month_ends, columns=mv_m.columns, dtype=float)
    for dt in month_ends:
        row = mv_m.loc[dt]
        ok = row.notna() & (row > 0) & valid_ind
        if ok.sum() < 2:
            continue
        r = row[ok]
        g = ind_codes[ok]
        tot = r.groupby(g, observed=True).transform("sum")
        share = r / tot.replace(0, np.nan)
        hhi_by_stock = (share ** 2).groupby(g, observed=True).transform("sum")
        hhi_m.loc[dt, hhi_by_stock.index] = hhi_by_stock.to_numpy()

    hhi_roll = hhi_m.rolling(36, min_periods=12).mean()
    daily = hhi_roll.reindex(prices.index).ffill()
    return _finalize(-daily, prices)


# ── Batch-3（现有字段 / 价量；严格 BS 版跳过）────────────────────────────────

def _sales_total_daily(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """sales ≈ |eps|/|npm| × shares（与 GP/SP 同源近似）。"""
    sales_ps, _ = _sales_est_per_share(financial, prices)
    if sales_ps is None:
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        return None
    sh = sh.reindex(index=prices.index, columns=prices.columns)
    return (sales_ps * sh).replace([np.inf, -np.inf], np.nan)


def factor_accruals_cf(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    应计资产比（Accruals）**CF 近似**：(eps−ocf_ps)×shares / avg(AT)。

    对齐 AbnormalAccruals 文中 CF 应计定义；**非** Sloan 资产负债表版
    （缺 ACT/CHE/LCT/DLC/DP）。Sign=-1 → 取负。
    """
    need = {"eps", "operating_cashflow", "total_assets"}
    if not need <= set(financial.columns):
        logger.warning(f"财务数据缺 {need - set(financial.columns)}，应计资产比 跳过")
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        logger.warning("应计资产比：无股本面板，跳过")
        return None
    eps_q = _quarterly_pivot(financial, "eps")
    ocf_q = _quarterly_pivot(financial, "operating_cashflow")
    at_q = _quarterly_pivot(financial, "total_assets")
    if eps_q is None or ocf_q is None or at_q is None:
        return None
    sh_q = sh.reindex(eps_q.index.union(sh.index)).sort_index().ffill()
    sh_q = sh_q.reindex(eps_q.index).reindex(columns=eps_q.columns)
    accr = (eps_q - ocf_q) * sh_q
    avg_at = 0.5 * (at_q + at_q.shift(4))
    raw = accr / avg_at.replace(0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    daily = pit_reindex_ffill(raw, prices.index)
    return _finalize(-daily, prices)


def factor_oper_prof_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    经营利润权益比（OperProf）**近似**：gpm×sales_est / (bvps×shares)。

    缺 COGS/SGA/利息明细；用毛利/账面权益代理 FF 经营盈利。
    对齐原文剔除市值最小三分位。Sign=+1。
    """
    need = {"gross_profit_margin", "eps", "net_profit_margin", "bvps"}
    if not need <= set(financial.columns):
        logger.warning(f"财务数据缺 {need - set(financial.columns)}，经营利润权益比 跳过")
        return None
    sales = _sales_total_daily(financial, prices, shares=shares)
    if sales is None:
        logger.warning("经营利润权益比：无法估计营收，跳过")
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        return None
    sh = sh.reindex(index=prices.index, columns=prices.columns)
    gpm = _pct_to_ratio(_pivot_financial(financial, "gross_profit_margin", prices))
    bvps = _pivot_financial(financial, "bvps", prices)
    ceq = bvps * sh
    op = (gpm * sales) / ceq.replace(0, np.nan)
    op = op.replace([np.inf, -np.inf], np.nan)
    # 剔除最小市值三分位（FF 样本习惯；按月抽样再 ffill，避免日频 apply）
    mv = resolve_mcap_panel(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if mv is not None:
        mv = mv.reindex(index=prices.index, columns=prices.columns)
        ym = mv.index.to_period("M")
        month_ends = mv.groupby(ym).tail(1).index
        small_m = pd.DataFrame(False, index=month_ends, columns=mv.columns)
        for dt in month_ends:
            row = mv.loc[dt]
            s = row.dropna()
            if len(s) < 30:
                continue
            try:
                q = pd.qcut(s, 3, labels=False, duplicates="drop")
            except ValueError:
                continue
            small_m.loc[dt, q.index] = (q == 0).to_numpy()
        small = small_m.reindex(prices.index).ffill().fillna(False).astype(bool)
        op = op.where(~small)
    return _finalize(op, prices)


def factor_xfin_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
    prices_raw: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    外部融资资产比（XFIN）**近似**：
    (Δshares×price + Δ(debt_ratio×AT)) / avg(AT)。

    缺 CF 表 sstk/prstkc/dltis 等；用股本与债务余额变化代理。Sign=-1 → 取负。
    """
    if not {"debt_ratio", "total_assets"} <= set(financial.columns):
        logger.warning("财务数据缺 debt_ratio/total_assets，外部融资资产比 跳过")
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        logger.warning("外部融资资产比：无股本面板，跳过")
        return None
    at_q = _quarterly_pivot(financial, "total_assets")
    dr_q = _quarterly_pivot(financial, "debt_ratio")
    if at_q is None or dr_q is None:
        return None
    dr_q = _pct_to_ratio(dr_q)
    debt_q = dr_q * at_q
    sh_q = sh.reindex(at_q.index.union(sh.index)).sort_index().ffill()
    sh_q = sh_q.reindex(at_q.index).reindex(columns=at_q.columns)
    px = prices_raw if prices_raw is not None else prices
    px_q = px.reindex(at_q.index.union(px.index)).sort_index().ffill()
    px_q = px_q.reindex(at_q.index).reindex(columns=at_q.columns)
    d_sh = sh_q - sh_q.shift(4)
    d_debt = debt_q - debt_q.shift(4)
    avg_at = 0.5 * (at_q + at_q.shift(4))
    xfin = (d_sh * px_q + d_debt) / avg_at.replace(0, np.nan)
    xfin = xfin.replace([np.inf, -np.inf], np.nan)
    # 与 Net*Finance 一致：极端值置空
    xfin = xfin.where(xfin.abs() <= 1.0)
    daily = pit_reindex_ffill(xfin, prices.index)
    return _finalize(-daily, prices)


def factor_ch_asset_turnover_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    资产周转变化（ChAssetTurnover）**近似**：Δ(sales_est/AT)。

    原定义用经营资产分母；此处用 total_assets。Sign=+1。
    """
    need = {"eps", "net_profit_margin", "total_assets"}
    if not need <= set(financial.columns):
        logger.warning(f"财务数据缺 {need - set(financial.columns)}，资产周转变化 跳过")
        return None
    sh = shares if shares is not None else _shares_panel(prices.index, prices.columns)
    if sh is None or sh.empty:
        logger.warning("资产周转变化：无股本，跳过")
        return None
    eps_q = _quarterly_pivot(financial, "eps")
    npm_q = _quarterly_pivot(financial, "net_profit_margin")
    at_q = _quarterly_pivot(financial, "total_assets")
    if eps_q is None or npm_q is None or at_q is None:
        return None
    npm_q = _pct_to_ratio(npm_q)
    sh_q = sh.reindex(eps_q.index.union(sh.index)).sort_index().ffill()
    sh_q = sh_q.reindex(eps_q.index).reindex(columns=eps_q.columns)
    sales_q = (eps_q.abs() / npm_q.abs().replace(0, np.nan)) * sh_q
    ato = sales_q / at_q.replace(0, np.nan)
    ato = ato.replace([np.inf, -np.inf], np.nan)
    ato = ato.where(ato >= 0)
    ch = ato - ato.shift(4)
    daily = pit_reindex_ffill(ch, prices.index)
    return _finalize(daily, prices)


def factor_op_leverage_approx(
    financial: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    经营杠杆（OPLeverage）**近似**：(1−gpm)×sales_est / AT。

    原定义 (COGS+SGA)/AT；缺 SGA，仅用估计 COGS。Sign=+1。
    """
    need = {"gross_profit_margin", "eps", "net_profit_margin", "total_assets"}
    if not need <= set(financial.columns):
        logger.warning(f"财务数据缺 {need - set(financial.columns)}，经营杠杆 跳过")
        return None
    sales = _sales_total_daily(financial, prices, shares=shares)
    if sales is None:
        return None
    gpm = _pct_to_ratio(_pivot_financial(financial, "gross_profit_margin", prices))
    at = _pivot_financial(financial, "total_assets", prices)
    cogs = (1.0 - gpm) * sales
    opl = cogs / at.replace(0, np.nan)
    opl = opl.replace([np.inf, -np.inf], np.nan)
    return _finalize(opl, prices)


def _align_market_ret(
    clean_ret: pd.DataFrame,
    market_prices: pd.DataFrame | pd.Series | None,
) -> pd.Series | None:
    from factors.barra_risk import market_return
    mkt = market_return(market_prices)
    if mkt is None:
        return None
    return mkt.reindex(clean_ret.index).astype(float)


def factor_coskewness(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
    market_prices: pd.DataFrame | pd.Series | None = None,
    window: int = 252,
    min_periods: int = 120,
) -> pd.DataFrame | None:
    """
    协偏度（Coskewness）：日频 CoskewACX 式
    E[r̃ m̃²] / (sd(r̃)·var(m̃))，过去 ``window`` 交易日。

    Sign=-1 → 取负。需 ``clean_ret`` + 市场指数（中证全指）。
    """
    if clean_ret is None:
        logger.warning("无 clean_ret，协偏度 跳过")
        return None
    if market_prices is None:
        logger.warning("无 market_prices，协偏度 跳过")
        return None
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    m = _align_market_ret(r, market_prices)
    if m is None or m.notna().sum() < min_periods:
        logger.warning("市场收益无效，协偏度 跳过")
        return None
    r_mu = r.rolling(window, min_periods=min_periods).mean()
    m_mu = m.rolling(window, min_periods=min_periods).mean()
    rt = r - r_mu
    mt = m - m_mu
    mt2 = mt ** 2
    # E[rt * mt2], sd(rt), E[mt2]≈var(m) when demeaned in-window
    cross = rt.mul(mt2, axis=0)
    e_cross = cross.rolling(window, min_periods=min_periods).mean()
    sd_r = rt.rolling(window, min_periods=min_periods).std()
    var_m = mt2.rolling(window, min_periods=min_periods).mean()
    cosk = e_cross / (sd_r.replace(0, np.nan).mul(var_m.replace(0, np.nan), axis=0))
    cosk = cosk.replace([np.inf, -np.inf], np.nan)
    return _finalize(-cosk, prices)


def factor_residual_momentum(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
    market_prices: pd.DataFrame | pd.Series | None = None,
    beta_window: int = 756,
    mom_window: int = 231,
    skip: int = 21,
    min_beta: int = 504,
    min_mom: int = 120,
) -> pd.DataFrame | None:
    """
    残差动量（ResidualMomentum）**CAPM 近似**：
    滚动市场回归残差 → 跳过近 ``skip`` 日后 ``mom_window`` 日均值/标准差。

    原定义用月频 FF3；A 股无免费 SMB/HML 面板，退化为单因子市场残差。Sign=+1。
    """
    if clean_ret is None:
        logger.warning("无 clean_ret，残差动量 跳过")
        return None
    if market_prices is None:
        logger.warning("无 market_prices，残差动量 跳过")
        return None
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    m = _align_market_ret(r, market_prices)
    if m is None or m.notna().sum() < min_beta:
        logger.warning("市场收益无效，残差动量 跳过")
        return None
    # 滚动 beta = cov(r,m)/var(m)
    # DataFrame.rolling.cov(Series) 在较新 pandas 可用；否则逐列退化
    var_m = m.rolling(beta_window, min_periods=min_beta).var()
    try:
        cov_rm = r.rolling(beta_window, min_periods=min_beta).cov(m)
    except Exception:
        cov_rm = pd.DataFrame(
            {c: r[c].rolling(beta_window, min_periods=min_beta).cov(m) for c in r.columns},
            index=r.index,
        )
    beta = cov_rm.div(var_m.replace(0, np.nan), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    lagged = resid.shift(skip)
    mu = lagged.rolling(mom_window, min_periods=min_mom).mean()
    sd = lagged.rolling(mom_window, min_periods=min_mom).std()
    raw = mu / sd.replace(0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return _finalize(raw, prices)


def factor_mom_season(
    prices: pd.DataFrame,
    clean_ret: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    季节动量（MomSeason, Heston/Sadka）：同年份月收益在滞后 23/35/47/59 月的均值。

    日频实现：先合成月收益，算信号后 ffill 到交易日。Sign=+1。
    """
    if clean_ret is None:
        logger.warning("无 clean_ret，季节动量 跳过")
        return None
    r = clean_ret.reindex(index=prices.index, columns=prices.columns)
    log1p = np.log1p(r.fillna(0.0).clip(lower=-0.999999))
    valid = r.notna().astype(float)
    ym = pd.Series(r.index.to_period("M"), index=r.index)
    sum_log = log1p.groupby(ym.values).sum()
    n_obs = valid.groupby(ym.values).sum()
    ret_m = np.expm1(sum_log)
    ret_m = ret_m.where(n_obs >= 10)
    # Period → 该月最后一个交易日
    last_by_period = ym.groupby(ym.values).apply(lambda s: s.index[-1])
    ret_m.index = pd.DatetimeIndex([last_by_period.loc[p] for p in ret_m.index])
    ret_m = ret_m.sort_index()
    lags = [23, 35, 47, 59]
    parts = [ret_m.shift(k) for k in lags]
    stack = np.stack([p.to_numpy(dtype=float) for p in parts], axis=0)
    valid_l = np.isfinite(stack)
    counts = valid_l.sum(axis=0)
    summed = np.where(valid_l, stack, 0.0).sum(axis=0)
    mean = np.divide(
        summed, counts, out=np.full_like(summed, np.nan, dtype=float), where=counts > 0,
    )
    mom = pd.DataFrame(mean, index=ret_m.index, columns=ret_m.columns)
    daily = mom.reindex(prices.index).ffill()
    return _finalize(daily, prices)


# ── 批量入口 ─────────────────────────────────────────────────────────────────

def get_opensource_ap_factors(
    prices: pd.DataFrame,
    financial: pd.DataFrame | None = None,
    prices_raw: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    market_prices: pd.DataFrame | pd.Series | None = None,
    industry_map: pd.DataFrame | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    factor_names: list | set | None = None,
) -> dict[str, pd.DataFrame]:
    """计算 OpenSourceAP Batch-1/2/3 子集；返回 {name: panel}。"""
    want = (
        set(factor_names) if factor_names is not None
        else set(OPENSOURCE_AP_FACTOR_NAMES)
    )
    want &= OPENSOURCE_AP_FACTOR_NAMES
    if not want:
        return {}

    out: dict[str, pd.DataFrame] = {}

    def _put(name: str, panel: pd.DataFrame | None) -> None:
        if panel is not None and name in want:
            out[name] = panel

    # 惰性股本面板（多因子共享）
    shares: pd.DataFrame | None = None
    need_shares = want & {
        "一年股本扩张", "五年股本扩张", "权益变化资产比", "毛利资产比",
        "应计资产比", "经营利润权益比", "外部融资资产比",
        "资产周转变化", "经营杠杆",
    }
    if need_shares:
        shares = _shares_panel(prices.index, prices.columns)

    if financial is not None and not financial.empty:
        if "资产增长" in want:
            _put("资产增长", factor_asset_growth(financial, prices))
        if "资产市值比" in want:
            _put(
                "资产市值比",
                factor_assets_to_market(
                    financial, prices, circ_mv=circ_mv, total_mv=total_mv,
                ),
            )
        if "现金流市值比" in want:
            _put(
                "现金流市值比",
                factor_cfp(financial, prices, prices_raw=prices_raw),
            )
        if "权益增长" in want:
            _put("权益增长", factor_cheq(financial, prices))
        if "盈利一致性" in want:
            _put("盈利一致性", factor_earnings_consistency(financial, prices))
        if "营收增长秩" in want:
            _put("营收增长秩", factor_mean_rank_rev_growth(financial, prices))
        if "盈利连增期数" in want:
            _put("盈利连增期数", factor_num_earn_increase(financial, prices))
        if "权益变化资产比" in want:
            _put("权益变化资产比", factor_del_equ(financial, prices, shares=shares))
        if "应计占比" in want:
            _put("应计占比", factor_pct_acc(financial, prices))
        if "现金流价格波动" in want:
            _put(
                "现金流价格波动",
                factor_var_cf(financial, prices, prices_raw=prices_raw),
            )
        if "毛利资产比" in want:
            _put("毛利资产比", factor_gp_approx(financial, prices, shares=shares))
        if "营收市值比" in want:
            _put(
                "营收市值比",
                factor_sp_approx(financial, prices, prices_raw=prices_raw),
            )
        if "净债务市值比" in want:
            _put(
                "净债务市值比",
                factor_net_debt_price_approx(
                    financial, prices, circ_mv=circ_mv, total_mv=total_mv,
                ),
            )
        if "综合债务融资" in want:
            _put(
                "综合债务融资",
                factor_composite_debt_issuance_approx(financial, prices),
            )
        # Batch-3 accounting（现有字段近似）
        if "应计资产比" in want:
            _put("应计资产比", factor_accruals_cf(financial, prices, shares=shares))
        if "经营利润权益比" in want:
            _put(
                "经营利润权益比",
                factor_oper_prof_approx(
                    financial, prices, shares=shares,
                    circ_mv=circ_mv, total_mv=total_mv,
                ),
            )
        if "外部融资资产比" in want:
            _put(
                "外部融资资产比",
                factor_xfin_approx(
                    financial, prices, shares=shares, prices_raw=prices_raw,
                ),
            )
        if "资产周转变化" in want:
            _put(
                "资产周转变化",
                factor_ch_asset_turnover_approx(financial, prices, shares=shares),
            )
        if "经营杠杆" in want:
            _put("经营杠杆", factor_op_leverage_approx(financial, prices, shares=shares))

    # 价量 / 股本（可不依赖 financial）
    if "一年股本扩张" in want:
        _put("一年股本扩张", factor_share_iss(prices, years=1, shares=shares))
    if "五年股本扩张" in want:
        _put("五年股本扩张", factor_share_iss(prices, years=5, shares=shares))
    if "综合股权融资" in want:
        _put(
            "综合股权融资",
            factor_comp_equ_iss(
                prices, clean_ret=clean_ret, circ_mv=circ_mv,
                total_mv=total_mv, financial=financial,
            ),
        )
    if "上市年龄" in want:
        _put("上市年龄", factor_firm_age(prices, listing_dates=listing_dates))
    if "年龄动量" in want:
        _put(
            "年龄动量",
            factor_firm_age_mom(
                prices, clean_ret=clean_ret, listing_dates=listing_dates,
            ),
        )
    if "月最大收益" in want:
        _put("月最大收益", factor_max_ret(prices, clean_ret=clean_ret))
    if "收益偏度" in want:
        _put("收益偏度", factor_return_skew(prices, clean_ret=clean_ret))
    if "行业集中度" in want:
        _put(
            "行业集中度",
            factor_herf_mcap(
                prices, circ_mv=circ_mv, total_mv=total_mv,
                financial=financial, industry_map=industry_map,
            ),
        )
    if "协偏度" in want:
        _put(
            "协偏度",
            factor_coskewness(
                prices, clean_ret=clean_ret, market_prices=market_prices,
            ),
        )
    if "残差动量" in want:
        _put(
            "残差动量",
            factor_residual_momentum(
                prices, clean_ret=clean_ret, market_prices=market_prices,
            ),
        )
    if "季节动量" in want:
        _put("季节动量", factor_mom_season(prices, clean_ret=clean_ret))

    if out:
        logger.info(
            f"OpenSourceAP 就绪: {len(out)} 个 → {sorted(out)} "
            f"({ {k: OPENSOURCE_AP_ACRONYM[k] for k in out} })"
        )
    return out
