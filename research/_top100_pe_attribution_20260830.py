"""Top100 PE 正负分组收益归因（PIT as-of 信号日）。

数据：
  - 全 WF 回测: results/xgb_h5_sizeind_w156_nob_wf_20260830/
  - holdings_top100_*.csv (信号日 + pipe 分隔 codes, 等权 1/100)
  - close_hfq.parquet (个股期间收益)
  - data/raw/pe_ttm.parquet (东财日频 pe_ttm, 当日可得, as-of 信号日)
  - data/raw/circ_mv.parquet (流通市值, 用于辅助对照)
  - data/raw/industry_map_panel.parquet (PIT 行业, 用于辅助对照)

口径：
  - 持仓等权 1/100
  - 个股期间收益 = close_hfq[next_signal_date] / close_hfq[signal_date] - 1
  - PE 分组按信号日 as-of pe_ttm：
      负 PE (亏损股): pe_ttm < 0
      正 PE (盈利股): pe_ttm > 0
      NaN/0: 单列 "未知/零"
  - 贡献 = (1/100) * 个股期间收益
  - 年化贡献近似 = 期间贡献均值 * 52 (周频)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "xgb_h5_sizeind_w156_nob_wf_20260830"
HOLD_CSV = RES / "holdings_top100_xgb_h5_w156_p_sparse_rt4.csv"
NAV_CSV = RES / "backtest_xgb_h5_w156_p_sparse_rt4_nav.csv"
CLOSE_HFQ = ROOT / "data" / "raw" / "close_hfq.parquet"
PE_TTM = ROOT / "data" / "raw" / "pe_ttm.parquet"
CIRC_MV = ROOT / "data" / "raw" / "circ_mv.parquet"
LIVE_CSV = RES / "candidates_20260828.csv"


def load_holdings():
    df = pd.read_csv(HOLD_CSV)
    df.columns = ["signal_date", "codes", "n"]
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["codes"] = df["codes"].str.split(" | ", regex=False)
    return df[["signal_date", "codes"]].sort_values("signal_date").reset_index(drop=True)


def nearest_idx(idx, t):
    return idx[idx.get_indexer([t], method="nearest")[0]]


def main():
    print("=" * 90)
    print("Top100 PE 正负分组收益归因 — xgb_h5_sizeind_w156_nob_wf_20260830")
    print("=" * 90)

    holdings = load_holdings()
    print(f"\n持仓期数: {len(holdings)}  起止: {holdings['signal_date'].iloc[0].date()} -> {holdings['signal_date'].iloc[-1].date()}")

    close = pd.read_parquet(CLOSE_HFQ)
    pe = pd.read_parquet(PE_TTM)
    circ_mv = pd.read_parquet(CIRC_MV)
    close.columns = close.columns.astype(str)
    pe.columns = pe.columns.astype(str)
    circ_mv.columns = circ_mv.columns.astype(str)

    # PE 覆盖检查
    pe_first = pe.index.min()
    pe_last = pe.index.max()
    print(f"pe_ttm 覆盖: {pe_first.date()} -> {pe_last.date()}  shape={pe.shape}")
    sig_dates = holdings["signal_date"]
    print(f"信号日范围: {sig_dates.min().date()} -> {sig_dates.max().date()}")
    # 检查信号日是否在 pe 覆盖内
    in_cov = ((sig_dates >= pe_first) & (sig_dates <= pe_last)).sum()
    print(f"信号日落在 pe 覆盖内: {in_cov}/{len(sig_dates)}")

    # NAV 对账
    nav = pd.read_csv(NAV_CSV)
    nav.columns = ["date"] + list(nav.columns[1:])
    nav["date"] = pd.to_datetime(nav["date"])

    records = []
    pe_missing_dates = []
    for i in range(len(holdings) - 1):
        t = holdings.loc[i, "signal_date"]
        t_next = holdings.loc[i + 1, "signal_date"]
        codes = holdings.loc[i, "codes"]

        # 个股期间收益 (close-to-close)
        t_c = nearest_idx(close.index, t)
        t_n = nearest_idx(close.index, t_next)
        px0 = close.loc[t_c].reindex(codes)
        px1 = close.loc[t_n].reindex(codes)
        ret = px1 / px0 - 1.0

        # PE as-of 信号日: 用 .asof 取 <= t 的最后一个有效值 (PIT, 不前视)
        # pe 是日频, 信号日当天 pe 已可得 (东财当日计算), 所以 asof(t) 即可
        pe_row = pe.reindex(index=[t], method="nearest").iloc[0] if t in pe.index or pe.index.get_indexer([t], method="nearest")[0] >= 0 else None
        # 更稳妥: 取 pe.loc[:t] 最后一行
        pe_sub = pe.loc[:t]
        if len(pe_sub) == 0:
            pe_vals = pd.Series(np.nan, index=codes)
            pe_missing_dates.append(t)
        else:
            pe_vals = pe_sub.iloc[-1].reindex(codes)

        # 市值 (辅助)
        t_m = nearest_idx(circ_mv.index, t)
        mv = circ_mv.loc[t_m].reindex(codes) / 1e8

        for code in codes:
            records.append({
                "signal_date": t,
                "year": t.year,
                "code": code,
                "ret": ret.get(code, np.nan),
                "pe_ttm": pe_vals.get(code, np.nan),
                "circ_mv_yi": mv.get(code, np.nan),
            })

    panel = pd.DataFrame(records)
    panel["weight"] = 1.0 / 100.0
    panel["contrib"] = panel["weight"] * panel["ret"]

    # PE 分组
    def pe_group(v):
        if pd.isna(v):
            return "未知/NaN"
        if v == 0:
            return "零PE"
        if v < 0:
            return "负PE(亏损)"
        return "正PE(盈利)"
    panel["pe_grp"] = panel["pe_ttm"].apply(pe_group)

    # 对账
    n_periods = len(holdings) - 1
    n_years = (holdings["signal_date"].iloc[-1] - holdings["signal_date"].iloc[0]).days / 365.25
    gross_period_mean = panel.groupby("signal_date")["contrib"].sum().mean()
    gross_annual = (1 + gross_period_mean) ** 52 - 1
    nav_total = nav["Top100"].iloc[-1] / nav["Top100"].iloc[0] - 1
    print(f"\n[对账] 期数={n_periods}  跨度={n_years:.2f}年")
    print(f"  逐期等权 gross 均值={gross_period_mean*100:.3f}%  -> 年化 ≈ {gross_annual*100:.2f}%")
    print(f"  NAV 累计={nav_total*100:.2f}%")
    if pe_missing_dates:
        print(f"  WARNING: {len(pe_missing_dates)} 个信号日 pe_ttm 完全缺失: {[d.date() for d in pe_missing_dates[:5]]}")

    # PE 覆盖率
    pe_cov = panel["pe_ttm"].notna().mean()
    print(f"  panel 内 pe_ttm 非空率: {pe_cov*100:.1f}%")

    # ============ 1. PE 正负分组总表 ============
    print("\n" + "=" * 90)
    print("1. PE 正负分组收益贡献总表")
    print("=" * 90)
    grp_order = ["负PE(亏损)", "正PE(盈利)", "零PE", "未知/NaN"]
    per_date = panel.groupby(["signal_date", "pe_grp"], observed=True)
    avg_n = per_date["code"].count().groupby("pe_grp").mean()
    avg_w = per_date["weight"].sum().groupby("pe_grp").mean()
    avg_contrib = per_date["contrib"].sum().groupby("pe_grp").mean()
    total_contrib = panel.groupby("pe_grp", observed=True)["contrib"].sum()
    avg_ret = panel.groupby("pe_grp", observed=True)["ret"].mean()

    pe_df = pd.DataFrame({
        "平均只数": avg_n,
        "平均权重": avg_w,
        "平均个股收益": avg_ret,
        "期间贡献均值": avg_contrib,
        "累计贡献": total_contrib,
    })
    pe_df = pe_df.reindex(grp_order)
    pe_df["年化贡献(近似)"] = pe_df["期间贡献均值"] * 52
    pe_df["占比(只数%)"] = pe_df["平均只数"]
    print(pe_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n  合计年化贡献(近似) = {pe_df['年化贡献(近似)'].sum()*100:.2f}%  (对账 gross 年化 {gross_annual*100:.2f}%)")

    # 负 PE vs 正 PE 直接对比
    neg = pe_df.loc["负PE(亏损)"]
    pos = pe_df.loc["正PE(盈利)"]
    print(f"\n  --- 负 PE vs 正 PE 直接对比 ---")
    print(f"  负PE 平均只数: {neg['平均只数']:.2f}  正PE 平均只数: {pos['平均只数']:.2f}")
    print(f"  负PE 平均个股收益: {neg['平均个股收益']*100:.3f}%  正PE: {pos['平均个股收益']*100:.3f}%  差: {(neg['平均个股收益']-pos['平均个股收益'])*100:.3f}%")
    print(f"  负PE 累计贡献: {neg['累计贡献']*100:.2f}%  正PE 累计贡献: {pos['累计贡献']*100:.2f}%")
    print(f"  负PE 年化贡献: {neg['年化贡献(近似)']*100:.2f}%  正PE 年化贡献: {pos['年化贡献(近似)']*100:.2f}%")

    # ============ 2. 分年 PE 正负贡献 ============
    print("\n" + "=" * 90)
    print("2. 分年 PE 正负贡献 (每年累计贡献 %)")
    print("=" * 90)
    yr = panel.groupby(["year", "pe_grp"], observed=True)["contrib"].sum().unstack(fill_value=0) * 100
    yr = yr.reindex(columns=grp_order)
    print(yr.round(2).to_string())

    # 分年只数占比
    print("\n  分年 PE 分组平均只数:")
    yr_n = panel.groupby(["year", "pe_grp"], observed=True)["code"].count().unstack(fill_value=0)
    yr_n = yr_n.reindex(columns=grp_order)
    print(yr_n.to_string())

    # 分年个股平均收益
    print("\n  分年 PE 分组个股平均期间收益 (%):")
    yr_ret = panel.groupby(["year", "pe_grp"], observed=True)["ret"].mean().unstack(fill_value=np.nan) * 100
    yr_ret = yr_ret.reindex(columns=grp_order)
    print(yr_ret.round(2).to_string())

    # ============ 3. 亏损股占比历史趋势 ============
    print("\n" + "=" * 90)
    print("3. 亏损股 (负 PE) 占比历史趋势")
    print("=" * 90)
    per_date_neg_n = panel[panel["pe_grp"] == "负PE(亏损)"].groupby("signal_date")["code"].count()
    per_date_total = panel.groupby("signal_date")["code"].count()
    neg_ratio = (per_date_neg_n / per_date_total).fillna(0)
    # 按半年聚合看趋势
    neg_ratio_sem = neg_ratio.resample("2MS").mean()
    print(f"  负 PE 占比: 全期均值 = {neg_ratio.mean()*100:.2f}%  最大 = {neg_ratio.max()*100:.2f}%  最小 = {neg_ratio.min()*100:.2f}%")
    print(f"\n  按半年聚合 (2MS) 负 PE 占比均值 (%):")
    print((neg_ratio_sem * 100).round(2).to_string())
    # 分年
    neg_ratio_yr = neg_ratio.resample("YS").mean()
    print(f"\n  分年负 PE 占比均值 (%):")
    print((neg_ratio_yr * 100).round(2).to_string())

    # ============ 4. 最新持仓 PE 分布 ============
    print("\n" + "=" * 90)
    print("4. 最新持仓 PE 分布")
    print("=" * 90)
    # 回测最后一期有收益的信号日
    last_t = panel["signal_date"].max()
    last_panel = panel[panel["signal_date"] == last_t].copy()
    print(f"\n  回测最后有收益期: 信号日 {last_t.date()}  持仓 {len(last_panel)} 只")
    last_pe = last_panel["pe_grp"].value_counts().reindex(grp_order, fill_value=0)
    print("  回测最后期 PE 分组只数:")
    print(last_pe.to_string())
    print(f"  负 PE 占比: {(last_pe.get('负PE(亏损)',0)/len(last_panel)*100):.1f}%")

    # live 08-28
    if LIVE_CSV.exists():
        live = pd.read_csv(LIVE_CSV, dtype={"code": str})
        live["code"] = live["code"].str.zfill(6)
        live_t = pd.to_datetime(live["signal_date"].iloc[0])
        # PE as-of live 信号日
        pe_sub = pe.loc[:live_t]
        if len(pe_sub):
            pe_live = pe_sub.iloc[-1].reindex(live["code"])
        else:
            pe_live = pd.Series(np.nan, index=live["code"])
        live["pe_ttm"] = pe_live.values
        live["pe_grp"] = live["pe_ttm"].apply(pe_group)
        print(f"\n  live {live_t.date()} 候选: {len(live)} 只")
        live_pe = live["pe_grp"].value_counts().reindex(grp_order, fill_value=0)
        print("  live PE 分组只数:")
        print(live_pe.to_string())
        print(f"  负 PE 占比: {(live_pe.get('负PE(亏损)',0)/len(live)*100):.1f}%")
        # 历史均值对照
        print(f"\n  历史负 PE 占比均值: {neg_ratio.mean()*100:.2f}%  (live {live_pe.get('负PE(亏损)',0)/len(live)*100:.1f}%)")
        # 负 PE 个股样本
        neg_live = live[live["pe_grp"] == "负PE(亏损)"].sort_values("pe_ttm")
        if len(neg_live):
            print(f"\n  live 负 PE 个股 (Top 10 最负):")
            print(neg_live.head(10)[["code", "name", "pe_ttm", "circ_mv_yi", "score"]].to_string(index=False))

    # ============ 5. 一句话结论 ============
    print("\n" + "=" * 90)
    print("5. 结论")
    print("=" * 90)
    neg_better = neg["平均个股收益"] > pos["平均个股收益"]
    neg_contrib_pos = neg["累计贡献"] > 0
    neg_share = neg["平均只数"] / 100
    print(f"  负 PE (亏损股) 平均占比: {neg_share*100:.1f}%")
    print(f"  负 PE 个股平均收益: {neg['平均个股收益']*100:.3f}%  vs 正 PE: {pos['平均个股收益']*100:.3f}%")
    print(f"  负 PE 跑得{'更好' if neg_better else '更差'} (差 {abs(neg['平均个股收益']-pos['平均个股收益'])*100:.3f}%/期)")
    print(f"  负 PE 累计贡献: {neg['累计贡献']*100:.2f}%  ({'正贡献' if neg_contrib_pos else '负贡献/拖累'})")
    systematic = neg_share > 0.15 and neg_contrib_pos
    print(f"  模型{'有' if systematic else '无明显'}系统性选亏损小盘倾向 (占比>{neg_share*100:.0f}% 且贡献{'正' if neg_contrib_pos else '非正'})")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
