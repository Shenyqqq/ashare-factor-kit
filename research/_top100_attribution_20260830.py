"""Top100 收益归因分析（市值分层 / 行业 / 个股）。

数据：
  - 全 WF 回测: results/xgb_h5_sizeind_w156_nob_wf_20260830/
  - holdings_top100_*.csv (信号日 + pipe 分隔 codes, 等权 1/100)
  - close_hfq.parquet / circ_mv.parquet / industry_map_panel.parquet

口径：
  - 持仓等权 (1/100)，无权重列，按等权近似。
  - 个股期间收益 = close_hfq[next_signal_date] / close_hfq[t] - 1 (close-to-close 近似)
  - 市值分层用 circ_mv（元）在信号日 as-of，转成亿元。
  - 行业用 industry_map_panel PIT as-of 信号日。
  - 贡献 = (1/100) * 个股期间收益；年化贡献 ≈ 期间贡献均值 * 52。
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
CIRC_MV = ROOT / "data" / "raw" / "circ_mv.parquet"
INDUSTRY = ROOT / "data" / "raw" / "industry_map_panel.parquet"

SIZE_BINS = [-np.inf, 20, 50, 200, 1000, np.inf]
SIZE_LABELS = ["微盘<20", "小盘20-50", "中盘50-200", "大盘200-1000", "超大盘>1000"]

# 申万 2021 一级行业代码 -> 名称（标准 31 行业，按本回测持仓出现的 25 个）
SW_L1_NAME = {
    "11": "农林牧渔", "22": "基础化工", "23": "钢铁", "24": "有色金属",
    "27": "电子", "28": "汽车", "31": "家用电器", "33": "家用电器",
    "34": "食品饮料", "35": "纺织服饰", "36": "建筑装饰", "37": "医药生物",
    "38": "机械设备", "41": "公用事业", "42": "交通运输", "43": "房地产",
    "44": "银行", "45": "商贸零售", "46": "社会服务", "47": "计算机",
    "48": "银行", "50": "食品饮料", "51": "商贸零售", "52": "交通运输",
    "60": "建筑材料", "61": "房地产", "62": "建筑装饰", "63": "电力设备",
    "64": "机械设备", "65": "国防军工", "71": "计算机", "72": "传媒",
    "73": "通信", "74": "煤炭", "75": "石油石化", "76": "环保",
    "NA": "未分类",
}


def load_sw_l2_names():
    import json
    p = ROOT / "research" / "output" / "_sw_name_map_20260830.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("l2_code_to_name", {})


def load_holdings():
    df = pd.read_csv(HOLD_CSV)
    df.columns = ["signal_date", "codes", "n"]
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["codes"] = df["codes"].str.split(" | ", regex=False)
    return df[["signal_date", "codes"]].sort_values("signal_date").reset_index(drop=True)


def pit_industry_asof(industry_panel, codes, date):
    sub = industry_panel[industry_panel["code"].isin(codes)].copy()
    sub["effective_date"] = pd.to_datetime(sub["effective_date"])
    sub["end_date"] = pd.to_datetime(sub["end_date"])
    mask = (sub["effective_date"] <= date) & (sub["end_date"].isna() | (sub["end_date"] > date))
    sub = sub[mask]
    # 防止同一 code 多行（重复 effective 区间），保留最后一行
    sub = sub.drop_duplicates(subset="code", keep="last")
    return sub.set_index("code")


def nearest_idx(idx, t):
    return idx[idx.get_indexer([t], method="nearest")[0]]


def main():
    print("=" * 80)
    print("Top100 收益归因 — xgb_h5_sizeind_w156_nob_wf_20260830")
    print("=" * 80)

    holdings = load_holdings()
    print(f"\n持仓期数: {len(holdings)}  起止: {holdings['signal_date'].iloc[0].date()} -> {holdings['signal_date'].iloc[-1].date()}")

    close = pd.read_parquet(CLOSE_HFQ)
    circ_mv = pd.read_parquet(CIRC_MV)
    industry = pd.read_parquet(INDUSTRY)
    sw_l2_names = load_sw_l2_names()
    close.columns = close.columns.astype(str)
    circ_mv.columns = circ_mv.columns.astype(str)

    nav = pd.read_csv(NAV_CSV).rename(columns={lambda c: c if c != nav.columns[0] else "date": "date"} if False else {pd.read_csv(NAV_CSV).columns[0]: "date"})
    nav["date"] = pd.to_datetime(nav["date"])

    records = []
    nav_rets = []
    for i in range(len(holdings) - 1):
        t = holdings.loc[i, "signal_date"]
        t_next = holdings.loc[i + 1, "signal_date"]
        codes = holdings.loc[i, "codes"]

        t_c = nearest_idx(close.index, t)
        t_n = nearest_idx(close.index, t_next)
        px0 = close.loc[t_c].reindex(codes)
        px1 = close.loc[t_n].reindex(codes)
        ret = px1 / px0 - 1.0

        t_m = nearest_idx(circ_mv.index, t)
        mv = circ_mv.loc[t_m].reindex(codes) / 1e8

        ind = pit_industry_asof(industry, codes, t)
        sw_l1 = ind["sw_l1"].astype(str)
        sw_l2 = ind["sw_l2"].astype(str)

        for code in codes:
            records.append({
                "signal_date": t,
                "year": t.year,
                "code": code,
                "ret": ret.get(code, np.nan),
                "circ_mv_yi": mv.get(code, np.nan),
                "sw_l1": sw_l1.get(code, "NA"),
                "sw_l2": sw_l2.get(code, "NA"),
            })

        nav0 = nav.loc[nav["date"] == t, "Top100"]
        nav1 = nav.loc[nav["date"] == t_next, "Top100"]
        if len(nav0) and len(nav1):
            nav_rets.append(float(nav1.iloc[0] / nav0.iloc[0] - 1.0))

    panel = pd.DataFrame(records)
    panel["weight"] = 1.0 / 100.0
    panel["contrib"] = panel["weight"] * panel["ret"]
    nav_arr = np.array(nav_rets)

    n_years = (holdings["signal_date"].iloc[-1] - holdings["signal_date"].iloc[0]).days / 365.25
    gross_period_mean = panel.groupby("signal_date")["contrib"].sum().mean()
    gross_annual = (1 + gross_period_mean) ** 52 - 1
    nav_annual = (1 + nav_arr.mean()) ** 52 - 1
    nav_total = nav["Top100"].iloc[-1] / nav["Top100"].iloc[0] - 1
    print(f"\n[对账] 期数={len(holdings)-1}  跨度={n_years:.2f}年")
    print(f"  逐期等权 gross 均值={gross_period_mean*100:.3f}%  -> 年化 ≈ {gross_annual*100:.2f}%")
    print(f"  NAV period 均值={nav_arr.mean()*100:.3f}%  -> 年化 ≈ {nav_annual*100:.2f}%")
    print(f"  NAV 累计={nav_total*100:.2f}%  (终值/起点)")
    print(f"  报告基线: 年化 23.3% / 超额 18.7%")

    # 1. 市值分层
    panel["size_bucket"] = pd.cut(panel["circ_mv_yi"], bins=SIZE_BINS, labels=SIZE_LABELS)
    print("\n" + "=" * 80)
    print("1. 市值分层收益贡献（流通市值，亿元）")
    print("=" * 80)
    per_date_size = panel.groupby(["signal_date", "size_bucket"], observed=True)
    avg_n = per_date_size["code"].count().groupby("size_bucket").mean()
    avg_w = per_date_size["weight"].sum().groupby("size_bucket").mean()
    avg_contrib = per_date_size["contrib"].sum().groupby("size_bucket").mean()
    total_contrib = panel.groupby("size_bucket", observed=True)["contrib"].sum()
    avg_mv = panel.groupby("size_bucket", observed=True)["circ_mv_yi"].mean()
    size_df = pd.DataFrame({
        "平均只数": avg_n,
        "平均权重": avg_w,
        "平均市值(亿)": avg_mv,
        "期间贡献均值": avg_contrib,
        "累计贡献": total_contrib,
    })
    size_df["年化贡献(近似)"] = size_df["期间贡献均值"] * 52
    print(size_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"  合计年化贡献(近似) = {size_df['年化贡献(近似)'].sum()*100:.2f}%  (对账 gross 年化 {gross_annual*100:.2f}%)")

    print("\n  分年市值分层贡献（每年累计贡献，%）：")
    yr_size = panel.groupby(["year", "size_bucket"], observed=True)["contrib"].sum().unstack(fill_value=0) * 100
    print(yr_size.round(2).to_string())

    # 2a. 行业一级
    print("\n" + "=" * 80)
    print("2a. 申万一级行业收益贡献")
    print("=" * 80)
    per_date_l1 = panel.groupby(["signal_date", "sw_l1"], observed=True)
    l1_n = per_date_l1["code"].count().groupby("sw_l1").mean()
    l1_w = per_date_l1["weight"].sum().groupby("sw_l1").mean()
    l1_period = per_date_l1["contrib"].sum().groupby("sw_l1").mean()
    l1_total = panel.groupby("sw_l1", observed=True)["contrib"].sum()
    l1_df = pd.DataFrame({"平均只数": l1_n, "平均权重": l1_w, "期间贡献均值": l1_period, "累计贡献": l1_total})
    l1_df["年化贡献(近似)"] = l1_df["期间贡献均值"] * 52
    l1_df = l1_df.sort_values("累计贡献", ascending=False)
    l1_df["行业名"] = [SW_L1_NAME.get(str(i), str(i)) for i in l1_df.index]
    print("\n  Top 10 赚钱一级行业:")
    print(l1_df.head(10)[["行业名", "平均只数", "平均权重", "累计贡献", "年化贡献(近似)"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n  Bottom 5 拖累一级行业:")
    print(l1_df.tail(5)[["行业名", "平均只数", "平均权重", "累计贡献", "年化贡献(近似)"]].to_string(float_format=lambda x: f"{x:.4f}"))

    pos = l1_df["累计贡献"].clip(lower=0).sort_values(ascending=False)
    pos_total = pos.sum()
    print(f"\n  行业集中度: 正贡献行业={(l1_df['累计贡献']>0).sum()}  负贡献行业={(l1_df['累计贡献']<0).sum()}")
    print(f"  Top3 行业占正贡献比重 = {pos.head(3).sum()/pos_total*100:.1f}%")
    print(f"  Top5 行业占正贡献比重 = {pos.head(5).sum()/pos_total*100:.1f}%")

    print("\n  分年 Top10 一级行业贡献（每年累计贡献，%）：")
    yr_l1 = panel.groupby(["year", "sw_l1"], observed=True)["contrib"].sum().unstack(fill_value=0) * 100
    yr_top = yr_l1[l1_df.head(10).index]
    yr_top.columns = [f"{c}({SW_L1_NAME.get(str(c),str(c))[:4]})" for c in yr_top.columns]
    print(yr_top.round(2).to_string())

    # 2b. 行业二级
    print("\n" + "=" * 80)
    print("2b. 申万二级行业收益贡献（Top 10 / Bottom 5）")
    print("=" * 80)
    per_date_l2 = panel.groupby(["signal_date", "sw_l2"], observed=True)
    l2_n = per_date_l2["code"].count().groupby("sw_l2").mean()
    l2_period = per_date_l2["contrib"].sum().groupby("sw_l2").mean()
    l2_total = panel.groupby("sw_l2", observed=True)["contrib"].sum()
    l2_df = pd.DataFrame({"平均只数": l2_n, "期间贡献均值": l2_period, "累计贡献": l2_total})
    l2_df["年化贡献(近似)"] = l2_df["期间贡献均值"] * 52
    l2_df = l2_df.sort_values("累计贡献", ascending=False)
    l2_df["行业名"] = [sw_l2_names.get(str(i), str(i)) for i in l2_df.index]
    print("\n  Top 10 赚钱二级行业:")
    print(l2_df.head(10)[["行业名", "平均只数", "累计贡献", "年化贡献(近似)"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n  Bottom 5 拖累二级行业:")
    print(l2_df.tail(5)[["行业名", "平均只数", "累计贡献", "年化贡献(近似)"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # 3. 个股 Top 20
    print("\n" + "=" * 80)
    print("3. 个股累计贡献 Top 20")
    print("=" * 80)
    stock = panel.groupby("code").agg(累计贡献=("contrib", "sum"), 入选次数=("code", "count"), 平均收益=("ret", "mean"))
    last_asof = panel.sort_values("signal_date").groupby("code").tail(1).set_index("code")[["circ_mv_yi", "sw_l1", "sw_l2"]]
    stock = stock.join(last_asof)
    stock["sw_l1名"] = [SW_L1_NAME.get(str(i), str(i)) for i in stock["sw_l1"]]
    stock["sw_l2名"] = [sw_l2_names.get(str(i), str(i)) for i in stock["sw_l2"]]
    stock = stock.sort_values("累计贡献", ascending=False)
    print("\n  Top 20 贡献个股:")
    print(stock.head(20)[["累计贡献", "入选次数", "平均收益", "circ_mv_yi", "sw_l1名", "sw_l2名"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n  Bottom 10 拖累个股:")
    print(stock.tail(10)[["累计贡献", "入选次数", "平均收益", "circ_mv_yi", "sw_l1名", "sw_l2名"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # 集中度：Top20 占总正贡献
    pos_stock = stock[stock["累计贡献"] > 0]["累计贡献"]
    print(f"\n  个股集中度: 正贡献个股={len(pos_stock)}  负贡献个股={(stock['累计贡献']<0).sum()}")
    print(f"  Top20 个股占正贡献比重 = {stock.head(20)['累计贡献'].clip(lower=0).sum()/pos_stock.sum()*100:.1f}%")
    print(f"  Top20 个股占总贡献(含负) = {stock.head(20)['累计贡献'].sum()/stock['累计贡献'].sum()*100:.1f}%")
    # Top20 个股市值/行业分布
    print("\n  Top20 个股市值分层分布:")
    print(pd.cut(stock.head(20)["circ_mv_yi"], bins=SIZE_BINS, labels=SIZE_LABELS).value_counts().to_string())
    print("\n  Top20 个股一级行业分布:")
    print(stock.head(20)["sw_l1名"].value_counts().to_string())

    # 4. 最新一期持仓（回测最后一期有收益的 + live 08-28）vs 历史赚钱模式
    print("\n" + "=" * 80)
    print("4. 最新持仓 vs 历史赚钱模式")
    print("=" * 80)
    # 回测最后一期有收益的信号日 = holdings 倒数第二期（panel 最后一期）
    last_t = panel["signal_date"].max()
    last_panel = panel[panel["signal_date"] == last_t].copy()
    print(f"\n  回测最后有收益期: 信号日 {last_t.date()}  持仓 {len(last_panel)} 只")
    last_size = pd.cut(last_panel["circ_mv_yi"], bins=SIZE_BINS, labels=SIZE_LABELS).value_counts().sort_index()
    print("  回测最后期市值分层分布:")
    print(last_size.to_string())
    last_panel["sw_l1名"] = [SW_L1_NAME.get(str(i), str(i)) for i in last_panel["sw_l1"]]
    print("\n  回测最后期一级行业分布:")
    print(last_panel["sw_l1名"].value_counts().head(10).to_string())

    # live 08-28（当前实际持仓）
    live_csv = RES / "candidates_20260828.csv"
    if live_csv.exists():
        live = pd.read_csv(live_csv, dtype={"code": str})
        live["code"] = live["code"].str.zfill(6)
        print(f"\n  live 08-28 候选: {len(live)} 只")
        if "circ_mv" in live.columns:
            live["circ_mv_yi"] = pd.to_numeric(live["circ_mv"], errors="coerce") / 1e8
            print("  live 08-28 市值分层分布:")
            print(pd.cut(live["circ_mv_yi"], bins=SIZE_BINS, labels=SIZE_LABELS).value_counts().sort_index().to_string())
        if "sw_l2" in live.columns:
            print("\n  live 08-28 二级行业分布 (Top 10):")
            print(live["sw_l2"].value_counts().head(10).to_string())

    # 历史赚钱模式对照
    print("\n  历史赚钱模式对照（累计贡献）:")
    print(f"    最赚市值层: {size_df['累计贡献'].idxmax()}  ({size_df['累计贡献'].max()*100:.2f}%)")
    print(f"    最赚一级行业: {l1_df['行业名'].iloc[0]}  ({l1_df['累计贡献'].max()*100:.2f}%)")
    print(f"    最赚二级行业: {l2_df['行业名'].iloc[0]}  ({l2_df['累计贡献'].max()*100:.2f}%)")
    # 历史持仓市值分布（平均只数）
    print("\n  历史持仓市值分层（平均只数占比）:")
    for b in SIZE_LABELS:
        n = size_df.loc[b, "平均只数"] if b in size_df.index else 0
        print(f"    {b}: {n:.1f} 只 ({n:.0f}%)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
