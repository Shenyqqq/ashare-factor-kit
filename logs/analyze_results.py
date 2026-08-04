"""
analyze_results.py — 汇总所有模型回测结果并给出 ensemble 组合建议

运行：python logs/analyze_results.py
"""
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# ── 1. 汇总 model_metrics_*.json ────────────────────────────────────────────
print("=" * 70)
print("【模型 IC 指标汇总】")
print("=" * 70)

metrics_rows = []
for fpath in sorted(Path(BASE).rglob("model_metrics_*.json")):
    with open(fpath, encoding="utf-8") as f:
        d = json.load(f)
    metrics_rows.append(d)

if metrics_rows:
    df_m = pd.DataFrame(metrics_rows).set_index("tag")
    df_m = df_m.sort_values("ICIR", ascending=False)
    print(df_m.to_string())
else:
    print("未找到 model_metrics_*.json 文件")

# ── 2. 汇总 backtest_*_nav.csv Q5 年化收益 ──────────────────────────────────
print("\n" + "=" * 70)
print("【各模型 Q5 累计收益（回测区间）】")
print("=" * 70)

nav_rows = []
for fpath in sorted(Path(BASE).rglob("backtest_*_nav.csv")):
    tag = Path(fpath).stem.replace("backtest_", "").replace("_nav", "")
    try:
        nav = pd.read_csv(fpath, index_col=0, parse_dates=True)
        # 找 Q5 列
        q5_col = [c for c in nav.columns if "Q5" in str(c) or "q5" in str(c)]
        if not q5_col:
            continue
        q5 = nav[q5_col[0]].dropna()
        total_ret = q5.iloc[-1] / q5.iloc[0] - 1
        years = (q5.index[-1] - q5.index[0]).days / 365
        ann_ret = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

        # Q1 对比
        q1_col = [c for c in nav.columns if "Q1" in str(c) or "q1" in str(c)]
        q1_ann = np.nan
        if q1_col:
            q1 = nav[q1_col[0]].dropna()
            q1_total = q1.iloc[-1] / q1.iloc[0] - 1
            q1_ann = (1 + q1_total) ** (1 / max(years, 0.1)) - 1

        nav_rows.append({
            "tag": tag,
            "Q5年化": round(ann_ret, 4),
            "Q1年化": round(q1_ann, 4),
            "Q5-Q1": round(ann_ret - q1_ann, 4),
            "单调性": "[正向]" if ann_ret > q1_ann else "[反向]",
        })
    except Exception as e:
        nav_rows.append({"tag": tag, "Q5年化": f"ERR:{e}"})

if nav_rows:
    df_n = pd.DataFrame(nav_rows).set_index("tag")
    df_n = df_n.sort_values("Q5-Q1", ascending=False)
    print(df_n.to_string())

# ── 3. 分组单调性检查（年度）────────────────────────────────────────────────
print("\n" + "=" * 70)
print("【年度 Q5 vs Q1 单调性分析】")
print("=" * 70)

for fpath in sorted(Path(BASE).rglob("backtest_*_annual.csv")):
    tag = Path(fpath).stem.replace("backtest_", "").replace("_annual", "")
    try:
        ann = pd.read_csv(fpath, index_col=0)
        q5_col = [c for c in ann.columns if "Q5" in str(c)]
        q1_col = [c for c in ann.columns if "Q1" in str(c)]
        if q5_col and q1_col:
            spread = ann[q5_col[0]] - ann[q1_col[0]]
            pos_years = (spread > 0).sum()
            total_years = spread.count()
            print(f"{tag:40s}  Q5>Q1的年份: {pos_years}/{total_years}  ({spread.values.round(3)})")
    except Exception as e:
        print(f"{tag}: {e}")

# ── 4. WQ 因子 IC 分析结果 ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("【WQ Alpha101 因子 IC 验证结果】")
print("=" * 70)

for period in [5, 20]:
    fpath = BASE / "research" / "output" / f"selected_factors_h{period}.json"
    if fpath.exists():
        with open(fpath, encoding="utf-8") as f:
            selected = json.load(f)
        factors = selected.get("factors", [])
        wq_selected = [x for x in factors if "WQ_" in x]
        all_factors = factors
        print(f"\nh{period} 白名单共 {len(all_factors)} 个因子，其中 WQ_ 因子: {wq_selected if wq_selected else '无（全部被剔除）'}")
    else:
        print(f"\nh{period}: IC 分析结果文件不存在，请先运行 ic_analysis")

# ── 5. Ensemble 组合建议 ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("【Ensemble 组合建议】")
print("=" * 70)

if metrics_rows:
    h5_models = {r["tag"].split("_")[0]: r for r in metrics_rows if "_h5" in r["tag"] and "w" not in r["tag"] and r["tag"].count("_") == 1}
    h20_models = {r["tag"].split("_")[0]: r for r in metrics_rows if "_h20" in r["tag"] and "w" not in r["tag"] and r["tag"].count("_") == 1}

    print("\n  h5 各模型 ICIR:")
    for m, r in sorted(h5_models.items(), key=lambda x: x[1].get("ICIR", -99), reverse=True):
        print(f"    {m:10s}: ICIR={r.get('ICIR','N/A'):.3f}  IC={r.get('IC均值','N/A'):.4f}")

    print("\n  h20 各模型 ICIR:")
    for m, r in sorted(h20_models.items(), key=lambda x: x[1].get("ICIR", -99), reverse=True):
        print(f"    {m:10s}: ICIR={r.get('ICIR','N/A'):.3f}  IC={r.get('IC均值','N/A'):.4f}")

    # 建议：保留 ICIR > 0.2 的模型
    print("\n  建议 ensemble 组合（ICIR > 0.2 的模型）:")
    for horizon, models_dict in [("h5", h5_models), ("h20", h20_models)]:
        keep = [m for m, r in models_dict.items() if r.get("ICIR", 0) > 0.2]
        print(f"    {horizon}: {keep if keep else '无达标模型，保留全部'}")

# ── 6. PBO + Deflated Sharpe Ratio 评估（AFML Ch11-15）─────────────────────
print("\n" + "=" * 70)
print("【PBO + Deflated Sharpe Ratio 评估】")
print("=" * 70)

try:
    from research.pbo import evaluate_experiments

    pbo_result = evaluate_experiments(results_dir=str(BASE / "results"), metric="ICIR")
    if pbo_result["n_experiments"] == 0:
        print(pbo_result["interpretation"])
    else:
        n_exp = pbo_result["n_experiments"]
        best_icir = pbo_result["best_icir"]
        best_tag = pbo_result["best_tag"]
        best_T = pbo_result["best_n_obs"]
        dsr = pbo_result["dsr"]
        print(f"  实验总数 N        : {n_exp}")
        print(f"  最优 ICIR         : {best_icir:+.4f}  (tag={best_tag}, T={best_T})")
        print(f"  Deflated Sharpe   : {dsr:.4f}")
        print(f"  解读              : {pbo_result['interpretation']}")
        print()
        print("  阈值参考：")
        print("    DSR < 0.95     -> 多试验校正后不显著，最优 ICIR 疑似噪声/过拟合")
        print("    DSR >= 0.95    -> 边缘显著，可谨慎使用")
        print("    DSR >= 0.99    -> 统计显著")
        print()
        if dsr < 0.95:
            print(f"  [过拟合警告] 最优 {best_tag} 的 ICIR={best_icir:+.4f} "
                  f"在 {n_exp} 次试验校正后 DSR={dsr:.3f}，不应直接采信；")
            print("             建议缩减试验空间、增加样本长度，或保留多个模型做 ensemble。")
        else:
            print(f"  [通过] 最优 {best_tag} 的 ICIR 在 {n_exp} 次试验校正后仍显著 (DSR={dsr:.3f})。")
except Exception as e:
    print(f"  PBO/DSR 评估失败: {e}")

print("\n" + "=" * 70)
print("分析完成。使用示例：")
print("  python run.py --mode ensemble --horizon 5 --models lgbm,xgb --skip-download --factor-config config/factor_configs.yaml")
print("=" * 70)
