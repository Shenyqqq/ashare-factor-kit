"""Save IC analysis outputs (driver.py compatible JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def save_results(
    period: int,
    summary_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    kept_factors: list,
    exclusion_reasons: dict,
    lookback_years: int = 0,
    lookback_date=None,
    ind_ic_df: pd.DataFrame | None = None,
    pure_ic_means: dict | None = None,
    meta: dict | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(OUTPUT_DIR / f"ic_summary_h{period}.csv", encoding="utf-8-sig")
    yearly_df.to_csv(OUTPUT_DIR / f"ic_yearly_h{period}.csv", encoding="utf-8-sig")

    if ind_ic_df is not None and not ind_ic_df.empty:
        ind_ic_df.to_csv(OUTPUT_DIR / f"ic_industry_h{period}.csv", encoding="utf-8-sig")

    if pure_ic_means:
        pure_df = pd.DataFrame.from_dict(
            pure_ic_means, orient="index", columns=["纯因子IC均值"]
        )
        pure_df["原始IC均值"] = summary_df["IC均值"].reindex(pure_df.index)
        pure_df["保留率"] = pure_df["纯因子IC均值"] / pure_df["原始IC均值"]
        pure_df.to_csv(OUTPUT_DIR / f"ic_barra_pure_h{period}.csv", encoding="utf-8-sig")

    ic_cols = [c for c in ["IC均值", "ICIR", "胜率", "NW_t统计量", "IC_after_cost"] if c in summary_df.columns]
    selection = {
        "horizon": period,
        "lookback_years": lookback_years if lookback_years > 0 else "full",
        "ic_start_date": str(lookback_date.date()) if lookback_date is not None else "all",
        "engine": "v2",
        "factors": kept_factors,
        "excluded": exclusion_reasons,
        "ic_stats": summary_df[ic_cols].to_dict(),
        "meta": meta or {},
    }
    json_path = OUTPUT_DIR / f"selected_factors_h{period}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)
    return json_path
