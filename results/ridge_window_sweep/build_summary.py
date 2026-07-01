"""Build summary.md from ridge_window_sweep artifacts."""
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def top30_ann_from_nav(nav_path: Path) -> float | None:
    nav = pd.read_csv(nav_path, index_col=0, parse_dates=True)
    if "Top30" not in nav.columns:
        return None
    s = nav["Top30"].dropna()
    if len(s) < 2:
        return None
    total = s.iloc[-1] / s.iloc[0] - 1
    years = (s.index[-1] - s.index[0]).days / 365.25
    return round((1 + total) ** (1 / max(years, 0.1)) - 1, 4)


def monotonicity_from_annual(annual_path: Path) -> float | None:
    ann = pd.read_csv(annual_path, index_col=0)
    q_cols = sorted([c for c in ann.columns if re.match(r"Q[1-5]", str(c))])
    if len(q_cols) < 5:
        return None
    ranks = list(range(1, len(q_cols) + 1))
    yearly = []
    for _, row in ann.iterrows():
        rets = [row[c] for c in q_cols]
        if any(pd.isna(rets)):
            continue
        corr = pd.Series(ranks).corr(pd.Series(rets), method="spearman")
        if corr is not None and not math.isnan(corr):
            yearly.append(corr)
    if not yearly:
        return None
    return round(float(pd.Series(yearly).mean()), 4)


def _read_log_text(log_path: Path) -> str:
    raw = log_path.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_wall_times(log_path: Path) -> dict[str, float | None]:
    """Parse per-run wall time (minutes) from sweep.log run markers."""
    if not log_path.exists():
        return {}
    lines = _read_log_text(log_path).splitlines()
    ts_pat = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
    run_starts: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = re.search(r"horizon=(\d+) train-windows=([\d,]+)", line)
        if line.startswith("===== Run ") and m:
            run_starts.append((f"h{m.group(1)}_{m.group(2)}", i))

    result: dict[str, float | None] = {}
    for idx, (key, start_i) in enumerate(run_starts):
        end_i = run_starts[idx + 1][1] if idx + 1 < len(run_starts) else len(lines)
        block = "\n".join(lines[start_i:end_i])
        stamps = []
        for sm in ts_pat.finditer(block):
            try:
                stamps.append(datetime.strptime(sm.group(1), "%Y-%m-%d %H:%M:%S.%f"))
            except ValueError:
                pass
        if len(stamps) >= 2:
            result[key] = round((stamps[-1] - stamps[0]).total_seconds() / 60.0, 1)
        else:
            result[key] = None
    return result


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return v


def main() -> None:
    log_failures = []
    log_path = BASE / "sweep.log"
    if log_path.exists():
        for line in _read_log_text(log_path).splitlines():
            if line.startswith("Run failed:"):
                log_failures.append(line.replace("Run failed: ", ""))

    wall_times = parse_wall_times(log_path)

    rows = []
    for mp in sorted(BASE.glob("model_metrics_ridge_h*.json")):
        with open(mp, encoding="utf-8") as f:
            d = json.load(f, parse_constant=lambda x: float("nan") if x == "NaN" else None)
        tag = d.get("tag", mp.stem.replace("model_metrics_", ""))
        h = re.search(r"_h(\d+)", tag)
        horizon = int(h.group(1)) if h else None
        wm = re.search(r"_w([\d-]+)", tag)
        windows = wm.group(1).replace("-", ",") if wm else ""
        bt_nav = BASE / f"backtest_{tag}_nav.csv"
        bt_ann = BASE / f"backtest_{tag}_annual.csv"
        n_pred = d.get("预测期数") or 0
        status = "OK" if n_pred > 0 and bt_nav.exists() else (
            "FAILED (no backtest)" if n_pred == 0 else "PARTIAL"
        )
        wt_key = f"h{horizon}_{windows}"
        rows.append({
            "horizon": horizon,
            "windows": windows,
            "IC": d.get("IC均值"),
            "ICIR": d.get("ICIR"),
            "IC>0%": d.get("IC>0胜率"),
            "monotonicity": monotonicity_from_annual(bt_ann) if bt_ann.exists() else None,
            "top30 ann": top30_ann_from_nav(bt_nav) if bt_nav.exists() else None,
            "wall min": wall_times.get(wt_key),
            "status": status,
        })

    rows.sort(key=lambda r: (r["horizon"] or 0, r["windows"]))

    out = BASE / "summary.md"
    lines = [
        "# Ridge train-window sweep",
        "",
        "Batch: `results/ridge_window_sweep/run_sweep.ps1` | "
        "optimizations: `optimization_notes.md` | log: `sweep.log`",
        "",
        f"**Completed with backtest:** "
        f"{sum(1 for r in rows if r['status'] == 'OK')}/{len(rows) or 6} runs.",
        "",
        "| horizon | windows (months) | IC mean | ICIR | IC>0% | monotonicity | top30 ann return | wall min | status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['horizon']} | {r['windows']} | {fmt(r['IC'])} | {fmt(r['ICIR'])} | "
            f"{fmt(r['IC>0%'])} | {fmt(r['monotonicity'])} | {fmt(r['top30 ann'])} | "
            f"{fmt(r['wall min'])} | {r['status']} |"
        )
    if log_failures:
        lines.extend(["", "## Failures / notes", ""])
        for f in log_failures:
            lines.append(f"- {f}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
