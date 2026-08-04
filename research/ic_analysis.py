"""
research/ic_analysis.py  —  v1 IC 分析入口（已退役为 v2 的薄 shim）。

历史：本模块曾是独立的 v1 IC 分析流水线，与 v2（research/ic/cli.py）共享
`research/output/selected_factors_h{period}.json` 输出路径但实现独立。
这导致一个陷阱：通过 v1 入口跑 smoke test 会用 v1 格式 JSON 覆盖 v2 的
富化输出（含 engine="v2"、meta、factors_orth 等字段），且 v1 的筛选逻辑
与 v2 不一致，结果不可混读。

修复（2026-07-03）：将本模块改为委托给 v2 引擎的薄 shim，统一代码路径。
v1 的独立实现已删除；如需历史实现请查 git history。

用法（与 v2 一致，详见 `python -m research.ic_analysis --help`）：
    python -m research.ic_analysis --period 5 --raw-select --save
    python -m research.ic_analysis --period 20 --barra --save --use-fdr --t-threshold 2.5

v1 独有的 `--neutralize` 标志已不再支持；行业中性化请用 v2 的
`--feature-neutralize`（在 strategies/ml.py 出口做 Barra+行业残差化）。
"""
from research.ic.cli import main as _v2_main
from research.ic.ic_series import compute_ic_series as _compute_ic_series

# 向后兼容：历史上 market_state.py / factor_event.py 从本模块导入
# compute_ic_series。v2 的同名函数签名是超集（tradable/min_stocks 可选），
# 直接 re-export 不会破坏既有 2-arg 调用。
compute_ic_series = _compute_ic_series


def main():
    _v2_main()


if __name__ == "__main__":
    main()
