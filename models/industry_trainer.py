"""
models/industry_trainer.py  —  分行业 Walk-Forward 训练器

策略：
    对每个申万二级行业单独训练一个模型（WalkForwardTrainer），
    最后把所有行业的预测得分合并为全市场截面得分。

优势：
    - 行业内特征分布差异大（金融/科技/消费用的因子有效性完全不同）
    - 同行业内排名更稳定，不被跨行业的尺度差异污染
    - 自动适配行业牛熊轮动（各行业模型独立调权）

用法：
    from strategies.ml import build_factor_dataset
    from models.industry_trainer import IndustryWalkForwardTrainer

    dataset = build_factor_dataset(..., factor_whitelist=whitelist, **extra_kwargs)
    trainer = IndustryWalkForwardTrainer(model_types=["lgbm","xgb"])
    score_df = trainer.fit_predict(dataset, industry_map=industry_map)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from loguru import logger

from models.trainer import (
    WalkForwardTrainer, MLDataset,
    MIN_STOCKS_PER_DATE, MODEL_TYPES,
    TRAIN_WINDOWS_MONTHS, VAL_WINDOW_MONTHS, REBALANCE_FREQ,
    LABEL_MODE_DEFAULT, spearman_ic,
)
from config.settings import PROCESSED_DIR

# 行业股票数低于此阈值时合并到申万一级桶
# 提高至 80：子训练器有独立的 embargo_periods（基于 hold_period），
# 行业股票过少会导致 purged train 后样本严重不足，且 IC 噪声大
MIN_STOCKS_INDUSTRY = 80


class IndustryWalkForwardTrainer:
    """
    分行业 Walk-Forward 训练器。

    fit_predict() 返回全市场因子得分 DataFrame，
    与 WalkForwardTrainer / DynamicFactorTrainer 接口完全兼容，可直接对接回测引擎。

    Parameters
    ----------
    model_types : list[str]
        传给每个行业子训练器的模型列表，默认 ["lgbm","xgb"]。
    train_windows : list[int]
        Walk-Forward 训练窗口月数，默认 TRAIN_WINDOWS_MONTHS（构造时按调仓频率转为期数）。
    val_window : int
        验证窗口月数，默认 VAL_WINDOW_MONTHS（6）。
    rebalance_freq : str
        调仓频率（与 run.py horizon 推断一致），用于月数→调仓期数转换。
    min_stocks : int
        行业股票数低于此值时合并到一级行业桶，默认 80。
    hold_period : int
        持仓周期（日），透传到子 WalkForwardTrainer，影响 embargo_periods 与 forward_return。
    label_mode : str | dict
        标签截面标准化模式，透传到子 WalkForwardTrainer。
    wf_selection : str
        多窗口/模型选择策略，透传到子 WalkForwardTrainer。
    ensemble_method : str
        多模型集成方法，透传到子 WalkForwardTrainer。
    output_rank : bool
        是否输出 pct rank，透传到子 WalkForwardTrainer。
    """

    def __init__(
        self,
        model_types: list = None,
        train_windows: list = None,
        val_window: int = VAL_WINDOW_MONTHS,
        min_stocks: int = MIN_STOCKS_INDUSTRY,
        rebalance_freq: str = None,
        hold_period: int = 20,
        label_mode: str = LABEL_MODE_DEFAULT,
        wf_selection: str = "ic_weighted",
        ensemble_method: str = "zscore",
        output_rank: bool = False,
    ):
        self.model_types   = model_types   or list(MODEL_TYPES)
        self.train_windows = train_windows or list(TRAIN_WINDOWS_MONTHS)
        self.val_window    = val_window
        self.min_stocks    = min_stocks
        self.rebalance_freq = rebalance_freq or REBALANCE_FREQ
        self.hold_period   = hold_period
        self.label_mode    = label_mode
        self.wf_selection  = wf_selection
        self.ensemble_method = ensemble_method
        self.output_rank   = output_rank

        # 诊断属性
        self.industry_ic:    dict[str, pd.Series] = {}
        self.small_industries: list[str]           = []
        self.score_df:       pd.DataFrame          = None
        self.ic_series:      pd.Series             = None

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _build_groups(
        self,
        all_stocks: pd.Index,
        industry_map: pd.Series,
    ) -> dict[str, list[str]]:
        """
        把股票按申万二级行业分组；股票数不足 min_stocks 的小行业
        合并到对应一级行业桶（_l1_XXX）。

        industry_map 应为 Series(stock_code → sw_l2)；
        若上层提供的是 DataFrame，取其 "sw_l2" 列。
        """
        if isinstance(industry_map, pd.DataFrame):
            l1_series = industry_map.get("sw_l1", pd.Series(dtype=str))
            l2_series = industry_map.get("sw_l2", industry_map.iloc[:, 0])
        else:
            l2_series = industry_map
            l1_series = pd.Series(dtype=str)

        l2 = l2_series.reindex(all_stocks).fillna("未分类")
        l1 = l1_series.reindex(all_stocks).fillna("未分类")

        groups: dict[str, list[str]] = {}
        for ind_l2, sub in l2.groupby(l2):
            stocks = sub.index.tolist()
            if len(stocks) < self.min_stocks:
                # 取该批股票里最多的一级行业作为桶名
                l1_val = l1.reindex(stocks).mode()
                bucket = f"_l1_{l1_val.iloc[0] if len(l1_val) else '其他'}"
                groups.setdefault(bucket, []).extend(stocks)
                self.small_industries.append(ind_l2)
            else:
                groups[ind_l2] = stocks

        logger.info(
            f"行业分组完成: {len(groups)} 个训练组 "
            f"（含 {len([g for g in groups if g.startswith('_l1_')])} 个合并桶），"
            f"小行业合并 {len(self.small_industries)} 个"
        )
        return groups

    def _slice_dataset(
        self,
        dataset: MLDataset,
        stocks: list[str],
    ) -> MLDataset | None:
        """
        从全市场 MLDataset 中裁剪出指定股票子集。
        返回 None 表示该行业有效股票数不足。
        """
        stocks_set = set(stocks)

        sliced_panel: dict[str, pd.DataFrame] = {}
        for name, df in dataset.factor_panel.items():
            cols = [c for c in df.columns if c in stocks_set]
            if cols:
                sliced_panel[name] = df[cols].dropna(how="all")

        fwd_cols = [c for c in dataset.forward_return.columns if c in stocks_set]
        if not fwd_cols or not sliced_panel:
            return None

        fwd = dataset.forward_return[fwd_cols]

        # 检查跨因子共同有效股票数
        common = set.intersection(*[set(df.columns) for df in sliced_panel.values()])
        common &= set(fwd_cols)
        if len(common) < self.min_stocks:
            return None

        common_list = sorted(common)
        return MLDataset(
            factor_panel    = {n: df[common_list] for n, df in sliced_panel.items()},
            forward_return  = fwd[common_list],
            rebalance_dates = dataset.rebalance_dates,
            feature_names   = dataset.feature_names,
        )

    def _train_one(
        self,
        group_name: str,
        sub_dataset: MLDataset,
    ) -> pd.DataFrame | None:
        """对单个行业子集训练 WalkForwardTrainer，返回得分 DataFrame 或 None。"""
        sub_tag = f"ind_{group_name}"
        # 每个行业子训练器独立的产物目录，避免互相覆盖 ml_factor_scores / diagnostics
        sub_artifact = PROCESSED_DIR / f"industry_{group_name}"
        trainer = WalkForwardTrainer(
            train_windows = self.train_windows,
            val_window    = self.val_window,
            model_types   = self.model_types,
            rebalance_freq=self.rebalance_freq,
            hold_period   = self.hold_period,
            label_mode    = self.label_mode,
            wf_selection  = self.wf_selection,
            ensemble_method=self.ensemble_method,
            output_rank   = self.output_rank,
            tag           = sub_tag,
            artifact_dir  = sub_artifact,
        )
        try:
            score = trainer.fit_predict(sub_dataset)
            self.industry_ic[group_name] = trainer.ic_series
            return score
        except Exception as exc:
            logger.warning(f"  [{group_name}] 训练失败: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────────────────────

    def fit_predict(
        self,
        dataset: MLDataset,
        industry_map: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        分行业训练，合并输出全市场因子得分。

        Parameters
        ----------
        dataset      : 全市场 MLDataset（由 build_factor_dataset 构建）
        industry_map : Series/DataFrame（stock_code → sw_l2），
                       传 None 时自动从 data/industry 加载（向后兼容）。

        Returns
        -------
        score_df : DataFrame(index=调仓日, columns=全市场股票)
                   越大越优先，与 WalkForwardTrainer 输出格式完全相同。
        """
        # ── 加载行业映射 ──────────────────────────────────────────────────────
        if industry_map is None:
            logger.warning("industry_map 未传入，从磁盘加载（建议在 run.py 层直接传入）")
            from data.industry.download_industry import load_industry_map
            industry_map = load_industry_map()

        # 全市场股票列表（取所有因子覆盖的并集）
        all_stocks = pd.Index(sorted(
            set.union(*[set(df.columns) for df in dataset.factor_panel.values()])
        ))

        # ── 行业分组 ──────────────────────────────────────────────────────────
        groups = self._build_groups(all_stocks, industry_map)

        # ── 逐行业训练 ────────────────────────────────────────────────────────
        logger.info(f"开始分行业训练，共 {len(groups)} 组，模型={self.model_types}")
        all_scores: list[pd.DataFrame] = []

        for i, (group_name, stocks) in enumerate(groups.items(), 1):
            logger.info(f"  [{i}/{len(groups)}] {group_name}: {len(stocks)} 只股票")
            sub = self._slice_dataset(dataset, stocks)
            if sub is None:
                logger.debug(f"    {group_name}: 有效股票不足 {self.min_stocks}，跳过")
                continue
            score = self._train_one(group_name, sub)
            if score is not None:
                all_scores.append(score)

        if not all_scores:
            raise RuntimeError("所有行业训练均失败，请检查数据")

        # ── 合并：行业内得分各自是 pct rank → 拼合后再做全截面 rank ──────────
        merged = pd.concat(all_scores, axis=1)
        # 向量化全截面 pct rank，替代逐行 apply（O(N×M) Python 调用 → 单次 C 实现）
        self.score_df = merged.rank(pct=True, axis=1, na_option="keep")
        self.score_df.index.name = "date"

        # ── 全市场样本外 IC ────────────────────────────────────────────────────
        ic_dict: dict = {}
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None or len(y) < MIN_STOCKS_PER_DATE:
                continue
            s = self.score_df.loc[date].dropna()
            common = s.index.intersection(y.index)
            if len(common) < MIN_STOCKS_PER_DATE:
                continue
            ic_dict[date] = spearman_ic(s.loc[common].values, y.loc[common].values)

        self.ic_series = pd.Series(ic_dict)

        # ── 汇报 ──────────────────────────────────────────────────────────────
        ic      = self.ic_series.dropna()
        ic_mean = ic.mean()
        icir    = ic_mean / ic.std() if ic.std() > 0 else 0
        win     = (ic > 0).mean()
        logger.info(
            f"分行业训练完成: IC={ic_mean:.4f}, ICIR={icir:.4f}, 胜率={win:.1%}, "
            f"成功行业={len(self.industry_ic)}/{len(groups)}"
        )
        self._log_industry_ic()

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.score_df.to_parquet(PROCESSED_DIR / "ml_industry_scores.parquet")
        return self.score_df

    # ──────────────────────────────────────────────────────────────────────────
    # 诊断工具
    # ──────────────────────────────────────────────────────────────────────────

    def _log_industry_ic(self, top_n: int = 15) -> None:
        if not self.industry_ic:
            return
        rows = [
            {"行业": name, "IC均值": ic.mean(), "ICIR": ic.mean() / (ic.std() + 1e-8), "样本数": len(ic)}
            for name, ic in self.industry_ic.items()
            if len(ic) >= 3
        ]
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values("IC均值", ascending=False).reset_index(drop=True)
        lines = [f"\n各行业 IC 排名（Top {min(top_n, len(df))}）:"]
        for _, r in df.head(top_n).iterrows():
            bar = "█" * max(0, int(r["IC均值"] * 100))
            lines.append(
                f"  {r['行业']:<20} IC={r['IC均值']:>+.4f}  ICIR={r['ICIR']:>+.3f}  {bar}"
            )
        logger.info("\n".join(lines))

    def ic_summary(self) -> pd.Series:
        """返回全市场样本外 IC 汇总，格式与 WalkForwardTrainer 一致。"""
        ic = self.ic_series.dropna()
        std = ic.std()
        return pd.Series({
            "IC均值":   round(ic.mean(), 4),
            "IC标准差": round(std, 4),
            "ICIR":     round(ic.mean() / std, 4) if std > 0 else 0.0,
            "IC>0胜率": round((ic > 0).mean(), 4),
            "行业组数": len(self.industry_ic),
            "预测期数": len(ic),
        })

    def save_metrics(self, tag: str, output_dir=None) -> None:
        """保存 model_metrics_{tag}.json，与全市场模型格式一致。"""
        from pathlib import Path
        summary = self.ic_summary()
        metrics = {
            "tag":      tag,
            "IC均值":   summary["IC均值"],
            "IC标准差": summary["IC标准差"],
            "ICIR":     summary["ICIR"],
            "IC>0胜率": summary["IC>0胜率"],
            "行业组数": summary["行业组数"],
            "预测期数": summary["预测期数"],
            "industry_ic": {
                k: {"IC均值": round(v.mean(), 4), "样本数": len(v)}
                for k, v in self.industry_ic.items()
            },
        }
        out = Path(output_dir) if output_dir else Path("results") / tag
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"model_metrics_{tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"指标已保存: {path}")
