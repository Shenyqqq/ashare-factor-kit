"""
models/industry_trainer.py  —  分行业 Walk-Forward 训练

策略：
    对每个申万二级行业单独训练一个模型（WalkForwardTrainer），
    最后把所有行业的预测得分合并为全市场截面得分。

优势：
    - 行业内特征分布差异大（金融/科技/消费用的因子有效性完全不同）
    - 同行业内排名更稳定，不被跨行业的尺度差异污染
    - 自动适配行业牛熊轮动（各行业模型独立调权）

设计细节：
    - 股票数 < MIN_STOCKS_INDUSTRY 的小行业 → 合并到对应的申万一级桶里训练
    - 各行业得分独立输出 → 最终合并时做截面 rank normalization，保证可比性
    - 合并后的得分与全市场模型得分兼容，可直接送入 backtest/engine.py

用法:
    from models.industry_trainer import IndustryWalkForwardTrainer
    trainer = IndustryWalkForwardTrainer(model_types=["lgbm", "xgb"])
    score_df = trainer.fit_predict(dataset)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from models.trainer import (
    WalkForwardTrainer, MLDataset, build_ml_dataset,
    MIN_STOCKS_PER_DATE, REBALANCE_FREQ, MODEL_TYPES,
    TRAIN_WINDOWS_MONTHS, VAL_WINDOW_MONTHS,
)
from config.settings import PROCESSED_DIR

MIN_STOCKS_INDUSTRY = 30   # 低于此数合并到一级行业


class IndustryWalkForwardTrainer:
    """
    分行业 Walk-Forward 训练器。

    fit_predict() 返回全市场因子得分 DataFrame，
    格式与 WalkForwardTrainer 完全相同，可直接对接回测引擎。

    额外属性：
        industry_ic: dict[行业名 → ic_series]   各行业样本外IC
        small_industries: list[str]              被合并的小行业
    """

    def __init__(
        self,
        model_types: list = MODEL_TYPES,
        train_windows: list = TRAIN_WINDOWS_MONTHS,
        val_window: int = VAL_WINDOW_MONTHS,
        min_stocks: int = MIN_STOCKS_INDUSTRY,
    ):
        self.model_types   = model_types
        self.train_windows = train_windows
        self.val_window    = val_window
        self.min_stocks    = min_stocks

        self.industry_ic: dict[str, pd.Series] = {}
        self.small_industries: list[str]        = []
        self.score_df: pd.DataFrame             = None
        self.ic_series: pd.Series               = None

    # ── 数据准备 ────────────────────────────────────────────────────────────────

    def _load_industry_map(self) -> pd.Series:
        """加载行业映射，返回 Series(code → industry_label)"""
        from data.industry.download_industry import load_industry_map
        ind_map = load_industry_map()
        return ind_map["sw_l2"]   # 申万二级

    def _get_industry_groups(
        self,
        all_stocks: pd.Index,
        industry_map: pd.Series,
        rebalance_dates: list,
        forward_return: pd.DataFrame,
    ) -> dict[str, list]:
        """
        把股票按行业分组，小行业合并到一级行业桶。
        返回 {group_label: [stock_codes]}
        """
        # 对齐到实际价格宇宙
        mapped = industry_map.reindex(all_stocks).fillna("未分类")

        from data.industry.download_industry import load_industry_map
        ind_map_full = load_industry_map()
        l1_map = ind_map_full["sw_l1"].reindex(all_stocks).fillna("未分类")

        groups: dict[str, list] = {}
        for l2, codes in mapped.groupby(mapped):
            stocks = codes.index.tolist()
            if len(stocks) < self.min_stocks:
                # 合并到一级行业
                l1 = l1_map[stocks[0]] if stocks else "未分类"
                bucket = f"_l1_{l1}"
                groups.setdefault(bucket, []).extend(stocks)
                self.small_industries.append(l2)
            else:
                groups[l2] = stocks

        logger.info(
            f"行业分组: {len(groups)}组 | "
            f"小行业合并: {len(self.small_industries)}个"
        )
        return groups

    # ── 核心训练 ─────────────────────────────────────────────────────────────────

    def _train_one_industry(
        self,
        group_name: str,
        stocks: list,
        dataset: MLDataset,
    ) -> pd.DataFrame | None:
        """
        对单个行业股票池训练 WalkForwardTrainer。
        返回该行业的因子得分 DataFrame（index=date, columns=stocks）
        """
        # 过滤 factor_panel 到该行业
        industry_factor = {
            name: df[df.columns.intersection(stocks)].dropna(how="all")
            for name, df in dataset.factor_panel.items()
        }
        fwd = dataset.forward_return[
            dataset.forward_return.columns.intersection(stocks)
        ].dropna(how="all")

        # 检查有效股票数
        common_stocks = sorted(
            set.intersection(*[set(df.columns) for df in industry_factor.values()])
            & set(fwd.columns)
        )
        if len(common_stocks) < self.min_stocks:
            logger.debug(f"  {group_name}: 有效股票数不足({len(common_stocks)})，跳过")
            return None

        # 裁剪因子面板到该行业
        industry_factor_filtered = {
            name: df[common_stocks]
            for name, df in industry_factor.items()
        }
        fwd_filtered = fwd[common_stocks]

        sub_dataset = MLDataset(
            factor_panel=industry_factor_filtered,
            forward_return=fwd_filtered,
            rebalance_dates=dataset.rebalance_dates,
            feature_names=dataset.feature_names,
        )

        trainer = WalkForwardTrainer(
            train_windows=self.train_windows,
            val_window=self.val_window,
            model_types=self.model_types,
        )

        try:
            score = trainer.fit_predict(sub_dataset)
            self.industry_ic[group_name] = trainer.ic_series
            return score
        except Exception as e:
            logger.warning(f"  {group_name} 训练失败: {e}")
            return None

    def fit_predict(self, dataset: MLDataset) -> pd.DataFrame:
        """
        主入口：分行业训练，合并输出全市场因子得分。

        dataset: 全市场 MLDataset（factor_panel 覆盖所有股票）
        """
        # 所有股票（取 factor_panel 中覆盖最广的因子列）
        all_stocks = sorted(
            set.union(*[set(df.columns) for df in dataset.factor_panel.values()])
        )
        industry_map = self._load_industry_map()

        groups = self._get_industry_groups(
            pd.Index(all_stocks), industry_map,
            dataset.rebalance_dates, dataset.forward_return,
        )

        logger.info(f"开始分行业训练，共 {len(groups)} 组...")
        all_scores: list[pd.DataFrame] = []

        for i, (group_name, stocks) in enumerate(groups.items()):
            logger.info(f"[{i+1}/{len(groups)}] {group_name}: {len(stocks)}只股票")
            score = self._train_one_industry(group_name, stocks, dataset)
            if score is not None:
                all_scores.append(score)

        if not all_scores:
            raise RuntimeError("所有行业训练均失败")

        # ── 合并：各行业分拼接，截面 rank normalization ──────────────────────
        # 每个调仓日，把各行业的得分拼在一起，再做全市场截面 rank
        merged = pd.concat(all_scores, axis=1)

        # 截面 rank normalization：确保全市场得分可比
        # 各行业输出的是 [0,1] rank，合并后再做一次全截面排名
        self.score_df = merged.apply(
            lambda row: row.rank(pct=True, na_option="keep"), axis=1
        )
        self.score_df.index.name = "date"

        # ── 全市场样本外 IC ──────────────────────────────────────────────────
        from models.trainer import spearman_ic
        ic_dict = {}
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = self.score_df.loc[date].dropna()
            y = y.reindex(s.index).dropna()
            s = s.loc[y.index]
            if len(s) >= MIN_STOCKS_PER_DATE:
                ic_dict[date] = spearman_ic(s.values, y.values)
        self.ic_series = pd.Series(ic_dict)

        # ── 汇报 ────────────────────────────────────────────────────────────────
        ic_mean = self.ic_series.mean()
        icir    = ic_mean / self.ic_series.std() if self.ic_series.std() > 0 else 0
        logger.info(
            f"分行业训练完成: "
            f"IC={ic_mean:.4f}, ICIR={icir:.4f}, "
            f"胜率={(self.ic_series>0).mean():.1%}"
        )
        self._print_industry_ic()

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.score_df.to_parquet(PROCESSED_DIR / "ml_industry_scores.parquet")
        return self.score_df

    def _print_industry_ic(self):
        """打印各行业IC排名"""
        if not self.industry_ic:
            return
        rows = [
            {"行业": name, "IC均值": ic.mean(), "样本数": len(ic)}
            for name, ic in self.industry_ic.items()
            if len(ic) > 0
        ]
        if not rows:
            return
        df = (pd.DataFrame(rows)
              .sort_values("IC均值", ascending=False)
              .reset_index(drop=True))
        logger.info("\n各行业IC排名（Top10）:")
        for _, r in df.head(10).iterrows():
            bar_len = max(0, int(r["IC均值"] * 100))
            bar = "█" * bar_len
            logger.info(f"  {r['行业']:<16} IC={r['IC均值']:>6.4f}  {bar}")

    def ic_summary(self) -> pd.Series:
        ic = self.ic_series.dropna()
        return pd.Series({
            "IC均值":   round(ic.mean(), 4),
            "IC标准差": round(ic.std(), 4),
            "ICIR":     round(ic.mean() / ic.std(), 4),
            "IC>0胜率": round((ic > 0).mean(), 4),
            "行业数":   len(self.industry_ic),
        })
