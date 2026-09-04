"""
models/analyzer.py  —  训练后可视化与诊断

所有方法均可独立调用，不互相依赖。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

from loguru import logger
from models.trainer import WalkForwardTrainer, spearman_ic


class MLAnalyzer:

    def __init__(self, trainer: WalkForwardTrainer):
        assert trainer.score_df is not None, "请先调用 trainer.fit_predict()"
        self.trainer = trainer

    def plot_ic(self):
        ic = self.trainer.ic_series.dropna()
        icir = ic.mean() / ic.std()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(
            f"样本外 IC（Walk-Forward）  IC均值={ic.mean():.4f}  "
            f"ICIR={icir:.4f}  胜率={(ic > 0).mean():.1%}", fontsize=11
        )

        ax = axes[0]
        ic.plot(ax=ax, alpha=0.4, color="steelblue", label="月度IC")
        ic.rolling(6).mean().plot(ax=ax, color="red", lw=1.8, label="6期滚动均值")
        ax.axhline(0,     color="black", lw=0.8)
        ax.axhline( 0.05, color="green", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(-0.05, color="green", lw=0.8, ls="--", alpha=0.6)
        ax.set_title("IC序列")
        ax.legend(fontsize=9)

        ax = axes[1]
        ic.hist(ax=ax, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(ic.mean(), color="red", lw=1.8, label=f"均值={ic.mean():.3f}")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("IC分布")
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.show()
        print(self.trainer.ic_summary().to_string())

    def plot_model_comparison(self):
        valid = {m: s.dropna() for m, s in self.trainer.model_ic.items()
                 if len(s.dropna()) > 0}
        if not valid:
            logger.warning("没有各模型独立IC数据")
            return

        stats = pd.DataFrame([{
            "模型": m,
            "IC均值": round(ic.mean(), 4),
            "ICIR":   round(ic.mean() / ic.std(), 4),
            "胜率":   round((ic > 0).mean(), 4),
        } for m, ic in valid.items()]).set_index("模型")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle("各模型样本外 IC 对比", fontsize=11)

        ax = axes[0]
        for m, ic in valid.items():
            ic.rolling(6).mean().plot(ax=ax, label=m, lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("6期滚动IC均值")
        ax.legend(fontsize=9)

        ax = axes[1]
        bars = ax.bar(stats.index, stats["IC均值"], edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("IC均值（Ridge=线性基准）")
        for bar, val in zip(bars, stats["IC均值"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.001 * np.sign(val),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()
        print(stats.to_string())

    def plot_quantile_returns(self, prices, n_groups=5, benchmark=None):
        score_df = self.trainer.score_df
        dates    = sorted(score_df.index)
        group_navs = {f"Q{g+1}": [1.0] for g in range(n_groups)}

        for i, date in enumerate(dates[:-1]):
            next_date = dates[i + 1]
            scores = score_df.loc[date].dropna()
            if len(scores) < n_groups * 5:
                for g in range(n_groups):
                    group_navs[f"Q{g+1}"].append(group_navs[f"Q{g+1}"][-1])
                continue
            for g in range(n_groups):
                lo, hi = g / n_groups, (g + 1) / n_groups
                codes = scores[(scores >= scores.quantile(lo)) &
                               (scores <  scores.quantile(hi))].index
                valid = [c for c in codes if c in prices.columns
                         and date in prices.index and next_date in prices.index]
                if not valid:
                    group_navs[f"Q{g+1}"].append(group_navs[f"Q{g+1}"][-1])
                    continue
                ret = (prices.loc[next_date, valid] / prices.loc[date, valid] - 1).mean()
                group_navs[f"Q{g+1}"].append(group_navs[f"Q{g+1}"][-1] * (1 + ret))

        nav_df = pd.DataFrame(group_navs, index=dates[:len(group_navs["Q1"])])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("分组净值（样本外 Walk-Forward）", fontsize=11)

        ax = axes[0]
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_groups))
        for i, col in enumerate(nav_df.columns):
            nav_df[col].plot(ax=ax, color=colors[i], label=col, lw=1.3)
        if benchmark is not None:
            bm = benchmark.reindex(nav_df.index, method="ffill")
            (bm / bm.iloc[0]).plot(ax=ax, color="black", lw=1, ls="--", label="基准")
        ax.set_title("各分组净值（Q1=低分，Q5=高分）")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ls_nav = nav_df["Q5"] / nav_df["Q1"]
        (ls_nav / ls_nav.iloc[0]).plot(ax=ax, color="purple", lw=1.5)
        ax.axhline(1, color="black", lw=0.8)
        ax.set_title("多空净值（Q5/Q1）")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        return nav_df

    def plot_shap(self, model_type="lgbm", window=None, sample_size=200):
        """单模型末折可视化；完整 WF 汇总请用 ``export_shap`` / 训练期 ``--shap``。"""
        from models.wf.shap_analysis import (
            compute_fold_shap_summary,
            subsample_rows,
        )

        if window is None:
            window = self.trainer.train_windows[-1]
        model = self.trainer.models.get((window, model_type))
        if model is None:
            logger.warning(f"找不到模型 ({model_type}, window={window})")
            return

        feature_names = self.trainer._dataset.feature_names
        last_date = self.trainer._dataset.rebalance_dates[-1]
        X_sample, _ = self.trainer._dataset.get_cross_section(last_date)
        if X_sample is None:
            return

        X_sample = X_sample.reindex(columns=feature_names).fillna(0)
        summary, method = compute_fold_shap_summary(
            model, model_type, X_sample.values, feature_names,
            max_samples=sample_size,
        )
        if summary.empty:
            if model_type == "ridge":
                # 与旧行为兼容：LinearExplainer 失败时仍画 |coef|
                coef_arr = (
                    model.coef_ if hasattr(model, "coef_")
                    else model.named_steps["model"].coef_
                )
                coef = pd.Series(
                    np.abs(coef_arr), index=feature_names[: len(coef_arr)]
                ).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(4, len(coef) * 0.4)))
                coef.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
                ax.set_title(f"Ridge |coef|（SHAP 不可用: {method}）")
                plt.tight_layout()
                plt.show()
                return coef
            logger.warning(f"SHAP 计算失败: {method}")
            return

        mean_abs = summary.set_index("feature")["mean_abs_shap"].sort_values()
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(mean_abs) * 0.4)))
        fig.suptitle(f"SHAP 分析（{model_type}, {method}）", fontsize=11)
        mean_abs.plot(kind="barh", ax=axes[0], color="steelblue", edgecolor="white")
        axes[0].set_title("平均 |SHAP值|")
        axes[0].set_xlabel("mean_|SHAP|")

        # 右图：若仍能拿到 raw SHAP 再画 beeswarm；否则画 share 条形
        try:
            import shap
            from models.wf.shap_analysis import compute_shap_values

            X_np = subsample_rows(X_sample.values, max_samples=sample_size)
            sv, _ = compute_shap_values(
                model, model_type, X_np, feature_names,
                max_samples=sample_size,
            )
            if sv is not None and model_type != "ridge":
                shap.summary_plot(
                    sv, X_np, feature_names=feature_names,
                    show=False, plot_size=None,
                )
            else:
                share = summary.set_index("feature")["share"].sort_values()
                share.plot(kind="barh", ax=axes[1], color="coral", edgecolor="white")
                axes[1].set_title("占比 share")
        except Exception as e:
            logger.debug(f"SHAP summary_plot 跳过: {e}")
            share = summary.set_index("feature")["share"].sort_values()
            share.plot(kind="barh", ax=axes[1], color="coral", edgecolor="white")
            axes[1].set_title("占比 share")

        plt.tight_layout()
        plt.show()
        return mean_abs

    def export_shap(
        self,
        top_n: int = 20,
        max_samples: int = 500,
        max_dates: int = 12,
    ) -> pd.DataFrame:
        """
        导出 SHAP 汇总到 ``artifact_dir``（若训练期已 ``--shap`` 则已有产物；
        本方法用末折模型 + 最近 OOS 截面补算/覆盖轻量路径）。
        """
        from models.wf.shap_analysis import compute_trainer_shap_recent

        return compute_trainer_shap_recent(
            self.trainer,
            max_samples=max_samples,
            max_dates=max_dates,
            top_n=top_n,
            export=True,
        )

    def plot_feature_importance(self):
        feature_names = self.trainer._dataset.feature_names
        results = {}
        for model_type in self.trainer.model_types:
            model = self.trainer.models.get((self.trainer.train_windows[-1], model_type))
            if model is None:
                continue
            try:
                if model_type == "ridge":
                    # ridge 为裸 Ridge 对象（无 Pipeline），直接访问 coef_
                    coef_arr = model.coef_ if hasattr(model, "coef_") else model.named_steps["model"].coef_
                    imp = np.abs(coef_arr)
                elif model_type in ("lgbm", "xgb"):
                    imp = model.feature_importances_
                elif model_type == "cat":
                    imp = model.get_feature_importance()
                results[model_type] = pd.Series(imp, index=feature_names)
            except Exception as e:
                logger.warning(f"{model_type} 特征重要性失败: {e}")

        if not results:
            return
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, max(4, len(feature_names) * 0.4)))
        if n == 1:
            axes = [axes]
        fig.suptitle("各模型特征重要性", fontsize=11)
        for ax, (mtype, imp) in zip(axes, results.items()):
            imp.sort_values(ascending=True).plot(
                kind="barh", ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(mtype)
        plt.tight_layout()
        plt.show()

    def plot_clustered_importance(
        self,
        model_type="lgbm",
        window=None,
        correlation_threshold: float = 0.7,
        n_repeats_cluster: int = 10,
        n_repeats_intra: int = 5,
        sample_size: int = 2000,
    ):
        """
        AFML Ch6 Clustered Feature Importance：先按相关性聚类，再按簇做 MDA，
        簇内单特征再分配重要性。避免相关因子重要性分裂被误删。

        在最后一个调仓日的截面 + 最近训练窗口的模型上评估。
        """
        from models.wf.clustered_importance import compute_clustered_importance

        if window is None:
            window = self.trainer.train_windows[-1]
        model = self.trainer.models.get((window, model_type))
        if model is None:
            logger.warning(f"找不到模型 ({model_type}, window={window})")
            return

        feature_names = self.trainer._dataset.feature_names
        last_date = self.trainer._dataset.rebalance_dates[-1]
        X_sample, y_sample = self.trainer._dataset.get_cross_section(last_date)
        if X_sample is None or y_sample is None:
            logger.warning("无可用截面数据评估 clustered importance")
            return

        X_sample = X_sample.iloc[:sample_size]
        y_sample = y_sample.iloc[:sample_size]
        # 对齐特征列名（与训练时一致）
        X_sample = X_sample[feature_names]

        try:
            df = compute_clustered_importance(
                model, X_sample, y_sample,
                correlation_threshold=correlation_threshold,
                n_repeats_cluster=n_repeats_cluster,
                n_repeats_intra=n_repeats_intra,
                scoring="ic",
            )
        except Exception as e:
            logger.warning(f"Clustered importance 计算失败: {e}")
            return

        if df.empty:
            logger.warning("Clustered importance 结果为空")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(df) * 0.35)))
        fig.suptitle(
            f"Clustered Feature Importance（AFML Ch6, {model_type}, "
            f"thresh={correlation_threshold})",
            fontsize=11,
        )

        # 左：簇级重要性
        clu = (df.groupby("cluster", as_index=False)
                 .agg(cluster_importance=("cluster_importance", "first"),
                      n_features=("feature", "count"))
                 .sort_values("cluster_importance", ascending=True))
        ax = axes[0]
        ax.barh(clu["cluster"], clu["cluster_importance"],
                color="steelblue", edgecolor="white")
        for y, (imp, n) in zip(clu["cluster"],
                               zip(clu["cluster_importance"], clu["n_features"])):
            ax.text(imp, y, f" n={n}", va="center", fontsize=8, color="gray")
        ax.set_title("簇级 MDA 重要性（IC 下降量）")
        ax.set_xlabel("importance (Δ IC)")
        ax.grid(alpha=0.3, axis="x")

        # 右：单特征再分配后总重要性
        ax = axes[1]
        sub = df.sort_values("total_importance", ascending=True)
        colors = plt.cm.tab20(sub["cluster"].astype("category").cat.codes / max(1, len(clu)))
        ax.barh(sub["feature"], sub["total_importance"], color=colors, edgecolor="white")
        ax.set_title("单特征总重要性（簇内再分配）")
        ax.set_xlabel("total importance")
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.show()
        print(df.to_string(index=False))
        return df

    def full_report(self, prices, benchmark=None):
        self.plot_ic()
        self.plot_model_comparison()
        self.plot_quantile_returns(prices, benchmark=benchmark)
        self.plot_feature_importance()
        # SHAP：优先画树模型；ridge 走 LinearExplainer / |coef| fallback
        for mt in self.trainer.model_types:
            if mt in ("lgbm", "xgb", "cat", "rf", "ridge"):
                self.plot_shap(model_type=mt)
        # 若训练期未开 --shap，用末折模型补一份 CSV/JSON
        if not getattr(self.trainer, "_shap_rows", None):
            try:
                self.export_shap()
            except Exception as e:
                logger.warning(f"SHAP 导出跳过: {e}")
        self.plot_clustered_importance(model_type="lgbm")
