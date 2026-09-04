"""Minimal SHAP analysis unit tests (no full WF / long train)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from models.wf.shap_analysis import (
    aggregate_shap_rows,
    append_shap_rows,
    compute_fold_shap_summary,
    export_shap_artifacts,
    subsample_rows,
    summarize_shap_matrix,
)


def test_subsample_rows_caps():
    X = np.arange(1000, dtype=float).reshape(200, 5)
    out = subsample_rows(X, max_samples=30, random_state=0)
    assert out.shape == (30, 5)


def test_ridge_shap_fallback_or_linear():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    # y driven by feature 0 and 2
    y = 2.0 * X[:, 0] - 1.5 * X[:, 2] + 0.05 * rng.normal(size=80)
    model = Ridge(alpha=1.0).fit(X, y)
    names = ["f0", "f1", "f2", "f3"]
    summary, method = compute_fold_shap_summary(
        model, "ridge", X, names, max_samples=50, random_state=1,
    )
    assert not summary.empty
    assert method in ("linear_explainer", "coef_times_centered")
    assert set(summary["feature"]) == set(names)
    # Dominant features should rank high in mean_|SHAP|
    top2 = set(summary.sort_values("mean_abs_shap", ascending=False)["feature"].head(2))
    assert "f0" in top2 or "f2" in top2
    assert abs(summary["share"].sum() - 1.0) < 1e-6


def test_tree_shap_lgbm():
    pytest.importorskip("shap")
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=100)
    model = lgb.LGBMRegressor(
        n_estimators=30, max_depth=3, verbosity=-1, random_state=0,
    )
    model.fit(X, y)
    names = ["a", "b", "c"]
    summary, method = compute_fold_shap_summary(
        model, "lgbm", X, names, max_samples=40, random_state=2,
    )
    assert not summary.empty, method
    assert method == "tree_explainer"
    top = summary.sort_values("mean_abs_shap", ascending=False).iloc[0]["feature"]
    assert top == "a"


def test_export_shap_artifacts(tmp_path: Path):
    rows = []
    for feat, mabs, w in [("x", 0.5, 1.0), ("y", 0.1, 1.0), ("x", 0.3, 1.0)]:
        append_shap_rows(
            rows,
            pd.DataFrame([{
                "feature": feat,
                "mean_abs_shap": mabs,
                "mean_shap": 0.1 if feat == "x" else -0.05,
                "share": 0.5,
            }]),
            model_type="lgbm",
            window=6,
            pred_date=pd.Timestamp("2020-01-31"),
            method="tree_explainer",
            weight=w,
        )
    paths = export_shap_artifacts(
        rows, tmp_path, "unit",
        top_n=2,
        model_weights={"lgbm": 1.0},
    )
    assert "summary" in paths
    assert paths["summary"].exists()
    assert paths["json"].exists()
    meta = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert meta["top_n"] == 2
    assert len(meta["top_features"]) <= 2
    agg = aggregate_shap_rows(rows, by=["feature"])
    assert agg.iloc[0]["feature"] == "x"


def test_summarize_share_sums_to_one():
    sv = np.array([[1.0, -2.0], [3.0, 0.0], [-1.0, 1.0]])
    df = summarize_shap_matrix(sv, ["p", "q"])
    assert abs(df["share"].sum() - 1.0) < 1e-9


def test_mlp_unsupported():
    class Dummy:
        pass

    summary, method = compute_fold_shap_summary(
        Dummy(), "mlp", np.ones((10, 2)), ["a", "b"],
    )
    assert summary.empty
    assert method.startswith("unsupported")


if __name__ == "__main__":
    test_subsample_rows_caps()
    test_ridge_shap_fallback_or_linear()
    test_tree_shap_lgbm()
    test_export_shap_artifacts(Path("_tmp_shap_test"))
    test_summarize_share_sums_to_one()
    test_mlp_unsupported()
    print("all ok")
