"""UI / Live 页超小 smoke：import、控件字符串、predict 入口可 mock。

不读 ``data/raw`` 真实 parquet、不训模型、不起 Streamlit server。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd

import ui.app as app


def test_import_ui_app():
    assert app.REPO_ROOT.is_dir()
    assert hasattr(app, "_render_live_tab")
    assert hasattr(app, "run_fold_predict")
    assert hasattr(app, "main")
    assert app.LIVE_TAB_LABEL == "Live 候选股"
    assert app.LIVE_PRIMARY_BUTTON == "按 fold 出分"
    assert app.LIVE_COVERAGE_BUTTON == "检查数据覆盖率"
    assert app.FLAGSHIP_WF_REL.endswith("xgb_h5_sizeind_w156_nob_wf_20260830")


def test_app_module_has_no_eager_parquet_calls():
    """``import ui.app`` 路径上不应有模块级 read_parquet / _load_data。"""
    src = Path(app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"read_parquet", "_load_data", "incremental_download"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        dumped = ast.dump(node)
        for name in banned:
            assert name not in dumped, f"模块级出现 {name}"


def test_live_page_copy_and_controls():
    src = inspect.getsource(app._render_live_tab)
    assert "LIVE_PRIMARY_BUTTON" in src
    assert "LIVE_COVERAGE_BUTTON" in src
    assert "Top30" in src
    assert "fit_date" in src
    assert "run_fold_predict" in src
    assert "models_manifest" in src
    help_txt = app.LIVE_PATH_HELP
    assert "predict_candidates" in help_txt
    assert "flagship_last_window" in help_txt
    assert "备选" in help_txt
    assert "下单" in help_txt


def test_predict_candidates_entrypoint_exists():
    from live.predict_from_wf_models import predict, predict_candidates

    assert callable(predict_candidates)
    assert predict is predict_candidates


def test_run_fold_predict_mocked(monkeypatch):
    sentinel = pd.DataFrame(
        {
            "code": ["000001"],
            "name": ["测试"],
            "rank": [1],
            "score": [0.5],
            "fit_date": ["2026-08-14"],
            "signal_date": ["2026-08-28"],
            "suggested_buy_date": ["2026-08-31"],
        }
    )
    fit = pd.Timestamp("2026-08-14")

    def fake(**kwargs):
        assert kwargs["as_of"] == "2026-08-28"
        assert kwargs["top_n"] == 30
        return sentinel, fit

    monkeypatch.setattr(
        "live.predict_from_wf_models.predict_candidates",
        fake,
    )
    top, got = app.run_fold_predict(
        as_of="2026-08-28",
        model_dir="results/fake_tag",
        top_n=30,
    )
    assert list(top["code"]) == ["000001"]
    assert got == fit


def test_candidate_sanity_flags_no_disk():
    df = pd.DataFrame(
        {
            "code": ["000001", "200001", "920001", "600530"],
            "name": ["平安银行", "深B测试", "北交所票", "*ST某股"],
        }
    )
    flags = app.candidate_sanity_flags(df)
    assert flags["n"] == 4
    assert flags["n_b"] == 1
    assert flags["n_92"] == 1
    assert flags["n_st_name"] == 1
    assert flags["b_codes"] == ["200001"]
    assert flags["st_codes"] == ["600530"]


def test_list_wf_model_dirs_tmp(tmp_path: Path):
    tag = tmp_path / "xgb_h5_sizeind_w156_nob_wf_20260830"
    (tag / "models").mkdir(parents=True)
    (tag / "models" / "models_manifest.json").write_text("[]", encoding="utf-8")
    (tmp_path / "no_models").mkdir()
    found = app.list_wf_model_dirs(results_dir=tmp_path)
    assert len(found) == 1
    assert "xgb_h5_sizeind_w156_nob_wf_20260830" in found[0]


def test_peek_coverage_missing_file_no_real_raw(tmp_path: Path):
    info = app.peek_raw_panel_coverage("circ_mv.parquet", root=tmp_path)
    assert info["exists"] is False
    assert app.peek_last_index_date("close_hfq.parquet", root=tmp_path) is None
