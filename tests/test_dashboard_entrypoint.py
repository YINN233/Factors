from pathlib import Path
import runpy

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_reserved_pages_directory_has_no_scripts():
    reserved = PROJECT_ROOT / "factors" / "dashboard" / "pages"
    assert not list(reserved.glob("*.py"))


def test_dashboard_entrypoint_imports_outside_project_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "factors" / "dashboard" / "app.py"),
        run_name="dashboard_import_check",
    )
    assert callable(namespace["main"])


def test_best_cache_prefers_latest_end_then_longest_coverage(tmp_path):
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "factors" / "dashboard" / "app.py"),
        run_name="dashboard_cache_check",
    )
    pd.DataFrame({"trade_date": ["2026-08-06"], "required": [1]}).to_parquet(
        tmp_path / "demo_20260801_20260806.parquet", index=False
    )
    pd.DataFrame({"trade_date": ["2026-08-06"], "required": [2]}).to_parquet(
        tmp_path / "demo_20170101_20260806.parquet", index=False
    )
    pd.DataFrame({"trade_date": ["2026-08-05"], "required": [3]}).to_parquet(
        tmp_path / "demo_20260805_20260805.parquet", index=False
    )
    selected = namespace["_best_cache"](tmp_path, "demo", required_columns={"required"})
    assert selected.name == "demo_20170101_20260806.parquet"
