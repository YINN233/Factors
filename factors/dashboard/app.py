"""Streamlit entry point for the local monitoring dashboard.

Run with ``streamlit run factors/dashboard/app.py`` after installing
Streamlit.  The app never calls Tushare while rendering; use the CLI refresh
command separately.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import sys


# Streamlit executes the entry file as a script and may set ``sys.path[0]``
# to this directory, especially after a page reload.  Make the repository
# root importable before loading the ``factors`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from factors.monitoring.contracts import default_raw_dir
from factors.dashboard.views import basis, etf_risk, overview, pair_compare


def _cache_range(path: Path, stem: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    match = re.fullmatch(rf"{re.escape(stem)}_(\d{{8}})_(\d{{8}})\.parquet", path.name)
    if not match:
        return None
    start, end = pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))
    return start, end


def _best_cache(directory: Path, stem: str, *, required_columns: set[str] | None = None) -> Path | None:
    candidates: list[tuple[pd.Timestamp, pd.Timedelta, float, Path]] = []
    for path in directory.glob(f"{stem}_*.parquet"):
        date_range = _cache_range(path, stem)
        if date_range is None:
            continue
        start, end = date_range
        if required_columns:
            try:
                columns = set(pd.read_parquet(path).columns)
            except Exception:
                continue
            if not required_columns.issubset(columns):
                continue
        candidates.append((end, end - start, path.stat().st_mtime, path))
    return max(candidates, key=lambda item: item[:3])[3] if candidates else None


def _read(path: Path | None) -> pd.DataFrame:
    return pd.read_parquet(path) if path is not None and path.exists() else pd.DataFrame()


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is not installed; run `pip install streamlit` first.") from exc
    st.set_page_config(page_title="股指期货监测驾驶舱", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px;}
        [data-testid="stMetric"] {border: 1px solid #d9dee7; border-radius: 6px; padding: 0.8rem; background: #ffffff;}
        [data-testid="stMetricLabel"] {font-size: 0.82rem; color: #4b5563;}
        [data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    raw_dir = Path(os.environ.get("MONITORING_RAW_DIR", default_raw_dir()))
    resolved_raw = raw_dir.resolve()
    if resolved_raw.name == "monitoring" and resolved_raw.parent.name == "raw" and resolved_raw.parent.parent.name == "data":
        processed = resolved_raw.parent.parent / "processed" / "monitoring"
    else:
        processed = resolved_raw / "processed" / "monitoring"
    summary_path = _best_cache(processed, "decision_summary", required_columns={"trade_date", "product"})
    pair_path = _best_cache(processed, "pair_basis", required_columns={"trade_date", "long_product", "short_product"})
    basis_path = _best_cache(processed, "basis_table", required_columns={"trade_date", "product"})
    groups_path = _best_cache(processed, "etf_groups", required_columns={"trade_date"})
    signals_path = _best_cache(processed, "etf_signals", required_columns={"trade_date", "signal_quality"})
    panel_path = _best_cache(processed, "etf_panel", required_columns={"trade_date", "ts_code"})
    validation_path = _best_cache(processed, "etf_validation", required_columns={"product", "split"})
    coverage_path = _best_cache(processed, "data_coverage", required_columns={"dataset", "status"})
    gaps_path = processed / "company_database_gaps.csv"
    audit_path = raw_dir / "data_availability.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()

    summary = _read(summary_path)
    pairs = _read(pair_path)
    basis_data = _read(basis_path)
    groups = _read(groups_path)
    signals = _read(signals_path)
    panel = _read(panel_path)
    validation = _read(validation_path)
    st.title("股指期货监测驾驶舱")
    if not summary.empty and "trade_date" in summary.columns:
        cutoff = pd.to_datetime(summary["trade_date"], errors="coerce").max()
        st.sidebar.metric("数据截止", cutoff.strftime("%Y-%m-%d") if pd.notna(cutoff) else "不可用")
    if summary_path is not None:
        updated = pd.Timestamp(summary_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        st.sidebar.caption(f"缓存更新：{updated}")
    coverage = _read(coverage_path)
    gaps = pd.read_csv(gaps_path) if gaps_path.exists() else pd.DataFrame()
    if not coverage.empty or not gaps.empty:
        with st.sidebar.expander("数据覆盖与公司数据库缺口", expanded=False):
            if not coverage.empty:
                st.dataframe(coverage, width="stretch", hide_index=True)
            if not gaps.empty:
                st.dataframe(gaps, width="stretch", hide_index=True)
    page = st.sidebar.radio("页面", ["今日摘要", "配对比较", "基差与期限结构", "ETF 风险偏好"])
    if page == "今日摘要":
        overview.render(st, summary, coverage)
    elif page == "配对比较":
        pair_compare.render(st, pairs, summary)
    elif page == "基差与期限结构":
        basis.render(st, basis_data, audit)
    else:
        etf_risk.render(st, groups, signals, panel, validation)


if __name__ == "__main__":
    main()
