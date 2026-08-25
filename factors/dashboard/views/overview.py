"""Daily decision-evidence overview."""

from __future__ import annotations

import pandas as pd

from factors.dashboard.components.formatters import display_frame, quality_text, status_text
from factors.monitoring.decision_summary import resolve_as_of_date


def select_summary_date(summary: pd.DataFrame, requested_date: object | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if summary.empty or "trade_date" not in summary.columns:
        raise ValueError("summary has no trade dates")
    data = summary.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    actual = resolve_as_of_date(data["trade_date"], requested_date)
    return data[data["trade_date"].eq(actual)].copy(), actual


def overview_alerts(selected: pd.DataFrame) -> list[str]:
    alerts: list[str] = []
    if selected.empty:
        return ["所选日期没有摘要数据。"]
    dividend = set(selected.get("dividend_status", pd.Series(dtype=str)).dropna().astype(str))
    if "partial" in dividend:
        alerts.append("部分分红：当前只计入已披露且可估值的事件。")
    if "unavailable" in dividend:
        alerts.append("分红未接入：相关品种仅展示原始基差。")
    concentration = selected.get("concentration_warning", pd.Series(False, index=selected.index))
    if concentration.fillna(False).astype(bool).any():
        alerts.append("单只 ETF 成交占比超过 50%，资金信号受单产品主导。")
    if selected.get("overall_evidence_status", pd.Series(dtype=str)).eq("insufficient").any():
        alerts.append("至少一个品种关键证据不足。")
    return alerts


def _mode(values: pd.Series, default: str = "insufficient") -> str:
    clean = values.dropna().astype(str)
    return clean.mode().iloc[0] if not clean.empty else default


def _is_stale(latest: pd.Timestamp, today: pd.Timestamp | None = None) -> bool:
    now = (today or pd.Timestamp.today()).normalize()
    if latest.normalize() >= now:
        return False
    return len(pd.bdate_range(latest.normalize() + pd.Timedelta(days=1), now)) > 3


def render(st, summary: pd.DataFrame, coverage: pd.DataFrame | None = None) -> None:
    st.header("今日摘要")
    if summary.empty:
        st.error("缺少今日摘要派生数据。请先运行 monitoring build/backfill。")
        return
    data = summary.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    latest = data["trade_date"].max()
    requested = st.date_input(
        "分析日期",
        value=latest.date(),
        min_value=data["trade_date"].min().date(),
        max_value=latest.date(),
    )
    selected, actual = select_summary_date(data, requested)
    if actual.date() != requested:
        st.info(f"{requested} 不是可用交易日，实际采用 {actual.date()}。")

    risk = _mode(selected["risk_appetite_status"])
    overall = _mode(selected["overall_evidence_status"])
    qualities = selected.get("signal_quality", pd.Series(["D"])).dropna().astype(str)
    quality = max(qualities, key=lambda value: {"A": 0, "B": 1, "C": 2, "D": 3}.get(value, 3)) if not qualities.empty else "D"
    status_cols = st.columns(4)
    status_cols[0].metric("实际交易日", actual.strftime("%Y-%m-%d"))
    status_cols[1].metric("整体风险偏好", status_text(risk))
    status_cols[2].metric("证据状态", status_text(overall))
    status_cols[3].metric("ETF证据质量", quality_text(quality))
    if _is_stale(latest):
        st.error(f"数据可能过期，最新交易日为 {latest.date()}。")
    for message in overview_alerts(selected):
        st.warning(message)

    st.subheader("四品种状态")
    columns = [
        "product",
        "main_contract",
        "expiry_date",
        "raw_annualized_basis",
        "adjusted_annualized_basis",
        "basis_percentile",
        "basis_status",
        "risk_appetite_status",
        "four_factor_status",
        "overall_evidence_status",
        "raw_basis_quality",
        "signal_quality",
    ]
    st.dataframe(display_frame(selected.sort_values("product"), columns), width="stretch", hide_index=True)

    st.subheader("证据矩阵")
    matrix = selected[
        [
            column
            for column in [
                "product",
                "basis_status",
                "risk_appetite_status",
                "four_factor_status",
                "dividend_status",
                "overall_evidence_status",
            ]
            if column in selected.columns
        ]
    ]
    st.dataframe(display_frame(matrix), width="stretch", hide_index=True)

    history = data[data["trade_date"].le(actual)].sort_values("trade_date").groupby("product", group_keys=False).tail(20)
    if not history.empty:
        curve = history.pivot_table(index="trade_date", columns="product", values="basis_percentile", aggfunc="last")
        st.subheader("最近20个交易日基差分位")
        st.line_chart(curve, height=280)
    with st.expander("数据口径与覆盖", expanded=False):
        if coverage is not None and not coverage.empty:
            st.dataframe(coverage, width="stretch", hide_index=True)
        st.dataframe(selected, width="stretch", hide_index=True)
    st.download_button(
        "下载当日摘要",
        selected.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"decision_summary_{actual.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
