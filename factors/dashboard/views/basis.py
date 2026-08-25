"""Basis and dividend research view."""

from __future__ import annotations

import pandas as pd

from factors.dashboard.components.formatters import display_frame, quality_text
from factors.monitoring.decision_summary import resolve_as_of_date


def select_basis_date(basis: pd.DataFrame, requested_date: object | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if basis.empty or "trade_date" not in basis.columns:
        raise ValueError("basis has no trade dates")
    data = basis.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    actual = resolve_as_of_date(data["trade_date"], requested_date)
    return data[data["trade_date"].eq(actual)].copy(), actual


def term_structure(selected: pd.DataFrame, product: str) -> pd.DataFrame:
    data = selected[selected["product"].eq(product)].copy()
    if "tenor_rank" in data.columns:
        data["tenor_rank"] = pd.to_numeric(data["tenor_rank"], errors="coerce")
        sort_columns = [column for column in ["tenor_rank", "expiry_date", "ts_code"] if column in data.columns]
        data = data.sort_values(sort_columns, kind="stable")
    return data.reset_index(drop=True)


def render(st, basis: pd.DataFrame, audit: pd.DataFrame | None = None) -> None:
    st.header("基差与期限结构")
    if basis.empty:
        st.error("缺少基差派生数据。请先运行 monitoring build/backfill。")
        return
    data = basis.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["expiry_date"] = pd.to_datetime(data["expiry_date"], errors="coerce")
    products = [product for product in ("IH", "IF", "IC", "IM") if product in set(data["product"])]
    controls = st.columns(2)
    product = controls[0].selectbox("品种", products, index=0)
    latest = data["trade_date"].max()
    requested = controls[1].date_input(
        "分析日期",
        value=latest.date(),
        min_value=data["trade_date"].min().date(),
        max_value=latest.date(),
    )
    selected, actual = select_basis_date(data, requested)
    selected_product = term_structure(selected, product)
    if actual.date() != requested:
        st.info(f"{requested} 不是可用交易日，实际采用 {actual.date()}。")
    if selected_product.empty:
        st.warning("所选日期和品种没有可用合约。")
        return

    main_mask = (
        selected_product["is_main"].fillna(False).astype(bool)
        if "is_main" in selected_product.columns
        else pd.Series(False, index=selected_product.index)
    )
    main = selected_product[main_mask]
    current = (main if not main.empty else selected_product).iloc[0]
    source = str(current.get("dividend_source", "unavailable"))
    metrics = st.columns(4)
    metrics[0].metric("主力合约", str(current.get("ts_code", "不可用")))
    raw = current.get("raw_annualized_basis")
    metrics[1].metric(
        "原始年化基差",
        "不可用" if pd.isna(raw) else f"{float(raw):.2%}",
        delta=quality_text(current.get("raw_basis_quality", "D")),
        delta_color="off",
    )
    adjusted = current.get("annualized_basis") if source != "unavailable" else pd.NA
    metrics[2].metric(
        "部分含分红年化基差",
        "不可用" if pd.isna(adjusted) else f"{float(adjusted):.2%}",
        delta=quality_text(current.get("basis_quality", "D")),
        delta_color="off",
    )
    percentile = current.get("raw_historical_percentile", current.get("historical_percentile"))
    metrics[3].metric("原始基差历史分位", "不可用" if pd.isna(percentile) else f"{float(percentile):.1f}")
    if source == "unavailable":
        st.warning("该日期未接入可用分红估计，含分红字段保持不可用。")
    elif source != "disclosed_events":
        st.warning("当前只计入已披露且可估值的分红事件，不包含卖方预测分红。")

    structure_columns = [
        "product",
        "ts_code",
        "expiry_date",
        "days_to_expiry",
        "tenor_rank",
        "raw_basis",
        "expected_dividend_points",
        "dividend_adjusted_basis",
        "raw_annualized_basis",
        "annualized_basis",
        "raw_historical_percentile",
        "raw_basis_quality",
        "basis_quality",
    ]
    st.subheader("当日期限结构")
    st.dataframe(display_frame(selected_product, structure_columns), width="stretch", hide_index=True)
    curve_columns = [column for column in ["raw_annualized_basis", "annualized_basis"] if column in selected_product]
    if curve_columns:
        curve = selected_product.set_index("expiry_date")[curve_columns].rename(
            columns={"raw_annualized_basis": "原始年化基差", "annualized_basis": "部分含分红年化基差"}
        )
        if source == "unavailable" and "部分含分红年化基差" in curve.columns:
            curve = curve.drop(columns="部分含分红年化基差")
        st.line_chart(curve, height=280)

    history_main = (
        data["is_main"].fillna(False).astype(bool)
        if "is_main" in data.columns
        else pd.Series(False, index=data.index)
    )
    history = data[data["product"].eq(product) & history_main & data["trade_date"].le(actual)].sort_values("trade_date")
    if not history.empty:
        st.subheader("主力原始年化基差历史")
        st.line_chart(history.set_index("trade_date")[["raw_annualized_basis"]].rename(columns={"raw_annualized_basis": product}), height=300)

    with st.expander("原始记录与刷新审计", expanded=False):
        st.dataframe(selected_product, width="stretch", hide_index=True)
        if audit is not None and not audit.empty:
            st.dataframe(audit.tail(40), width="stretch", hide_index=True)
    st.download_button(
        "下载基差明细",
        history.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"basis_{product}_{actual.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
