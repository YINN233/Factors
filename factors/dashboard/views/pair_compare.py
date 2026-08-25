"""Directional index-futures pair comparison view."""

from __future__ import annotations

import pandas as pd

from factors.dashboard.components.formatters import TENOR_LABELS, display_frame, quality_text, status_text
from factors.monitoring.decision_summary import resolve_as_of_date


def select_pair_date(
    pairs: pd.DataFrame,
    long_product: str,
    short_product: str,
    tenor_rank: int,
    requested_date: object | None,
) -> tuple[pd.Series, pd.DataFrame, pd.Timestamp]:
    if long_product == short_product:
        raise ValueError("long and short products must differ")
    data = pairs.copy()
    data["trade_date"] = pd.to_datetime(data.get("trade_date"), errors="coerce")
    history = data[
        data["long_product"].eq(long_product)
        & data["short_product"].eq(short_product)
        & pd.to_numeric(data["tenor_rank"], errors="coerce").eq(int(tenor_rank))
    ].sort_values("trade_date")
    if history.empty:
        raise ValueError("no pair history for selected direction and tenor")
    actual = resolve_as_of_date(history["trade_date"], requested_date)
    visible_history = history[history["trade_date"].le(actual)].copy()
    row = visible_history[visible_history["trade_date"].eq(actual)].iloc[-1]
    return row, visible_history, actual


def pair_evidence_matrix(row: pd.Series, *, risk_status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_group": "pair_basis",
                "status": row.get("pair_structure_status", "insufficient"),
                "quality": row.get("pair_quality", "D"),
                "reason": "同期限原始年化基差差值及其PIT历史分位",
            },
            {
                "evidence_group": "etf_environment",
                "status": risk_status,
                "quality": "C",
                "reason": "仅表示整体风险偏好，不判断指数相对强弱",
            },
            {
                "evidence_group": "relative_exposure",
                "status": "unavailable",
                "quality": "D",
                "reason": "行业、主题及AI相对暴露尚未接入",
            },
        ]
    )


def render(st, pairs: pd.DataFrame, summary: pd.DataFrame | None = None) -> None:
    st.header("配对比较")
    if pairs.empty:
        st.error("缺少配对基差派生数据。请先运行 monitoring build/backfill。")
        return
    products = sorted(set(pairs["long_product"].dropna().astype(str)) | set(pairs["short_product"].dropna().astype(str)))
    if len(products) < 2:
        st.error("可用品种不足，无法进行配对比较。")
        return
    controls = st.columns(4)
    long_default = products.index("IC") if "IC" in products else 0
    long_product = controls[0].selectbox("多头品种", products, index=long_default)
    short_options = [product for product in products if product != long_product]
    short_default = short_options.index("IM") if "IM" in short_options else 0
    short_product = controls[1].selectbox("空头品种", short_options, index=short_default)
    direction = pairs[pairs["long_product"].eq(long_product) & pairs["short_product"].eq(short_product)]
    tenors = sorted(pd.to_numeric(direction["tenor_rank"], errors="coerce").dropna().astype(int).unique())
    if not tenors:
        st.error("所选方向没有可比较期限。")
        return
    tenor_rank = controls[2].selectbox(
        "期限",
        tenors,
        index=0,
        format_func=lambda value: TENOR_LABELS.get(value, f"第{value}期限"),
    )
    direction_dates = pd.to_datetime(direction["trade_date"], errors="coerce").dropna()
    requested = controls[3].date_input(
        "分析日期",
        value=direction_dates.max().date(),
        min_value=direction_dates.min().date(),
        max_value=direction_dates.max().date(),
    )
    row, history, actual = select_pair_date(pairs, long_product, short_product, tenor_rank, requested)
    if actual.date() != requested:
        st.info(f"{requested} 不是可用交易日，实际采用 {actual.date()}。")

    metrics = st.columns(4)
    metrics[0].metric("多头合约", str(row.get("long_contract", "不可用")))
    metrics[1].metric("空头合约", str(row.get("short_contract", "不可用")))
    spread = row.get("pair_basis_spread")
    metrics[2].metric("配对基差差值", "不可用" if pd.isna(spread) else f"{float(spread):.2%}")
    metrics[3].metric(
        "基差结构",
        status_text(row.get("pair_structure_status")),
        delta=quality_text(row.get("pair_quality")),
        delta_color="off",
    )

    comparison_columns = [
        "trade_date",
        "long_product",
        "short_product",
        "tenor_rank",
        "long_contract",
        "short_contract",
        "long_expiry_date",
        "short_expiry_date",
        "expiry_gap_days",
        "long_raw_annualized_basis",
        "short_raw_annualized_basis",
        "pair_basis_spread",
        "pair_historical_percentile",
        "pair_structure_status",
        "pair_quality",
    ]
    st.subheader("两腿对照")
    st.dataframe(display_frame(pd.DataFrame([row]), comparison_columns), width="stretch", hide_index=True)

    chart = history.set_index("trade_date")[["pair_historical_percentile"]].copy()
    chart["有利阈值"] = 20.0
    chart["不利阈值"] = 80.0
    chart = chart.rename(columns={"pair_historical_percentile": "配对历史分位"})
    st.subheader("配对历史分位")
    st.line_chart(chart, height=300)

    risk_status = "insufficient"
    if summary is not None and not summary.empty:
        current = summary.copy()
        current["trade_date"] = pd.to_datetime(current["trade_date"], errors="coerce")
        current = current[current["trade_date"].eq(actual)]
        if not current.empty and "risk_appetite_status" in current.columns:
            risk_status = str(current["risk_appetite_status"].mode().iloc[0])
    matrix = pair_evidence_matrix(row, risk_status=risk_status)
    st.subheader("证据矩阵")
    st.dataframe(display_frame(matrix), width="stretch", hide_index=True)
    if row.get("pair_structure_status") == "favorable":
        st.success("当前同期限基差结构对指定多空方向有利；这不代表相对收益观点已经得到验证。")
    elif row.get("pair_structure_status") == "unfavorable":
        st.warning("当前同期限基差结构对指定多空方向不利。")
    else:
        st.info("当前配对基差结构中性或证据不足。")
    with st.expander("计算依据", expanded=False):
        st.code("配对基差差值 = 多头原始年化基差 - 空头原始年化基差")
        st.dataframe(history.tail(120), width="stretch", hide_index=True)
    st.download_button(
        "下载配对历史",
        history.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"pair_{long_product}_{short_product}_tenor{tenor_rank}.csv",
        mime="text/csv",
    )
