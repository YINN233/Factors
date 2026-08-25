"""ETF risk-appetite research view."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.dashboard.components.formatters import display_frame, quality_text, status_text
from factors.monitoring.decision_summary import (
    classify_risk_appetite,
    resolve_as_of_date,
    signal_freshness,
)


def select_etf_date(signals: pd.DataFrame, requested_date: object | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if signals.empty or "trade_date" not in signals.columns:
        raise ValueError("ETF signals have no trade dates")
    data = signals.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    actual = resolve_as_of_date(data["trade_date"], requested_date)
    return data[data["trade_date"].eq(actual)].copy(), actual


def _direction_status(value: object, quality: object) -> str:
    if str(quality).upper() == "D" or pd.isna(value):
        return "insufficient"
    number = float(value)
    if number == -1:
        return "strong"
    if number == 1:
        return "weak"
    return "mixed"


def etf_status_snapshot(row: pd.Series, product: str) -> dict[str, object]:
    quality = str(row.get("signal_quality", "D"))
    return {
        "risk_appetite_status": classify_risk_appetite(
            row.get("volume_signal"), row.get("turnover_signal"), quality
        ),
        "volume_status": _direction_status(row.get("volume_signal"), quality),
        "turnover_status": _direction_status(row.get("turnover_signal"), quality),
        "share_status": _direction_status(row.get("share_adjusted_signal", np.nan), quality),
        "four_factor_status": _direction_status(row.get(f"{product}_four_factor_signal", np.nan), quality),
        "quality": quality,
        "concentration_warning": bool(row.get("concentration_warning", False)),
    }


def render(
    st,
    groups: pd.DataFrame,
    signals: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
) -> None:
    st.header("ETF 风险偏好")
    if groups.empty or signals.empty:
        st.error("缺少 ETF 风险偏好派生数据。请先运行 monitoring build/backfill。")
        return
    data = signal_freshness(signals)
    products = [product for product in ("IH", "IF", "IC", "IM") if f"{product}_four_factor_signal" in data.columns]
    if not products:
        st.error("ETF 信号中缺少四品种行情修正字段。")
        return
    controls = st.columns(2)
    product = controls[0].selectbox("行情修正品种", products, index=0)
    latest = data["trade_date"].max()
    requested = controls[1].date_input(
        "分析日期",
        value=latest.date(),
        min_value=data["trade_date"].min().date(),
        max_value=latest.date(),
    )
    selected, actual = select_etf_date(data, requested)
    if actual.date() != requested:
        st.info(f"{requested} 不是可用交易日，实际采用 {actual.date()}。")
    row = selected.iloc[-1]
    snapshot = etf_status_snapshot(row, product)
    metrics = st.columns(4)
    metrics[0].metric("整体风险偏好", status_text(snapshot["risk_appetite_status"]))
    metrics[1].metric("成交量证据", status_text(snapshot["volume_status"]))
    metrics[2].metric("换手率证据", status_text(snapshot["turnover_status"]))
    metrics[3].metric(
        f"{product}价格修正",
        status_text(snapshot["four_factor_status"]),
        delta=quality_text(snapshot["quality"]),
        delta_color="off",
    )
    if snapshot["concentration_warning"]:
        st.warning("单只 ETF 成交额占所在分组超过 50%，当前资金信号受单产品主导。")
    last_consensus = row.get("last_consensus_date")
    age = row.get("signal_age_days")
    if pd.notna(last_consensus):
        st.caption(f"最近一次成交与换手方向一致：{pd.Timestamp(last_consensus).date()}；当前状态持续 {int(age)} 个交易日。")

    evidence = pd.DataFrame(
        [
            {"evidence_group": "volume", "status": snapshot["volume_status"], "quality": snapshot["quality"], "reason": "5日成交活跃度相对变化"},
            {"evidence_group": "turnover", "status": snapshot["turnover_status"], "quality": snapshot["quality"], "reason": "5日换手率相对变化"},
            {"evidence_group": "shares", "status": snapshot["share_status"], "quality": snapshot["quality"], "reason": "20日ETF份额变化修正"},
            {"evidence_group": f"{product}_price", "status": snapshot["four_factor_status"], "quality": snapshot["quality"], "reason": "对应期货20日行情修正"},
        ]
    )
    st.subheader("资金证据")
    st.dataframe(display_frame(evidence), width="stretch", hide_index=True)

    history = data[data["trade_date"].le(actual)].tail(120)
    chart_columns = [column for column in ["volume_spread", "turnover_spread", "share_spread"] if column in history.columns]
    if chart_columns:
        st.subheader("最近120个交易日资金差值")
        chart = history.set_index("trade_date")[chart_columns].rename(
            columns={"volume_spread": "成交量差值", "turnover_spread": "换手率差值", "share_spread": "份额差值"}
        )
        st.line_chart(chart, height=300)

    product_rows = []
    for current_product in products:
        current = etf_status_snapshot(row, current_product)
        product_rows.append(
            {
                "product": current_product,
                "risk_appetite_status": current["risk_appetite_status"],
                "four_factor_status": current["four_factor_status"],
                "signal_quality": current["quality"],
            }
        )
    st.subheader("四品种行情修正")
    st.dataframe(display_frame(pd.DataFrame(product_rows)), width="stretch", hide_index=True)

    if validation is not None and not validation.empty:
        st.subheader("分阶段成本后验证")
        validation_columns = [
            "product",
            "split",
            "observations",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "turnover",
            "cost_bps_per_turnover",
        ]
        st.dataframe(display_frame(validation, validation_columns), width="stretch", hide_index=True)

    with st.expander("ETF 分类、集中度与原始聚合", expanded=False):
        if panel is not None and not panel.empty:
            latest_panel = panel.copy()
            latest_panel["trade_date"] = pd.to_datetime(latest_panel["trade_date"], errors="coerce")
            latest_panel = latest_panel[latest_panel["trade_date"].le(actual)].sort_values("trade_date").groupby("ts_code", group_keys=False).tail(1)
            detail_columns = [column for column in ["ts_code", "category", "risk_bucket", "reviewed", "trade_date", "close", "amount", "fd_share"] if column in latest_panel]
            st.dataframe(latest_panel[detail_columns], width="stretch", hide_index=True)
        st.dataframe(groups.tail(40), width="stretch", hide_index=True)
        st.dataframe(selected, width="stretch", hide_index=True)
    st.download_button(
        "下载 ETF 信号",
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name="etf_signals.csv",
        mime="text/csv",
    )
