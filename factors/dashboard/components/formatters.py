"""Display-only formatting for the monitoring dashboard."""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd


COLUMN_LABELS = {
    "trade_date": "交易日",
    "product": "品种",
    "main_contract": "主力合约",
    "ts_code": "合约代码",
    "expiry_date": "到期日",
    "long_product": "多头品种",
    "short_product": "空头品种",
    "long_contract": "多头合约",
    "short_contract": "空头合约",
    "long_expiry_date": "多头到期日",
    "short_expiry_date": "空头到期日",
    "expiry_gap_days": "到期差(天)",
    "days_to_expiry": "剩余天数",
    "tenor_rank": "期限序号",
    "close": "期货收盘",
    "index_close": "指数收盘",
    "raw_basis": "原始基差",
    "expected_dividend_points": "已披露分红点数",
    "dividend_adjusted_basis": "部分含分红基差",
    "raw_annualized_basis": "原始年化基差",
    "adjusted_annualized_basis": "部分含分红年化基差",
    "annualized_basis": "含分红年化基差",
    "long_raw_annualized_basis": "多头原始年化基差",
    "short_raw_annualized_basis": "空头原始年化基差",
    "pair_basis_spread": "配对基差差值",
    "historical_percentile": "历史分位",
    "raw_historical_percentile": "原始基差历史分位",
    "basis_percentile": "历史分位",
    "pair_historical_percentile": "配对历史分位",
    "basis_status": "基差状态",
    "pair_structure_status": "配对结构",
    "risk_appetite_status": "风险偏好",
    "four_factor_status": "价格修正状态",
    "overall_evidence_status": "证据状态",
    "dividend_status": "分红口径",
    "dividend_source": "分红来源",
    "basis_quality": "含分红质量",
    "raw_basis_quality": "原始基差质量",
    "pair_quality": "配对质量",
    "signal_quality": "ETF信号质量",
    "last_consensus_date": "最近一致日期",
    "signal_age_days": "信号持续交易日",
    "concentration_warning": "集中度警告",
    "evidence_group": "证据组",
    "status": "状态",
    "quality": "质量",
    "reason": "依据",
    "split": "样本阶段",
    "observations": "样本数",
    "annual_return": "年化收益",
    "annual_volatility": "年化波动",
    "sharpe": "夏普比率",
    "max_drawdown": "最大回撤",
    "turnover": "平均换手",
    "cost_bps_per_turnover": "单次换手成本(bps)",
}

STATUS_LABELS = {
    "strong": "风险偏好偏强",
    "weak": "风险偏好偏弱",
    "mixed": "证据混合",
    "insufficient": "证据不足",
    "limited_support": "支持有限",
    "cheap": "相对便宜",
    "neutral": "中性",
    "rich": "相对偏贵",
    "favorable": "基差结构有利",
    "unfavorable": "基差结构不利",
    "partial": "部分已披露",
    "disclosed": "已披露",
    "unavailable": "未接入",
    "stale": "数据过期",
    "fresh": "数据正常",
}

QUALITY_LABELS = {
    "A": "A · 原始且已验证",
    "B": "B · 原始/有限制",
    "C": "C · 代理/人工口径",
    "D": "D · 证据不足",
}

VALUE_LABELS = {
    "pair_basis": "配对基差结构",
    "etf_environment": "ETF整体环境",
    "relative_exposure": "相对暴露",
    "volume": "成交量",
    "turnover": "换手率",
    "shares": "ETF份额",
    "development": "开发期",
    "validation": "验证期",
    "holdout": "留出期",
}

TENOR_LABELS = {1: "近月", 2: "次月", 3: "季月", 4: "次季月"}

PERCENT_COLUMNS = {
    "raw_annualized_basis",
    "adjusted_annualized_basis",
    "annualized_basis",
    "long_raw_annualized_basis",
    "short_raw_annualized_basis",
    "pair_basis_spread",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "turnover",
}
PERCENTILE_COLUMNS = {
    "historical_percentile",
    "raw_historical_percentile",
    "basis_percentile",
    "pair_historical_percentile",
}
DATE_COLUMNS = {"trade_date", "expiry_date", "long_expiry_date", "short_expiry_date", "last_consensus_date"}
STATUS_COLUMNS = {"basis_status", "pair_structure_status", "risk_appetite_status", "four_factor_status", "overall_evidence_status", "dividend_status", "status"}
QUALITY_COLUMNS = {"basis_quality", "raw_basis_quality", "pair_quality", "signal_quality", "quality"}


def _missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_percent(value: object, digits: int = 2) -> str:
    if _missing(value):
        return "不可用"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "不可用"
    return f"{number:.{digits}%}"


def quality_text(value: object) -> str:
    if _missing(value):
        return "D · 证据不足"
    return QUALITY_LABELS.get(str(value).upper(), str(value))


def status_text(value: object) -> str:
    if _missing(value):
        return "不可用"
    return STATUS_LABELS.get(str(value), str(value))


def _format_date(value: object) -> str:
    if _missing(value):
        return "不可用"
    date = pd.to_datetime(value, errors="coerce")
    return "不可用" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d")


def _format_number(value: object, digits: int = 1) -> str:
    if _missing(value):
        return "不可用"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def display_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a formatted copy while preserving raw download data."""

    selected = [column for column in (columns or list(frame.columns)) if column in frame.columns]
    out = frame[selected].copy()
    formatters: dict[str, Callable[[object], str]] = {}
    for column in selected:
        if column in PERCENT_COLUMNS:
            formatters[column] = format_percent
        elif column in PERCENTILE_COLUMNS:
            formatters[column] = _format_number
        elif column in DATE_COLUMNS:
            formatters[column] = _format_date
        elif column in STATUS_COLUMNS:
            formatters[column] = status_text
        elif column in QUALITY_COLUMNS:
            formatters[column] = quality_text
    for column, formatter in formatters.items():
        out[column] = out[column].map(formatter)
    for column in [value for value in ("evidence_group", "split") if value in out.columns]:
        out[column] = out[column].map(lambda value: VALUE_LABELS.get(str(value), str(value)))
    if "tenor_rank" in out.columns:
        out["tenor_rank"] = out["tenor_rank"].map(
            lambda value: TENOR_LABELS.get(int(value), f"第{int(value)}期限") if pd.notna(value) else "不可用"
        )
    return out.rename(columns={column: COLUMN_LABELS.get(column, column) for column in selected})
