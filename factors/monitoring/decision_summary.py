"""Auditable dashboard summaries built from basis and ETF evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ensure_columns


def resolve_as_of_date(available_dates: object, target_date: object | None = None) -> pd.Timestamp:
    """Return the latest available date not later than the requested date."""

    dates = pd.to_datetime(pd.Series(available_dates), errors="coerce").dropna().drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("no available dates")
    if target_date is None:
        return pd.Timestamp(dates.iloc[-1])
    target = pd.Timestamp(target_date)
    eligible = dates[dates <= target]
    if eligible.empty:
        raise ValueError(f"no date on or before {target.date()}")
    return pd.Timestamp(eligible.iloc[-1])


def classify_basis_status(percentile: object, quality: object) -> str:
    value = pd.to_numeric(pd.Series([percentile]), errors="coerce").iloc[0]
    if str(quality).upper() == "D" or pd.isna(value):
        return "insufficient"
    if float(value) <= 20.0:
        return "cheap"
    if float(value) >= 80.0:
        return "rich"
    return "neutral"


def classify_risk_appetite(volume_signal: object, turnover_signal: object, quality: object) -> str:
    if str(quality).upper() == "D":
        return "insufficient"
    volume = pd.to_numeric(pd.Series([volume_signal]), errors="coerce").iloc[0]
    turnover = pd.to_numeric(pd.Series([turnover_signal]), errors="coerce").iloc[0]
    if pd.isna(volume) or pd.isna(turnover):
        return "insufficient"
    if volume == -1 and turnover == -1:
        return "strong"
    if volume == 1 and turnover == 1:
        return "weak"
    return "mixed"


def _direction_status(value: object, quality: object) -> str:
    if str(quality).upper() == "D":
        return "insufficient"
    signal = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(signal):
        return "insufficient"
    if signal == -1:
        return "strong"
    if signal == 1:
        return "weak"
    return "mixed"


def signal_freshness(signals: pd.DataFrame) -> pd.DataFrame:
    """Track the last day on which volume and turnover signals agreed."""

    ensure_columns(signals, ["trade_date", "volume_signal", "turnover_signal"])
    out = signals.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date", kind="stable").reset_index(drop=True)
    last_date = pd.NaT
    age: int | pd._libs.missing.NAType = pd.NA
    dates: list[pd.Timestamp | pd.NaT] = []
    ages: list[int | pd._libs.missing.NAType] = []
    for row in out[["trade_date", "volume_signal", "turnover_signal"]].itertuples(index=False):
        volume = pd.to_numeric(pd.Series([row.volume_signal]), errors="coerce").iloc[0]
        turnover = pd.to_numeric(pd.Series([row.turnover_signal]), errors="coerce").iloc[0]
        agrees = pd.notna(volume) and pd.notna(turnover) and volume == turnover and volume in (-1, 1)
        if agrees:
            last_date = row.trade_date
            age = 0
        elif pd.notna(last_date):
            age = int(age) + 1
        dates.append(last_date)
        ages.append(age)
    out["last_consensus_date"] = pd.to_datetime(pd.Series(dates), errors="coerce")
    out["signal_age_days"] = pd.Series(ages, dtype="Int64")
    return out


def _dividend_status(value: object) -> str:
    source = str(value)
    if source == "disclosed_events":
        return "disclosed"
    if source in {"partial_disclosed_events", "estimated_events", "proxy"}:
        return "partial"
    return "unavailable"


def _overall_status(row: pd.Series) -> str:
    if row.get("raw_basis_quality") == "D" or row.get("signal_quality") == "D":
        return "insufficient"
    if row.get("basis_status") == "insufficient" or row.get("risk_appetite_status") == "insufficient":
        return "insufficient"
    if row.get("basis_status") == "neutral" or row.get("risk_appetite_status") == "mixed":
        return "mixed"
    return "limited_support"


def _evidence_reason(row: pd.Series) -> str:
    return ";".join(
        [
            f"basis={row.get('basis_status', 'insufficient')}",
            f"risk={row.get('risk_appetite_status', 'insufficient')}",
            f"basis_quality={row.get('raw_basis_quality', 'D')}",
            f"signal_quality={row.get('signal_quality', 'D')}",
            "relative_exposure=unavailable",
        ]
    )


def build_decision_summary(basis: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Build one conservative dashboard row per date and futures product."""

    if basis.empty or signals.empty:
        return pd.DataFrame()
    ensure_columns(
        basis,
        [
            "trade_date",
            "product",
            "ts_code",
            "expiry_date",
            "is_main",
            "raw_annualized_basis",
            "annualized_basis",
            "dividend_source",
            "basis_quality",
        ],
    )
    ensure_columns(
        signals,
        ["trade_date", "volume_signal", "turnover_signal", "signal_quality", "concentration_warning"],
    )
    base = basis.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce")
    base["expiry_date"] = pd.to_datetime(base["expiry_date"], errors="coerce")
    base = base[base["is_main"].fillna(False).astype(bool)].copy()
    keys = ["trade_date", "product"]
    if base.duplicated(keys).any():
        raise ValueError("basis contains multiple main contracts per product/date")
    if "raw_basis_quality" not in base.columns:
        base["raw_basis_quality"] = "D"
    if "raw_historical_percentile" not in base.columns:
        base["raw_historical_percentile"] = np.nan
    base = base.rename(
        columns={
            "ts_code": "main_contract",
            "raw_historical_percentile": "basis_percentile",
        }
    )

    signal_data = signal_freshness(signals)
    signal_columns = [
        "trade_date",
        "volume_signal",
        "turnover_signal",
        "share_adjusted_signal",
        "signal_quality",
        "concentration_warning",
        "last_consensus_date",
        "signal_age_days",
    ]
    for product in sorted(base["product"].dropna().astype(str).unique()):
        column = f"{product}_four_factor_signal"
        if column in signal_data.columns:
            signal_columns.append(column)
    signal_columns = list(dict.fromkeys(column for column in signal_columns if column in signal_data.columns))
    signal_data = signal_data[signal_columns].drop_duplicates("trade_date", keep="last")
    out = base.merge(signal_data, on="trade_date", how="left", validate="many_to_one")
    out["adjusted_annualized_basis"] = out["annualized_basis"].where(
        out["dividend_source"].ne("unavailable")
    )
    out["basis_status"] = [
        classify_basis_status(percentile, quality)
        for percentile, quality in zip(out["basis_percentile"], out["raw_basis_quality"])
    ]
    out["dividend_status"] = out["dividend_source"].map(_dividend_status)
    out["volume_status"] = [
        _direction_status(value, quality) for value, quality in zip(out["volume_signal"], out["signal_quality"])
    ]
    out["turnover_status"] = [
        _direction_status(value, quality) for value, quality in zip(out["turnover_signal"], out["signal_quality"])
    ]
    out["risk_appetite_status"] = [
        classify_risk_appetite(volume, turnover, quality)
        for volume, turnover, quality in zip(out["volume_signal"], out["turnover_signal"], out["signal_quality"])
    ]
    four_factor_values = []
    for row in out.itertuples(index=False):
        column = f"{row.product}_four_factor_signal"
        four_factor_values.append(getattr(row, column, np.nan))
    out["four_factor_status"] = [
        _direction_status(value, quality) for value, quality in zip(four_factor_values, out["signal_quality"])
    ]
    out["overall_evidence_status"] = out.apply(_overall_status, axis=1)
    out["evidence_reasons"] = out.apply(_evidence_reason, axis=1)
    keep = [
        "trade_date",
        "product",
        "main_contract",
        "expiry_date",
        "raw_annualized_basis",
        "adjusted_annualized_basis",
        "basis_percentile",
        "basis_status",
        "dividend_status",
        "volume_status",
        "turnover_status",
        "risk_appetite_status",
        "four_factor_status",
        "last_consensus_date",
        "signal_age_days",
        "concentration_warning",
        "basis_quality",
        "raw_basis_quality",
        "signal_quality",
        "overall_evidence_status",
        "evidence_reasons",
    ]
    return out[keep].sort_values(["trade_date", "product"], kind="stable").reset_index(drop=True)

