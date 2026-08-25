"""Data quality and point-in-time audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .contracts import AvailabilityRecord, QualityGrade, date_text


def audit_frame(
    df: pd.DataFrame,
    dataset: str,
    *,
    key_columns: Iterable[str] = (),
    date_column: str | None = "trade_date",
    requested_start: str | None = None,
    requested_end: str | None = None,
    source: str = "tushare",
    quality_grade: QualityGrade | str = QualityGrade.B,
    point_in_time: bool = False,
    proxy: bool = False,
    as_of_date: str | None = None,
) -> dict:
    """Summarize coverage, duplicates and future-dated rows.

    The function is intentionally pure: callers decide where to persist the
    audit record.  ``future_rows`` is a hard warning and should block a
    historical signal build.
    """

    keys = list(key_columns)
    missing_keys = sorted(set(keys).difference(df.columns))
    duplicate_rows = int(df.duplicated(keys).sum()) if not missing_keys and keys else 0
    date_min = date_max = None
    future_rows = 0
    observed_dates = 0
    expected_dates = 0
    if date_column and date_column in df.columns and not df.empty:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            date_min = valid_dates.min().strftime("%Y-%m-%d")
            date_max = valid_dates.max().strftime("%Y-%m-%d")
            observed_dates = int(valid_dates.nunique())
            if as_of_date is not None:
                future_rows = int((valid_dates > pd.Timestamp(as_of_date)).sum())
            if requested_start and requested_end:
                expected_dates = int(
                    pd.date_range(requested_start, requested_end, freq="B").size
                )

    coverage = None
    if expected_dates:
        coverage = min(1.0, observed_dates / expected_dates)

    return {
        "dataset": dataset,
        "source": source,
        "rows": int(len(df)),
        "date_min": date_min,
        "date_max": date_max,
        "coverage": coverage,
        "missing_keys": missing_keys,
        "duplicate_rows": duplicate_rows,
        "future_rows": future_rows,
        "quality_grade": str(quality_grade),
        "point_in_time": bool(point_in_time),
        "proxy": bool(proxy),
        "requested_start": date_text(requested_start),
        "requested_end": date_text(requested_end),
    }


def append_audit(records: list[dict], path: str | Path) -> pd.DataFrame:
    """Append audit records to a CSV without losing previous runs."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(records)
    if target.exists():
        previous = pd.read_csv(target)
        current = pd.concat([previous, current], ignore_index=True)
    current.to_csv(target, index=False)
    return current


def assert_no_future_rows(df: pd.DataFrame, date_column: str, as_of_date: str) -> None:
    """Fail fast when a PIT data set contains records after its as-of date."""

    if date_column not in df.columns:
        return
    dates = pd.to_datetime(df[date_column], errors="coerce")
    future = df.loc[dates > pd.Timestamp(as_of_date)]
    if not future.empty:
        raise ValueError(
            f"{len(future)} rows in {date_column} are after as_of_date={as_of_date}"
        )


def quality_grade_for(*, native: bool, point_in_time: bool, proxy: bool, validated: bool) -> QualityGrade:
    """Apply the conservative A/B/C/D policy from the design document."""

    if proxy:
        return QualityGrade.C
    if validated and native and point_in_time:
        return QualityGrade.A
    if native or point_in_time:
        return QualityGrade.B
    return QualityGrade.D
