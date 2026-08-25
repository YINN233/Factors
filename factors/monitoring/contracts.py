"""Shared contracts for the index-futures monitoring pipeline.

The monitoring code deliberately keeps source metadata alongside values.  A
proxy can be useful for exploration, but it must never look like a native
production field in downstream reports.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


INDEX_CODES: dict[str, str] = {
    "IH": "000016.SH",
    "IF": "000300.SH",
    "IC": "000905.SH",
    "IM": "000852.SH",
}
FUTURE_PRODUCTS = tuple(INDEX_CODES)


class QualityGrade(StrEnum):
    """Evidence quality used by the dashboard."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class AvailabilityRecord:
    """One endpoint/data-set coverage record."""

    dataset: str
    source: str
    requested_start: str | None = None
    requested_end: str | None = None
    rows: int = 0
    date_min: str | None = None
    date_max: str | None = None
    coverage: float | None = None
    missing_reason: str | None = None
    quality_grade: str = QualityGrade.B.value
    point_in_time: bool = False
    proxy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_date(value: Any) -> pd.Timestamp | None:
    """Parse a Tushare date-like value without raising on missing values."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    result = pd.to_datetime(value, errors="coerce")
    if pd.isna(result):
        return None
    return pd.Timestamp(result)


def date_text(value: Any) -> str | None:
    parsed = as_date(value)
    return parsed.strftime("%Y-%m-%d") if parsed is not None else None


def normalize_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with known date columns normalized to ``datetime64``."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def normalize_future_code(code: str | None) -> str | None:
    """Convert CFFEX's ``.CFX`` suffix to the display convention only."""

    if code is None or pd.isna(code):
        return None
    text = str(code)
    return text.replace(".CFX", ".CFE")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a concise error when a required normalized field is absent."""

    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")


def default_raw_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "monitoring"
