"""
External alpha metadata and validation helpers.

External signals are treated as unverified research inputs by default.  They
must pass the local validation gate before they are allowed into model feature
sets or portfolio construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Sequence

import pandas as pd


class AlphaAvailability(str, Enum):
    DIRECT = "direct"
    PROXY = "proxy"
    PARTIAL = "partial"
    PRECOMPUTED = "precomputed"
    SKIPPED = "skipped"


class AlphaValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ExternalAlphaMetadata:
    source: str
    source_factor_id: str
    factor_name: str
    version: str
    release_date: str | None = None
    expression: str = ""
    local_expression: str = ""
    description: str = ""
    required_columns: Sequence[str] = ()
    missing_columns: Sequence[str] = ()
    availability: AlphaAvailability = AlphaAvailability.DIRECT
    validation_status: AlphaValidationStatus = AlphaValidationStatus.PENDING
    proxy_reason: str = ""
    skip_reason: str = ""
    used_in_model: bool = False

    def to_record(self) -> dict:
        out = asdict(self)
        out["availability"] = self.availability.value
        out["validation_status"] = self.validation_status.value
        out["required_columns"] = ",".join(self.required_columns)
        out["missing_columns"] = ",".join(self.missing_columns)
        return out


STANDARD_ALPHA_COLUMNS = [
    "trade_date",
    "ts_code",
    "factor_name",
    "factor_value",
    "source",
    "version",
    "release_date",
]


def metadata_frame(metadata: Iterable[ExternalAlphaMetadata]) -> pd.DataFrame:
    return pd.DataFrame([item.to_record() for item in metadata])


def validate_external_alpha_panel(
    panel: pd.DataFrame,
    required_columns: Sequence[str] = STANDARD_ALPHA_COLUMNS,
) -> pd.DataFrame:
    missing = sorted(set(required_columns) - set(panel.columns))
    if missing:
        raise ValueError(f"external alpha panel missing columns: {missing}")
    out = panel[list(required_columns)].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce")
    out["factor_value"] = pd.to_numeric(out["factor_value"], errors="coerce")
    if out["trade_date"].isna().any():
        raise ValueError("external alpha panel contains invalid trade_date values")
    if out[["ts_code", "factor_name", "source", "version"]].isna().any().any():
        raise ValueError("external alpha panel contains missing identifier values")
    return out.sort_values(["trade_date", "factor_name", "ts_code"]).reset_index(drop=True)


def wide_to_external_alpha_panel(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    source: str,
    version: str,
    release_date: str | None = None,
) -> pd.DataFrame:
    required = {"trade_date", "ts_code"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"wide alpha frame missing columns: {missing}")
    panel = df[["trade_date", "ts_code"] + list(factor_cols)].melt(
        id_vars=["trade_date", "ts_code"],
        value_vars=list(factor_cols),
        var_name="factor_name",
        value_name="factor_value",
    )
    panel["source"] = source
    panel["version"] = version
    panel["release_date"] = release_date
    return validate_external_alpha_panel(panel)

