"""Evidence registry and conservative decision-status aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .contracts import QualityGrade


@dataclass(frozen=True)
class QualityGate:
    min_coverage: float = 0.95
    min_independent_groups: int = 2
    require_point_in_time: bool = True
    require_cost_adjusted_validation: bool = True


def evidence_quality(
    *,
    native: bool,
    point_in_time: bool,
    proxy: bool,
    coverage: float | None,
    sample_out_of_sample: bool,
    cost_adjusted: bool,
    gate: QualityGate = QualityGate(),
) -> QualityGrade:
    """Grade evidence without mistaking a proxy for a native fact."""

    if proxy:
        return QualityGrade.C
    sufficient_coverage = coverage is not None and coverage >= gate.min_coverage
    if native and point_in_time and sufficient_coverage and sample_out_of_sample and cost_adjusted:
        return QualityGrade.A
    if native and (point_in_time or sufficient_coverage):
        return QualityGrade.B
    return QualityGrade.D


def build_evidence_record(
    *,
    evidence_id: str,
    evidence_group: str,
    direction: int | float | None,
    description: str,
    source: str,
    native: bool,
    point_in_time: bool,
    proxy: bool,
    coverage: float | None,
    sample_out_of_sample: bool = False,
    cost_adjusted: bool = False,
    sample_size: int | None = None,
    as_of_date: str | None = None,
    gate: QualityGate = QualityGate(),
) -> dict:
    grade = evidence_quality(
        native=native,
        point_in_time=point_in_time,
        proxy=proxy,
        coverage=coverage,
        sample_out_of_sample=sample_out_of_sample,
        cost_adjusted=cost_adjusted,
        gate=gate,
    )
    return {
        "evidence_id": evidence_id,
        "evidence_group": evidence_group,
        "direction": pd.NA if direction is None else int(direction),
        "description": description,
        "source": source,
        "native": native,
        "point_in_time": point_in_time,
        "proxy": proxy,
        "coverage": coverage,
        "sample_out_of_sample": sample_out_of_sample,
        "cost_adjusted": cost_adjusted,
        "sample_size": sample_size,
        "as_of_date": as_of_date,
        "quality_grade": grade.value,
    }


def decision_status(records: pd.DataFrame, *, gate: QualityGate = QualityGate()) -> dict:
    """Summarize evidence by independent group, not by raw indicator count."""

    required = {"evidence_group", "direction", "quality_grade"}
    missing = required.difference(records.columns)
    if missing:
        raise KeyError(f"missing evidence columns: {', '.join(sorted(missing))}")
    if records.empty:
        return {
            "status": "insufficient",
            "support_groups": 0,
            "opposition_groups": 0,
            "eligible_groups": 0,
            "reason": "no_evidence",
        }
    eligible = records[records["quality_grade"].isin([QualityGrade.A.value, QualityGrade.B.value])].copy()
    if gate.require_point_in_time and "point_in_time" in eligible.columns:
        eligible = eligible[eligible["point_in_time"].fillna(False)]
    if gate.require_cost_adjusted_validation and "cost_adjusted" in eligible.columns:
        eligible = eligible[eligible["cost_adjusted"].fillna(False)]
    if eligible.empty:
        return {
            "status": "insufficient",
            "support_groups": 0,
            "opposition_groups": 0,
            "eligible_groups": 0,
            "reason": "no_quality_eligible_evidence",
        }
    group_direction = eligible.groupby("evidence_group")["direction"].median()
    support = int((group_direction > 0).sum())
    opposition = int((group_direction < 0).sum())
    total = int(group_direction.size)
    if total < gate.min_independent_groups:
        status = "insufficient"
        reason = "too_few_independent_groups"
    elif support >= gate.min_independent_groups and opposition == 0:
        status = "support"
        reason = "independent_groups_agree"
    elif opposition >= gate.min_independent_groups and support == 0:
        status = "oppose"
        reason = "independent_groups_oppose"
    else:
        status = "mixed"
        reason = "evidence_conflicts"
    return {
        "status": status,
        "support_groups": support,
        "opposition_groups": opposition,
        "eligible_groups": total,
        "reason": reason,
    }


def records_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Create a stable evidence table for persistence and dashboard display."""

    frame = pd.DataFrame(list(records))
    if not frame.empty and "as_of_date" in frame.columns:
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    return frame
