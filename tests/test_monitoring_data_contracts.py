import pandas as pd

from factors.monitoring.data_audit import assert_no_future_rows, audit_frame, quality_grade_for
from factors.monitoring.contracts import QualityGrade


def test_audit_detects_duplicate_and_future_rows():
    frame = pd.DataFrame({"trade_date": ["2026-01-02", "2026-01-02", "2026-01-03"], "code": ["A", "A", "B"]})
    result = audit_frame(frame, "demo", key_columns=["trade_date", "code"], as_of_date="2026-01-02")
    assert result["duplicate_rows"] == 1
    assert result["future_rows"] == 1


def test_point_in_time_guard_rejects_future_rows():
    frame = pd.DataFrame({"ann_date": ["2026-01-01", "2026-01-03"]})
    try:
        assert_no_future_rows(frame, "ann_date", "2026-01-02")
    except ValueError as exc:
        assert "as_of_date" in str(exc)
    else:
        raise AssertionError("future row was not rejected")


def test_proxy_is_never_graded_as_native_a():
    assert quality_grade_for(native=True, point_in_time=True, proxy=True, validated=True) is QualityGrade.C
