import numpy as np
import pandas as pd

from factors.monitoring.evidence_registry import build_evidence_record, decision_status, records_frame
from factors.monitoring.walk_forward import walk_forward_summary


def test_decision_status_counts_independent_groups_not_indicators():
    records = records_frame(
        [
            build_evidence_record(
                evidence_id="basis", evidence_group="basis", direction=1, description="carry", source="tushare",
                native=True, point_in_time=True, proxy=False, coverage=1.0, sample_out_of_sample=True, cost_adjusted=True,
            ),
            build_evidence_record(
                evidence_id="volume", evidence_group="funding", direction=1, description="volume", source="tushare",
                native=True, point_in_time=True, proxy=False, coverage=1.0, sample_out_of_sample=True, cost_adjusted=True,
            ),
            build_evidence_record(
                evidence_id="turnover", evidence_group="funding", direction=1, description="turnover", source="tushare",
                native=True, point_in_time=True, proxy=False, coverage=1.0, sample_out_of_sample=True, cost_adjusted=True,
            ),
        ]
    )
    result = decision_status(records)
    assert result["support_groups"] == 2
    assert result["eligible_groups"] == 2
    assert result["status"] == "support"


def test_proxy_does_not_become_high_confidence():
    record = build_evidence_record(
        evidence_id="weight_proxy", evidence_group="basis", direction=1, description="proxy", source="tushare",
        native=True, point_in_time=True, proxy=True, coverage=1.0, sample_out_of_sample=True, cost_adjusted=True,
    )
    assert record["quality_grade"] == "C"
    assert decision_status(records_frame([record]))["status"] == "insufficient"


def test_walk_forward_lags_signal_and_reports_costs():
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "signal": [1, 1, -1, -1, 1, 1],
            "return": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )
    result = walk_forward_summary(
        frame,
        signal_column="signal",
        return_column="return",
        development_end="2020-01-03",
        validation_end="2020-01-05",
        cost_bps_per_turnover=10,
    )
    assert set(result["split"]) == {"development", "validation", "holdout"}
    assert result["observations"].sum() == 5
    assert np.isfinite(result.loc[result["split"] == "holdout", "annual_return"]).all()
