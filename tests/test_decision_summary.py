import numpy as np
import pandas as pd
import pytest

from factors.monitoring.decision_summary import (
    build_decision_summary,
    classify_basis_status,
    classify_risk_appetite,
    resolve_as_of_date,
    signal_freshness,
)


def test_risk_appetite_mapping_and_conflict():
    assert classify_risk_appetite(-1, -1, "C") == "strong"
    assert classify_risk_appetite(1, 1, "C") == "weak"
    assert classify_risk_appetite(-1, 1, "C") == "mixed"
    assert classify_risk_appetite(-1, -1, "D") == "insufficient"
    assert classify_risk_appetite(np.nan, -1, "C") == "insufficient"


@pytest.mark.parametrize(
    ("percentile", "quality", "expected"),
    [(20, "B", "cheap"), (50, "B", "neutral"), (80, "B", "rich"), (10, "D", "insufficient"), (np.nan, "B", "insufficient")],
)
def test_basis_status_thresholds(percentile, quality, expected):
    assert classify_basis_status(percentile, quality) == expected


def test_signal_freshness_tracks_last_agreement_by_trading_rows():
    signals = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]),
            "volume_signal": [-1, -1, 1, 1],
            "turnover_signal": [-1, 1, -1, 1],
        }
    )
    result = signal_freshness(signals)
    assert result["last_consensus_date"].tolist() == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-06")]
    assert result["signal_age_days"].tolist() == [0, 1, 2, 0]


def test_resolve_as_of_date_uses_prior_date_and_never_future():
    dates = pd.to_datetime(["2026-08-03", "2026-08-05"])
    assert resolve_as_of_date(dates, "2026-08-04") == pd.Timestamp("2026-08-03")
    assert resolve_as_of_date(dates, None) == pd.Timestamp("2026-08-05")
    with pytest.raises(ValueError, match="no date on or before"):
        resolve_as_of_date(dates, "2026-08-01")


def test_build_summary_keeps_partial_dividend_and_c_quality_explicit():
    basis = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-05", "2026-08-06"]),
            "product": ["IC", "IC"],
            "ts_code": ["IC2609.CFX", "IC2609.CFX"],
            "expiry_date": pd.to_datetime(["2026-09-18", "2026-09-18"]),
            "is_main": [True, True],
            "raw_annualized_basis": [-0.08, -0.09],
            "annualized_basis": [-0.08, -0.085],
            "raw_historical_percentile": [15.0, 10.0],
            "dividend_source": ["unavailable", "partial_disclosed_events"],
            "basis_quality": ["D", "C"],
            "raw_basis_quality": ["B", "B"],
        }
    )
    signals = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-05", "2026-08-06"]),
            "volume_signal": [-1.0, -1.0],
            "turnover_signal": [-1.0, -1.0],
            "share_adjusted_signal": [-1.0, -1.0],
            "IC_four_factor_signal": [-1.0, -1.0],
            "signal_quality": ["C", "C"],
            "concentration_warning": [False, False],
        }
    )
    result = build_decision_summary(basis, signals)
    latest = result.iloc[-1]
    assert latest["basis_status"] == "cheap"
    assert latest["risk_appetite_status"] == "strong"
    assert latest["dividend_status"] == "partial"
    assert latest["basis_quality"] == "C"
    assert latest["raw_basis_quality"] == "B"
    assert latest["overall_evidence_status"] == "limited_support"
    assert "high_confidence" not in set(result["overall_evidence_status"])


def test_summary_does_not_turn_unavailable_adjustment_into_zero():
    basis = pd.DataFrame(
        {
            "trade_date": ["2026-08-06"],
            "product": ["IF"],
            "ts_code": ["IF2609.CFX"],
            "expiry_date": ["2026-09-18"],
            "is_main": [True],
            "raw_annualized_basis": [-0.03],
            "annualized_basis": [-0.03],
            "raw_historical_percentile": [50.0],
            "dividend_source": ["unavailable"],
            "basis_quality": ["D"],
            "raw_basis_quality": ["B"],
        }
    )
    signals = pd.DataFrame(
        {
            "trade_date": ["2026-08-06"],
            "volume_signal": [1.0],
            "turnover_signal": [1.0],
            "share_adjusted_signal": [1.0],
            "IF_four_factor_signal": [1.0],
            "signal_quality": ["C"],
            "concentration_warning": [False],
        }
    )
    result = build_decision_summary(basis, signals)
    assert pd.isna(result.loc[0, "adjusted_annualized_basis"])
    assert result.loc[0, "dividend_status"] == "unavailable"

