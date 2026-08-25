import pandas as pd
import pytest

from factors.dashboard.views.pair_compare import pair_evidence_matrix, select_pair_date


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-07", "2026-08-10"]),
            "long_product": ["IC", "IC"],
            "short_product": ["IM", "IM"],
            "tenor_rank": [1, 1],
            "pair_structure_status": ["favorable", "neutral"],
            "pair_quality": ["B", "B"],
            "pair_basis_spread": [-0.02, 0.0],
            "pair_historical_percentile": [15.0, 50.0],
        }
    )


def test_pair_page_uses_prior_available_date():
    row, history, actual = select_pair_date(_pairs(), "IC", "IM", 1, "2026-08-09")
    assert actual == pd.Timestamp("2026-08-07")
    assert row["pair_structure_status"] == "favorable"
    assert history["trade_date"].max() == actual


def test_pair_evidence_never_claims_relative_exposure():
    row = _pairs().iloc[0]
    matrix = pair_evidence_matrix(row, risk_status="strong")
    relative = matrix[matrix["evidence_group"].eq("relative_exposure")].iloc[0]
    assert relative["status"] == "unavailable"
    assert relative["quality"] == "D"


def test_pair_page_rejects_missing_direction():
    with pytest.raises(ValueError, match="no pair history"):
        select_pair_date(_pairs(), "IM", "IC", 1, "2026-08-09")

