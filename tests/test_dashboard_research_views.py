import pandas as pd

from factors.dashboard.views.basis import select_basis_date, term_structure
from factors.dashboard.views.etf_risk import select_etf_date, etf_status_snapshot


def test_basis_research_view_uses_prior_date_and_selected_product():
    basis = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-07", "2026-08-07", "2026-08-10"]),
            "product": ["IC", "IC", "IC"],
            "tenor_rank": [1, 2, 1],
            "expiry_date": pd.to_datetime(["2026-09-18", "2026-10-16", "2026-09-18"]),
            "raw_annualized_basis": [-0.08, -0.06, -0.07],
        }
    )
    selected, actual = select_basis_date(basis, "2026-08-09")
    assert actual == pd.Timestamp("2026-08-07")
    curve = term_structure(selected, "IC")
    assert curve["tenor_rank"].tolist() == [1, 2]


def test_etf_research_view_reports_actual_date_and_state():
    signals = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-07", "2026-08-10"]),
            "volume_signal": [-1.0, 1.0],
            "turnover_signal": [-1.0, -1.0],
            "IC_four_factor_signal": [-1.0, 1.0],
            "signal_quality": ["C", "C"],
            "concentration_warning": [False, False],
        }
    )
    selected, actual = select_etf_date(signals, "2026-08-09")
    assert actual == pd.Timestamp("2026-08-07")
    snapshot = etf_status_snapshot(selected.iloc[-1], "IC")
    assert snapshot["risk_appetite_status"] == "strong"
    assert snapshot["four_factor_status"] == "strong"
    assert snapshot["quality"] == "C"
