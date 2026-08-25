import pandas as pd

from factors.dashboard.views.overview import overview_alerts, select_summary_date


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-07", "2026-08-07", "2026-08-10", "2026-08-10"]),
            "product": ["IF", "IC", "IF", "IC"],
            "dividend_status": ["partial", "partial", "partial", "partial"],
            "concentration_warning": [False, True, False, False],
            "overall_evidence_status": ["mixed", "mixed", "limited_support", "limited_support"],
        }
    )


def test_overview_non_trading_date_returns_actual_prior_date():
    selected, actual = select_summary_date(_summary(), "2026-08-09")
    assert actual == pd.Timestamp("2026-08-07")
    assert set(selected["trade_date"]) == {actual}


def test_overview_alerts_include_partial_dividend_and_concentration():
    selected, _ = select_summary_date(_summary(), "2026-08-07")
    alerts = overview_alerts(selected)
    assert any("部分分红" in value for value in alerts)
    assert any("单只 ETF" in value for value in alerts)

