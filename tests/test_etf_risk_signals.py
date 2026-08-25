import numpy as np
import pandas as pd

from factors.monitoring.etf_risk_signals import SignalWindows, aggregate_risk_groups, build_etf_panel, build_risk_signals
from factors.monitoring.etf_universe import (
    apply_classification,
    classification_review_template,
    load_classification,
    stock_exchange_fund_universe,
)


def _fund_basic():
    return pd.DataFrame(
        {
            "ts_code": ["L.SH", "R.SH"],
            "fund_type": ["股票型", "股票型"],
            "type": ["股票型", "股票型"],
            "list_date": ["2020-01-01", "2020-01-01"],
            "delist_date": [pd.NaT, pd.NaT],
            "name": ["宽基 ETF", "科技主题 ETF"],
            "benchmark": ["沪深300", "科技指数"],
        }
    )


def test_universe_is_not_classified_without_manual_mapping():
    universe = stock_exchange_fund_universe(_fund_basic(), as_of_date="2026-08-01")
    assert set(universe["classification_hint"]) == {"scale_review", "industry_or_theme_review"}


def test_reviewed_classification_and_signal_pipeline():
    dates = pd.date_range("2026-07-20", periods=10, freq="B")
    daily_rows, share_rows = [], []
    for i, date in enumerate(dates):
        for code, risk in [("L.SH", "low_risk"), ("R.SH", "risk")]:
            volume = 100.0 + i * 20 if risk == "low_risk" else 100.0
            daily_rows.append({"ts_code": code, "trade_date": date, "close": 10.0, "vol": volume, "amount": volume * 100.0})
            share_rows.append({"ts_code": code, "trade_date": date, "fd_share": 100.0 + (i * 10 if risk == "low_risk" else 0.0)})
    classification = pd.DataFrame(
        {
            "ts_code": ["L.SH", "R.SH"],
            "category": ["scale", "theme"],
            "risk_bucket": ["low_risk", "risk"],
            "effective_start": ["2020-01-01", "2020-01-01"],
            "effective_end": [pd.NaT, pd.NaT],
            "classification_basis": ["manual", "manual"],
            "reviewed": [True, True],
        }
    )
    classified = apply_classification(pd.DataFrame(daily_rows), classification)
    panel = build_etf_panel(pd.DataFrame(daily_rows), pd.DataFrame(share_rows), classified)
    groups = aggregate_risk_groups(panel)
    returns = pd.DataFrame({"trade_date": dates, "product": ["IF"] * len(dates), "return": [0.01] * len(dates)})
    signals = build_risk_signals(
        groups,
        returns,
        windows=SignalWindows(activity_window=1, share_window=1, price_window=1),
    )
    last = signals.iloc[-1]
    assert last["volume_signal"] == 1.0
    assert last["activity_consensus_signal"] == 1.0
    assert last["IF_four_factor_signal"] == 1.0
    assert last["signal_quality"] == "C"
    assert np.isclose(panel.loc[0, "turnover_proxy"], 1.0)
    assert "low_risk_largest_amount_share" in groups.columns


def test_classification_template_never_assigns_unreviewed_labels():
    template = classification_review_template(_fund_basic(), as_of_date="2026-08-01")
    assert len(template) == 2
    assert template["reviewed"].eq(False).all()
    assert template["risk_bucket"].isna().all()


def test_invalid_reviewed_classification_is_rejected(tmp_path):
    path = tmp_path / "classification.csv"
    pd.DataFrame(
        {
            "ts_code": ["L.SH"],
            "category": ["unknown"],
            "risk_bucket": ["risk"],
            "effective_start": ["2020-01-01"],
            "effective_end": [pd.NaT],
            "classification_basis": ["manual"],
            "reviewed": [True],
        }
    ).to_csv(path, index=False)
    try:
        load_classification(path)
    except ValueError as exc:
        assert "invalid_category" in str(exc)
    else:
        raise AssertionError("invalid reviewed classification was accepted")
