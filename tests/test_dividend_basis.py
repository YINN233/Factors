import numpy as np
import pandas as pd

from factors.monitoring.dividend_basis import (
    add_historical_percentile,
    build_basis_table,
    dividend_points_for_expiry,
    reweight_index_snapshot,
)
from factors.monitoring.pipeline import build_basis_from_cache


def test_basis_formula_and_annualization():
    futures = pd.DataFrame(
        {
            "product": ["IF"],
            "ts_code": ["IF2608.CFX"],
            "trade_date": ["2026-08-01"],
            "close": [99.0],
            "expiry_date": ["2026-08-11"],
        }
    )
    index = pd.DataFrame({"product": ["IF"], "trade_date": ["2026-08-01"], "close": [100.0]})
    points = pd.DataFrame(
        {
            "product": ["IF"],
            "ts_code": ["IF2608.CFX"],
            "trade_date": ["2026-08-01"],
            "expected_dividend_points": [2.0],
            "dividend_source": ["disclosed"],
        }
    )
    result = build_basis_table(futures, index, dividend_points=points)
    assert result.loc[0, "raw_basis"] == -1.0
    assert result.loc[0, "dividend_adjusted_basis"] == 1.0
    assert np.isclose(result.loc[0, "annualized_basis"], 0.365)


def test_dividend_points_use_known_events_and_do_not_fill_missing_price():
    events = pd.DataFrame(
        {
            "ts_code": ["A", "B"],
            "ann_date": ["2026-07-01", "2026-07-01"],
            "ex_date": ["2026-08-05", "2026-08-05"],
            "cash_div": [10.0, 10.0],
        }
    )
    weights = pd.DataFrame({"con_code": ["A", "B"], "trade_date": ["2026-07-31", "2026-07-31"], "weight": [50.0, 50.0]})
    prices = pd.DataFrame({"ts_code": ["A"], "trade_date": ["2026-08-01"], "close": [10.0]})
    total, detail = dividend_points_for_expiry(
        events,
        weights,
        100.0,
        as_of_date="2026-08-01",
        expiry_date="2026-08-20",
        stock_prices=prices,
    )
    # A pays 1 per share / 10 price = 10% yield * 50% * 100 = 5 points.
    assert np.isclose(total, 5.0)
    assert bool(detail.loc[detail["ts_code"] == "A", "included"].iloc[0])
    assert not bool(detail.loc[detail["ts_code"] == "B", "included"].iloc[0])


def test_monthly_weight_proxy_reweights_by_price():
    weights = pd.DataFrame(
        {
            "con_code": ["A", "B"],
            "trade_date": ["2026-08-03", "2026-08-03"],
            "weight": [50.0, 50.0],
        }
    )
    prices = pd.DataFrame(
        {
            "ts_code": ["A", "B", "A", "B"],
            "trade_date": ["2026-08-03", "2026-08-03", "2026-08-04", "2026-08-04"],
            "close": [10.0, 10.0, 20.0, 10.0],
        }
    )
    result = reweight_index_snapshot(weights, prices, start_date="2026-08-03", end_date="2026-08-04")
    day = result[result["trade_date"] == pd.Timestamp("2026-08-04")].set_index("con_code")
    assert np.isclose(day.loc["A", "weight"], 200 / 3)
    assert np.isclose(day["weight"].sum(), 100.0)
    assert set(day["weight_method"]) == {"monthly_reweighted_proxy"}


def test_historical_percentile_is_grouped():
    basis = pd.DataFrame({"product": ["IF", "IF", "IH"], "annualized_basis": [0.1, 0.2, 0.3]})
    out = add_historical_percentile(basis)
    assert np.isclose(out.loc[0, "historical_percentile"], 50.0)
    assert np.isclose(out.loc[1, "historical_percentile"], 100.0)


def test_point_in_time_percentile_does_not_use_future_observations():
    basis = pd.DataFrame(
        {
            "product": ["IF", "IF"],
            "tenor": [1, 1],
            "trade_date": ["2026-08-01", "2026-08-02"],
            "annualized_basis": [0.1, 0.2],
        }
    )
    from factors.monitoring.dividend_basis import add_historical_percentile

    out = add_historical_percentile(
        basis,
        group_columns=("product", "tenor"),
        point_in_time=True,
    )
    assert np.isclose(out.loc[0, "historical_percentile"], 100.0)
    assert np.isclose(out.loc[1, "historical_percentile"], 100.0)


def test_missing_weight_is_not_treated_as_zero_dividend():
    events = pd.DataFrame(
        {
            "ts_code": ["A"],
            "ann_date": ["2026-07-01"],
            "ex_date": ["2026-08-05"],
            "cash_div": [10.0],
        }
    )
    total, detail = dividend_points_for_expiry(
        events,
        pd.DataFrame({"con_code": ["B"], "trade_date": ["2026-07-31"], "weight": [100.0]}),
        100.0,
        as_of_date="2026-08-01",
        expiry_date="2026-08-20",
        stock_prices=pd.DataFrame({"ts_code": ["A"], "trade_date": ["2026-08-01"], "close": [10.0]}),
    )
    assert total == 0.0
    assert not bool(detail.loc[0, "included"])


def test_cached_basis_marks_all_missing_dividend_inputs_unavailable(tmp_path):
    futures = pd.DataFrame(
        {
            "product": ["IF", "IF"],
            "ts_code": ["IF2608.CFX", "IF2608.CFX"],
            "trade_date": ["2026-08-03", "2026-08-04"],
            "close": [99.0, 98.0],
            "expiry_date": ["2026-08-21", "2026-08-21"],
        }
    )
    index = pd.DataFrame(
        {"product": ["IF", "IF"], "trade_date": ["2026-08-03", "2026-08-04"], "close": [100.0, 100.0]}
    )
    mapping = pd.DataFrame(
        {"product": ["IF", "IF"], "trade_date": ["2026-08-03", "2026-08-04"], "mapping_ts_code": ["IF2608.CFX", "IF2608.CFX"]}
    )
    futures.to_parquet(tmp_path / "futures_daily_20260803_20260804.parquet", index=False)
    index.to_parquet(tmp_path / "index_daily_20260803_20260804.parquet", index=False)
    mapping.to_parquet(tmp_path / "futures_mapping_20260803_20260804.parquet", index=False)
    result = build_basis_from_cache(tmp_path, start_date="20260803", end_date="20260804", cache=True)
    assert set(result["dividend_source"]) == {"unavailable"}
    assert set(result["basis_quality"]) == {"D"}
    assert set(result["raw_basis_quality"]) == {"B"}
    assert result["raw_historical_percentile"].notna().all()
    assert set(result["historical_percentile_basis"]) == {"raw_annualized_basis"}
    assert (tmp_path / "processed" / "monitoring" / "basis_table_20260803_20260804.parquet").exists()
