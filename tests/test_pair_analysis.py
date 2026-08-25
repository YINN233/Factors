import numpy as np
import pandas as pd
import pytest

from factors.monitoring.pair_analysis import (
    build_pair_basis_history,
    classify_pair_structure,
)


def _basis_fixture() -> pd.DataFrame:
    rows = []
    values = {
        "2026-08-03": {"IF": -0.02, "IC": -0.08},
        "2026-08-04": {"IF": -0.01, "IC": -0.05},
        "2026-08-05": {"IF": -0.03, "IC": -0.06},
    }
    for trade_date, products in values.items():
        for product, value in products.items():
            rows.append(
                {
                    "trade_date": trade_date,
                    "product": product,
                    "ts_code": f"{product}2609.CFX",
                    "expiry_date": "2026-09-18",
                    "days_to_expiry": (pd.Timestamp("2026-09-18") - pd.Timestamp(trade_date)).days,
                    "tenor_rank": 1,
                    "raw_annualized_basis": value,
                    "annualized_basis": value,
                    "dividend_source": "unavailable",
                    "basis_quality": "D",
                    "raw_basis_quality": "B",
                    "is_main": True,
                }
            )
    return pd.DataFrame(rows)


def test_pair_spread_is_directional_and_reverses_with_legs():
    result = build_pair_basis_history(_basis_fixture())
    date = pd.Timestamp("2026-08-03")
    long_ic = result[
        result["trade_date"].eq(date)
        & result["long_product"].eq("IC")
        & result["short_product"].eq("IF")
    ].iloc[0]
    long_if = result[
        result["trade_date"].eq(date)
        & result["long_product"].eq("IF")
        & result["short_product"].eq("IC")
    ].iloc[0]
    assert np.isclose(long_ic["pair_basis_spread"], -0.06)
    assert np.isclose(long_if["pair_basis_spread"], 0.06)
    assert np.isclose(long_ic["pair_basis_spread"], -long_if["pair_basis_spread"])


def test_pair_history_is_unique_and_excludes_same_product_pairs():
    result = build_pair_basis_history(_basis_fixture())
    keys = ["trade_date", "long_product", "short_product", "tenor_rank"]
    assert not result.duplicated(keys).any()
    assert not result["long_product"].eq(result["short_product"]).any()
    assert set(zip(result["long_product"], result["short_product"])) == {("IF", "IC"), ("IC", "IF")}


def test_pair_requires_comparable_expiry_dates():
    basis = _basis_fixture()
    basis.loc[basis["product"].eq("IC"), "expiry_date"] = "2026-10-16"
    result = build_pair_basis_history(basis)
    assert set(result["pair_structure_status"]) == {"insufficient"}
    assert result["pair_historical_percentile"].isna().all()


def test_pair_point_in_time_percentile_does_not_change_when_future_is_added():
    basis = _basis_fixture()
    initial = build_pair_basis_history(basis[basis["trade_date"].ne("2026-08-05")])
    full = build_pair_basis_history(basis)
    keys = ["trade_date", "long_product", "short_product", "tenor_rank"]
    left = initial[keys + ["pair_historical_percentile"]].sort_values(keys).reset_index(drop=True)
    right = full[full["trade_date"].le(pd.Timestamp("2026-08-04"))][keys + ["pair_historical_percentile"]]
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize(
    ("percentile", "quality", "expected"),
    [(20.0, "B", "favorable"), (50.0, "B", "neutral"), (80.0, "B", "unfavorable"), (10.0, "D", "insufficient"), (np.nan, "B", "insufficient")],
)
def test_pair_structure_thresholds(percentile, quality, expected):
    assert classify_pair_structure(percentile, quality) == expected


def test_pair_missing_raw_basis_is_not_filled_with_zero():
    basis = _basis_fixture()
    basis.loc[(basis["product"].eq("IC")) & (basis["trade_date"].eq("2026-08-04")), "raw_annualized_basis"] = np.nan
    result = build_pair_basis_history(basis)
    row = result[
        result["trade_date"].eq(pd.Timestamp("2026-08-04"))
        & result["long_product"].eq("IC")
        & result["short_product"].eq("IF")
    ].iloc[0]
    assert pd.isna(row["pair_basis_spread"])
    assert row["pair_structure_status"] == "insufficient"


def test_pair_input_requires_different_products():
    one_product = _basis_fixture().query("product == 'IF'")
    assert build_pair_basis_history(one_product).empty

