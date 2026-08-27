import numpy as np
import pandas as pd

from factors.risk.multifrequency_specific_risk import (
    SpecificRiskConfig,
    aggregate_complete_period_returns,
    apply_structural_shrinkage,
    blend_frequency_variances,
    estimate_multifrequency_specific_risk,
)


def test_period_aggregation_is_non_overlapping_and_excludes_current_period():
    dates = pd.date_range("2024-01-01", "2024-03-15", freq="B")
    residuals = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["A"] * len(dates),
            "specific_return": np.ones(len(dates)) * 0.001,
        }
    )

    monthly = aggregate_complete_period_returns(residuals, frequency="M")

    assert monthly["period"].astype(str).tolist() == ["2024-01", "2024-02"]
    assert monthly["observations"].sum() == len(dates[dates < "2024-03-01"])
    assert np.allclose(monthly["period_return"], monthly["observations"] * 0.001)


def test_frequency_blend_renormalizes_available_variance_weights():
    frame = pd.DataFrame(
        {
            "specific_variance_daily_component": [4.0, np.nan],
            "specific_variance_monthly_component": [9.0, 9.0],
            "specific_variance_quarterly_component": [np.nan, 16.0],
        }
    )

    out = blend_frequency_variances(frame)

    assert np.isclose(out.loc[0, "specific_variance_blended"], (0.5 * 4.0 + 0.3 * 9.0) / 0.8)
    assert np.isclose(out.loc[0, "specific_weight_daily"], 0.5 / 0.8)
    assert np.isclose(out.loc[1, "specific_variance_blended"], (0.3 * 9.0 + 0.2 * 16.0) / 0.5)
    assert out.loc[1, "specific_weight_daily"] == 0.0


def test_structural_shrinkage_uses_group_then_industry_then_market_prior():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {
            "trade_date": [date] * 6,
            "ts_code": list("ABCDEF"),
            "industry_sw_l1_code": ["I1", "I1", "I1", "I2", "I2", "I3"],
            "total_mv": [10, 11, 12, 100, 110, 1000],
            "specific_variance_blended": [1.0, 2.0, 3.0, 8.0, 10.0, np.nan],
            "specific_effective_observations": [252, 126, 63, 252, 126, 0],
        }
    )

    out = apply_structural_shrinkage(frame, group_minimum=3, reliability_target=252, size_buckets=1)

    by_code = out.set_index("ts_code")
    assert by_code.loc["A", "specific_prior_level"] == "industry_size"
    assert by_code.loc["D", "specific_prior_level"] == "market"
    assert by_code.loc["F", "specific_prior_level"] == "market"
    assert by_code.loc["F", "specific_reliability"] == 0.0
    assert np.isfinite(by_code.loc["F", "specific_variance_daily"])


def _specific_panel(end: str = "2024-12-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-02", end, freq="B")
    residual_rows = []
    exposure_rows = []
    for stock_idx, code in enumerate(["A", "B", "C", "D", "E", "F"]):
        values = np.sin(np.arange(len(dates)) / (7.0 + stock_idx)) * (0.005 + stock_idx * 0.0002)
        for date, value in zip(dates, values):
            residual_rows.append({"trade_date": date, "ts_code": code, "specific_return": value})
            exposure_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "industry_sw_l1_code": "I1" if stock_idx < 3 else "I2",
                    "total_mv": 10.0 * (stock_idx + 1),
                }
            )
    return pd.DataFrame(residual_rows), pd.DataFrame(exposure_rows)


def test_multifrequency_specific_risk_outputs_daily_monthly_quarterly_components():
    residuals, exposures = _specific_panel()
    config = SpecificRiskConfig(
        daily_window=20,
        daily_min_periods=10,
        daily_half_life=5,
        monthly_window=6,
        monthly_min_periods=3,
        monthly_half_life=3,
        quarterly_window=4,
        quarterly_min_periods=2,
        quarterly_half_life=2,
        group_minimum=3,
        reliability_target=20,
    )

    out = estimate_multifrequency_specific_risk(residuals, exposures, config=config)
    latest = out[out["trade_date"] == out["trade_date"].max()]

    assert latest["specific_variance_daily_component"].notna().all()
    assert latest["specific_variance_monthly_component"].notna().all()
    assert latest["specific_variance_quarterly_component"].notna().all()
    assert (latest["specific_variance_daily"] >= 0).all()
    assert np.allclose(
        latest[["specific_weight_daily", "specific_weight_monthly", "specific_weight_quarterly"]].sum(axis=1),
        1.0,
    )
    assert np.allclose(
        latest["specific_risk_annualized"],
        np.sqrt(latest["specific_variance_daily"] * 252.0),
    )


def test_future_residual_does_not_change_previous_specific_risk():
    residuals, exposures = _specific_panel(end="2024-11-29")
    config = SpecificRiskConfig(
        daily_window=20,
        daily_min_periods=10,
        daily_half_life=5,
        monthly_window=4,
        monthly_min_periods=2,
        monthly_half_life=2,
        quarterly_window=3,
        quarterly_min_periods=2,
        quarterly_half_life=2,
        group_minimum=3,
        reliability_target=20,
    )
    baseline = estimate_multifrequency_specific_risk(residuals, exposures, config=config)
    future_date = pd.Timestamp("2024-12-02")
    future_residuals = pd.concat(
        [
            residuals,
            pd.DataFrame(
                {
                    "trade_date": [future_date] * 6,
                    "ts_code": list("ABCDEF"),
                    "specific_return": [0.50] * 6,
                }
            ),
        ],
        ignore_index=True,
    )
    future_exposures = pd.concat(
        [
            exposures,
            pd.DataFrame(
                {
                    "trade_date": [future_date] * 6,
                    "ts_code": list("ABCDEF"),
                    "industry_sw_l1_code": ["I1"] * 3 + ["I2"] * 3,
                    "total_mv": [10, 20, 30, 40, 50, 60],
                }
            ),
        ],
        ignore_index=True,
    )
    extended = estimate_multifrequency_specific_risk(future_residuals, future_exposures, config=config)

    target_date = pd.Timestamp("2024-11-29")
    columns = ["specific_variance_daily", "specific_variance_monthly_component", "specific_variance_quarterly_component"]
    left = baseline[baseline["trade_date"] == target_date].sort_values("ts_code")[columns]
    right = extended[extended["trade_date"] == target_date].sort_values("ts_code")[columns]
    assert np.allclose(left, right, equal_nan=True)
