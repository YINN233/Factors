"""Smoke tests for the local CNE6-style risk model.

These tests use synthetic data and never call tushare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.data.cne6_builder import _gap_aware_price_returns, asof_to_daily, build_analyst_sentiment_features
from factors.risk.cne6_descriptors import descriptor_metadata
from factors.risk.cne6_exposures import compute_descriptor_exposures, compute_style_exposures
from factors.risk.cne6_regression import _regression_work, run_factor_return_regression
from factors.risk.cne6_risk_model import rolling_factor_covariance, rolling_specific_risk
from factors.reports.cne6_dynamic_risk_calibration import (
    add_next_available_date,
    build_calibrated_forecasts,
    forecast_mapping,
)
from factors.reports.cne6_portfolio_attribution import _covariance_matrix, run_attribution
from factors.reports.cne6_risk_weighted_regression import (
    add_lagged_specific_risk,
    clip_positive_weights,
    fit_weights,
    fit_wls_beta,
    risk_asof_mapping,
    weight_diagnostics,
    weighted_r2,
)


def make_cne6_panel(n_stocks: int = 70, n_days: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(706)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    rows = []
    market_ret = pd.Series(rng.normal(0.0002, 0.012, n_days), index=dates)
    for stock_idx in range(n_stocks):
        code = f"{stock_idx:06d}.SZ"
        price = 10.0 + stock_idx * 0.1
        beta = 0.7 + (stock_idx % 9) * 0.08
        total_mv = 3e5 + stock_idx * 2e4
        for date in dates:
            ret = beta * market_ret.loc[date] + rng.normal(0.0001, 0.012)
            open_px = price * (1.0 + rng.normal(0.0, 0.003))
            close_px = max(0.5, open_px * (1.0 + ret))
            high_px = max(open_px, close_px) * (1.0 + abs(rng.normal(0.0, 0.004)))
            low_px = min(open_px, close_px) * (1.0 - abs(rng.normal(0.0, 0.004)))
            volume = float(rng.integers(500_000, 5_000_000))
            amount = volume * close_px
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "industry": f"industry_{stock_idx % 7}",
                    "open_adj": open_px,
                    "high_adj": high_px,
                    "low_adj": low_px,
                    "close_adj": close_px,
                    "volume": volume,
                    "amount": amount,
                    "turnover_rate": rng.uniform(0.2, 4.0),
                    "total_mv": total_mv * (1.0 + rng.normal(0.0, 0.02)),
                    "pb": rng.uniform(0.8, 5.0),
                    "pe_ttm": rng.uniform(6.0, 35.0),
                    "ps_ttm": rng.uniform(0.5, 8.0),
                    "dv_ttm": rng.uniform(0.0, 4.0),
                    "dv_ratio": rng.uniform(0.0, 2.0),
                    "revenue_yoy": rng.normal(0.08, 0.12),
                    "net_profit_yoy": rng.normal(0.06, 0.20),
                    "asset_turnover_yoy": rng.normal(0.02, 0.06),
                    "roe_ttm": rng.normal(0.10, 0.03),
                    "roa_ttm": rng.normal(0.04, 0.015),
                    "gross_margin_ttm": rng.normal(0.25, 0.08),
                    "cashflow_to_profit": rng.normal(1.0, 0.2),
                    "debt_to_assets": rng.uniform(0.2, 0.75),
                    "n_cashflow_act_ttm": rng.normal(1e8, 2e7),
                    "net_profit_ttm": rng.normal(8e7, 2e7),
                    "total_assets": rng.normal(1.5e9, 2e8),
                    "buy_lg_amount": rng.normal(2e6, 5e5),
                    "sell_lg_amount": rng.normal(2e6, 5e5),
                    "buy_elg_amount": rng.normal(8e5, 2e5),
                    "sell_elg_amount": rng.normal(8e5, 2e5),
                    "net_mf_amount": rng.normal(0.0, 1e6),
                    "analyst_report_count_90": float(stock_idx % 5),
                    "analyst_org_count_180": float(stock_idx % 7),
                    "analyst_rating_score_180": rng.choice([0.0, 0.5, 1.0]),
                    "analyst_target_upside_180": rng.normal(0.15, 0.05),
                    "analyst_eps_revision_180": rng.normal(0.05, 0.03),
                    "csi500_index_weight": 100.0 / n_stocks,
                }
            )
            price = close_px

    panel = pd.DataFrame(rows).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel["returns_1d"] = panel.groupby("ts_code", sort=False)["close_adj"].pct_change()
    panel["fwd_1d_return"] = panel.groupby("ts_code", sort=False)["close_adj"].shift(-1) / panel["close_adj"] - 1.0
    panel["csi500_return"] = panel["trade_date"].map(market_ret)
    return panel


def test_descriptor_metadata_flags_available_columns():
    panel = make_cne6_panel(n_stocks=8, n_days=30)
    meta = descriptor_metadata(panel.columns)
    assert {"descriptor", "style", "expression", "description", "is_available"}.issubset(meta.columns)
    assert meta.loc[meta["descriptor"] == "log_total_mv", "is_available"].iloc[0]
    assert meta.loc[meta["descriptor"] == "analyst_report_count_90", "is_available"].iloc[0]
    assert set(meta.loc[meta["style"] == "sentiment", "source"]) == {"tushare.report_rc"}
    assert meta["style"].nunique() >= 8


def test_gap_aware_returns_do_not_bridge_missing_trading_rows():
    panel = pd.DataFrame(
        {
            "ts_code": ["A", "A", "B", "B", "B"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "close_adj": [10.0, 12.0, 20.0, 21.0, 22.0],
        }
    ).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    returns, forward = _gap_aware_price_returns(panel)
    a_rows = panel["ts_code"] == "A"
    b_rows = panel["ts_code"] == "B"
    assert returns.loc[a_rows].isna().all()
    assert forward.loc[a_rows].isna().all()
    assert np.isclose(returns.loc[b_rows].iloc[1], 0.05)
    assert np.isclose(forward.loc[b_rows].iloc[1], 1.0 / 21.0)


def test_cne6_exposure_regression_and_risk_smoke():
    panel = make_cne6_panel()
    descriptors, metadata = compute_descriptor_exposures(panel)
    style = compute_style_exposures(descriptors, metadata)
    style_cols = [c for c in style.columns if c.startswith("style_") and not c.endswith("_n")]

    assert "beta_252" in descriptors.columns
    assert "style_size" in style.columns
    latest = style[style["trade_date"] == style["trade_date"].max()]
    assert latest["style_size"].notna().mean() > 0.9
    assert abs(float(latest["style_size"].mean())) < 1e-8
    assert len(style_cols) >= 8

    factor_returns, residuals, diagnostics = run_factor_return_regression(panel, style)
    ok = diagnostics[diagnostics["regression_status"] == "ok"]
    assert not ok.empty
    assert ok["n_obs"].min() >= 30
    assert "country" in factor_returns.columns
    assert residuals["specific_return"].notna().any()

    covariance = rolling_factor_covariance(factor_returns, windows=(20,))
    specific_risk = rolling_specific_risk(residuals, windows=(20,))
    assert not covariance.empty
    variances = covariance[covariance["factor_i"] == covariance["factor_j"]]
    assert (variances["covariance"] >= -1e-12).all()
    assert specific_risk["specific_risk_20"].notna().any()
    assert (specific_risk["specific_risk_20"].dropna() >= 0).all()


def test_regression_return_modes_align_exposures_and_returns():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    panel = pd.DataFrame(
        {
            "trade_date": list(dates) * 2,
            "ts_code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "industry": ["银行"] * 3 + ["电子"] * 3,
            "close_adj": [10.0, 11.0, 13.2, 20.0, 19.0, 20.9],
            "returns_1d": [np.nan, 0.10, 0.20, np.nan, -0.05, 0.10],
            "fwd_1d_return": [0.10, 0.20, np.nan, -0.05, 0.10, np.nan],
            "total_mv": [100.0, 110.0, 120.0, 200.0, 190.0, 210.0],
        }
    )
    style = pd.DataFrame(
        {
            "trade_date": list(dates) * 2,
            "ts_code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "style_size": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        }
    )

    forward, _ = _regression_work(panel, style, return_mode="forward_1d")
    same_day, _ = _regression_work(panel, style, return_mode="same_day")
    lagged, _ = _regression_work(panel, style, return_mode="lagged_exposure_1d")

    key = (forward["ts_code"] == "000001.SZ") & (forward["trade_date"] == pd.Timestamp("2024-01-02"))
    assert np.isclose(forward.loc[key, "_y"].iloc[0], 0.10)
    key = (same_day["ts_code"] == "000001.SZ") & (same_day["trade_date"] == pd.Timestamp("2024-01-03"))
    assert np.isclose(same_day.loc[key, "_y"].iloc[0], 0.10)
    assert np.isclose(same_day.loc[key, "style_size"].iloc[0], 2.0)
    key = (lagged["ts_code"] == "000001.SZ") & (lagged["trade_date"] == pd.Timestamp("2024-01-03"))
    assert np.isclose(lagged.loc[key, "_y"].iloc[0], 0.10)
    assert np.isclose(lagged.loc[key, "style_size"].iloc[0], 1.0)
    assert np.isclose(lagged.loc[key, "total_mv"].iloc[0], 100.0)


def test_dynamic_risk_forecast_mapping_uses_t_minus_lag():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    mapping = forecast_mapping(dates, lag_days=3)

    assert mapping["target_date"].iloc[0] == dates[3]
    assert mapping["forecast_asof_date"].iloc[0] == dates[0]
    assert mapping["target_date"].iloc[-1] == dates[5]
    assert mapping["forecast_asof_date"].iloc[-1] == dates[2]


def test_dynamic_risk_calibration_uses_only_available_past_ratios():
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    validation = pd.DataFrame(
        {
            "target_date": dates[3:],
            "forecast_asof_date": dates[:-3],
            "block": ["factor_all"] * 5,
            "risk_window": [252] * 5,
            "lag_days": [3] * 5,
            "predicted_variance": [1.0] * 5,
            "realized_variance": [2.0, 4.0, 8.0, 16.0, 32.0],
            "raw_ratio": [2.0, 4.0, 8.0, 16.0, 32.0],
            "n_items": [3] * 5,
        }
    )
    validation = add_next_available_date(validation, dates)
    calibrated = build_calibrated_forecasts(
        validation,
        calibration_windows=(2,),
        ratio_clip=(0.05, 20.0),
        multiplier_clip=(0.25, 20.0),
    )
    w2 = calibrated[calibrated["calibration_window"] == 2].sort_values("target_date")

    assert w2.empty

    later_dates = pd.date_range("2024-01-02", periods=12, freq="B")
    validation = pd.DataFrame(
        {
            "target_date": later_dates[3:],
            "forecast_asof_date": later_dates[:-3],
            "block": ["factor_all"] * 9,
            "risk_window": [252] * 9,
            "lag_days": [3] * 9,
            "predicted_variance": [1.0] * 9,
            "realized_variance": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
            "raw_ratio": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
            "n_items": [3] * 9,
        }
    )
    validation = add_next_available_date(validation, later_dates)
    calibrated = build_calibrated_forecasts(
        validation,
        calibration_windows=(2,),
        ratio_clip=(0.05, 20.0),
        multiplier_clip=(0.25, 20.0),
    )
    w2 = calibrated[calibrated["calibration_window"] == 2].sort_values("target_date")

    assert not w2.empty
    first = w2.iloc[0]
    assert first["target_date"] > later_dates[6]
    assert np.isclose(first["calibration_multiplier"], 3.0)


def test_ledoit_wolf_full_covariance_emits_upper_triangle():
    rng = np.random.default_rng(729)
    dates = pd.date_range("2024-01-02", periods=32, freq="B")
    factor_returns = pd.DataFrame(
        {
            "trade_date": dates,
            "country": rng.normal(0.0, 0.01, len(dates)),
            "style_size": rng.normal(0.0, 0.01, len(dates)),
            "industry_银行": rng.normal(0.0, 0.01, len(dates)),
            "industry_电子": rng.normal(0.0, 0.01, len(dates)),
        }
    )

    covariance = rolling_factor_covariance(
        factor_returns,
        windows=(20,),
        covariance_method="ledoit_wolf",
        lw_full_windows=(20,),
    )

    latest = covariance[covariance["trade_date"] == covariance["trade_date"].max()]
    assert set(latest["covariance_method"]) == {"ledoit_wolf"}
    assert set(latest["covariance_type"]) == {"ledoit_wolf_full"}
    assert latest["shrinkage"].notna().all()
    assert len(latest) == 10
    assert len(latest[latest["factor_i"] != latest["factor_j"]]) == 6


def test_covariance_matrix_symmetrizes_one_sided_rows():
    date = pd.Timestamp("2024-01-31")
    covariance = pd.DataFrame(
        {
            "trade_date": [date, date, date],
            "window": [252, 252, 252],
            "factor_i": ["industry_银行", "industry_电子", "industry_银行"],
            "factor_j": ["industry_银行", "industry_电子", "industry_电子"],
            "covariance": [0.0004, 0.0009, 0.0002],
        }
    )

    mat = _covariance_matrix(covariance, date, 252, ["industry_银行", "industry_电子"])

    assert np.isclose(mat.loc["industry_银行", "industry_电子"], 0.0002)
    assert np.isclose(mat.loc["industry_电子", "industry_银行"], 0.0002)


def test_portfolio_attribution_uses_industry_covariance_submatrix():
    date = pd.Timestamp("2024-01-31")
    panel = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "industry": ["银行", "电子"],
            "csi500_index_weight": [0.5, 0.5],
        }
    )
    style = pd.DataFrame({"trade_date": [date, date], "ts_code": ["000001.SZ", "000002.SZ"]})
    weights = pd.DataFrame(
        {
            "scenario": ["test", "test"],
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "weight": [0.8, 0.2],
        }
    )
    covariance = pd.DataFrame(
        {
            "trade_date": [date, date, date],
            "window": [252, 252, 252],
            "factor_i": ["industry_银行", "industry_电子", "industry_银行"],
            "factor_j": ["industry_银行", "industry_电子", "industry_电子"],
            "covariance": [0.0004, 0.0004, 0.0002],
        }
    )
    specific_risk = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "specific_risk_252": [0.0, 0.0],
        }
    )

    _, risk, _ = run_attribution(weights, panel, style, covariance, specific_risk)

    expected = np.array([0.3, -0.3]) @ np.array([[0.0004, 0.0002], [0.0002, 0.0004]]) @ np.array([0.3, -0.3])
    assert np.isclose(risk["industry_var_daily"].iloc[0], expected)
    assert risk["industry_var_daily"].iloc[0] < 2 * (0.3**2) * 0.0004


def test_cne6_builder_asof_does_not_use_future_fundamentals():
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "trade_date": pd.to_datetime(["2024-03-29", "2024-04-10", "2024-04-16"]),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "available_date": [pd.Timestamp("2024-04-15")],
            "roe_ttm": [0.12],
        }
    )
    out = asof_to_daily(daily, fundamentals)
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-10"), "roe_ttm"].isna().all()
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-16"), "roe_ttm"].iloc[0] == 0.12


def test_analyst_sentiment_features_use_report_date_plus_one():
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 4,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-04-10", "2024-04-16"]),
            "close": [10.0, 10.0, 11.0, 12.0],
        }
    )
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_date": pd.to_datetime(["2024-01-02", "2024-04-15"]),
            "org_name": ["券商A", "券商B"],
            "rating": ["买入", "增持"],
            "eps": [1.0, 1.2],
            "max_price": [15.0, 18.0],
            "min_price": [13.0, 16.0],
        }
    )

    out = build_analyst_sentiment_features(daily, report_rc)
    by_date = out.set_index("trade_date")

    assert by_date.loc[pd.Timestamp("2024-01-02"), "analyst_report_count_90"] == 0.0
    assert by_date.loc[pd.Timestamp("2024-01-03"), "analyst_report_count_90"] == 1.0
    assert by_date.loc[pd.Timestamp("2024-04-10"), "analyst_org_count_180"] == 1.0
    assert by_date.loc[pd.Timestamp("2024-04-10"), "analyst_rating_score_180"] == 1.0
    assert by_date.loc[pd.Timestamp("2024-04-16"), "analyst_report_count_90"] == 1.0
    assert by_date.loc[pd.Timestamp("2024-04-16"), "analyst_org_count_180"] == 2.0
    assert np.isclose(by_date.loc[pd.Timestamp("2024-01-03"), "analyst_target_upside_180"], 0.4)


def test_portfolio_attribution_flags_outside_model_universe():
    date = pd.Timestamp("2024-01-31")
    panel = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "industry": ["银行", "地产"],
            "csi500_index_weight": [50.0, 50.0],
        }
    )
    style = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "style_size": [0.2, -0.2],
        }
    )
    weights = pd.DataFrame(
        {
            "scenario": ["test", "test"],
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "999999.SZ"],
            "weight": [0.4, 0.6],
        }
    )
    covariance = pd.DataFrame(
        {
            "trade_date": [date],
            "window": [252],
            "factor_i": ["style_size"],
            "factor_j": ["style_size"],
            "covariance": [0.0001],
        }
    )
    specific_risk = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "specific_risk_252": [0.02, 0.03],
        }
    )
    exposures, risk, summary = run_attribution(weights, panel, style, covariance, specific_risk)

    diagnostic = exposures[exposures["factor"] == "outside_model_universe"]
    assert diagnostic["active_exposure"].iloc[0] == 0.6
    assert risk["out_of_model_weight"].iloc[0] == 0.6
    assert summary["out_of_model_weight_latest"].iloc[0] == 0.6
    assert "industry_unknown" not in set(exposures["factor"])


def test_risk_weighted_regression_maps_specific_risk_from_t_minus_3():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    mapping = risk_asof_mapping(dates, lag_days=3)

    assert mapping["trade_date"].iloc[0] == dates[3]
    assert mapping["risk_asof_date"].iloc[0] == dates[0]

    work = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "_y": np.arange(len(dates), dtype=float),
        }
    )
    specific_risk = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "specific_risk_252": np.arange(10.0, 10.0 + len(dates)),
        }
    )

    out = add_lagged_specific_risk(work, specific_risk, lag_days=3)
    by_date = out.set_index("trade_date")

    assert np.isnan(by_date.loc[dates[2], "specific_risk_252"])
    assert by_date.loc[dates[3], "risk_asof_date"] == dates[0]
    assert by_date.loc[dates[3], "specific_risk_252"] == 10.0
    assert by_date.loc[dates[5], "risk_asof_date"] == dates[2]
    assert by_date.loc[dates[5], "specific_risk_252"] == 12.0


def test_cross_evaluated_r2_can_be_much_lower_than_own_weight_r2():
    x = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0])
    fit_weights = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.001])

    beta, _ = fit_wls_beta(y, x, fit_weights)
    fitted = x @ beta
    own_r2 = weighted_r2(y, fitted, fit_weights)
    equal_r2 = weighted_r2(y, fitted, np.ones_like(fit_weights))

    assert own_r2 > 0.98
    assert equal_r2 < 0.30


def test_risk_weight_clipping_reduces_concentration():
    raw = pd.Series([1.0] * 100 + [1000.0])
    clipped = clip_positive_weights(raw, clip_quantiles=(0.01, 0.99), min_count=20)

    raw_diag = weight_diagnostics(raw, n_obs=len(raw))
    clipped_diag = weight_diagnostics(clipped, n_obs=len(clipped))

    assert clipped.max() < raw.max()
    assert clipped_diag["max_weight_share"] < raw_diag["max_weight_share"]
    assert clipped_diag["effective_sample_share"] > raw_diag["effective_sample_share"]


def test_rank_specific_risk_weight_keeps_broad_effective_sample():
    frame = pd.DataFrame(
        {
            "total_mv": np.linspace(100.0, 200.0, 100),
            "specific_risk_252": np.linspace(0.01, 0.08, 100),
        }
    )

    base = fit_weights(frame, "sqrt_mv")
    ranked = fit_weights(frame, "sqrt_mv_x_low_specific_rank_252_t3")
    ratio = ranked / base
    diag = weight_diagnostics(ranked, n_obs=len(ranked))

    assert ratio.min() >= 0.75
    assert ratio.max() <= 1.25
    assert diag["effective_sample_share"] > 0.90
