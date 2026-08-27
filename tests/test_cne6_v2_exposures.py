import numpy as np
import pandas as pd

from factors.data.cne6_builder import build_analyst_sentiment_features, build_fundamental_features
from factors.risk.cne6_v2_exposures import (
    _finite_ewm_moments,
    combine_style_exposures_v2,
    compute_raw_descriptors_v2,
    compute_v2_exposures,
    weighted_residualize_by_date,
)
from factors.risk.cne6_v2_spec import DescriptorSpecV2, STYLE_FACTORS_V2, descriptor_specs_v2


def test_v2_spec_has_15_styles_49_descriptors_and_normalized_weights():
    specs = descriptor_specs_v2()

    assert len(STYLE_FACTORS_V2) == 15
    assert len(specs) == 49
    assert len({spec.name for spec in specs}) == 49
    assert {spec.style for spec in specs} == set(STYLE_FACTORS_V2)
    totals = pd.Series({style: sum(spec.weight for spec in specs if spec.style == style) for style in STYLE_FACTORS_V2})
    assert np.allclose(totals.to_numpy(), 1.0)
    assert all(spec.direction in {-1.0, 1.0} for spec in specs)


def test_finite_ewm_uses_only_the_requested_window():
    values = np.array([1.0, 2.0, np.nan, 4.0, 8.0])

    mean, variance = _finite_ewm_moments(values, window=3, half_life=2, min_periods=2)

    decay = np.exp(-np.log(2.0) / 2.0)
    expected_values = np.array([8.0, 4.0])
    expected_weights = np.array([1.0, decay])
    expected_mean = np.average(expected_values, weights=expected_weights)
    expected_variance = np.average((expected_values - expected_mean) ** 2, weights=expected_weights)
    assert np.isclose(mean[-1], expected_mean)
    assert np.isclose(variance[-1], expected_variance)


def test_style_combination_requires_60_percent_scheduled_weight():
    date = pd.Timestamp("2024-01-02")
    descriptors = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["A", "B"],
            "large_part": [1.0, np.nan],
            "small_part": [np.nan, 2.0],
        }
    )
    specs = [
        DescriptorSpecV2("large_part", "demo", 0.60, "large", ("x",)),
        DescriptorSpecV2("small_part", "demo", 0.40, "small", ("y",)),
    ]

    out = combine_style_exposures_v2(descriptors, specs=specs, min_effective_weight=0.60, standardize=False)

    assert out.loc[0, "style_demo"] == 1.0
    assert out.loc[0, "style_demo_effective_weight"] == 0.60
    assert np.isnan(out.loc[1, "style_demo"])
    assert out.loc[1, "style_demo_effective_weight"] == 0.40


def test_weighted_residualization_removes_control_exposure_each_date():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 5,
            "target": [1.0, 3.0, 5.0, 7.0, 12.0],
            "control": [0.0, 1.0, 2.0, 3.0, 4.0],
            "weight": [1.0, 2.0, 1.0, 2.0, 1.0],
        }
    )

    residual = weighted_residualize_by_date(frame, "target", ["control"], "weight")

    valid = residual.notna()
    x = np.column_stack([np.ones(valid.sum()), frame.loc[valid, "control"]])
    w = frame.loc[valid, "weight"].to_numpy()
    inner = x.T @ (w * residual.loc[valid].to_numpy())
    assert np.allclose(inner, 0.0, atol=1e-10)


def _market_panel(days: int = 540) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    rows = []
    for stock_idx, code in enumerate(["A", "B"]):
        market = np.sin(np.arange(days) / 20.0) * 0.002
        returns = market * (0.8 + stock_idx * 0.4) + np.cos(np.arange(days) / 13.0) * 0.001
        price = 10.0 * np.cumprod(1.0 + returns)
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "returns_1d": returns[i],
                    "csi500_return": market[i],
                    "close_adj": price[i],
                    "total_mv": 1.0 + stock_idx,
                    "turnover_rate": 2.0 + stock_idx,
                    "amount": 1000.0 + i,
                    "pb": 2.0,
                    "pe_ttm": 10.0,
                    "n_cashflow_act_ttm": 10_000.0,
                    "total_assets": 30_000.0,
                    "total_liab": 10_000.0,
                    "net_profit_ttm": 2_000.0,
                    "dv_ttm": 1.0,
                    "dv_ratio": 20.0,
                }
            )
    return pd.DataFrame(rows)


def test_raw_descriptor_units_and_return_signs():
    panel = _market_panel()

    out = compute_raw_descriptors_v2(panel)
    latest = out[out["trade_date"] == out["trade_date"].max()].set_index("ts_code")

    assert np.isclose(latest.loc["A", "book_to_price"], 0.5)
    assert np.isclose(latest.loc["A", "earnings_yield"], 0.1)
    assert np.isclose(latest.loc["A", "cashflow_to_price"], 1.0)
    assert np.isfinite(latest.loc["A", "beta_252_ewma"])
    assert np.isfinite(latest.loc["A", "dastd_252"])
    assert np.isclose(latest.loc["A", "reversal_5d"], -panel[panel["ts_code"] == "A"]["close_adj"].pct_change(5).iloc[-1])


def test_past_descriptors_do_not_change_when_future_return_changes():
    panel = _market_panel(days=300)
    changed = panel.copy()
    last_date = changed["trade_date"].max()
    changed.loc[changed["trade_date"] == last_date, "returns_1d"] = 0.50

    original_out = compute_raw_descriptors_v2(panel)
    changed_out = compute_raw_descriptors_v2(changed)
    prior_date = panel["trade_date"].drop_duplicates().sort_values().iloc[-2]
    cols = ["beta_252_ewma", "dastd_252", "rstr_6m_ex_1m"]
    left = original_out[original_out["trade_date"] == prior_date].sort_values("ts_code")[cols]
    right = changed_out[changed_out["trade_date"] == prior_date].sort_values("ts_code")[cols]
    assert np.allclose(left, right, equal_nan=True)


def test_v2_exposure_pipeline_emits_all_style_contract_columns():
    panel = _market_panel(days=300)
    panel["csi500_member"] = True

    descriptors, styles, admission = compute_v2_exposures(panel)

    assert len(descriptors) == len(panel)
    assert len(styles) == len(panel)
    assert {f"style_{style}" for style in STYLE_FACTORS_V2}.issubset(styles.columns)
    assert set(admission["descriptor"]) == {spec.name for spec in descriptor_specs_v2()}


def test_fundamental_builder_adds_v2_same_quarter_growth_fields():
    end_dates = pd.date_range("2022-03-31", periods=8, freq="QE")
    ann_dates = end_dates + pd.Timedelta(days=30)
    quarter = np.tile(np.arange(1, 5), 2)
    revenue_q = np.array([100.0, 110.0, 120.0, 130.0, 150.0, 165.0, 180.0, 195.0])
    profit_q = revenue_q * 0.10
    finance_q = revenue_q * 0.01
    capex_q = np.array([10.0, 11.0, 12.0, 13.0, 15.0, 16.5, 18.0, 19.5])

    def cumulative_by_year(values):
        return np.concatenate([np.cumsum(values[:4]), np.cumsum(values[4:])])

    base = {"ts_code": ["A"] * 8, "ann_date": ann_dates, "end_date": end_dates}
    income = pd.DataFrame(
        {
            **base,
            "total_revenue": cumulative_by_year(revenue_q),
            "operate_profit": cumulative_by_year(profit_q),
            "fin_exp": cumulative_by_year(finance_q),
            "n_income_attr_p": cumulative_by_year(profit_q * 0.8),
            "oper_cost": cumulative_by_year(revenue_q * 0.6),
        }
    )
    cashflow = pd.DataFrame(
        {
            **base,
            "n_cashflow_act": cumulative_by_year(profit_q),
            "c_pay_acq_const_fiolta": cumulative_by_year(capex_q),
        }
    )
    total_assets = np.array([1000, 1020, 1040, 1060, 1200, 1224, 1248, 1272], dtype=float)
    inventories = np.array([100, 101, 102, 103, 120, 121.2, 122.4, 123.6], dtype=float)
    current_assets = total_assets * 0.45
    current_liab = total_assets * 0.25
    balance = pd.DataFrame(
        {
            **base,
            "total_assets": total_assets,
            "total_liab": total_assets * 0.40,
            "total_hldr_eqy_exc_min_int": total_assets * 0.60,
            "inventories": inventories,
            "total_cur_assets": current_assets,
            "total_cur_liab": current_liab,
        }
    )
    fina = pd.DataFrame(
        {
            **base,
            "eps": np.array([0.20, 0.42, 0.66, 0.92, 0.30, 0.63, 0.99, 1.38]),
            "roa": np.linspace(5.0, 8.5, 8),
        }
    )

    out = build_fundamental_features(
        {"income": income, "cashflow": cashflow, "balancesheet": balance, "fina_indicator": fina}
    )
    latest = out.iloc[-1]

    assert np.isclose(latest["asset_growth"], 0.20)
    assert np.isclose(latest["inventory_growth"], 0.20)
    assert np.isclose(latest["working_capital_growth"], 0.20)
    assert np.isclose(latest["eps_growth"], 0.50)
    assert np.isclose(latest["roe_growth"], latest["roe_ttm"] - out.iloc[-5]["roe_ttm"])
    assert np.isclose(latest["operating_margin_ttm"], 0.10)
    assert np.isclose(latest["book_leverage"], 1.0 / 0.60)
    assert np.isclose(latest["inverse_interest_coverage"], 0.10)
    assert np.isfinite(latest["earnings_stability"])


def test_analyst_forward_eps_is_available_from_report_date_plus_one():
    daily = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [10.0, 10.0],
        }
    )
    reports = pd.DataFrame(
        {
            "ts_code": ["A"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "eps": [1.25],
        }
    )

    out = build_analyst_sentiment_features(daily, reports).set_index("trade_date")

    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "analyst_forward_eps_180"])
    assert out.loc[pd.Timestamp("2024-01-03"), "analyst_forward_eps_180"] == 1.25
