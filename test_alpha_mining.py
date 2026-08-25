"""
Smoke tests for the alpha-mining scaffold.  These tests use synthetic data and
do not call tushare.
"""

import numpy as np
import pandas as pd

from factors.alpha.candidates import available_candidates
from factors.alpha.fundamental_factory import fundamental_operator_candidates
from factors.alpha.miner import AlphaMiner, AlphaMiningConfig
from factors.alpha.operators import (
    cs_robust_zscore,
    group_rank,
    safe_div,
    ts_argmax,
    ts_decay_linear,
    ts_delta,
    ts_max_drawdown,
    ts_median,
    ts_rank,
    ts_wma,
    ts_zscore,
)
from factors.alpha.external_alpha import validate_external_alpha_panel, wide_to_external_alpha_panel
from factors.alpha.public_factors import calculate_public_factors, public_factor_availability
from factors.alpha.validation import add_forward_rank_labels, validate_factors
from factors.data.fundamental_builder import asof_to_daily
from factors.models.xgb_alpha import AlphaModelConfig, train_predict_alpha_model
from factors.portfolio.constrained_optimizer import OptimizerConfig, optimize_constrained_weights
from factors.portfolio.style_exposures import STYLE_COLUMNS, compute_style_exposures
from factors.reports.fundamental_diagnostics import (
    composite_score,
    daily_factor_stats,
    summarize_daily_stats,
)
from factors.reports.index_enhancement import _cap_and_normalize, summarize_returns
from factors.reports.enhancement_stability import (
    rolling_stability,
    summarize_distribution,
    summarize_periods,
)
from factors.reports.guoxin_co_momentum import _weighted_extreme, compute_rolling_comomentum


def make_synthetic_panel(n_stocks: int = 30, n_days: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    rows = []
    for stock_idx in range(n_stocks):
        code = f"{stock_idx:06d}.SZ"
        price = 10.0 + stock_idx * 0.02
        for date in dates:
            ret = rng.normal(0.0005, 0.015)
            open_px = price * (1.0 + rng.normal(0.0, 0.004))
            close_px = open_px * (1.0 + ret)
            high_px = max(open_px, close_px) * (1.0 + abs(rng.normal(0.0, 0.003)))
            low_px = min(open_px, close_px) * (1.0 - abs(rng.normal(0.0, 0.003)))
            volume = rng.integers(1_000_000, 8_000_000)
            amount = volume * close_px
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "open_adj": open_px,
                    "high_adj": high_px,
                    "low_adj": low_px,
                    "close_adj": close_px,
                    "volume": float(volume),
                    "amount": amount,
                    "turnover_rate": rng.uniform(0.2, 3.0),
                    "industry": f"ind{stock_idx % 5}",
                    "index_weight": 1.0 / n_stocks,
                }
            )
            price = close_px

    df = pd.DataFrame(rows).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    momentum_5 = df.groupby("ts_code")["close_adj"].pct_change(5)
    df["label"] = momentum_5.fillna(0.0) + rng.normal(0.0, 0.005, len(df))
    return df


def test_safe_div_removes_infinite_values():
    result = safe_div(pd.Series([1.0, 2.0]), pd.Series([0.0, 2.0]))
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.0


def test_alpha_miner_runs_and_selects_candidates():
    df = make_synthetic_panel()
    candidates = available_candidates(df, windows=(5, 20), include_fundamental=False)
    miner = AlphaMiner(
        candidates,
        config=AlphaMiningConfig(min_abs_ic=0.01, min_coverage=0.5, max_pair_corr=0.9),
    )
    result = miner.run(df)

    assert not result.factor_values.empty
    assert not result.summary.empty
    assert "mom_close_5" in result.summary["factor"].tolist()
    assert result.selected
    assert result.summary["coverage"].between(0, 1).all()


def test_time_series_extra_operators():
    df = pd.DataFrame(
        {
            "ts_code": ["a"] * 5,
            "trade_date": pd.date_range("2024-01-01", periods=5),
            "x": [1.0, 2.0, 4.0, 3.0, 5.0],
        }
    )
    delta = ts_delta(df, "x", periods=2)
    rank = ts_rank(df, "x", window=3)

    assert np.isnan(delta.iloc[0])
    assert delta.iloc[2] == 3.0
    assert np.isnan(rank.iloc[1])
    assert rank.iloc[2] == 1.0
    assert np.isnan(ts_zscore(df, "x", window=3).iloc[1])
    assert cs_robust_zscore(df, "x").notna().all()
    assert ts_decay_linear(df, "x", window=3).iloc[2] == (1.0 + 4.0 + 12.0) / 6.0
    assert ts_argmax(df, "x", window=3).iloc[3] == 2.0 / 3.0
    assert ts_median(df, "x", window=3).iloc[2] == 2.0
    assert np.isclose(ts_wma(df, "x", window=3).iloc[2], (1.0 + 4.0 + 12.0) / 6.0)
    assert ts_max_drawdown(df, "x", window=3).iloc[3] < 0

    grouped = df.assign(industry=["a", "a", "a", "b", "b"])
    assert group_rank(grouped, "x").between(0, 1).all()


def test_guoxin_co_momentum_weighting_smoke():
    values = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
    keys = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
    expected = 0.01 * 1.0 + 0.02 * (2 ** (-1 / 3)) + 0.03 * (2 ** (-2 / 3))
    assert np.isclose(_weighted_extreme(values, keys, n=3, largest=True), expected)

    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    stock_ret = np.linspace(-0.02, 0.03, 25)
    industry_ret = np.linspace(0.01, 0.05, 25)
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "000001.SZ",
            "stock_ret": stock_ret,
            "volume_price": stock_ret,
            "industry_ret": industry_ret,
            "market_ret": industry_ret / 2.0,
        }
    )
    factors = compute_rolling_comomentum(panel, window=20, momentum_n=5, reversal_n=15)
    last_window = industry_ret[-20:]
    weights = 2 ** (-(np.arange(5, dtype=float)) / 5)
    expected_vicm = float(np.sum(weights * last_window[-5:][::-1]))
    assert np.isclose(factors["vicm"].iloc[-1], expected_vicm)
    assert np.isclose(factors["cmc"].iloc[-1], factors["vicm"].iloc[-1] - factors["vicr"].iloc[-1])


def test_external_alpha_panel_validation_and_public_factors():
    df = make_synthetic_panel(n_stocks=18, n_days=150)
    rng = np.random.default_rng(19)
    df["pe_ttm"] = rng.uniform(5.0, 40.0, len(df))
    df["pb"] = rng.uniform(0.5, 6.0, len(df))
    df["ps_ttm"] = rng.uniform(0.2, 8.0, len(df))
    df["total_mv"] = rng.uniform(1e6, 1e8, len(df))
    df["log_mv"] = np.log(df["total_mv"])
    df["operating_cf_margin_ttm"] = rng.normal(0.08, 0.03, len(df))
    df["cashflow_to_profit"] = rng.normal(1.0, 0.2, len(df))
    df["roe_ttm"] = rng.normal(0.1, 0.03, len(df))
    df["debt_to_assets"] = rng.uniform(0.2, 0.8, len(df))
    df["csi500_index_weight"] = 1.0 / 18

    factor_values, metadata = calculate_public_factors(df)
    assert "rl_24_price_reversal_30d" in factor_values.columns
    assert metadata["availability"].isin(["direct", "proxy", "partial", "skipped"]).all()
    assert not public_factor_availability(df).empty

    panel = wide_to_external_alpha_panel(
        factor_values[["trade_date", "ts_code", "rl_24_price_reversal_30d"]].dropna().head(20),
        ["rl_24_price_reversal_30d"],
        source="toy",
        version="v1",
        release_date="2026-01-01",
    )
    checked = validate_external_alpha_panel(panel)
    assert checked["factor_value"].notna().all()


def test_validation_model_and_constrained_optimizer_smoke():
    df = make_synthetic_panel(n_stocks=35, n_days=180)
    rng = np.random.default_rng(23)
    df["csi500_index_weight"] = 1.0 / 35
    df["total_mv"] = rng.uniform(1e6, 1e8, len(df))
    df["log_mv"] = np.log(df["total_mv"])
    df["pe_ttm"] = rng.uniform(5.0, 40.0, len(df))
    df["pb"] = rng.uniform(0.5, 6.0, len(df))
    df["ps_ttm"] = rng.uniform(0.2, 8.0, len(df))
    df["roe_ttm"] = rng.normal(0.1, 0.03, len(df))
    df["cashflow_to_profit"] = rng.normal(1.0, 0.2, len(df))
    df["debt_to_assets"] = rng.uniform(0.2, 0.8, len(df))
    df = add_forward_rank_labels(df, price_col="close_adj", horizons=(1, 5))
    df["toy_factor"] = df.groupby("trade_date")["fwd_5d_rank"].transform(lambda s: s.rank(pct=True)) + rng.normal(0, 0.01, len(df))

    summary, _ = validate_factors(df, ["toy_factor"])
    assert not summary.empty
    assert summary.loc[0, "coverage"] > 0.9

    model = train_predict_alpha_model(
        df,
        ["toy_factor", "log_mv", "pb"],
        config=AlphaModelConfig(train_end="2023-04-30", valid_start="2023-05-01", valid_end="2023-07-31", test_start="2023-08-01", ytd_start="2023-08-01"),
    )
    assert not model.predictions.empty
    assert "pred_rank" in model.predictions.columns

    styles = compute_style_exposures(df)
    assert set(STYLE_COLUMNS).issubset(styles.columns)
    latest_date = df["trade_date"].max()
    sub = df[df["trade_date"] == latest_date].merge(styles[styles["trade_date"] == latest_date], on=["trade_date", "ts_code"], how="left")
    sub["alpha"] = rng.normal(0, 1, len(sub))
    weights = optimize_constrained_weights(
        sub,
        "alpha",
        style_cols=["style_size", "style_value"],
        config=OptimizerConfig(max_industry_active=0.10, max_style_active=0.50, max_active_share=0.40),
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights.min() >= -1e-8


def test_fundamental_pit_asof_alignment():
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 4,
            "trade_date": pd.to_datetime(["2024-03-29", "2024-04-01", "2024-04-15", "2024-04-16"]),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": [pd.Timestamp("2023-12-31")],
            "available_date": [pd.Timestamp("2024-04-15")],
            "roe_ttm": [0.12],
        }
    )
    out = asof_to_daily(daily, fundamentals)

    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-01"), "roe_ttm"].isna().all()
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-15"), "roe_ttm"].iloc[0] == 0.12
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-16"), "roe_ttm"].iloc[0] == 0.12


def test_fundamental_candidates_run_on_synthetic_panel():
    df = make_synthetic_panel(n_stocks=25, n_days=60)
    rng = np.random.default_rng(11)
    df["roe_ttm"] = rng.normal(0.10, 0.03, len(df))
    df["roa_ttm"] = rng.normal(0.04, 0.01, len(df))
    df["cashflow_to_profit"] = rng.normal(1.0, 0.2, len(df))
    df["revenue_yoy"] = rng.normal(0.08, 0.1, len(df))
    df["net_profit_yoy"] = rng.normal(0.08, 0.15, len(df))
    df["n_cashflow_act_ttm"] = rng.normal(1e8, 2e7, len(df))
    df["net_profit_ttm"] = rng.normal(8e7, 2e7, len(df))
    df["total_liab"] = rng.normal(5e8, 1e8, len(df))
    df["debt_to_assets"] = rng.uniform(0.2, 0.8, len(df))
    df["pb"] = rng.uniform(0.8, 5.0, len(df))
    df["pe_ttm"] = rng.uniform(5.0, 40.0, len(df))
    df["ps_ttm"] = rng.uniform(0.5, 10.0, len(df))
    df["dv_ttm"] = rng.uniform(0.0, 4.0, len(df))
    df["total_mv"] = rng.uniform(1e6, 1e8, len(df))
    df["log_mv"] = np.log(df["total_mv"])
    df["free_cashflow_ttm"] = rng.normal(5e7, 2e7, len(df))
    df["operating_cf_margin_ttm"] = rng.normal(0.08, 0.03, len(df))
    df["ocf_to_or"] = rng.normal(0.10, 0.03, len(df))
    df["ocf_to_profit"] = rng.normal(1.0, 0.2, len(df))
    df["net_margin_ttm"] = rng.normal(0.06, 0.02, len(df))
    df["current_ratio"] = rng.uniform(0.6, 3.0, len(df))
    df["quick_ratio"] = rng.uniform(0.4, 2.5, len(df))
    df["working_capital_pressure"] = rng.normal(0.2, 0.1, len(df))
    df["inv_turn"] = rng.uniform(0.2, 10.0, len(df))
    df["ar_turn"] = rng.uniform(0.2, 10.0, len(df))
    df["grossprofit_margin"] = rng.normal(20.0, 8.0, len(df))
    df["netprofit_margin"] = rng.normal(8.0, 4.0, len(df))
    df["eps"] = rng.normal(0.7, 0.3, len(df))
    df["bps"] = rng.normal(5.0, 2.0, len(df))
    df["cash_to_liab"] = rng.uniform(0.02, 1.2, len(df))
    df["asset_turnover_ttm"] = rng.uniform(0.1, 2.0, len(df))
    df["inventories"] = rng.uniform(1e7, 2e8, len(df))
    df["accounts_receiv"] = rng.uniform(1e7, 2e8, len(df))
    df["total_assets"] = rng.uniform(5e8, 3e9, len(df))
    df["money_cap"] = rng.uniform(5e7, 5e8, len(df))
    df["n_cashflow_act_ttm"] = rng.normal(1e8, 2e7, len(df))
    df["total_revenue_ttm"] = rng.normal(1e9, 2e8, len(df))
    df["capex_to_assets"] = rng.uniform(0.0, 0.15, len(df))

    generated = [candidate for candidate in fundamental_operator_candidates() if candidate.is_available(df)]
    generated_names = {candidate.name for candidate in generated}
    assert "fund_atom_low_pb" in generated_names
    assert "fund_combo_roe_low_pb" in generated_names
    assert "fund_combo_revenue_profit_cash_hmean" in generated_names
    assert "fund_ind_neu_roe_low_pb" in generated_names
    assert "fund_atom_cashflow_to_liab" in generated_names
    assert "fund_combo_reported_ocf_to_profit_low_pb" in generated_names
    assert "fund_combo_fcf_margin_low_ps" in generated_names
    assert "fund_ind_neu_accrual_cash_low_pb" in generated_names

    generated_by_name = {candidate.name: candidate for candidate in generated}
    smoke_generated_names = [
        "fund_atom_low_pb",
        "fund_combo_roe_low_pb",
        "fund_combo_revenue_profit_cash_hmean",
        "fund_ind_neu_roe_low_pb",
        "fund_atom_cashflow_to_liab",
        "fund_combo_reported_ocf_to_profit_low_pb",
        "fund_combo_fcf_margin_low_ps",
        "fund_ind_neu_accrual_cash_low_pb",
    ]
    for name in smoke_generated_names:
        values = generated_by_name[name].calculate(df)
        assert values.notna().any()
        assert len(values) == len(df)

    legacy_candidates = available_candidates(df, windows=(5,), factor_set="fundamental")
    names = {candidate.name for candidate in legacy_candidates}
    assert "quality_roe_ocf" in names
    assert "roe_value_pb" in names
    assert "free_cashflow_yield_quality" in names
    assert "solvency_liquidity_quality" in names
    assert "industry_neutral_roe_value_pb" in names
    assert "size_neutral_earnings_yield_quality" in names
    assert "decayed_quality_growth_20" in names

    legacy_smoke_names = {
        "quality_roe_ocf",
        "roe_value_pb",
        "industry_neutral_roe_value_pb",
    }
    candidates = [candidate for candidate in legacy_candidates if candidate.name in legacy_smoke_names]
    candidates.extend(generated_by_name[name] for name in smoke_generated_names[:2])
    miner = AlphaMiner(candidates, config=AlphaMiningConfig(min_abs_ic=0.0, min_coverage=0.5))
    result = miner.run(df)
    assert not result.summary.empty
    assert result.factor_values.filter(regex="quality_roe_ocf|roe_value_pb|fund_atom_low_pb").notna().any().all()


def test_fundamental_diagnostics_helpers():
    dates = pd.date_range("2026-01-02", periods=4, freq="B")
    rows = []
    for date in dates:
        for i in range(24):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"{i:06d}.SZ",
                    "factor_a": float(i),
                    "factor_b": float(23 - i),
                    "label": float(i) / 100.0,
                }
            )
    df = pd.DataFrame(rows)
    df["final_score"] = composite_score(df, ["factor_a", "factor_b"])
    daily = daily_factor_stats(df, "factor_a", n_groups=5)
    summary = summarize_daily_stats(daily, "ytd")

    assert df["final_score"].notna().all()
    assert np.isclose(df["final_score"].std(), 0.0)
    assert daily["RankIC"].dropna().gt(0.99).all()
    assert summary.loc[0, "period"] == "2026_YTD"
    assert summary.loc[0, "long_short_mean"] > 0


def test_index_enhancement_helpers():
    weights = _cap_and_normalize(pd.Series([0.9, 0.05, 0.05], index=list("abc")), max_weight=0.6)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.6000001

    daily = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-02", periods=5, freq="B"),
            "portfolio_return_net": [0.02, 0.01, -0.01, 0.0, 0.01],
            "benchmark_return": [0.01, 0.0, -0.02, 0.0, 0.0],
            "active_return": [0.01, 0.01, 0.01, 0.0, 0.01],
            "turnover": [0.5, 0.0, 0.1, 0.0, 0.0],
            "rebalanced": [True, False, True, False, False],
            "active_share": [0.1] * 5,
            "max_industry_active": [0.02] * 5,
            "n_holdings": [10] * 5,
        }
    )
    summary = summarize_returns(daily, "toy")
    assert summary["excess_total_return"] > 0
    assert summary["information_ratio"] > 0


def test_enhancement_stability_helpers():
    daily = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-02", periods=30, freq="B"),
            "portfolio_return_net": [0.002, -0.001, 0.001] * 10,
            "benchmark_return": [0.001, -0.0015, 0.0] * 10,
            "active_return": [0.001, 0.0005, 0.001] * 10,
            "turnover": [0.02] * 30,
            "rebalanced": [i % 10 == 0 for i in range(30)],
        }
    )
    month = summarize_periods(daily, "month")
    dist = summarize_distribution(daily)
    rolling = rolling_stability(daily, windows=(10,))

    assert month["daily_win_rate"].iloc[0] == 1.0
    assert dist.loc[dist["window"] == "ytd_2026", "daily_win_rate"].iloc[0] == 1.0
    assert rolling["win_rate_10d"].dropna().eq(1.0).all()


if __name__ == "__main__":
    test_safe_div_removes_infinite_values()
    test_alpha_miner_runs_and_selects_candidates()
    test_time_series_extra_operators()
    test_fundamental_pit_asof_alignment()
    test_fundamental_candidates_run_on_synthetic_panel()
    test_fundamental_diagnostics_helpers()
    test_index_enhancement_helpers()
    test_enhancement_stability_helpers()
    print("alpha mining smoke tests passed")
