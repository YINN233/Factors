import pandas as pd

from factors.monitoring.backfill import annual_ranges, latest_constituent_codes, merge_range_caches
from factors.monitoring.pipeline import build_dashboard_derivatives_from_cache


def test_annual_ranges_respect_partial_first_and_last_years():
    assert annual_ranges("20201215", "20220203") == [
        ("20201215", "20201231"),
        ("20210101", "20211231"),
        ("20220101", "20220203"),
    ]


def test_merge_range_caches_deduplicates_warmup_rows(tmp_path):
    first = pd.DataFrame(
        {"product": ["IF", "IF"], "trade_date": ["2020-12-31", "2021-01-04"], "value": [1, 2]}
    )
    second = pd.DataFrame(
        {"product": ["IF", "IF"], "trade_date": ["2021-01-04", "2022-01-04"], "value": [3, 4]}
    )
    first.to_parquet(tmp_path / "demo_20200101_20201231.parquet", index=False)
    second.to_parquet(tmp_path / "demo_20210101_20221231.parquet", index=False)
    result = merge_range_caches(
        tmp_path,
        stem="demo",
        ranges=[("20200101", "20201231"), ("20210101", "20221231")],
        output_start="20200101",
        output_end="20221231",
        key_columns=["product", "trade_date"],
    )
    assert len(result) == 3
    date = pd.Timestamp("2021-01-04")
    assert result.loc[result["trade_date"].eq(date), "value"].iloc[0] == 3


def test_latest_constituents_are_selected_per_product():
    weights = pd.DataFrame(
        {
            "product": ["IF", "IF", "IF", "IC", "IC"],
            "con_code": ["A", "B", "C", "X", "Y"],
            "trade_date": ["2025-12-31", "2026-07-31", "2026-07-31", "2026-06-30", "2026-07-31"],
        }
    )
    assert latest_constituent_codes(weights, "2026-08-01") == ["B", "C", "Y"]


def test_dashboard_derivatives_are_written_from_processed_inputs(tmp_path):
    target = tmp_path / "processed" / "monitoring"
    target.mkdir(parents=True)
    basis = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-05", "2026-08-05", "2026-08-06", "2026-08-06"]),
            "product": ["IF", "IC", "IF", "IC"],
            "ts_code": ["IF2609.CFX", "IC2609.CFX", "IF2609.CFX", "IC2609.CFX"],
            "expiry_date": pd.to_datetime(["2026-09-18"] * 4),
            "days_to_expiry": [44, 44, 43, 43],
            "tenor_rank": [1, 1, 1, 1],
            "is_main": [True, True, True, True],
            "raw_annualized_basis": [-0.02, -0.08, -0.01, -0.06],
            "annualized_basis": [-0.02, -0.08, -0.01, -0.06],
            "raw_historical_percentile": [50.0, 20.0, 70.0, 30.0],
            "dividend_source": ["unavailable"] * 4,
            "basis_quality": ["D"] * 4,
            "raw_basis_quality": ["B"] * 4,
        }
    )
    signals = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-05", "2026-08-06"]),
            "volume_signal": [-1.0, -1.0],
            "turnover_signal": [-1.0, -1.0],
            "share_adjusted_signal": [-1.0, -1.0],
            "IF_four_factor_signal": [-1.0, -1.0],
            "IC_four_factor_signal": [-1.0, -1.0],
            "signal_quality": ["C", "C"],
            "concentration_warning": [False, False],
        }
    )
    basis.to_parquet(target / "basis_table_20260805_20260806.parquet", index=False)
    signals.to_parquet(target / "etf_signals_20260805_20260806.parquet", index=False)

    summary, pairs = build_dashboard_derivatives_from_cache(
        tmp_path,
        start_date="20260805",
        end_date="20260806",
    )

    assert len(summary) == 4
    assert set(zip(pairs["long_product"], pairs["short_product"])) == {("IF", "IC"), ("IC", "IF")}
    assert (target / "decision_summary_20260805_20260806.parquet").exists()
    assert (target / "pair_basis_20260805_20260806.parquet").exists()
