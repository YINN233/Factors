import pandas as pd

from factors.monitoring.coverage import build_coverage_report, company_database_gaps


def test_coverage_uses_observed_index_calendar(tmp_path):
    dates = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"])
    pd.DataFrame({"ts_code": ["IDX"] * 3, "trade_date": dates}).to_parquet(
        tmp_path / "index_daily_20260801_20260805.parquet", index=False
    )
    pd.DataFrame({"ts_code": ["IF1"] * 2, "trade_date": dates[:2]}).to_parquet(
        tmp_path / "futures_daily_20260801_20260805.parquet", index=False
    )
    report = build_coverage_report(tmp_path, start_date="20260801", end_date="20260805")
    futures = report[report["dataset"].eq("futures_daily")].iloc[0]
    assert futures["date_coverage"] == 2 / 3
    assert futures["status"] == "available"


def test_company_database_gaps_are_explicit():
    gaps = company_database_gaps()
    assert "daily_index_weights" in set(gaps["dataset"])
    assert set(gaps["priority"]).issubset({"high", "medium"})


def test_coverage_includes_dashboard_derivatives(tmp_path):
    raw = tmp_path / "raw" / "monitoring"
    processed = tmp_path / "processed" / "monitoring"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    dates = pd.to_datetime(["2026-08-05", "2026-08-06"])
    pd.DataFrame({"ts_code": ["IDX", "IDX"], "trade_date": dates}).to_parquet(
        raw / "index_daily_20260805_20260806.parquet", index=False
    )
    pd.DataFrame({"product": ["IF", "IF"], "trade_date": dates}).to_parquet(
        processed / "decision_summary_20260805_20260806.parquet", index=False
    )
    pd.DataFrame(
        {
            "long_product": ["IC", "IC"],
            "short_product": ["IM", "IM"],
            "trade_date": dates,
        }
    ).to_parquet(processed / "pair_basis_20260805_20260806.parquet", index=False)

    report = build_coverage_report(
        raw,
        processed_dir=processed,
        start_date="20260805",
        end_date="20260806",
    )
    derived = report[report["dataset"].isin(["decision_summary", "pair_basis"])]
    assert set(derived["status"]) == {"available"}
    assert dict(zip(derived["dataset"], derived["quality_grade"])) == {
        "decision_summary": "C",
        "pair_basis": "B",
    }
