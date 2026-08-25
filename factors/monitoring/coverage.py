"""Coverage summaries and explicit company-database gap reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_SPECS = (
    ("futures_daily", "trade_date", "ts_code", "B", False),
    ("futures_mapping", "trade_date", "product", "B", False),
    ("index_daily", "trade_date", "ts_code", "B", False),
    ("index_weights", "trade_date", "con_code", "C", True),
    ("etf_daily_universe", "trade_date", "ts_code", "B", False),
    ("etf_shares_universe", "trade_date", "ts_code", "B", False),
)

DERIVED_DATASET_SPECS = (
    ("decision_summary", "trade_date", "product", "C", True),
    ("pair_basis", "trade_date", "long_product", "B", False),
)


def _default_processed_dir(raw_dir: Path) -> Path:
    resolved = raw_dir.resolve()
    if resolved.name == "monitoring" and resolved.parent.name == "raw" and resolved.parent.parent.name == "data":
        return resolved.parent.parent / "processed" / "monitoring"
    return resolved / "processed" / "monitoring"


def build_coverage_report(
    raw_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize consolidated caches against the observed index calendar."""

    root = Path(raw_dir)
    reference_path = root / f"index_daily_{start_date}_{end_date}.parquet"
    reference_dates: set[pd.Timestamp] = set()
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        if "trade_date" in reference.columns:
            reference_dates = set(pd.to_datetime(reference["trade_date"], errors="coerce").dropna().unique())
    rows: list[dict] = []
    derived_root = Path(processed_dir) if processed_dir is not None else _default_processed_dir(root)
    specs = [
        (root, *spec) for spec in DATASET_SPECS
    ] + [
        (derived_root, *spec) for spec in DERIVED_DATASET_SPECS
    ]
    for dataset_root, stem, date_column, entity_column, grade, proxy in specs:
        path = dataset_root / f"{stem}_{start_date}_{end_date}.parquet"
        if not path.exists():
            rows.append(
                {
                    "dataset": stem,
                    "rows": 0,
                    "date_min": pd.NaT,
                    "date_max": pd.NaT,
                    "observed_dates": 0,
                    "reference_dates": len(reference_dates),
                    "date_coverage": 0.0,
                    "entities": 0,
                    "quality_grade": "D",
                    "proxy": proxy,
                    "status": "missing",
                }
            )
            continue
        frame = pd.read_parquet(path)
        dates = pd.to_datetime(frame.get(date_column), errors="coerce").dropna() if date_column in frame else pd.Series(dtype="datetime64[ns]")
        observed = set(dates.unique())
        if stem == "index_weights":
            coverage = pd.NA
        elif reference_dates:
            coverage = len(observed.intersection(reference_dates)) / len(reference_dates)
        else:
            coverage = pd.NA
        rows.append(
            {
                "dataset": stem,
                "rows": len(frame),
                "date_min": dates.min() if not dates.empty else pd.NaT,
                "date_max": dates.max() if not dates.empty else pd.NaT,
                "observed_dates": len(observed),
                "reference_dates": len(reference_dates),
                "date_coverage": coverage,
                "entities": frame[entity_column].nunique() if entity_column in frame else 0,
                "quality_grade": grade if not frame.empty else "D",
                "proxy": proxy,
                "status": "available" if not frame.empty else "empty",
            }
        )
    return pd.DataFrame(rows)


def company_database_gaps() -> pd.DataFrame:
    """List source limitations that Tushare cannot remove in this MVP."""

    return pd.DataFrame(
        [
            {
                "priority": "high",
                "dataset": "daily_index_weights",
                "reason": "Tushare 仅提供月末/调仓快照，日度重估仍是 proxy",
                "required_fields": "index_code, con_code, trade_date, weight",
                "impact": "分红点数和行业归因无法逐日精确复刻报告",
            },
            {
                "priority": "high",
                "dataset": "historical_dividend_forecast_snapshots",
                "reason": "Tushare 可重建公告事件，但缺少卖方历史预测快照",
                "required_fields": "as_of_date, ts_code, forecast_cash_div, forecast_ex_date, status",
                "impact": "历史含预测分红基差只能标记为不完整",
            },
            {
                "priority": "medium",
                "dataset": "historical_etf_classification",
                "reason": "报告分类无公开标准字段，当前为人工审核初始池",
                "required_fields": "ts_code, category, effective_start, effective_end, review_source",
                "impact": "ETF 信号存在分类覆盖和存活偏差，最高为 C 级",
            },
            {
                "priority": "medium",
                "dataset": "wind_report_validation_snapshot",
                "reason": "需要报告同口径的 2026-07-29 分红预测与合约表",
                "required_fields": "contract, index_close, forecast_dividend_points, adjusted_basis",
                "impact": "只能解释 Tushare proxy 与报告值的差异，不能声称逐点复刻",
            },
        ]
    )


def write_coverage_outputs(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist coverage and external-source gaps for the dashboard."""

    target = Path(processed_dir)
    target.mkdir(parents=True, exist_ok=True)
    coverage = build_coverage_report(
        raw_dir,
        processed_dir=target,
        start_date=start_date,
        end_date=end_date,
    )
    gaps = company_database_gaps()
    coverage.to_parquet(target / f"data_coverage_{start_date}_{end_date}.parquet", index=False)
    coverage.to_csv(target / f"data_coverage_{start_date}_{end_date}.csv", index=False)
    gaps.to_csv(target / "company_database_gaps.csv", index=False)
    return coverage, gaps
