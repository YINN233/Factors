"""Validate locally reproducible public alpha factors on CSI500."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factors.alpha.public_factors import (
    augment_public_factor_fields,
    calculate_public_factors,
    load_adjusted_price_panel,
    public_factor_availability,
)
from factors.alpha.validation import (
    ValidationConfig,
    add_forward_rank_labels,
    factor_correlation,
    select_features,
    validate_factors,
)


BASE_COLUMNS = [
    "trade_date",
    "ts_code",
    "industry",
    "csi500_index_weight",
    "operating_cf_margin_ttm",
    "ocf_to_or",
    "roe_ttm",
    "cashflow_to_profit",
    "debt_to_assets",
    "pb",
    "pe_ttm",
    "ps_ttm",
    "total_mv",
    "log_mv",
]


def load_csi500_context(processed_dir: Path, splits: list[str], suffix: str = "000905_SH") -> pd.DataFrame:
    frames = []
    for split in splits:
        path = processed_dir / f"{split}_fundamental_{suffix}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df[[col for col in BASE_COLUMNS if col in df.columns]])
    if not frames:
        raise FileNotFoundError("No CSI500 processed fundamental split files found")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_validation_panel(raw_dir: Path, processed_dir: Path, start: str, end: str, splits: list[str]) -> pd.DataFrame:
    context = load_csi500_context(processed_dir, splits)
    prices = load_adjusted_price_panel(raw_dir, start=start, end=end, ts_codes=context["ts_code"].unique())
    price_cols = [
        "trade_date",
        "ts_code",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "volume",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "log_mv",
    ]
    prices = prices[[col for col in price_cols if col in prices.columns]]
    panel = context.merge(prices, on=["trade_date", "ts_code"], how="left", suffixes=("_fund", ""))
    for col in ["pb", "pe_ttm", "ps_ttm", "total_mv", "log_mv"]:
        fund_col = f"{col}_fund"
        if fund_col in panel.columns:
            panel[col] = panel[col].fillna(panel[fund_col]) if col in panel.columns else panel[fund_col]
            panel = panel.drop(columns=[fund_col])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return augment_public_factor_fields(panel, raw_dir=raw_dir, start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260706")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--output", default="outputs/csi500_xgb_constrained_index_enhancement")
    parser.add_argument("--label", default="fwd_5d_rank")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    panel = build_validation_panel(Path(args.raw_dir), Path(args.processed_dir), args.start, args.end, splits)
    panel = add_forward_rank_labels(panel, price_col="close_adj", horizons=(1, 5))
    factor_values, metadata = calculate_public_factors(panel)
    factor_values.to_parquet(output_dir / "public_factor_values.parquet", index=False)
    public_factor_availability(panel).to_csv(output_dir / "public_factor_availability.csv", index=False)
    metadata.to_csv(output_dir / "public_factor_metadata.csv", index=False)

    factor_cols = [col for col in factor_values.columns if col not in {"trade_date", "ts_code"}]
    validation_input = panel[["trade_date", "ts_code", args.label]].merge(factor_values, on=["trade_date", "ts_code"], how="left")
    summary, details = validate_factors(validation_input, factor_cols, config=ValidationConfig(label_col=args.label))
    corr = factor_correlation(validation_input, factor_cols)
    selected = select_features(summary, validation_input, max_pair_corr=0.90)

    summary.to_csv(output_dir / "public_factor_validation_summary.csv", index=False)
    corr.to_csv(output_dir / "public_factor_correlation.csv", index=False)
    pd.DataFrame({"factor": selected}).to_csv(output_dir / "selected_model_features.csv", index=False)
    if details:
        pd.concat(details.values(), ignore_index=True).to_csv(output_dir / "public_factor_daily_ic.csv", index=False)

    print(f"panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()}")
    print(f"computed factors={len(factor_cols)}, selected={len(selected)}")
    if not summary.empty:
        cols = ["factor", "validation_status", "coverage", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean", "validation_reason"]
        print(summary[cols].head(20).to_string(index=False))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
