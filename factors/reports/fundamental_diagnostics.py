"""
Diagnostics for mined fundamental factors.

The module reads saved factor values and processed panels, then reports
single-factor IC/RankIC, quantile long-short label spread, and composite-score
industry/stock exposure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_RECOMMENDED_FACTORS = [
    "dupont_value_quality",
    "cash_buffer_value",
    "robust_margin_value_ps",
    "growth_value_balance",
    "eps_bps_value_quality",
    "quality_growth_hmean",
    "industry_neutral_roe_value_pb",
    "liquidity_solvency_value",
]


def _read_existing_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    schema_names = set(pq.ParquetFile(path).schema.names)
    use_cols = [col for col in columns if col in schema_names]
    return pd.read_parquet(path, columns=use_cols)


def load_factor_panel(
    splits: Iterable[str],
    output_root: Path,
    processed_dir: Path,
    factors: list[str],
    suffix: str = "000905_SH",
) -> pd.DataFrame:
    frames = []
    for split in splits:
        factor_path = output_root / f"fundamental_alpha_csi500_{split}" / "factor_values.parquet"
        base_path = processed_dir / f"{split}_fundamental_{suffix}.parquet"
        factor_df = _read_existing_columns(factor_path, ["trade_date", "ts_code", "label"] + factors)
        base_df = _read_existing_columns(
            base_path,
            [
                "trade_date",
                "ts_code",
                "industry",
                "total_mv",
                "log_mv",
                "csi500_index_weight",
                "close_adj",
            ],
        )
        df = factor_df.merge(base_df, on=["trade_date", "ts_code"], how="left")
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def daily_factor_stats(
    df: pd.DataFrame,
    factor: str,
    label_col: str = "label",
    n_groups: int = 5,
) -> pd.DataFrame:
    records = []
    for date, sub in df[["trade_date", "ts_code", factor, label_col]].groupby("trade_date", sort=False):
        sub = sub.dropna(subset=[factor, label_col])
        if len(sub) < max(20, n_groups):
            continue
        rank_ic = sub[factor].corr(sub[label_col], method="spearman")
        ic = sub[factor].corr(sub[label_col], method="pearson")
        groups = pd.qcut(sub[factor], n_groups, labels=False, duplicates="drop")
        sub = sub.assign(group=groups)
        top_group = sub["group"].max()
        bottom_group = sub["group"].min()
        if pd.isna(top_group) or pd.isna(bottom_group) or top_group == bottom_group:
            long_short = np.nan
            top_mean = np.nan
            bottom_mean = np.nan
        else:
            top_mean = sub.loc[sub["group"] == top_group, label_col].mean()
            bottom_mean = sub.loc[sub["group"] == bottom_group, label_col].mean()
            long_short = top_mean - bottom_mean
        records.append(
            {
                "trade_date": date,
                "factor": factor,
                "IC": ic,
                "RankIC": rank_ic,
                "n_stocks": len(sub),
                "coverage": sub[factor].notna().mean(),
                "top_label_mean": top_mean,
                "bottom_label_mean": bottom_mean,
                "long_short": long_short,
            }
        )
    return pd.DataFrame(records)


def summarize_daily_stats(
    daily: pd.DataFrame,
    period_type: str,
    label_horizon: int = 10,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    daily = daily.copy()
    if period_type == "year":
        daily["period"] = daily["trade_date"].dt.year.astype(str)
    elif period_type == "month":
        daily["period"] = daily["trade_date"].dt.to_period("M").astype(str)
    elif period_type == "ytd":
        max_year = daily["trade_date"].dt.year.max()
        daily = daily[daily["trade_date"].dt.year == max_year].copy()
        daily["period"] = f"{max_year}_YTD"
    elif period_type == "full":
        daily["period"] = "full"
    else:
        raise ValueError("period_type must be one of: year, month, ytd, full")

    records = []
    for (factor, period), sub in daily.groupby(["factor", "period"], sort=False):
        rank_ic = sub["RankIC"].dropna()
        ic = sub["IC"].dropna()
        ls = sub["long_short"].dropna()
        records.append(
            {
                "factor": factor,
                "period_type": period_type,
                "period": period,
                "start_date": sub["trade_date"].min(),
                "end_date": sub["trade_date"].max(),
                "n_dates": sub["trade_date"].nunique(),
                "n_stocks_avg": sub["n_stocks"].mean(),
                "IC_mean": ic.mean(),
                "IC_IR": ic.mean() / (ic.std() + 1e-9),
                "IC_positive_ratio": (ic > 0).mean(),
                "RankIC_mean": rank_ic.mean(),
                "RankIC_IR": rank_ic.mean() / (rank_ic.std() + 1e-9),
                "RankIC_positive_ratio": (rank_ic > 0).mean(),
                "top_label_mean": sub["top_label_mean"].mean(),
                "bottom_label_mean": sub["bottom_label_mean"].mean(),
                "long_short_mean": ls.mean(),
                "long_short_sum": ls.sum(),
                "long_short_win_rate": (ls > 0).mean(),
                "label_horizon": label_horizon,
            }
        )
    return pd.DataFrame(records)


def composite_score(df: pd.DataFrame, factors: list[str]) -> pd.Series:
    pieces = []
    for factor in factors:
        if factor not in df.columns:
            continue
        rank = df.groupby("trade_date", sort=False)[factor].rank(pct=True)
        pieces.append(rank - 0.5)
    if not pieces:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=True)


def score_exposure_tables(
    df: pd.DataFrame,
    score_col: str = "final_score",
    ytd_year: int | None = None,
) -> dict[str, pd.DataFrame]:
    if ytd_year is None:
        ytd_year = int(df["trade_date"].dt.year.max())
    latest_date = df["trade_date"].max()
    latest = df[df["trade_date"] == latest_date].copy()
    latest["final_rank_pct"] = latest[score_col].rank(pct=True)
    latest["is_top_decile"] = latest["final_rank_pct"] >= 0.9
    latest["is_bottom_decile"] = latest["final_rank_pct"] <= 0.1

    universe_count = len(latest)
    top_count = max(1, int(latest["is_top_decile"].sum()))
    bottom_count = max(1, int(latest["is_bottom_decile"].sum()))
    industry_latest = (
        latest.groupby("industry", dropna=False)
        .agg(
            universe_count=("ts_code", "count"),
            avg_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            avg_rank_pct=("final_rank_pct", "mean"),
            top_decile_count=("is_top_decile", "sum"),
            bottom_decile_count=("is_bottom_decile", "sum"),
            index_weight_sum=("csi500_index_weight", "sum"),
            total_mv_sum=("total_mv", "sum"),
        )
        .reset_index()
    )
    industry_latest["universe_share"] = industry_latest["universe_count"] / universe_count
    industry_latest["top_decile_share"] = industry_latest["top_decile_count"] / top_count
    industry_latest["bottom_decile_share"] = industry_latest["bottom_decile_count"] / bottom_count
    industry_latest["top_minus_universe_share"] = industry_latest["top_decile_share"] - industry_latest["universe_share"]
    industry_latest = industry_latest.sort_values(["avg_score", "top_minus_universe_share"], ascending=False)

    ytd = df[df["trade_date"].dt.year == ytd_year].copy()
    ytd["daily_rank_pct"] = ytd.groupby("trade_date", sort=False)[score_col].rank(pct=True)
    ytd["is_top_decile"] = ytd["daily_rank_pct"] >= 0.9
    ytd_stock = (
        ytd.groupby("ts_code")
        .agg(
            ytd_avg_score=(score_col, "mean"),
            ytd_median_score=(score_col, "median"),
            ytd_avg_rank_pct=("daily_rank_pct", "mean"),
            ytd_top_decile_days=("is_top_decile", "sum"),
            ytd_days=("trade_date", "nunique"),
            ytd_label_mean=("label", "mean"),
        )
        .reset_index()
    )
    latest_cols = [
        "ts_code",
        "industry",
        "total_mv",
        "csi500_index_weight",
        score_col,
        "final_rank_pct",
    ]
    ytd_stock = ytd_stock.merge(latest[latest_cols], on="ts_code", how="left")
    ytd_stock = ytd_stock.sort_values(["ytd_avg_score", "final_rank_pct"], ascending=False)

    industry_day = (
        ytd.groupby(["trade_date", "industry"], dropna=False)
        .agg(
            avg_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            top_decile_share=("is_top_decile", "mean"),
            count=("ts_code", "count"),
        )
        .reset_index()
    )
    industry_ytd = (
        industry_day.groupby("industry", dropna=False)
        .agg(
            ytd_avg_score=("avg_score", "mean"),
            ytd_median_score=("median_score", "mean"),
            ytd_top_decile_share=("top_decile_share", "mean"),
            ytd_avg_count=("count", "mean"),
            ytd_days=("trade_date", "nunique"),
        )
        .reset_index()
        .sort_values(["ytd_avg_score", "ytd_top_decile_share"], ascending=False)
    )

    return {
        "latest_stock_scores": latest.sort_values(score_col, ascending=False),
        "latest_industry_exposure": industry_latest,
        "ytd_stock_score_exposure": ytd_stock,
        "ytd_industry_score_exposure": industry_ytd,
    }


def industry_factor_rankic(df: pd.DataFrame, factors: list[str], year: int, min_stocks: int = 10) -> pd.DataFrame:
    ytd = df[df["trade_date"].dt.year == year].copy()
    records = []
    for factor in factors:
        for (date, industry), sub in ytd.groupby(["trade_date", "industry"], sort=False):
            sub = sub.dropna(subset=[factor, "label"])
            if len(sub) < min_stocks:
                continue
            records.append(
                {
                    "factor": factor,
                    "trade_date": date,
                    "industry": industry,
                    "RankIC": sub[factor].corr(sub["label"], method="spearman"),
                    "n_stocks": len(sub),
                }
            )
    daily = pd.DataFrame(records)
    if daily.empty:
        return daily
    out = (
        daily.groupby(["factor", "industry"])
        .agg(
            RankIC_mean=("RankIC", "mean"),
            RankIC_IR=("RankIC", lambda s: s.mean() / (s.std() + 1e-9)),
            positive_ratio=("RankIC", lambda s: (s > 0).mean()),
            n_dates=("trade_date", "nunique"),
            n_stocks_avg=("n_stocks", "mean"),
        )
        .reset_index()
        .sort_values(["factor", "RankIC_mean"], ascending=[True, False])
    )
    return out


def add_stock_names(df: pd.DataFrame, stock_company_path: Path) -> pd.DataFrame:
    if not stock_company_path.exists():
        return df
    names = pd.read_parquet(stock_company_path, columns=["ts_code", "name"])
    return df.merge(names, on="ts_code", how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--summary-path", default="outputs/fundamental_alpha_csi500_summary/stable_factors.csv")
    parser.add_argument("--output", default="outputs/fundamental_alpha_csi500_diagnostics")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--label-horizon", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable = pd.read_csv(args.summary_path)
    stable = stable.sort_values(["min_abs_rankic", "avg_abs_rankic"], ascending=False)
    stable_factors = stable["factor"].tolist()
    recommended = [factor for factor in DEFAULT_RECOMMENDED_FACTORS if factor in stable_factors]
    if len(recommended) < args.top_n:
        recommended = stable_factors[: args.top_n]
    else:
        recommended = recommended[: args.top_n]

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    panel = load_factor_panel(
        splits=splits,
        output_root=Path(args.output_root),
        processed_dir=Path(args.processed_dir),
        factors=stable_factors,
    )
    panel["final_score"] = composite_score(panel, recommended)

    daily_frames = [daily_factor_stats(panel, factor) for factor in stable_factors]
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(output_dir / "daily_factor_performance.csv", index=False)

    period_summary = pd.concat(
        [
            summarize_daily_stats(daily, "full", label_horizon=args.label_horizon),
            summarize_daily_stats(daily, "year", label_horizon=args.label_horizon),
            summarize_daily_stats(daily, "month", label_horizon=args.label_horizon),
            summarize_daily_stats(daily, "ytd", label_horizon=args.label_horizon),
        ],
        ignore_index=True,
    )
    period_summary.to_csv(output_dir / "factor_period_performance.csv", index=False)

    ytd_year = int(panel["trade_date"].dt.year.max())
    exposures = score_exposure_tables(panel, ytd_year=ytd_year)
    stock_company = Path("data/raw/stock_company.parquet")
    for name, table in exposures.items():
        if "ts_code" in table.columns:
            table = add_stock_names(table, stock_company)
        table.to_csv(output_dir / f"{name}.csv", index=False)

    ind_rankic = industry_factor_rankic(panel, recommended, year=ytd_year)
    ind_rankic.to_csv(output_dir / "ytd_industry_factor_rankic.csv", index=False)

    pd.Series(recommended, name="factor").to_csv(output_dir / "recommended_factors_used.csv", index=False)
    print(f"panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()}")
    print(f"stable factors={len(stable_factors)}, recommended={recommended}")
    print(f"latest date={panel['trade_date'].max().date()}, ytd year={ytd_year}")
    print(f"wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
