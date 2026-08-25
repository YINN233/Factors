"""
Validation gate for mined and external alpha factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from factors.alpha.external_alpha import AlphaValidationStatus


@dataclass(frozen=True)
class ValidationConfig:
    label_col: str = "fwd_5d_rank"
    min_coverage: float = 0.50
    min_train_abs_rankic: float = 1e-6
    min_valid_rankic: float = 0.0
    min_test_rankic: float = 0.0
    min_ytd_rankic: float = -0.01
    min_ytd_positive_ratio: float = 0.45
    max_pair_corr: float = 0.90
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    ytd_start: str = "2026-01-01"


def add_forward_rank_labels(
    df: pd.DataFrame,
    price_col: str = "close_adj",
    horizons: Sequence[int] = (1, 5),
    date_col: str = "trade_date",
    code_col: str = "ts_code",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([code_col, date_col]).reset_index(drop=True)
    for horizon in horizons:
        ret_col = f"fwd_{horizon}d_return"
        rank_col = f"fwd_{horizon}d_rank"
        future = out.groupby(code_col, sort=False)[price_col].shift(-horizon)
        out[ret_col] = future / out[price_col] - 1.0
        out[rank_col] = out.groupby(date_col, sort=False)[ret_col].rank(pct=True) - 0.5
    return out


def daily_ic(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    label_rank_col = f"__rank_{label_col}"
    factor_rank_col = f"__rank_{factor_col}"
    cols = [date_col, factor_col, label_col]
    if label_rank_col in df.columns:
        cols.append(label_rank_col)
    if factor_rank_col in df.columns:
        cols.append(factor_rank_col)
    valid = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_col, label_col])
    if valid.empty:
        return pd.DataFrame(columns=[date_col, "IC", "RankIC", "n_stocks"])
    valid = valid.copy()
    if factor_rank_col in valid.columns:
        valid["_x_rank"] = valid[factor_rank_col]
    else:
        valid["_x_rank"] = valid.groupby(date_col, sort=False)[factor_col].rank(pct=True)
    if label_rank_col in valid.columns:
        valid["_y_rank"] = valid[label_rank_col]
    else:
        valid["_y_rank"] = valid.groupby(date_col, sort=False)[label_col].rank(pct=True)

    def _corr_frame(x_col: str, y_col: str, out_col: str) -> pd.DataFrame:
        grouped = valid.groupby(date_col, sort=False)
        mx = grouped[x_col].transform("mean")
        my = grouped[y_col].transform("mean")
        dx = valid[x_col] - mx
        dy = valid[y_col] - my
        tmp = pd.DataFrame({date_col: valid[date_col], "_cov": dx * dy, "_vx": dx * dx, "_vy": dy * dy})
        stats = tmp.groupby(date_col, sort=False).agg(_cov=("_cov", "sum"), _vx=("_vx", "sum"), _vy=("_vy", "sum"))
        stats[out_col] = stats["_cov"] / np.sqrt(stats["_vx"] * stats["_vy"])
        return stats[[out_col]]

    ic = _corr_frame(factor_col, label_col, "IC")
    rankic = _corr_frame("_x_rank", "_y_rank", "RankIC")
    counts = valid.groupby(date_col, sort=False).size().rename("n_stocks")
    out = ic.join(rankic).join(counts).reset_index()
    out = out[out["n_stocks"] >= 5]
    return out


def _period_mask(dates: pd.Series, start: str | None = None, end: str | None = None) -> pd.Series:
    mask = pd.Series(True, index=dates.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    return mask


def summarize_ic(ic_df: pd.DataFrame, prefix: str, start: str | None = None, end: str | None = None) -> dict:
    if ic_df.empty:
        return {
            f"{prefix}_rankic_mean": np.nan,
            f"{prefix}_rankic_ir": np.nan,
            f"{prefix}_rankic_positive_ratio": np.nan,
            f"{prefix}_ic_mean": np.nan,
            f"{prefix}_n_days": 0,
        }
    dates = pd.to_datetime(ic_df["trade_date"])
    sub = ic_df[_period_mask(dates, start, end)]
    rankic = sub["RankIC"].dropna()
    ic = sub["IC"].dropna()
    if rankic.empty:
        return {
            f"{prefix}_rankic_mean": np.nan,
            f"{prefix}_rankic_ir": np.nan,
            f"{prefix}_rankic_positive_ratio": np.nan,
            f"{prefix}_ic_mean": np.nan,
            f"{prefix}_n_days": 0,
        }
    return {
        f"{prefix}_rankic_mean": float(rankic.mean()),
        f"{prefix}_rankic_ir": float(rankic.mean() / (rankic.std(ddof=1) + 1e-12)),
        f"{prefix}_rankic_positive_ratio": float((rankic > 0).mean()),
        f"{prefix}_ic_mean": float(ic.mean()) if not ic.empty else np.nan,
        f"{prefix}_n_days": int(len(rankic)),
    }


def monthly_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    if ic_df.empty:
        return pd.DataFrame()
    out = ic_df.copy()
    out["month"] = pd.to_datetime(out["trade_date"]).dt.to_period("M").astype(str)
    return (
        out.groupby("month")
        .agg(
            rankic_mean=("RankIC", "mean"),
            rankic_positive_ratio=("RankIC", lambda s: float((s > 0).mean())),
            n_days=("RankIC", "count"),
        )
        .reset_index()
    )


def factor_group_returns(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    n_groups: int = 5,
    date_col: str = "trade_date",
    rank_col: str | None = None,
    rank_direction: float = 1.0,
) -> pd.DataFrame:
    cols = [date_col, factor_col, label_col]
    if rank_col and rank_col in df.columns:
        cols.append(rank_col)
    valid = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.DataFrame()
    counts = valid.groupby(date_col, sort=False)[factor_col].transform("size")
    valid = valid[counts >= n_groups * 5].copy()
    if valid.empty:
        return pd.DataFrame()
    if rank_col and rank_col in valid.columns:
        ranks = valid[rank_col]
        if rank_direction < 0:
            ranks = 1.0 - ranks
    else:
        ranks = valid.groupby(date_col, sort=False)[factor_col].rank(pct=True, method="first")
    valid["group"] = np.minimum((ranks * n_groups).astype(int), n_groups - 1)
    out = valid.groupby([date_col, "group"], sort=False)[label_col].mean().unstack("group")
    out = out.rename(columns={idx: f"G{idx + 1}" for idx in range(n_groups)}).reset_index()
    if not out.empty and f"G{n_groups}" in out:
        out["top_bottom"] = out[f"G{n_groups}"] - out["G1"]
    return out


def validate_factors(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    config: ValidationConfig = ValidationConfig(),
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[f"__rank_{config.label_col}"] = work.groupby(date_col, sort=False)[config.label_col].rank(pct=True)
    existing_factors = [factor for factor in factor_cols if factor in work.columns]
    if existing_factors:
        ranks = work.groupby(date_col, sort=False)[existing_factors].rank(pct=True)
        ranks.columns = [f"__rank_{col}" for col in ranks.columns]
        work = pd.concat([work, ranks], axis=1)
        coverage_by_factor = work[existing_factors].notna().groupby(work[date_col], sort=False).mean().mean()
    else:
        coverage_by_factor = pd.Series(dtype=float)
    details: dict[str, pd.DataFrame] = {}
    rows = []
    n_dates = work[date_col].nunique()
    for factor in existing_factors:
        coverage = float(coverage_by_factor.get(factor, 0.0))
        ic_raw = daily_ic(work, factor, config.label_col, date_col=date_col)
        train_summary = summarize_ic(ic_raw, "train", end=config.train_end)
        direction = np.sign(train_summary["train_rankic_mean"])
        if pd.isna(direction) or abs(train_summary["train_rankic_mean"]) < config.min_train_abs_rankic:
            direction = 1.0
        ic_adj = ic_raw.copy()
        if not ic_adj.empty:
            ic_adj["IC"] = ic_adj["IC"] * direction
            ic_adj["RankIC"] = ic_adj["RankIC"] * direction
        details[factor] = ic_adj.assign(factor=factor)

        row = {
            "factor": factor,
            "coverage": coverage,
            "n_dates": n_dates,
            "direction": float(direction),
        }
        row.update(summarize_ic(ic_adj, "full"))
        row.update(summarize_ic(ic_adj, "train", end=config.train_end))
        row.update(summarize_ic(ic_adj, "valid", start=config.valid_start, end=config.valid_end))
        row.update(summarize_ic(ic_adj, "test", start=config.test_start))
        row.update(summarize_ic(ic_adj, "ytd_2026", start=config.ytd_start))
        groups = factor_group_returns(
            work,
            factor,
            config.label_col,
            rank_col=f"__rank_{factor}",
            rank_direction=float(direction),
        )
        row["group_top_bottom_mean"] = float(groups["top_bottom"].mean()) if not groups.empty and "top_bottom" in groups else np.nan
        row["group_top_bottom_positive_ratio"] = (
            float((groups["top_bottom"] > 0).mean()) if not groups.empty and "top_bottom" in groups else np.nan
        )

        status = AlphaValidationStatus.PASSED.value
        reason = []
        if coverage < config.min_coverage:
            status = AlphaValidationStatus.FAILED.value
            reason.append("low_coverage")
        if row["valid_rankic_mean"] <= config.min_valid_rankic:
            status = AlphaValidationStatus.FAILED.value
            reason.append("valid_rankic_nonpositive")
        if row["test_rankic_mean"] < config.min_test_rankic:
            status = AlphaValidationStatus.QUARANTINED.value if status != AlphaValidationStatus.FAILED.value else status
            reason.append("test_rankic_negative")
        if (
            row["ytd_2026_rankic_mean"] < config.min_ytd_rankic
            or row["ytd_2026_rankic_positive_ratio"] < config.min_ytd_positive_ratio
        ):
            status = AlphaValidationStatus.QUARANTINED.value if status != AlphaValidationStatus.FAILED.value else status
            reason.append("ytd_2026_weak")
        row["validation_status"] = status
        row["validation_reason"] = ",".join(reason)
        rows.append(row)

    return pd.DataFrame(rows), details


def factor_correlation(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    date_col: str = "trade_date",
) -> pd.DataFrame:
    rows = []
    for date, sub in df.groupby(date_col, sort=False):
        valid_cols = [col for col in factor_cols if col in sub]
        corr = sub[valid_cols].corr(method="spearman", min_periods=20)
        if corr.empty:
            continue
        corr = corr.stack(dropna=True).reset_index()
        corr.columns = ["factor_a", "factor_b", "corr"]
        corr["trade_date"] = date
        rows.append(corr[corr["factor_a"] < corr["factor_b"]])
    if not rows:
        return pd.DataFrame(columns=["factor_a", "factor_b", "mean_abs_corr"])
    out = pd.concat(rows, ignore_index=True)
    out["abs_corr"] = out["corr"].abs()
    return out.groupby(["factor_a", "factor_b"]).agg(mean_abs_corr=("abs_corr", "mean")).reset_index()


def select_features(
    summary: pd.DataFrame,
    values: pd.DataFrame,
    max_pair_corr: float = 0.90,
    status_col: str = "validation_status",
    date_col: str = "trade_date",
) -> list[str]:
    eligible = summary[summary[status_col] == AlphaValidationStatus.PASSED.value].copy()
    if eligible.empty:
        return []
    eligible = eligible.sort_values(["test_rankic_mean", "valid_rankic_mean", "coverage"], ascending=False)
    eligible_factors = [factor for factor in eligible["factor"] if factor in values.columns]
    if not eligible_factors:
        return []

    # Use cross-sectional percentile ranks once, then a Pearson correlation
    # matrix on those ranks. This preserves the intended Spearman-like
    # redundancy screen without re-ranking the full panel for every pair.
    ranked_values = values.groupby(date_col, sort=False)[eligible_factors].rank(pct=True)
    corr_matrix = ranked_values.corr(method="pearson", min_periods=20)

    selected: list[str] = []
    for factor in eligible_factors:
        keep = True
        for chosen in selected:
            corr = corr_matrix.loc[factor, chosen] if factor in corr_matrix.index and chosen in corr_matrix.columns else np.nan
            if pd.notna(corr) and abs(corr) > max_pair_corr:
                keep = False
                break
        if keep:
            selected.append(factor)
    return selected
