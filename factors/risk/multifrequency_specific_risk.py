"""Daily, monthly, and quarterly specific-risk estimation for CNE6 V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpecificRiskConfig:
    daily_window: int = 252
    daily_min_periods: int = 126
    daily_half_life: int = 63
    monthly_window: int = 36
    monthly_min_periods: int = 18
    monthly_half_life: int = 12
    quarterly_window: int = 20
    quarterly_min_periods: int = 8
    quarterly_half_life: int = 8
    minimum_period_completeness: float = 0.60
    daily_weight: float = 0.50
    monthly_weight: float = 0.30
    quarterly_weight: float = 0.20
    group_minimum: int = 5
    reliability_target: int = 252
    minimum_reliability: float = 0.20
    annualization_days: int = 252


def _finite_ewm_moments(
    values: np.ndarray,
    window: int,
    half_life: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    decay = float(np.exp(-np.log(2.0) / float(half_life)))
    powers = decay ** np.arange(len(values), dtype=float)

    def _weighted_sum(data: np.ndarray) -> np.ndarray:
        prefix = np.cumsum(data / powers)
        selected = prefix.copy()
        if len(data) > window:
            selected[window:] -= prefix[:-window]
        return powers * selected

    denominator = _weighted_sum(valid.astype(float))
    count_prefix = np.cumsum(valid.astype(float))
    counts = count_prefix.copy()
    if len(values) > window:
        counts[window:] -= count_prefix[:-window]
    mean = np.divide(
        _weighted_sum(filled),
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    second = np.divide(
        _weighted_sum(filled * filled),
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    variance = np.maximum(second - mean * mean, 0.0)
    mean[counts < min_periods] = np.nan
    variance[counts < min_periods] = np.nan
    return mean, variance, counts


def aggregate_complete_period_returns(
    specific_returns: pd.DataFrame,
    frequency: str,
    minimum_completeness: float = 0.60,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate additive residuals into completed, non-overlapping periods."""

    if frequency not in {"M", "Q"}:
        raise ValueError("frequency must be 'M' or 'Q'")
    required = {"trade_date", "ts_code", "specific_return"}
    missing = sorted(required.difference(specific_returns.columns))
    if missing:
        raise ValueError(f"specific returns missing required columns: {missing}")
    work = specific_returns[["trade_date", "ts_code", "specific_return"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["specific_return"] = pd.to_numeric(work["specific_return"], errors="coerce")
    work = work.dropna(subset=["trade_date", "ts_code"])
    if work.empty:
        return pd.DataFrame(
            columns=["ts_code", "period", "period_end", "period_return", "observations", "expected_days", "completeness"]
        )
    cutoff = pd.Timestamp(as_of_date) if as_of_date is not None else work["trade_date"].max()
    work["period"] = work["trade_date"].dt.to_period(frequency)
    current_period = cutoff.to_period(frequency)
    work = work[work["period"] < current_period].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["ts_code", "period", "period_end", "period_return", "observations", "expected_days", "completeness"]
        )

    calendar = (
        work[["trade_date", "period"]]
        .drop_duplicates()
        .groupby("period", sort=True)["trade_date"]
        .agg(period_end="max", expected_days="count")
        .reset_index()
    )
    aggregate = (
        work.groupby(["ts_code", "period"], sort=True)["specific_return"]
        .agg(period_return="sum", observations="count")
        .reset_index()
        .merge(calendar, on="period", how="left", validate="many_to_one")
    )
    aggregate["completeness"] = aggregate["observations"] / aggregate["expected_days"]
    aggregate = aggregate[aggregate["completeness"] >= minimum_completeness].copy()
    return aggregate.sort_values(["ts_code", "period_end"]).reset_index(drop=True)


def _daily_variance_component(
    specific_returns: pd.DataFrame,
    config: SpecificRiskConfig,
) -> pd.DataFrame:
    work = specific_returns[["trade_date", "ts_code", "specific_return"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["specific_return"] = pd.to_numeric(work["specific_return"], errors="coerce")
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    work["specific_variance_daily_component"] = np.nan
    work["specific_effective_observations"] = 0.0
    for _, positions in work.groupby("ts_code", sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        _, variance, counts = _finite_ewm_moments(
            work.loc[idx, "specific_return"].to_numpy(dtype=float),
            config.daily_window,
            config.daily_half_life,
            config.daily_min_periods,
        )
        work.loc[idx, "specific_variance_daily_component"] = variance
        work.loc[idx, "specific_effective_observations"] = counts
    return work.drop(columns="specific_return")


def _period_variance_component(
    aggregates: pd.DataFrame,
    window: int,
    min_periods: int,
    half_life: int,
    output_column: str,
) -> pd.DataFrame:
    if aggregates.empty:
        return pd.DataFrame(columns=["ts_code", "period_end", output_column])
    work = aggregates.sort_values(["ts_code", "period_end"]).reset_index(drop=True).copy()
    work[output_column] = np.nan
    for _, positions in work.groupby("ts_code", sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        _, period_variance, _ = _finite_ewm_moments(
            work.loc[idx, "period_return"].to_numpy(dtype=float),
            window,
            half_life,
            min_periods,
        )
        mean_days, _, _ = _finite_ewm_moments(
            work.loc[idx, "expected_days"].to_numpy(dtype=float),
            window,
            half_life,
            min_periods,
        )
        daily_variance = np.divide(
            period_variance,
            mean_days,
            out=np.full(len(idx), np.nan),
            where=mean_days > 0,
        )
        work.loc[idx, output_column] = daily_variance
    return work[["ts_code", "period_end", output_column]]


def _asof_period_component(
    daily_keys: pd.DataFrame,
    period_component: pd.DataFrame,
    output_column: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=daily_keys.index, dtype=float)
    groups = dict(tuple(period_component.groupby("ts_code", sort=False))) if not period_component.empty else {}
    for code, left in daily_keys.groupby("ts_code", sort=False):
        right = groups.get(code)
        if right is None or right.empty:
            continue
        ordered = left.sort_values("trade_date")
        merged = pd.merge_asof(
            ordered[["trade_date"]],
            right.sort_values("period_end")[["period_end", output_column]],
            left_on="trade_date",
            right_on="period_end",
            direction="backward",
            allow_exact_matches=False,
        )
        result.loc[ordered.index] = merged[output_column].to_numpy(dtype=float)
    return result


def blend_frequency_variances(
    frame: pd.DataFrame,
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> pd.DataFrame:
    out = frame.copy()
    columns = [
        "specific_variance_daily_component",
        "specific_variance_monthly_component",
        "specific_variance_quarterly_component",
    ]
    values = out.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    planned = np.asarray(weights, dtype=float)
    if (planned < 0).any() or planned.sum() <= 0:
        raise ValueError("specific-risk blend weights must be non-negative and non-zero")
    available = values.notna().to_numpy(dtype=float)
    actual = available * planned[None, :]
    denominator = actual.sum(axis=1)
    actual = np.divide(actual, denominator[:, None], out=np.zeros_like(actual), where=denominator[:, None] > 0)
    blended = np.nansum(values.to_numpy(dtype=float) * actual, axis=1)
    blended[denominator <= 0] = np.nan
    out["specific_variance_blended"] = blended
    out["specific_weight_daily"] = actual[:, 0]
    out["specific_weight_monthly"] = actual[:, 1]
    out["specific_weight_quarterly"] = actual[:, 2]
    return out


def apply_structural_shrinkage(
    frame: pd.DataFrame,
    group_minimum: int = 5,
    reliability_target: int = 252,
    minimum_reliability: float = 0.20,
    size_buckets: int = 5,
) -> pd.DataFrame:
    """Shrink individual estimates toward industry-size, industry, or market priors."""

    out = frame.copy()
    out["specific_prior_variance"] = np.nan
    out["specific_prior_level"] = pd.Series(pd.NA, index=out.index, dtype="string")
    for _, sub in out.groupby("trade_date", sort=False):
        idx = sub.index
        values = pd.to_numeric(sub["specific_variance_blended"], errors="coerce")
        market_prior = float(values.median())
        log_size = np.log(pd.to_numeric(sub["total_mv"], errors="coerce").where(lambda item: item > 0))
        if size_buckets <= 1:
            bucket = pd.Series(0, index=idx, dtype="Int64")
        else:
            try:
                bucket = pd.qcut(log_size, q=size_buckets, labels=False, duplicates="drop").astype("Int64")
            except ValueError:
                bucket = pd.Series(pd.NA, index=idx, dtype="Int64")
        temporary = pd.DataFrame(
            {
                "industry": sub["industry_sw_l1_code"].astype("string"),
                "bucket": bucket,
                "variance": values,
            },
            index=idx,
        )
        group_stats = temporary.groupby(["industry", "bucket"], dropna=False)["variance"].agg(["median", "count"])
        industry_stats = temporary.groupby("industry", dropna=False)["variance"].agg(["median", "count"])
        group_prior = temporary.set_index(["industry", "bucket"]).index.map(
            group_stats["median"].where(group_stats["count"] >= group_minimum)
        )
        industry_prior = temporary["industry"].map(
            industry_stats["median"].where(industry_stats["count"] >= group_minimum)
        )
        group_prior = pd.Series(group_prior.to_numpy(dtype=float), index=idx)
        industry_prior = pd.Series(industry_prior.to_numpy(dtype=float), index=idx)
        selected_prior = group_prior.fillna(industry_prior).fillna(market_prior)
        level = pd.Series("market", index=idx, dtype="string")
        level.loc[industry_prior.notna()] = "industry"
        level.loc[group_prior.notna()] = "industry_size"
        out.loc[idx, "specific_prior_variance"] = selected_prior
        out.loc[idx, "specific_prior_level"] = level

    observations = pd.to_numeric(out["specific_effective_observations"], errors="coerce").fillna(0.0)
    reliability = (observations / float(reliability_target)).clip(lower=minimum_reliability, upper=1.0)
    blended = pd.to_numeric(out["specific_variance_blended"], errors="coerce")
    reliability = reliability.where(blended.notna(), 0.0)
    out["specific_reliability"] = reliability
    out["specific_variance_daily"] = (
        reliability * blended.fillna(0.0)
        + (1.0 - reliability) * pd.to_numeric(out["specific_prior_variance"], errors="coerce")
    )
    return out


def estimate_multifrequency_specific_risk(
    specific_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    config: SpecificRiskConfig | None = None,
) -> pd.DataFrame:
    config = SpecificRiskConfig() if config is None else config
    if specific_returns.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("specific returns contain duplicate trade_date/ts_code rows")
    required_exposures = {"trade_date", "ts_code", "industry_sw_l1_code", "total_mv"}
    missing = sorted(required_exposures.difference(exposures.columns))
    if missing:
        raise ValueError(f"specific-risk exposures missing required columns: {missing}")

    daily = _daily_variance_component(specific_returns, config)
    monthly_aggregates = aggregate_complete_period_returns(
        specific_returns,
        "M",
        minimum_completeness=config.minimum_period_completeness,
    )
    quarterly_aggregates = aggregate_complete_period_returns(
        specific_returns,
        "Q",
        minimum_completeness=config.minimum_period_completeness,
    )
    monthly = _period_variance_component(
        monthly_aggregates,
        config.monthly_window,
        config.monthly_min_periods,
        config.monthly_half_life,
        "specific_variance_monthly_component",
    )
    quarterly = _period_variance_component(
        quarterly_aggregates,
        config.quarterly_window,
        config.quarterly_min_periods,
        config.quarterly_half_life,
        "specific_variance_quarterly_component",
    )
    daily["specific_variance_monthly_component"] = _asof_period_component(
        daily, monthly, "specific_variance_monthly_component"
    )
    daily["specific_variance_quarterly_component"] = _asof_period_component(
        daily, quarterly, "specific_variance_quarterly_component"
    )
    exposure_columns = ["trade_date", "ts_code", "industry_sw_l1_code", "total_mv"]
    exposure_frame = exposures[exposure_columns].copy()
    exposure_frame["trade_date"] = pd.to_datetime(exposure_frame["trade_date"])
    if exposure_frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("specific-risk exposures contain duplicate trade_date/ts_code rows")
    work = daily.merge(exposure_frame, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    work = blend_frequency_variances(
        work,
        weights=(config.daily_weight, config.monthly_weight, config.quarterly_weight),
    )
    work = apply_structural_shrinkage(
        work,
        group_minimum=config.group_minimum,
        reliability_target=config.reliability_target,
        minimum_reliability=config.minimum_reliability,
    )
    work["specific_risk_annualized"] = np.sqrt(
        pd.to_numeric(work["specific_variance_daily"], errors="coerce").clip(lower=0)
        * config.annualization_days
    )
    return work.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
