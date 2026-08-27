"""Descriptor calculation and weighted style exposures for CNE6 V2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from factors.risk.cne6_v2_spec import DescriptorSpecV2, descriptor_specs_v2


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_div(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = pd.to_numeric(numerator, errors="coerce") / denominator.where(denominator.abs() > eps)
    return result.replace([np.inf, -np.inf], np.nan)


def _finite_ewm_moments(
    values: np.ndarray,
    window: int,
    half_life: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    decay = float(np.exp(-np.log(2.0) / float(half_life)))
    powers = decay ** np.arange(len(values), dtype=float)

    def _finite_weighted_sum(data: np.ndarray) -> np.ndarray:
        scaled_prefix = np.cumsum(data / powers)
        window_prefix = scaled_prefix.copy()
        if len(data) > window:
            window_prefix[window:] -= scaled_prefix[:-window]
        return powers * window_prefix

    denominator = _finite_weighted_sum(valid.astype(float))
    count_prefix = np.cumsum(valid.astype(float))
    count = count_prefix.copy()
    if len(values) > window:
        count[window:] -= count_prefix[:-window]
    mean = np.divide(
        _finite_weighted_sum(filled),
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    second = np.divide(
        _finite_weighted_sum(filled * filled),
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    variance = np.maximum(second - mean * mean, 0.0)
    mean[count < min_periods] = np.nan
    variance[count < min_periods] = np.nan
    return mean, variance


def _rolling_ewm_stat(
    frame: pd.DataFrame,
    values: pd.Series,
    window: int,
    half_life: int,
    min_periods: int,
    stat: str,
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, positions in frame.groupby("ts_code", sort=False).indices.items():
        position_array = np.asarray(positions, dtype=int)
        mean, variance = _finite_ewm_moments(
            pd.to_numeric(values.iloc[position_array], errors="coerce").to_numpy(dtype=float),
            window,
            half_life,
            min_periods,
        )
        output.iloc[position_array] = mean if stat == "mean" else np.sqrt(variance)
    return output


def _rolling_ewm_beta(
    frame: pd.DataFrame,
    window: int,
    half_life: int,
    min_periods: int,
    downside: bool = False,
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    stock = _num(frame, "returns_1d")
    market = _num(frame, "csi500_return")
    for _, positions in frame.groupby("ts_code", sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        x = stock.iloc[idx].to_numpy(dtype=float)
        y = market.iloc[idx].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if downside:
            valid &= y < 0
        x_masked = np.where(valid, x, np.nan)
        y_masked = np.where(valid, y, np.nan)
        mean_x, _ = _finite_ewm_moments(x_masked, window, half_life, min_periods)
        mean_y, var_y = _finite_ewm_moments(y_masked, window, half_life, min_periods)
        mean_xy, _ = _finite_ewm_moments(x_masked * y_masked, window, half_life, min_periods)
        covariance = mean_xy - mean_x * mean_y
        beta = np.divide(covariance, var_y, out=np.full(len(idx), np.nan), where=var_y > 1e-20)
        output.iloc[idx] = beta
    return output


def _rolling_transform(
    frame: pd.DataFrame,
    values: pd.Series,
    window: int,
    function: str,
    min_periods: int,
) -> pd.Series:
    temporary = pd.DataFrame({"ts_code": frame["ts_code"].to_numpy(), "value": values.to_numpy()}, index=frame.index)
    grouped = temporary.groupby("ts_code", sort=False)["value"]
    if function == "sum":
        return grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).sum())
    if function == "mean":
        return grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).mean())
    if function == "std":
        return grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).std())
    if function == "max":
        return grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).max())
    if function == "min":
        return grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).min())
    raise ValueError(f"unsupported rolling function: {function}")


def _group_shift(frame: pd.DataFrame, values: pd.Series, periods: int) -> pd.Series:
    temporary = pd.DataFrame({"ts_code": frame["ts_code"].to_numpy(), "value": values.to_numpy()}, index=frame.index)
    return temporary.groupby("ts_code", sort=False)["value"].shift(periods)


def _group_pct_change(frame: pd.DataFrame, values: pd.Series, periods: int) -> pd.Series:
    shifted = _group_shift(frame, values, periods)
    return _safe_div(values, shifted) - 1.0


def compute_raw_descriptors_v2(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate all V2 raw descriptors while preserving unavailable fields."""

    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    output = frame[["trade_date", "ts_code"]].copy()

    total_mv_rmb = _num(frame, "total_mv") * 10_000.0
    log_size = np.log(total_mv_rmb.where(total_mv_rmb > 0))
    output["log_total_mv"] = log_size
    output["nonlinear_size_residual"] = log_size**3

    beta_252 = _rolling_ewm_beta(frame, 252, 63, 126)
    output["beta_252_ewma"] = beta_252
    output["beta_504_ewma"] = _rolling_ewm_beta(frame, 504, 126, 252)
    output["downside_beta_252"] = _rolling_ewm_beta(frame, 252, 63, 60, downside=True)

    returns = _num(frame, "returns_1d")
    market = _num(frame, "csi500_return")
    output["dastd_252"] = _rolling_ewm_stat(frame, returns, 252, 42, 126, "std")
    residual = returns - beta_252 * market
    output["hsigma_252"] = _rolling_ewm_stat(frame, residual, 252, 63, 126, "std")
    log_excess = np.log1p(returns.where(returns > -1.0)) - np.log1p(market.where(market > -1.0))
    monthly_proxy = _rolling_transform(frame, log_excess, 21, "sum", 10)
    cmra_high = _rolling_transform(frame, monthly_proxy, 252, "max", 126)
    cmra_low = _rolling_transform(frame, monthly_proxy, 252, "min", 126)
    output["cmra_12m"] = cmra_high - cmra_low

    turnover = _num(frame, "turnover_rate") / 100.0
    output["stom_21"] = np.log(_rolling_transform(frame, turnover, 21, "sum", 10).where(lambda value: value > 0))
    output["stoq_63"] = np.log((_rolling_transform(frame, turnover, 63, "sum", 32) / 3.0).where(lambda value: value > 0))
    output["stoa_252"] = np.log((_rolling_transform(frame, turnover, 252, "sum", 126) / 12.0).where(lambda value: value > 0))
    amount_rmb = _num(frame, "amount") * 1_000.0
    amihud = _safe_div(returns.abs(), amount_rmb)
    output["amihud_63"] = np.log(_rolling_transform(frame, amihud, 63, "mean", 32).where(lambda value: value > 0))
    output["turnover_stability_63"] = _rolling_transform(frame, turnover, 63, "std", 32)

    shifted_excess = _group_shift(frame, log_excess, 21)
    output["rstr_12m_ex_1m"] = _rolling_transform(frame, shifted_excess, 231, "sum", 116)
    output["rstr_6m_ex_1m"] = _rolling_transform(frame, shifted_excess, 105, "sum", 53)
    output["momentum_ewma_252"] = _rolling_ewm_stat(frame, shifted_excess, 231, 126, 116, "mean")
    close = _num(frame, "close_adj")
    output["reversal_5d"] = -_group_pct_change(frame, close, 5)
    output["reversal_21d"] = -_group_pct_change(frame, close, 21)

    pb = _num(frame, "pb")
    pe = _num(frame, "pe_ttm")
    output["book_to_price"] = _safe_div(pd.Series(1.0, index=frame.index), pb.where(pb > 0))
    output["earnings_yield"] = _safe_div(pd.Series(1.0, index=frame.index), pe.where(pe > 0))
    output["cashflow_to_price"] = _safe_div(_num(frame, "n_cashflow_act_ttm"), total_mv_rmb)
    analyst_eps = _num(frame, "analyst_forward_eps_180")
    raw_close = _num(frame, "close")
    output["forecast_earnings_yield"] = _safe_div(analyst_eps, raw_close.where(raw_close > 0))

    direct_columns = [
        "revenue_yoy",
        "net_profit_yoy",
        "eps_growth",
        "roe_growth",
        "asset_turnover_yoy",
        "roe_ttm",
        "roa_ttm",
        "gross_margin_ttm",
        "operating_margin_ttm",
        "cashflow_to_profit",
        "earnings_stability",
        "asset_growth",
        "capex_growth",
        "inventory_growth",
        "working_capital_growth",
        "debt_to_assets",
        "book_leverage",
        "inverse_interest_coverage",
        "dv_ttm",
        "dv_ratio",
        "analyst_report_count_90",
        "analyst_org_count_180",
        "analyst_rating_score_180",
        "analyst_target_upside_180",
        "analyst_eps_revision_180",
    ]
    for column in direct_columns:
        output[column] = _num(frame, column)
    output["accrual_quality"] = _safe_div(
        _num(frame, "n_cashflow_act_ttm") - _num(frame, "net_profit_ttm"),
        _num(frame, "total_assets"),
    )
    output["market_leverage"] = _safe_div(
        _num(frame, "total_liab"),
        _num(frame, "total_liab") + total_mv_rmb,
    )

    specs = descriptor_specs_v2()
    for spec in specs:
        if spec.name not in output.columns:
            output[spec.name] = np.nan
        output[spec.name] = pd.to_numeric(output[spec.name], errors="coerce") * spec.direction
    return output[["trade_date", "ts_code"] + [spec.name for spec in specs]]


def weighted_residualize_by_date(
    frame: pd.DataFrame,
    target: str,
    controls: Sequence[str],
    weight_column: str,
) -> pd.Series:
    """Return WLS residuals independently for each cross-section."""

    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, sub in frame.groupby("trade_date", sort=False):
        columns = [target, weight_column] + list(controls)
        valid = sub[columns].replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[pd.to_numeric(valid[weight_column], errors="coerce") > 0]
        if len(valid) < len(controls) + 2:
            continue
        x = np.column_stack([np.ones(len(valid))] + [pd.to_numeric(valid[col], errors="coerce") for col in controls])
        y = pd.to_numeric(valid[target], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(valid[weight_column], errors="coerce").to_numpy(dtype=float)
        root = np.sqrt(weights / weights.mean())
        beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
        residuals.loc[valid.index] = y - x @ beta
    return residuals


def _standardize_one(values: pd.Series, weights: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() < 2:
        return output
    selected = values.loc[valid]
    median = selected.median()
    mad = (selected - median).abs().median()
    if pd.notna(mad) and mad > 0:
        clipped = selected.clip(median - 5.0 * mad, median + 5.0 * mad)
    else:
        lower, upper = selected.quantile([0.01, 0.99])
        clipped = selected.clip(lower, upper)
    selected_weights = weights.loc[valid]
    center = float(np.average(clipped, weights=selected_weights))
    scale = float(clipped.std(ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        output.loc[valid] = 0.0
    else:
        output.loc[valid] = (clipped - center) / scale
    return output


def robust_standardize_by_date(frame: pd.DataFrame, column: str, weight_column: str) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, sub in frame.groupby("trade_date", sort=False):
        output.loc[sub.index] = _standardize_one(sub[column], sub[weight_column])
    return output


def combine_style_exposures_v2(
    descriptor_exposures: pd.DataFrame,
    specs: Sequence[DescriptorSpecV2] | None = None,
    min_effective_weight: float = 0.60,
    standardize: bool = True,
    weight_column: str = "_sqrt_mv_weight",
) -> pd.DataFrame:
    specs = list(descriptor_specs_v2() if specs is None else specs)
    output = descriptor_exposures[["trade_date", "ts_code"]].copy()
    if weight_column in descriptor_exposures.columns:
        output[weight_column] = descriptor_exposures[weight_column]
    else:
        output[weight_column] = 1.0
    styles = list(dict.fromkeys(spec.style for spec in specs))
    for style in styles:
        style_specs = [spec for spec in specs if spec.style == style]
        names = [spec.name for spec in style_specs]
        weights = pd.Series({spec.name: spec.weight for spec in style_specs})
        values = descriptor_exposures.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
        available_weight = values.notna().mul(weights, axis=1).sum(axis=1)
        weighted_sum = values.fillna(0.0).mul(weights, axis=1).sum(axis=1)
        combined = weighted_sum / available_weight.where(available_weight >= min_effective_weight)
        output[f"style_{style}"] = combined
        output[f"style_{style}_effective_weight"] = available_weight
        output[f"style_{style}_n"] = values.notna().sum(axis=1)
        if standardize:
            output[f"style_{style}"] = robust_standardize_by_date(output, f"style_{style}", weight_column)
    return output


def _trailing_descriptor_availability(
    frame: pd.DataFrame,
    descriptor_columns: Sequence[str],
    lookback: int = 252,
    min_periods: int = 126,
    minimum_coverage: float = 0.70,
) -> pd.DataFrame:
    dates = pd.Index(pd.to_datetime(frame["trade_date"]).drop_duplicates()).sort_values()
    daily = frame.groupby("trade_date", sort=True)[list(descriptor_columns)].agg(lambda series: series.notna().mean())
    trailing = daily.shift(1).rolling(lookback, min_periods=min_periods).mean()
    allowed = trailing >= minimum_coverage
    allowed = allowed.reindex(dates)
    allowed.index.name = "trade_date"
    return allowed


def compute_v2_exposures(
    panel: pd.DataFrame,
    minimum_descriptor_coverage: float = 0.70,
    min_effective_weight: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build standardized descriptor and style exposures with diagnostics."""

    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    raw = compute_raw_descriptors_v2(frame)
    raw["_sqrt_mv_weight"] = np.sqrt(_num(frame, "total_mv").clip(lower=0)).replace(0, np.nan)
    if "csi500_member" in frame.columns:
        member = frame["csi500_member"].fillna(False).astype(bool)
        raw = raw.loc[member].reset_index(drop=True)
    descriptor_columns = [spec.name for spec in descriptor_specs_v2()]

    standardized = raw[["trade_date", "ts_code", "_sqrt_mv_weight"]].copy()
    for column in descriptor_columns:
        standardized[column] = robust_standardize_by_date(raw, column, "_sqrt_mv_weight")

    standardized["nonlinear_size_residual"] = weighted_residualize_by_date(
        standardized,
        "nonlinear_size_residual",
        ["log_total_mv"],
        "_sqrt_mv_weight",
    )
    standardized["nonlinear_size_residual"] = robust_standardize_by_date(
        standardized, "nonlinear_size_residual", "_sqrt_mv_weight"
    )

    availability = _trailing_descriptor_availability(
        standardized,
        descriptor_columns,
        minimum_coverage=minimum_descriptor_coverage,
    )
    availability_long = availability.stack().rename("is_admitted").reset_index()
    availability_long = availability_long.rename(columns={"level_1": "descriptor"})
    for column in descriptor_columns:
        admitted = standardized["trade_date"].map(availability[column]).fillna(False)
        standardized.loc[~admitted, column] = np.nan

    styles = combine_style_exposures_v2(
        standardized,
        min_effective_weight=min_effective_weight,
        standardize=True,
    )
    for target, controls in {
        "style_nonlinear_size": ["style_size"],
        "style_residual_volatility": ["style_size", "style_beta"],
        "style_liquidity": ["style_size"],
    }.items():
        styles[target] = weighted_residualize_by_date(styles, target, controls, "_sqrt_mv_weight")
        styles[target] = robust_standardize_by_date(styles, target, "_sqrt_mv_weight")

    return standardized, styles, availability_long
