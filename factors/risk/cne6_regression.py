"""Daily cross-sectional factor return regression for CNE6-style exposures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _next_return(panel: pd.DataFrame) -> pd.Series:
    if "fwd_1d_return" in panel.columns:
        return pd.to_numeric(panel["fwd_1d_return"], errors="coerce")
    if "close_adj" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    return panel.groupby("ts_code", sort=False)["close_adj"].shift(-1) / panel["close_adj"] - 1.0


def _same_day_return(panel: pd.DataFrame) -> pd.Series:
    if "returns_1d" in panel.columns:
        return pd.to_numeric(panel["returns_1d"], errors="coerce")
    if "close_adj" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    return panel.groupby("ts_code", sort=False)["close_adj"].pct_change()


def _regression_work(
    panel: pd.DataFrame,
    style_exposures: pd.DataFrame,
    return_mode: str = "forward_1d",
) -> tuple[pd.DataFrame, list[str]]:
    valid_modes = {"forward_1d", "same_day", "lagged_exposure_1d"}
    if return_mode not in valid_modes:
        raise ValueError(f"return_mode must be one of {sorted(valid_modes)}, got {return_mode!r}")

    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    style_exposures = style_exposures.copy()
    style_exposures["trade_date"] = pd.to_datetime(style_exposures["trade_date"])
    style_cols = [c for c in style_exposures.columns if c.startswith("style_") and not c.endswith("_n")]

    work = panel.merge(style_exposures[["trade_date", "ts_code"] + style_cols], on=["trade_date", "ts_code"], how="left")
    if "industry" not in work.columns:
        work["industry"] = "unknown"
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if return_mode == "forward_1d":
        work["_y"] = _next_return(work)
    else:
        work["_y"] = _same_day_return(work)
        if return_mode == "lagged_exposure_1d":
            exposure_cols = [c for c in ["industry", "total_mv"] + style_cols if c in work.columns]
            work[exposure_cols] = work.groupby("ts_code", sort=False)[exposure_cols].shift(1)

    work = work.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return work, style_cols


def _standardize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return values * 0.0
    return (values - values.mean(skipna=True)) / std


def _wls(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    root_w = np.sqrt(weights / np.nanmean(weights))
    xw = x * root_w[:, None]
    yw = y * root_w
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    ss_res = float(np.nansum(weights * resid**2))
    y_bar = float(np.average(y, weights=weights))
    ss_tot = float(np.nansum(weights * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan
    cond = float(np.linalg.cond(xw)) if xw.size else np.nan
    return beta, resid, r2, cond


def _pool_rare_industries(industry: pd.Series, min_obs: int) -> pd.Series:
    """Pool sparse daily industry buckets without changing the stock universe."""

    industry = industry.fillna("unknown").astype(str)
    if min_obs <= 1:
        return industry
    counts = industry.value_counts(dropna=False)
    rare = counts[counts < min_obs].index
    return industry.where(~industry.isin(rare), "other")


def _robust_wls(
    y: np.ndarray,
    x: np.ndarray,
    base_weights: np.ndarray,
    max_iter: int = 4,
    huber_k: float = 1.345,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Huber-style iteratively reweighted least squares.

    The returned R2 is always evaluated with the original WLS weights, so the
    robust fit cannot manufacture a better score merely by changing the
    evaluation metric.
    """

    base_weights = np.where(np.isfinite(base_weights) & (base_weights > 0), base_weights, 1.0)
    fit_weights = base_weights.copy()
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(max_iter):
        beta, *_ = np.linalg.lstsq(x * np.sqrt(fit_weights / np.nanmean(fit_weights))[:, None], y * np.sqrt(fit_weights / np.nanmean(fit_weights)), rcond=None)
        resid = y - x @ beta
        center = float(np.median(resid))
        scale = float(1.4826 * np.median(np.abs(resid - center)))
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        u = np.abs(resid - center) / (huber_k * scale)
        robust_multiplier = np.ones_like(u)
        clipped = np.isfinite(u) & (u > 1.0)
        robust_multiplier[clipped] = 1.0 / u[clipped]
        new_weights = base_weights * robust_multiplier
        if np.max(np.abs(new_weights - fit_weights) / np.maximum(fit_weights, 1e-12)) < 1e-4:
            fit_weights = new_weights
            break
        fit_weights = new_weights

    fitted = x @ beta
    resid = y - fitted
    ss_res = float(np.sum(base_weights * resid**2))
    y_bar = float(np.average(y, weights=base_weights))
    ss_tot = float(np.sum(base_weights * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan
    return beta, resid, r2


def _date_regression(
    sub: pd.DataFrame,
    style_cols: list[str],
    industry_min_obs: int = 0,
    fit_method: str = "wls",
    winsor_quantiles: tuple[float, float] | None = None,
    include_industry: bool = True,
) -> tuple[dict, pd.DataFrame]:
    date = sub["trade_date"].iloc[0]
    sub = sub.copy()
    sub["_y"] = pd.to_numeric(sub["_y"], errors="coerce")
    use_styles = [col for col in style_cols if col in sub.columns and sub[col].notna().mean() >= 0.5]
    needed = ["_y", "ts_code", "industry"]
    valid = sub.dropna(subset=needed).copy()
    if valid.empty:
        return {"trade_date": date, "regression_status": "failed_empty"}, pd.DataFrame()

    style_values = []
    for col in use_styles:
        z = _standardize(valid[col]).fillna(0.0)
        style_values.append(z.to_numpy())
    if include_industry:
        industry = _pool_rare_industries(valid["industry"], industry_min_obs)
        dummies = pd.get_dummies(industry, prefix="industry", dtype=float)
        effective_industries = int(dummies.shape[1])
        if dummies.shape[1] > 1:
            dummies = dummies.iloc[:, 1:]
        elif dummies.shape[1] == 1:
            dummies = dummies.iloc[:, 0:0]
    else:
        dummies = pd.DataFrame(index=valid.index)
        effective_industries = 0

    columns = ["country"] + use_styles + dummies.columns.tolist()
    parts = [np.ones(len(valid))]
    parts.extend(style_values)
    if not dummies.empty:
        parts.extend(dummies.to_numpy().T)
    x = np.column_stack(parts)
    y = valid["_y"].to_numpy(dtype=float)
    if "total_mv" in valid.columns:
        weights = np.sqrt(pd.to_numeric(valid["total_mv"], errors="coerce").clip(lower=0).fillna(0.0).to_numpy())
    else:
        weights = np.ones(len(valid))

    min_obs = max(30, x.shape[1] + 5)
    if len(valid) < min_obs or (include_industry and effective_industries < 2):
        return {
            "trade_date": date,
            "regression_status": "failed_insufficient_sample",
            "n_obs": int(len(valid)),
            "n_factors": int(x.shape[1]),
            "n_industries": effective_industries,
            "n_styles": int(len(use_styles)),
        }, pd.DataFrame()

    fit_y = y.copy()
    if winsor_quantiles is not None:
        lower_q, upper_q = winsor_quantiles
        lower, upper = np.nanquantile(fit_y, [lower_q, upper_q])
        fit_y = np.clip(fit_y, lower, upper)

    try:
        if fit_method == "wls":
            beta, _, _, cond = _wls(fit_y, x, weights)
            fitted = x @ beta
            resid = y - fitted
            ss_res = float(np.sum(weights * resid**2))
            y_bar = float(np.average(y, weights=weights))
            ss_tot = float(np.sum(weights * (y - y_bar) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan
        elif fit_method == "huber":
            beta, resid, r2 = _robust_wls(fit_y, x, weights)
            resid = y - x @ beta
            ss_res = float(np.sum(weights * resid**2))
            y_bar = float(np.average(y, weights=weights))
            ss_tot = float(np.sum(weights * (y - y_bar) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan
            root_w = np.sqrt(weights / np.nanmean(weights))
            cond = float(np.linalg.cond(x * root_w[:, None])) if x.size else np.nan
        else:
            raise ValueError(f"fit_method must be 'wls' or 'huber', got {fit_method!r}")
    except np.linalg.LinAlgError:
        return {
            "trade_date": date,
            "regression_status": "failed_linalg",
            "n_obs": int(len(valid)),
            "n_factors": int(x.shape[1]),
            "n_industries": effective_industries,
            "n_styles": int(len(use_styles)),
        }, pd.DataFrame()

    row = {
        "trade_date": date,
        "regression_status": "ok",
        "n_obs": int(len(valid)),
        "n_factors": int(x.shape[1]),
        "n_industries": effective_industries,
        "n_styles": int(len(use_styles)),
        "r2": r2,
        "adj_r2": 1.0 - (1.0 - r2) * (len(valid) - 1) / max(len(valid) - x.shape[1], 1) if np.isfinite(r2) else np.nan,
        "condition_number": cond,
        "resid_mean": float(np.nanmean(resid)),
        "resid_std": float(np.nanstd(resid)),
    }
    for col, val in zip(columns, beta):
        row[col] = float(val)

    residuals = valid[["trade_date", "ts_code"]].copy()
    residuals["fitted_return"] = x @ beta
    residuals["specific_return"] = resid
    return row, residuals


def run_factor_return_regression(
    panel: pd.DataFrame,
    style_exposures: pd.DataFrame,
    return_mode: str = "forward_1d",
    industry_min_obs: int = 0,
    fit_method: str = "wls",
    winsor_quantiles: tuple[float, float] | None = None,
    include_industry: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work, style_cols = _regression_work(panel, style_exposures, return_mode=return_mode)
    factor_rows = []
    resid_frames = []
    for _, sub in work.groupby("trade_date", sort=False):
        row, resid = _date_regression(
            sub,
            style_cols,
            industry_min_obs=industry_min_obs,
            fit_method=fit_method,
            winsor_quantiles=winsor_quantiles,
            include_industry=include_industry,
        )
        factor_rows.append(row)
        if not resid.empty:
            resid_frames.append(resid)
    factor_returns = pd.DataFrame(factor_rows)
    diagnostics_cols = [
        "trade_date",
        "regression_status",
        "n_obs",
        "n_factors",
        "n_industries",
        "n_styles",
        "r2",
        "adj_r2",
        "condition_number",
        "resid_mean",
        "resid_std",
    ]
    diagnostics = factor_returns[[c for c in diagnostics_cols if c in factor_returns.columns]].copy()
    diagnostics["return_mode"] = return_mode
    factor_cols = [c for c in factor_returns.columns if c not in diagnostics_cols]
    factor_returns = factor_returns[["trade_date"] + factor_cols].copy()
    residuals = pd.concat(resid_frames, ignore_index=True) if resid_frames else pd.DataFrame(columns=["trade_date", "ts_code", "fitted_return", "specific_return"])
    return factor_returns, residuals, diagnostics


def constrained_regression_work(
    panel: pd.DataFrame,
    style_exposures: pd.DataFrame,
    industry_column: str = "industry_sw_l1_code",
) -> tuple[pd.DataFrame, list[str]]:
    """Align t returns with t-1 V2 style, industry, and size exposures."""

    required_panel = {"trade_date", "ts_code", "returns_1d", industry_column, "total_mv"}
    missing = sorted(required_panel.difference(panel.columns))
    if missing:
        raise ValueError(f"V2 regression panel missing required columns: {missing}")
    required_styles = {"trade_date", "ts_code"}
    missing_styles = sorted(required_styles.difference(style_exposures.columns))
    if missing_styles:
        raise ValueError(f"V2 style exposures missing required columns: {missing_styles}")

    left = panel.copy()
    right = style_exposures.copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    right["trade_date"] = pd.to_datetime(right["trade_date"])
    style_columns = sorted(
        column
        for column in right.columns
        if column.startswith("style_")
        and not column.endswith("_n")
        and not column.endswith("_effective_weight")
    )
    work = left.merge(
        right[["trade_date", "ts_code"] + style_columns],
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    work["_y"] = pd.to_numeric(work["returns_1d"], errors="coerce")
    lagged_columns = [industry_column, "total_mv"] + style_columns
    work[lagged_columns] = work.groupby("ts_code", sort=False)[lagged_columns].shift(1)
    if "csi500_member" in work.columns:
        work = work.loc[work["csi500_member"].fillna(False).astype(bool)].copy()
    return work.sort_values(["trade_date", "ts_code"]).reset_index(drop=True), style_columns


def _constraint_null_space(constraint: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
    rank = int((singular_values > tolerance).sum())
    return vh[rank:].T


def _constrained_date_regression(
    sub: pd.DataFrame,
    style_columns: list[str],
    industry_column: str,
    min_observations: int,
) -> tuple[dict, pd.DataFrame]:
    date = pd.Timestamp(sub["trade_date"].iloc[0])
    work = sub.copy()
    work["_y"] = pd.to_numeric(work["_y"], errors="coerce")
    usable_styles = [
        column
        for column in style_columns
        if column in work.columns and work[column].notna().mean() >= 0.50
    ]
    valid = work.dropna(subset=["_y", "ts_code", industry_column, "total_mv"]).copy()
    valid["total_mv"] = pd.to_numeric(valid["total_mv"], errors="coerce")
    valid = valid[valid["total_mv"] > 0]
    if valid.empty:
        return {"trade_date": date, "regression_status": "failed_empty"}, pd.DataFrame()

    for column in usable_styles:
        valid[column] = pd.to_numeric(valid[column], errors="coerce").fillna(0.0)
    industries = valid[industry_column].astype("string")
    industry_dummies = pd.get_dummies(industries, prefix="industry", dtype=float)
    industry_columns = sorted(industry_dummies.columns)
    industry_dummies = industry_dummies[industry_columns]
    if len(industry_columns) < 2:
        return {
            "trade_date": date,
            "regression_status": "failed_insufficient_industries",
            "n_obs": int(len(valid)),
            "n_industries": int(len(industry_columns)),
        }, pd.DataFrame()

    x_parts = [np.ones(len(valid), dtype=float)]
    x_parts.extend(valid[column].to_numpy(dtype=float) for column in usable_styles)
    x_parts.extend(industry_dummies.to_numpy(dtype=float).T)
    x = np.column_stack(x_parts)
    factor_columns = ["country"] + usable_styles + industry_columns
    y = valid["_y"].to_numpy(dtype=float)
    market_cap = valid["total_mv"].to_numpy(dtype=float)
    regression_weights = np.sqrt(market_cap)

    industry_caps = (
        pd.DataFrame({"industry": industries.to_numpy(), "market_cap": market_cap})
        .groupby("industry", sort=False)["market_cap"]
        .sum()
    )
    industry_codes = [column.removeprefix("industry_") for column in industry_columns]
    cap_weights = np.array([industry_caps.loc[code] for code in industry_codes], dtype=float)
    cap_weights = cap_weights / cap_weights.sum()
    constraint = np.zeros((1, x.shape[1]), dtype=float)
    constraint[0, 1 + len(usable_styles):] = cap_weights
    null_space = _constraint_null_space(constraint)

    required = max(min_observations, null_space.shape[1] + 5)
    if len(valid) < required:
        return {
            "trade_date": date,
            "regression_status": "failed_insufficient_sample",
            "n_obs": int(len(valid)),
            "n_factors": int(x.shape[1]),
            "n_industries": int(len(industry_columns)),
            "n_styles": int(len(usable_styles)),
        }, pd.DataFrame()

    root_weight = np.sqrt(regression_weights / regression_weights.mean())
    reduced_x = x @ null_space
    weighted_x = reduced_x * root_weight[:, None]
    weighted_y = y * root_weight
    rank = int(np.linalg.matrix_rank(weighted_x))
    if rank < reduced_x.shape[1]:
        return {
            "trade_date": date,
            "regression_status": "failed_rank_deficient",
            "n_obs": int(len(valid)),
            "n_factors": int(x.shape[1]),
            "n_industries": int(len(industry_columns)),
            "n_styles": int(len(usable_styles)),
            "matrix_rank": rank,
        }, pd.DataFrame()

    try:
        theta, *_ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "trade_date": date,
            "regression_status": "failed_linalg",
            "n_obs": int(len(valid)),
            "n_factors": int(x.shape[1]),
            "n_industries": int(len(industry_columns)),
            "n_styles": int(len(usable_styles)),
        }, pd.DataFrame()

    beta = null_space @ theta
    fitted = x @ beta
    residual = y - fitted
    y_bar = float(np.average(y, weights=regression_weights))
    ss_res = float(np.sum(regression_weights * residual**2))
    ss_tot = float(np.sum(regression_weights * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan
    condition = float(np.linalg.cond(weighted_x))
    constraint_residual = float((constraint @ beta)[0])

    row = {
        "trade_date": date,
        "regression_status": "ok",
        "n_obs": int(len(valid)),
        "n_factors": int(x.shape[1]),
        "n_industries": int(len(industry_columns)),
        "n_styles": int(len(usable_styles)),
        "matrix_rank": rank,
        "r2": r2,
        "adj_r2": 1.0 - (1.0 - r2) * (len(valid) - 1) / max(len(valid) - reduced_x.shape[1], 1) if np.isfinite(r2) else np.nan,
        "condition_number": condition,
        "industry_constraint_residual": constraint_residual,
        "resid_mean": float(np.mean(residual)),
        "resid_std": float(np.std(residual)),
    }
    row.update({column: float(value) for column, value in zip(factor_columns, beta)})
    residuals = valid[["trade_date", "ts_code"]].copy()
    residuals["fitted_return"] = fitted
    residuals["specific_return"] = residual
    return row, residuals


def run_constrained_factor_return_regression(
    panel: pd.DataFrame,
    style_exposures: pd.DataFrame,
    industry_column: str = "industry_sw_l1_code",
    min_observations: int = 30,
    fail_if_all_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate V2 country, style, and all-industry returns with a linear constraint."""

    work, style_columns = constrained_regression_work(
        panel,
        style_exposures,
        industry_column=industry_column,
    )
    factor_rows = []
    residual_frames = []
    for _, sub in work.groupby("trade_date", sort=True):
        row, residuals = _constrained_date_regression(
            sub,
            style_columns,
            industry_column,
            min_observations,
        )
        factor_rows.append(row)
        if not residuals.empty:
            residual_frames.append(residuals)
    all_rows = pd.DataFrame(factor_rows)
    if all_rows.empty or not all_rows.get("regression_status", pd.Series(dtype="string")).eq("ok").any():
        if fail_if_all_invalid:
            raise ValueError("no valid constrained regressions")
    diagnostic_columns = [
        "trade_date",
        "regression_status",
        "n_obs",
        "n_factors",
        "n_industries",
        "n_styles",
        "matrix_rank",
        "r2",
        "adj_r2",
        "condition_number",
        "industry_constraint_residual",
        "resid_mean",
        "resid_std",
    ]
    diagnostics = all_rows.reindex(columns=diagnostic_columns)
    factor_columns = [column for column in all_rows.columns if column not in diagnostic_columns]
    factor_returns = all_rows.reindex(columns=["trade_date"] + [column for column in factor_columns if column != "trade_date"])
    residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame(columns=["trade_date", "ts_code", "fitted_return", "specific_return"])
    )
    return factor_returns, residuals, diagnostics


def run(
    panel_path: Path,
    exposures_path: Path,
    output_dir: Path,
    return_mode: str = "forward_1d",
    industry_min_obs: int = 0,
    fit_method: str = "wls",
    winsor_quantiles: tuple[float, float] | None = None,
    include_industry: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    style = pd.read_parquet(exposures_path)
    factor_returns, residuals, diagnostics = run_factor_return_regression(
        panel,
        style,
        return_mode=return_mode,
        industry_min_obs=industry_min_obs,
        fit_method=fit_method,
        winsor_quantiles=winsor_quantiles,
        include_industry=include_industry,
    )
    factor_returns.to_csv(output_dir / "factor_returns.csv", index=False)
    residuals.to_parquet(output_dir / "specific_returns.parquet", index=False)
    diagnostics.to_csv(output_dir / "regression_diagnostics.csv", index=False)
    print(f"wrote CNE6-style factor returns to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--exposures", default="outputs/cne6_reproduction/style_exposures.parquet")
    parser.add_argument("--output", default="outputs/cne6_reproduction")
    parser.add_argument("--return-mode", choices=["forward_1d", "same_day", "lagged_exposure_1d"], default="forward_1d")
    parser.add_argument("--industry-min-obs", type=int, default=0)
    parser.add_argument("--fit-method", choices=["wls", "huber"], default="wls")
    parser.add_argument("--winsor-quantiles", default=None, help="optional fit-only return quantiles, e.g. 0.01,0.99")
    parser.add_argument("--no-industry", action="store_true")
    args = parser.parse_args()
    quantiles = None
    if args.winsor_quantiles:
        values = tuple(float(item.strip()) for item in args.winsor_quantiles.split(","))
        if len(values) != 2:
            raise ValueError("--winsor-quantiles expects lower,upper")
        quantiles = values
    run(
        Path(args.panel),
        Path(args.exposures),
        Path(args.output),
        return_mode=args.return_mode,
        industry_min_obs=args.industry_min_obs,
        fit_method=args.fit_method,
        winsor_quantiles=quantiles,
        include_industry=not args.no_industry,
    )


if __name__ == "__main__":
    main()
