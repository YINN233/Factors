"""EWMA/Newey-West covariance with Monte Carlo Eigenfactor adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class FactorBasis:
    display_columns: tuple[str, ...]
    independent_columns: tuple[str, ...]
    forward_matrix: np.ndarray
    restore_matrix: np.ndarray
    reference_industry: str
    industry_weights: tuple[float, ...]


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: np.ndarray
    observations: int
    raw_min_eigenvalue: float
    negative_eigenvalue_mass: float


@dataclass(frozen=True)
class EigenfactorAdjustment:
    matrix: np.ndarray
    original_eigenvalues: np.ndarray
    adjusted_eigenvalues: np.ndarray
    multipliers: np.ndarray
    simulations: int
    clipped_multipliers: int


@dataclass(frozen=True)
class PublishedCovariance:
    matrix: np.ndarray
    columns: tuple[str, ...]
    method: str
    fallback_reason: str
    basis: FactorBasis
    base_estimate: CovarianceEstimate | None
    adjustment: EigenfactorAdjustment | None


def _project_psd(matrix: np.ndarray, tolerance: float = 0.0) -> tuple[np.ndarray, float, float]:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    raw_minimum = float(eigenvalues.min()) if eigenvalues.size else np.nan
    negative_mass = float(-eigenvalues[eigenvalues < 0].sum())
    clipped = np.maximum(eigenvalues, tolerance)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    return (projected + projected.T) / 2.0, raw_minimum, negative_mass


def build_independent_factor_basis(
    factor_returns: pd.DataFrame,
    industry_weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, FactorBasis]:
    """Convert country plus constrained industries to a full-rank contrast basis."""

    if "country" not in factor_returns.columns:
        raise ValueError("factor returns must contain country")
    display_columns = tuple(factor_returns.columns)
    industry_columns = tuple(sorted(column for column in display_columns if column.startswith("industry_")))
    if len(industry_columns) < 2:
        raise ValueError("at least two industry factors are required for an independent basis")
    other_columns = tuple(column for column in display_columns if column != "country" and column not in industry_columns)
    reference = industry_columns[-1]
    non_reference = industry_columns[:-1]

    if industry_weights is None:
        normalized_weights = {column: 1.0 / len(industry_columns) for column in industry_columns}
    else:
        raw_weights = np.array([float(industry_weights.get(column, 0.0)) for column in industry_columns])
        if not np.isfinite(raw_weights).all() or (raw_weights < 0).any() or raw_weights.sum() <= 0:
            raise ValueError("industry weights must be finite, non-negative, and sum to a positive value")
        raw_weights = raw_weights / raw_weights.sum()
        normalized_weights = dict(zip(industry_columns, raw_weights))

    independent_columns = ("country_reference",) + other_columns + tuple(
        f"contrast_{column.removeprefix('industry_')}" for column in non_reference
    )
    display_index = {column: index for index, column in enumerate(display_columns)}
    independent_index = {column: index for index, column in enumerate(independent_columns)}
    forward = np.zeros((len(independent_columns), len(display_columns)), dtype=float)
    forward[independent_index["country_reference"], display_index["country"]] = 1.0
    forward[independent_index["country_reference"], display_index[reference]] = 1.0
    for column in other_columns:
        forward[independent_index[column], display_index[column]] = 1.0
    for industry, contrast in zip(non_reference, independent_columns[1 + len(other_columns):]):
        forward[independent_index[contrast], display_index[industry]] = 1.0
        forward[independent_index[contrast], display_index[reference]] = -1.0

    restore = np.zeros((len(display_columns), len(independent_columns)), dtype=float)
    restore[display_index["country"], independent_index["country_reference"]] = 1.0
    for column in other_columns:
        restore[display_index[column], independent_index[column]] = 1.0
    contrast_columns = independent_columns[1 + len(other_columns):]
    non_reference_weights = np.array([normalized_weights[column] for column in non_reference], dtype=float)
    for contrast, weight in zip(contrast_columns, non_reference_weights):
        restore[display_index["country"], independent_index[contrast]] = weight
        restore[display_index[reference], independent_index[contrast]] = -weight
    for industry, own_contrast in zip(non_reference, contrast_columns):
        for contrast, weight in zip(contrast_columns, non_reference_weights):
            restore[display_index[industry], independent_index[contrast]] = -weight
        restore[display_index[industry], independent_index[own_contrast]] += 1.0

    numeric = factor_returns.loc[:, display_columns].apply(pd.to_numeric, errors="coerce")
    independent_values = numeric.to_numpy(dtype=float) @ forward.T
    independent = pd.DataFrame(independent_values, index=factor_returns.index, columns=independent_columns)
    basis = FactorBasis(
        display_columns=display_columns,
        independent_columns=independent_columns,
        forward_matrix=forward,
        restore_matrix=restore,
        reference_industry=reference,
        industry_weights=tuple(normalized_weights[column] for column in industry_columns),
    )
    return independent, basis


def ewma_newey_west_covariance(
    factor_returns: pd.DataFrame,
    window: int = 504,
    min_periods: int = 252,
    half_life: int = 90,
    newey_west_lags: int = 2,
) -> CovarianceEstimate:
    """Estimate a finite-history EWMA/Newey-West covariance without zero filling."""

    numeric = factor_returns.apply(pd.to_numeric, errors="coerce").tail(window)
    complete = numeric.dropna(axis=0, how="any")
    if len(complete) < min_periods:
        raise ValueError(
            f"insufficient complete factor-return history: {len(complete)} < {min_periods}"
        )
    values = complete.to_numpy(dtype=float)
    ages = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * ages / float(half_life))
    weights = weights / weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = centered.T @ (centered * weights[:, None])
    for lag in range(1, newey_west_lags + 1):
        lag_weights = weights[lag:].copy()
        lag_weights = lag_weights / lag_weights.sum()
        autocovariance = centered[lag:].T @ (centered[:-lag] * lag_weights[:, None])
        bartlett = 1.0 - lag / float(newey_west_lags + 1)
        covariance += bartlett * (autocovariance + autocovariance.T)
    projected, raw_minimum, negative_mass = _project_psd(covariance)
    return CovarianceEstimate(
        matrix=projected,
        observations=len(complete),
        raw_min_eigenvalue=raw_minimum,
        negative_eigenvalue_mass=negative_mass,
    )


def eigenfactor_adjust_covariance(
    base_covariance: np.ndarray,
    history_length: int,
    simulations: int = 500,
    half_life: int = 90,
    newey_west_lags: int = 2,
    random_seed: int = 729,
    multiplier_bounds: tuple[float, float] = (0.5, 2.0),
) -> EigenfactorAdjustment:
    """Correct eigenvalue rank bias using covariance-consistent simulations."""

    base, _, _ = _project_psd(base_covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(base)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(random_seed)
    ratios = []
    minimum = max(20, history_length // 2)
    for _ in range(simulations):
        simulated = rng.normal(size=(history_length, base.shape[0])) @ positive_root.T
        estimate = ewma_newey_west_covariance(
            pd.DataFrame(simulated),
            window=history_length,
            min_periods=minimum,
            half_life=half_life,
            newey_west_lags=newey_west_lags,
        ).matrix
        sample_eigenvalues, sample_eigenvectors = np.linalg.eigh(estimate)
        sample_order = np.argsort(sample_eigenvalues)[::-1]
        sample_eigenvalues = sample_eigenvalues[sample_order]
        sample_eigenvectors = sample_eigenvectors[:, sample_order]
        true_variances = np.sum(sample_eigenvectors * (base @ sample_eigenvectors), axis=0)
        ratio = np.divide(
            true_variances,
            sample_eigenvalues,
            out=np.ones_like(true_variances),
            where=sample_eigenvalues > 1e-20,
        )
        ratios.append(ratio)
    raw_multipliers = np.mean(np.vstack(ratios), axis=0)
    lower, upper = multiplier_bounds
    multipliers = np.clip(raw_multipliers, lower, upper)
    adjusted_eigenvalues = np.maximum(eigenvalues * multipliers, 0.0)
    adjusted = (eigenvectors * adjusted_eigenvalues) @ eigenvectors.T
    adjusted, _, _ = _project_psd(adjusted)
    return EigenfactorAdjustment(
        matrix=adjusted,
        original_eigenvalues=eigenvalues,
        adjusted_eigenvalues=adjusted_eigenvalues,
        multipliers=multipliers,
        simulations=simulations,
        clipped_multipliers=int(np.count_nonzero(np.abs(raw_multipliers - multipliers) > 1e-12)),
    )


def eigenfactor_effective_dates(trading_dates: pd.Series | pd.Index | np.ndarray) -> pd.DataFrame:
    """Map each month-end calibration to the following trading day."""

    dates = pd.Index(pd.to_datetime(trading_dates).dropna().unique()).sort_values()
    rows = []
    for index in range(len(dates) - 1):
        current = pd.Timestamp(dates[index])
        following = pd.Timestamp(dates[index + 1])
        if current.to_period("M") != following.to_period("M"):
            rows.append({"calibration_date": current, "effective_date": following})
    return pd.DataFrame(rows, columns=["calibration_date", "effective_date"])


def restore_display_covariance(independent_covariance: np.ndarray, basis: FactorBasis) -> np.ndarray:
    restored = basis.restore_matrix @ independent_covariance @ basis.restore_matrix.T
    projected, _, _ = _project_psd(restored)
    return projected


def estimate_eigenfactor_covariance(
    factor_history: pd.DataFrame,
    industry_weights: Mapping[str, float] | None = None,
    window: int = 504,
    min_periods: int = 252,
    half_life: int = 90,
    newey_west_lags: int = 2,
    simulations: int = 500,
    random_seed: int = 729,
) -> PublishedCovariance:
    independent, basis = build_independent_factor_basis(factor_history, industry_weights=industry_weights)
    base = ewma_newey_west_covariance(
        independent,
        window=window,
        min_periods=min_periods,
        half_life=half_life,
        newey_west_lags=newey_west_lags,
    )
    adjustment = eigenfactor_adjust_covariance(
        base.matrix,
        history_length=base.observations,
        simulations=simulations,
        half_life=half_life,
        newey_west_lags=newey_west_lags,
        random_seed=random_seed,
    )
    display = restore_display_covariance(adjustment.matrix, basis)
    return PublishedCovariance(
        matrix=display,
        columns=basis.display_columns,
        method="eigenfactor",
        fallback_reason="",
        basis=basis,
        base_estimate=base,
        adjustment=adjustment,
    )


def estimate_covariance_with_fallback(
    factor_history: pd.DataFrame,
    industry_weights: Mapping[str, float] | None = None,
    window: int = 504,
    min_periods: int = 252,
    half_life: int = 90,
    newey_west_lags: int = 2,
    simulations: int = 500,
    random_seed: int = 729,
) -> PublishedCovariance:
    try:
        return estimate_eigenfactor_covariance(
            factor_history,
            industry_weights=industry_weights,
            window=window,
            min_periods=min_periods,
            half_life=half_life,
            newey_west_lags=newey_west_lags,
            simulations=simulations,
            random_seed=random_seed,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        independent, basis = build_independent_factor_basis(factor_history, industry_weights=industry_weights)
        complete = independent.tail(window).dropna(axis=0, how="any")
        if len(complete) < 2:
            raise ValueError(f"Eigenfactor failed and Ledoit-Wolf has insufficient history: {error}") from error
        covariance = LedoitWolf().fit(complete.to_numpy(dtype=float)).covariance_
        display = restore_display_covariance(covariance, basis)
        return PublishedCovariance(
            matrix=display,
            columns=basis.display_columns,
            method="ledoit_wolf_fallback",
            fallback_reason=str(error),
            basis=basis,
            base_estimate=None,
            adjustment=None,
        )


def covariance_to_long(
    covariance: np.ndarray,
    columns: tuple[str, ...] | list[str],
    trade_date: pd.Timestamp,
    covariance_type: str,
) -> pd.DataFrame:
    rows = []
    for i, factor_i in enumerate(columns):
        for j in range(i, len(columns)):
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "factor_i": factor_i,
                    "factor_j": columns[j],
                    "covariance": float(covariance[i, j]),
                    "covariance_type": covariance_type,
                }
            )
    return pd.DataFrame(rows)
