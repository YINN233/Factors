import numpy as np
import pandas as pd

from factors.risk.eigenfactor_covariance import (
    build_independent_factor_basis,
    eigenfactor_adjust_covariance,
    eigenfactor_effective_dates,
    estimate_covariance_with_fallback,
    estimate_eigenfactor_covariance,
    ewma_newey_west_covariance,
)


def test_independent_basis_round_trip_preserves_constrained_factor_returns():
    factor_returns = pd.DataFrame(
        {
            "country": [0.01, 0.02],
            "style_size": [0.003, -0.002],
            "industry_A": [0.006, 0.003],
            "industry_B": [-0.002, -0.001],
            "industry_C": [-0.004, -0.002],
        }
    )

    independent, basis = build_independent_factor_basis(factor_returns)
    restored = independent.to_numpy() @ basis.restore_matrix.T

    assert basis.display_columns == tuple(factor_returns.columns)
    assert independent.shape[1] == factor_returns.shape[1] - 1
    assert np.allclose(restored, factor_returns.to_numpy(), atol=1e-12)
    assert np.allclose(basis.forward_matrix @ basis.restore_matrix, np.eye(independent.shape[1]))


def test_ewma_newey_west_covariance_is_symmetric_psd():
    rng = np.random.default_rng(11)
    innovations = rng.normal(0.0, 0.01, size=(320, 4))
    returns = np.zeros_like(innovations)
    for index in range(1, len(returns)):
        returns[index] = 0.35 * returns[index - 1] + innovations[index]
    frame = pd.DataFrame(returns, columns=list("ABCD"))

    estimate = ewma_newey_west_covariance(
        frame,
        window=300,
        min_periods=252,
        half_life=90,
        newey_west_lags=2,
    )

    assert estimate.observations == 300
    assert np.allclose(estimate.matrix, estimate.matrix.T, atol=1e-14)
    assert np.linalg.eigvalsh(estimate.matrix).min() >= -1e-12
    assert estimate.raw_min_eigenvalue <= np.linalg.eigvalsh(estimate.matrix).min() + 1e-12


def test_eigenfactor_adjustment_is_deterministic_bounded_and_psd():
    base = np.array(
        [
            [0.0004, 0.00010, 0.00005],
            [0.0001, 0.00025, 0.00004],
            [0.00005, 0.00004, 0.00010],
        ]
    )

    first = eigenfactor_adjust_covariance(
        base,
        history_length=100,
        simulations=30,
        half_life=40,
        newey_west_lags=1,
        random_seed=729,
    )
    second = eigenfactor_adjust_covariance(
        base,
        history_length=100,
        simulations=30,
        half_life=40,
        newey_west_lags=1,
        random_seed=729,
    )

    assert np.allclose(first.matrix, second.matrix)
    assert np.allclose(first.multipliers, second.multipliers)
    assert ((first.multipliers >= 0.5) & (first.multipliers <= 2.0)).all()
    assert np.linalg.eigvalsh(first.matrix).min() >= -1e-12
    assert not np.allclose(first.matrix, base)


def test_month_end_eigenfactor_adjustment_becomes_effective_next_trading_day():
    dates = pd.to_datetime(
        ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"]
    )

    mapping = eigenfactor_effective_dates(dates)

    assert mapping.iloc[0]["calibration_date"] == pd.Timestamp("2024-01-31")
    assert mapping.iloc[0]["effective_date"] == pd.Timestamp("2024-02-01")
    assert mapping.iloc[1]["calibration_date"] == pd.Timestamp("2024-02-29")
    assert mapping.iloc[1]["effective_date"] == pd.Timestamp("2024-03-01")


def _factor_history(rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    industry_a = rng.normal(0.0, 0.01, rows)
    industry_b = rng.normal(0.0, 0.01, rows)
    industry_c = -(industry_a + industry_b)
    return pd.DataFrame(
        {
            "country": rng.normal(0.0, 0.01, rows),
            "style_size": rng.normal(0.0, 0.005, rows),
            "industry_A": industry_a,
            "industry_B": industry_b,
            "industry_C": industry_c,
        }
    )


def test_production_estimator_restores_psd_display_covariance():
    result = estimate_eigenfactor_covariance(
        _factor_history(),
        window=260,
        min_periods=252,
        simulations=20,
        random_seed=17,
    )

    assert result.method == "eigenfactor"
    assert result.fallback_reason == ""
    assert result.matrix.shape == (5, 5)
    assert np.linalg.eigvalsh(result.matrix).min() >= -1e-12
    assert len(result.adjustment.multipliers) == 4


def test_production_estimator_records_ledoit_wolf_fallback_reason():
    result = estimate_covariance_with_fallback(
        _factor_history(rows=40),
        window=40,
        min_periods=100,
        simulations=5,
    )

    assert result.method == "ledoit_wolf_fallback"
    assert "insufficient complete factor-return history" in result.fallback_reason
    assert np.linalg.eigvalsh(result.matrix).min() >= -1e-12
