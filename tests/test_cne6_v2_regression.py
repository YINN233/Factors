import numpy as np
import pandas as pd
import pytest

from factors.risk.cne6_regression import (
    constrained_regression_work,
    run_constrained_factor_return_regression,
)


def _known_factor_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    industries = ["801010.SI", "801020.SI", "801030.SI"]
    industry_returns = {"801010.SI": 0.006, "801020.SI": -0.002, "801030.SI": -0.004}
    panel_rows = []
    style_rows = []
    for industry_idx, industry in enumerate(industries):
        for stock_idx in range(30):
            code = f"{industry_idx}{stock_idx:05d}.SZ"
            base_style = (stock_idx - 14.5) / 10.0
            for date_idx, date in enumerate(dates):
                style = base_style + date_idx * 0.10
                style_rows.append({"trade_date": date, "ts_code": code, "style_size": style})
                if date_idx == 0:
                    stock_return = np.nan
                else:
                    lagged_style = base_style + (date_idx - 1) * 0.10
                    stock_return = 0.01 + 0.02 * lagged_style + industry_returns[industry]
                panel_rows.append(
                    {
                        "trade_date": date,
                        "ts_code": code,
                        "returns_1d": stock_return,
                        "industry_sw_l1_code": industry,
                        "total_mv": 100.0,
                        "csi500_member": True,
                    }
                )
    return pd.DataFrame(panel_rows), pd.DataFrame(style_rows)


def test_constrained_regression_recovers_known_factor_returns():
    panel, styles = _known_factor_panel()

    factor_returns, residuals, diagnostics = run_constrained_factor_return_regression(panel, styles)

    successful = diagnostics[diagnostics["regression_status"] == "ok"]
    assert len(successful) == 4
    row = factor_returns.set_index("trade_date").loc[pd.Timestamp("2024-01-03")]
    assert np.isclose(row["country"], 0.01, atol=1e-10)
    assert np.isclose(row["style_size"], 0.02, atol=1e-10)
    assert np.isclose(row["industry_801010.SI"], 0.006, atol=1e-10)
    assert np.isclose(row["industry_801020.SI"], -0.002, atol=1e-10)
    assert np.isclose(row["industry_801030.SI"], -0.004, atol=1e-10)
    assert residuals["specific_return"].abs().max() < 1e-10
    assert successful["industry_constraint_residual"].abs().max() < 1e-12


def test_constrained_regression_is_invariant_to_input_order():
    panel, styles = _known_factor_panel()
    baseline, _, _ = run_constrained_factor_return_regression(panel, styles)

    shuffled, _, _ = run_constrained_factor_return_regression(
        panel.sample(frac=1.0, random_state=7),
        styles.sample(frac=1.0, random_state=9),
    )

    columns = sorted(set(baseline.columns).intersection(shuffled.columns).difference({"trade_date"}))
    left = baseline.sort_values("trade_date")[columns].to_numpy()
    right = shuffled.sort_values("trade_date")[columns].to_numpy()
    assert np.allclose(left, right, equal_nan=True)


def test_constrained_work_uses_t_minus_one_exposures():
    panel, styles = _known_factor_panel()

    work, style_columns = constrained_regression_work(panel, styles)

    assert style_columns == ["style_size"]
    stock = work[work["ts_code"] == "000000.SZ"].set_index("trade_date")
    assert np.isnan(stock.loc[pd.Timestamp("2024-01-02"), "style_size"])
    assert stock.loc[pd.Timestamp("2024-01-03"), "style_size"] == styles.loc[
        (styles["ts_code"] == "000000.SZ") & (styles["trade_date"] == pd.Timestamp("2024-01-02")),
        "style_size",
    ].iloc[0]


def test_constrained_regression_rejects_unknown_industry_rows():
    panel, styles = _known_factor_panel()
    panel["industry_sw_l1_code"] = pd.NA

    with pytest.raises(ValueError, match="no valid constrained regressions"):
        run_constrained_factor_return_regression(panel, styles, fail_if_all_invalid=True)
