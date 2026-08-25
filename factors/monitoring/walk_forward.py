"""Small walk-forward evaluation helpers for monitoring signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _performance_summary(strategy_returns: pd.Series, periods_per_year: int = 252) -> dict:
    values = strategy_returns.dropna()
    if values.empty:
        return {"observations": 0, "annual_return": np.nan, "annual_volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    wealth = (1.0 + values).cumprod()
    annual_return = wealth.iloc[-1] ** (periods_per_year / len(values)) - 1.0
    annual_volatility = values.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "observations": int(len(values)),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def walk_forward_summary(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    return_column: str,
    date_column: str = "trade_date",
    development_end: str = "2022-12-31",
    validation_end: str = "2024-12-31",
    cost_bps_per_turnover: float = 0.0,
) -> pd.DataFrame:
    """Evaluate a lagged daily signal in development/validation/holdout splits.

    The signal is lagged one row before multiplying the market return, which
    prevents same-close look-ahead.  Costs are charged only when position
    changes and are deliberately explicit in the output.
    """

    required = {date_column, signal_column, return_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {', '.join(sorted(missing))}")
    data = frame[[date_column, signal_column, return_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    data["position"] = pd.to_numeric(data[signal_column], errors="coerce").shift(1)
    data["market_return"] = pd.to_numeric(data[return_column], errors="coerce")
    data["turnover"] = data["position"].diff().abs().fillna(0.0)
    data["cost"] = data["turnover"] * float(cost_bps_per_turnover) / 10_000.0
    data["strategy_return"] = data["position"] * data["market_return"] - data["cost"]
    development_end = pd.Timestamp(development_end)
    validation_end = pd.Timestamp(validation_end)
    splits = {
        "development": data[data[date_column] <= development_end],
        "validation": data[(data[date_column] > development_end) & (data[date_column] <= validation_end)],
        "holdout": data[data[date_column] > validation_end],
    }
    rows = []
    for split, values in splits.items():
        row = {"split": split, **_performance_summary(values["strategy_return"])}
        row["turnover"] = float(values["turnover"].mean()) if not values.empty else np.nan
        row["cost_bps_per_turnover"] = float(cost_bps_per_turnover)
        rows.append(row)
    return pd.DataFrame(rows)
