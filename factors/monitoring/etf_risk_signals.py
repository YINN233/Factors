"""Risk-appetite aggregates and conservative ETF timing signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import ensure_columns


@dataclass(frozen=True)
class SignalWindows:
    activity_window: int = 5
    share_window: int = 20
    price_window: int = 20
    optimization_window: int = 60


def build_etf_panel(
    etf_daily: pd.DataFrame,
    etf_shares: pd.DataFrame,
    classified_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Combine daily price/volume, shares and reviewed ETF classification."""

    ensure_columns(etf_daily, ["ts_code", "trade_date", "close", "vol", "amount"])
    ensure_columns(etf_shares, ["ts_code", "trade_date", "fd_share"])
    keys = ["ts_code", "trade_date"]
    daily = etf_daily.copy()
    shares = etf_shares.copy()
    for frame in (daily, shares):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    panel = daily.merge(shares[keys + ["fd_share"]], on=keys, how="inner")
    keep = classified_daily[keys + ["category", "risk_bucket", "reviewed"]].drop_duplicates(keys)
    panel = panel.merge(keep, on=keys, how="inner")
    for col in ("close", "vol", "amount", "fd_share"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel[(panel["close"] > 0) & (panel["fd_share"] > 0)].copy()
    # Tushare fund_daily.amount is in CNY thousands and fund_share.fd_share is
    # in ten-thousand shares.  Convert both sides to CNY before calculating a
    # turnover proxy; retaining the unit columns makes this auditable.
    panel["amount_yuan"] = panel["amount"] * 1_000.0
    panel["market_value_proxy"] = panel["close"] * panel["fd_share"] * 10_000.0
    panel["turnover_proxy"] = panel["amount_yuan"] / panel["market_value_proxy"].replace(0, np.nan)
    panel["turnover_proxy"] = panel["turnover_proxy"].replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(keys).reset_index(drop=True)


def aggregate_risk_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ETF activity into low-risk and risk buckets by trading day."""

    ensure_columns(panel, ["trade_date", "risk_bucket", "vol", "amount", "fd_share", "market_value_proxy"])
    rows = panel[panel["risk_bucket"].isin(["low_risk", "risk"])].copy()
    if rows.empty:
        return pd.DataFrame()
    grouped = (
        rows.groupby(["trade_date", "risk_bucket"], as_index=False)
        .agg(
            etf_count=("ts_code", "nunique"),
            volume=("vol", "sum"),
            amount=("amount", "sum"),
            shares=("fd_share", "sum"),
            market_value=("market_value_proxy", "sum"),
        )
    )
    grouped["turnover"] = grouped["amount"] / grouped["market_value"].replace(0, np.nan)
    # ``amount`` is still in CNY thousands at this stage.
    grouped["turnover"] *= 1_000.0
    concentration = (
        rows.groupby(["trade_date", "risk_bucket"], as_index=False)
        .agg(
            max_volume=("vol", "max"),
            max_amount=("amount", "max"),
            max_market_value=("market_value_proxy", "max"),
            total_volume=("vol", "sum"),
            total_amount=("amount", "sum"),
            total_market_value=("market_value_proxy", "sum"),
        )
    )
    concentration["largest_volume_share"] = concentration["max_volume"] / concentration["total_volume"].replace(0, np.nan)
    concentration["largest_amount_share"] = concentration["max_amount"] / concentration["total_amount"].replace(0, np.nan)
    concentration["largest_market_value_share"] = concentration["max_market_value"] / concentration["total_market_value"].replace(0, np.nan)
    concentration = concentration[
        ["trade_date", "risk_bucket", "largest_volume_share", "largest_amount_share", "largest_market_value_share"]
    ]
    grouped = grouped.merge(concentration, on=["trade_date", "risk_bucket"], how="left")
    wide = grouped.pivot(index="trade_date", columns="risk_bucket")
    wide.columns = [f"{bucket}_{metric}" for metric, bucket in wide.columns]
    wide = wide.reset_index()
    expected = [
        f"{bucket}_{metric}"
        for bucket in ("low_risk", "risk")
        for metric in (
            "etf_count", "volume", "amount", "shares", "market_value", "turnover",
            "largest_volume_share", "largest_amount_share", "largest_market_value_share",
        )
    ]
    for col in expected:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide.sort_values("trade_date").reset_index(drop=True)


def _hold_last_signal(values: pd.Series) -> pd.Series:
    """Hold the previous position when two activity signals disagree."""

    result: list[float] = []
    previous = 0.0
    for value in values:
        if pd.isna(value):
            result.append(previous)
        else:
            previous = float(value)
            result.append(previous)
    return pd.Series(result, index=values.index, dtype=float)


def build_risk_signals(
    groups: pd.DataFrame,
    futures_returns: pd.DataFrame | None = None,
    *,
    windows: SignalWindows = SignalWindows(),
) -> pd.DataFrame:
    """Calculate the report-inspired volume/turnover/share/price signals.

    Signals are evidence fields, not orders.  Window values are configurable
    because the source report does not disclose all parameters.
    """

    if groups.empty:
        return groups.copy()
    out = groups.copy().sort_values("trade_date").reset_index(drop=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    activity = max(1, int(windows.activity_window))
    for metric in ("volume", "turnover"):
        for bucket in ("low_risk", "risk"):
            col = f"{bucket}_{metric}"
            rolling = out[col].rolling(activity, min_periods=activity).mean()
            out[f"{col}_activity"] = rolling
            out[f"{col}_change"] = rolling.pct_change(activity)
        out[f"{metric}_spread"] = out[f"low_risk_{metric}_change"] - out[f"risk_{metric}_change"]
        out[f"{metric}_signal"] = np.where(out[f"{metric}_spread"].notna(), np.where(out[f"{metric}_spread"] >= 0, 1.0, -1.0), np.nan)

    for bucket in ("low_risk", "risk"):
        out[f"{bucket}_share_change"] = out[f"{bucket}_shares"].pct_change(max(1, int(windows.share_window)))
    out["share_spread"] = out["low_risk_share_change"] - out["risk_share_change"]
    agreement = np.where(
        (out["volume_signal"] == 1.0) & (out["turnover_signal"] == 1.0),
        1.0,
        np.where((out["volume_signal"] == -1.0) & (out["turnover_signal"] == -1.0), -1.0, np.nan),
    )
    out["activity_consensus_signal"] = _hold_last_signal(pd.Series(agreement, index=out.index))
    out["share_adjusted_signal"] = np.where(
        out["share_spread"].notna() & (out["share_spread"] < 0),
        -out["activity_consensus_signal"],
        out["activity_consensus_signal"],
    )

    if futures_returns is not None and not futures_returns.empty:
        ensure_columns(futures_returns, ["trade_date", "product", "return"])
        fr = futures_returns.copy()
        fr["trade_date"] = pd.to_datetime(fr["trade_date"], errors="coerce")
        pivot = fr.pivot_table(index="trade_date", columns="product", values="return", aggfunc="last")
        for product in pivot.columns:
            out = out.merge(pivot[[product]].rename(columns={product: f"{product}_return"}), left_on="trade_date", right_index=True, how="left")
            out[f"{product}_price_change"] = (1.0 + out[f"{product}_return"]).rolling(max(1, int(windows.price_window))).apply(np.prod, raw=True) - 1.0
            condition = (out["share_spread"] < 0) & (out[f"{product}_price_change"] < 0)
            out[f"{product}_four_factor_signal"] = np.where(condition, -out["activity_consensus_signal"], out["activity_consensus_signal"])
    out["signal_quality"] = np.where(
        out[["volume_spread", "turnover_spread", "share_spread"]].notna().all(axis=1),
        "C",
        "D",
    )
    concentration_columns = [
        column
        for column in ("low_risk_largest_amount_share", "risk_largest_amount_share")
        if column in out.columns
    ]
    out["concentration_warning"] = (
        out[concentration_columns].max(axis=1).gt(0.5) if concentration_columns else False
    )
    out["activity_window"] = activity
    out["share_window"] = max(1, int(windows.share_window))
    out["price_window"] = max(1, int(windows.price_window))
    out["optimization_window"] = max(1, int(windows.optimization_window))
    out["parameters_disclosed"] = False
    out["signal_interpretation"] = "+1=低风险ETF相对占优；-1=风险型ETF相对占优；仅作证据，不是下单指令"
    return out
