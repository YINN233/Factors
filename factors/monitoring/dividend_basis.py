"""Dividend-point and futures-basis calculations.

The functions here are data-frame based and do not call Tushare.  Keeping the
calculation layer pure makes it possible to test the financial formulas with
fixtures and to swap a company database adapter later.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import INDEX_CODES, ensure_columns


DIVIDEND_DATE_COLUMNS = ["ann_date", "end_date", "record_date", "ex_date", "pay_date"]


def normalize_dividend_events(
    events: pd.DataFrame,
    *,
    cash_div_per_n_shares: float = 10.0,
) -> pd.DataFrame:
    """Normalize Tushare dividend events without changing the raw amount.

    Tushare's ``cash_div`` is conventionally quoted per ten shares.  The
    derived ``cash_div_per_share`` is kept alongside the raw field so the
    conversion remains auditable.
    """

    required = {"ts_code", "ann_date"}
    if events.empty:
        return events.copy()
    ensure_columns(events, required)
    out = events.copy()
    for column in DIVIDEND_DATE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "cash_div" not in out.columns:
        out["cash_div"] = np.nan
    out["cash_div"] = pd.to_numeric(out["cash_div"], errors="coerce")
    if "cash_div_tax" in out.columns:
        out["cash_div_tax"] = pd.to_numeric(out["cash_div_tax"], errors="coerce")
        out["cash_div_used"] = out["cash_div"].where(out["cash_div"].gt(0), out["cash_div_tax"])
    else:
        out["cash_div_used"] = out["cash_div"]
    out["cash_div_per_share"] = out["cash_div_used"] / float(cash_div_per_n_shares)
    out["event_known"] = out["ann_date"].notna()
    out["ex_date_known"] = out.get("ex_date", pd.Series(index=out.index)).notna()
    dedupe_cols = [column for column in ("ts_code", "end_date", "ex_date", "cash_div_per_share") if column in out.columns]
    return out.drop_duplicates(dedupe_cols).reset_index(drop=True) if dedupe_cols else out


def _latest_weights(weights: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Select the latest available weight snapshot at or before a date."""

    if weights.empty:
        return weights.copy()
    w = weights.copy()
    date_col = "weight_date" if "weight_date" in w.columns else "trade_date"
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w = w[w[date_col] <= as_of_date].copy()
    if w.empty:
        return w
    latest = w[date_col].max()
    return w[w[date_col] == latest].copy()


def reweight_index_snapshot(
    weights: pd.DataFrame,
    stock_prices: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Approximate daily index weights from periodic snapshots and prices.

    For each snapshot interval, current constituent weights are adjusted by
    relative price moves and renormalized.  The result is explicitly marked as
    ``monthly_reweighted_proxy``; it is not a substitute for official daily
    weights when constituents change intra-period.
    """

    if weights.empty:
        return pd.DataFrame()
    ensure_columns(weights, ["con_code", "weight"])
    p = stock_prices.copy()
    price_code = "ts_code" if "ts_code" in p.columns else "con_code"
    ensure_columns(p, [price_code, "trade_date", "close"])
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.rename(columns={price_code: "con_code"})
    p = p.dropna(subset=["con_code", "trade_date", "close"])
    p = p[p["close"] > 0]

    w = weights.copy()
    date_col = "weight_date" if "weight_date" in w.columns else "trade_date"
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    w = w.dropna(subset=["con_code", date_col, "weight"])
    w = w.sort_values([date_col, "con_code"])
    snapshots = list(w.groupby(date_col, sort=True))
    calendar = pd.date_range(start_date, end_date, freq="B")
    records: list[pd.DataFrame] = []

    for i, (snapshot_date, snapshot) in enumerate(snapshots):
        interval_start = max(pd.Timestamp(snapshot_date), calendar.min())
        next_date = snapshots[i + 1][0] if i + 1 < len(snapshots) else calendar.max() + pd.Timedelta(days=1)
        interval_end = min(pd.Timestamp(next_date) - pd.Timedelta(days=1), calendar.max())
        if interval_start > interval_end:
            continue
        dates = calendar[(calendar >= interval_start) & (calendar <= interval_end)]
        if len(dates) == 0:
            continue
        codes = snapshot["con_code"].astype(str).unique().tolist()
        prices = p[p["con_code"].astype(str).isin(codes)]
        pivot = prices.pivot_table(index="trade_date", columns="con_code", values="close", aggfunc="last")
        pivot = pivot.reindex(dates).ffill()
        base = pivot.iloc[0]
        base = base.replace(0, np.nan)
        base_weight = snapshot.set_index("con_code")["weight"]
        for date in dates:
            price_row = pivot.loc[date]
            relative = price_row / base
            adjusted = base_weight.mul(relative, fill_value=np.nan).dropna()
            if adjusted.empty or adjusted.sum() <= 0:
                current = base_weight.copy()
            else:
                current = adjusted / adjusted.sum() * 100.0
            records.append(
                pd.DataFrame(
                    {
                        "con_code": current.index.astype(str),
                        "trade_date": date,
                        "weight": current.values,
                        "weight_method": "monthly_reweighted_proxy",
                    }
                )
            )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _event_price_map(
    events: pd.DataFrame,
    stock_prices: pd.DataFrame | None,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Attach a conservative price used to turn cash dividends into yields."""

    out = events.copy()
    if "stock_close" in out.columns:
        out["stock_close"] = pd.to_numeric(out["stock_close"], errors="coerce")
        return out
    if stock_prices is None or stock_prices.empty:
        out["stock_close"] = np.nan
        return out
    p = stock_prices.copy()
    ensure_columns(p, ["ts_code", "trade_date", "close"])
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p[p["trade_date"] <= as_of_date].dropna(subset=["ts_code", "trade_date", "close"])
    p = p.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
    return out.merge(p[["ts_code", "close"]].rename(columns={"close": "stock_close"}), on="ts_code", how="left")


def dividend_points_for_expiry(
    events: pd.DataFrame,
    weights: pd.DataFrame,
    index_close: float,
    *,
    as_of_date: str | pd.Timestamp,
    expiry_date: str | pd.Timestamp,
    stock_prices: pd.DataFrame | None = None,
    product: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """Estimate index dividend points known at ``as_of_date`` until expiry.

    Events without a known ex-date or usable stock price are returned in the
    detail frame with ``included=False`` rather than silently contributing
    zero.  This keeps the dashboard honest about the estimate's coverage.
    """

    as_of = pd.Timestamp(as_of_date)
    expiry = pd.Timestamp(expiry_date)
    if pd.isna(as_of) or pd.isna(expiry) or expiry <= as_of:
        return 0.0, pd.DataFrame()
    normalized = normalize_dividend_events(events)
    if normalized.empty:
        return 0.0, pd.DataFrame()
    normalized = normalized[normalized["ann_date"].notna() & (normalized["ann_date"] <= as_of)].copy()
    if "ex_date" not in normalized.columns:
        normalized["ex_date"] = pd.NaT
    normalized["included"] = normalized["ex_date"].between(as_of, expiry, inclusive="right")
    normalized = _event_price_map(normalized, stock_prices, as_of_date=as_of)
    normalized["dividend_yield"] = normalized["cash_div_per_share"] / normalized["stock_close"]
    normalized["included"] &= normalized["stock_close"].gt(0)
    w = weights.copy()
    if not w.empty:
        w["trade_date"] = pd.to_datetime(w["trade_date"], errors="coerce") if "trade_date" in w.columns else pd.NaT
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
        # Use the latest available snapshot at the as-of date for all known future events.
        latest = _latest_weights(w, as_of)
        if not latest.empty:
            normalized = normalized.merge(
                latest[["con_code", "weight"]],
                left_on="ts_code",
                right_on="con_code",
                how="left",
            )
        else:
            normalized["weight"] = np.nan
    else:
        normalized["weight"] = np.nan
    normalized["weight_fraction"] = normalized["weight"] / 100.0
    # An announced event without a usable price, positive amount, or matching
    # index weight is not an included dividend estimate.  This prevents
    # ``sum(skipna=True)`` from silently turning an unknown contribution into
    # zero while still retaining the row for audit/debugging.
    normalized["included"] &= normalized["cash_div_per_share"].gt(0)
    normalized["included"] &= normalized["weight"].gt(0)
    normalized["point_contribution"] = (
        normalized["dividend_yield"] * normalized["weight_fraction"] * float(index_close)
    )
    normalized.loc[~normalized["included"], "point_contribution"] = 0.0
    if product is not None:
        normalized["product"] = product
    total = float(normalized.loc[normalized["included"], "point_contribution"].sum())
    return total, normalized


def build_basis_table(
    futures: pd.DataFrame,
    index_daily: pd.DataFrame,
    *,
    dividend_points: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Merge futures and index quotes and calculate raw/adjusted annualized basis."""

    if futures.empty or index_daily.empty:
        return pd.DataFrame()
    f = futures.copy()
    i = index_daily.copy()
    ensure_columns(f, ["product", "ts_code", "trade_date", "close"])
    ensure_columns(i, ["product", "trade_date", "close"])
    f["trade_date"] = pd.to_datetime(f["trade_date"], errors="coerce")
    i["trade_date"] = pd.to_datetime(i["trade_date"], errors="coerce")
    f["close"] = pd.to_numeric(f["close"], errors="coerce")
    i["close"] = pd.to_numeric(i["close"], errors="coerce")
    if "expiry_date" not in f.columns:
        f["expiry_date"] = pd.NaT
    f["expiry_date"] = pd.to_datetime(f["expiry_date"], errors="coerce")
    i = i.rename(columns={"close": "index_close"})
    keep_i = i[["product", "trade_date", "index_close"]].drop_duplicates()
    out = f.merge(keep_i, on=["product", "trade_date"], how="left")
    out["days_to_expiry"] = (out["expiry_date"] - out["trade_date"]).dt.days
    out["raw_basis"] = out["close"] - out["index_close"]
    if dividend_points is not None and not dividend_points.empty:
        dp = dividend_points.copy()
        dp["trade_date"] = pd.to_datetime(dp["trade_date"], errors="coerce")
        if "dividend_source" not in dp.columns:
            dp["dividend_source"] = "unknown"
        out = out.merge(
            dp[["product", "ts_code", "trade_date", "expected_dividend_points", "dividend_source"]],
            on=["product", "ts_code", "trade_date"],
            how="left",
        )
    if "expected_dividend_points" not in out.columns:
        out["expected_dividend_points"] = 0.0
        out["dividend_source"] = "unavailable"
    else:
        # A partial as-of dividend table is expected in the MVP.  Make the
        # missing side of the left join explicit instead of leaving NaN to be
        # interpreted as a known zero by a caller.
        out["dividend_source"] = out["dividend_source"].fillna("unavailable")
    out["expected_dividend_points"] = pd.to_numeric(out["expected_dividend_points"], errors="coerce").fillna(0.0)
    out["dividend_adjusted_basis"] = out["raw_basis"] + out["expected_dividend_points"]
    denominator = out["index_close"].replace(0, np.nan)
    days = out["days_to_expiry"].where(out["days_to_expiry"].gt(0))
    out["raw_annualized_basis"] = out["raw_basis"] / denominator * 365.0 / days
    out["annualized_basis"] = out["dividend_adjusted_basis"] / denominator * 365.0 / days
    if as_of_date is not None:
        out = out[out["trade_date"] <= pd.Timestamp(as_of_date)].copy()
    return out.sort_values(["trade_date", "product", "expiry_date", "ts_code"]).reset_index(drop=True)


def build_contract_dividend_points(
    futures: pd.DataFrame,
    index_daily: pd.DataFrame,
    events: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    stock_prices: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build known dividend points for each contract/date row.

    This is intentionally usable for a single as-of date in the dashboard.
    A full historical run can be expensive because each contract/date pair
    needs an event-time weight and dividend-price lookup.
    """

    if futures.empty or index_daily.empty or events.empty or weights.empty:
        return pd.DataFrame(
            columns=["product", "ts_code", "trade_date", "expected_dividend_points", "dividend_source"]
        )
    f = futures.copy()
    idx = index_daily.copy()
    f["trade_date"] = pd.to_datetime(f["trade_date"], errors="coerce")
    f["expiry_date"] = pd.to_datetime(f["expiry_date"], errors="coerce")
    idx["trade_date"] = pd.to_datetime(idx["trade_date"], errors="coerce")
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        f = f[f["trade_date"] == as_of].copy()
        idx = idx[idx["trade_date"] == as_of].copy()
    rows: list[dict] = []
    for row in f[["product", "ts_code", "trade_date", "expiry_date"]].drop_duplicates().itertuples(index=False):
        match = idx[(idx["product"] == row.product) & (idx["trade_date"] == row.trade_date)]
        if match.empty or pd.isna(row.expiry_date):
            continue
        index_close = float(match.iloc[0]["close"])
        weight_subset = weights.copy()
        if "product" in weight_subset.columns:
            weight_subset = weight_subset[weight_subset["product"] == row.product]
        total, detail = dividend_points_for_expiry(
            events,
            weight_subset,
            index_close,
            as_of_date=row.trade_date,
            expiry_date=row.expiry_date,
            stock_prices=stock_prices,
            product=row.product,
        )
        included = int(detail["included"].sum()) if not detail.empty and "included" in detail.columns else 0
        announced = int(detail["event_known"].sum()) if not detail.empty and "event_known" in detail.columns else 0
        # ``disclosed_events`` means every announced, in-window event that
        # can be valued with the supplied price and weight inputs.  If some
        # rows are not usable, retain a partial status rather than presenting
        # the estimate as complete.
        usable_announced = int(
            (
                detail.get("event_known", pd.Series(dtype=bool)).fillna(False)
                & detail.get("ex_date_known", pd.Series(dtype=bool)).fillna(False)
                & detail.get("stock_close", pd.Series(dtype=float)).gt(0).fillna(False)
                & detail.get("weight", pd.Series(dtype=float)).gt(0).fillna(False)
            ).sum()
        ) if not detail.empty else 0
        if usable_announced == 0:
            source = "unavailable"
        elif usable_announced == announced:
            source = "disclosed_events"
        else:
            source = "partial_disclosed_events"
        rows.append(
            {
                "product": row.product,
                "ts_code": row.ts_code,
                "trade_date": row.trade_date,
                "expected_dividend_points": total,
                "dividend_source": source,
                "included_dividend_events": included,
                "known_dividend_events": announced,
                "usable_dividend_events": usable_announced,
            }
        )
    return pd.DataFrame(rows)


def add_historical_percentile(
    basis: pd.DataFrame,
    *,
    value_column: str = "annualized_basis",
    group_columns: Iterable[str] = ("product",),
    point_in_time: bool = False,
    date_column: str = "trade_date",
) -> pd.DataFrame:
    """Add a historical percentile (0-100), optionally without look-ahead."""

    if basis.empty:
        return basis.copy()
    out = basis.copy()
    groups = list(group_columns)
    if not point_in_time:
        out["historical_percentile"] = out.groupby(groups, dropna=False)[value_column].rank(pct=True) * 100.0
        return out
    ensure_columns(out, [date_column, value_column, *groups])
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    out["historical_percentile"] = np.nan
    for _, group in out.groupby(groups, dropna=False, sort=False):
        ordered = group.sort_values(date_column, kind="stable")
        observed: list[float] = []
        for index, raw_value in ordered[value_column].items():
            value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            left = bisect_left(observed, float(value))
            right = bisect_right(observed, float(value))
            # Average-rank percentile, including the current observation.
            percentile = ((left + 1 + right + 1) / 2.0) / (len(observed) + 1) * 100.0
            out.at[index, "historical_percentile"] = percentile
            insort(observed, float(value))
    return out


def latest_term_structure(basis: pd.DataFrame, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the latest available contract row for each product."""

    if basis.empty:
        return basis.copy()
    out = basis.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    date = pd.Timestamp(as_of_date) if as_of_date is not None else out["trade_date"].max()
    out = out[out["trade_date"] == date].copy()
    return out.sort_values(["product", "expiry_date", "ts_code"]).reset_index(drop=True)
