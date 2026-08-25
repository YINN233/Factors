"""Cache-to-derived pipeline used by the CLI and Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import default_raw_dir
from .decision_summary import build_decision_summary
from .dividend_basis import add_historical_percentile, build_basis_table, build_contract_dividend_points
from .etf_risk_signals import SignalWindows, aggregate_risk_groups, build_etf_panel, build_risk_signals
from .etf_universe import apply_classification, load_classification
from .pair_analysis import build_pair_basis_history
from .walk_forward import walk_forward_summary


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _raw_path(raw_dir: Path, stem: str, start_date: str | None = None, end_date: str | None = None) -> Path:
    suffix = f"_{start_date}_{end_date}" if start_date and end_date else ""
    return raw_dir / f"{stem}{suffix}.parquet"


def _processed_dir(raw_dir: Path) -> Path:
    # The repository default ends in ``data/raw/monitoring``.  For a custom
    # raw directory (common in tests and company-DB adapters), keep derived
    # files below that directory rather than accidentally resolving to a
    # filesystem root such as ``/dev/processed``.
    resolved = raw_dir.resolve()
    if resolved.name == "monitoring" and resolved.parent.name == "raw" and resolved.parent.parent.name == "data":
        target = resolved.parent.parent / "processed" / "monitoring"
    else:
        target = resolved / "processed" / "monitoring"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _normalise_dividend_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Make unavailable dividend inputs explicit for every contract row.

    The dividend calculation is intentionally run only for the requested
    as-of date in the first MVP.  Rows from earlier dates therefore have no
    dividend estimate.  A left join leaves those statuses as ``NaN``; keeping
    that value would make the dashboard look like an unlabelled estimate and
    could accidentally promote it in downstream quality filters.
    """

    if frame.empty:
        return frame
    out = frame.copy()
    if "dividend_source" not in out.columns:
        out["dividend_source"] = "unavailable"
    out["dividend_source"] = out["dividend_source"].fillna("unavailable").replace({"": "unavailable"})
    if "expected_dividend_points" not in out.columns:
        out["expected_dividend_points"] = 0.0
    out["expected_dividend_points"] = pd.to_numeric(
        out["expected_dividend_points"], errors="coerce"
    ).fillna(0.0)
    return out


def build_basis_from_cache(
    raw_dir: str | Path | None = None,
    *,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Build basis data from cached raw frames.

    If constituent dividend inputs are absent, the output retains an
    ``unavailable`` dividend source instead of silently treating zero as a
    known dividend estimate.
    """

    root = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    futures = _read(_raw_path(root, "futures_daily", start_date, end_date))
    indices = _read(_raw_path(root, "index_daily", start_date, end_date))
    mapping = _read(_raw_path(root, "futures_mapping", start_date, end_date))
    if futures.empty or indices.empty:
        return pd.DataFrame()
    futures["trade_date"] = pd.to_datetime(futures["trade_date"], errors="coerce")
    indices["trade_date"] = pd.to_datetime(indices["trade_date"], errors="coerce")
    if not mapping.empty:
        mapping["trade_date"] = pd.to_datetime(mapping["trade_date"], errors="coerce")
        futures = futures.merge(
            mapping[["product", "trade_date", "mapping_ts_code"]].drop_duplicates(),
            on=["product", "trade_date"],
            how="left",
        )
        futures["is_main"] = futures["ts_code"].eq(futures["mapping_ts_code"])
    else:
        futures["is_main"] = False

    as_of = pd.Timestamp(as_of_date) if as_of_date else futures["trade_date"].max()
    events = _read(_raw_path(root, "dividend_events_constituents"))
    weights = _read(_raw_path(root, "index_weights", start_date, end_date))
    stock_prices = _read(_raw_path(root, "stock_daily_constituents", start_date, end_date))
    if stock_prices.empty:
        stock_prices = _read(_raw_path(root, "stock_daily_constituents_snapshot"))
    current_futures = futures[futures["trade_date"] == as_of].copy()
    current_indices = indices[indices["trade_date"] == as_of].copy()
    if not events.empty and not weights.empty and not stock_prices.empty and not current_futures.empty:
        dividend_points = build_contract_dividend_points(
            current_futures,
            current_indices,
            events,
            weights,
            stock_prices=stock_prices,
            as_of_date=as_of,
        )
    else:
        dividend_points = pd.DataFrame(
            {
                "product": current_futures.get("product", pd.Series(dtype=str)),
                "ts_code": current_futures.get("ts_code", pd.Series(dtype=str)),
                "trade_date": current_futures.get("trade_date", pd.Series(dtype="datetime64[ns]")),
                "expected_dividend_points": 0.0,
                "dividend_source": "unavailable",
            }
        )
    basis = build_basis_table(futures, indices, dividend_points=dividend_points, as_of_date=end_date)
    if not basis.empty:
        basis = _normalise_dividend_status(basis)
        # Historical dividend forecasts are not available in the MVP.  Keep
        # the percentile useful by falling back to raw basis only on rows
        # whose dividend adjustment is explicitly unavailable, and retain
        # the chosen input so the comparison is auditable in the dashboard.
        has_dividend_input = basis["dividend_source"].ne("unavailable")
        basis["historical_percentile_input"] = basis["annualized_basis"].where(
            has_dividend_input,
            basis["raw_annualized_basis"],
        )
        basis["historical_percentile_basis"] = "dividend_adjusted_annualized_basis"
        basis.loc[~has_dividend_input, "historical_percentile_basis"] = "raw_annualized_basis"
        # Compare like-for-like expiries and only use observations known up
        # to each date.  Ranking all contracts on the full sample would mix
        # tenor effects and leak future basis levels into historical views.
        basis["tenor_rank"] = (
            basis.sort_values(["trade_date", "product", "days_to_expiry", "ts_code"])
            .groupby(["trade_date", "product"], dropna=False)
            .cumcount()
            .add(1)
        )
        basis["raw_basis_quality"] = "D"
        basis.loc[basis["raw_annualized_basis"].notna(), "raw_basis_quality"] = "B"
        raw_ranked = add_historical_percentile(
            basis,
            value_column="raw_annualized_basis",
            group_columns=("product", "tenor_rank"),
            point_in_time=True,
        )
        basis["raw_historical_percentile"] = raw_ranked["historical_percentile"]
        basis = add_historical_percentile(
            basis,
            value_column="historical_percentile_input",
            group_columns=("product", "tenor_rank"),
            point_in_time=True,
        )
        basis["basis_quality"] = basis["dividend_source"].map(
            lambda value: (
                "B"
                if value == "disclosed_events"
                else "C"
                if value in {"partial_disclosed_events", "estimated_events", "proxy"}
                else "D"
            )
        )
    if cache:
        out_path = _processed_dir(root) / f"basis_table_{start_date}_{end_date}.parquet"
        basis.to_parquet(out_path, index=False)
    return basis


def build_etf_signals_from_cache(
    raw_dir: str | Path | None = None,
    *,
    start_date: str,
    end_date: str,
    classification_path: str | Path | None = None,
    windows: SignalWindows = SignalWindows(),
    cost_bps_per_turnover: float = 5.0,
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ETF group aggregates and signals from cached raw data."""

    root = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    daily = _read(_raw_path(root, "etf_daily_universe", start_date, end_date))
    shares = _read(_raw_path(root, "etf_shares_universe", start_date, end_date))
    if classification_path is None:
        classification_path = Path(__file__).with_name("etf_classification.csv")
    classification = load_classification(classification_path)
    if daily.empty or shares.empty or classification.empty:
        return pd.DataFrame(), pd.DataFrame()
    classified = apply_classification(daily, classification)
    panel = build_etf_panel(daily, shares, classified)
    groups = aggregate_risk_groups(panel)

    futures = _read(_raw_path(root, "futures_daily", start_date, end_date))
    mapping = _read(_raw_path(root, "futures_mapping", start_date, end_date))
    futures_returns = pd.DataFrame()
    if not futures.empty and not mapping.empty:
        futures["trade_date"] = pd.to_datetime(futures["trade_date"], errors="coerce")
        mapping["trade_date"] = pd.to_datetime(mapping["trade_date"], errors="coerce")
        mapping_keys = mapping[["product", "trade_date", "mapping_ts_code"]].drop_duplicates(
            ["product", "trade_date"], keep="last"
        )
        futures = futures.merge(mapping_keys, on=["product", "trade_date"], how="left")
        futures = futures[futures["ts_code"].eq(futures["mapping_ts_code"])].copy()
        futures_returns = futures.sort_values(["product", "trade_date"])
        if "pre_close" in futures_returns.columns:
            previous = pd.to_numeric(futures_returns["pre_close"], errors="coerce").replace(0, pd.NA)
            futures_returns["return"] = pd.to_numeric(futures_returns["close"], errors="coerce") / previous - 1.0
        else:
            # Group by the actual contract so a mapping roll never creates a
            # synthetic return from two different contract price levels.
            futures_returns["return"] = futures_returns.groupby("ts_code")["close"].pct_change()
        futures_returns = futures_returns[["trade_date", "product", "return"]].dropna()
    signals = build_risk_signals(groups, futures_returns, windows=windows)
    validation_rows: list[pd.DataFrame] = []
    for product in ("IH", "IF", "IC", "IM"):
        signal_column = f"{product}_four_factor_signal"
        return_column = f"{product}_return"
        if signal_column not in signals.columns or return_column not in signals.columns:
            continue
        summary = walk_forward_summary(
            signals,
            signal_column=signal_column,
            return_column=return_column,
            cost_bps_per_turnover=cost_bps_per_turnover,
        )
        summary.insert(0, "product", product)
        summary.insert(1, "signal", "four_factor")
        validation_rows.append(summary)
    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    if cache:
        target = _processed_dir(root)
        panel.to_parquet(target / f"etf_panel_{start_date}_{end_date}.parquet", index=False)
        groups.to_parquet(target / f"etf_groups_{start_date}_{end_date}.parquet", index=False)
        signals.to_parquet(target / f"etf_signals_{start_date}_{end_date}.parquet", index=False)
        validation.to_parquet(target / f"etf_validation_{start_date}_{end_date}.parquet", index=False)
    return groups, signals


def build_dashboard_derivatives_from_cache(
    raw_dir: str | Path | None = None,
    *,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build decision-summary and directional pair tables from derivatives."""

    root = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    target = _processed_dir(root)
    basis = _read(target / f"basis_table_{start_date}_{end_date}.parquet")
    signals = _read(target / f"etf_signals_{start_date}_{end_date}.parquet")
    summary = build_decision_summary(basis, signals)
    pairs = build_pair_basis_history(basis)
    if cache:
        summary.to_parquet(target / f"decision_summary_{start_date}_{end_date}.parquet", index=False)
        pairs.to_parquet(target / f"pair_basis_{start_date}_{end_date}.parquet", index=False)
    return summary, pairs
