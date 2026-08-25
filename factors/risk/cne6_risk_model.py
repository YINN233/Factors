"""Rolling covariance and specific-risk estimation for CNE6-style factor returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.covariance import LedoitWolf


def _full_covariance_cols(factor_cols: list[str]) -> list[str]:
    return [col for col in factor_cols if col == "country" or col.startswith("style_")]


def _parse_windows(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _prepare_factor_returns(factor_returns: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = factor_returns.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.sort_values("trade_date").reset_index(drop=True)
    factor_cols = [c for c in work.columns if c != "trade_date"]
    for col in factor_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work, factor_cols


def _estimate_covariance(hist: pd.DataFrame, method: str) -> tuple[pd.DataFrame, float]:
    if method == "ledoit_wolf":
        # Missing factor returns mostly come from omitted base industries or
        # temporarily unavailable factor returns.  For covariance estimation we
        # treat those as neutral common-factor returns instead of dropping the
        # whole date from a high-dimensional window.
        matrix = hist.fillna(0.0).to_numpy(dtype=float)
        if matrix.shape[0] < 2 or matrix.shape[1] == 0:
            return pd.DataFrame(index=hist.columns, columns=hist.columns, dtype=float), np.nan
        estimator = LedoitWolf().fit(matrix)
        cov = pd.DataFrame(estimator.covariance_, index=hist.columns, columns=hist.columns)
        return cov, float(estimator.shrinkage_)
    return hist.cov(), np.nan


def _covariance_records(
    date: pd.Timestamp,
    window: int,
    cov: pd.DataFrame,
    valid_cols: list[str],
    full_cols: list[str],
    method: str,
    shrinkage: float,
) -> list[dict]:
    rows = []
    full_set = set(full_cols)
    for factor in valid_cols:
        value = cov.loc[factor, factor]
        if pd.notna(value):
            rows.append(
                {
                    "trade_date": date,
                    "window": window,
                    "factor_i": factor,
                    "factor_j": factor,
                    "covariance": float(value),
                    "covariance_type": "ledoit_wolf_full" if method == "ledoit_wolf" else ("sample_full" if factor in full_set else "sample_diagonal_only"),
                    "covariance_method": method,
                    "shrinkage": shrinkage,
                }
            )
    for pos_i, factor_i in enumerate(full_cols):
        for factor_j in full_cols[pos_i + 1:]:
            value = cov.loc[factor_i, factor_j]
            if pd.notna(value):
                rows.append(
                    {
                        "trade_date": date,
                        "window": window,
                        "factor_i": factor_i,
                        "factor_j": factor_j,
                        "covariance": float(value),
                        "covariance_type": "ledoit_wolf_full" if method == "ledoit_wolf" else "sample_full",
                        "covariance_method": method,
                        "shrinkage": shrinkage,
                    }
                )
    return rows


def iter_rolling_factor_covariance_batches(
    factor_returns: pd.DataFrame,
    windows: tuple[int, ...] = (60, 120, 252),
    full_all: bool = False,
    covariance_method: str = "ledoit_wolf",
    lw_full_windows: tuple[int, ...] = (252,),
    batch_rows: int = 250_000,
):
    work, factor_cols = _prepare_factor_returns(factor_returns)
    full_cols_default = _full_covariance_cols(factor_cols)
    rows: list[dict] = []
    for window in windows:
        min_periods = max(20, window // 2)
        use_lw_full = covariance_method == "ledoit_wolf" and (full_all or window in lw_full_windows)
        method = "ledoit_wolf" if use_lw_full else "sample"
        for idx in range(len(work)):
            if idx + 1 < min_periods:
                continue
            hist = work.iloc[max(0, idx + 1 - window): idx + 1][factor_cols]
            valid_cols = [col for col in factor_cols if hist[col].notna().sum() >= min_periods]
            if not valid_cols:
                continue
            hist = hist[valid_cols]
            cov, shrinkage = _estimate_covariance(hist, method)
            date = work.loc[idx, "trade_date"]
            if full_all or use_lw_full:
                full_cols = valid_cols
            else:
                full_cols = [col for col in full_cols_default if col in valid_cols]
            rows.extend(_covariance_records(date, window, cov, valid_cols, full_cols, method, shrinkage))
            if len(rows) >= batch_rows:
                yield pd.DataFrame.from_records(rows)
                rows = []
    if rows:
        yield pd.DataFrame.from_records(rows)


def rolling_factor_covariance(
    factor_returns: pd.DataFrame,
    windows: tuple[int, ...] = (60, 120, 252),
    full_all: bool = False,
    covariance_method: str = "ledoit_wolf",
    lw_full_windows: tuple[int, ...] = (252,),
) -> pd.DataFrame:
    """Return rolling covariance in a compact long format.

    The production default uses a 252-day Ledoit-Wolf full covariance matrix
    for all common factors and keeps shorter windows compact. Off-diagonal
    entries are stored once; consumers should symmetrize when building a dense
    matrix.
    """
    batches = list(
        iter_rolling_factor_covariance_batches(
            factor_returns,
            windows=windows,
            full_all=full_all,
            covariance_method=covariance_method,
            lw_full_windows=lw_full_windows,
        )
    )
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def rolling_specific_risk(specific_returns: pd.DataFrame, windows: tuple[int, ...] = (60, 120, 252)) -> pd.DataFrame:
    work = specific_returns.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    out = work[["trade_date", "ts_code"]].copy()
    values = pd.to_numeric(work["specific_return"], errors="coerce")
    for window in windows:
        min_periods = max(20, window // 2)
        out[f"specific_risk_{window}"] = values.groupby(work["ts_code"], sort=False).transform(
            lambda s: s.rolling(window, min_periods=min_periods).std()
        )
    return out


def covariance_diagnostics(covariance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, sub in covariance.groupby("window", sort=True):
        diag = sub[sub["factor_i"] == sub["factor_j"]]
        methods = ",".join(sorted(sub.get("covariance_method", pd.Series(dtype=object)).dropna().astype(str).unique()))
        types = ",".join(sorted(sub.get("covariance_type", pd.Series(dtype=object)).dropna().astype(str).unique()))
        rows.append(
            {
                "metric": "factor_covariance",
                "window": int(window),
                "dates": int(sub["trade_date"].nunique()),
                "factors": int(pd.Index(sub["factor_i"]).union(sub["factor_j"]).nunique()),
                "negative_variance_rows": int((diag["covariance"] < -1e-12).sum()),
                "rows": int(len(sub)),
                "covariance_method": methods,
                "covariance_type": types,
                "mean_shrinkage": round(float(pd.to_numeric(sub.get("shrinkage", pd.Series(dtype=float)), errors="coerce").mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def specific_risk_diagnostics(specific_risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in specific_risk.columns if c.startswith("specific_risk_")]:
        rows.append(
            {
                "metric": "specific_risk",
                "window": int(col.rsplit("_", 1)[1]),
                "dates": int(specific_risk.loc[specific_risk[col].notna(), "trade_date"].nunique()),
                "factors": np.nan,
                "negative_variance_rows": int((specific_risk[col] < -1e-12).sum()),
                "rows": int(specific_risk[col].notna().sum()),
                "covariance_method": "",
                "covariance_type": "",
                "mean_shrinkage": np.nan,
            }
        )
    return pd.DataFrame(rows)


def risk_model_diagnostics(covariance: pd.DataFrame, specific_risk: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([covariance_diagnostics(covariance), specific_risk_diagnostics(specific_risk)], ignore_index=True)


def _write_covariance_parquet(covariance_batches, output_path: Path) -> pd.DataFrame:
    writer = None
    stats: dict[int, dict] = {}
    try:
        for batch in covariance_batches:
            if batch.empty:
                continue
            for window, sub in batch.groupby("window", sort=False):
                window = int(window)
                stat = stats.setdefault(
                    window,
                    {
                        "dates": set(),
                        "factors": set(),
                        "negative_variance_rows": 0,
                        "rows": 0,
                        "covariance_method": set(),
                        "covariance_type": set(),
                        "shrinkage_sum": 0.0,
                        "shrinkage_count": 0,
                    },
                )
                stat["dates"].update(pd.to_datetime(sub["trade_date"]).dropna().unique())
                stat["factors"].update(sub["factor_i"].dropna().astype(str).unique())
                stat["factors"].update(sub["factor_j"].dropna().astype(str).unique())
                diag = sub[sub["factor_i"] == sub["factor_j"]]
                stat["negative_variance_rows"] += int((pd.to_numeric(diag["covariance"], errors="coerce") < -1e-12).sum())
                stat["rows"] += int(len(sub))
                if "covariance_method" in sub.columns:
                    stat["covariance_method"].update(sub["covariance_method"].dropna().astype(str).unique())
                if "covariance_type" in sub.columns:
                    stat["covariance_type"].update(sub["covariance_type"].dropna().astype(str).unique())
                if "shrinkage" in sub.columns:
                    shrink = pd.to_numeric(sub["shrinkage"], errors="coerce").dropna()
                    stat["shrinkage_sum"] += float(shrink.sum())
                    stat["shrinkage_count"] += int(len(shrink))
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame().to_parquet(output_path, index=False)
        return pd.DataFrame()
    rows = []
    for window, stat in sorted(stats.items()):
        rows.append(
            {
                "metric": "factor_covariance",
                "window": int(window),
                "dates": int(len(stat["dates"])),
                "factors": int(len(stat["factors"])),
                "negative_variance_rows": int(stat["negative_variance_rows"]),
                "rows": int(stat["rows"]),
                "covariance_method": ",".join(sorted(stat["covariance_method"])),
                "covariance_type": ",".join(sorted(stat["covariance_type"])),
                "mean_shrinkage": round(stat["shrinkage_sum"] / stat["shrinkage_count"], 6) if stat["shrinkage_count"] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    factor_returns_path: Path,
    specific_returns_path: Path,
    output_dir: Path,
    full_all: bool = False,
    covariance_method: str = "ledoit_wolf",
    lw_full_windows: tuple[int, ...] = (252,),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_returns = pd.read_csv(factor_returns_path, parse_dates=["trade_date"])
    specific_returns = pd.read_parquet(specific_returns_path)
    covariance_path = output_dir / "factor_covariance_rolling.parquet"
    covariance_batches = iter_rolling_factor_covariance_batches(
        factor_returns,
        full_all=full_all,
        covariance_method=covariance_method,
        lw_full_windows=lw_full_windows,
    )
    covariance_diag = _write_covariance_parquet(covariance_batches, covariance_path)
    specific_risk = rolling_specific_risk(specific_returns)
    diagnostics = pd.concat([covariance_diag, specific_risk_diagnostics(specific_risk)], ignore_index=True)
    specific_risk.to_parquet(output_dir / "specific_risk.parquet", index=False)
    diagnostics.to_csv(output_dir / "risk_model_diagnostics.csv", index=False)
    print(f"wrote CNE6-style risk model outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-returns", default="outputs/cne6_reproduction/factor_returns.csv")
    parser.add_argument("--specific-returns", default="outputs/cne6_reproduction/specific_returns.parquet")
    parser.add_argument("--output", default="outputs/cne6_reproduction")
    parser.add_argument("--full-all", action="store_true", help="emit full covariance matrix for all factors; can be very large")
    parser.add_argument("--covariance-method", choices=["sample", "ledoit_wolf"], default="ledoit_wolf")
    parser.add_argument("--lw-full-windows", default="252", help="comma-separated windows that use full Ledoit-Wolf covariance")
    args = parser.parse_args()
    run(
        Path(args.factor_returns),
        Path(args.specific_returns),
        Path(args.output),
        full_all=args.full_all,
        covariance_method=args.covariance_method,
        lw_full_windows=_parse_windows(args.lw_full_windows),
    )


if __name__ == "__main__":
    main()
