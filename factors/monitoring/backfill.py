"""Resumable annual backfill and cache consolidation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def annual_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into calendar-year requests."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError(f"invalid backfill range: {start_date}..{end_date}")
    ranges: list[tuple[str, str]] = []
    for year in range(start.year, end.year + 1):
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right = min(end, pd.Timestamp(year=year, month=12, day=31))
        ranges.append((left.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
    return ranges


def merge_range_caches(
    raw_dir: str | Path,
    *,
    stem: str,
    ranges: Iterable[tuple[str, str]],
    output_start: str,
    output_end: str,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Merge completed ranges; fail rather than label partial data complete."""

    root = Path(raw_dir)
    paths = [root / f"{stem}_{start}_{end}.parquet" for start, end in ranges]
    missing = [path for path in paths if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"missing {stem} backfill caches: {names}{suffix}")
    frames = [pd.read_parquet(path) for path in paths]
    nonempty = [frame for frame in frames if not frame.empty]
    merged = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if not merged.empty:
        for column in ("trade_date", "ann_date", "list_date", "delist_date", "ex_date"):
            if column in merged.columns:
                merged[column] = pd.to_datetime(merged[column], errors="coerce")
        missing_keys = sorted(set(key_columns).difference(merged.columns))
        if missing_keys:
            raise KeyError(f"{stem} missing merge keys: {', '.join(missing_keys)}")
        sort_columns = [column for column in ("trade_date", *key_columns) if column in merged.columns]
        merged = (
            merged.sort_values(sort_columns, kind="stable")
            .drop_duplicates(list(key_columns), keep="last")
            .reset_index(drop=True)
        )
    output = root / f"{stem}_{output_start}_{output_end}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    return merged


def latest_constituent_codes(weights: pd.DataFrame, as_of_date: str) -> list[str]:
    """Return the union of each product's latest available constituents."""

    required = {"con_code", "trade_date"}
    missing = sorted(required.difference(weights.columns))
    if missing:
        raise KeyError(f"index weights missing columns: {', '.join(missing)}")
    frame = weights.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame[frame["trade_date"] <= pd.Timestamp(as_of_date)].dropna(subset=["trade_date", "con_code"])
    if frame.empty:
        return []
    group_columns = ["product"] if "product" in frame.columns else ["index_code"]
    latest_dates = frame.groupby(group_columns, dropna=False)["trade_date"].transform("max")
    return sorted(frame.loc[frame["trade_date"].eq(latest_dates), "con_code"].astype(str).unique())
