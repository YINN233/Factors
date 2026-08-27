"""Tushare adapter for the index-futures monitoring dashboard.

This module is intentionally separate from the equity alpha fetcher.  It
stores raw responses with enough metadata to audit source, date coverage and
point-in-time limitations before any signal is calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Iterable, Sequence

import pandas as pd

from factors.data.fetcher import get_tushare_pro
from factors.monitoring.contracts import FUTURE_PRODUCTS, INDEX_CODES, default_raw_dir, normalize_dates
from factors.monitoring.data_audit import append_audit, audit_frame


@dataclass(frozen=True)
class FetchConfig:
    raw_dir: Path
    retries: int = 3
    retry_seconds: float = 2.0
    max_calls_per_minute: int = 420


def _date_chunks(start_date: str, end_date: str, months: int = 6) -> list[tuple[str, str]]:
    """Split a calendar range so Tushare row limits stay observable."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def _overlaps(contract: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    listed = pd.to_datetime(contract.get("list_date"), errors="coerce")
    delisted = pd.to_datetime(contract.get("delist_date"), errors="coerce")
    return (pd.isna(delisted) or delisted >= start) and (pd.isna(listed) or listed <= end)


def _is_tradeable_equity_contract(row: pd.Series) -> bool:
    """Exclude Tushare's ``IF.CFX``/``IFL1.CFX`` aliases from daily quotes."""

    symbol = str(row.get("symbol") or str(row.get("ts_code", "")).split(".")[0])
    product = str(row.get("fut_code", ""))
    return product in FUTURE_PRODUCTS and bool(re.fullmatch(rf"{product}\d{{4}}", symbol))


class IndexFuturesFetcher:
    """Fetch and cache only the data sets used by the first monitoring MVP."""

    def __init__(
        self,
        pro=None,
        *,
        token: str | None = None,
        raw_dir: str | Path | None = None,
        retries: int = 3,
        retry_seconds: float = 2.0,
    ) -> None:
        root = Path(raw_dir) if raw_dir is not None else default_raw_dir()
        root.mkdir(parents=True, exist_ok=True)
        self.pro = pro or get_tushare_pro(token)
        self.config = FetchConfig(root, retries, retry_seconds)
        self._audit_records: list[dict] = []
        self._rate_window_started = time.monotonic()
        self._calls_in_window = 0

    @property
    def raw_dir(self) -> Path:
        return self.config.raw_dir

    @property
    def audit_path(self) -> Path:
        return self.raw_dir / "data_availability.csv"

    def flush_audit(self) -> pd.DataFrame:
        """Persist the accumulated coverage records and clear the buffer."""

        if not self._audit_records:
            return pd.DataFrame()
        records, self._audit_records = self._audit_records, []
        return append_audit(records, self.audit_path)

    def _call(self, dataset: str, fn, *args, **kwargs) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                self._throttle()
                frame = fn(*args, **kwargs)
                self._calls_in_window += 1
                return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            except Exception as exc:  # Tushare uses strings for rate and permission errors.
                last_error = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(self.config.retry_seconds * (attempt + 1))
        self._audit_records.append(
            {
                "dataset": dataset,
                "source": "tushare",
                "rows": 0,
                "missing_reason": f"{type(last_error).__name__}: {last_error}",
                "quality_grade": "D",
                "point_in_time": False,
                "proxy": False,
            }
        )
        raise RuntimeError(f"Tushare request failed for {dataset}") from last_error

    def _throttle(self) -> None:
        """Respect a conservative request ceiling for per-code bulk pulls."""

        elapsed = time.monotonic() - self._rate_window_started
        if elapsed >= 60:
            self._rate_window_started = time.monotonic()
            self._calls_in_window = 0
            return
        if self._calls_in_window >= self.config.max_calls_per_minute:
            time.sleep(60 - elapsed + 1)
            self._rate_window_started = time.monotonic()
            self._calls_in_window = 0

    def _cache_path(self, name: str, start_date: str | None = None, end_date: str | None = None) -> Path:
        suffix = f"_{start_date}_{end_date}" if start_date and end_date else ""
        return self.raw_dir / f"{name}{suffix}.parquet"

    @staticmethod
    def _cache_covers(
        path: Path,
        *,
        products: Iterable[str] | None = None,
        ts_codes: Iterable[str] | None = None,
    ) -> pd.DataFrame | None:
        """Read a cache only when it contains every requested code/bucket."""

        if not path.exists():
            return None
        cached = pd.read_parquet(path)
        if cached.empty:
            return None
        if products is not None:
            wanted = {str(value).upper() for value in products}
            if "product" in cached.columns:
                available = set(cached["product"].dropna().astype(str).str.upper())
            elif "index_code" in cached.columns:
                reverse = {code: product for product, code in INDEX_CODES.items()}
                available = {reverse.get(str(code), str(code)) for code in cached["index_code"].dropna().astype(str)}
            else:
                return None
            if not wanted.issubset(available):
                return None
        if ts_codes is not None:
            wanted_codes = {str(value) for value in ts_codes}
            if "ts_code" not in cached.columns:
                return None
            available_codes = set(cached["ts_code"].dropna().astype(str))
            if not wanted_codes.issubset(available_codes):
                return None
        return cached

    def _store(
        self,
        frame: pd.DataFrame,
        *,
        dataset: str,
        path: Path,
        key_columns: Sequence[str],
        requested_start: str | None = None,
        requested_end: str | None = None,
        point_in_time: bool = False,
        proxy: bool = False,
        quality_grade: str = "B",
    ) -> pd.DataFrame:
        out = normalize_dates(
            frame,
            ["trade_date", "ann_date", "end_date", "record_date", "ex_date", "pay_date", "list_date", "delist_date", "last_ddate"],
        )
        if not out.empty:
            out = out.drop_duplicates().copy()
            out["source"] = "tushare"
            out["fetch_time"] = pd.Timestamp.now(tz="UTC")
        out.to_parquet(path, index=False)
        self._audit_records.append(
            audit_frame(
                out,
                dataset,
                key_columns=key_columns,
                requested_start=requested_start,
                requested_end=requested_end,
                source="tushare",
                quality_grade=quality_grade,
                point_in_time=point_in_time,
                proxy=proxy,
            )
        )
        return out

    def fetch_futures_basic(self, *, cache: bool = True) -> pd.DataFrame:
        path = self._cache_path("futures_basic_cffex")
        if cache and path.exists():
            cached = pd.read_parquet(path)
            if "symbol" in cached.columns:
                cached = cached[cached.apply(_is_tradeable_equity_contract, axis=1)].copy()
            if not cached.empty:
                return cached
        frame = self._call("fut_basic", self.pro.fut_basic, exchange="CFFEX")
        frame = frame[frame.apply(_is_tradeable_equity_contract, axis=1)].copy()
        return self._store(frame, dataset="futures_basic", path=path, key_columns=["ts_code"])

    def fetch_futures_daily(
        self,
        start_date: str,
        end_date: str,
        *,
        products: Iterable[str] = FUTURE_PRODUCTS,
        cache: bool = True,
    ) -> pd.DataFrame:
        path = self._cache_path("futures_daily", start_date, end_date)
        products = tuple(str(value).upper() for value in products)
        if cache:
            cached = self._cache_covers(path, products=products)
            if cached is not None:
                return cached
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        basic = self.fetch_futures_basic(cache=cache)
        product_set = set(products)
        contracts = basic[basic["fut_code"].isin(product_set)].copy()
        contracts = contracts[contracts.apply(_overlaps, axis=1, args=(start, end))]
        frames: list[pd.DataFrame] = []
        for contract in contracts.itertuples(index=False):
            frame = self._call(
                f"fut_daily:{contract.ts_code}",
                self.pro.fut_daily,
                ts_code=contract.ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frame["product"] = contract.fut_code
                frame["expiry_date"] = contract.delist_date
                frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="futures_daily",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )

    def fetch_futures_mapping(
        self,
        start_date: str,
        end_date: str,
        *,
        products: Iterable[str] = FUTURE_PRODUCTS,
        cache: bool = True,
    ) -> pd.DataFrame:
        path = self._cache_path("futures_mapping", start_date, end_date)
        products = tuple(str(value).upper() for value in products)
        if cache:
            cached = self._cache_covers(path, products=products)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for product in products:
            frame = self._call(
                f"fut_mapping:{product}",
                self.pro.fut_mapping,
                ts_code=f"{product}.CFX",
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frame["product"] = product
                frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="futures_mapping",
            path=path,
            key_columns=["product", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )

    def fetch_index_daily(
        self,
        start_date: str,
        end_date: str,
        *,
        products: Iterable[str] = FUTURE_PRODUCTS,
        cache: bool = True,
    ) -> pd.DataFrame:
        path = self._cache_path("index_daily", start_date, end_date)
        products = tuple(str(value).upper() for value in products)
        if cache:
            cached = self._cache_covers(path, products=products)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for product in products:
            index_code = INDEX_CODES[product]
            frame = self._call(
                f"index_daily:{index_code}",
                self.pro.index_daily,
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frame["product"] = product
                frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="index_daily",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )

    def fetch_index_weights(
        self,
        start_date: str,
        end_date: str,
        *,
        products: Iterable[str] = FUTURE_PRODUCTS,
        months_per_chunk: int = 3,
        warmup_days: int = 370,
        cache: bool = True,
    ) -> pd.DataFrame:
        path = self._cache_path("index_weights", start_date, end_date)
        products = tuple(str(value).upper() for value in products)
        if cache:
            cached = self._cache_covers(path, products=products)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        # A current as-of date often falls between month-end snapshots.  Pull
        # one year of warm-up so the latest prior snapshot is available.
        requested_start = pd.Timestamp(start_date)
        weight_start = max(
            pd.Timestamp("2017-01-01"),
            requested_start - pd.Timedelta(days=max(0, int(warmup_days))),
        )
        for product in products:
            index_code = INDEX_CODES[product]
            for chunk_start, chunk_end in _date_chunks(weight_start.strftime("%Y%m%d"), end_date, months_per_chunk):
                frame = self._call(
                    f"index_weight:{index_code}",
                    self.pro.index_weight,
                    index_code=index_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                if not frame.empty:
                    frame["product"] = product
                    frame["weight_method"] = "monthly_snapshot_tushare"
                    frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="index_weights",
            path=path,
            key_columns=["index_code", "con_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
            proxy=True,
            quality_grade="C",
        )

    def fetch_stock_snapshot(
        self,
        ts_codes: Iterable[str],
        trade_date: str,
        *,
        cache_tag: str = "constituents_snapshot",
        cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch one market-wide stock snapshot and keep requested codes."""

        codes = {str(value) for value in ts_codes}
        path = self._cache_path(f"stock_daily_{cache_tag}")
        if cache:
            cached = self._cache_covers(path, ts_codes=codes)
            if cached is not None and "trade_date" in cached.columns:
                dates = pd.to_datetime(cached["trade_date"], errors="coerce")
                if dates.eq(pd.Timestamp(trade_date)).all():
                    return cached
        frame = self._call(
            f"daily_snapshot:{trade_date}",
            self.pro.daily,
            trade_date=trade_date,
        )
        if not frame.empty:
            frame = frame[frame["ts_code"].astype(str).isin(codes)].copy()
        return self._store(
            frame,
            dataset="stock_daily_snapshot",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=trade_date,
            requested_end=trade_date,
        )

    def fetch_stock_daily(
        self,
        ts_codes: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        cache_tag: str = "constituents",
        months_per_chunk: int = 6,
        cache: bool = True,
    ) -> pd.DataFrame:
        codes = tuple(sorted({str(value) for value in ts_codes}))
        path = self._cache_path(f"stock_daily_{cache_tag}", start_date, end_date)
        if cache:
            cached = self._cache_covers(path, ts_codes=codes)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, months_per_chunk):
                frame = self._call(
                    f"daily:{ts_code}:{chunk_start}:{chunk_end}",
                    self.pro.daily,
                    ts_code=ts_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                if not frame.empty:
                    frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="stock_daily",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )

    def fetch_dividends(
        self,
        ts_codes: Iterable[str],
        *,
        cache_tag: str = "constituents",
        start_date: str | None = None,
        end_date: str | None = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        codes = tuple(sorted({str(value) for value in ts_codes}))
        path = self._cache_path(f"dividend_events_{cache_tag}", start_date, end_date)
        if cache:
            cached = self._cache_covers(path, ts_codes=codes)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            # ``dividend`` is naturally bounded by one stock's corporate
            # actions and its public API does not consistently support a
            # start/end range.  Fetch once per code and apply the requested
            # announcement-date filter locally.
            frame = self._call(f"dividend:{ts_code}", self.pro.dividend, ts_code=ts_code)
            if not frame.empty:
                frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if start_date and end_date and "ann_date" in merged.columns:
            ann = pd.to_datetime(merged["ann_date"], errors="coerce")
            merged = merged[(ann >= pd.Timestamp(start_date)) & (ann <= pd.Timestamp(end_date))].copy()
        return self._store(
            merged,
            dataset="dividend_events",
            path=path,
            key_columns=["ts_code", "ann_date", "div_proc", "ex_date", "cash_div_tax"],
            requested_start=start_date,
            requested_end=end_date,
            point_in_time=True,
        )

    def fetch_etf_basic(self, *, cache: bool = True) -> pd.DataFrame:
        path = self._cache_path("etf_basic")
        if cache and path.exists():
            return pd.read_parquet(path)
        frame = self._call("fund_basic", self.pro.fund_basic, market="E")
        return self._store(frame, dataset="etf_basic", path=path, key_columns=["ts_code"])

    def fetch_etf_daily(
        self,
        ts_codes: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        cache_tag: str = "universe",
        months_per_chunk: int = 6,
        cache: bool = True,
    ) -> pd.DataFrame:
        codes = tuple(sorted({str(value) for value in ts_codes}))
        path = self._cache_path(f"etf_daily_{cache_tag}", start_date, end_date)
        if cache:
            cached = self._cache_covers(path, ts_codes=codes)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, months_per_chunk):
                frame = self._call(
                    f"fund_daily:{ts_code}:{chunk_start}:{chunk_end}",
                    self.pro.fund_daily,
                    ts_code=ts_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                if not frame.empty:
                    frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="etf_daily",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )

    def fetch_etf_shares(
        self,
        ts_codes: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        cache_tag: str = "universe",
        months_per_chunk: int = 6,
        cache: bool = True,
    ) -> pd.DataFrame:
        codes = tuple(sorted({str(value) for value in ts_codes}))
        path = self._cache_path(f"etf_shares_{cache_tag}", start_date, end_date)
        if cache:
            cached = self._cache_covers(path, ts_codes=codes)
            if cached is not None:
                return cached
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, months_per_chunk):
                frame = self._call(
                    f"fund_share:{ts_code}:{chunk_start}:{chunk_end}",
                    self.pro.fund_share,
                    ts_code=ts_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                if not frame.empty:
                    frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._store(
            merged,
            dataset="etf_shares",
            path=path,
            key_columns=["ts_code", "trade_date"],
            requested_start=start_date,
            requested_end=end_date,
        )
