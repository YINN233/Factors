"""Command line entry points for refreshing monitoring data and derivatives."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .backfill import annual_ranges, latest_constituent_codes, merge_range_caches
from .contracts import FUTURE_PRODUCTS, default_raw_dir
from .coverage import write_coverage_outputs
from .pipeline import (
    _processed_dir,
    build_basis_from_cache,
    build_dashboard_derivatives_from_cache,
    build_etf_signals_from_cache,
)
from .etf_universe import classification_review_template, load_classification
from factors.data.index_futures_fetcher import IndexFuturesFetcher


def _products(value: str) -> list[str]:
    values = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values).difference(FUTURE_PRODUCTS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown products: {', '.join(invalid)}")
    return values


def refresh(args: argparse.Namespace) -> None:
    fetcher = IndexFuturesFetcher(token=args.token, raw_dir=args.raw_dir)
    products = _products(args.products)
    use_cache = not args.force
    fetcher.fetch_futures_basic(cache=use_cache)
    if args.mode in ("futures", "all"):
        fetcher.fetch_futures_daily(args.start, args.end, products=products, cache=use_cache)
        fetcher.fetch_futures_mapping(args.start, args.end, products=products, cache=use_cache)
        fetcher.fetch_index_daily(args.start, args.end, products=products, cache=use_cache)
        weights = fetcher.fetch_index_weights(args.start, args.end, products=products, cache=use_cache)
        if args.with_dividends and not weights.empty:
            codes = weights["con_code"].dropna().astype(str).unique().tolist()
            fetcher.fetch_dividends(codes, cache=use_cache)
            fetcher.fetch_stock_daily(codes, args.start, args.end, cache=use_cache)
        build_basis_from_cache(args.raw_dir, start_date=args.start, end_date=args.end)
    if args.mode in ("etf", "all"):
        fetcher.fetch_etf_basic(cache=use_cache)
        if args.etf_codes:
            codes = [code.strip() for code in args.etf_codes.split(",") if code.strip()]
            fetcher.fetch_etf_daily(codes, args.start, args.end, cache=use_cache)
            fetcher.fetch_etf_shares(codes, args.start, args.end, cache=use_cache)
            build_etf_signals_from_cache(args.raw_dir, start_date=args.start, end_date=args.end)
    if args.mode == "all":
        summary, pairs = build_dashboard_derivatives_from_cache(
            args.raw_dir,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"dashboard summary rows: {len(summary)}, pair rows: {len(pairs)}")
    audit = fetcher.flush_audit()
    print(f"refresh complete: audit table has {len(audit)} rows")


def build(args: argparse.Namespace) -> None:
    if args.mode in ("futures", "all"):
        basis = build_basis_from_cache(args.raw_dir, start_date=args.start, end_date=args.end)
        print(f"basis rows: {len(basis)}")
    if args.mode in ("etf", "all"):
        groups, signals = build_etf_signals_from_cache(args.raw_dir, start_date=args.start, end_date=args.end)
        print(f"ETF group rows: {len(groups)}, signal rows: {len(signals)}")
    if args.mode == "all":
        summary, pairs = build_dashboard_derivatives_from_cache(
            args.raw_dir,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"dashboard summary rows: {len(summary)}, pair rows: {len(pairs)}")


def prepare_etf_classification(args: argparse.Namespace) -> None:
    """Export an auditable ETF review queue without assigning risk labels."""

    fetcher = IndexFuturesFetcher(token=args.token, raw_dir=args.raw_dir)
    basic = fetcher.fetch_etf_basic(cache=not args.force)
    if args.etf_codes:
        codes = {code.strip() for code in args.etf_codes.split(",") if code.strip()}
        basic = basic[basic["ts_code"].astype(str).isin(codes)].copy()
    template = classification_review_template(basic, as_of_date=args.as_of_date)
    target = Path(args.output)
    if target.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {target}; pass --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(target, index=False)
    audit = fetcher.flush_audit()
    print(f"ETF review rows: {len(template)}; wrote {target}; audit table rows: {len(audit)}")


def _classification_codes(path: Path) -> list[str]:
    classification = load_classification(path)
    if classification.empty:
        return []
    reviewed = classification[classification["reviewed"].fillna(False).astype(bool)]
    return sorted(reviewed["ts_code"].dropna().astype(str).unique())


def backfill(args: argparse.Namespace) -> None:
    """Run resumable annual downloads, consolidate, and build derivatives."""

    fetcher = IndexFuturesFetcher(token=args.token, raw_dir=args.raw_dir)
    products = _products(args.products)
    ranges = annual_ranges(args.start, args.end)
    use_cache = not args.force
    classification_path = Path(args.classification_path)
    etf_codes = [code.strip() for code in args.etf_codes.split(",") if code.strip()]
    if not etf_codes:
        etf_codes = _classification_codes(classification_path)
    if args.mode in ("etf", "all") and not etf_codes:
        raise ValueError("ETF backfill requires reviewed classification rows or --etf-codes")

    fetcher.fetch_futures_basic(cache=use_cache)
    if args.mode in ("etf", "all"):
        fetcher.fetch_etf_basic(cache=use_cache)
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        print(f"backfill {index}/{len(ranges)}: {chunk_start}..{chunk_end}")
        try:
            if args.mode in ("futures", "all"):
                fetcher.fetch_futures_daily(chunk_start, chunk_end, products=products, cache=use_cache)
                fetcher.fetch_futures_mapping(chunk_start, chunk_end, products=products, cache=use_cache)
                fetcher.fetch_index_daily(chunk_start, chunk_end, products=products, cache=use_cache)
                fetcher.fetch_index_weights(
                    chunk_start,
                    chunk_end,
                    products=products,
                    warmup_days=370 if index == 1 else 0,
                    cache=use_cache,
                )
            if args.mode in ("etf", "all"):
                fetcher.fetch_etf_daily(etf_codes, chunk_start, chunk_end, cache=use_cache)
                fetcher.fetch_etf_shares(etf_codes, chunk_start, chunk_end, cache=use_cache)
        finally:
            fetcher.flush_audit()

    if args.mode in ("futures", "all"):
        futures = merge_range_caches(
            args.raw_dir,
            stem="futures_daily",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["ts_code", "trade_date"],
        )
        merge_range_caches(
            args.raw_dir,
            stem="futures_mapping",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["product", "trade_date"],
        )
        indices = merge_range_caches(
            args.raw_dir,
            stem="index_daily",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["ts_code", "trade_date"],
        )
        weights = merge_range_caches(
            args.raw_dir,
            stem="index_weights",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["index_code", "con_code", "trade_date"],
        )
        if args.with_dividends and not weights.empty and not indices.empty:
            latest_date = pd.to_datetime(indices["trade_date"], errors="coerce").max().strftime("%Y%m%d")
            codes = latest_constituent_codes(weights, latest_date)
            print(f"dividend snapshot: {len(codes)} current constituent codes at {latest_date}")
            fetcher.fetch_dividends(codes, cache=use_cache)
            fetcher.fetch_stock_snapshot(codes, latest_date, cache=use_cache)
            fetcher.flush_audit()
        basis = build_basis_from_cache(args.raw_dir, start_date=args.start, end_date=args.end)
        print(f"consolidated futures={len(futures)}, basis={len(basis)}")

    if args.mode in ("etf", "all"):
        daily = merge_range_caches(
            args.raw_dir,
            stem="etf_daily_universe",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["ts_code", "trade_date"],
        )
        shares = merge_range_caches(
            args.raw_dir,
            stem="etf_shares_universe",
            ranges=ranges,
            output_start=args.start,
            output_end=args.end,
            key_columns=["ts_code", "trade_date"],
        )
        groups, signals = build_etf_signals_from_cache(
            args.raw_dir,
            start_date=args.start,
            end_date=args.end,
            classification_path=classification_path,
            cost_bps_per_turnover=args.cost_bps,
        )
        print(f"consolidated ETF daily={len(daily)}, shares={len(shares)}, signals={len(signals)}")
    if args.mode == "all":
        summary, pairs = build_dashboard_derivatives_from_cache(
            args.raw_dir,
            start_date=args.start,
            end_date=args.end,
        )
        insufficient = int(summary["overall_evidence_status"].eq("insufficient").sum()) if not summary.empty else 0
        print(
            f"dashboard summary={len(summary)}, pair rows={len(pairs)}, "
            f"insufficient summary rows={insufficient}"
        )
    coverage, gaps = write_coverage_outputs(
        args.raw_dir,
        _processed_dir(Path(args.raw_dir)),
        start_date=args.start,
        end_date=args.end,
    )
    available = int(coverage["status"].eq("available").sum())
    print(f"coverage datasets available={available}/{len(coverage)}; company DB gaps={len(gaps)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="股指期货监测数据刷新和派生指标")
    sub = root.add_subparsers(dest="command", required=True)
    for command, fn in (("refresh", refresh), ("build", build)):
        p = sub.add_parser(command)
        p.set_defaults(handler=fn)
        p.add_argument("--start", default="20170101")
        p.add_argument("--end", default=date.today().strftime("%Y%m%d"))
        p.add_argument("--mode", choices=["futures", "etf", "all"], default="futures")
        p.add_argument("--products", default=",".join(FUTURE_PRODUCTS))
        p.add_argument("--etf-codes", default="", help="comma-separated ETF codes for the first refresh")
        p.add_argument("--raw-dir", type=Path, default=default_raw_dir())
        p.add_argument("--token", default=None, help="optional token; prefer TUSHARE_TOKEN")
        p.add_argument("--force", action="store_true", help="ignore and rebuild matching raw cache files")
        p.add_argument("--with-dividends", action="store_true", help="also fetch constituent dividends and stock prices; slow for full history")
    review = sub.add_parser(
        "prepare-etf-classification",
        help="从 Tushare 基金主表生成待人工审核的 ETF 分类队列",
    )
    review.set_defaults(handler=prepare_etf_classification)
    review.add_argument("--as-of-date", default=None)
    review.add_argument("--etf-codes", default="", help="comma-separated codes; empty means all eligible stock funds")
    review.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("etf_classification_review.csv"),
    )
    review.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    review.add_argument("--token", default=None, help="optional token; prefer TUSHARE_TOKEN")
    review.add_argument("--force", action="store_true", help="refresh fund master and overwrite the review queue")
    history = sub.add_parser("backfill", help="按年度断点续传并合并 2017 年至今的监测数据")
    history.set_defaults(handler=backfill)
    history.add_argument("--start", default="20170101")
    history.add_argument("--end", default=date.today().strftime("%Y%m%d"))
    history.add_argument("--mode", choices=["futures", "etf", "all"], default="all")
    history.add_argument("--products", default=",".join(FUTURE_PRODUCTS))
    history.add_argument("--etf-codes", default="", help="empty means all reviewed classification codes")
    history.add_argument(
        "--classification-path",
        type=Path,
        default=Path(__file__).with_name("etf_classification.csv"),
    )
    history.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    history.add_argument("--token", default=None, help="optional token; prefer TUSHARE_TOKEN")
    history.add_argument("--cost-bps", type=float, default=5.0)
    history.add_argument("--with-dividends", action="store_true")
    history.add_argument("--force", action="store_true")
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.handler(args)
