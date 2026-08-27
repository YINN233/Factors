"""Tushare data fetcher for the local CNE6-style risk model."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "cne6_reproduction"


def get_tushare_pro(token: str | None = None, token_env: str = "TUSHARE_TOKEN"):
    token = token or os.environ.get(token_env)
    if not token:
        raise ValueError(f"Missing tushare token. Set {token_env} or pass --token-env pointing to an environment variable.")
    return ts.pro_api(token)


def _date_str(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _index_tag(index_code: str) -> str:
    return index_code.replace(".", "_")


def _cache_path(name: str, start: str, end: str) -> Path:
    return RAW_DIR / f"cne6_{name}_{start}_{end}.parquet"


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("trade_date", "ann_date", "f_ann_date", "end_date", "list_date", "delist_date", "report_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], format="%Y%m%d", errors="coerce")
    if "vol" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"vol": "volume"})
    return out


def _rate_limit(window: dict, max_calls_per_minute: int) -> None:
    now = time.monotonic()
    if now - window["start"] >= 60:
        window["start"] = now
        window["calls"] = 0
    if window["calls"] >= max_calls_per_minute:
        sleep_for = max(0.0, 60.0 - (now - window["start"])) + 1.0
        time.sleep(sleep_for)
        window["start"] = time.monotonic()
        window["calls"] = 0


def _call_with_limit(fetch: Callable, kwargs: dict, window: dict, max_calls_per_minute: int) -> pd.DataFrame:
    _rate_limit(window, max_calls_per_minute)
    try:
        df = fetch(**kwargs)
        window["calls"] += 1
        return df
    except Exception as exc:
        if "频率" not in str(exc) and "rate" not in str(exc).lower():
            raise
        time.sleep(65)
        window["start"] = time.monotonic()
        window["calls"] = 1
        return fetch(**kwargs)


def _fetch_by_code(
    fetch: Callable,
    codes: Iterable[str],
    start: str,
    end: str,
    desc: str,
    fields: str | None = None,
    max_calls_per_minute: int = 420,
) -> pd.DataFrame:
    frames = []
    window = {"start": time.monotonic(), "calls": 0}
    for code in tqdm(list(codes), desc=desc):
        kwargs = {"ts_code": code, "start_date": start, "end_date": end}
        if fields:
            kwargs["fields"] = fields
        df = _call_with_limit(fetch, kwargs, window, max_calls_per_minute)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _normalize_dates(pd.concat(frames, ignore_index=True).drop_duplicates())


def _fetch_index_weight(pro, index_code: str, start: str, end: str, cache: bool = True) -> pd.DataFrame:
    name = f"index_weight_{_index_tag(index_code)}"
    path = _cache_path(name, start, end)
    if cache and path.exists():
        return pd.read_parquet(path)

    frames = []
    for month_start in tqdm(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS"), desc="fetch cne6 index_weight"):
        chunk_start = max(month_start, pd.Timestamp(start))
        chunk_end = min(month_start + pd.offsets.MonthEnd(), pd.Timestamp(end))
        df = pro.index_weight(index_code=index_code, start_date=_date_str(chunk_start), end_date=_date_str(chunk_end))
        if not df.empty:
            frames.append(df)
    out = _normalize_dates(pd.concat(frames, ignore_index=True).drop_duplicates()) if frames else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def _fetch_stock_basic(pro, cache: bool = True) -> pd.DataFrame:
    path = RAW_DIR / "cne6_stock_basic.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    frames = []
    for status in ("L", "D", "P"):
        df = pro.stock_basic(exchange="", list_status=status, fields=fields)
        if not df.empty:
            frames.append(df)
    out = _normalize_dates(pd.concat(frames, ignore_index=True).drop_duplicates()) if frames else pd.DataFrame()
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def _codes_from_index_weight(index_weight: pd.DataFrame) -> list[str]:
    if index_weight.empty:
        return []
    code_col = "con_code" if "con_code" in index_weight.columns else "ts_code"
    return sorted(index_weight[code_col].dropna().astype(str).unique().tolist())


def _fetch_table_by_code(
    pro,
    table: str,
    codes: list[str],
    start: str,
    end: str,
    fields: list[str],
    cache: bool = True,
) -> pd.DataFrame:
    path = _cache_path(table, start, end)
    if cache and path.exists():
        return pd.read_parquet(path)
    fetch = getattr(pro, table)
    out = _fetch_by_code(fetch, codes, start, end, f"fetch cne6 {table}", fields=",".join(fields))
    if not out.empty:
        sort_cols = [col for col in ["ts_code", "trade_date", "end_date", "ann_date"] if col in out.columns]
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


TABLE_FIELDS = {
    "daily": ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"],
    "adj_factor": ["ts_code", "trade_date", "adj_factor"],
    "daily_basic": [
        "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
        "total_share", "float_share", "free_share", "total_mv", "circ_mv",
    ],
    "moneyflow": ["ts_code", "trade_date", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"],
    "income": [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "total_revenue", "revenue", "oper_cost", "total_cogs", "sell_exp", "admin_exp",
        "fin_exp", "rd_exp", "operate_profit", "total_profit", "n_income",
        "n_income_attr_p", "net_profit", "basic_eps", "diluted_eps",
    ],
    "balancesheet": [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int",
        "money_cap", "accounts_receiv", "inventories", "fix_assets", "intan_assets", "goodwill",
        "st_borr", "lt_borr", "notes_payable", "acct_payable",
    ],
    "cashflow": [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "n_cashflow_act", "net_profit", "c_fr_sale_sg", "c_paid_to_for_empl", "c_paid_for_taxes",
        "free_cashflow", "im_net_cashflow_oper_act", "n_cash_flows_fnc_act",
        "n_cash_flows_inv_act", "c_pay_acq_const_fiolta",
    ],
    "fina_indicator": [
        "ts_code", "ann_date", "end_date", "roe", "roe_dt", "roa", "grossprofit_margin",
        "netprofit_margin", "op_yoy", "or_yoy", "netprofit_yoy", "ocf_to_or",
        "ocf_to_profit", "asset_turnover", "debt_to_assets", "current_ratio", "quick_ratio",
        "inv_turn", "ar_turn", "eps", "bps", "cfps",
    ],
    "report_rc": [
        "ts_code", "name", "report_date", "report_title", "report_type", "classify",
        "org_name", "author_name", "quarter", "op_rt", "op_pr", "tp", "np",
        "eps", "pe", "rd", "roe", "ev_ebitda", "rating", "max_price",
        "min_price", "imp_dg", "create_time",
    ],
}


def audit_dataframes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table, df in frames.items():
        if df.empty:
            rows.append({"table": table, "field": "__table__", "year": "all", "rows": 0, "non_null": 0, "coverage": 0.0, "start": "", "end": "", "n_codes": 0})
            continue
        date_col = next((col for col in ["trade_date", "ann_date", "report_date", "end_date", "list_date"] if col in df.columns), None)
        start = pd.to_datetime(df[date_col]).min() if date_col else pd.NaT
        end = pd.to_datetime(df[date_col]).max() if date_col else pd.NaT
        n_codes = int(df["ts_code"].nunique()) if "ts_code" in df.columns else int(df.get("con_code", pd.Series(dtype=object)).nunique())
        rows.append(
            {
                "table": table,
                "field": "__table__",
                "year": "all",
                "rows": int(len(df)),
                "non_null": int(len(df)),
                "coverage": 1.0,
                "start": "" if pd.isna(start) else start.date().isoformat(),
                "end": "" if pd.isna(end) else end.date().isoformat(),
                "n_codes": n_codes,
            }
        )
        if date_col:
            years = pd.to_datetime(df[date_col]).dt.year
            for field in df.columns:
                grouped = df[field].notna().groupby(years, sort=True)
                for year, valid in grouped:
                    rows.append(
                        {
                            "table": table,
                            "field": field,
                            "year": int(year) if pd.notna(year) else "unknown",
                            "rows": int(len(valid)),
                            "non_null": int(valid.sum()),
                            "coverage": float(valid.mean()) if len(valid) else 0.0,
                            "start": "",
                            "end": "",
                            "n_codes": n_codes,
                        }
                    )
    return pd.DataFrame(rows)


def run_fetch_all(
    start: str = "20100101",
    end: str = "20260722",
    index_code: str = "000905.SH",
    token: str | None = None,
    token_env: str = "TUSHARE_TOKEN",
    cache: bool = True,
    tables: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pro = get_tushare_pro(token=token, token_env=token_env)
    frames: dict[str, pd.DataFrame] = {}
    frames[f"index_weight_{_index_tag(index_code)}"] = _fetch_index_weight(pro, index_code, start, end, cache=cache)
    codes = _codes_from_index_weight(frames[f"index_weight_{_index_tag(index_code)}"])
    if not codes:
        raise RuntimeError(f"No index constituents fetched for {index_code}")

    requested = set(tables or TABLE_FIELDS)
    if "stock_basic" in requested:
        frames["stock_basic"] = _fetch_stock_basic(pro, cache=cache)
    for table, fields in TABLE_FIELDS.items():
        if table not in requested:
            continue
        frames[table] = _fetch_table_by_code(pro, table, codes, start, end, fields, cache=cache)

    audit = audit_dataframes(frames)
    audit.to_csv(OUTPUT_DIR / "data_availability.csv", index=False)
    print(f"fetched CNE6-style raw data for {len(codes)} historical {index_code} constituents")
    print(f"wrote availability audit to {OUTPUT_DIR / 'data_availability.csv'}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="20260722")
    parser.add_argument("--index-code", default="000905.SH")
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    parser.add_argument("--tables", default="", help="comma-separated table names; empty means all data tables")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    tables = [item.strip() for item in args.tables.split(",") if item.strip()] or None
    run_fetch_all(start=args.start, end=args.end, index_code=args.index_code, token_env=args.token_env, cache=not args.no_cache, tables=tables)


if __name__ == "__main__":
    main()
