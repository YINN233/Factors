"""
数据下载模块：从 tushare 拉取指数成分股、日行情、行业分类和财报数据。
"""

import os
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import tushare as ts
from tqdm import tqdm


# 默认数据存储路径
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_tushare_pro(token: Optional[str] = None) -> ts.pro_api:
    """初始化 tushare pro 接口。"""
    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError(
            "请设置 TUSHARE_TOKEN 环境变量，或在调用时传入 token。"
            "获取方式：https://tushare.pro/register"
        )
    return ts.pro_api(token)


def _open_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    """Return open A-share trade dates as YYYYMMDD strings."""
    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    if cal.empty:
        return [d.strftime("%Y%m%d") for d in pd.date_range(start_date, end_date, freq="B")]
    return sorted(cal["cal_date"].astype(str).tolist())


def _fetch_by_trade_date(
    pro,
    fetch_fn: Callable,
    start_date: str,
    end_date: str,
    desc: str,
    max_calls_per_minute: int = 420,
) -> pd.DataFrame:
    """Fetch tushare endpoints that are naturally bounded by a trade date."""
    frames = []
    window_start = time.monotonic()
    calls_in_window = 0
    for trade_date in tqdm(_open_trade_dates(pro, start_date, end_date), desc=desc):
        if calls_in_window >= max_calls_per_minute:
            elapsed = time.monotonic() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed + 1)
            window_start = time.monotonic()
            calls_in_window = 0
        try:
            df = fetch_fn(trade_date=trade_date)
            calls_in_window += 1
        except Exception as exc:
            if "频率超限" not in str(exc):
                raise
            time.sleep(65)
            window_start = time.monotonic()
            calls_in_window = 1
            df = fetch_fn(trade_date=trade_date)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    return out


def _fetch_by_ann_month(
    fetch_fn: Callable,
    start_date: str,
    end_date: str,
    desc: str,
    fields: str | None = None,
) -> pd.DataFrame:
    """Fetch financial-statement endpoints by announcement-date month."""
    frames = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    month_starts = pd.date_range(start_ts.replace(day=1), end_ts, freq="MS")
    for start in tqdm(month_starts, desc=desc):
        month_start = max(start, pd.Timestamp(start_date))
        month_end = min(start + pd.offsets.MonthEnd(), pd.Timestamp(end_date))
        kwargs = {
            "start_date": month_start.strftime("%Y%m%d"),
            "end_date": month_end.strftime("%Y%m%d"),
        }
        if fields:
            kwargs["fields"] = fields
        df = fetch_fn(**kwargs)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    for col in ("ann_date", "f_ann_date", "end_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], format="%Y%m%d", errors="coerce")
    return out


def _fetch_by_ts_code(
    fetch_fn: Callable,
    ts_codes: Iterable[str],
    start_date: str,
    end_date: str,
    desc: str,
    fields: str | None = None,
    max_calls_per_minute: int = 450,
) -> pd.DataFrame:
    """Fetch endpoints that require ``ts_code``."""
    frames = []
    window_start = time.monotonic()
    calls_in_window = 0
    for ts_code in tqdm(list(ts_codes), desc=desc):
        if calls_in_window >= max_calls_per_minute:
            elapsed = time.monotonic() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed + 1)
            window_start = time.monotonic()
            calls_in_window = 0
        kwargs = {"ts_code": ts_code, "start_date": start_date, "end_date": end_date}
        if fields:
            kwargs["fields"] = fields
        try:
            df = fetch_fn(**kwargs)
            calls_in_window += 1
        except Exception as exc:
            if "频率超限" not in str(exc):
                raise
            time.sleep(65)
            window_start = time.monotonic()
            calls_in_window = 1
            df = fetch_fn(**kwargs)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    for col in ("ann_date", "f_ann_date", "end_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], format="%Y%m%d", errors="coerce")
    return out


def _index_tag(index_code: str) -> str:
    return index_code.replace(".", "_")


def _cache_name(name: str, start_date: str, end_date: str, cache_tag: str | None = None) -> str:
    if cache_tag:
        return f"{name}_{cache_tag}_{start_date}_{end_date}.parquet"
    return f"{name}_{start_date}_{end_date}.parquet"


def _codes_from_index_cache(index_code: str, start_date: str, end_date: str) -> list[str]:
    cache_path = RAW_DIR / f"index_weight_{_index_tag(index_code)}_{start_date}_{end_date}.parquet"
    if not cache_path.exists():
        return []
    idx = pd.read_parquet(cache_path)
    code_col = "con_code" if "con_code" in idx.columns else "ts_code"
    return sorted(idx[code_col].dropna().astype(str).unique().tolist())


def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tushare daily column names used by downstream builders."""
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    return df


def fetch_index_weight(
    pro,
    index_code: str = "000300.SH",
    start_date: str = "20180101",
    end_date: str = "20241231",
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取指数成分股权重（沪深300）。
    tushare 的 index_weight 接口返回每个交易日的成分股权重。
    """
    cache_path = RAW_DIR / f"index_weight_{index_code.replace('.', '_')}_{start_date}_{end_date}.parquet"
    if cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            min_date = pd.to_datetime(cached["trade_date"]).min()
            max_date = pd.to_datetime(cached["trade_date"]).max()
            if min_date <= pd.to_datetime(start_date) and max_date >= pd.to_datetime(end_date) - pd.Timedelta(days=45):
                return cached

    frames = []
    for start in tqdm(pd.date_range(start_date, end_date, freq="YS"), desc="fetch index_weight"):
        year_end = min(start + pd.offsets.YearEnd(), pd.Timestamp(end_date))
        df_year = pro.index_weight(
            index_code=index_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=year_end.strftime("%Y%m%d"),
        )
        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values(["trade_date", "con_code"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_daily(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取 A 股日行情（open, high, low, close, volume, amount 等）。
    tushare 的 daily 接口一次最多返回 5000 条，需要按股票分批或按日期分批拉取。
    这里采用按日期分批的策略（每天最多 5000 只股票，全 A 约 5000+，刚好够）。
    """
    cache_path = RAW_DIR / f"daily_{start_date}_{end_date}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = _fetch_by_trade_date(pro, pro.daily, start_date, end_date, "fetch daily")
    if df.empty:
        return pd.DataFrame()
    df = _normalize_daily_columns(df)
    # 保留全 A 股（不过滤科创板/北交所），后续 builder 会按成分股权重筛选
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_daily_basic(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取每日指标（turnover_rate, total_mv 等）。
    """
    cache_path = RAW_DIR / f"daily_basic_{start_date}_{end_date}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = _fetch_by_trade_date(pro, pro.daily_basic, start_date, end_date, "fetch daily_basic")
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_moneyflow(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取个股资金流向，用于补齐公开因子中的主力/大单资金字段。
    """
    cache_path = RAW_DIR / f"moneyflow_{start_date}_{end_date}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    fields = ",".join(
        [
            "ts_code",
            "trade_date",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ]
    )
    df = _fetch_by_trade_date(
        pro,
        lambda trade_date: pro.moneyflow(trade_date=trade_date, fields=fields),
        start_date,
        end_date,
        "fetch moneyflow",
    )
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_stock_company(
    pro,
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取上市公司基本信息，包括行业分类。
    """
    cache_path = RAW_DIR / "stock_company.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    if df.empty or not {"ts_code", "name", "industry"}.issubset(df.columns):
        frames = []
        for exchange in ("SZSE", "SSE", "BSE"):
            company = pro.stock_company(exchange=exchange)
            if not company.empty:
                frames.append(company)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    for col in ("ts_code", "industry", "name"):
        if col not in df.columns:
            df[col] = "未知" if col != "ts_code" else pd.NA
    df = df[["ts_code", "industry", "name"]].dropna(subset=["ts_code"]).copy()
    df["industry"] = df["industry"].fillna("未知")
    df["name"] = df["name"].fillna(df["ts_code"])

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_adj_factor(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    cache: bool = True,
) -> pd.DataFrame:
    """
    获取复权因子，用于计算复权价格。
    """
    cache_path = RAW_DIR / f"adj_factor_{start_date}_{end_date}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = _fetch_by_trade_date(pro, pro.adj_factor, start_date, end_date, "fetch adj_factor")
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_income(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    ts_codes: Optional[Iterable[str]] = None,
    cache_tag: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = RAW_DIR / _cache_name("income", start_date, end_date, cache_tag)
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    fields = ",".join(
        [
            "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
            "total_revenue", "revenue", "oper_cost", "total_cogs", "sell_exp", "admin_exp",
            "fin_exp", "rd_exp", "operate_profit", "total_profit", "n_income",
            "n_income_attr_p", "net_profit", "basic_eps", "diluted_eps",
        ]
    )
    ts_codes = list(ts_codes or _codes_from_index_cache("000300.SH", start_date, end_date))
    if not ts_codes:
        raise ValueError("fetch_income requires ts_codes or cached HS300 index_weight data")
    df = _fetch_by_ts_code(pro.income, ts_codes, start_date, end_date, "fetch income", fields=fields)
    if not df.empty:
        df = df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        if cache:
            df.to_parquet(cache_path, index=False)
    return df


def fetch_balancesheet(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    ts_codes: Optional[Iterable[str]] = None,
    cache_tag: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = RAW_DIR / _cache_name("balancesheet", start_date, end_date, cache_tag)
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    fields = ",".join(
        [
            "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
            "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
            "total_hldr_eqy_inc_min_int", "money_cap", "accounts_receiv", "inventories",
            "fix_assets", "intan_assets", "goodwill", "st_borr", "lt_borr",
            "notes_payable", "acct_payable",
        ]
    )
    ts_codes = list(ts_codes or _codes_from_index_cache("000300.SH", start_date, end_date))
    if not ts_codes:
        raise ValueError("fetch_balancesheet requires ts_codes or cached HS300 index_weight data")
    df = _fetch_by_ts_code(pro.balancesheet, ts_codes, start_date, end_date, "fetch balancesheet", fields=fields)
    if not df.empty:
        df = df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        if cache:
            df.to_parquet(cache_path, index=False)
    return df


def fetch_cashflow(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    ts_codes: Optional[Iterable[str]] = None,
    cache_tag: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = RAW_DIR / _cache_name("cashflow", start_date, end_date, cache_tag)
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    fields = ",".join(
        [
            "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
            "n_cashflow_act", "net_profit", "c_fr_sale_sg", "c_paid_to_for_empl",
            "c_paid_for_taxes", "free_cashflow", "im_net_cashflow_oper_act",
            "n_cash_flows_fnc_act", "n_cash_flows_inv_act", "c_pay_acq_const_fiolta",
        ]
    )
    ts_codes = list(ts_codes or _codes_from_index_cache("000300.SH", start_date, end_date))
    if not ts_codes:
        raise ValueError("fetch_cashflow requires ts_codes or cached HS300 index_weight data")
    df = _fetch_by_ts_code(pro.cashflow, ts_codes, start_date, end_date, "fetch cashflow", fields=fields)
    if not df.empty:
        df = df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        if cache:
            df.to_parquet(cache_path, index=False)
    return df


def fetch_fina_indicator(
    pro,
    start_date: str = "20180101",
    end_date: str = "20241231",
    ts_codes: Optional[Iterable[str]] = None,
    cache_tag: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = RAW_DIR / _cache_name("fina_indicator", start_date, end_date, cache_tag)
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    fields = ",".join(
        [
            "ts_code", "ann_date", "end_date", "roe", "roe_dt", "roa",
            "grossprofit_margin", "netprofit_margin", "op_yoy", "or_yoy",
            "netprofit_yoy", "ocf_to_or", "ocf_to_profit", "asset_turnover",
            "debt_to_assets", "current_ratio", "quick_ratio", "inv_turn",
            "ar_turn", "eps", "bps", "cfps",
        ]
    )
    ts_codes = list(ts_codes or _codes_from_index_cache("000300.SH", start_date, end_date))
    if not ts_codes:
        raise ValueError("fetch_fina_indicator requires ts_codes or cached HS300 index_weight data")
    df = _fetch_by_ts_code(pro.fina_indicator, ts_codes, start_date, end_date, "fetch fina_indicator", fields=fields)
    if not df.empty:
        df = df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        if cache:
            df.to_parquet(cache_path, index=False)
    return df


def run_fetch_all(
    token: Optional[str] = None,
    start_date: str = "20180101",
    end_date: str = "20241231",
    index_code: str = "000300.SH",
):
    """一键拉取全部所需原始数据。"""
    pro = get_tushare_pro(token)
    fetch_index_weight(pro, index_code=index_code, start_date=start_date, end_date=end_date)
    index_codes = _codes_from_index_cache(index_code, start_date, end_date)
    cache_tag = None if index_code == "000300.SH" else _index_tag(index_code)
    fetch_daily(pro, start_date=start_date, end_date=end_date)
    fetch_daily_basic(pro, start_date=start_date, end_date=end_date)
    fetch_moneyflow(pro, start_date=start_date, end_date=end_date)
    fetch_stock_company(pro)
    fetch_adj_factor(pro, start_date=start_date, end_date=end_date)
    fetch_income(pro, start_date=start_date, end_date=end_date, ts_codes=index_codes, cache_tag=cache_tag)
    fetch_balancesheet(pro, start_date=start_date, end_date=end_date, ts_codes=index_codes, cache_tag=cache_tag)
    fetch_cashflow(pro, start_date=start_date, end_date=end_date, ts_codes=index_codes, cache_tag=cache_tag)
    fetch_fina_indicator(pro, start_date=start_date, end_date=end_date, ts_codes=index_codes, cache_tag=cache_tag)
    print("✅ 全部原始数据下载完成。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="tushare token")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--index-code", default="000300.SH")
    args = parser.parse_args()
    run_fetch_all(token=args.token, start_date=args.start, end_date=args.end, index_code=args.index_code)
