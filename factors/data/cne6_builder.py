"""Build a point-in-time CSI500 panel for the local CNE6-style risk model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factors.data.cne6_fetcher import RAW_DIR, OUTPUT_DIR, _index_tag, audit_dataframes


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


INCOME_INTERVAL_FIELDS = [
    "total_revenue", "revenue", "oper_cost", "total_cogs", "sell_exp", "admin_exp",
    "fin_exp", "rd_exp", "operate_profit", "total_profit", "n_income",
    "n_income_attr_p", "net_profit",
]
CASHFLOW_INTERVAL_FIELDS = [
    "n_cashflow_act", "c_fr_sale_sg", "c_paid_to_for_empl", "c_paid_for_taxes",
    "free_cashflow", "im_net_cashflow_oper_act", "n_cash_flows_fnc_act",
    "n_cash_flows_inv_act", "c_pay_acq_const_fiolta",
]
BALANCE_FIELDS = [
    "total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int",
    "total_cur_assets", "total_cur_liab",
    "money_cap", "accounts_receiv", "inventories", "fix_assets", "intan_assets", "goodwill",
    "st_borr", "lt_borr", "notes_payable", "acct_payable",
]
FINA_FIELDS = [
    "roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin", "op_yoy",
    "or_yoy", "netprofit_yoy", "ocf_to_or", "ocf_to_profit", "asset_turnover",
    "debt_to_assets", "current_ratio", "quick_ratio", "inv_turn", "ar_turn",
    "eps", "bps", "cfps",
]


def _safe_div(x: pd.Series, y: pd.Series, eps: float = 1e-12) -> pd.Series:
    out = x.astype(float) / y.astype(float).where(y.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def _gap_aware_price_returns(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate one-trading-day returns without bridging missing stock rows.

    A stock can be absent from the daily table because it was suspended or not
    yet listed.  A plain groupby ``pct_change`` would treat the next available
    row as a one-day return and inject a multi-day price move into the daily
    regression.  The global trading-date rank makes both backward and forward
    labels explicit while preserving the panel's existing row index.
    """

    dates = pd.Index(pd.to_datetime(panel["trade_date"], errors="coerce").dropna().unique()).sort_values()
    date_rank = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    rank = pd.to_datetime(panel["trade_date"], errors="coerce").map(date_rank)
    grouped = panel.groupby("ts_code", sort=False)
    prev_rank = grouped["trade_date"].shift(1).map(date_rank)
    next_rank = grouped["trade_date"].shift(-1).map(date_rank)
    prev_close = grouped["close_adj"].shift(1)
    next_close = grouped["close_adj"].shift(-1)

    returns = pd.to_numeric(panel["close_adj"], errors="coerce") / pd.to_numeric(prev_close, errors="coerce") - 1.0
    fwd = pd.to_numeric(next_close, errors="coerce") / pd.to_numeric(panel["close_adj"], errors="coerce") - 1.0
    returns = returns.where((rank - prev_rank) == 1.0)
    fwd = fwd.where((next_rank - rank) == 1.0)
    return returns, fwd


def _date_filter(df: pd.DataFrame, start: str, end: str, date_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[(out[date_col] >= pd.Timestamp(start)) & (out[date_col] <= pd.Timestamp(end))].copy()


def _read_cache_candidates(
    prefixes: Iterable[str],
    start: str,
    end: str,
    date_col: str = "trade_date",
    ts_codes: set[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for prefix in prefixes:
        paths = []
        exact_path = RAW_DIR / f"{prefix}.parquet"
        if exact_path.exists():
            paths.append(exact_path)
        for candidate in sorted(RAW_DIR.glob(f"{prefix}_*.parquet")):
            stem = candidate.stem
            if prefix in {"daily", "cne6_daily"} and stem.startswith(f"{prefix}_basic_"):
                continue
            paths.append(candidate)
        for path in paths:
            read_columns = columns
            if columns:
                try:
                    available_cols = set(pq.ParquetFile(path).schema.names)
                    read_columns = [col for col in columns if col in available_cols]
                    if "ts_code" in available_cols and "ts_code" not in read_columns:
                        read_columns.append("ts_code")
                    if date_col in available_cols and date_col not in read_columns:
                        read_columns.append(date_col)
                except Exception:
                    read_columns = columns
            try:
                filters = None
                if ts_codes and "stock_basic" not in prefix and "stock_company" not in prefix:
                    filters = [("ts_code", "in", sorted(ts_codes))]
                df = pd.read_parquet(path, columns=read_columns, filters=filters)
            except TypeError:
                try:
                    df = pd.read_parquet(path, columns=read_columns)
                except Exception:
                    continue
            except Exception:
                continue
            if ts_codes and "ts_code" in df.columns:
                df = df[df["ts_code"].isin(ts_codes)]
            df = _date_filter(df, start, end, date_col)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    for col in ["trade_date", "ann_date", "f_ann_date", "end_date", "list_date", "delist_date", "report_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "vol" in out.columns:
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(out["vol"])
            out = out.drop(columns=["vol"])
        else:
            out = out.rename(columns={"vol": "volume"})
    if "ts_code" in out.columns and date_col in out.columns:
        out = out.sort_values(["ts_code", date_col]).drop_duplicates(["ts_code", date_col], keep="last")
    elif "con_code" in out.columns and date_col in out.columns:
        out = out.sort_values(["con_code", date_col]).drop_duplicates(["con_code", date_col], keep="last")
    return out


def load_raw_inputs(start: str, end: str, index_code: str = "000905.SH") -> dict[str, pd.DataFrame]:
    tag = _index_tag(index_code)
    index_weight = _read_cache_candidates([f"cne6_index_weight_{tag}", f"index_weight_{tag}"], start, end, "trade_date")
    if not index_weight.empty:
        code_col = "con_code" if "con_code" in index_weight.columns else "ts_code"
        codes = set(index_weight[code_col].dropna().astype(str))
    else:
        codes = None

    daily_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "volume", "amount"]
    basic_cols = [
        "ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_mv", "circ_mv",
    ]
    money_cols = ["ts_code", "trade_date", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]
    report_rc_cols = [
        "ts_code", "report_date", "report_title", "report_type", "classify",
        "org_name", "author_name", "quarter", "op_rt", "op_pr", "tp", "np",
        "eps", "pe", "rd", "roe", "ev_ebitda", "rating", "max_price",
        "min_price", "imp_dg", "create_time",
    ]
    return {
        "daily": _read_cache_candidates(["cne6_daily", "daily"], start, end, "trade_date", ts_codes=codes, columns=daily_cols),
        "adj_factor": _read_cache_candidates(["cne6_adj_factor", "adj_factor"], start, end, "trade_date", ts_codes=codes, columns=["ts_code", "trade_date", "adj_factor"]),
        "daily_basic": _read_cache_candidates(["cne6_daily_basic", "daily_basic"], start, end, "trade_date", ts_codes=codes, columns=basic_cols),
        "moneyflow": _read_cache_candidates(["cne6_moneyflow", "moneyflow"], start, end, "trade_date", ts_codes=codes, columns=money_cols),
        "report_rc": _read_cache_candidates(["cne6_report_rc", "report_rc"], start, end, "report_date", ts_codes=codes, columns=report_rc_cols),
        f"index_weight_{tag}": index_weight,
        "income": _read_cache_candidates(["cne6_income", f"income_{tag}", "income"], start, end, "ann_date", ts_codes=codes),
        "balancesheet": _read_cache_candidates(["cne6_balancesheet", f"balancesheet_{tag}", "balancesheet"], start, end, "ann_date", ts_codes=codes),
        "cashflow": _read_cache_candidates(["cne6_cashflow", f"cashflow_{tag}", "cashflow"], start, end, "ann_date", ts_codes=codes),
        "fina_indicator": _read_cache_candidates(["cne6_fina_indicator", f"fina_indicator_{tag}", "fina_indicator"], start, end, "ann_date", ts_codes=codes),
        "stock_basic": _read_cache_candidates(["cne6_stock_basic", "stock_company"], "19000101", end, "list_date"),
    }


def _normalize_statement_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("ann_date", "f_ann_date", "end_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "f_ann_date" in out.columns:
        out["available_date"] = out["f_ann_date"].fillna(out["ann_date"])
    else:
        out["available_date"] = out["ann_date"]
    return out.dropna(subset=["ts_code", "end_date", "available_date"])


def _dedupe_first_announcement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _normalize_statement_dates(df)
    out = out.sort_values(["ts_code", "end_date", "available_date"]).reset_index(drop=True)
    return out.drop_duplicates(["ts_code", "end_date"], keep="first")


def _add_quarterly_interval_features(df: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["ts_code", "end_date"]).copy()
    out["_year"] = out["end_date"].dt.year
    out["_quarter"] = out["end_date"].dt.quarter
    for field in [f for f in fields if f in out.columns]:
        values = pd.to_numeric(out[field], errors="coerce")
        prev_cum = values.groupby([out["ts_code"], out["_year"]], sort=False).shift(1)
        single = values.where(out["_quarter"] == 1, values - prev_cum)
        out[f"{field}_q"] = single
        by_code = out.groupby("ts_code", sort=False)[f"{field}_q"]
        out[f"{field}_ttm"] = by_code.transform(lambda s: s.rolling(4, min_periods=4).sum())
        out[f"{field}_yoy"] = _safe_div(single, by_code.shift(4)) - 1.0
    return out.drop(columns=["_year", "_quarter"])


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _first_available(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def _pct_to_decimal(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.abs().median()
    return s / 100.0 if pd.notna(med) and med > 2 else s


def build_fundamental_features(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    income = _add_quarterly_interval_features(_dedupe_first_announcement(inputs.get("income", pd.DataFrame())), INCOME_INTERVAL_FIELDS)
    balance = _dedupe_first_announcement(inputs.get("balancesheet", pd.DataFrame()))
    cashflow = _add_quarterly_interval_features(_dedupe_first_announcement(inputs.get("cashflow", pd.DataFrame())), CASHFLOW_INTERVAL_FIELDS)
    fina = _dedupe_first_announcement(inputs.get("fina_indicator", pd.DataFrame()))

    if income.empty and balance.empty and cashflow.empty and fina.empty:
        return pd.DataFrame(columns=["ts_code", "available_date"])

    income = income.rename(columns={"available_date": "income_ann_date"})
    balance = balance.rename(columns={"available_date": "balance_ann_date"})
    cashflow = cashflow.rename(columns={"available_date": "cashflow_ann_date"})
    fina = fina.rename(columns={"available_date": "fina_ann_date"})

    keep_income = ["ts_code", "end_date", "income_ann_date"] + [c for c in income.columns if c.endswith(("_q", "_ttm", "_yoy"))]
    keep_balance = ["ts_code", "end_date", "balance_ann_date"] + [c for c in BALANCE_FIELDS if c in balance.columns]
    keep_cashflow = ["ts_code", "end_date", "cashflow_ann_date"] + [c for c in cashflow.columns if c.endswith(("_q", "_ttm", "_yoy"))]
    keep_fina = ["ts_code", "end_date", "fina_ann_date"] + [c for c in FINA_FIELDS if c in fina.columns]

    frames = []
    if not income.empty:
        frames.append(income[[c for c in keep_income if c in income.columns]])
    if not balance.empty:
        frames.append(balance[[c for c in keep_balance if c in balance.columns]])
    if not cashflow.empty:
        frames.append(cashflow[[c for c in keep_cashflow if c in cashflow.columns]])
    if not fina.empty:
        frames.append(fina[[c for c in keep_fina if c in fina.columns]])
    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on=["ts_code", "end_date"], how="outer")

    ann_cols = [c for c in ["income_ann_date", "balance_ann_date", "cashflow_ann_date", "fina_ann_date"] if c in df.columns]
    df["available_date"] = df[ann_cols].max(axis=1)
    df = df.dropna(subset=["available_date"]).sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    revenue_ttm = _first_available(df, ["total_revenue_ttm", "revenue_ttm"])
    net_profit_ttm = _first_available(df, ["n_income_attr_p_ttm", "net_profit_ttm", "n_income_ttm"])
    ocf_ttm = _first_available(df, ["n_cashflow_act_ttm", "im_net_cashflow_oper_act_ttm"])
    total_assets = _num(df, "total_assets")
    total_liab = _num(df, "total_liab")
    equity = _first_available(df, ["total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"])

    df["total_revenue_ttm"] = revenue_ttm
    df["net_profit_ttm"] = net_profit_ttm
    df["n_cashflow_act_ttm"] = ocf_ttm
    df["roe_ttm"] = _safe_div(net_profit_ttm, equity)
    df["roa_ttm"] = _safe_div(net_profit_ttm, total_assets)
    df["gross_margin_ttm"] = _safe_div(revenue_ttm - _num(df, "oper_cost_ttm"), revenue_ttm)
    df["operating_margin_ttm"] = _safe_div(_num(df, "operate_profit_ttm"), revenue_ttm)
    df["net_margin_ttm"] = _safe_div(net_profit_ttm, revenue_ttm)
    df["operating_cf_margin_ttm"] = _safe_div(ocf_ttm, revenue_ttm)
    df["cashflow_to_profit"] = _safe_div(ocf_ttm, net_profit_ttm)
    df["debt_to_assets"] = _num(df, "debt_to_assets").fillna(_safe_div(total_liab, total_assets))
    df["asset_turnover_ttm"] = _safe_div(revenue_ttm, total_assets)
    df["free_cashflow_ttm"] = _num(df, "free_cashflow_ttm")
    df["revenue_yoy"] = _first_available(df, ["total_revenue_yoy", "revenue_yoy"])
    df["net_profit_yoy"] = _first_available(df, ["n_income_attr_p_yoy", "net_profit_yoy"])
    df["asset_turnover_yoy"] = df.groupby("ts_code", sort=False)["asset_turnover_ttm"].pct_change(4)
    by_code = df.groupby("ts_code", sort=False)
    working_capital = _num(df, "total_cur_assets") - _num(df, "total_cur_liab")
    capex_ttm = _num(df, "c_pay_acq_const_fiolta_ttm")
    eps = _num(df, "eps")
    df["eps_growth"] = _safe_div(eps, eps.groupby(df["ts_code"], sort=False).shift(4)) - 1.0
    df["roe_growth"] = df["roe_ttm"] - by_code["roe_ttm"].shift(4)
    df["asset_growth"] = _safe_div(
        total_assets,
        total_assets.groupby(df["ts_code"], sort=False).shift(4),
    ) - 1.0
    df["capex_growth"] = _safe_div(capex_ttm, capex_ttm.groupby(df["ts_code"], sort=False).shift(4)) - 1.0
    inventories = _num(df, "inventories")
    df["inventory_growth"] = _safe_div(inventories, inventories.groupby(df["ts_code"], sort=False).shift(4)) - 1.0
    df["working_capital_growth"] = _safe_div(
        working_capital,
        working_capital.groupby(df["ts_code"], sort=False).shift(4),
    ) - 1.0
    df["book_leverage"] = _safe_div(total_assets, equity)
    df["inverse_interest_coverage"] = _safe_div(_num(df, "fin_exp_ttm"), _num(df, "operate_profit_ttm"))
    reported_roa = _pct_to_decimal(_num(df, "roa"))
    df["earnings_stability"] = -reported_roa.groupby(df["ts_code"], sort=False).transform(
        lambda values: values.rolling(8, min_periods=8).std()
    )

    for col in ["roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin", "ocf_to_or", "ocf_to_profit", "debt_to_assets"]:
        if col in df.columns:
            df[col] = _pct_to_decimal(df[col])

    keep = [
        "ts_code", "end_date", "available_date", "total_revenue_ttm", "net_profit_ttm",
        "n_cashflow_act_ttm", "total_assets", "total_liab", "roe_ttm", "roa_ttm",
        "gross_margin_ttm", "net_margin_ttm", "operating_cf_margin_ttm",
        "operating_margin_ttm", "cashflow_to_profit", "debt_to_assets", "asset_turnover_ttm",
        "free_cashflow_ttm", "revenue_yoy", "net_profit_yoy", "asset_turnover_yoy",
        "eps_growth", "roe_growth", "asset_growth", "capex_growth", "inventory_growth",
        "working_capital_growth", "book_leverage", "inverse_interest_coverage", "earnings_stability",
    ] + [c for c in FINA_FIELDS if c in df.columns]
    return df[[c for c in dict.fromkeys(keep) if c in df.columns]].sort_values(["ts_code", "available_date"]).reset_index(drop=True)


def asof_to_daily(daily_keys: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    keys = daily_keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    if fundamentals.empty:
        return keys
    funds = fundamentals.copy()
    funds["available_date"] = pd.to_datetime(funds["available_date"])
    pieces = []
    fund_groups = dict(tuple(funds.groupby("ts_code", sort=False)))
    for code, left in keys.groupby("ts_code", sort=False):
        right = fund_groups.get(code)
        left = left.sort_values("trade_date")
        if right is None or right.empty:
            pieces.append(left)
            continue
        merged = pd.merge_asof(left, right.sort_values("available_date"), left_on="trade_date", right_on="available_date", direction="backward")
        if "ts_code_x" in merged.columns:
            merged = merged.rename(columns={"ts_code_x": "ts_code"}).drop(columns=["ts_code_y"], errors="ignore")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


ANALYST_FEATURE_COLUMNS = [
    "analyst_report_count_90",
    "analyst_org_count_180",
    "analyst_rating_score_180",
    "analyst_target_upside_180",
    "analyst_eps_revision_180",
    "analyst_forward_eps_180",
]


def _rating_to_score(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    positive_strong = ["强烈推荐", "强推", "买入", "推荐", "跑赢大市", "跑赢行业", "优于大市", "优于行业", "buy", "outperform"]
    positive_mid = ["增持", "审慎增持", "谨慎推荐", "overweight", "add"]
    neutral = ["中性", "持有", "同步大市", "同步行业", "无评级", "未评级", "neutral", "hold", "market perform"]
    negative_mid = ["减持", "低于大市", "弱于大市", "低于行业", "弱于行业", "underweight", "reduce", "underperform"]
    negative_strong = ["卖出", "回避", "sell"]
    if any(item in text for item in positive_strong):
        return 1.0
    if any(item in text for item in positive_mid):
        return 0.5
    if any(item in text for item in neutral):
        return 0.0
    if any(item in text for item in negative_strong):
        return -1.0
    if any(item in text for item in negative_mid):
        return -0.5
    return np.nan


def _safe_nanmean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _safe_nanmedian(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _window_mean(values: np.ndarray, start_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    counts = np.r_[0, np.cumsum(valid.astype(int))]
    win_sum = sums[end_idx] - sums[start_idx]
    win_count = counts[end_idx] - counts[start_idx]
    out = np.full(len(end_idx), np.nan, dtype=float)
    ok = win_count > 0
    out[ok] = win_sum[ok] / win_count[ok]
    return out


def _empty_analyst_features(keys: pd.DataFrame) -> pd.DataFrame:
    out = keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    for col in ANALYST_FEATURE_COLUMNS:
        out[col] = np.nan
    return out


def build_analyst_sentiment_features(daily: pd.DataFrame, report_rc: pd.DataFrame) -> pd.DataFrame:
    keys = daily[["ts_code", "trade_date"]].drop_duplicates().copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    if report_rc.empty or not {"ts_code", "report_date"}.issubset(report_rc.columns):
        return _empty_analyst_features(keys)

    reports = report_rc.copy()
    reports["report_date"] = pd.to_datetime(reports["report_date"], errors="coerce")
    reports["available_date"] = reports["report_date"] + pd.Timedelta(days=1)
    reports = reports.dropna(subset=["ts_code", "available_date"]).sort_values(["ts_code", "available_date"]).reset_index(drop=True)
    if reports.empty:
        return _empty_analyst_features(keys)

    for col in ["eps", "max_price", "min_price"]:
        reports[col] = pd.to_numeric(reports[col], errors="coerce") if col in reports.columns else np.nan
    reports["rating_score"] = reports["rating"].map(_rating_to_score) if "rating" in reports.columns else np.nan
    target_cols = [col for col in ["max_price", "min_price"] if col in reports.columns]
    reports["target_price_mid"] = reports[target_cols].mean(axis=1, skipna=True) if target_cols else np.nan

    price_col = "close" if "close" in daily.columns else "close_adj" if "close_adj" in daily.columns else None
    daily_cols = ["ts_code", "trade_date"] + ([price_col] if price_col else [])
    daily_work = daily[daily_cols].drop_duplicates(["ts_code", "trade_date"]).copy()
    daily_work["trade_date"] = pd.to_datetime(daily_work["trade_date"])
    if price_col:
        daily_work["_close_for_target"] = pd.to_numeric(daily_work[price_col], errors="coerce")
    else:
        daily_work["_close_for_target"] = np.nan

    report_groups = dict(tuple(reports.groupby("ts_code", sort=False)))
    pieces = []
    for code, left in daily_work.groupby("ts_code", sort=False):
        left = left.sort_values("trade_date").copy()
        right = report_groups.get(code)
        if right is None or right.empty:
            for col in ANALYST_FEATURE_COLUMNS:
                left[col] = np.nan
            pieces.append(left[["ts_code", "trade_date"] + ANALYST_FEATURE_COLUMNS])
            continue

        right = right.sort_values("available_date").reset_index(drop=True)
        event_dates = right["available_date"].to_numpy(dtype="datetime64[ns]")
        trade_dates = left["trade_date"].to_numpy(dtype="datetime64[ns]")
        close = left["_close_for_target"].to_numpy(dtype=float)
        org = right["org_name"].astype("object").to_numpy() if "org_name" in right.columns else np.array([None] * len(right), dtype=object)
        rating = pd.to_numeric(right["rating_score"], errors="coerce").to_numpy(dtype=float)
        target = pd.to_numeric(right["target_price_mid"], errors="coerce").to_numpy(dtype=float)
        eps = pd.to_numeric(right["eps"], errors="coerce").to_numpy(dtype=float)

        end_idx = np.searchsorted(event_dates, trade_dates, side="right")
        start_90 = np.searchsorted(event_dates, trade_dates - np.timedelta64(90, "D"), side="left")
        start_180 = np.searchsorted(event_dates, trade_dates - np.timedelta64(180, "D"), side="left")
        start_360 = np.searchsorted(event_dates, trade_dates - np.timedelta64(360, "D"), side="left")

        report_count_90 = (end_idx - start_90).astype(float)
        org_count_180 = np.zeros(len(left), dtype=float)
        for org_name in pd.Series(org).dropna().unique():
            org_dates = event_dates[org == org_name]
            org_start = np.searchsorted(org_dates, trade_dates - np.timedelta64(180, "D"), side="left")
            org_end = np.searchsorted(org_dates, trade_dates, side="right")
            org_count_180 += (org_end > org_start).astype(float)

        rating_score_180 = _window_mean(rating, start_180, end_idx)
        target_mean_180 = _window_mean(target, start_180, end_idx)
        target_upside_180 = _safe_div(pd.Series(target_mean_180, index=left.index), left["_close_for_target"]).to_numpy(dtype=float)
        target_upside_180 = target_upside_180 - 1.0

        eps_recent = _window_mean(eps, start_180, end_idx)
        eps_prior = _window_mean(eps, start_360, start_180)
        eps_revision_180 = _safe_div(pd.Series(eps_recent, index=left.index), pd.Series(eps_prior, index=left.index)).to_numpy(dtype=float) - 1.0

        out = left[["ts_code", "trade_date"]].copy()
        out["analyst_report_count_90"] = report_count_90
        out["analyst_org_count_180"] = org_count_180
        out["analyst_rating_score_180"] = rating_score_180
        out["analyst_target_upside_180"] = target_upside_180
        out["analyst_eps_revision_180"] = eps_revision_180
        out["analyst_forward_eps_180"] = eps_recent
        pieces.append(out)

    return pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _index_weight_daily(daily_keys: pd.DataFrame, index_weight: pd.DataFrame, index_code: str) -> pd.DataFrame:
    weight_col = "csi500_index_weight" if index_code == "000905.SH" else f"index_weight_{_index_tag(index_code)}"
    if index_weight.empty:
        out = daily_keys[["ts_code", "trade_date"]].drop_duplicates().copy()
        out[weight_col] = 0.0
        return out
    weights = index_weight.rename(columns={"con_code": "ts_code", "weight": weight_col})[["ts_code", "trade_date", weight_col]].copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    keys = daily_keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    date_map = pd.DataFrame({"trade_date": sorted(keys["trade_date"].unique())})
    snapshots = pd.DataFrame({"snapshot_date": sorted(weights["trade_date"].dropna().unique())})
    date_map = pd.merge_asof(
        date_map.sort_values("trade_date"),
        snapshots.sort_values("snapshot_date"),
        left_on="trade_date",
        right_on="snapshot_date",
        direction="backward",
    ).dropna(subset=["snapshot_date"])
    pieces = []
    for snapshot_date, dates in date_map.groupby("snapshot_date", sort=False):
        snap = weights[weights["trade_date"] == snapshot_date][["ts_code", weight_col]].copy()
        if snap.empty:
            continue
        date_frame = dates[["trade_date"]].copy()
        snap["_key"] = 1
        date_frame["_key"] = 1
        expanded = snap.merge(date_frame, on="_key", how="inner").drop(columns="_key")
        pieces.append(expanded)
    if not pieces:
        keys[weight_col] = 0.0
        return keys
    expanded_weights = pd.concat(pieces, ignore_index=True)
    out = keys.merge(expanded_weights, on=["ts_code", "trade_date"], how="left")
    out[weight_col] = out[weight_col].fillna(0.0)
    return out


def build_cne6_panel(
    start: str = "20100101",
    end: str = "20260722",
    index_code: str = "000905.SH",
    include_non_members: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time panel while preserving a full history internally.

    Returns are calculated before the daily index-membership filter.  This is
    important for stocks that leave and later re-enter the index: their price
    change is still a one-trading-day return, rather than a multi-month jump
    caused by missing rows in the filtered panel.
    """
    inputs = load_raw_inputs(start, end, index_code=index_code)
    tag = _index_tag(index_code)
    daily = inputs["daily"]
    adj = inputs["adj_factor"]
    basic = inputs["daily_basic"]
    index_weight = inputs[f"index_weight_{tag}"]
    if daily.empty or adj.empty or index_weight.empty:
        missing = [name for name, df in [("daily", daily), ("adj_factor", adj), (f"index_weight_{tag}", index_weight)] if df.empty]
        raise FileNotFoundError(f"Missing required CNE6 raw inputs: {missing}")

    codes = set(index_weight["con_code"].dropna().astype(str)) if "con_code" in index_weight.columns else set(index_weight["ts_code"].dropna().astype(str))
    daily = daily[daily["ts_code"].isin(codes)].copy()
    adj = adj[adj["ts_code"].isin(codes)].copy()
    basic = basic[basic["ts_code"].isin(codes)].copy() if not basic.empty else basic

    if "volume" not in daily.columns and "vol" in daily.columns:
        daily = daily.rename(columns={"vol": "volume"})
    panel = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    for col in ["open", "high", "low", "close"]:
        panel[f"{col}_adj"] = pd.to_numeric(panel[col], errors="coerce") * pd.to_numeric(panel["adj_factor"], errors="coerce")

    basic_cols = [
        "ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
        "total_mv", "circ_mv",
    ]
    if not basic.empty:
        panel = panel.merge(basic[[c for c in basic_cols if c in basic.columns]], on=["ts_code", "trade_date"], how="left")

    money = inputs["moneyflow"]
    if not money.empty:
        money_cols = ["ts_code", "trade_date", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]
        panel = panel.merge(money[[c for c in money_cols if c in money.columns]], on=["ts_code", "trade_date"], how="left")

    stock_basic = inputs["stock_basic"]
    if not stock_basic.empty:
        cols = [c for c in ["ts_code", "name", "industry", "market", "exchange", "list_date", "list_status"] if c in stock_basic.columns]
        panel = panel.merge(stock_basic[cols].drop_duplicates("ts_code"), on="ts_code", how="left")
    if "industry" not in panel.columns:
        panel["industry"] = "unknown"
    panel["industry"] = panel["industry"].fillna("unknown")

    weights_daily = _index_weight_daily(panel[["ts_code", "trade_date"]], index_weight, index_code)
    panel = panel.merge(weights_daily, on=["ts_code", "trade_date"], how="left")
    weight_col = "csi500_index_weight" if index_code == "000905.SH" else f"index_weight_{tag}"
    panel[weight_col] = panel[weight_col].fillna(0.0)
    panel["csi500_member"] = (panel[weight_col] > 0).astype(bool)

    analyst_features = build_analyst_sentiment_features(panel[["ts_code", "trade_date", "close"]], inputs.get("report_rc", pd.DataFrame()))
    panel = panel.merge(analyst_features, on=["ts_code", "trade_date"], how="left")

    fundamentals = build_fundamental_features(inputs)
    daily_fundamentals = asof_to_daily(panel[["ts_code", "trade_date"]], fundamentals)
    panel = panel.merge(daily_fundamentals, on=["ts_code", "trade_date"], how="left")

    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel["returns_1d"], panel["fwd_1d_return"] = _gap_aware_price_returns(panel)
    panel["log_mv"] = np.log(pd.to_numeric(panel["total_mv"], errors="coerce").where(pd.to_numeric(panel["total_mv"], errors="coerce") > 0))

    bench = panel.loc[panel["csi500_member"], ["trade_date", "ts_code", weight_col, "returns_1d"]].dropna(subset=["returns_1d"]).copy()
    bench["_w"] = pd.to_numeric(bench[weight_col], errors="coerce").clip(lower=0)
    bench["_w"] = bench["_w"] / bench.groupby("trade_date", sort=False)["_w"].transform("sum")
    bench_ret = bench.groupby("trade_date", sort=False).apply(lambda s: float((s["_w"] * s["returns_1d"]).sum())).rename("csi500_return").reset_index()
    panel = panel.merge(bench_ret, on="trade_date", how="left")
    if not include_non_members:
        panel = panel.loc[panel["csi500_member"]].copy()
    return panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def run(start: str, end: str, index_code: str, output_name: str = "cne6_csi500_daily_panel.parquet") -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = build_cne6_panel(start=start, end=end, index_code=index_code, include_non_members=True)
    history_path = PROCESSED_DIR / "cne6_csi500_daily_history.parquet"
    history.to_parquet(history_path, index=False)
    panel = history.loc[history["csi500_member"]].copy()
    out_path = PROCESSED_DIR / output_name
    panel.to_parquet(out_path, index=False)
    audit_dataframes(
        {
            "cne6_csi500_daily_panel": panel,
            "cne6_csi500_daily_history": history,
        }
    ).to_csv(OUTPUT_DIR / "panel_availability.csv", index=False)
    print(f"wrote CNE6 CSI500 panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()} to {out_path}")
    print(f"wrote continuous history rows={len(history)}, stocks={history['ts_code'].nunique()} to {history_path}")
    return panel


def build_cne6_v2_panels(
    start: str,
    end: str,
    index_code: str = "000905.SH",
    industry_members: pd.DataFrame | None = None,
    industry_coverage_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build versioned history/member panels with PIT SW2021 L1 industries."""

    from factors.data.cne6_industry import (
        SW_MEMBERS_FILE,
        attach_pit_industry,
        industry_coverage_audit,
    )

    if industry_members is None:
        members_path = RAW_DIR / SW_MEMBERS_FILE
        if not members_path.exists():
            raise FileNotFoundError(
                f"Missing SW2021 membership cache: {members_path}. "
                "Run factors.data.cne6_industry first."
            )
        industry_members = pd.read_parquet(members_path)

    history = build_cne6_panel(
        start=start,
        end=end,
        index_code=index_code,
        include_non_members=True,
    )
    history_v2 = attach_pit_industry(history, industry_members)
    panel_v2 = history_v2.loc[history_v2["csi500_member"]].copy()
    matched = panel_v2["industry_sw_l1_code"].notna()
    coverage = float(matched.mean()) if len(panel_v2) else 0.0
    if coverage < industry_coverage_threshold:
        missing = panel_v2.loc[~matched, ["trade_date", "ts_code"]].head(20).to_dict("records")
        raise ValueError(
            f"SW2021 industry coverage {coverage:.4%} is below required "
            f"{industry_coverage_threshold:.4%}; sample={missing}"
        )
    audit = industry_coverage_audit(panel_v2)
    return history_v2, panel_v2, audit


def run_v2(
    start: str,
    end: str,
    index_code: str = "000905.SH",
    history_output_name: str = "cne6_csi500_daily_history_v2.parquet",
    panel_output_name: str = "cne6_csi500_daily_panel_v2.parquet",
) -> pd.DataFrame:
    """Write V2 panels without touching the legacy panel artifacts."""

    history, panel, audit = build_cne6_v2_panels(start, end, index_code=index_code)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    v2_output = ROOT_DIR / "outputs" / "cne6_enhanced_v2"
    v2_output.mkdir(parents=True, exist_ok=True)
    history.to_parquet(PROCESSED_DIR / history_output_name, index=False)
    panel.to_parquet(PROCESSED_DIR / panel_output_name, index=False)
    audit.to_csv(v2_output / "industry_mapping_audit.csv", index=False)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="20260722")
    parser.add_argument("--index-code", default="000905.SH")
    parser.add_argument("--output-name", default="cne6_csi500_daily_panel.parquet")
    args = parser.parse_args()
    run(args.start, args.end, args.index_code, args.output_name)


if __name__ == "__main__":
    main()
