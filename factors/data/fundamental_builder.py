"""
Point-in-time fundamental data builder.

Financial statements are low-frequency and must be aligned by announcement
date before they are combined with daily price/volume data.  This module keeps
the transformed fundamental panel separate from the existing processed splits
and writes augmented split files named ``train_fundamental.parquet`` etc.
"""

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from factors.alpha.operators import safe_div


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


INCOME_INTERVAL_FIELDS = [
    "total_revenue", "revenue", "oper_cost", "total_cogs", "sell_exp",
    "admin_exp", "fin_exp", "rd_exp", "operate_profit", "total_profit",
    "n_income", "n_income_attr_p", "net_profit",
]

CASHFLOW_INTERVAL_FIELDS = [
    "n_cashflow_act", "c_fr_sale_sg", "c_paid_to_for_empl", "c_paid_for_taxes",
    "free_cashflow", "im_net_cashflow_oper_act", "n_cash_flows_fnc_act",
    "n_cash_flows_inv_act", "c_pay_acq_const_fiolta",
]

BALANCE_FIELDS = [
    "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int", "money_cap", "accounts_receiv", "inventories",
    "fix_assets", "intan_assets", "goodwill", "st_borr", "lt_borr",
    "notes_payable", "acct_payable",
]

FINA_FIELDS = [
    "roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin",
    "op_yoy", "or_yoy", "netprofit_yoy", "ocf_to_or", "ocf_to_profit",
    "asset_turnover", "debt_to_assets", "current_ratio", "quick_ratio",
    "inv_turn", "ar_turn", "eps", "bps", "cfps",
]

VALUATION_FIELDS = [
    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
    "circ_mv", "turnover_rate_f", "volume_ratio",
]


def _index_tag(index_code: str) -> str:
    return index_code.replace(".", "_")


def _weight_col(index_code: str) -> str:
    if index_code == "000300.SH":
        return "index_weight"
    if index_code == "000905.SH":
        return "csi500_index_weight"
    return f"index_weight_{_index_tag(index_code)}"


def _read_raw(
    name: str,
    start_date: str,
    end_date: str,
    raw_suffix: str = "",
) -> pd.DataFrame:
    suffix = f"_{raw_suffix}" if raw_suffix else ""
    path = RAW_DIR / f"{name}{suffix}_{start_date}_{end_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw fundamental data: {path}")
    return pd.read_parquet(path)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("ann_date", "f_ann_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "f_ann_date" in df.columns:
        df["available_date"] = df["f_ann_date"].fillna(df["ann_date"])
    else:
        df["available_date"] = df["ann_date"]
    return df.dropna(subset=["ts_code", "end_date", "available_date"])


def _dedupe_first_announcement(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first announcement for each stock/report period."""
    if df.empty:
        return df
    df = _normalize_dates(df)
    df = df.sort_values(["ts_code", "end_date", "available_date"]).reset_index(drop=True)
    return df.drop_duplicates(["ts_code", "end_date"], keep="first")


def _add_quarterly_interval_features(
    df: pd.DataFrame,
    fields: Iterable[str],
) -> pd.DataFrame:
    """Convert cumulative statement fields to single-quarter, TTM, YoY and QoQ."""
    df = df.sort_values(["ts_code", "end_date"]).copy()
    df["_year"] = df["end_date"].dt.year
    df["_quarter"] = df["end_date"].dt.quarter

    for field in [f for f in fields if f in df.columns]:
        values = pd.to_numeric(df[field], errors="coerce")
        prev_cum = values.groupby([df["ts_code"], df["_year"]], sort=False).shift(1)
        single = values.where(df["_quarter"] == 1, values - prev_cum)
        df[f"{field}_q"] = single
        by_code = df.groupby("ts_code", sort=False)[f"{field}_q"]
        df[f"{field}_ttm"] = by_code.transform(lambda s: s.rolling(4, min_periods=4).sum())
        df[f"{field}_yoy"] = safe_div(single, by_code.shift(4)) - 1.0
        df[f"{field}_qoq"] = safe_div(single, by_code.shift(1)) - 1.0

    return df.drop(columns=["_year", "_quarter"])


def _pct_to_decimal(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    median_abs = s.abs().median()
    if pd.notna(median_abs) and median_abs > 2:
        return s / 100.0
    return s


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


def build_fundamental_features(
    start_date: str = "20180101",
    end_date: str = "20260706",
    raw_suffix: str = "",
) -> pd.DataFrame:
    income = _add_quarterly_interval_features(
        _dedupe_first_announcement(_read_raw("income", start_date, end_date, raw_suffix=raw_suffix)),
        INCOME_INTERVAL_FIELDS,
    )
    balance = _dedupe_first_announcement(_read_raw("balancesheet", start_date, end_date, raw_suffix=raw_suffix))
    cashflow = _add_quarterly_interval_features(
        _dedupe_first_announcement(_read_raw("cashflow", start_date, end_date, raw_suffix=raw_suffix)),
        CASHFLOW_INTERVAL_FIELDS,
    )
    fina = _dedupe_first_announcement(_read_raw("fina_indicator", start_date, end_date, raw_suffix=raw_suffix))

    income = income.rename(columns={"available_date": "income_ann_date"})
    balance = balance.rename(columns={"available_date": "balance_ann_date"})
    cashflow = cashflow.rename(columns={"available_date": "cashflow_ann_date"})
    fina = fina.rename(columns={"available_date": "fina_ann_date"})

    keep_income = ["ts_code", "end_date", "income_ann_date"] + [
        col for col in income.columns if col.endswith(("_q", "_ttm", "_yoy", "_qoq"))
    ]
    keep_balance = ["ts_code", "end_date", "balance_ann_date"] + [
        col for col in BALANCE_FIELDS if col in balance.columns
    ]
    keep_cashflow = ["ts_code", "end_date", "cashflow_ann_date"] + [
        col for col in cashflow.columns if col.endswith(("_q", "_ttm", "_yoy", "_qoq"))
    ]
    keep_fina = ["ts_code", "end_date", "fina_ann_date"] + [
        col for col in FINA_FIELDS if col in fina.columns
    ]

    df = income[keep_income].merge(balance[keep_balance], on=["ts_code", "end_date"], how="outer")
    df = df.merge(cashflow[keep_cashflow], on=["ts_code", "end_date"], how="outer")
    df = df.merge(fina[keep_fina], on=["ts_code", "end_date"], how="outer")

    ann_cols = [col for col in ["income_ann_date", "balance_ann_date", "cashflow_ann_date", "fina_ann_date"] if col in df.columns]
    df["available_date"] = df[ann_cols].max(axis=1)
    df = df.dropna(subset=["available_date"]).sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    revenue_ttm = _first_available(df, ["total_revenue_ttm", "revenue_ttm"])
    net_profit_ttm = _first_available(df, ["n_income_attr_p_ttm", "net_profit_ttm", "n_income_ttm"])
    ocf_ttm = _first_available(df, ["n_cashflow_act_ttm", "im_net_cashflow_oper_act_ttm"])
    equity = _first_available(df, ["total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"])
    total_assets = _num(df, "total_assets")

    df["total_revenue_ttm"] = revenue_ttm
    df["net_profit_ttm"] = net_profit_ttm
    df["n_cashflow_act_ttm"] = ocf_ttm
    df["roe_ttm"] = safe_div(net_profit_ttm, equity)
    df["roa_ttm"] = safe_div(net_profit_ttm, total_assets)
    df["gross_margin_ttm"] = safe_div(revenue_ttm - _num(df, "oper_cost_ttm"), revenue_ttm)
    df["net_margin_ttm"] = safe_div(net_profit_ttm, revenue_ttm)
    df["operating_cf_margin_ttm"] = safe_div(ocf_ttm, revenue_ttm)
    df["cashflow_to_profit"] = safe_div(ocf_ttm, net_profit_ttm)
    df["rd_expense_intensity"] = safe_div(_num(df, "rd_exp_ttm"), revenue_ttm)
    df["capex_to_assets"] = safe_div(_num(df, "c_pay_acq_const_fiolta_ttm"), total_assets)
    debt_calc = safe_div(_num(df, "total_liab"), total_assets)
    df["debt_to_assets"] = _num(df, "debt_to_assets").fillna(debt_calc)
    df["asset_turnover_ttm"] = safe_div(revenue_ttm, total_assets)
    df["free_cashflow_ttm"] = _num(df, "free_cashflow_ttm")
    df["cash_to_liab"] = safe_div(_num(df, "money_cap"), _num(df, "total_liab"))
    df["working_capital_pressure"] = safe_div(
        _num(df, "accounts_receiv") + _num(df, "inventories") - _num(df, "acct_payable"),
        total_assets,
    )

    df["revenue_yoy"] = _first_available(df, ["total_revenue_yoy", "revenue_yoy"])
    df["net_profit_yoy"] = _first_available(df, ["n_income_attr_p_yoy", "net_profit_yoy"])
    df["ocf_yoy"] = _first_available(df, ["n_cashflow_act_yoy", "im_net_cashflow_oper_act_yoy"])
    df["gross_margin_yoy"] = df.groupby("ts_code", sort=False)["gross_margin_ttm"].pct_change(4)
    df["asset_turnover_yoy"] = df.groupby("ts_code", sort=False)["asset_turnover_ttm"].pct_change(4)

    for col in ["roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin", "ocf_to_or", "ocf_to_profit", "debt_to_assets"]:
        if col in df.columns:
            df[col] = _pct_to_decimal(df[col])

    feature_cols = [
        "ts_code", "end_date", "available_date",
        "total_revenue_ttm", "net_profit_ttm", "n_cashflow_act_ttm",
        "total_assets", "total_liab", "roe_ttm", "roa_ttm", "gross_margin_ttm",
        "net_margin_ttm", "operating_cf_margin_ttm", "cashflow_to_profit",
        "rd_expense_intensity", "capex_to_assets", "debt_to_assets",
        "asset_turnover_ttm", "free_cashflow_ttm", "cash_to_liab",
        "working_capital_pressure", "money_cap", "accounts_receiv",
        "inventories", "acct_payable", "revenue_yoy", "net_profit_yoy",
        "ocf_yoy", "gross_margin_yoy", "asset_turnover_yoy",
    ] + [col for col in FINA_FIELDS if col in df.columns]
    feature_cols = list(dict.fromkeys([col for col in feature_cols if col in df.columns]))
    return df[feature_cols].sort_values(["ts_code", "available_date"]).reset_index(drop=True)


def asof_to_daily(daily_keys: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    daily_keys = daily_keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    daily_keys["trade_date"] = pd.to_datetime(daily_keys["trade_date"])
    fundamentals = fundamentals.copy()
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"])

    pieces = []
    fund_groups = dict(tuple(fundamentals.groupby("ts_code", sort=False)))
    for code, left in daily_keys.groupby("ts_code", sort=False):
        right = fund_groups.get(code)
        left = left.sort_values("trade_date")
        if right is None or right.empty:
            pieces.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right.sort_values("available_date"),
            left_on="trade_date",
            right_on="available_date",
            direction="backward",
        )
        if "ts_code_x" in merged.columns:
            merged = merged.rename(columns={"ts_code_x": "ts_code"}).drop(columns=["ts_code_y"], errors="ignore")
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _load_daily_basic_valuation(start_date: str, end_date: str) -> pd.DataFrame:
    path = RAW_DIR / f"daily_basic_{start_date}_{end_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = ["ts_code", "trade_date"] + VALUATION_FIELDS
    try:
        daily_basic = pd.read_parquet(path, columns=cols)
    except Exception:
        daily_basic = pd.read_parquet(path)
        cols = ["ts_code", "trade_date"] + [col for col in VALUATION_FIELDS if col in daily_basic.columns]
        daily_basic = daily_basic[cols].copy()
    daily_basic["trade_date"] = pd.to_datetime(daily_basic["trade_date"])
    return daily_basic


def _load_index_weight_daily(
    daily_keys: pd.DataFrame,
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    path = RAW_DIR / f"index_weight_{_index_tag(index_code)}_{start_date}_{end_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing index weight data: {path}")
    weights = pd.read_parquet(path).rename(columns={"con_code": "ts_code", "weight": _weight_col(index_code)})
    weights = weights[["ts_code", "trade_date", _weight_col(index_code)]].copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    keys = daily_keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])

    pieces = []
    weight_groups = dict(tuple(weights.groupby("ts_code", sort=False)))
    for code, left in keys.groupby("ts_code", sort=False):
        right = weight_groups.get(code)
        left = left.sort_values("trade_date")
        if right is None or right.empty:
            left[_weight_col(index_code)] = 0.0
            pieces.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right.sort_values("trade_date"),
            on="trade_date",
            direction="backward",
        )
        if "ts_code_x" in merged.columns:
            merged = merged.rename(columns={"ts_code_x": "ts_code"}).drop(columns=["ts_code_y"], errors="ignore")
        merged[_weight_col(index_code)] = merged[_weight_col(index_code)].fillna(0.0)
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def build_and_save(
    start_date: str = "20180101",
    end_date: str = "20260706",
    processed_dir: str | Path = PROCESSED_DIR,
    output_name: str = "fundamental_daily.parquet",
    merge_splits: bool = True,
    raw_suffix: str = "",
    split_suffix: str = "",
    index_code: str | None = None,
    index_universe_only: bool = False,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    fundamentals = build_fundamental_features(start_date=start_date, end_date=end_date, raw_suffix=raw_suffix)

    split_paths = {
        split: processed_dir / f"{split}.parquet"
        for split in ("train", "valid", "test")
        if (processed_dir / f"{split}.parquet").exists()
    }
    if not split_paths:
        raise FileNotFoundError(f"No processed split files found under {processed_dir}")

    daily_parts = []
    valuation = _load_daily_basic_valuation(start_date, end_date) if merge_splits else pd.DataFrame()
    if merge_splits:
        for split, path in split_paths.items():
            if index_code and index_universe_only:
                base_keys = pd.read_parquet(path, columns=["ts_code", "trade_date"]).drop_duplicates()
                index_daily = _load_index_weight_daily(base_keys, index_code, start_date, end_date)
                weight_col = _weight_col(index_code)
                index_daily = index_daily[index_daily[weight_col] > 0].copy()
                daily_keys = index_daily[["ts_code", "trade_date"]].drop_duplicates()
                daily_fundamental = asof_to_daily(daily_keys, fundamentals)
                daily_parts.append(daily_fundamental.merge(index_daily, on=["ts_code", "trade_date"], how="left"))

                base = pd.read_parquet(path)
                base = base.merge(index_daily, on=["ts_code", "trade_date"], how="inner")
                del base_keys, index_daily
            else:
                base = pd.read_parquet(path)
                daily_keys = base[["ts_code", "trade_date"]].drop_duplicates()
                daily_fundamental = asof_to_daily(daily_keys, fundamentals)
                if index_code:
                    index_daily = _load_index_weight_daily(daily_keys, index_code, start_date, end_date)
                    daily_fundamental = daily_fundamental.merge(index_daily, on=["ts_code", "trade_date"], how="left")
                    daily_fundamental[_weight_col(index_code)] = daily_fundamental[_weight_col(index_code)].fillna(0.0)
                daily_parts.append(daily_fundamental)

            augmented = base.merge(daily_fundamental, on=["ts_code", "trade_date"], how="left")
            if not valuation.empty:
                missing_valuation = [col for col in valuation.columns if col not in augmented.columns]
                if missing_valuation:
                    augmented = augmented.merge(
                        valuation[["ts_code", "trade_date"] + missing_valuation],
                        on=["ts_code", "trade_date"],
                        how="left",
                    )
            split_tag = f"_{split_suffix}" if split_suffix else ""
            augmented.to_parquet(processed_dir / f"{split}_fundamental{split_tag}.parquet", index=False)
            del base, daily_keys, daily_fundamental, augmented
            gc.collect()
    else:
        for path in split_paths.values():
            daily_keys = pd.read_parquet(path, columns=["ts_code", "trade_date"]).drop_duplicates()
            daily_fundamental = asof_to_daily(daily_keys, fundamentals)
            if index_code:
                index_daily = _load_index_weight_daily(daily_keys, index_code, start_date, end_date)
                daily_fundamental = daily_fundamental.merge(index_daily, on=["ts_code", "trade_date"], how="left")
                daily_fundamental[_weight_col(index_code)] = daily_fundamental[_weight_col(index_code)].fillna(0.0)
            daily_parts.append(daily_fundamental)
            del daily_keys, daily_fundamental
            gc.collect()

    daily_fundamental = (
        pd.concat(daily_parts, ignore_index=True)
        .drop_duplicates(["ts_code", "trade_date"])
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )
    output_path = processed_dir / output_name
    daily_fundamental.to_parquet(output_path, index=False)

    return daily_fundamental


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260706")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output-name", default="fundamental_daily.parquet")
    parser.add_argument("--raw-suffix", default="")
    parser.add_argument("--split-suffix", default="")
    parser.add_argument("--index-code", default=None)
    parser.add_argument("--index-universe-only", action="store_true")
    parser.add_argument("--no-merge-splits", action="store_true")
    args = parser.parse_args()

    out = build_and_save(
        start_date=args.start,
        end_date=args.end,
        processed_dir=args.processed_dir,
        output_name=args.output_name,
        raw_suffix=args.raw_suffix,
        split_suffix=args.split_suffix,
        index_code=args.index_code,
        index_universe_only=args.index_universe_only,
        merge_splits=not args.no_merge_splits,
    )
    print(
        "fundamental_daily rows="
        f"{len(out)}, stocks={out['ts_code'].nunique()}, "
        f"dates={out['trade_date'].nunique()}"
    )
