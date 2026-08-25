"""Style exposure construction and active exposure diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.alpha import operators as op


STYLE_COLUMNS = [
    "style_size",
    "style_value",
    "style_momentum",
    "style_volatility",
    "style_liquidity",
    "style_quality",
    "style_leverage",
]


def _z(df: pd.DataFrame, values: pd.Series, name: str) -> pd.Series:
    tmp = df[["trade_date"]].copy()
    tmp[name] = values.replace([np.inf, -np.inf], np.nan)
    tmp[name] = op.cs_winsorize(tmp, name, 0.01, 0.99)
    return op.cs_zscore(tmp, name)


def compute_style_exposures(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["trade_date", "ts_code"]].copy()
    work = df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])

    if "log_mv" in work:
        out["style_size"] = _z(work, work["log_mv"], "_size")
    elif "total_mv" in work:
        out["style_size"] = _z(work, np.log(work["total_mv"].where(work["total_mv"] > 0)), "_size")
    else:
        out["style_size"] = 0.0

    inv_pe = op.safe_div(pd.Series(1.0, index=work.index), work["pe_ttm"]) if "pe_ttm" in work else 0.0
    inv_pb = op.safe_div(pd.Series(1.0, index=work.index), work["pb"]) if "pb" in work else 0.0
    out["style_value"] = _z(work, inv_pe + inv_pb, "_value")

    if "close_adj" in work:
        work["_ret20"] = op.ts_pct(work, "close_adj", 20)
        work["_ret60"] = op.ts_pct(work, "close_adj", 60)
        work["_ret1"] = op.ts_pct(work, "close_adj", 1)
        out["style_momentum"] = _z(work, 0.5 * work["_ret20"] + 0.5 * work["_ret60"], "_momentum")
        vol20 = op.ts_std(work, "_ret1", 20, min_periods=10)
        vol60 = op.ts_std(work, "_ret1", 60, min_periods=30)
        out["style_volatility"] = _z(work, 0.5 * vol20 + 0.5 * vol60, "_volatility")
    else:
        out["style_momentum"] = 0.0
        out["style_volatility"] = 0.0

    liquidity = pd.Series(0.0, index=work.index)
    if "amount" in work:
        liquidity = liquidity + np.log1p(work["amount"].clip(lower=0))
    if "turnover_rate" in work:
        liquidity = liquidity + work["turnover_rate"]
    out["style_liquidity"] = _z(work, liquidity, "_liquidity")

    quality = pd.Series(0.0, index=work.index)
    if "roe_ttm" in work:
        quality = quality + work["roe_ttm"]
    if "cashflow_to_profit" in work:
        quality = quality + work["cashflow_to_profit"]
    out["style_quality"] = _z(work, quality, "_quality")

    leverage = work["debt_to_assets"] if "debt_to_assets" in work else pd.Series(0.0, index=work.index)
    out["style_leverage"] = _z(work, leverage, "_leverage")

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def style_active_exposure(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark_col: str = "csi500_index_weight",
) -> pd.DataFrame:
    style_cols = [col for col in STYLE_COLUMNS if col in panel.columns]
    merged = weights.merge(panel[["trade_date", "ts_code", benchmark_col] + style_cols], on=["trade_date", "ts_code"], how="left")
    rows = []
    for date, sub in merged.groupby("trade_date", sort=False):
        bench = sub[benchmark_col].clip(lower=0).astype(float)
        if bench.sum() > 0:
            bench = bench / bench.sum()
        port = sub["weight"].fillna(0.0).astype(float)
        row = {"trade_date": date}
        for col in style_cols:
            row[col] = float((port * sub[col].fillna(0.0)).sum() - (bench * sub[col].fillna(0.0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)

