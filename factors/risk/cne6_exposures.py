"""CNE6-style descriptor and style exposure construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.risk.cne6_descriptors import descriptor_metadata, descriptor_specs


STYLE_FACTORS = [
    "size",
    "volatility",
    "liquidity",
    "momentum",
    "value",
    "growth",
    "quality",
    "dividend_yield",
    "sentiment",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(x: pd.Series, y: pd.Series, eps: float = 1e-12) -> pd.Series:
    out = x.astype(float) / y.astype(float).where(y.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_by_code(df: pd.DataFrame, values: pd.Series, window: int, func: str, min_periods: int | None = None) -> pd.Series:
    tmp = pd.DataFrame({"ts_code": df["ts_code"].to_numpy(), "_v": values.to_numpy()}, index=df.index)
    min_periods = min_periods or max(5, window // 2)
    grouped = tmp.groupby("ts_code", sort=False)["_v"]
    if func == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    if func == "sum":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    if func == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    if func == "max":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    if func == "min":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
    raise ValueError(f"unsupported rolling func: {func}")


def _pct_change(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    return df.groupby("ts_code", sort=False)[col].pct_change(periods)


def _shift(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    return df.groupby("ts_code", sort=False)[col].shift(periods)


def _rolling_beta(df: pd.DataFrame, window: int = 252) -> pd.Series:
    if not {"returns_1d", "csi500_return"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, dtype=float)

    def _one(sub: pd.DataFrame) -> pd.Series:
        ret = pd.to_numeric(sub["returns_1d"], errors="coerce")
        mkt = pd.to_numeric(sub["csi500_return"], errors="coerce")
        cov = ret.rolling(window, min_periods=max(40, window // 2)).cov(mkt)
        var = mkt.rolling(window, min_periods=max(40, window // 2)).var()
        return _safe_div(cov, var)

    return df.groupby("ts_code", group_keys=False, sort=False).apply(_one)


def _cs_winsor_zscore(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"trade_date": df["trade_date"].to_numpy(), "_v": pd.to_numeric(values, errors="coerce")}, index=df.index)
    grouped = tmp.groupby("trade_date", sort=False)["_v"]
    lower = grouped.transform(lambda s: s.quantile(0.01))
    upper = grouped.transform(lambda s: s.quantile(0.99))
    clipped = tmp["_v"].clip(lower=lower, upper=upper)
    mean = clipped.groupby(tmp["trade_date"], sort=False).transform("mean")
    std = clipped.groupby(tmp["trade_date"], sort=False).transform("std")
    out = _safe_div(clipped - mean, std)
    return out.mask(std.isna() | (std == 0), 0.0).mask(tmp["_v"].isna())


def _descriptor_raw_values(df: pd.DataFrame, cross_section_mask: pd.Series | None = None) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    total_mv = _num(df, "total_mv")
    log_mv = np.log(total_mv.where(total_mv > 0))
    out["log_total_mv"] = log_mv
    cs_group = df["trade_date"]
    cs_log_mv = log_mv if cross_section_mask is None else log_mv.where(cross_section_mask)
    mean_log_mv = cs_log_mv.groupby(cs_group, sort=False).transform("mean")
    log_mv_sq = log_mv**2
    mean_log_mv_sq = log_mv_sq.where(cross_section_mask) if cross_section_mask is not None else log_mv_sq
    mean_log_mv_sq = mean_log_mv_sq.groupby(cs_group, sort=False).transform("mean")
    centered_x = log_mv - mean_log_mv
    centered_y = log_mv_sq - mean_log_mv_sq
    cov_xy = (centered_x * centered_y).where(cross_section_mask) if cross_section_mask is not None else centered_x * centered_y
    var_x = (centered_x**2).where(cross_section_mask) if cross_section_mask is not None else centered_x**2
    cov_xy = cov_xy.groupby(cs_group, sort=False).transform("sum")
    var_x = var_x.groupby(cs_group, sort=False).transform("sum")
    slope = _safe_div(cov_xy, var_x)
    out["mid_cap_proxy"] = centered_y - slope * centered_x

    beta = _rolling_beta(df, 252)
    out["beta_252"] = beta
    ret = _num(df, "returns_1d")
    out["daily_std_252"] = _rolling_by_code(df, ret, 252, "std", min_periods=126)
    high_max = _rolling_by_code(df, _num(df, "high_adj"), 252, "max", min_periods=126)
    low_min = _rolling_by_code(df, _num(df, "low_adj"), 252, "min", min_periods=126)
    out["cumulative_range_252"] = np.log(_safe_div(high_max, low_min))
    resid = ret - beta * _num(df, "csi500_return")
    out["residual_volatility_proxy"] = _rolling_by_code(df, resid, 252, "std", min_periods=126)

    turnover = _num(df, "turnover_rate")
    out["avg_turnover_21"] = _rolling_by_code(df, turnover, 21, "mean", min_periods=10)
    out["avg_turnover_63"] = _rolling_by_code(df, turnover, 63, "mean", min_periods=21)
    out["avg_amount_21"] = np.log1p(_rolling_by_code(df, _num(df, "amount").clip(lower=0), 21, "mean", min_periods=10))
    out["turnover_stability_63"] = -_rolling_by_code(df, turnover, 63, "std", min_periods=21)

    out["ret_252_ex_21"] = _safe_div(_shift(df, "close_adj", 21), _shift(df, "close_adj", 252)) - 1.0 if "close_adj" in df.columns else pd.Series(np.nan, index=df.index)
    out["ret_126"] = _pct_change(df, "close_adj", 126) if "close_adj" in df.columns else pd.Series(np.nan, index=df.index)
    out["short_reversal_21"] = -(_pct_change(df, "close_adj", 21) if "close_adj" in df.columns else pd.Series(np.nan, index=df.index))

    out["book_to_price"] = _safe_div(pd.Series(1.0, index=df.index), _num(df, "pb"))
    out["earnings_yield"] = _safe_div(pd.Series(1.0, index=df.index), _num(df, "pe_ttm"))
    out["sales_to_price"] = _safe_div(pd.Series(1.0, index=df.index), _num(df, "ps_ttm"))
    out["cashflow_to_price"] = _safe_div(_num(df, "n_cashflow_act_ttm"), total_mv)

    out["revenue_yoy"] = _num(df, "revenue_yoy")
    out["net_profit_yoy"] = _num(df, "net_profit_yoy")
    out["roe_growth"] = _num(df, "roe_ttm") - df.groupby("ts_code", sort=False)["roe_ttm"].shift(252) if "roe_ttm" in df.columns else pd.Series(np.nan, index=df.index)
    out["asset_turnover_yoy"] = _num(df, "asset_turnover_yoy")

    out["roe_ttm"] = _num(df, "roe_ttm")
    out["roa_ttm"] = _num(df, "roa_ttm")
    out["gross_margin_ttm"] = _num(df, "gross_margin_ttm")
    out["cashflow_to_profit"] = _num(df, "cashflow_to_profit")
    out["low_leverage"] = -_num(df, "debt_to_assets")
    out["accrual_quality"] = _safe_div(_num(df, "n_cashflow_act_ttm") - _num(df, "net_profit_ttm"), _num(df, "total_assets"))

    out["dv_ttm"] = _num(df, "dv_ttm")
    out["dv_ratio"] = _num(df, "dv_ratio")

    out["analyst_report_count_90"] = _num(df, "analyst_report_count_90")
    out["analyst_org_count_180"] = _num(df, "analyst_org_count_180")
    out["analyst_rating_score_180"] = _num(df, "analyst_rating_score_180")
    out["analyst_target_upside_180"] = _num(df, "analyst_target_upside_180")
    out["analyst_eps_revision_180"] = _num(df, "analyst_eps_revision_180")

    return out


def compute_descriptor_exposures(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if "csi500_member" in df.columns:
        member_mask = df["csi500_member"].fillna(False).astype(bool)
    else:
        member_mask = pd.Series(True, index=df.index)
    target = df.loc[member_mask].copy()
    raw = _descriptor_raw_values(df, cross_section_mask=member_mask)
    specs = descriptor_specs()

    out = target[["trade_date", "ts_code"]].copy()
    for spec in specs:
        if spec.is_available(df.columns):
            out[spec.name] = _cs_winsor_zscore(target, raw[spec.name].loc[target.index])
        else:
            out[spec.name] = np.nan

    metadata = descriptor_metadata(df.columns)
    return out, metadata


def compute_style_exposures(descriptor_exposures: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out = descriptor_exposures[["trade_date", "ts_code"]].copy()
    for style in STYLE_FACTORS:
        desc = metadata.loc[(metadata["style"] == style) & (metadata["is_available"]), "descriptor"].tolist()
        if not desc:
            out[f"style_{style}"] = np.nan
            out[f"style_{style}_n"] = 0
            continue
        values = descriptor_exposures[desc].replace([np.inf, -np.inf], np.nan)
        out[f"style_{style}"] = values.mean(axis=1, skipna=True)
        out[f"style_{style}_n"] = values.notna().sum(axis=1)
        out.loc[out[f"style_{style}_n"] == 0, f"style_{style}"] = np.nan
        out[f"style_{style}"] = _cs_winsor_zscore(out, out[f"style_{style}"])
    return out


def compute_industry_exposures(panel: pd.DataFrame, industry_col: str = "industry") -> pd.DataFrame:
    out = panel[["trade_date", "ts_code"]].copy()
    industry = panel[industry_col].fillna("unknown").astype(str) if industry_col in panel.columns else pd.Series("unknown", index=panel.index)
    dummies = pd.get_dummies(industry, prefix="industry", dtype=float)
    return pd.concat([out.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def style_coverage_by_year(style: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = style.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    for col in [c for c in work.columns if c.startswith("style_") and not c.endswith("_n")]:
        grouped = work.groupby("year", sort=True)[col]
        for year, values in grouped:
            rows.append({"style": col, "year": int(year), "rows": int(len(values)), "coverage": float(values.notna().mean())})
    return pd.DataFrame(rows)


def run(panel_path: Path, output_dir: Path, history_panel_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    history_path = history_panel_path
    if history_path is None:
        candidate = panel_path.parent / "cne6_csi500_daily_history.parquet"
        history_path = candidate if candidate.exists() else panel_path
    feature_panel = pd.read_parquet(history_path)
    descriptors, metadata = compute_descriptor_exposures(feature_panel)
    style = compute_style_exposures(descriptors, metadata)
    industries = compute_industry_exposures(panel)

    descriptors.to_parquet(output_dir / "descriptor_exposures.parquet", index=False)
    style.to_parquet(output_dir / "style_exposures.parquet", index=False)
    industries.to_parquet(output_dir / "industry_exposures.parquet", index=False)
    metadata.to_csv(output_dir / "descriptor_metadata.csv", index=False)
    style_coverage_by_year(style).to_csv(output_dir / "style_coverage_by_year.csv", index=False)
    style_cols = [c for c in style.columns if c.startswith("style_") and not c.endswith("_n")]
    style[style_cols].corr().to_csv(output_dir / "style_correlation.csv")
    print(f"wrote CNE6-style exposures to {output_dir} using history={history_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--history-panel", default=None)
    parser.add_argument("--output", default="outputs/cne6_reproduction")
    args = parser.parse_args()
    run(Path(args.panel), Path(args.output), history_panel_path=Path(args.history_panel) if args.history_panel else None)


if __name__ == "__main__":
    main()
