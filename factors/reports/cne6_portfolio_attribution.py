"""Portfolio risk attribution using local CNE6-style risk model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_weights(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce").clip(lower=0).fillna(0.0)
    total = values.sum()
    if total <= 0:
        return pd.Series(0.0, index=s.index)
    return values / total


def _latest_by_date(df: pd.DataFrame, date_col: str, target_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col].dropna().unique())
    dates = dates[dates <= target_date]
    if len(dates) == 0:
        return df.iloc[0:0].copy()
    return df[pd.to_datetime(df[date_col]) == dates.max()].copy()


def _covariance_matrix(covariance: pd.DataFrame, date: pd.Timestamp, window: int, factors: list[str]) -> pd.DataFrame:
    sub = covariance[(pd.to_datetime(covariance["trade_date"]) <= date) & (covariance["window"] == window)].copy()
    if sub.empty:
        return pd.DataFrame(0.0, index=factors, columns=factors)
    latest = pd.to_datetime(sub["trade_date"]).max()
    sub = sub[pd.to_datetime(sub["trade_date"]) == latest]
    return _covariance_matrix_from_rows(sub, factors)


def _covariance_matrix_from_rows(sub: pd.DataFrame | None, factors: list[str]) -> pd.DataFrame:
    mat_values = np.zeros((len(factors), len(factors)))
    if sub is None or sub.empty:
        return pd.DataFrame(mat_values, index=factors, columns=factors)
    factor_pos = {factor: idx for idx, factor in enumerate(factors)}
    row_idx = sub["factor_i"].map(factor_pos)
    col_idx = sub["factor_j"].map(factor_pos)
    mask = row_idx.notna() & col_idx.notna()
    if mask.any():
        r = row_idx[mask].astype(int).to_numpy()
        c = col_idx[mask].astype(int).to_numpy()
        values = pd.to_numeric(sub.loc[mask, "covariance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mat_values[r, c] = values
        off_diag = r != c
        mat_values[c[off_diag], r[off_diag]] = values[off_diag]
    return pd.DataFrame(mat_values, index=factors, columns=factors)


def _covariance_rows_by_target_date(
    covariance: pd.DataFrame,
    target_dates: pd.Series | pd.Index | list[pd.Timestamp],
    window: int | None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if window is None or "window" not in covariance.columns:
        cov_window = covariance.copy()
    else:
        cov_window = covariance[covariance["window"] == window].copy()
    if cov_window.empty:
        return {}
    cov_window["trade_date"] = pd.to_datetime(cov_window["trade_date"])
    cov_dates = pd.Index(cov_window["trade_date"].dropna().unique()).sort_values()
    target_index = pd.Index(pd.to_datetime(pd.Series(target_dates).dropna().unique())).sort_values()
    latest_by_target: dict[pd.Timestamp, pd.Timestamp] = {}
    for target in target_index:
        pos = cov_dates.searchsorted(target, side="right") - 1
        if pos >= 0:
            latest_by_target[pd.Timestamp(target)] = pd.Timestamp(cov_dates[pos])
    needed_dates = set(latest_by_target.values())
    if not needed_dates:
        return {}
    needed_rows = cov_window[cov_window["trade_date"].isin(needed_dates)]
    rows_by_cov_date = {pd.Timestamp(date): sub for date, sub in needed_rows.groupby("trade_date", sort=False)}
    return {target: rows_by_cov_date[cov_date] for target, cov_date in latest_by_target.items() if cov_date in rows_by_cov_date}


def _active_vector(
    sub: pd.DataFrame,
    active: pd.Series,
    style_cols: list[str],
    in_model: pd.Series | None = None,
    industry_col: str = "industry",
) -> tuple[dict[str, float], list[dict]]:
    exposures: dict[str, float] = {}
    rows = []
    if in_model is None:
        in_model = pd.Series(True, index=sub.index)
    in_model = in_model.reindex(sub.index, fill_value=False).astype(bool)
    for col in style_cols:
        value = float((active[in_model] * pd.to_numeric(sub.loc[in_model, col], errors="coerce").fillna(0.0)).sum())
        exposures[col] = value
        rows.append({"factor": col, "factor_type": "style", "active_exposure": value})
    industry = sub.loc[in_model, industry_col].fillna("unknown").astype(str) if industry_col in sub.columns else pd.Series("unknown", index=sub.index[in_model])
    for ind, idx in industry.groupby(industry, sort=False).groups.items():
        factor = f"industry_{ind}"
        value = float(active.loc[list(idx)].sum())
        exposures[factor] = value
        rows.append({"factor": factor, "factor_type": "industry", "active_exposure": value})
    return exposures, rows


def _realized_te(daily_returns: pd.DataFrame | None, scenario: str, date: pd.Timestamp, window: int = 60) -> float:
    if daily_returns is None or daily_returns.empty or "active_return" not in daily_returns.columns:
        return np.nan
    sub = daily_returns[daily_returns["scenario"] == scenario].copy() if "scenario" in daily_returns.columns else daily_returns.copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub = sub[sub["trade_date"] <= date].sort_values("trade_date").tail(window)
    if len(sub) < max(20, window // 2):
        return np.nan
    return float(pd.to_numeric(sub["active_return"], errors="coerce").std() * np.sqrt(252))


def run_attribution(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    style: pd.DataFrame,
    covariance: pd.DataFrame,
    specific_risk: pd.DataFrame,
    daily_returns: pd.DataFrame | None = None,
    benchmark_col: str = "csi500_index_weight",
    window: int | None = 252,
    industry_col: str = "industry",
    specific_variance_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    style = style.copy()
    style["trade_date"] = pd.to_datetime(style["trade_date"])
    covariance = covariance.copy()
    covariance["trade_date"] = pd.to_datetime(covariance["trade_date"])
    specific_risk = specific_risk.copy()
    specific_risk["trade_date"] = pd.to_datetime(specific_risk["trade_date"])
    style_cols = [
        c
        for c in style.columns
        if c.startswith("style_")
        and not c.endswith("_n")
        and not c.endswith("_effective_weight")
    ]

    exposure_rows = []
    risk_rows = []
    summary_rows = []
    scenarios = weights["scenario"].dropna().unique().tolist() if "scenario" in weights.columns else ["portfolio"]
    if "scenario" not in weights.columns:
        weights["scenario"] = "portfolio"

    panel_by_date = {date: sub for date, sub in panel.groupby("trade_date", sort=False)}
    style_by_date = {date: sub for date, sub in style.groupby("trade_date", sort=False)}
    covariance_rows_by_date = _covariance_rows_by_target_date(covariance, weights["trade_date"], window)

    for scenario in scenarios:
        w_scenario = weights[weights["scenario"] == scenario].copy()
        for date, w_date in w_scenario.groupby("trade_date", sort=True):
            p_date = panel_by_date.get(date)
            s_date = style_by_date.get(date)
            if p_date is None or s_date is None:
                continue
            sub = p_date.merge(s_date[["trade_date", "ts_code"] + style_cols], on=["trade_date", "ts_code"], how="left")
            sub = sub.drop_duplicates("ts_code").set_index("ts_code")
            port = w_date.set_index("ts_code")["weight"]
            idx = sub.index.union(port.index)
            sub = sub.reindex(idx)
            in_model = sub[benchmark_col].notna() if benchmark_col in sub.columns else pd.Series(False, index=idx)
            port = _normalize_weights(port.reindex(idx, fill_value=0.0))
            bench = _normalize_weights(sub[benchmark_col].fillna(0.0) if benchmark_col in sub.columns else pd.Series(0.0, index=idx))
            active = port - bench

            sub_reset = sub.reset_index(drop=False)
            active_reset = pd.Series(active.to_numpy(), index=sub_reset.index)
            in_model_reset = pd.Series(in_model.to_numpy(), index=sub_reset.index)
            out_of_model_weight = float(port.loc[~in_model].sum())
            benchmark_model_weight = float(bench.loc[in_model].sum())

            exposures, rows = _active_vector(
                sub_reset,
                active_reset,
                style_cols,
                in_model=in_model_reset,
                industry_col=industry_col,
            )
            for row in rows:
                row.update({"scenario": scenario, "trade_date": date})
                exposure_rows.append(row)
            exposure_rows.append(
                {
                    "factor": "outside_model_universe",
                    "factor_type": "diagnostic",
                    "active_exposure": out_of_model_weight,
                    "scenario": scenario,
                    "trade_date": date,
                }
            )

            factors = list(exposures)
            cov = _covariance_matrix_from_rows(covariance_rows_by_date.get(pd.Timestamp(date)), factors)
            vec = pd.Series(exposures).reindex(factors).fillna(0.0)
            factor_var = float(vec.to_numpy() @ cov.reindex(index=factors, columns=factors).fillna(0.0).to_numpy() @ vec.to_numpy())
            style_factors = [f for f in factors if f.startswith("style_") or f == "country"]
            industry_factors = [f for f in factors if f.startswith("industry_")]
            style_var = float(vec.reindex(style_factors, fill_value=0.0).to_numpy() @ cov.reindex(index=style_factors, columns=style_factors).fillna(0.0).to_numpy() @ vec.reindex(style_factors, fill_value=0.0).to_numpy()) if style_factors else 0.0
            industry_var = float(vec.reindex(industry_factors, fill_value=0.0).to_numpy() @ cov.reindex(index=industry_factors, columns=industry_factors).fillna(0.0).to_numpy() @ vec.reindex(industry_factors, fill_value=0.0).to_numpy()) if industry_factors else 0.0

            sr_date = _latest_by_date(specific_risk, "trade_date", date).set_index("ts_code")
            if specific_variance_col is not None and specific_variance_col in sr_date.columns:
                specific_variance = pd.to_numeric(
                    sr_date.reindex(idx)[specific_variance_col], errors="coerce"
                )
            else:
                sr_col = f"specific_risk_{window}"
                sr = pd.to_numeric(sr_date.reindex(idx)[sr_col], errors="coerce") if sr_col in sr_date.columns else pd.Series(np.nan, index=idx)
                specific_variance = sr**2
            fallback = specific_variance.median(skipna=True)
            specific_variance = specific_variance.fillna(0.0 if pd.isna(fallback) else fallback)
            specific_var = float(((active**2) * specific_variance).sum())
            total_var = max(factor_var + specific_var, 0.0)
            predicted_te = float(np.sqrt(total_var * 252))
            realized_te = _realized_te(daily_returns, scenario, date)
            risk_rows.append(
                {
                    "scenario": scenario,
                    "trade_date": date,
                    "window": window if window is not None else "eigenfactor",
                    "factor_var_daily": factor_var,
                    "style_var_daily": style_var,
                    "industry_var_daily": industry_var,
                    "specific_var_daily": specific_var,
                    "total_var_daily": total_var,
                    "predicted_te_annual": predicted_te,
                    "realized_te_annual_60d": realized_te,
                    "active_share": float(0.5 * active.abs().sum()),
                    "n_portfolio_names": int((port > 1e-10).sum()),
                    "portfolio_model_weight": float(port.loc[in_model].sum()),
                    "out_of_model_weight": out_of_model_weight,
                    "benchmark_model_weight": benchmark_model_weight,
                }
            )
    exposures = pd.DataFrame(exposure_rows)
    risk = pd.DataFrame(risk_rows)
    if not risk.empty:
        summary = risk.groupby("scenario", sort=False).agg(
            dates=("trade_date", "nunique"),
            predicted_te_mean=("predicted_te_annual", "mean"),
            predicted_te_latest=("predicted_te_annual", "last"),
            realized_te_60d_latest=("realized_te_annual_60d", "last"),
            active_share_mean=("active_share", "mean"),
            out_of_model_weight_mean=("out_of_model_weight", "mean"),
            out_of_model_weight_latest=("out_of_model_weight", "last"),
            portfolio_model_weight_latest=("portfolio_model_weight", "last"),
            style_var_share=("style_var_daily", lambda s: float(s.sum() / max(risk.loc[s.index, "total_var_daily"].sum(), 1e-20))),
            industry_var_share=("industry_var_daily", lambda s: float(s.sum() / max(risk.loc[s.index, "total_var_daily"].sum(), 1e-20))),
            specific_var_share=("specific_var_daily", lambda s: float(s.sum() / max(risk.loc[s.index, "total_var_daily"].sum(), 1e-20))),
        ).reset_index()
    else:
        summary = pd.DataFrame()
    return exposures, risk, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv")
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--style", default="outputs/cne6_reproduction/style_exposures.parquet")
    parser.add_argument("--covariance", default="outputs/cne6_reproduction/factor_covariance_rolling.parquet")
    parser.add_argument("--specific-risk", default="outputs/cne6_reproduction/specific_risk.parquet")
    parser.add_argument("--daily-returns", default="outputs/csi500_xgb_constrained_index_enhancement/constrained_daily_returns.csv")
    parser.add_argument("--output", default="outputs/cne6_reproduction")
    parser.add_argument("--window", type=int, default=252)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(args.weights, parse_dates=["trade_date"])
    panel = pd.read_parquet(args.panel)
    style = pd.read_parquet(args.style)
    covariance = pd.read_parquet(args.covariance)
    specific_risk = pd.read_parquet(args.specific_risk)
    daily_returns_path = Path(args.daily_returns)
    daily_returns = pd.read_csv(daily_returns_path, parse_dates=["trade_date"]) if daily_returns_path.exists() else None
    exposures, risk, summary = run_attribution(weights, panel, style, covariance, specific_risk, daily_returns=daily_returns, window=args.window)
    exposures.to_csv(output_dir / "portfolio_active_exposures.csv", index=False)
    risk.to_csv(output_dir / "portfolio_risk_attribution.csv", index=False)
    risk[["scenario", "trade_date", "predicted_te_annual", "realized_te_annual_60d"]].to_csv(output_dir / "predicted_vs_realized_te.csv", index=False)
    summary.to_csv(output_dir / "portfolio_risk_summary.csv", index=False)
    print(f"wrote CNE6-style portfolio attribution to {output_dir}")


if __name__ == "__main__":
    main()
