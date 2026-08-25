"""Risk-weighted CNE6 cross-sectional regression diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors.risk.cne6_regression import _regression_work, _standardize


RETURN_MODE_LABELS = {
    "same_day": "当天暴露解释当天收益",
    "forward_1d": "当天暴露解释下一日收益",
    "lagged_exposure_1d": "昨日暴露解释今日收益",
}

FIT_WEIGHT_LABELS = {
    "sqrt_mv": "市值平方根权重",
    "equal": "等权",
    "sqrt_mv_x_low_specific_rank_60_t3": "市值平方根 * t-3 60日低特异风险排名",
    "sqrt_mv_x_low_specific_rank_120_t3": "市值平方根 * t-3 120日低特异风险排名",
    "sqrt_mv_x_low_specific_rank_252_t3": "市值平方根 * t-3 252日低特异风险排名",
    "inv_specific_60_t3": "t-3 60日特异风险倒方差",
    "inv_specific_120_t3": "t-3 120日特异风险倒方差",
    "inv_specific_252_t3": "t-3 252日特异风险倒方差",
    "inv_specific_vol_252_t3": "t-3 252日特异风险倒数",
    "sqrt_mv_div_specific_vol_252_t3": "市值平方根 / t-3 252日特异风险",
}

FIT_WEIGHT_PLOT_LABELS = {
    "sqrt_mv": "sqrt_mv",
    "equal": "equal",
    "sqrt_mv_x_low_specific_rank_60_t3": "sqrt_mv*rank60",
    "sqrt_mv_x_low_specific_rank_120_t3": "sqrt_mv*rank120",
    "sqrt_mv_x_low_specific_rank_252_t3": "sqrt_mv*rank252",
    "inv_specific_60_t3": "inv_var_60",
    "inv_specific_120_t3": "inv_var_120",
    "inv_specific_252_t3": "inv_var_252",
    "inv_specific_vol_252_t3": "inv_vol_252",
    "sqrt_mv_div_specific_vol_252_t3": "sqrt_mv/inv_vol",
}

EVAL_WEIGHT_LABELS = {
    "own": "自权重评价",
    "sqrt_mv": "市值平方根评价",
    "equal": "等权评价",
}

DEFAULT_FIT_WEIGHTS = tuple(FIT_WEIGHT_LABELS)
DEFAULT_EVAL_WEIGHTS = ("own", "sqrt_mv", "equal")
DEFAULT_RETURN_MODES = ("same_day", "forward_1d")


def _md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无数据。"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_datetime64_any_dtype(show[col]):
            show[col] = pd.to_datetime(show[col]).dt.date.astype(str)
        elif pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    columns = [str(col) for col in show.columns]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in show.itertuples(index=False):
        values = ["" if pd.isna(item) else str(item).replace("|", "/") for item in row]
        lines.append("|" + "|".join(values) + "|")
    if len(df) > max_rows:
        values = ["..."] + [f"仅展示前 {max_rows} 行，共 {len(df)} 行"] + [""] * max(0, len(columns) - 2)
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def _parse_csv_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def risk_asof_mapping(trade_dates: pd.Series | pd.Index, lag_days: int = 3) -> pd.DataFrame:
    """Map target dates to the lagged date whose risk estimate is observable."""

    dates = pd.Index(pd.to_datetime(pd.Series(trade_dates).dropna().unique())).sort_values()
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    if len(dates) <= lag_days:
        return pd.DataFrame(columns=["trade_date", "risk_asof_date"])
    return pd.DataFrame(
        {
            "trade_date": dates[lag_days:],
            "risk_asof_date": dates[:-lag_days],
        }
    )


def add_lagged_specific_risk(work: pd.DataFrame, specific_risk: pd.DataFrame, lag_days: int = 3) -> pd.DataFrame:
    """Attach t-lag specific-risk columns to each target date and stock."""

    risk_cols = [c for c in specific_risk.columns if c.startswith("specific_risk_")]
    if not risk_cols:
        return work.copy()
    mapping = risk_asof_mapping(work["trade_date"], lag_days=lag_days)
    out = work.merge(mapping, on="trade_date", how="left")
    risk = specific_risk[["trade_date", "ts_code"] + risk_cols].copy()
    risk["trade_date"] = pd.to_datetime(risk["trade_date"])
    risk = risk.rename(columns={"trade_date": "risk_asof_date"})
    out = out.merge(risk, on=["risk_asof_date", "ts_code"], how="left")
    return out


def market_cap_weights(frame: pd.DataFrame) -> pd.Series:
    if "total_mv" not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype=float)
    weights = np.sqrt(pd.to_numeric(frame["total_mv"], errors="coerce").clip(lower=0).fillna(0.0))
    weights = pd.Series(weights, index=frame.index, dtype=float)
    return weights.where(np.isfinite(weights) & (weights > 0), 1.0)


def clip_positive_weights(
    values: pd.Series | np.ndarray,
    clip_quantiles: tuple[float, float] | None = (0.01, 0.99),
    min_count: int = 20,
) -> pd.Series:
    weights = pd.Series(values, copy=True, dtype=float)
    weights = weights.where(np.isfinite(weights) & (weights > 0))
    valid = weights.dropna()
    if clip_quantiles is not None and len(valid) >= min_count:
        lower_q, upper_q = clip_quantiles
        lower = float(valid.quantile(lower_q))
        upper = float(valid.quantile(upper_q))
        if np.isfinite(lower) and np.isfinite(upper) and upper > 0 and upper >= lower:
            weights = weights.clip(lower=lower, upper=upper)
    return weights


def effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return float("nan")
    denom = float(np.sum(values**2))
    if denom <= 0:
        return float("nan")
    return float((np.sum(values) ** 2) / denom)


def weight_diagnostics(weights: pd.Series | np.ndarray, n_obs: int | None = None) -> dict:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return {
            "effective_sample_size": np.nan,
            "effective_sample_share": np.nan,
            "max_weight_share": np.nan,
            "p99_p50_weight_ratio": np.nan,
            "weight_cv": np.nan,
        }
    ess = effective_sample_size(values)
    total = float(np.sum(values))
    p50 = float(np.percentile(values, 50))
    p99 = float(np.percentile(values, 99))
    obs = int(n_obs) if n_obs is not None else int(len(values))
    return {
        "effective_sample_size": ess,
        "effective_sample_share": float(ess / obs) if obs > 0 and np.isfinite(ess) else np.nan,
        "max_weight_share": float(np.max(values) / total) if total > 0 else np.nan,
        "p99_p50_weight_ratio": float(p99 / p50) if p50 > 0 else np.nan,
        "weight_cv": float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else np.nan,
    }


def fit_weights(frame: pd.DataFrame, strategy: str) -> pd.Series:
    """Build per-stock WLS fit weights for one date."""

    if strategy == "sqrt_mv":
        return market_cap_weights(frame)
    if strategy == "equal":
        return pd.Series(1.0, index=frame.index, dtype=float)

    mv = market_cap_weights(frame)
    if strategy in {
        "sqrt_mv_x_low_specific_rank_60_t3",
        "sqrt_mv_x_low_specific_rank_120_t3",
        "sqrt_mv_x_low_specific_rank_252_t3",
    }:
        risk_col = strategy.replace("sqrt_mv_x_low_specific_rank_", "specific_risk_").replace("_t3", "")
        if risk_col not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        sigma = pd.to_numeric(frame[risk_col], errors="coerce")
        valid_sigma = sigma.where(np.isfinite(sigma) & (sigma > 0))
        rank_pct = valid_sigma.rank(pct=True, ascending=False)
        multiplier = 0.75 + 0.50 * rank_pct
        raw = mv * multiplier
        return raw.where(np.isfinite(raw) & (raw > 0) & valid_sigma.notna())

    if strategy == "inv_specific_60_t3":
        risk_col = "specific_risk_60"
        sigma_power = 2.0
        use_mv = False
    elif strategy == "inv_specific_120_t3":
        risk_col = "specific_risk_120"
        sigma_power = 2.0
        use_mv = False
    elif strategy == "inv_specific_252_t3":
        risk_col = "specific_risk_252"
        sigma_power = 2.0
        use_mv = False
    elif strategy == "inv_specific_vol_252_t3":
        risk_col = "specific_risk_252"
        sigma_power = 1.0
        use_mv = False
    elif strategy == "sqrt_mv_div_specific_vol_252_t3":
        risk_col = "specific_risk_252"
        sigma_power = 1.0
        use_mv = True
    else:
        raise ValueError(f"unknown fit weight strategy: {strategy}")

    if risk_col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    sigma = pd.to_numeric(frame[risk_col], errors="coerce")
    raw = 1.0 / (sigma**sigma_power)
    if use_mv:
        raw = raw * mv
    raw = raw.where(np.isfinite(raw) & (raw > 0) & (sigma > 0))
    return clip_positive_weights(raw, clip_quantiles=(0.01, 0.99), min_count=20)


def eval_weights(frame: pd.DataFrame, eval_weight: str, own_weights: pd.Series) -> pd.Series:
    if eval_weight == "own":
        return pd.Series(own_weights, index=frame.index, dtype=float)
    if eval_weight == "sqrt_mv":
        return market_cap_weights(frame)
    if eval_weight == "equal":
        return pd.Series(1.0, index=frame.index, dtype=float)
    raise ValueError(f"unknown eval weight strategy: {eval_weight}")


def weighted_r2(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(y) & np.isfinite(fitted) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    y_use = y[mask]
    fitted_use = fitted[mask]
    weights_use = weights[mask]
    resid = y_use - fitted_use
    ss_res = float(np.sum(weights_use * resid**2))
    y_bar = float(np.average(y_use, weights=weights_use))
    ss_tot = float(np.sum(weights_use * (y_use - y_bar) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else float("nan")


def weighted_resid_std(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(y) & np.isfinite(fitted) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) == 0:
        return float("nan")
    resid = y[mask] - fitted[mask]
    return float(np.sqrt(np.average(resid**2, weights=weights[mask])))


def fit_wls_beta(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
    if np.isnan(weights).any():
        raise ValueError("fit weights contain invalid values")
    root_w = np.sqrt(weights / np.nanmean(weights))
    xw = x * root_w[:, None]
    yw = y * root_w
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    cond = float(np.linalg.cond(xw)) if xw.size else float("nan")
    return beta, cond


def _date_design(valid: pd.DataFrame, style_cols: list[str]) -> tuple[np.ndarray, np.ndarray, int, int, list[str]]:
    use_styles = [col for col in style_cols if col in valid.columns and valid[col].notna().mean() >= 0.5]
    style_values = []
    for col in use_styles:
        style_values.append(_standardize(valid[col]).fillna(0.0).to_numpy())

    industry = valid["industry"].fillna("unknown").astype(str)
    dummies = pd.get_dummies(industry, prefix="industry", dtype=float)
    effective_industries = int(dummies.shape[1])
    if dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]
    elif dummies.shape[1] == 1:
        dummies = dummies.iloc[:, 0:0]

    columns = ["country"] + use_styles + dummies.columns.tolist()
    parts = [np.ones(len(valid))]
    parts.extend(style_values)
    if not dummies.empty:
        parts.extend(dummies.to_numpy().T)
    x = np.column_stack(parts)
    y = pd.to_numeric(valid["_y"], errors="coerce").to_numpy(dtype=float)
    return y, x, effective_industries, len(use_styles), columns


def _failed_row(
    date: pd.Timestamp,
    return_mode: str,
    fit_weight: str,
    eval_weight_name: str,
    status: str,
    n_obs: int,
) -> dict:
    return {
        "trade_date": date,
        "return_mode": return_mode,
        "return_mode_cn": RETURN_MODE_LABELS.get(return_mode, return_mode),
        "fit_weight": fit_weight,
        "fit_weight_cn": FIT_WEIGHT_LABELS.get(fit_weight, fit_weight),
        "eval_weight": eval_weight_name,
        "eval_weight_cn": EVAL_WEIGHT_LABELS.get(eval_weight_name, eval_weight_name),
        "regression_status": status,
        "n_obs": int(n_obs),
        "n_factors": np.nan,
        "n_industries": np.nan,
        "n_styles": np.nan,
        "r2": np.nan,
        "adj_r2": np.nan,
        "condition_number": np.nan,
        "resid_std_weighted": np.nan,
        "effective_sample_size": np.nan,
        "effective_sample_share": np.nan,
        "max_weight_share": np.nan,
        "p99_p50_weight_ratio": np.nan,
        "weight_cv": np.nan,
    }


def date_weighted_regression(
    sub: pd.DataFrame,
    style_cols: list[str],
    return_mode: str,
    fit_weight_name: str,
    eval_weight_names: tuple[str, ...] = DEFAULT_EVAL_WEIGHTS,
) -> list[dict]:
    date = pd.to_datetime(sub["trade_date"].iloc[0])
    base = sub.dropna(subset=["_y", "ts_code", "industry"]).copy()
    if base.empty:
        return [_failed_row(date, return_mode, fit_weight_name, "own", "failed_empty", 0)]

    candidate_weights = fit_weights(base, fit_weight_name)
    valid_mask = np.isfinite(candidate_weights.to_numpy(dtype=float)) & (candidate_weights.to_numpy(dtype=float) > 0)
    valid = base.loc[valid_mask].copy()
    if valid.empty:
        return [_failed_row(date, return_mode, fit_weight_name, "own", "failed_no_fit_weight", 0)]

    y, x, effective_industries, n_styles, _ = _date_design(valid, style_cols)
    n_obs = int(len(valid))
    n_factors = int(x.shape[1])
    min_obs = max(30, n_factors + 5)
    if n_obs < min_obs or effective_industries < 2:
        return [_failed_row(date, return_mode, fit_weight_name, "own", "failed_insufficient_sample", n_obs)]

    own_weights = fit_weights(valid, fit_weight_name)
    try:
        beta, cond = fit_wls_beta(y, x, own_weights.to_numpy(dtype=float))
    except (np.linalg.LinAlgError, ValueError):
        return [_failed_row(date, return_mode, fit_weight_name, "own", "failed_linalg", n_obs)]

    fitted = x @ beta
    diag = weight_diagnostics(own_weights, n_obs=n_obs)
    rows = []
    for eval_weight_name in eval_weight_names:
        e_weights = eval_weights(valid, eval_weight_name, own_weights)
        r2 = weighted_r2(y, fitted, e_weights.to_numpy(dtype=float))
        adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / max(n_obs - n_factors, 1) if np.isfinite(r2) else np.nan
        row = {
            "trade_date": date,
            "return_mode": return_mode,
            "return_mode_cn": RETURN_MODE_LABELS.get(return_mode, return_mode),
            "fit_weight": fit_weight_name,
            "fit_weight_cn": FIT_WEIGHT_LABELS.get(fit_weight_name, fit_weight_name),
            "eval_weight": eval_weight_name,
            "eval_weight_cn": EVAL_WEIGHT_LABELS.get(eval_weight_name, eval_weight_name),
            "regression_status": "ok",
            "n_obs": n_obs,
            "n_factors": n_factors,
            "n_industries": effective_industries,
            "n_styles": int(n_styles),
            "r2": r2,
            "adj_r2": adj_r2,
            "condition_number": cond,
            "resid_std_weighted": weighted_resid_std(y, fitted, e_weights.to_numpy(dtype=float)),
        }
        row.update(diag)
        rows.append(row)
    return rows


def run_weighted_regressions(
    panel: pd.DataFrame,
    style_exposures: pd.DataFrame,
    specific_risk: pd.DataFrame,
    return_modes: tuple[str, ...] = DEFAULT_RETURN_MODES,
    fit_weight_names: tuple[str, ...] = DEFAULT_FIT_WEIGHTS,
    eval_weight_names: tuple[str, ...] = DEFAULT_EVAL_WEIGHTS,
    risk_lag_days: int = 3,
) -> pd.DataFrame:
    rows = []
    for return_mode in return_modes:
        print(f"running return_mode={return_mode}")
        work, style_cols = _regression_work(panel, style_exposures, return_mode=return_mode)
        work = add_lagged_specific_risk(work, specific_risk, lag_days=risk_lag_days)
        work = work.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        for idx, (_, sub) in enumerate(work.groupby("trade_date", sort=False), start=1):
            if idx % 250 == 0:
                print(f"  processed {idx} dates for {return_mode}")
            for fit_weight_name in fit_weight_names:
                rows.extend(
                    date_weighted_regression(
                        sub,
                        style_cols,
                        return_mode=return_mode,
                        fit_weight_name=fit_weight_name,
                        eval_weight_names=eval_weight_names,
                    )
                )
    daily = pd.DataFrame(rows)
    daily["risk_lag_days"] = int(risk_lag_days)
    return daily


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    ok = daily[daily["regression_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    grouped = ok.groupby(["return_mode", "return_mode_cn", "fit_weight", "fit_weight_cn", "eval_weight", "eval_weight_cn"], sort=False)
    out = grouped.agg(
        起始日期=("trade_date", "min"),
        结束日期=("trade_date", "max"),
        成功回归天数=("r2", "count"),
        平均样本股票数=("n_obs", "mean"),
        平均因子数=("n_factors", "mean"),
        平均行业数=("n_industries", "mean"),
        平均R2=("r2", "mean"),
        R2中位数=("r2", "median"),
        平均调整R2=("adj_r2", "mean"),
        R2超过0_5占比=("r2", lambda s: float((s >= 0.5).mean())),
        平均条件数=("condition_number", "mean"),
        平均加权残差标准差=("resid_std_weighted", "mean"),
        平均有效样本数=("effective_sample_size", "mean"),
        平均有效样本占比=("effective_sample_share", "mean"),
        平均最大权重占比=("max_weight_share", "mean"),
        平均P99_P50权重比=("p99_p50_weight_ratio", "mean"),
    ).reset_index()
    return out


def comparable_summary(daily: pd.DataFrame, fit_weight_names: tuple[str, ...]) -> pd.DataFrame:
    ok = daily[daily["regression_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    frames = []
    target_count = len(fit_weight_names)
    for (return_mode, eval_weight_name), sub in ok.groupby(["return_mode", "eval_weight"], sort=False):
        date_counts = sub.groupby("trade_date")["fit_weight"].nunique()
        common_dates = date_counts[date_counts == target_count].index
        if len(common_dates) == 0:
            continue
        comp = sub[sub["trade_date"].isin(common_dates)].copy()
        summary = summarize_daily(comp)
        if summary.empty:
            continue
        summary["共同样本起始日期"] = pd.to_datetime(common_dates).min()
        summary["共同样本结束日期"] = pd.to_datetime(common_dates).max()
        summary["共同样本天数"] = int(len(common_dates))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    ok = daily[daily["regression_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["year"] = pd.to_datetime(ok["trade_date"]).dt.year
    out = (
        ok.groupby(["return_mode", "return_mode_cn", "fit_weight", "fit_weight_cn", "eval_weight", "eval_weight_cn", "year"], sort=True)
        .agg(
            成功回归天数=("r2", "count"),
            平均R2=("r2", "mean"),
            R2中位数=("r2", "median"),
            R2超过0_5占比=("r2", lambda s: float((s >= 0.5).mean())),
            平均有效样本占比=("effective_sample_share", "mean"),
            平均最大权重占比=("max_weight_share", "mean"),
        )
        .reset_index()
    )
    return out


def weight_diagnostics_daily(daily: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_date",
        "return_mode",
        "return_mode_cn",
        "fit_weight",
        "fit_weight_cn",
        "regression_status",
        "n_obs",
        "effective_sample_size",
        "effective_sample_share",
        "max_weight_share",
        "p99_p50_weight_ratio",
        "weight_cv",
    ]
    own = daily[daily["eval_weight"] == "own"].copy()
    return own[[c for c in cols if c in own.columns]].drop_duplicates()


def _save_summary_plot(summary: pd.DataFrame, output: Path) -> str | None:
    same = summary[summary["return_mode"] == "same_day"].copy()
    if same.empty:
        return None
    pivot = same.pivot(index="fit_weight", columns="eval_weight", values="平均R2").reindex(index=list(DEFAULT_FIT_WEIGHTS))
    pivot = pivot[[c for c in DEFAULT_EVAL_WEIGHTS if c in pivot.columns]]
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.0))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = {"own": "#4C78A8", "sqrt_mv": "#F58518", "equal": "#54A24B"}
    for offset, col in enumerate(pivot.columns):
        ax.bar(x + (offset - 1) * width, pivot[col], width=width, label=col, color=colors.get(col))
    ax.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.0, label="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([FIT_WEIGHT_PLOT_LABELS.get(idx, idx) for idx in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("Mean R2")
    ax.set_title("Same-day CNE6 regression R2 by fit and evaluation weight")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output / "risk_weighted_r2_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _save_cross_eval_timeseries(daily: pd.DataFrame, output: Path) -> str | None:
    sub = daily[
        (daily["return_mode"] == "same_day")
        & (daily["eval_weight"] == "sqrt_mv")
        & (daily["regression_status"] == "ok")
    ].copy()
    if sub.empty:
        return None
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    fig, ax = plt.subplots(figsize=(12, 5.0))
    for fit_weight_name, part in sub.groupby("fit_weight", sort=False):
        part = part.sort_values("trade_date")
        label = FIT_WEIGHT_PLOT_LABELS.get(fit_weight_name, fit_weight_name)
        ax.plot(part["trade_date"], part["r2"].rolling(60, min_periods=20).mean(), linewidth=1.0, label=label)
    ax.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.0, label="0.5")
    ax.set_ylabel("R2, 60d mean")
    ax.set_title("Same-day R2 under official sqrt_mv evaluation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = output / "risk_weighted_same_day_cross_eval_r2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _save_weight_concentration(weight_diag: pd.DataFrame, output: Path) -> str | None:
    sub = weight_diag[(weight_diag["return_mode"] == "same_day") & (weight_diag["regression_status"] == "ok")].copy()
    if sub.empty:
        return None
    grouped = (
        sub.groupby("fit_weight", sort=False)
        .agg(
            effective_sample_share=("effective_sample_share", "mean"),
            max_weight_share=("max_weight_share", "mean"),
            p99_p50_weight_ratio=("p99_p50_weight_ratio", "mean"),
        )
        .reindex(list(DEFAULT_FIT_WEIGHTS))
        .dropna(how="all")
    )
    if grouped.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(grouped.index))
    labels = [FIT_WEIGHT_PLOT_LABELS.get(idx, idx) for idx in grouped.index]
    axes[0].bar(x, grouped["effective_sample_share"], color="#4C78A8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Effective sample share")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, grouped["max_weight_share"], color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_title("Max weight share")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Same-day fit-weight concentration")
    fig.tight_layout()
    path = output / "risk_weight_concentration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _select_summary(summary: pd.DataFrame, return_mode: str, eval_weight_name: str) -> pd.DataFrame:
    cols = [
        "fit_weight",
        "fit_weight_cn",
        "eval_weight",
        "eval_weight_cn",
        "起始日期",
        "结束日期",
        "成功回归天数",
        "平均样本股票数",
        "平均因子数",
        "平均R2",
        "R2中位数",
        "平均调整R2",
        "R2超过0_5占比",
        "平均条件数",
        "平均有效样本数",
        "平均有效样本占比",
        "平均最大权重占比",
        "平均P99_P50权重比",
    ]
    sub = summary[(summary["return_mode"] == return_mode) & (summary["eval_weight"] == eval_weight_name)].copy()
    return sub[[c for c in cols if c in sub.columns]]


def _value(summary: pd.DataFrame, return_mode: str, fit_weight_name: str, eval_weight_name: str, column: str = "平均R2") -> float:
    sub = summary[
        (summary["return_mode"] == return_mode)
        & (summary["fit_weight"] == fit_weight_name)
        & (summary["eval_weight"] == eval_weight_name)
    ]
    if sub.empty or column not in sub.columns:
        return float("nan")
    return float(sub[column].iloc[0])


def _write_report(
    doc: Path,
    output: Path,
    summary: pd.DataFrame,
    comparable: pd.DataFrame,
    by_year: pd.DataFrame,
    fig_names: list[str],
) -> None:
    baseline_same = _value(summary, "same_day", "sqrt_mv", "sqrt_mv")
    baseline_forward = _value(summary, "forward_1d", "sqrt_mv", "sqrt_mv")

    same_own = _select_summary(summary, "same_day", "own").sort_values("平均R2", ascending=False)
    same_sqrt = _select_summary(summary, "same_day", "sqrt_mv").sort_values("平均R2", ascending=False)
    same_equal = _select_summary(summary, "same_day", "equal").sort_values("平均R2", ascending=False)

    best_own = same_own.iloc[0] if not same_own.empty else None
    best_sqrt = same_sqrt.iloc[0] if not same_sqrt.empty else None
    best_equal = same_equal.iloc[0] if not same_equal.empty else None

    comparable_same_sqrt = comparable[
        (comparable["return_mode"] == "same_day") & (comparable["eval_weight"] == "sqrt_mv")
    ].copy()
    comparable_same_sqrt = comparable_same_sqrt.sort_values("平均R2", ascending=False)

    year_same_sqrt = by_year[
        (by_year["return_mode"] == "same_day") & (by_year["eval_weight"] == "sqrt_mv")
    ].copy()
    year_same_sqrt = year_same_sqrt.sort_values(["year", "fit_weight"])

    conclusion = []
    if best_own is not None:
        conclusion.append(
            f"在当天收益分解口径下，自权重评价最高的是 `{best_own['fit_weight']}`，平均 R2 为 {best_own['平均R2']:.4f}。"
        )
        if (
            "平均有效样本数" in best_own
            and "平均因子数" in best_own
            and pd.notna(best_own["平均有效样本数"])
            and pd.notna(best_own["平均因子数"])
            and float(best_own["平均有效样本数"]) < float(best_own["平均因子数"])
        ):
            conclusion.append(
                f"但这个结果不能当成模型达标，因为它的平均有效样本数只有 {best_own['平均有效样本数']:.1f}，明显小于平均因子数 {best_own['平均因子数']:.1f}，更像权重过度集中后的数值过拟合。"
            )
    if best_sqrt is not None:
        diff = float(best_sqrt["平均R2"]) - baseline_same
        conclusion.append(
            f"但是换成正式基线的市值平方根权重评价后，最高的是 `{best_sqrt['fit_weight']}`，平均 R2 为 {best_sqrt['平均R2']:.4f}，相对 `sqrt_mv` 基线变化 {diff:+.4f}。"
        )
        if float(best_sqrt["平均R2"]) < 0.5:
            conclusion.append("这个结果只能算小幅改善，仍然没有达到 0.5 参考线。")
    if best_equal is not None:
        conclusion.append(
            f"等权评价下最高的是 `{best_equal['fit_weight']}`，平均 R2 为 {best_equal['平均R2']:.4f}。"
        )
    if best_own is not None and best_sqrt is not None and str(best_own["fit_weight"]) != str(best_sqrt["fit_weight"]):
        conclusion.append(
            "这说明最高自权重 R2 和更可比较的交叉评价 R2 并不是同一个结论，不能只拿最高的自权重 R2 说模型已经达到商业风险模型的水平。"
        )

    lines = [
        "# CNE6 风险加权横截面回归实验报告",
        "",
        "## 1. 这次在做什么",
        "",
        "前面我已经检查过两个方向：第一，收益口径从下一日收益换成当天收益以后，平均 R2 只小幅提高；第二，动态风险校验模块主要修正的是风险预测偏差，不会直接改变每日横截面回归 R2。所以这次我单独检查回归权重本身：如果把低特异风险股票放得更重，横截面回归是不是会更接近导师提到的 0.5。",
        "",
        "我没有改正式 CNE6 复现结果，而是新建了一个实验输出目录。这样原始 `sqrt(total_mv)` 基线、风险加权实验和动态风险校验可以分开看。",
        "",
        "## 2. 方法",
        "",
        "回归的因子暴露仍然沿用前一版 CNE6-style 复现：国家因子、9 个风格因子和申万行业哑变量。主要变化是 WLS 拟合权重。所有特异风险信息都只使用 `t-3` 已经能看到的 `specific_risk_60/120/252`；直接倒数和倒方差权重做 1%/99% 截尾，rank 权重则限制在 0.75-1.25 倍之间，避免极少数股票权重过大。",
        "",
            "|fit_weight|含义|我怎么看这个权重|",
            "|---|---|---|",
            "|sqrt_mv|`sqrt(total_mv)`|原始正式基线，偏向更大、更有流动性的股票。|",
            "|equal|等权|不额外偏向大市值股票，用来做参照。|",
            "|sqrt_mv_x_low_specific_rank_60_t3|`sqrt(total_mv) * rank_adjust_60`|用低特异风险排名做 0.75-1.25 倍温和调整，主要看轻微风险降权是否有效。|",
            "|sqrt_mv_x_low_specific_rank_120_t3|`sqrt(total_mv) * rank_adjust_120`|比 60 日更平滑。|",
            "|sqrt_mv_x_low_specific_rank_252_t3|`sqrt(total_mv) * rank_adjust_252`|一年窗口的温和风险降权，权重集中度应该接近原始市值权重。|",
            "|inv_specific_60_t3|`1 / specific_risk_60(t-3)^2`|强调最近 60 日特异噪声低的股票，反应快但也可能更不稳定。|",
            "|inv_specific_120_t3|`1 / specific_risk_120(t-3)^2`|比 60 日平滑一些。|",
            "|inv_specific_252_t3|`1 / specific_risk_252(t-3)^2`|用一年特异风险，最稳，但也最容易形成低波动股票偏置。|",
        "|inv_specific_vol_252_t3|`1 / specific_risk_252(t-3)`|比倒方差更温和，降低权重集中。|",
        "|sqrt_mv_div_specific_vol_252_t3|`sqrt(total_mv) / specific_risk_252(t-3)`|同时考虑市值和特异风险，是更接近工程可用版本的折中方案。|",
        "",
        "评价时我没有只看拟合权重自身的 R2，而是每个 beta 都同时算三种 R2：自权重评价、`sqrt_mv` 评价、等权评价。这里最关键的是交叉评价。如果某个方案只在自权重下很高，但换成 `sqrt_mv` 或等权以后下降，就说明它主要是在低噪声股票上解释得好，不代表整个股票池解释力提高。",
        "",
        "## 3. 总体结论",
        "",
        " ".join(conclusion) if conclusion else "本次实验没有生成可用结论。",
            "",
            f"作为参照，原始 `sqrt_mv` 拟合并用 `sqrt_mv` 评价时，`same_day` 平均 R2 为 {baseline_same:.4f}。",
            "",
            "我的判断是：如果导师关心的是风险模型横截面收益分解能力，最应该看 `same_day + sqrt_mv评价` 或共同样本里的交叉评价结果；如果只看倒特异风险的自权重 R2，很容易把指标权重变化误读成模型能力提升。",
            "",
        "## 4. 当天收益口径：自权重评价",
        "",
            _md_table(same_own, max_rows=20),
            "",
            "这张表回答的是：每个方案在自己最强调的股票集合上解释得怎么样。但我不会把这里最高的 R2 当成最终结论，尤其是当平均有效样本数明显小于平均因子数时，R2 接近 1 反而说明回归自由度已经被权重压坏了。",
        "",
        "## 5. 当天收益口径：市值平方根评价",
        "",
        _md_table(same_sqrt, max_rows=20),
        "",
        "这张表更适合和正式复现基线比较。因为评价权重固定成 `sqrt_mv`，所以不同拟合权重之间的 R2 更可比。",
        "",
        "## 6. 当天收益口径：等权评价",
        "",
        _md_table(same_equal, max_rows=20),
        "",
        "等权评价能检查结果是不是只对大市值股票有效。如果一个方案在自权重下很高，但等权下明显变差，我会把它看成样本权重选择带来的局部效果。",
        "",
        "## 7. 共同样本对比",
        "",
        _md_table(comparable_same_sqrt, max_rows=30),
        "",
        "特异风险权重需要历史滚动窗口，所以早期日期会少一些。共同样本表只保留所有拟合权重都能成功回归的日期，用来减少样本区间不同带来的影响。",
        "",
        "## 8. 分年度表现",
        "",
        _md_table(year_same_sqrt, max_rows=140),
        "",
        "分年度结果主要看 `same_day + sqrt_mv评价`。如果某一年风险加权方案明显好，后续需要结合当年市场状态继续拆：是行业主线更集中、低波动股票更强，还是高特异风险股票的残差冲击太大。",
        "",
        "## 9. 图表",
        "",
    ]
    for fig_name in fig_names:
        lines.extend([f"![](../outputs/cne6_risk_weighted_regression/{fig_name})", ""])
    lines.extend(
        [
            "## 10. 我对 R2 的理解",
            "",
            "这次实验最容易误读的地方是：R2 不是一个脱离权重存在的绝对指标。WLS 的 R2 本来就是按权重计算的。如果我把高特异风险股票权重压得很低，残差平方自然会小很多，自权重 R2 就可能大幅提高。更严重的是，当有效样本数小于因子数时，回归几乎可以在被高权重强调的少数股票上把收益拟合到接近完美，这时 R2 接近 1 不是好消息，而是过拟合和矩阵病态的信号。",
            "",
            "所以我会把结果分成两层：第一层是诊断层，倒特异风险自权重 R2 可以告诉我噪声股票确实在拖累回归；第二层是可比较层，`sqrt_mv` 和等权交叉评价才更接近正式模型能不能改进。如果交叉评价没有提高，我不会建议直接把倒特异风险权重替换进正式 CNE6 复现。",
            "",
            "## 11. 后续可以怎么做",
            "",
            "后续更值得继续检查的是暴露本身，而不是继续调权重把 R2 做高。比如行业分类能不能换成更稳定的中信/申万一级口径，风格因子是否需要正交化，极值处理和缺失填补是否过粗，以及部分风格描述子是否本来就不是 CNE6 商业模型里的真实定义。风险加权可以作为诊断工具保留，但不能单独作为 R2 达标的证据。",
            "",
        ]
    )
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def run(
    panel_path: Path,
    style_path: Path,
    specific_risk_path: Path,
    output: Path,
    doc: Path,
    return_modes: tuple[str, ...] = DEFAULT_RETURN_MODES,
    fit_weight_names: tuple[str, ...] = DEFAULT_FIT_WEIGHTS,
    eval_weight_names: tuple[str, ...] = DEFAULT_EVAL_WEIGHTS,
    risk_lag_days: int = 3,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    style = pd.read_parquet(style_path)
    specific_risk = pd.read_parquet(specific_risk_path)

    daily = run_weighted_regressions(
        panel,
        style,
        specific_risk,
        return_modes=return_modes,
        fit_weight_names=fit_weight_names,
        eval_weight_names=eval_weight_names,
        risk_lag_days=risk_lag_days,
    )
    summary = summarize_daily(daily)
    comparable = comparable_summary(daily, fit_weight_names)
    by_year = summarize_by_year(daily)
    weight_diag = weight_diagnostics_daily(daily)

    daily.to_csv(output / "risk_weighted_regression_daily.csv", index=False)
    summary.to_csv(output / "risk_weighted_regression_summary.csv", index=False)
    comparable.to_csv(output / "risk_weighted_regression_comparable_summary.csv", index=False)
    by_year.to_csv(output / "risk_weighted_regression_by_year.csv", index=False)
    weight_diag.to_csv(output / "risk_weight_diagnostics.csv", index=False)

    fig_names = []
    for fig_name in [
        _save_summary_plot(summary, output),
        _save_cross_eval_timeseries(daily, output),
        _save_weight_concentration(weight_diag, output),
    ]:
        if fig_name:
            fig_names.append(fig_name)
    _write_report(doc, output, summary, comparable, by_year, fig_names)
    print(f"wrote CNE6 risk-weighted regression experiment to {output} and {doc}")


def write_report_from_existing(output: Path, doc: Path) -> None:
    daily = pd.read_csv(output / "risk_weighted_regression_daily.csv", parse_dates=["trade_date"])
    summary = pd.read_csv(output / "risk_weighted_regression_summary.csv", parse_dates=["起始日期", "结束日期"])
    comparable = pd.read_csv(output / "risk_weighted_regression_comparable_summary.csv")
    by_year = pd.read_csv(output / "risk_weighted_regression_by_year.csv")
    weight_diag = pd.read_csv(output / "risk_weight_diagnostics.csv", parse_dates=["trade_date"])
    fig_names = []
    for fig_name in [
        _save_summary_plot(summary, output),
        _save_cross_eval_timeseries(daily, output),
        _save_weight_concentration(weight_diag, output),
    ]:
        if fig_name:
            fig_names.append(fig_name)
    _write_report(doc, output, summary, comparable, by_year, fig_names)
    print(f"wrote CNE6 risk-weighted regression report to {doc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--style", default="outputs/cne6_reproduction/style_exposures.parquet")
    parser.add_argument("--specific-risk", default="outputs/cne6_reproduction/specific_risk.parquet")
    parser.add_argument("--output", default="outputs/cne6_risk_weighted_regression")
    parser.add_argument("--doc", default="docs/cne6_risk_weighted_regression_experiment_2026-08-04.md")
    parser.add_argument("--return-modes", default=",".join(DEFAULT_RETURN_MODES))
    parser.add_argument("--fit-weights", default=",".join(DEFAULT_FIT_WEIGHTS))
    parser.add_argument("--eval-weights", default=",".join(DEFAULT_EVAL_WEIGHTS))
    parser.add_argument("--risk-lag-days", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    doc = Path(args.doc)
    if args.reuse_existing:
        write_report_from_existing(output, doc)
        return

    run(
        Path(args.panel),
        Path(args.style),
        Path(args.specific_risk),
        output,
        doc,
        return_modes=_parse_csv_list(args.return_modes, DEFAULT_RETURN_MODES),
        fit_weight_names=_parse_csv_list(args.fit_weights, DEFAULT_FIT_WEIGHTS),
        eval_weight_names=_parse_csv_list(args.eval_weights, DEFAULT_EVAL_WEIGHTS),
        risk_lag_days=args.risk_lag_days,
    )


if __name__ == "__main__":
    main()
