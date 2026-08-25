"""Dynamic validation and calibration for local CNE6-style risk forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BLOCK_ORDER = ["factor_all", "factor_country", "factor_style", "factor_industry", "specific"]
BLOCK_CN = {
    "factor_all": "共同因子整体",
    "factor_country": "国家因子",
    "factor_style": "风格因子",
    "factor_industry": "行业因子",
    "specific": "个股特异风险",
}


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "无数据。"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    columns = [str(c) for c in show.columns]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in show.itertuples(index=False):
        values = ["" if pd.isna(item) else str(item).replace("|", "/") for item in row]
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def _factor_block(factor: str) -> str:
    if factor == "country":
        return "factor_country"
    if factor.startswith("style_"):
        return "factor_style"
    if factor.startswith("industry_"):
        return "factor_industry"
    return "factor_other"


def _trading_calendar(*frames: pd.DataFrame) -> pd.Index:
    dates: list[pd.Timestamp] = []
    for frame in frames:
        if "trade_date" in frame.columns:
            dates.extend(pd.to_datetime(frame["trade_date"].dropna().unique()).tolist())
    return pd.Index(sorted(pd.unique(pd.Series(dates))))


def forecast_mapping(trade_dates: pd.Series | pd.Index, lag_days: int = 3) -> pd.DataFrame:
    dates = pd.Index(pd.to_datetime(pd.Series(trade_dates).dropna().unique())).sort_values()
    if len(dates) <= lag_days:
        return pd.DataFrame(columns=["target_date", "forecast_asof_date"])
    return pd.DataFrame(
        {
            "target_date": dates[lag_days:],
            "forecast_asof_date": dates[:-lag_days],
        }
    )


def add_next_available_date(validation: pd.DataFrame, calendar: pd.Index) -> pd.DataFrame:
    validation = validation.copy()
    calendar = pd.Index(pd.to_datetime(calendar)).sort_values()
    pos = calendar.searchsorted(pd.to_datetime(validation["target_date"]), side="right")
    validation["available_date"] = pd.NaT
    valid = pos < len(calendar)
    values = pd.Series(pd.NaT, index=validation.index, dtype="datetime64[ns]")
    values.loc[valid] = calendar.take(pos[valid])
    validation["available_date"] = values
    return validation


def _latest_covariance_dates(cov_diag: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    cov_dates = pd.Index(pd.to_datetime(cov_diag["trade_date"].dropna().unique())).sort_values()
    mapping = mapping.copy()
    pos = cov_dates.searchsorted(pd.to_datetime(mapping["forecast_asof_date"]), side="right") - 1
    mapping["covariance_date"] = pd.NaT
    valid = pos >= 0
    mapping.loc[valid, "covariance_date"] = cov_dates.take(pos[valid])
    return mapping.dropna(subset=["covariance_date"])


def build_factor_block_validation(
    factor_returns: pd.DataFrame,
    covariance: pd.DataFrame,
    lag_days: int = 3,
    risk_window: int = 252,
    eps: float = 1e-12,
) -> pd.DataFrame:
    factor_returns = factor_returns.copy()
    factor_returns["trade_date"] = pd.to_datetime(factor_returns["trade_date"])
    mapping = forecast_mapping(factor_returns["trade_date"], lag_days=lag_days)
    if mapping.empty:
        return pd.DataFrame()

    covariance = covariance.copy()
    covariance["trade_date"] = pd.to_datetime(covariance["trade_date"])
    cov_diag = covariance[
        (covariance["window"] == risk_window)
        & (covariance["factor_i"] == covariance["factor_j"])
    ][["trade_date", "factor_i", "covariance"]].copy()
    cov_diag = cov_diag.rename(columns={"factor_i": "factor", "covariance": "predicted_variance"})
    cov_diag["predicted_variance"] = pd.to_numeric(cov_diag["predicted_variance"], errors="coerce")
    mapping = _latest_covariance_dates(cov_diag, mapping)
    if mapping.empty:
        return pd.DataFrame()

    factor_cols = [c for c in factor_returns.columns if c != "trade_date"]
    realized = factor_returns.melt(id_vars="trade_date", value_vars=factor_cols, var_name="factor", value_name="realized_return")
    realized = realized.rename(columns={"trade_date": "target_date"})
    realized["realized_return"] = pd.to_numeric(realized["realized_return"], errors="coerce")
    realized = realized.dropna(subset=["realized_return"])
    realized = realized.merge(mapping, on="target_date", how="inner")
    realized = realized.merge(
        cov_diag,
        left_on=["covariance_date", "factor"],
        right_on=["trade_date", "factor"],
        how="inner",
        suffixes=("", "_cov"),
    )
    realized = realized[realized["predicted_variance"] > eps].copy()
    if realized.empty:
        return pd.DataFrame()
    realized["realized_variance"] = realized["realized_return"] ** 2
    realized["block"] = realized["factor"].map(_factor_block)
    realized = realized[realized["block"].isin(["factor_country", "factor_style", "factor_industry"])]

    rows = []
    group_cols = ["target_date", "forecast_asof_date", "block"]
    block = (
        realized.groupby(group_cols, sort=True)
        .agg(
            predicted_variance=("predicted_variance", "sum"),
            realized_variance=("realized_variance", "sum"),
            n_items=("factor", "nunique"),
        )
        .reset_index()
    )
    rows.append(block)
    all_block = (
        realized.groupby(["target_date", "forecast_asof_date"], sort=True)
        .agg(
            predicted_variance=("predicted_variance", "sum"),
            realized_variance=("realized_variance", "sum"),
            n_items=("factor", "nunique"),
        )
        .reset_index()
    )
    all_block["block"] = "factor_all"
    rows.append(all_block)
    out = pd.concat(rows, ignore_index=True)
    out["raw_ratio"] = out["realized_variance"] / out["predicted_variance"].clip(lower=eps)
    out["risk_window"] = risk_window
    out["lag_days"] = lag_days
    return out[["target_date", "forecast_asof_date", "block", "risk_window", "lag_days", "predicted_variance", "realized_variance", "raw_ratio", "n_items"]]


def build_specific_block_validation(
    specific_returns: pd.DataFrame,
    specific_risk: pd.DataFrame,
    trade_dates: pd.Series | pd.Index,
    lag_days: int = 3,
    risk_window: int = 252,
    eps: float = 1e-12,
) -> pd.DataFrame:
    mapping = forecast_mapping(trade_dates, lag_days=lag_days)
    if mapping.empty:
        return pd.DataFrame()
    sr_col = f"specific_risk_{risk_window}"
    specific_returns = specific_returns[["trade_date", "ts_code", "specific_return"]].copy()
    specific_returns["trade_date"] = pd.to_datetime(specific_returns["trade_date"])
    specific_returns["specific_return"] = pd.to_numeric(specific_returns["specific_return"], errors="coerce")
    specific_returns = specific_returns.dropna(subset=["specific_return"])
    specific_returns = specific_returns.rename(columns={"trade_date": "target_date"})
    specific_returns = specific_returns.merge(mapping, on="target_date", how="inner")

    specific_risk = specific_risk[["trade_date", "ts_code", sr_col]].copy()
    specific_risk["trade_date"] = pd.to_datetime(specific_risk["trade_date"])
    specific_risk = specific_risk.rename(columns={"trade_date": "forecast_asof_date", sr_col: "specific_risk"})
    specific_risk["specific_risk"] = pd.to_numeric(specific_risk["specific_risk"], errors="coerce")

    merged = specific_returns.merge(specific_risk, on=["forecast_asof_date", "ts_code"], how="inner")
    merged = merged.dropna(subset=["specific_risk"])
    merged["predicted_variance"] = merged["specific_risk"] ** 2
    merged = merged[merged["predicted_variance"] > eps].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["realized_variance"] = merged["specific_return"] ** 2
    out = (
        merged.groupby(["target_date", "forecast_asof_date"], sort=True)
        .agg(
            predicted_variance=("predicted_variance", "sum"),
            realized_variance=("realized_variance", "sum"),
            n_items=("ts_code", "nunique"),
        )
        .reset_index()
    )
    out["block"] = "specific"
    out["raw_ratio"] = out["realized_variance"] / out["predicted_variance"].clip(lower=eps)
    out["risk_window"] = risk_window
    out["lag_days"] = lag_days
    return out[["target_date", "forecast_asof_date", "block", "risk_window", "lag_days", "predicted_variance", "realized_variance", "raw_ratio", "n_items"]]


def build_calibrated_forecasts(
    validation: pd.DataFrame,
    calibration_windows: tuple[int, ...] = (20, 40, 60, 126),
    ratio_clip: tuple[float, float] = (0.05, 20.0),
    multiplier_clip: tuple[float, float] = (0.25, 4.0),
    eps: float = 1e-12,
) -> pd.DataFrame:
    validation = validation.copy()
    validation["target_date"] = pd.to_datetime(validation["target_date"]).astype("datetime64[ns]")
    validation["forecast_asof_date"] = pd.to_datetime(validation["forecast_asof_date"]).astype("datetime64[ns]")
    validation["available_date"] = pd.to_datetime(validation["available_date"]).astype("datetime64[ns]")
    validation["ratio_for_multiplier"] = validation["raw_ratio"].clip(lower=ratio_clip[0], upper=ratio_clip[1])
    validation["realized_variance_for_multiplier"] = validation["predicted_variance"] * validation["ratio_for_multiplier"]

    frames = []
    base = validation.copy()
    base["calibration_window"] = 0
    base["calibration_multiplier"] = 1.0
    frames.append(base)

    for window in calibration_windows:
        calibrated_parts = []
        for block, sub in validation.groupby("block", sort=False):
            hist = sub[["available_date", "predicted_variance", "realized_variance_for_multiplier"]].dropna().sort_values("available_date").copy()
            min_periods = min(window, max(5, window // 4))
            rolling_predicted = hist["predicted_variance"].rolling(window, min_periods=min_periods).sum()
            rolling_realized = hist["realized_variance_for_multiplier"].rolling(window, min_periods=min_periods).sum()
            hist["calibration_multiplier"] = (rolling_realized / rolling_predicted.clip(lower=eps)).clip(
                lower=multiplier_clip[0],
                upper=multiplier_clip[1],
            )
            left = sub.sort_values("forecast_asof_date").copy()
            merged = pd.merge_asof(
                left,
                hist[["available_date", "calibration_multiplier"]].dropna().sort_values("available_date"),
                left_on="forecast_asof_date",
                right_on="available_date",
                direction="backward",
            )
            merged = merged.drop(columns=["available_date_y"]).rename(columns={"available_date_x": "available_date"})
            merged["block"] = block
            merged["calibration_window"] = window
            calibrated_parts.append(merged)
        frames.append(pd.concat(calibrated_parts, ignore_index=True) if calibrated_parts else pd.DataFrame())

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["calibration_multiplier"]).copy()
    out["corrected_predicted_variance"] = out["predicted_variance"] * out["calibration_multiplier"]
    out["corrected_ratio"] = out["realized_variance"] / out["corrected_predicted_variance"].clip(lower=eps)
    out["raw_abs_log_error"] = np.abs(np.log(out["predicted_variance"].clip(lower=eps) / out["realized_variance"].clip(lower=eps)))
    out["corrected_abs_log_error"] = np.abs(np.log(out["corrected_predicted_variance"].clip(lower=eps) / out["realized_variance"].clip(lower=eps)))
    return out


def summarize_calibration(calibrated: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    rows = []
    for (block, window), sub in calibrated.groupby(["block", "calibration_window"], sort=True):
        raw_bias = sub["realized_variance"].sum() / max(float(sub["predicted_variance"].sum()), eps)
        corrected_bias = sub["realized_variance"].sum() / max(float(sub["corrected_predicted_variance"].sum()), eps)
        rows.append(
            {
                "block": block,
                "中文风险块": BLOCK_CN.get(block, block),
                "calibration_window": int(window),
                "dates": int(sub["target_date"].nunique()),
                "raw_bias_ratio": raw_bias,
                "corrected_bias_ratio": corrected_bias,
                "raw_abs_log_error": float(sub["raw_abs_log_error"].mean()),
                "corrected_abs_log_error": float(sub["corrected_abs_log_error"].mean()),
                "log_error_improvement": float(sub["raw_abs_log_error"].mean() - sub["corrected_abs_log_error"].mean()),
                "multiplier_mean": float(sub["calibration_multiplier"].mean()),
                "multiplier_median": float(sub["calibration_multiplier"].median()),
            }
        )
    out = pd.DataFrame(rows)
    out["block_order"] = out["block"].map({block: idx for idx, block in enumerate(BLOCK_ORDER)}).fillna(99)
    return out.sort_values(["block_order", "calibration_window"]).drop(columns=["block_order"]).reset_index(drop=True)


def summarize_validation_distribution(validation: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    rows = []
    for block, sub in validation.groupby("block", sort=True):
        rows.append(
            {
                "block": block,
                "中文风险块": BLOCK_CN.get(block, block),
                "dates": int(sub["target_date"].nunique()),
                "aggregate_bias_ratio": float(sub["realized_variance"].sum() / max(float(sub["predicted_variance"].sum()), eps)),
                "mean_daily_ratio": float(sub["raw_ratio"].mean()),
                "median_daily_ratio": float(sub["raw_ratio"].median()),
                "p75_daily_ratio": float(sub["raw_ratio"].quantile(0.75)),
                "p95_daily_ratio": float(sub["raw_ratio"].quantile(0.95)),
                "p99_daily_ratio": float(sub["raw_ratio"].quantile(0.99)),
                "max_daily_ratio": float(sub["raw_ratio"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["block_order"] = out["block"].map({block: idx for idx, block in enumerate(BLOCK_ORDER)}).fillna(99)
    return out.sort_values(["block_order"]).drop(columns=["block_order"]).reset_index(drop=True)


def summarize_calibration_by_year(calibrated: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    work = calibrated.copy()
    work["year"] = pd.to_datetime(work["target_date"]).dt.year
    rows = []
    for (block, window, year), sub in work.groupby(["block", "calibration_window", "year"], sort=True):
        raw_bias = sub["realized_variance"].sum() / max(float(sub["predicted_variance"].sum()), eps)
        corrected_bias = sub["realized_variance"].sum() / max(float(sub["corrected_predicted_variance"].sum()), eps)
        rows.append(
            {
                "block": block,
                "中文风险块": BLOCK_CN.get(block, block),
                "calibration_window": int(window),
                "year": int(year),
                "dates": int(sub["target_date"].nunique()),
                "raw_bias_ratio": raw_bias,
                "corrected_bias_ratio": corrected_bias,
                "corrected_abs_log_error": float(sub["corrected_abs_log_error"].mean()),
                "multiplier_mean": float(sub["calibration_multiplier"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["block_order"] = out["block"].map({block: idx for idx, block in enumerate(BLOCK_ORDER)}).fillna(99)
    return out.sort_values(["block_order", "calibration_window", "year"]).drop(columns=["block_order"]).reset_index(drop=True)


def _save_bias_plot(summary: pd.DataFrame, output: Path) -> str:
    import matplotlib.pyplot as plt

    plot = summary[summary["calibration_window"].isin([0, 20, 40, 60, 126])].copy()
    blocks = [b for b in BLOCK_ORDER if b in set(plot["block"])]
    windows = sorted(plot["calibration_window"].unique())
    x = np.arange(len(blocks))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, window in enumerate(windows):
        vals = []
        for block in blocks:
            sub = plot[(plot["block"] == block) & (plot["calibration_window"] == window)]
            vals.append(float(sub["corrected_bias_ratio"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (idx - (len(windows) - 1) / 2) * width, vals, width=width, label=f"w={window}")
    ax.axhline(1.0, color="#D62728", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=20, ha="right")
    ax.set_ylabel("realized / corrected predicted variance")
    ax.set_title("CNE6 dynamic calibration bias by block")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = output / "calibration_bias_by_window.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _save_error_plot(summary: pd.DataFrame, output: Path) -> str:
    import matplotlib.pyplot as plt

    plot = summary[summary["calibration_window"].isin([0, 20, 40, 60, 126])].copy()
    blocks = [b for b in BLOCK_ORDER if b in set(plot["block"])]
    windows = sorted(plot["calibration_window"].unique())
    x = np.arange(len(blocks))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, window in enumerate(windows):
        vals = []
        for block in blocks:
            sub = plot[(plot["block"] == block) & (plot["calibration_window"] == window)]
            vals.append(float(sub["corrected_abs_log_error"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (idx - (len(windows) - 1) / 2) * width, vals, width=width, label=f"w={window}")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=20, ha="right")
    ax.set_ylabel("mean abs log error")
    ax.set_title("CNE6 dynamic calibration error by block")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = output / "calibration_error_by_window.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _best_window_table(summary: pd.DataFrame) -> pd.DataFrame:
    nonzero = summary[summary["calibration_window"] > 0].copy()
    if nonzero.empty:
        return pd.DataFrame()
    best = nonzero.sort_values(["block", "corrected_abs_log_error"]).groupby("block", sort=False).head(1)
    return best[
        [
            "block",
            "中文风险块",
            "calibration_window",
            "corrected_bias_ratio",
            "corrected_abs_log_error",
            "log_error_improvement",
            "multiplier_mean",
        ]
    ].reset_index(drop=True)


def write_report(
    doc: Path,
    output: Path,
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    distribution: pd.DataFrame,
    bias_fig: str,
    error_fig: str,
) -> None:
    best = _best_window_table(summary)
    baseline = summary[summary["calibration_window"] == 0][
        ["block", "中文风险块", "dates", "raw_bias_ratio", "raw_abs_log_error"]
    ].copy()
    focus = summary[
        summary["block"].isin(["factor_all", "factor_style", "factor_industry", "specific"])
    ][
        [
            "block",
            "中文风险块",
            "calibration_window",
            "dates",
            "corrected_bias_ratio",
            "corrected_abs_log_error",
            "log_error_improvement",
            "multiplier_mean",
        ]
    ].copy()
    by_year_focus = by_year[
        (by_year["block"].isin(["factor_all", "specific"]))
        & (by_year["calibration_window"].isin([0, 20, 60, 126]))
    ].copy()
    lines = [
        "# CNE6 风险预测校验与动态修正报告",
        "",
        "## 1. 这次在做什么",
        "",
        "导师提到的校验和动态修正，我理解为风险模型预测之后的事后反馈机制：先在 `t-3` 用当时能看到的数据预测 `t` 的风险，等 `t+1` 真实收益和残差都出来以后，再回头看 `t` 的风险预测是偏高还是偏低，并把近期偏差滚动反馈到后面的预测里。",
        "",
        "这次没有覆盖原始 CNE6 风险模型文件，而是在原始 252 日 Ledoit-Wolf 因子协方差和个股特异风险上加一层校准乘数。这样原始模型、校验结果和修正结果可以分开看。",
        "",
        "## 2. 方法",
        "",
        "对目标日 `t`，预测时点设成 `t-3`。共同因子风险用 `t-3` 之前最近一期 252 日协方差对角线作为预测方差，用 `t` 日因子收益平方作为实现方差代理；个股特异风险用 `t-3` 的 `specific_risk_252` 作为预测方差，用 `t` 日特异收益平方作为实现方差代理。`t` 日偏差只在下一个交易日之后进入滚动校准，避免未来函数。",
        "",
        "动态修正用的公式很简单：",
        "",
        "```text",
        "校准乘数 = 最近 N 个已经可见交易日的 sum(实现方差) / sum(预测方差)",
        "修正后预测方差 = 原始预测方差 * 校准乘数",
        "偏差比率 = 实现方差 / 预测方差",
        "```",
        "",
        "这里比较了 20、40、60、126 个交易日窗口，其中 126 日约等于导师提到的六个月，20/40/60 用来观察更短校验窗口能不能更快适应市场状态。",
        "",
        "## 3. 原始预测偏差",
        "",
        _md_table(baseline, max_rows=20),
        "",
        "这里 `raw_bias_ratio` 大于 1 表示原始风险预测偏低，小于 1 表示原始风险预测偏高。",
        "",
        "## 4. 偏差分布",
        "",
        _md_table(distribution, max_rows=20),
        "",
        "这张表是理解结果的关键。比如某些风险块的总体 `aggregate_bias_ratio` 大于 1，但 `median_daily_ratio` 小于 1，说明大多数普通日期反而是预测偏高，只有少数极端日期预测严重偏低，并且这些极端日期把总体偏差拉上去了。所以动态修正不能只盯着一个平均数，需要同时看日度误差和尾部偏差。",
        "",
        "## 5. 动态修正总体结果",
        "",
        _md_table(focus, max_rows=80),
        "",
        "## 6. 每个风险块日度 log 误差最低窗口",
        "",
        _md_table(best, max_rows=20),
        "",
        "## 7. 分年度结果",
        "",
        _md_table(by_year_focus, max_rows=80),
        "",
        "## 8. 图表",
        "",
        f"![](../outputs/cne6_dynamic_risk_calibration/{bias_fig})",
        "",
        f"![](../outputs/cne6_dynamic_risk_calibration/{error_fig})",
        "",
        "## 9. 我的理解",
        "",
        "这套校验模块的价值不在于把风险模型变成收益预测模型，而是检查风险预测有没有系统性偏差。如果某个风险块长期 `realized / predicted` 大于 1，就说明这个风险块原来被低估；如果长期小于 1，就说明原来偏保守。动态修正相当于在原始风险模型外面加一个可解释的反馈环。",
        "",
        "我会比较谨慎地看短窗口结果：20 日窗口反应最快，当前在 log 误差上通常最好，但它不一定能修好总体偏差，因为总体偏差经常由极少数尾部日期决定。126 日窗口更稳，但如果市场状态快速变化，它可能调整太慢。后续如果要接入组合归因，我更倾向先把这个模块作为诊断层使用，再单独检验校准后预测 TE 和实现 TE 是否更接近，而不是直接宣称乘数修正已经让风险模型完全变准。",
        "",
        "## 10. 后续可以怎么接",
        "",
        "下一步可以把 `daily_calibrated_forecasts.csv` 里的校准乘数接到组合风险归因里：共同因子协方差按对应因子块乘以 factor multiplier，个股特异风险按 specific multiplier 调整。这样能检验校准后预测 TE 和实现 TE 是否更接近。",
        "",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def run(
    factor_returns_path: Path,
    factor_covariance_path: Path,
    specific_returns_path: Path,
    specific_risk_path: Path,
    output: Path,
    doc: Path,
    lag_days: int = 3,
    risk_window: int = 252,
    calibration_windows: tuple[int, ...] = (20, 40, 60, 126),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    factor_returns = pd.read_csv(factor_returns_path, parse_dates=["trade_date"])
    covariance = pd.read_parquet(
        factor_covariance_path,
        columns=["trade_date", "window", "factor_i", "factor_j", "covariance"],
    )
    specific_returns = pd.read_parquet(specific_returns_path, columns=["trade_date", "ts_code", "specific_return"])
    specific_risk = pd.read_parquet(specific_risk_path, columns=["trade_date", "ts_code", f"specific_risk_{risk_window}"])

    factor_validation = build_factor_block_validation(
        factor_returns,
        covariance,
        lag_days=lag_days,
        risk_window=risk_window,
    )
    specific_validation = build_specific_block_validation(
        specific_returns,
        specific_risk,
        factor_returns["trade_date"],
        lag_days=lag_days,
        risk_window=risk_window,
    )
    validation = pd.concat([factor_validation, specific_validation], ignore_index=True)
    calendar = _trading_calendar(factor_returns, specific_returns)
    validation = add_next_available_date(validation, calendar)
    validation = validation.dropna(subset=["available_date"]).copy()
    calibrated = build_calibrated_forecasts(validation, calibration_windows=calibration_windows)
    summary = summarize_calibration(calibrated)
    distribution = summarize_validation_distribution(validation)
    by_year = summarize_calibration_by_year(calibrated)

    validation.to_csv(output / "daily_block_validation.csv", index=False)
    calibrated.to_csv(output / "daily_calibrated_forecasts.csv", index=False)
    summary.to_csv(output / "calibration_summary.csv", index=False)
    distribution.to_csv(output / "validation_ratio_distribution.csv", index=False)
    by_year.to_csv(output / "calibration_by_year.csv", index=False)
    bias_fig = _save_bias_plot(summary, output)
    error_fig = _save_error_plot(summary, output)
    write_report(doc, output, summary, by_year, distribution, bias_fig, error_fig)
    print(f"wrote CNE6 dynamic risk calibration outputs to {output} and {doc}")


def _parse_windows(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-returns", default="outputs/cne6_reproduction/factor_returns.csv")
    parser.add_argument("--factor-covariance", default="outputs/cne6_reproduction/factor_covariance_rolling.parquet")
    parser.add_argument("--specific-returns", default="outputs/cne6_reproduction/specific_returns.parquet")
    parser.add_argument("--specific-risk", default="outputs/cne6_reproduction/specific_risk.parquet")
    parser.add_argument("--output", default="outputs/cne6_dynamic_risk_calibration")
    parser.add_argument("--doc", default="docs/cne6_dynamic_risk_validation_calibration_2026-08-04.md")
    parser.add_argument("--lag-days", type=int, default=3)
    parser.add_argument("--risk-window", type=int, default=252)
    parser.add_argument("--calibration-windows", default="20,40,60,126")
    args = parser.parse_args()
    run(
        Path(args.factor_returns),
        Path(args.factor_covariance),
        Path(args.specific_returns),
        Path(args.specific_risk),
        Path(args.output),
        Path(args.doc),
        lag_days=args.lag_days,
        risk_window=args.risk_window,
        calibration_windows=_parse_windows(args.calibration_windows),
    )


if __name__ == "__main__":
    main()
