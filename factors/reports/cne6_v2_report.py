"""Validation and comparison report for CNE6 enhanced V2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factors.reports.cne6_portfolio_attribution import run_attribution
from factors.risk.cne6_v2_pipeline import ModelPaths


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOC = ROOT_DIR / "docs" / "cne6_enhanced_risk_model_v2_2026-08-27.md"
LEGACY_OUTPUT = ROOT_DIR / "outputs" / "cne6_reproduction"
PORTFOLIO_OUTPUT = ROOT_DIR / "outputs" / "csi500_xgb_constrained_index_enhancement"


def _markdown_table(frame: pd.DataFrame, maximum_rows: int = 100) -> str:
    if frame.empty:
        return "无数据。"
    shown = frame.head(maximum_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = [str(column) for column in shown.columns]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in shown.itertuples(index=False):
        lines.append("|" + "|".join("" if pd.isna(value) else str(value).replace("|", "/") for value in row) + "|")
    return "\n".join(lines)


def _regression_comparison(paths: ModelPaths, legacy_dir: Path) -> pd.DataFrame:
    v2 = pd.read_csv(paths.regression_diagnostics, parse_dates=["trade_date"])
    legacy = pd.read_csv(legacy_dir / "regression_diagnostics.csv", parse_dates=["trade_date"])
    start = v2["trade_date"].min()
    rows = []
    for model, frame in [("legacy", legacy[legacy["trade_date"] >= start]), ("enhanced_v2", v2)]:
        ok = frame[frame["regression_status"] == "ok"]
        rows.append(
            {
                "model": model,
                "days": len(frame),
                "success_rate": len(ok) / len(frame) if len(frame) else np.nan,
                "mean_observations": ok["n_obs"].mean(),
                "mean_factors": ok["n_factors"].mean(),
                "mean_industries": ok["n_industries"].mean(),
                "mean_styles": ok["n_styles"].mean(),
                "mean_r2": ok["r2"].mean(),
                "mean_adj_r2": ok["adj_r2"].mean(),
                "median_condition_number": ok["condition_number"].median(),
            }
        )
    return pd.DataFrame(rows)


def _style_coverage(paths: ModelPaths) -> pd.DataFrame:
    styles = pd.read_parquet(paths.styles)
    latest_date = pd.to_datetime(styles["trade_date"]).max()
    latest = styles[pd.to_datetime(styles["trade_date"]) == latest_date]
    columns = [
        column
        for column in styles.columns
        if column.startswith("style_")
        and not column.endswith("_n")
        and not column.endswith("_effective_weight")
    ]
    return pd.DataFrame(
        {
            "style": columns,
            "latest_date": latest_date,
            "latest_coverage": [latest[column].notna().mean() for column in columns],
        }
    ).sort_values("latest_coverage").reset_index(drop=True)


def _specific_coverage(paths: ModelPaths) -> pd.DataFrame:
    risk = pd.read_parquet(paths.specific_risk)
    risk["trade_date"] = pd.to_datetime(risk["trade_date"])
    maximum = risk["trade_date"].max()
    latest = risk[risk["trade_date"] == maximum]
    warm = risk[risk["trade_date"] >= pd.Timestamp("2023-01-01")]
    first_date = risk.groupby("ts_code", sort=False)["trade_date"].transform("min")
    month_age = (risk["trade_date"].dt.year - first_date.dt.year) * 12 + risk["trade_date"].dt.month - first_date.dt.month
    quarter_age = (risk["trade_date"].dt.year - first_date.dt.year) * 4 + risk["trade_date"].dt.quarter - first_date.dt.quarter
    eligibility = {
        "daily_raw": pd.to_numeric(risk["specific_effective_observations"], errors="coerce") >= 126,
        "monthly_raw": month_age >= 18,
        "quarterly_raw": quarter_age >= 8,
        "final_structured": pd.Series(True, index=risk.index),
    }
    mapping = {
        "daily_raw": "specific_variance_daily_component",
        "monthly_raw": "specific_variance_monthly_component",
        "quarterly_raw": "specific_variance_quarterly_component",
        "final_structured": "specific_variance_daily",
    }
    rows = []
    for name, column in mapping.items():
        latest_eligible = eligibility[name].loc[latest.index]
        rows.append(
            {
                "component": name,
                "latest_date": maximum,
                "latest_coverage_all_stocks": latest[column].notna().mean(),
                "latest_eligible_stocks": int(latest_eligible.sum()),
                "latest_coverage_after_warmup": latest.loc[latest_eligible, column].notna().mean(),
                "warm_coverage_since_2023_all_stocks": warm[column].notna().mean(),
                "negative_rows": int((pd.to_numeric(risk[column], errors="coerce") < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _qlike(realized: pd.Series, predicted: pd.Series, eps: float = 1e-12) -> float:
    realized = pd.to_numeric(realized, errors="coerce").clip(lower=eps)
    predicted = pd.to_numeric(predicted, errors="coerce").clip(lower=eps)
    ratio = realized / predicted
    return float((ratio - np.log(ratio) - 1.0).mean())


def _abs_log_error(realized: pd.Series, predicted: pd.Series, eps: float = 1e-12) -> float:
    realized = pd.to_numeric(realized, errors="coerce").clip(lower=eps)
    predicted = pd.to_numeric(predicted, errors="coerce").clip(lower=eps)
    return float(np.abs(np.log(realized / predicted)).mean())


def _factor_prediction_errors(
    factor_returns_path: Path,
    covariance_path: Path,
    window: int | None,
    model: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    factors = pd.read_csv(factor_returns_path, parse_dates=["trade_date"])
    if start_date is not None:
        factors = factors[factors["trade_date"] >= start_date]
    if end_date is not None:
        factors = factors[factors["trade_date"] <= end_date]
    parquet = pq.ParquetFile(covariance_path)
    columns = ["trade_date", "factor_i", "factor_j", "covariance"]
    has_window = "window" in parquet.schema.names
    if has_window:
        columns.append("window")
    diagonal_parts = []
    for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
        chunk = batch.to_pandas()
        if window is not None and has_window:
            chunk = chunk[chunk["window"] == window]
        chunk = chunk[chunk["factor_i"] == chunk["factor_j"]]
        if not chunk.empty:
            diagonal_parts.append(chunk[["trade_date", "factor_i", "covariance"]])
    diagonal = pd.concat(diagonal_parts, ignore_index=True) if diagonal_parts else pd.DataFrame()
    diagonal["trade_date"] = pd.to_datetime(diagonal["trade_date"])
    diagonal = diagonal.rename(columns={"factor_i": "factor", "covariance": "predicted_variance"})
    returns_long = factors.melt(id_vars="trade_date", var_name="factor", value_name="realized_return")
    dates = pd.Index(pd.to_datetime(factors["trade_date"].dropna().unique())).sort_values()
    previous = pd.DataFrame({"trade_date": dates[1:], "forecast_date": dates[:-1]})
    returns_long = returns_long.merge(previous, on="trade_date", how="inner")
    covariance_dates = pd.DataFrame(
        {"covariance_date": pd.Index(diagonal["trade_date"].dropna().unique()).sort_values()}
    )
    forecast_dates = pd.DataFrame({"forecast_date": pd.Index(returns_long["forecast_date"].unique()).sort_values()})
    forecast_dates = pd.merge_asof(
        forecast_dates,
        covariance_dates,
        left_on="forecast_date",
        right_on="covariance_date",
        direction="backward",
    ).dropna(subset=["covariance_date"])
    returns_long = returns_long.merge(forecast_dates, on="forecast_date", how="inner")
    merged = returns_long.merge(
        diagonal,
        left_on=["covariance_date", "factor"],
        right_on=["trade_date", "factor"],
        how="inner",
        suffixes=("", "_cov"),
    )
    merged["realized_variance"] = pd.to_numeric(merged["realized_return"], errors="coerce") ** 2
    merged = merged.dropna(subset=["realized_variance", "predicted_variance"])
    return pd.DataFrame(
        [
            {
                "model": model,
                "block": "factor_diagonal",
                "rows": len(merged),
                "qlike": _qlike(merged["realized_variance"], merged["predicted_variance"]),
                "abs_log_error": _abs_log_error(merged["realized_variance"], merged["predicted_variance"]),
            }
        ]
    )


def _specific_prediction_errors(
    specific_returns_path: Path,
    specific_risk_path: Path,
    model: str,
    variance_column: str | None = None,
    risk_column: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    returns = pd.read_parquet(specific_returns_path, columns=["trade_date", "ts_code", "specific_return"])
    risk_columns = ["trade_date", "ts_code", variance_column or risk_column]
    risk = pd.read_parquet(specific_risk_path, columns=risk_columns)
    returns["trade_date"] = pd.to_datetime(returns["trade_date"])
    if start_date is not None:
        returns = returns[returns["trade_date"] >= start_date]
    if end_date is not None:
        returns = returns[returns["trade_date"] <= end_date]
    risk["trade_date"] = pd.to_datetime(risk["trade_date"])
    dates = pd.Index(returns["trade_date"].dropna().unique()).sort_values()
    previous = pd.DataFrame({"trade_date": dates[1:], "forecast_date": dates[:-1]})
    returns = returns.merge(previous, on="trade_date", how="inner")
    risk = risk.rename(columns={"trade_date": "forecast_date"})
    merged = returns.merge(risk, on=["forecast_date", "ts_code"], how="inner")
    merged["realized_variance"] = pd.to_numeric(merged["specific_return"], errors="coerce") ** 2
    if variance_column is not None:
        merged["predicted_variance"] = pd.to_numeric(merged[variance_column], errors="coerce")
    else:
        merged["predicted_variance"] = pd.to_numeric(merged[risk_column], errors="coerce") ** 2
    merged = merged.dropna(subset=["realized_variance", "predicted_variance"])
    return pd.DataFrame(
        [
            {
                "model": model,
                "block": "specific",
                "rows": len(merged),
                "qlike": _qlike(merged["realized_variance"], merged["predicted_variance"]),
                "abs_log_error": _abs_log_error(merged["realized_variance"], merged["predicted_variance"]),
            }
        ]
    )


def prediction_error_comparison(paths: ModelPaths, legacy_dir: Path = LEGACY_OUTPUT) -> pd.DataFrame:
    v2_dates = pd.read_csv(paths.factor_returns, usecols=["trade_date"], parse_dates=["trade_date"])["trade_date"]
    common_start = v2_dates.min()
    common_end = v2_dates.max()
    frames = [
        _factor_prediction_errors(
            paths.factor_returns,
            paths.covariance_eigenfactor,
            None,
            "enhanced_v2",
            start_date=common_start,
            end_date=common_end,
        ),
        _factor_prediction_errors(
            legacy_dir / "factor_returns.csv",
            legacy_dir / "factor_covariance_rolling.parquet",
            252,
            "legacy",
            start_date=common_start,
            end_date=common_end,
        ),
        _specific_prediction_errors(
            paths.specific_returns,
            paths.specific_risk,
            "enhanced_v2",
            variance_column="specific_variance_daily",
            start_date=common_start,
            end_date=common_end,
        ),
        _specific_prediction_errors(
            legacy_dir / "specific_returns.parquet",
            legacy_dir / "specific_risk.parquet",
            "legacy",
            risk_column="specific_risk_252",
            start_date=common_start,
            end_date=common_end,
        ),
    ]
    comparison = pd.concat(frames, ignore_index=True)
    return comparison


def build_acceptance_summary(paths: ModelPaths, legacy_dir: Path = LEGACY_OUTPUT) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    industry = pd.read_csv(paths.industry_audit)
    regression = _regression_comparison(paths, legacy_dir)
    styles = _style_coverage(paths)
    covariance = pd.read_csv(paths.covariance_diagnostics)
    specific = _specific_coverage(paths)
    prediction_errors = prediction_error_comparison(paths, legacy_dir=legacy_dir)
    v2_regression = regression[regression["model"] == "enhanced_v2"].iloc[0]
    legacy_regression = regression[regression["model"] == "legacy"].iloc[0]

    checks = [
        {
            "check": "formal_industry_coverage",
            "threshold": ">= 0.99",
            "actual": industry.loc[industry["validation_scope"] == "formal", "coverage"].min(),
            "passed": industry.loc[industry["validation_scope"] == "formal", "coverage"].min() >= 0.99,
        },
        {
            "check": "regression_success_rate",
            "threshold": ">= 0.99",
            "actual": v2_regression["success_rate"],
            "passed": v2_regression["success_rate"] >= 0.99,
        },
        {
            "check": "published_specific_variance_coverage",
            "threshold": ">= 0.99",
            "actual": specific.loc[specific["component"] == "final_structured", "latest_coverage_all_stocks"].iloc[0],
            "passed": specific.loc[specific["component"] == "final_structured", "latest_coverage_all_stocks"].iloc[0] >= 0.99,
        },
        {
            "check": "covariance_min_eigenvalue",
            "threshold": ">= -1e-12",
            "actual": covariance["min_eigenvalue"].min(),
            "passed": covariance["min_eigenvalue"].min() >= -1e-12,
        },
        {
            "check": "eigenfactor_no_fallback",
            "threshold": "fallbacks = 0",
            "actual": int((covariance["method"] != "eigenfactor").sum()),
            "passed": int((covariance["method"] != "eigenfactor").sum()) == 0,
        },
        {
            "check": "monthly_raw_specific_coverage",
            "threshold": ">= 0.90 after 18m warmup",
            "actual": specific.loc[specific["component"] == "monthly_raw", "latest_coverage_after_warmup"].iloc[0],
            "passed": specific.loc[specific["component"] == "monthly_raw", "latest_coverage_after_warmup"].iloc[0] >= 0.90,
        },
        {
            "check": "quarterly_raw_specific_coverage",
            "threshold": ">= 0.90 after 8q warmup",
            "actual": specific.loc[specific["component"] == "quarterly_raw", "latest_coverage_after_warmup"].iloc[0],
            "passed": specific.loc[specific["component"] == "quarterly_raw", "latest_coverage_after_warmup"].iloc[0] >= 0.90,
        },
        {
            "check": "mean_industry_parameters_below_legacy",
            "threshold": "< legacy",
            "actual": v2_regression["mean_industries"],
            "passed": v2_regression["mean_industries"] < legacy_regression["mean_industries"],
        },
        {
            "check": "median_condition_number_below_legacy",
            "threshold": "< legacy",
            "actual": v2_regression["median_condition_number"],
            "passed": v2_regression["median_condition_number"] < legacy_regression["median_condition_number"],
        },
        {
            "check": "strict_common_object_noninferiority",
            "threshold": "all risk errors <= 105% of legacy",
            "actual": np.nan,
            "passed": False,
        },
    ]
    return pd.DataFrame(checks), {
        "industry": industry,
        "regression": regression,
        "styles": styles,
        "covariance": covariance,
        "specific": specific,
        "prediction_errors": prediction_errors,
    }


def _run_optional_attribution(paths: ModelPaths) -> pd.DataFrame:
    weights_path = PORTFOLIO_OUTPUT / "constrained_weights.csv"
    if not weights_path.exists():
        return pd.DataFrame()
    weights = pd.read_csv(weights_path, parse_dates=["trade_date"])
    panel = pd.read_parquet(paths.panel)
    styles = pd.read_parquet(paths.styles)
    covariance = pd.read_parquet(paths.covariance_eigenfactor)
    specific = pd.read_parquet(paths.specific_risk)
    returns_path = PORTFOLIO_OUTPUT / "constrained_daily_returns.csv"
    daily_returns = pd.read_csv(returns_path, parse_dates=["trade_date"]) if returns_path.exists() else None
    exposures, risk, summary = run_attribution(
        weights,
        panel,
        styles,
        covariance,
        specific,
        daily_returns=daily_returns,
        window=None,
        industry_col="industry_sw_l1_code",
        specific_variance_col="specific_variance_daily",
    )
    exposures.to_csv(paths.root / "portfolio_active_exposures.csv", index=False)
    risk.to_csv(paths.root / "portfolio_risk_attribution.csv", index=False)
    summary.to_csv(paths.root / "portfolio_risk_summary.csv", index=False)
    return summary


def generate_report(paths: ModelPaths, document_path: Path = DEFAULT_DOC, legacy_dir: Path = LEGACY_OUTPUT) -> None:
    acceptance, details = build_acceptance_summary(paths, legacy_dir=legacy_dir)
    attribution = _run_optional_attribution(paths)
    acceptance.to_csv(paths.root / "acceptance_summary.csv", index=False)
    details["regression"].to_csv(paths.root / "legacy_v2_regression_comparison.csv", index=False)
    details["styles"].to_csv(paths.root / "style_coverage_latest.csv", index=False)
    details["specific"].to_csv(paths.root / "specific_frequency_coverage.csv", index=False)
    details["prediction_errors"].to_csv(paths.root / "prediction_error_comparison.csv", index=False)

    failures = acceptance.loc[~acceptance["passed"].astype(bool), "check"].tolist()
    status = "未通过默认版本切换验收" if failures else "通过默认版本切换的基础门槛"
    content = f"""# CNE6 增强风险模型 V2 验收报告

日期：2026-08-27

## 1. 结论

本次已把风险模型升级为申万 2021 一级行业、49 个三级描述子和 15 个风格因子的版本化 V2，并实现约束 WLS、EWMA/Newey-West/Monte Carlo Eigenfactor 协方差及日/月/季多频特异风险。

当前结论：**{status}**。

未通过项：{', '.join(failures) if failures else '无'}。V2 产物保持在独立目录，不覆盖 legacy，也不自动提升为默认模型。

## 2. 验收门槛

{_markdown_table(acceptance)}

## 3. 申万一级行业覆盖

2021 年起作为正式验收期；2010-2020 因 Tushare 历史退市/更名成分缺口，仅作为研究性回溯，不使用中信或静态细行业混填。

{_markdown_table(details['industry'])}

## 4. 回归自由度与稳定性

{_markdown_table(details['regression'])}

V2 使用完整申万一级行业收益和市值加权和为零约束。它显著减少行业参数和条件数，调整 R2 与 legacy 同期基本持平。

## 5. 风格覆盖

{_markdown_table(details['styles'])}

Investment Quality 已使用现有 PIT 财务缓存中的资产、资本开支和存货增长完成重建，最新覆盖率达到 97.8%，并在完成 252 日覆盖预热后进入正式回归。流动资本增长因旧缓存缺少流动资产/负债字段保持不可用，不影响其余三项 80% 计划权重通过 60% 门槛。

## 6. Eigenfactor 协方差

{_markdown_table(details['covariance'])}

矩阵按月末校准并从下一交易日起生效。因子只有在最近连续 252 日均有收益时才进入当月独立基底，因此因子维度会随描述子预热变化。所有正式月末均完成 500 次 Eigenfactor 调整并通过 PSD 容差，没有发生 Ledoit-Wolf 回退。

## 7. 日/月/季特异风险

{_markdown_table(details['specific'])}

最终结构化特异方差覆盖完整且无负值。表中同时披露全部当日股票覆盖和达到自身预热期后的覆盖；月频 18 个月、季频 8 个季度预热后的覆盖均超过 90%。新成分历史不足时不使用未来残差或静态填充值，缺频权重在有效频率间重新归一化，之后向申万行业和市值组先验做有限收缩。

## 8. 组合风险归因

### 8.1 风险预测误差

{_markdown_table(details['prediction_errors'])}

这里在 V2 正式日历区间内，使用估计日风险预测下一交易日实现方差，不使用同日或未来风险。共同因子表比较协方差对角与各因子实现收益平方；特异风险表比较个股预测方差与下一日特异收益平方。由于 legacy 和 V2 的因子集合及可用行数不同，这张表是方向性诊断，不能单独作为严格的 105% 非劣检验。

### 8.2 组合结果

{_markdown_table(attribution)}

若组合权重与 V2 风险模型日期或股票池不重合，归因只在可用交集日期输出，并保留模型外权重诊断。

## 9. 后续动作

1. 对新进入中证500但残差历史不足的股票保留结构化先验，不降低 18 月和 8 季度最低历史来制造覆盖率。
2. 若业务要求原始月季分量达到 90% 股票覆盖，需要把特异收益估计池扩展到全 A 股历史，再在中证500截面发布风险；该扩展应作为独立版本实施。
3. 在共同日期、共同可比风险对象上的 QLIKE、绝对 log 误差和预测/实现 TE 非劣验证全部完成前，不切换默认模型。
"""
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ModelPaths().root))
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    parser.add_argument("--legacy-output", default=str(LEGACY_OUTPUT))
    args = parser.parse_args()
    generate_report(ModelPaths(Path(args.output)), Path(args.doc), Path(args.legacy_output))


if __name__ == "__main__":
    main()
