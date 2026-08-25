"""Ablation study for CSI500 constrained index enhancement."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from factors.alpha.validation import add_forward_rank_labels
from factors.models.xgb_alpha import AlphaModelConfig, train_predict_alpha_model
from factors.portfolio.constrained_optimizer import OptimizerConfig
from factors.portfolio.style_exposures import STYLE_COLUMNS, compute_style_exposures
from factors.reports.constrained_index_enhancement import (
    _old_score,
    _read_factor_values,
    run_constrained_backtest,
)
from factors.reports.index_enhancement import YTD_CORE3, summarize_returns
from factors.reports.public_factor_validation import build_validation_panel


def _log(message: str) -> None:
    print(f"[ablation_index_enhancement] {message}", flush=True)


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def _num(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    table = df.copy().fillna("")
    lines = [
        "| " + " | ".join(str(col) for col in table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in table.columns) + " |")
    return "\n".join(lines)


def _format_summary(df: pd.DataFrame) -> str:
    cols = [
        "scenario",
        "period",
        "feature_set",
        "construction",
        "n_features",
        "excess_total_return",
        "tracking_error",
        "information_ratio",
        "active_max_drawdown",
        "daily_active_win_rate",
        "monthly_active_win_rate",
        "avg_active_share",
        "avg_max_industry_active",
    ]
    table = df[[col for col in cols if col in df.columns]].copy()
    for col in [
        "excess_total_return",
        "tracking_error",
        "active_max_drawdown",
        "daily_active_win_rate",
        "monthly_active_win_rate",
        "avg_active_share",
        "avg_max_industry_active",
    ]:
        if col in table:
            table[col] = table[col].map(_pct)
    if "information_ratio" in table:
        table["information_ratio"] = table["information_ratio"].map(_num)
    return _markdown_table(table)


def _plot_excess_nav(daily: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "ablation_excess_nav.png"
    plt.figure(figsize=(11, 6))
    for scenario, sub in daily.groupby("scenario", sort=False):
        sub = sub.sort_values("trade_date")
        plt.plot(sub["trade_date"], sub["excess_nav"], label=scenario, linewidth=1.0)
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.legend(fontsize=7, ncol=2)
    plt.title("Ablation Scenario Excess NAV")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_period_bars(summary: pd.DataFrame, output_dir: Path, period: str) -> str:
    path = output_dir / f"ablation_{period}_excess_return.png"
    sub = summary[summary["period"] == period].copy()
    if sub.empty:
        return ""
    sub = sub.sort_values("excess_total_return")
    plt.figure(figsize=(11, max(4, 0.35 * len(sub))))
    plt.barh(sub["scenario"], sub["excess_total_return"], color="#4c78a8")
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title(f"Ablation {period} Excess Total Return")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _feature_sets(selected_public: list[str], model_panel: pd.DataFrame) -> dict[str, list[str]]:
    public = [col for col in selected_public if col in model_panel.columns]
    fundamental = [col for col in YTD_CORE3 if col in model_panel.columns]
    return {
        "fundamental_only": fundamental,
        "public_only": public,
        "public_plus_fundamental": public + fundamental,
    }


def _scenario_specs(feature_sets: dict[str, list[str]]) -> list[dict]:
    specs: list[dict] = [
        {
            "scenario": "old_core3_score_tilt",
            "feature_set": "old_core3_score",
            "construction": "score_tilt",
            "score_col": "score_current_exp",
            "model_key": None,
            "old_exp_score": True,
            "style_cols": [],
            "config": OptimizerConfig(max_stock_weight=0.02),
            "n_features": len(YTD_CORE3),
        },
        {
            "scenario": "old_core3_tight_constraint",
            "feature_set": "old_core3_score",
            "construction": "industry_style_tight",
            "score_col": "score_current_exp",
            "model_key": None,
            "old_exp_score": False,
            "style_cols": STYLE_COLUMNS,
            "config": OptimizerConfig(max_industry_active=0.01, max_style_active=0.10, max_active_share=0.12),
            "n_features": len(YTD_CORE3),
        },
    ]
    for key, features in feature_sets.items():
        specs.extend(
            [
                {
                    "scenario": f"xgb_{key}_score_tilt",
                    "feature_set": key,
                    "construction": "score_tilt",
                    "score_col": f"pred_rank_{key}",
                    "model_key": key,
                    "old_exp_score": True,
                    "style_cols": [],
                    "config": OptimizerConfig(max_stock_weight=0.02),
                    "n_features": len(features),
                },
                {
                    "scenario": f"xgb_{key}_tight_constraint",
                    "feature_set": key,
                    "construction": "industry_style_tight",
                    "score_col": f"pred_rank_{key}",
                    "model_key": key,
                    "old_exp_score": False,
                    "style_cols": STYLE_COLUMNS,
                    "config": OptimizerConfig(max_industry_active=0.01, max_style_active=0.10, max_active_share=0.12),
                    "n_features": len(features),
                },
            ]
        )
    return specs


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    feature_sets: pd.DataFrame,
    report_path: Path,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    nav_png = _plot_excess_nav(pd.read_csv(output_dir / "ablation_daily_returns.csv", parse_dates=["trade_date"]), output_dir)
    test_png = _plot_period_bars(summary, output_dir, "test")
    ytd_png = _plot_period_bars(summary, output_dir, "ytd_2026")

    test = summary[summary["period"] == "test"].copy()
    ytd = summary[summary["period"] == "ytd_2026"].copy()
    best_test = test.sort_values("information_ratio", ascending=False).head(1)
    public_tight = test[test["scenario"] == "xgb_public_only_tight_constraint"]
    combined_tight = test[test["scenario"] == "xgb_public_plus_fundamental_tight_constraint"]
    fundamental_tight = test[test["scenario"] == "xgb_fundamental_only_tight_constraint"]
    backends = sorted(model_summary["backend"].dropna().unique().tolist()) if "backend" in model_summary else []
    if backends == ["xgboost"]:
        backend_note = "- 本实验使用原生 `XGBRegressor`，不是 sklearn fallback；模型后端变化会影响与旧报告的数值可比性。"
    else:
        backend_text = ", ".join(backends) if backends else "unknown"
        backend_note = f"- 本实验模型后端为 `{backend_text}`；如果不是 `xgboost`，需要明确它是替代模型结果。"

    lines = [
        "# 中证500指增严格对照实验",
        "",
        "日期：2026-07-13",
        "",
        "## 结论",
        "",
        "- 本实验不新增因子，只改变特征集合和组合构造方式，用来拆解公开因子、旧基本面因子、行业/风格约束分别贡献了什么。",
        f"- 测试期 IR 最高的场景是 `{best_test['scenario'].iloc[0] if not best_test.empty else ''}`，测试期超额 {_pct(best_test['excess_total_return'].iloc[0]) if not best_test.empty else ''}，IR {_num(best_test['information_ratio'].iloc[0]) if not best_test.empty else ''}。",
        f"- 公开因子-only + tight 约束测试期超额 {_pct(public_tight['excess_total_return'].iloc[0]) if not public_tight.empty else ''}；基本面-only + tight 约束测试期超额 {_pct(fundamental_tight['excess_total_return'].iloc[0]) if not fundamental_tight.empty else ''}；公开+基本面 + tight 约束测试期超额 {_pct(combined_tight['excess_total_return'].iloc[0]) if not combined_tight.empty else ''}。",
        "- 如果 score tilt 强而 tight 约束弱，说明信号更像选股排序；如果 tight 仍稳，才说明它能进入指数增强框架。本轮重点看后者。",
        "",
        "## 特征集合",
        "",
        _markdown_table(feature_sets),
        "",
        "## 场景表现",
        "",
        _format_summary(summary),
        "",
        f"![](../{output_dir}/{nav_png})" if nav_png else "",
        "",
        f"![](../{output_dir}/{test_png})" if test_png else "",
        "",
        f"![](../{output_dir}/{ytd_png})" if ytd_png else "",
        "",
        "## 模型 RankIC",
        "",
        _markdown_table(model_summary),
        "",
        "## 批判性解读",
        "",
        "- `score_tilt` 只做基准权重上的指数倾斜和个股上限，基本不控制行业/风格，是多因子选股更接近的形态。",
        "- `industry_style_tight` 同时约束行业、风格和 active share，更接近导师要求的指数增强框架。",
        backend_note,
        "",
        "## 输出文件",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 日度回测：`{output_dir / 'ablation_daily_returns.csv'}`",
        f"- 方案摘要：`{output_dir / 'ablation_scenario_summary.csv'}`",
        f"- 模型摘要：`{output_dir / 'ablation_model_summary.csv'}`",
        f"- 调仓权重：`{output_dir / 'ablation_weights.csv'}`",
    ]
    report_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--source-output", default="outputs/csi500_xgb_constrained_index_enhancement")
    parser.add_argument("--output", default="outputs/csi500_xgb_ablation_index_enhancement")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260706")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--frequency", default="monthly")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--report", default="docs/csi500_xgb_ablation_index_enhancement_2026-07-13.md")
    args = parser.parse_args()

    output_dir = Path(args.output)
    source_dir = Path(args.source_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    _log("building labeled panel")
    panel = build_validation_panel(Path(args.raw_dir), Path(args.processed_dir), args.start, args.end, splits)
    panel = add_forward_rank_labels(panel, price_col="close_adj", horizons=(1, 5))

    _log("loading existing public and fundamental factors")
    public_values = pd.read_parquet(source_dir / "public_factor_values.parquet")
    public_values["trade_date"] = pd.to_datetime(public_values["trade_date"])
    selected_public = pd.read_csv(source_dir / "selected_public_features.csv")["factor"].tolist()
    fundamental = _read_factor_values(Path(args.output_root), splits, YTD_CORE3)
    model_panel = panel.merge(public_values, on=["trade_date", "ts_code"], how="left")
    model_panel = model_panel.merge(fundamental, on=["trade_date", "ts_code"], how="left")
    feature_sets = _feature_sets(selected_public, model_panel)
    feature_table = pd.DataFrame(
        {
            "feature_set": key,
            "n_features": len(features),
            "features": ", ".join(features[:12]) + (" ..." if len(features) > 12 else ""),
        }
        for key, features in feature_sets.items()
    )
    feature_table.to_csv(output_dir / "ablation_feature_sets.csv", index=False)

    _log("training ablation models")
    predictions = {}
    model_rows = []
    importance_frames = []
    for key, features in feature_sets.items():
        if not features:
            continue
        model = train_predict_alpha_model(model_panel, features, config=AlphaModelConfig(label_col="fwd_5d_rank"))
        pred_col = f"pred_rank_{key}"
        predictions[key] = model.predictions[["trade_date", "ts_code", "pred_rank"]].rename(columns={"pred_rank": pred_col})
        model.summary.assign(feature_set=key, backend=model.backend, n_features=len(features)).to_csv(
            output_dir / f"model_summary_{key}.csv", index=False
        )
        model.predictions.to_parquet(output_dir / f"predictions_{key}.parquet", index=False)
        model.feature_importance.assign(feature_set=key).to_csv(output_dir / f"feature_importance_{key}.csv", index=False)
        model_rows.append(model.summary.assign(feature_set=key, backend=model.backend, n_features=len(features)))
        importance_frames.append(model.feature_importance.assign(feature_set=key))

    model_summary = pd.concat(model_rows, ignore_index=True)
    model_summary.to_csv(output_dir / "ablation_model_summary.csv", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(output_dir / "ablation_feature_importance.csv", index=False)

    _log("computing shared style exposures")
    style = compute_style_exposures(model_panel)
    bt_panel = model_panel[
        ["trade_date", "ts_code", "industry", "csi500_index_weight", "fwd_1d_return"]
    ].copy()
    bt_panel = bt_panel.merge(style, on=["trade_date", "ts_code"], how="left")
    bt_panel["score_current_exp"] = _old_score(model_panel)
    for pred in predictions.values():
        bt_panel = bt_panel.merge(pred, on=["trade_date", "ts_code"], how="left")

    _log("running ablation backtests")
    daily_frames = []
    weight_frames = []
    rows = []
    specs = _scenario_specs(feature_sets)
    for spec in specs:
        if spec["score_col"] not in bt_panel.columns:
            continue
        daily, weights = run_constrained_backtest(
            bt_panel,
            spec["scenario"],
            spec["score_col"],
            args.frequency,
            spec["config"],
            style_cols=spec["style_cols"],
            cost_bps=args.cost_bps,
            old_exp_score=spec["old_exp_score"],
        )
        daily_frames.append(daily)
        weight_frames.append(weights)
        for period, start in [("full", None), ("test", "2025-01-01"), ("ytd_2026", "2026-01-01")]:
            row = summarize_returns(daily, period, start=start)
            row.update(
                {
                    "scenario": spec["scenario"],
                    "feature_set": spec["feature_set"],
                    "construction": spec["construction"],
                    "n_features": spec["n_features"],
                }
            )
            rows.append(row)

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_weights = pd.concat(weight_frames, ignore_index=True)
    summary = pd.DataFrame(rows)
    all_daily.to_csv(output_dir / "ablation_daily_returns.csv", index=False)
    all_weights.to_csv(output_dir / "ablation_weights.csv", index=False)
    summary.to_csv(output_dir / "ablation_scenario_summary.csv", index=False)

    report_path = _write_report(output_dir, summary, model_summary, feature_table, Path(args.report))
    print(summary[summary["period"].eq("test")][["scenario", "excess_total_return", "tracking_error", "information_ratio", "active_max_drawdown"]].to_string(index=False))
    print(f"wrote outputs to {output_dir}")
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
