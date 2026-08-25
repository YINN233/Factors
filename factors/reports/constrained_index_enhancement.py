"""CSI500 XGBoost constrained index-enhancement backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.alpha.public_factors import calculate_public_factors
from factors.alpha.validation import ValidationConfig, add_forward_rank_labels, select_features, validate_factors
from factors.models.xgb_alpha import AlphaModelConfig, train_predict_alpha_model
from factors.portfolio.constrained_optimizer import OptimizerConfig, optimize_constrained_weights, weight_diagnostics
from factors.portfolio.style_exposures import STYLE_COLUMNS, compute_style_exposures
from factors.reports.index_enhancement import YTD_CORE3, make_target_weights, rebalance_dates, summarize_returns
from factors.reports.public_factor_validation import build_validation_panel


def _log(message: str) -> None:
    print(f"[constrained_index_enhancement] {message}", flush=True)


def _read_factor_values(output_root: Path, splits: list[str], factors: list[str]) -> pd.DataFrame:
    frames = []
    for split in splits:
        path = output_root / f"fundamental_alpha_csi500_{split}" / "factor_values.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        keep = ["trade_date", "ts_code"] + [col for col in factors if col in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def _turnover(prev: pd.Series | None, target: pd.Series) -> float:
    if prev is None or prev.empty:
        return float(0.5 * target.abs().sum())
    idx = prev.index.union(target.index)
    return float(0.5 * (target.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum())


def _aligned_sum(weights: pd.Series, returns: pd.Series) -> float:
    aligned = returns.reindex(weights.index).fillna(0.0)
    return float((weights * aligned).sum())


def _drift_weights(weights: pd.Series, returns: pd.Series, portfolio_return: float) -> pd.Series:
    aligned = returns.reindex(weights.index).fillna(0.0)
    if 1.0 + portfolio_return <= 0:
        return weights
    drifted = weights * (1.0 + aligned)
    return drifted / drifted.sum()


def _old_score(panel: pd.DataFrame) -> pd.Series:
    ranks = []
    for factor in YTD_CORE3:
        if factor in panel.columns:
            ranks.append(panel.groupby("trade_date", sort=False)[factor].rank(pct=True) - 0.5)
    if not ranks:
        return pd.Series(0.0, index=panel.index)
    return pd.concat(ranks, axis=1).mean(axis=1)


def run_constrained_backtest(
    panel: pd.DataFrame,
    scenario: str,
    alpha_col: str,
    frequency: str,
    optimizer_config: OptimizerConfig,
    style_cols: list[str],
    cost_bps: float = 5.0,
    old_exp_score: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_by_date = {date: sub for date, sub in panel.groupby("trade_date", sort=False)}
    ret_by_date = {date: sub.set_index("ts_code")["fwd_1d_return"] for date, sub in panel.groupby("trade_date", sort=False)}
    rebal_dates = rebalance_dates(panel["trade_date"], frequency)
    current_w: pd.Series | None = None
    records = []
    weights_records = []

    for date in sorted(panel_by_date):
        sub = panel_by_date[date]
        returns_today = ret_by_date.get(date)
        if returns_today is None:
            continue
        turnover = 0.0
        rebalanced = False
        if current_w is None or date in rebal_dates:
            if old_exp_score:
                target = make_target_weights(sub, score_col=alpha_col, strength=0.25, max_weight=optimizer_config.max_stock_weight)
            else:
                target = optimize_constrained_weights(
                    sub,
                    alpha_col=alpha_col,
                    style_cols=style_cols,
                    prev_weights=current_w,
                    config=optimizer_config,
                )
            turnover = _turnover(current_w, target)
            current_w = target
            rebalanced = True
            for code, weight in current_w.items():
                weights_records.append({"scenario": scenario, "trade_date": date, "ts_code": code, "weight": weight})

        bench = sub.set_index("ts_code")[optimizer_config.benchmark_col].clip(lower=0).astype(float)
        bench = bench / bench.sum()
        portfolio_gross = _aligned_sum(current_w, returns_today)
        benchmark_return = _aligned_sum(bench, returns_today)
        cost = turnover * cost_bps / 10000.0
        portfolio_net = portfolio_gross - cost
        diag = weight_diagnostics(current_w, sub, style_cols=style_cols, config=optimizer_config)
        records.append(
            {
                "scenario": scenario,
                "trade_date": date,
                "portfolio_return_net": portfolio_net,
                "portfolio_return_gross": portfolio_gross,
                "benchmark_return": benchmark_return,
                "active_return": portfolio_net - benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "rebalanced": rebalanced,
                "n_holdings": int((current_w > 1e-8).sum()),
                **diag,
            }
        )
        current_w = _drift_weights(current_w, returns_today, portfolio_gross)

    daily = pd.DataFrame(records)
    weights = pd.DataFrame(weights_records)
    if not daily.empty:
        daily["portfolio_nav"] = (1.0 + daily["portfolio_return_net"]).cumprod()
        daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
        daily["excess_nav"] = daily["portfolio_nav"] / daily["benchmark_nav"]
    return daily, weights


def scenario_configs() -> dict[str, OptimizerConfig]:
    return {
        "xgb_no_style_constraint": OptimizerConfig(max_industry_active=0.02, max_style_active=99.0, max_active_share=0.20),
        "xgb_industry_style_tight": OptimizerConfig(max_industry_active=0.01, max_style_active=0.10, max_active_share=0.12),
        "xgb_industry_style_mid": OptimizerConfig(max_industry_active=0.015, max_style_active=0.20, max_active_share=0.18),
        "xgb_industry_style_loose": OptimizerConfig(max_industry_active=0.025, max_style_active=0.30, max_active_share=0.25),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260706")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--output", default="outputs/csi500_xgb_constrained_index_enhancement")
    parser.add_argument("--frequency", default="monthly")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    _log("building CSI500 validation panel")
    panel = build_validation_panel(Path(args.raw_dir), Path(args.processed_dir), args.start, args.end, splits)
    panel = add_forward_rank_labels(panel, price_col="close_adj", horizons=(1, 5))
    _log(f"panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()}")

    _log("calculating public factors")
    public_values, metadata = calculate_public_factors(panel)
    metadata.to_csv(output_dir / "public_factor_metadata.csv", index=False)

    _log("validating public factors")
    public_cols = [col for col in public_values.columns if col not in {"trade_date", "ts_code"}]
    validation_input = panel[["trade_date", "ts_code", "fwd_5d_rank"]].merge(public_values, on=["trade_date", "ts_code"], how="left")
    public_summary, _ = validate_factors(validation_input, public_cols, config=ValidationConfig(label_col="fwd_5d_rank"))
    selected_public = select_features(public_summary, validation_input, max_pair_corr=0.90)
    public_summary.to_csv(output_dir / "public_factor_validation_summary.csv", index=False)
    pd.DataFrame({"factor": selected_public}).to_csv(output_dir / "selected_public_features.csv", index=False)
    public_values.to_parquet(output_dir / "public_factor_values.parquet", index=False)
    _log(f"public factors computed={len(public_cols)}, selected={len(selected_public)}")

    _log("training alpha model")
    fundamental = _read_factor_values(Path(args.output_root), splits, YTD_CORE3)
    model_panel = panel.merge(public_values, on=["trade_date", "ts_code"], how="left")
    model_panel = model_panel.merge(fundamental, on=["trade_date", "ts_code"], how="left")
    feature_cols = [col for col in selected_public + YTD_CORE3 if col in model_panel.columns]
    if not feature_cols:
        feature_cols = [col for col in YTD_CORE3 if col in model_panel.columns]

    model = train_predict_alpha_model(model_panel, feature_cols, config=AlphaModelConfig(label_col="fwd_5d_rank"))
    model.predictions.to_parquet(output_dir / "xgb_predictions.parquet", index=False)
    model.summary.to_csv(output_dir / "xgb_model_summary.csv", index=False)
    model.feature_importance.to_csv(output_dir / "xgb_feature_importance.csv", index=False)
    _log(f"model backend={model.backend}, features={len(feature_cols)}")

    _log("computing style exposures")
    style = compute_style_exposures(model_panel)
    style.to_parquet(output_dir / "style_exposures.parquet", index=False)
    bt_panel = model_panel.merge(model.predictions[["trade_date", "ts_code", "pred_rank"]], on=["trade_date", "ts_code"], how="left")
    bt_panel = bt_panel.merge(style, on=["trade_date", "ts_code"], how="left")
    bt_panel["score_current_exp"] = _old_score(bt_panel)
    bt_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "csi500_index_weight",
        "fwd_1d_return",
        "pred_rank",
        "score_current_exp",
    ] + [col for col in STYLE_COLUMNS if col in bt_panel.columns]
    bt_panel = bt_panel[[col for col in bt_cols if col in bt_panel.columns]].copy()

    daily_frames = []
    weight_frames = []
    old_cfg = OptimizerConfig(max_stock_weight=0.02)
    _log("running scenario current_exp_score")
    daily, weights = run_constrained_backtest(
        bt_panel,
        "current_exp_score",
        "score_current_exp",
        args.frequency,
        old_cfg,
        style_cols=[],
        cost_bps=args.cost_bps,
        old_exp_score=True,
    )
    daily_frames.append(daily)
    weight_frames.append(weights)

    for scenario, cfg in scenario_configs().items():
        _log(f"running scenario {scenario}")
        styles = [] if scenario == "xgb_no_style_constraint" else STYLE_COLUMNS
        daily, weights = run_constrained_backtest(
            bt_panel,
            scenario,
            "pred_rank",
            args.frequency,
            cfg,
            style_cols=styles,
            cost_bps=args.cost_bps,
        )
        daily_frames.append(daily)
        weight_frames.append(weights)

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_weights = pd.concat(weight_frames, ignore_index=True)
    all_daily.to_csv(output_dir / "constrained_daily_returns.csv", index=False)
    all_weights.to_csv(output_dir / "constrained_weights.csv", index=False)

    _log("summarizing scenarios")
    rows = []
    for scenario, sub in all_daily.groupby("scenario", sort=False):
        for period, start in [("full", None), ("test", "2025-01-01"), ("ytd_2026", "2026-01-01")]:
            row = summarize_returns(sub, period, start=start)
            row["scenario"] = scenario
            row["backend"] = model.backend
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)

    print(f"backend={model.backend}, features={len(feature_cols)}, public_selected={len(selected_public)}")
    print(summary[["scenario", "period", "excess_total_return", "tracking_error", "information_ratio", "active_max_drawdown"]].to_string(index=False))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
