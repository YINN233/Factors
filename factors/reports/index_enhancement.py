"""
CSI500 index-enhancement backtest for mined fundamental factors.

Signals are observed at close T.  Target weights are formed at close T and are
applied to T -> T+1 close-to-close returns.  The benchmark is the daily
normalized CSI500 constituent weight from the point-in-time panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RECOMMENDED8 = [
    "dupont_value_quality",
    "cash_buffer_value",
    "robust_margin_value_ps",
    "growth_value_balance",
    "eps_bps_value_quality",
    "quality_growth_hmean",
    "industry_neutral_roe_value_pb",
    "liquidity_solvency_value",
]

YTD_CORE3 = [
    "eps_bps_value_quality",
    "quality_growth_hmean",
    "industry_neutral_roe_value_pb",
]


def _read_existing_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema.names)
    return pd.read_parquet(path, columns=[col for col in columns if col in schema])


def _cap_and_normalize(weights: pd.Series, max_weight: float | None = None) -> pd.Series:
    weights = weights.clip(lower=0).astype(float)
    if weights.sum() <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    weights = weights / weights.sum()
    if max_weight is None or max_weight <= 0:
        return weights

    capped = pd.Series(0.0, index=weights.index, dtype=float)
    remaining = weights.copy()
    remaining_budget = 1.0
    while not remaining.empty:
        scaled = remaining / remaining.sum() * remaining_budget
        over = scaled > max_weight
        if not over.any():
            capped.loc[scaled.index] = scaled
            break
        capped.loc[scaled.index[over]] = max_weight
        remaining_budget = 1.0 - capped.sum()
        remaining = remaining.loc[~over]
        if remaining_budget <= 1e-12:
            break
    return capped / capped.sum()


def load_signal_panel(
    processed_dir: Path,
    output_root: Path,
    splits: Iterable[str],
    factors: list[str],
    suffix: str = "000905_SH",
) -> pd.DataFrame:
    frames = []
    for split in splits:
        factor_path = output_root / f"fundamental_alpha_csi500_{split}" / "factor_values.parquet"
        base_path = processed_dir / f"{split}_fundamental_{suffix}.parquet"
        factor_df = _read_existing_columns(factor_path, ["trade_date", "ts_code"] + factors)
        base_df = _read_existing_columns(
            base_path,
            [
                "trade_date",
                "ts_code",
                "industry",
                "total_mv",
                "log_mv",
                "csi500_index_weight",
            ],
        )
        df = factor_df.merge(base_df, on=["trade_date", "ts_code"], how="left")
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def load_forward_returns(
    raw_dir: Path,
    start: str,
    end: str,
    ts_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    daily_path = raw_dir / f"daily_{start}_{end}.parquet"
    adj_path = raw_dir / f"adj_factor_{start}_{end}.parquet"
    daily = pd.read_parquet(daily_path, columns=["trade_date", "ts_code", "close"])
    adj = pd.read_parquet(adj_path, columns=["trade_date", "ts_code", "adj_factor"])
    if ts_codes is not None:
        ts_codes = set(ts_codes)
        daily = daily[daily["ts_code"].isin(ts_codes)]
        adj = adj[adj["ts_code"].isin(ts_codes)]
    prices = daily.merge(adj, on=["trade_date", "ts_code"], how="left")
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["close_adj"] = prices["close"] * prices["adj_factor"]
    prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    prices["fwd_return_1d"] = prices.groupby("ts_code", sort=False)["close_adj"].shift(-1) / prices["close_adj"] - 1.0
    return prices[["trade_date", "ts_code", "fwd_return_1d"]].dropna(subset=["fwd_return_1d"])


def add_scores(panel: pd.DataFrame, factor_sets: dict[str, list[str]]) -> pd.DataFrame:
    panel = panel.copy()
    for name, factors in factor_sets.items():
        ranks = []
        for factor in factors:
            if factor not in panel.columns:
                continue
            ranks.append(panel.groupby("trade_date", sort=False)[factor].rank(pct=True) - 0.5)
        if ranks:
            panel[f"score_{name}"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=True)
    return panel


def rebalance_dates(dates: pd.Series, frequency: str) -> set[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(sorted(pd.unique(dates))))
    if frequency == "daily":
        return set(unique)
    if frequency == "weekly":
        return set(unique.groupby(unique.dt.to_period("W-FRI")).max())
    if frequency == "monthly":
        return set(unique.groupby(unique.dt.to_period("M")).max())
    raise ValueError("frequency must be one of: daily, weekly, monthly")


def make_target_weights(
    sub: pd.DataFrame,
    score_col: str,
    strength: float,
    max_weight: float | None,
) -> pd.Series:
    sub = sub.dropna(subset=["ts_code"]).copy()
    base = sub["csi500_index_weight"].astype(float).clip(lower=0)
    if base.sum() <= 0:
        base = pd.Series(1.0, index=sub.index)
    base = base / base.sum()

    score = sub[score_col].astype(float).fillna(0.0)
    std = score.std()
    if pd.isna(std) or std == 0:
        z = pd.Series(0.0, index=sub.index)
    else:
        z = (score - score.mean()) / std
    raw = base * np.exp(strength * z.clip(-4, 4))
    target = _cap_and_normalize(pd.Series(raw.to_numpy(), index=sub["ts_code"].to_numpy()), max_weight=max_weight)
    return target


def _aligned_sum(weights: pd.Series, returns: pd.Series) -> float:
    aligned = returns.reindex(weights.index)
    if aligned.empty:
        return 0.0
    return float((weights * aligned.fillna(0.0)).sum())


def _drift_weights(weights: pd.Series, returns: pd.Series, portfolio_return: float) -> pd.Series:
    aligned = returns.reindex(weights.index).fillna(0.0)
    if aligned.empty or 1.0 + portfolio_return <= 0:
        return weights
    drifted = weights * (1.0 + aligned)
    drifted = drifted / drifted.sum()
    return drifted


def _turnover(prev: pd.Series | None, target: pd.Series) -> float:
    if prev is None or prev.empty:
        return float(0.5 * target.abs().sum())
    idx = prev.index.union(target.index)
    return float(0.5 * (target.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum())


def _industry_active(
    target: pd.Series,
    sub: pd.DataFrame,
) -> tuple[float, float]:
    info = sub.set_index("ts_code")[["industry", "csi500_index_weight"]].copy()
    if info.empty:
        return np.nan, np.nan
    bench = info["csi500_index_weight"].clip(lower=0)
    bench = bench / bench.sum()
    port = target.rename("weight").to_frame().join(info[["industry"]], how="left")
    bench_ind = info.assign(bench=bench).groupby("industry")["bench"].sum()
    port_ind = port.groupby("industry")["weight"].sum()
    active = port_ind.sub(bench_ind, fill_value=0.0)
    return float(active.abs().max()), float(active.abs().sum() / 2.0)


def run_backtest(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    score_col: str,
    frequency: str,
    strength: float,
    max_weight: float | None = 0.02,
    cost_bps: float = 5.0,
    compute_daily_industry: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_by_date = {date: sub for date, sub in panel.groupby("trade_date", sort=False)}
    ret_by_date = {
        date: sub.set_index("ts_code")["fwd_return_1d"]
        for date, sub in returns.groupby("trade_date", sort=False)
    }
    rebal_dates = rebalance_dates(panel["trade_date"], frequency)
    current_w: pd.Series | None = None
    records = []
    weight_records = []

    for date in sorted(panel_by_date):
        sub = panel_by_date[date]
        returns_today = ret_by_date.get(date)
        if returns_today is None:
            continue

        turnover = 0.0
        rebalanced = False
        if current_w is None or date in rebal_dates:
            target = make_target_weights(sub, score_col=score_col, strength=strength, max_weight=max_weight)
            turnover = _turnover(current_w, target)
            current_w = target
            rebalanced = True
            for code, weight in current_w.items():
                weight_records.append({"trade_date": date, "ts_code": code, "weight": weight})

        bench = sub.set_index("ts_code")["csi500_index_weight"].astype(float).clip(lower=0)
        bench = bench / bench.sum()
        portfolio_return_gross = _aligned_sum(current_w, returns_today)
        benchmark_return = _aligned_sum(bench, returns_today)
        cost = turnover * cost_bps / 10000.0
        portfolio_return_net = portfolio_return_gross - cost
        bench_on_port = bench.reindex(current_w.index, fill_value=0.0)
        bench_not_held = bench.loc[~bench.index.isin(current_w.index)].sum()
        active_share = 0.5 * ((current_w - bench_on_port).abs().sum() + bench_not_held)
        if compute_daily_industry:
            max_industry_active, industry_active_share = _industry_active(current_w, sub)
        else:
            max_industry_active, industry_active_share = np.nan, np.nan
        records.append(
            {
                "trade_date": date,
                "portfolio_return_gross": portfolio_return_gross,
                "portfolio_return_net": portfolio_return_net,
                "benchmark_return": benchmark_return,
                "active_return": portfolio_return_net - benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "active_share": active_share,
                "max_industry_active": max_industry_active,
                "industry_active_share": industry_active_share,
                "n_holdings": int((current_w > 1e-8).sum()),
                "rebalanced": rebalanced,
            }
        )
        current_w = _drift_weights(current_w, returns_today, portfolio_return_gross)

    daily = pd.DataFrame(records)
    weights = pd.DataFrame(weight_records)
    if not daily.empty:
        daily["portfolio_nav"] = (1.0 + daily["portfolio_return_net"]).cumprod()
        daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
        daily["excess_nav"] = daily["portfolio_nav"] / daily["benchmark_nav"]
    return daily, weights


def summarize_returns(daily: pd.DataFrame, period_name: str, start: str | None = None, end: str | None = None) -> dict:
    if daily.empty:
        return {}
    df = daily.copy()
    if start:
        df = df[df["trade_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["trade_date"] <= pd.Timestamp(end)]
    if df.empty:
        return {}
    active = df["active_return"].dropna()
    port_total = float((1.0 + df["portfolio_return_net"]).prod() - 1.0)
    bench_total = float((1.0 + df["benchmark_return"]).prod() - 1.0)
    excess_total = float((1.0 + df["portfolio_return_net"]).prod() / (1.0 + df["benchmark_return"]).prod() - 1.0)
    n = len(df)
    ann_factor = 252.0 / n
    active_nav = (1.0 + active).cumprod()
    drawdown = active_nav / active_nav.cummax() - 1.0
    month = df.assign(month=df["trade_date"].dt.to_period("M"))
    monthly = month.groupby("month").apply(
        lambda x: (1.0 + x["portfolio_return_net"]).prod() / (1.0 + x["benchmark_return"]).prod() - 1.0,
        include_groups=False,
    )
    return {
        "period": period_name,
        "start_date": df["trade_date"].min(),
        "end_date": df["trade_date"].max(),
        "n_days": n,
        "portfolio_total_return": port_total,
        "benchmark_total_return": bench_total,
        "excess_total_return": excess_total,
        "portfolio_annual_return": (1.0 + port_total) ** ann_factor - 1.0,
        "benchmark_annual_return": (1.0 + bench_total) ** ann_factor - 1.0,
        "annual_excess_arith": active.mean() * 252.0,
        "tracking_error": active.std(ddof=1) * np.sqrt(252.0),
        "information_ratio": active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252.0) + 1e-12),
        "active_max_drawdown": float(drawdown.min()),
        "daily_active_win_rate": float((active > 0).mean()),
        "monthly_active_win_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "avg_daily_turnover": df["turnover"].mean(),
        "avg_rebalance_turnover": df.loc[df["rebalanced"], "turnover"].mean(),
        "avg_active_share": df["active_share"].mean(),
        "avg_max_industry_active": df["max_industry_active"].mean(),
        "avg_n_holdings": df["n_holdings"].mean(),
    }


def latest_weight_tables(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    stock_company_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_date = weights["trade_date"].max()
    latest_w = weights[weights["trade_date"] == latest_date].copy()
    latest_panel = panel[panel["trade_date"] == latest_date].copy()
    latest_panel = latest_panel.rename(columns={"csi500_index_weight": "benchmark_weight"})
    latest_panel["benchmark_weight"] = latest_panel["benchmark_weight"].clip(lower=0)
    latest_panel["benchmark_weight"] = latest_panel["benchmark_weight"] / latest_panel["benchmark_weight"].sum()
    out = latest_w.merge(
        latest_panel[["ts_code", "industry", "total_mv", "benchmark_weight"]],
        on="ts_code",
        how="left",
    )
    if stock_company_path.exists():
        names = pd.read_parquet(stock_company_path, columns=["ts_code", "name"])
        out = out.merge(names, on="ts_code", how="left")
    out["active_weight"] = out["weight"] - out["benchmark_weight"].fillna(0.0)
    out = out.sort_values("weight", ascending=False)
    ind = (
        out.groupby("industry")
        .agg(portfolio_weight=("weight", "sum"), benchmark_weight=("benchmark_weight", "sum"), n_holdings=("ts_code", "count"))
        .reset_index()
    )
    ind["active_weight"] = ind["portfolio_weight"] - ind["benchmark_weight"]
    ind = ind.sort_values("active_weight", ascending=False)
    return out, ind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260706")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output", default="outputs/fundamental_alpha_csi500_index_enhancement")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--strengths", default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--frequencies", default="weekly,monthly")
    parser.add_argument("--max-weight", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    frequencies = [x.strip() for x in args.frequencies.split(",") if x.strip()]
    factor_sets = {"recommended8": RECOMMENDED8, "ytd_core3": YTD_CORE3}
    all_factors = sorted(set(RECOMMENDED8 + YTD_CORE3))

    panel = load_signal_panel(
        processed_dir=Path(args.processed_dir),
        output_root=Path(args.output_root),
        splits=splits,
        factors=all_factors,
    )
    panel = add_scores(panel, factor_sets)
    returns = load_forward_returns(
        raw_dir=Path(args.raw_dir),
        start=args.start,
        end=args.end,
        ts_codes=panel["ts_code"].unique(),
    )

    scenario_rows = []
    period_rows = []
    daily_by_scenario = {}
    weights_by_scenario = {}
    for factor_set in factor_sets:
        score_col = f"score_{factor_set}"
        for frequency in frequencies:
            for strength in strengths:
                daily, weights = run_backtest(
                    panel=panel,
                    returns=returns,
                    score_col=score_col,
                    frequency=frequency,
                    strength=strength,
                    max_weight=args.max_weight,
                    cost_bps=args.cost_bps,
                )
                scenario = f"{factor_set}_{frequency}_s{strength:g}"
                daily.insert(0, "scenario", scenario)
                daily.insert(1, "factor_set", factor_set)
                daily.insert(2, "frequency", frequency)
                daily.insert(3, "strength", strength)
                daily_by_scenario[scenario] = daily
                weights_by_scenario[scenario] = weights.assign(scenario=scenario, factor_set=factor_set, frequency=frequency, strength=strength)

                full = summarize_returns(daily, "full")
                test = summarize_returns(daily, "test", start="2023-01-01")
                ytd = summarize_returns(daily, "2026_ytd", start="2026-01-01")
                for row in [full, test, ytd]:
                    row.update({"scenario": scenario, "factor_set": factor_set, "frequency": frequency, "strength": strength})
                    period_rows.append(row)
                scenario_rows.append(test | {"scenario": scenario, "factor_set": factor_set, "frequency": frequency, "strength": strength})

    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["information_ratio", "excess_total_return"], ascending=False)
    period_summary = pd.DataFrame(period_rows).sort_values(["scenario", "period"])
    scenario_summary.to_csv(output_dir / "scenario_summary_test_period.csv", index=False)
    period_summary.to_csv(output_dir / "period_summary.csv", index=False)

    eligible = period_summary.pivot(index="scenario", columns="period", values=["information_ratio", "excess_total_return"])
    eligible = eligible.dropna()
    positive_ytd = eligible[("excess_total_return", "2026_ytd")] > 0
    if positive_ytd.any():
        best_scenario = eligible.loc[positive_ytd, ("information_ratio", "test")].idxmax()
    else:
        best_scenario = scenario_summary.iloc[0]["scenario"]

    best_daily = daily_by_scenario[best_scenario]
    best_weights = weights_by_scenario[best_scenario]
    best_daily.to_csv(output_dir / "best_daily_returns.csv", index=False)
    best_weights.to_csv(output_dir / "best_rebalance_weights.csv", index=False)

    holdings, industry = latest_weight_tables(best_weights, panel, Path("data/raw/stock_company.parquet"))
    holdings.to_csv(output_dir / "best_latest_holdings.csv", index=False)
    industry.to_csv(output_dir / "best_latest_industry_active.csv", index=False)

    monthly = best_daily.assign(month=best_daily["trade_date"].dt.to_period("M").astype(str)).groupby("month").apply(
        lambda x: pd.Series(
            {
                "portfolio_return": (1.0 + x["portfolio_return_net"]).prod() - 1.0,
                "benchmark_return": (1.0 + x["benchmark_return"]).prod() - 1.0,
                "excess_return": (1.0 + x["portfolio_return_net"]).prod() / (1.0 + x["benchmark_return"]).prod() - 1.0,
                "turnover": x["turnover"].sum(),
            }
        ),
        include_groups=False,
    )
    monthly.to_csv(output_dir / "best_monthly_returns.csv")

    print(f"panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()}")
    print(f"scenarios={len(scenario_summary)}, best={best_scenario}")
    print(scenario_summary.head(10)[[
        "scenario",
        "excess_total_return",
        "annual_excess_arith",
        "tracking_error",
        "information_ratio",
        "active_max_drawdown",
        "avg_rebalance_turnover",
        "avg_active_share",
        "avg_max_industry_active",
    ]].to_string(index=False))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
