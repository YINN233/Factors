"""
Win-rate and active-return stability diagnostics for index enhancement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _period_key(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "year":
        return dates.dt.year.astype(str)
    if freq == "quarter":
        return dates.dt.to_period("Q").astype(str)
    if freq == "month":
        return dates.dt.to_period("M").astype(str)
    raise ValueError("freq must be one of: year, quarter, month")


def summarize_periods(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["period"] = _period_key(df["trade_date"], freq)
    records = []
    for period, sub in df.groupby("period", sort=False):
        active = sub["active_return"].dropna()
        monthly_like = (1.0 + sub["portfolio_return_net"]).prod() / (1.0 + sub["benchmark_return"]).prod() - 1.0
        active_nav = (1.0 + active).cumprod()
        drawdown = active_nav / active_nav.cummax() - 1.0
        records.append(
            {
                "period_type": freq,
                "period": period,
                "start_date": sub["trade_date"].min(),
                "end_date": sub["trade_date"].max(),
                "n_days": len(sub),
                "win_days": int((active > 0).sum()),
                "flat_days": int((active == 0).sum()),
                "loss_days": int((active < 0).sum()),
                "daily_win_rate": (active > 0).mean(),
                "active_mean": active.mean(),
                "active_median": active.median(),
                "active_std": active.std(ddof=1),
                "active_sum": active.sum(),
                "active_compound": monthly_like,
                "tracking_error": active.std(ddof=1) * np.sqrt(252.0),
                "information_ratio": active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252.0) + 1e-12),
                "active_max_drawdown": drawdown.min(),
                "p05": active.quantile(0.05),
                "p25": active.quantile(0.25),
                "p75": active.quantile(0.75),
                "p95": active.quantile(0.95),
                "turnover_sum": sub["turnover"].sum(),
                "rebalance_count": int(sub["rebalanced"].sum()),
            }
        )
    return pd.DataFrame(records)


def summarize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "full": df,
        "test_2023_2026": df[df["trade_date"] >= "2023-01-01"],
        "ytd_2026": df[df["trade_date"] >= "2026-01-01"],
    }
    records = []
    for name, sub in windows.items():
        active = sub["active_return"].dropna()
        if active.empty:
            continue
        records.append(
            {
                "window": name,
                "start_date": sub["trade_date"].min(),
                "end_date": sub["trade_date"].max(),
                "n_days": len(active),
                "daily_win_rate": (active > 0).mean(),
                "mean": active.mean(),
                "median": active.median(),
                "std": active.std(ddof=1),
                "skew": active.skew(),
                "kurt": active.kurt(),
                "min": active.min(),
                "p01": active.quantile(0.01),
                "p05": active.quantile(0.05),
                "p25": active.quantile(0.25),
                "p75": active.quantile(0.75),
                "p95": active.quantile(0.95),
                "p99": active.quantile(0.99),
                "max": active.max(),
                "positive_mean": active[active > 0].mean(),
                "negative_mean": active[active < 0].mean(),
                "gain_loss_ratio": active[active > 0].mean() / abs(active[active < 0].mean()),
            }
        )
    return pd.DataFrame(records)


def rolling_stability(df: pd.DataFrame, windows: tuple[int, ...] = (20, 60, 120)) -> pd.DataFrame:
    out = df[["trade_date", "active_return"]].copy()
    for window in windows:
        roll = out["active_return"].rolling(window, min_periods=max(5, window // 2))
        out[f"win_rate_{window}d"] = roll.apply(lambda s: (s > 0).mean(), raw=False)
        out[f"mean_{window}d"] = roll.mean()
        out[f"te_{window}d"] = roll.std(ddof=1) * np.sqrt(252.0)
        out[f"ir_{window}d"] = out[f"mean_{window}d"] * 252.0 / (out[f"te_{window}d"] + 1e-12)
        out[f"active_sum_{window}d"] = roll.sum()
    active_nav = (1.0 + out["active_return"]).cumprod()
    out["active_nav"] = active_nav
    out["active_drawdown"] = active_nav / active_nav.cummax() - 1.0
    return out


def plot_diagnostics(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    rolling: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active = daily["active_return"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(active, bins=80, color="#315a89", alpha=0.78)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(active.mean(), color="#c43b3b", linewidth=1.5, label=f"mean={active.mean():.5f}")
    ax.set_title("Daily Active Return Distribution")
    ax.set_xlabel("daily active return")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "daily_active_return_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    month = monthly.copy()
    colors = np.where(month["active_compound"] >= 0, "#2f7d4f", "#b94b4b")
    ax.bar(month["period"], month["active_compound"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Monthly Compounded Active Return")
    ax.set_xlabel("month")
    ax.set_ylabel("active return")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_active_return_bars.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    recent = monthly[monthly["period"].astype(str).str.startswith(("2024", "2025", "2026"))].copy()
    colors = np.where(recent["daily_win_rate"] >= 0.5, "#2f7d4f", "#b94b4b")
    ax.bar(recent["period"], recent["daily_win_rate"], color=colors)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_title("Monthly Daily Win Rate Since 2024")
    ax.set_xlabel("month")
    ax.set_ylabel("daily win rate")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_daily_win_rate_since_2024.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling["trade_date"], rolling["win_rate_20d"], label="20d", color="#315a89")
    ax.plot(rolling["trade_date"], rolling["win_rate_60d"], label="60d", color="#8d6b22")
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_title("Rolling Daily Active Win Rate")
    ax.set_ylabel("win rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_win_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling["trade_date"], rolling["ir_60d"], label="60d rolling IR", color="#315a89")
    ax.plot(rolling["trade_date"], rolling["ir_120d"], label="120d rolling IR", color="#8d6b22")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Rolling Active Information Ratio")
    ax.set_ylabel("IR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_information_ratio.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling["trade_date"], rolling["active_nav"] - 1.0, label="active cumulative return", color="#315a89")
    ax.fill_between(
        rolling["trade_date"],
        rolling["active_drawdown"],
        0,
        color="#b94b4b",
        alpha=0.25,
        label="active drawdown",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Active Cumulative Return and Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "active_nav_and_drawdown.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/fundamental_alpha_csi500_index_enhancement/best_daily_returns.csv")
    parser.add_argument("--output", default="outputs/fundamental_alpha_csi500_index_enhancement_stability")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.input, parse_dates=["trade_date"])
    daily = daily.sort_values("trade_date").reset_index(drop=True)

    year = summarize_periods(daily, "year")
    quarter = summarize_periods(daily, "quarter")
    month = summarize_periods(daily, "month")
    dist = summarize_distribution(daily)
    rolling = rolling_stability(daily)

    year.to_csv(output_dir / "winrate_by_year.csv", index=False)
    quarter.to_csv(output_dir / "winrate_by_quarter.csv", index=False)
    month.to_csv(output_dir / "winrate_by_month.csv", index=False)
    dist.to_csv(output_dir / "active_return_distribution_summary.csv", index=False)
    rolling.to_csv(output_dir / "rolling_stability.csv", index=False)
    daily.to_csv(output_dir / "daily_active_returns.csv", index=False)
    plot_diagnostics(daily, month, rolling, output_dir)

    print(f"rows={len(daily)}, start={daily['trade_date'].min().date()}, end={daily['trade_date'].max().date()}")
    print("distribution")
    print(dist.to_string(index=False))
    print("recent years")
    print(year.tail(5)[["period", "daily_win_rate", "active_compound", "tracking_error", "information_ratio", "active_max_drawdown"]].to_string(index=False))
    print(f"wrote stability diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
