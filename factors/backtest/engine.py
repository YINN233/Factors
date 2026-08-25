"""
轻量回测引擎：日频截面调仓，费后净值计算。
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    参数：
        weights_df: DataFrame[trade_date, ts_code, weight] 每天的持仓权重
        prices_df: DataFrame[trade_date, ts_code, close_adj] 复权收盘价
        benchmark_weight_df: DataFrame[trade_date, ts_code, index_weight] 基准权重
        fee_rate: 双边手续费率，默认 0.0015（0.15%）
        impact_cost: 冲击成本系数，默认 turnover * 0.001（0.1%）
    """

    def __init__(
        self,
        weights_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        benchmark_weight_df: pd.DataFrame,
        fee_rate: float = 0.0015,
        impact_cost: float = 0.001,
    ):
        self.weights = weights_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.prices = prices_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.benchmark = benchmark_weight_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.fee_rate = fee_rate
        self.impact_cost = impact_cost

        # 计算个股日收益
        self.prices["return_1d"] = self.prices.groupby("ts_code")["close_adj"].pct_change()

    def run(self) -> pd.DataFrame:
        """
        运行回测，返回日频统计 DataFrame。
        """
        records = []
        dates = sorted(self.weights["trade_date"].unique())
        prev_w = None

        for date in dates:
            w_today = self.weights[self.weights["trade_date"] == date]
            price_today = self.prices[self.prices["trade_date"] == date]
            bm_today = self.benchmark[self.benchmark["trade_date"] == date]

            # 合并权重和收益
            merged = w_today.merge(
                price_today[["ts_code", "return_1d"]],
                on="ts_code",
                how="left",
            )
            merged = merged.merge(
                bm_today[["ts_code", "index_weight"]],
                on="ts_code",
                how="left",
            )
            merged["index_weight"] = merged["index_weight"].fillna(0.0)
            merged["return_1d"] = merged["return_1d"].fillna(0.0)

            w = merged["weight"].values.astype(np.float32)
            r = merged["return_1d"].values.astype(np.float32)
            bm = merged["index_weight"].values.astype(np.float32)
            bm = bm / (bm.sum() + 1e-9)  # 归一化

            # 组合收益
            portfolio_ret = float(np.dot(w, r))
            benchmark_ret = float(np.dot(bm, r))

            # 换手率 & 费后收益
            if prev_w is not None:
                # 对齐股票池（取交集，缺失的权重视为 0）
                codes_today = set(merged["ts_code"].values)
                # 简化：假设股票池变化不大，直接用长度对齐
                min_len = min(len(w), len(prev_w))
                turnover = 0.5 * np.sum(np.abs(w[:min_len] - prev_w[:min_len]))
                # 若股票池扩大/收缩，剩余部分也算变化
                if len(w) != len(prev_w):
                    turnover += 0.5 * abs(len(w) - len(prev_w)) * 0.01  # 粗略估计
            else:
                turnover = 0.5 * np.sum(np.abs(w))  # 建仓成本

            fee = turnover * (self.fee_rate + self.impact_cost)
            portfolio_ret_net = portfolio_ret - fee
            benchmark_ret_net = benchmark_ret  # 基准不调仓，无手续费

            records.append({
                "trade_date": date,
                "portfolio_return": portfolio_ret,
                "portfolio_return_net": portfolio_ret_net,
                "benchmark_return": benchmark_ret,
                "benchmark_return_net": benchmark_ret_net,
                "excess_return": portfolio_ret_net - benchmark_ret_net,
                "turnover": turnover,
                "fee": fee,
            })

            prev_w = w.copy()

        self.daily_df = pd.DataFrame(records)
        self.daily_df["portfolio_cum"] = (1 + self.daily_df["portfolio_return_net"]).cumprod() - 1
        self.daily_df["benchmark_cum"] = (1 + self.daily_df["benchmark_return_net"]).cumprod() - 1
        self.daily_df["excess_cum"] = self.daily_df["portfolio_cum"] - self.daily_df["benchmark_cum"]

        return self.daily_df

    def summary(self) -> Dict[str, float]:
        """回测统计摘要。"""
        df = self.daily_df.copy()
        n = len(df)
        if n == 0:
            return {}

        excess = df["excess_return"].values
        portfolio = df["portfolio_return_net"].values
        benchmark = df["benchmark_return_net"].values

        # 年化（按252交易日）
        ann_excess = np.mean(excess) * 252
        ann_portfolio = np.mean(portfolio) * 252
        ann_benchmark = np.mean(benchmark) * 252

        # 波动率
        vol_excess = np.std(excess, ddof=1) * np.sqrt(252)
        vol_portfolio = np.std(portfolio, ddof=1) * np.sqrt(252)

        # IR
        ir = ann_excess / (vol_excess + 1e-9)

        # 最大回撤（超额曲线）
        cum = df["excess_cum"].values
        rolling_max = np.maximum.accumulate(cum)
        drawdown = cum - rolling_max
        max_dd = drawdown.min()

        # 胜率
        win_rate = np.mean(excess > 0)

        # 月均换手
        avg_turnover = df["turnover"].mean()

        return {
            "annualized_return": ann_portfolio,
            "annualized_benchmark": ann_benchmark,
            "annualized_excess": ann_excess,
            "tracking_error": vol_excess,
            "information_ratio": ir,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "sharpe_ratio": ann_portfolio / (vol_portfolio + 1e-9),
            "n_days": n,
        }

    def plot(self, save_path: Optional[str] = None):
        """绘制累计净值曲线。"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        ax.plot(self.daily_df["trade_date"], self.daily_df["portfolio_cum"], label="Portfolio", color="blue")
        ax.plot(self.daily_df["trade_date"], self.daily_df["benchmark_cum"], label="Benchmark", color="gray")
        ax.set_title("Cumulative Net Value")
        ax.legend()
        ax.set_ylabel("Cumulative Return")

        ax = axes[1]
        ax.plot(self.daily_df["trade_date"], self.daily_df["excess_cum"], label="Excess", color="red")
        ax.set_title("Excess Cumulative Return")
        ax.legend()
        ax.set_ylabel("Excess Return")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 回测图表保存至 {save_path}")
        plt.close()


if __name__ == "__main__":
    # 简单测试
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(0)
    n = 300

    weights = pd.DataFrame({
        "trade_date": np.repeat(dates, n),
        "ts_code": [f"{i:06d}.SZ" for i in range(n)] * 100,
        "weight": np.random.rand(n * 100),
    })
    # 每天归一化权重
    weights = weights.groupby("trade_date").apply(lambda x: x.assign(weight=x["weight"] / x["weight"].sum()))
    weights = weights.reset_index(drop=True)

    prices = pd.DataFrame({
        "trade_date": np.repeat(dates, n),
        "ts_code": [f"{i:06d}.SZ" for i in range(n)] * 100,
        "close_adj": np.cumprod(1 + np.random.randn(n * 100) * 0.01),
    })

    bm = pd.DataFrame({
        "trade_date": np.repeat(dates, n),
        "ts_code": [f"{i:06d}.SZ" for i in range(n)] * 100,
        "index_weight": np.random.rand(n * 100),
    })
    bm = bm.groupby("trade_date").apply(lambda x: x.assign(index_weight=x["index_weight"] / x["index_weight"].sum()))
    bm = bm.reset_index(drop=True)

    engine = BacktestEngine(weights, prices, bm)
    engine.run()
    stats = engine.summary()
    print(stats)
    engine.plot("tmp_backtest.png")
