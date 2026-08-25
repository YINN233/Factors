"""
因子评估体系：对给定的因子值进行全面评估。
输入：DataFrame 包含 [trade_date, ts_code, factor, label, (可选) return_next]
输出：IC、RankIC、分组收益、多空对冲、turnover 等。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FactorEvaluator:
    """
    对单因子进行评估，核心方法：
      - evaluate(): 一键输出全部指标
      - ic_analysis(): IC / RankIC 序列统计
      - group_test(): 按因子分 N 组，计算各组累计收益
      - long_short(): Top vs Bottom 多空对冲
      - turnover_analysis(): 相邻期因子权重变化
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factor_col: str = "factor",
        label_col: str = "label",
        date_col: str = "trade_date",
        code_col: str = "ts_code",
    ):
        """
        df 必须包含至少：date_col, code_col, factor_col, label_col
        label_col 应为未来收益（用于计算 IC 和分组收益）。
        """
        self.df = df[[date_col, code_col, factor_col, label_col]].copy()
        self.factor_col = factor_col
        self.label_col = label_col
        self.date_col = date_col
        self.code_col = code_col
        self.df = self.df.sort_values([date_col, code_col]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. IC 分析
    # ------------------------------------------------------------------
    def ic_analysis(self) -> pd.DataFrame:
        """
        日频截面计算 IC 和 RankIC，返回时间序列。
        """
        def _ic(x: pd.Series, y: pd.Series) -> float:
            return x.corr(y, method="pearson")

        def _rank_ic(x: pd.Series, y: pd.Series) -> float:
            return x.corr(y, method="spearman")

        records = []
        for date, sub in self.df.groupby(self.date_col):
            sub = sub.dropna(subset=[self.factor_col, self.label_col])
            if len(sub) < 3:
                continue
            ic = _ic(sub[self.factor_col], sub[self.label_col])
            rank_ic = _rank_ic(sub[self.factor_col], sub[self.label_col])
            records.append({
                self.date_col: date,
                "IC": ic,
                "RankIC": rank_ic,
                "n_stocks": len(sub),
            })

        ic_df = pd.DataFrame(records)
        return ic_df

    def ic_summary(self, ic_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """IC 序列的统计摘要。"""
        if ic_df is None:
            ic_df = self.ic_analysis()
        ic = ic_df["IC"].dropna()
        rank_ic = ic_df["RankIC"].dropna()
        return {
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "IC_IR": ic.mean() / (ic.std() + 1e-9),
            "IC_positive_ratio": (ic > 0).mean(),
            "RankIC_mean": rank_ic.mean(),
            "RankIC_std": rank_ic.std(),
            "RankIC_IR": rank_ic.mean() / (rank_ic.std() + 1e-9),
            "RankIC_positive_ratio": (rank_ic > 0).mean(),
        }

    # ------------------------------------------------------------------
    # 2. 分组测试
    # ------------------------------------------------------------------
    def group_test(self, n_groups: int = 5) -> pd.DataFrame:
        """
        每天按因子值分 n_groups 组，计算每组的平均未来收益。
        返回每天各组收益的时间序列。
        """
        records = []
        for date, sub in self.df.groupby(self.date_col):
            sub = sub.dropna(subset=[self.factor_col, self.label_col])
            if len(sub) < n_groups:
                continue
            # 按因子值分组
            sub["group"] = pd.qcut(sub[self.factor_col], n_groups, labels=False, duplicates="drop")
            group_mean = sub.groupby("group")[self.label_col].mean().to_dict()
            row = {self.date_col: date}
            for g in range(n_groups):
                row[f"G{g+1}"] = group_mean.get(g, np.nan)
            records.append(row)

        group_df = pd.DataFrame(records)
        return group_df

    def group_cumsum(self, group_df: Optional[pd.DataFrame] = None, n_groups: int = 5) -> pd.DataFrame:
        """各组收益的累计和（近似累计净值）。"""
        if group_df is None:
            group_df = self.group_test(n_groups=n_groups)
        group_df = group_df.sort_values(self.date_col)
        for g in range(n_groups):
            col = f"G{g+1}"
            if col in group_df.columns:
                group_df[f"{col}_cum"] = group_df[col].fillna(0).cumsum()
        return group_df

    # ------------------------------------------------------------------
    # 3. 多空对冲
    # ------------------------------------------------------------------
    def long_short(self, n_groups: int = 5, group_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Top 组（因子最大） - Bottom 组（因子最小）的日频收益。
        """
        if group_df is None:
            group_df = self.group_test(n_groups=n_groups)
        group_df = group_df.sort_values(self.date_col).reset_index(drop=True)
        top_col = f"G{n_groups}"
        bot_col = "G1"
        group_df["long_short"] = group_df[top_col] - group_df[bot_col]
        group_df["long_short_cum"] = group_df["long_short"].fillna(0).cumsum()
        return group_df

    def long_short_stats(self, ls_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """多空对冲的统计指标。"""
        if ls_df is None:
            ls_df = self.long_short()
        ls = ls_df["long_short"].dropna()
        if len(ls) == 0:
            return {}
        cum = ls_df["long_short_cum"].dropna()
        # 最大回撤
        rolling_max = cum.cummax()
        drawdown = cum - rolling_max
        max_drawdown = drawdown.min()

        return {
            "annualized_return": ls.mean() * 252,
            "annualized_volatility": ls.std() * np.sqrt(252),
            "sharpe_ratio": (ls.mean() * 252) / (ls.std() * np.sqrt(252) + 1e-9),
            "max_drawdown": max_drawdown,
            "win_rate": (ls > 0).mean(),
        }

    # ------------------------------------------------------------------
    # 4. Turnover
    # ------------------------------------------------------------------
    def turnover_analysis(self) -> pd.DataFrame:
        """
        计算相邻两期的因子排名 turnover（或权重变化）。
        这里采用：每天按因子值计算 zscore 后的权重向量，计算 L1 变化。
        """
        records = []
        dates = sorted(self.df[self.date_col].unique())
        prev_weights = None
        for i, date in enumerate(dates):
            sub = self.df[self.df[self.date_col] == date]
            sub = sub.dropna(subset=[self.factor_col])
            if len(sub) < 2:
                continue
            # 简单等权或按因子值 zscore 归一化权重
            fac = sub[self.factor_col].values
            fac_z = (fac - fac.mean()) / (fac.std() + 1e-9)
            # 映射到非负权重并归一化
            w = np.maximum(fac_z, 0)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones_like(w) / len(w)

            codes = sub[self.code_col].values
            weight_map = dict(zip(codes, w))

            if prev_weights is not None:
                all_codes = set(weight_map) | set(prev_weights)
                turnover = 0.5 * sum(
                    abs(weight_map.get(code, 0.0) - prev_weights.get(code, 0.0))
                    for code in all_codes
                )
                records.append({self.date_col: date, "turnover": turnover})
            prev_weights = weight_map

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 5. 一键评估
    # ------------------------------------------------------------------
    def evaluate(self, n_groups: int = 5, save_dir: Optional[str] = None) -> Dict:
        """运行全部评估，打印并返回结果字典。"""
        print("=" * 50)
        print("因子评估报告")
        print("=" * 50)

        # IC
        ic_df = self.ic_analysis()
        ic_sum = self.ic_summary(ic_df)
        print("\n【IC 分析】")
        for k, v in ic_sum.items():
            print(f"  {k}: {v:.4f}")

        # 分组
        group_df = self.group_test(n_groups=n_groups)
        group_cum = self.group_cumsum(group_df, n_groups=n_groups)
        print(f"\n【分组测试】({n_groups} 组)")
        for g in range(n_groups):
            col = f"G{g+1}_cum"
            if col in group_cum.columns:
                final = group_cum[col].iloc[-1]
                print(f"  组 {g+1} 累计收益: {final:.4f}")

        # 多空
        ls_df = self.long_short(n_groups=n_groups, group_df=group_df)
        ls_stats = self.long_short_stats(ls_df)
        print("\n【多空对冲】")
        for k, v in ls_stats.items():
            print(f"  {k}: {v:.4f}")

        # Turnover
        to_df = self.turnover_analysis()
        if not to_df.empty:
            print(f"\n【Turnover】日均: {to_df['turnover'].mean():.4f}")

        # 可视化
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self._plot_all(ic_df, group_cum, ls_df, to_df, save_path, n_groups)

        return {
            "ic": ic_sum,
            "group": group_cum,
            "long_short": ls_stats,
            "turnover": to_df["turnover"].mean() if not to_df.empty else None,
        }

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def _plot_all(
        self,
        ic_df: pd.DataFrame,
        group_cum: pd.DataFrame,
        ls_df: pd.DataFrame,
        to_df: pd.DataFrame,
        save_dir: Path,
        n_groups: int,
    ):
        # IC 时间序列
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(ic_df[self.date_col], ic_df["IC"], label="IC", alpha=0.7)
        ax.plot(ic_df[self.date_col], ic_df["RankIC"], label="RankIC", alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("IC / RankIC Time Series")
        ax.legend()
        ax.set_xlabel("Date")

        # 分组累计收益
        ax = axes[0, 1]
        for g in range(n_groups):
            col = f"G{g+1}_cum"
            if col in group_cum.columns:
                ax.plot(group_cum[self.date_col], group_cum[col], label=f"G{g+1}")
        ax.set_title("Group Cumulative Returns")
        ax.legend()
        ax.set_xlabel("Date")

        # 多空对冲
        ax = axes[1, 0]
        ax.plot(ls_df[self.date_col], ls_df["long_short_cum"], label="Long-Short Cum", color="red")
        ax.set_title("Long-Short Cumulative Return")
        ax.legend()
        ax.set_xlabel("Date")

        # Turnover
        ax = axes[1, 1]
        if not to_df.empty:
            ax.plot(to_df[self.date_col], to_df["turnover"], label="Turnover", color="green")
            ax.set_title("Turnover")
            ax.legend()
            ax.set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(save_dir / "factor_evaluation.png", dpi=150)
        plt.close()
        print(f"📊 图表已保存至 {save_dir / 'factor_evaluation.png'}")


if __name__ == "__main__":
    # 简单测试
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    df = pd.DataFrame({
        "trade_date": np.repeat(dates, 300),
        "ts_code": [f"{i:06d}.SZ" for i in range(300)] * 100,
        "factor": np.random.randn(300 * 100),
        "label": np.random.randn(300 * 100) * 0.02,
    })
    # 让 factor 和 label 有点相关性
    df["label"] = df["factor"] * 0.01 + np.random.randn(len(df)) * 0.02

    ev = FactorEvaluator(df)
    result = ev.evaluate(n_groups=5, save_dir="tmp_eval")
    print("\n评估完成。")
