"""
组合表现报告生成器：基于 BacktestEngine 的回测结果，输出 Markdown 报告。
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class PortfolioReport:
    """
    封装 BacktestEngine 的输出，生成结构化的 Markdown 组合研究报告。
    """

    def __init__(self, stats: Dict, strategy_name: str = "Index_Enhancement"):
        self.stats = stats
        self.strategy_name = strategy_name

    def to_markdown(self) -> str:
        lines = [
            f"# {self.strategy_name} 组合表现报告",
            "",
            "## 1. 收益统计",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 年化组合收益 | {self.stats.get('annualized_return', 0):.2%} |",
            f"| 年化基准收益 | {self.stats.get('annualized_benchmark', 0):.2%} |",
            f"| 年化超额收益 | {self.stats.get('annualized_excess', 0):.2%} |",
            "",
            "## 2. 风险指标",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 跟踪误差 (TE) | {self.stats.get('tracking_error', 0):.2%} |",
            f"| 信息比率 (IR) | {self.stats.get('information_ratio', 0):.2f} |",
            f"| 最大回撤 | {self.stats.get('max_drawdown', 0):.2%} |",
            f"| 胜率 | {self.stats.get('win_rate', 0):.2%} |",
            f"| 夏普比率 | {self.stats.get('sharpe_ratio', 0):.2f} |",
            "",
            "## 3. 交易成本",
            "",
            f"| 日均换手率 | {self.stats.get('avg_turnover', 0):.2%} |",
            "",
            "## 4. 图表",
            "",
            "- 累计净值曲线见回测输出目录中的 `backtest.png`",
            "",
            "---",
            "*自动生成，仅供参考*",
        ]
        return "\n".join(lines)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        print(f"📄 组合报告已保存至 {path}")


if __name__ == "__main__":
    dummy = {
        "annualized_return": 0.15,
        "annualized_benchmark": 0.08,
        "annualized_excess": 0.07,
        "tracking_error": 0.05,
        "information_ratio": 1.40,
        "max_drawdown": -0.08,
        "win_rate": 0.62,
        "sharpe_ratio": 1.20,
        "avg_turnover": 0.35,
    }
    report = PortfolioReport(dummy, strategy_name="Two_Stage_300")
    print(report.to_markdown())
