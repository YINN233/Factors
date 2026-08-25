"""
因子评估报告生成器：基于 FactorEvaluator 的评估结果，输出 Markdown 报告。
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class FactorReport:
    """
    封装 FactorEvaluator 的输出，生成结构化的 Markdown 因子研究报告。
    """

    def __init__(self, evaluator_result: Dict, factor_name: str = "GRU_Alpha"):
        self.result = evaluator_result
        self.factor_name = factor_name

    def to_markdown(self) -> str:
        lines = [
            f"# {self.factor_name} 因子评估报告",
            "",
            "## 1. IC / RankIC 分析",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ]
        ic = self.result.get("ic", {})
        for k, v in ic.items():
            lines.append(f"| {k} | {v:.4f} |")

        lines.extend([
            "",
            "## 2. 多空对冲表现",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ])
        ls = self.result.get("long_short", {})
        for k, v in ls.items():
            lines.append(f"| {k} | {v:.4f} |")

        lines.extend([
            "",
            "## 3. Turnover",
            "",
            f"- 日均 Turnover: {self.result.get('turnover', 'N/A')}",
            "",
            "## 4. 图表",
            "",
            "- 因子评估图表见 `factor_evaluation.png`",
            "",
            "---",
            "*自动生成，仅供参考*",
        ])
        return "\n".join(lines)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        print(f"📄 因子报告已保存至 {path}")


if __name__ == "__main__":
    # 示例
    dummy = {
        "ic": {"IC_mean": 0.05, "IC_IR": 0.30, "RankIC_mean": 0.04, "RankIC_IR": 0.25},
        "long_short": {"annualized_return": 0.12, "sharpe_ratio": 1.5, "max_drawdown": -0.05},
        "turnover": 0.45,
    }
    report = FactorReport(dummy, factor_name="GRU_Alpha")
    print(report.to_markdown())
