"""
Alpha candidate computation, evaluation, and redundancy-aware selection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from factors.alpha.candidates import AlphaCandidate, available_candidates
from factors.alpha.evaluator import FactorEvaluator


@dataclass
class AlphaMiningConfig:
    label_col: str = "label"
    date_col: str = "trade_date"
    code_col: str = "ts_code"
    n_groups: int = 5
    min_abs_ic: float = 0.005
    min_coverage: float = 0.55
    max_pair_corr: float = 0.85
    include_turnover: bool = True


@dataclass
class AlphaMiningResult:
    factor_values: pd.DataFrame
    summary: pd.DataFrame
    selected: List[str]


class AlphaMiner:
    def __init__(
        self,
        candidates: Iterable[AlphaCandidate],
        config: Optional[AlphaMiningConfig] = None,
    ):
        self.candidates = list(candidates)
        self.config = config or AlphaMiningConfig()

    def compute_factor_values(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        base_cols = [cfg.date_col, cfg.code_col, cfg.label_col]
        panel = df.sort_values([cfg.code_col, cfg.date_col]).reset_index(drop=True).copy()

        factor_columns = {}
        for candidate in self.candidates:
            values = candidate.calculate(panel)
            factor_columns[candidate.name] = values.to_numpy()
        factor_df = pd.DataFrame(factor_columns, index=panel.index)
        result = pd.concat([panel[base_cols].copy(), factor_df], axis=1)
        return result.sort_values([cfg.date_col, cfg.code_col]).reset_index(drop=True)

    def evaluate(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        records = []
        for candidate in self.candidates:
            if candidate.name not in factor_values.columns:
                continue

            eval_df = factor_values[
                [cfg.date_col, cfg.code_col, candidate.name, cfg.label_col]
            ].rename(columns={candidate.name: "factor"})
            coverage = eval_df["factor"].notna().mean()
            eval_df = eval_df.dropna(subset=["factor", cfg.label_col])
            if eval_df.empty:
                continue

            evaluator = FactorEvaluator(
                eval_df,
                factor_col="factor",
                label_col=cfg.label_col,
                date_col=cfg.date_col,
                code_col=cfg.code_col,
            )
            ic_df = evaluator.ic_analysis()
            if ic_df.empty:
                continue

            ic_sum = evaluator.ic_summary(ic_df)
            turnover_df = evaluator.turnover_analysis() if cfg.include_turnover else pd.DataFrame()
            records.append(
                {
                    "factor": candidate.name,
                    "family": candidate.family,
                    "expression": candidate.expression,
                    "description": candidate.description,
                    "window": candidate.window,
                    "complexity": candidate.complexity,
                    "coverage": coverage,
                    "n_ic_dates": len(ic_df),
                    "IC_mean": ic_sum["IC_mean"],
                    "IC_IR": ic_sum["IC_IR"],
                    "RankIC_mean": ic_sum["RankIC_mean"],
                    "RankIC_IR": ic_sum["RankIC_IR"],
                    "turnover": (
                        turnover_df["turnover"].mean() if not turnover_df.empty else pd.NA
                    ),
                }
            )

        if not records:
            return pd.DataFrame()
        summary = pd.DataFrame(records)
        summary["score"] = summary["RankIC_mean"].abs() * summary["coverage"] / summary["complexity"].clip(lower=1)
        return summary.sort_values("score", ascending=False).reset_index(drop=True)

    def select(self, factor_values: pd.DataFrame, summary: pd.DataFrame) -> List[str]:
        cfg = self.config
        if summary.empty:
            return []

        eligible = summary[
            (summary["coverage"] >= cfg.min_coverage)
            & (summary["RankIC_mean"].abs() >= cfg.min_abs_ic)
        ].copy()
        if eligible.empty:
            return []

        ordered_factors = [factor for factor in eligible["factor"] if factor in factor_values.columns]
        corr_matrix = factor_values[ordered_factors].corr(min_periods=20).abs() if ordered_factors else pd.DataFrame()
        selected: List[str] = []
        for factor in ordered_factors:
            if not selected:
                selected.append(factor)
                continue
            corr = corr_matrix.loc[factor, selected] if factor in corr_matrix.index else pd.Series(dtype=float)
            max_corr = corr.max() if not corr.empty else 0.0
            if pd.isna(max_corr) or max_corr < cfg.max_pair_corr:
                selected.append(factor)
        return selected

    def run(self, df: pd.DataFrame) -> AlphaMiningResult:
        factor_values = self.compute_factor_values(df)
        summary = self.evaluate(factor_values)
        selected = self.select(factor_values, summary)
        return AlphaMiningResult(factor_values=factor_values, summary=summary, selected=selected)

    def save(self, result: AlphaMiningResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result.factor_values.to_parquet(output_path / "factor_values.parquet", index=False)
        result.summary.to_csv(output_path / "candidate_summary.csv", index=False)
        pd.Series(result.selected, name="factor").to_csv(output_path / "selected_factors.csv", index=False)


def mine_default_daily_factors(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    windows: Iterable[int] = (5, 20),
    factor_set: str = "all",
    config: Optional[AlphaMiningConfig] = None,
) -> AlphaMiningResult:
    candidates = available_candidates(df, windows=windows, factor_set=factor_set)
    miner = AlphaMiner(candidates, config=config)
    result = miner.run(df)
    if output_dir is not None:
        miner.save(result, output_dir)
    return result
