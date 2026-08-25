"""Descriptor metadata for the local CNE6-style risk model.

The exact MSCI Barra CNE6 methodology is proprietary. This module keeps the
local descriptor list explicit so reports can distinguish directly available
data from proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    style: str
    expression: str
    description: str
    required_columns: Sequence[str]
    availability: str = "direct"
    source: str = "tushare"

    def is_available(self, columns: Sequence[str]) -> bool:
        present = set(columns)
        return all(col in present for col in self.required_columns)


def descriptor_specs() -> list[DescriptorSpec]:
    return [
        DescriptorSpec(
            "log_total_mv",
            "size",
            "log(total_mv)",
            "Total market capitalization exposure. Larger companies often carry different liquidity and benchmark-driven risk.",
            ["total_mv"],
        ),
        DescriptorSpec(
            "mid_cap_proxy",
            "size",
            "residual(log(total_mv)^2 | 1 + log(total_mv))",
            "Non-linear size exposure from the squared log market cap after removing its linear size component.",
            ["total_mv"],
            availability="proxy",
        ),
        DescriptorSpec(
            "beta_252",
            "volatility",
            "rolling_cov(stock_return, benchmark_return, 252) / rolling_var(benchmark_return, 252)",
            "Market beta over roughly one trading year.",
            ["returns_1d", "csi500_return"],
        ),
        DescriptorSpec(
            "daily_std_252",
            "volatility",
            "rolling_std(stock_return, 252)",
            "One-year realized volatility.",
            ["returns_1d"],
        ),
        DescriptorSpec(
            "cumulative_range_252",
            "volatility",
            "log(rolling_max(high_adj, 252) / rolling_min(low_adj, 252))",
            "One-year realized price range.",
            ["high_adj", "low_adj"],
        ),
        DescriptorSpec(
            "residual_volatility_proxy",
            "volatility",
            "rolling_std(stock_return - beta_252 * benchmark_return, 252)",
            "Residual volatility proxy after removing market beta.",
            ["returns_1d", "csi500_return"],
            availability="proxy",
        ),
        DescriptorSpec(
            "avg_turnover_21",
            "liquidity",
            "rolling_mean(turnover_rate, 21)",
            "Short-window turnover exposure.",
            ["turnover_rate"],
        ),
        DescriptorSpec(
            "avg_turnover_63",
            "liquidity",
            "rolling_mean(turnover_rate, 63)",
            "Quarterly turnover exposure.",
            ["turnover_rate"],
        ),
        DescriptorSpec(
            "avg_amount_21",
            "liquidity",
            "log1p(rolling_mean(amount, 21))",
            "Short-window trading amount exposure.",
            ["amount"],
        ),
        DescriptorSpec(
            "turnover_stability_63",
            "liquidity",
            "-rolling_std(turnover_rate, 63)",
            "Stable turnover proxy. Higher values mean less unstable trading activity.",
            ["turnover_rate"],
            availability="proxy",
        ),
        DescriptorSpec(
            "ret_252_ex_21",
            "momentum",
            "close_adj.shift(21) / close_adj.shift(252) - 1",
            "Twelve-month momentum excluding the most recent month.",
            ["close_adj"],
        ),
        DescriptorSpec(
            "ret_126",
            "momentum",
            "close_adj / close_adj.shift(126) - 1",
            "Six-month price momentum.",
            ["close_adj"],
        ),
        DescriptorSpec(
            "short_reversal_21",
            "momentum",
            "-(close_adj / close_adj.shift(21) - 1)",
            "Short-term reversal proxy.",
            ["close_adj"],
            availability="proxy",
        ),
        DescriptorSpec(
            "book_to_price",
            "value",
            "1 / pb",
            "Book-to-price valuation exposure.",
            ["pb"],
        ),
        DescriptorSpec(
            "earnings_yield",
            "value",
            "1 / pe_ttm",
            "Earnings yield exposure.",
            ["pe_ttm"],
        ),
        DescriptorSpec(
            "sales_to_price",
            "value",
            "1 / ps_ttm",
            "Sales-to-price exposure.",
            ["ps_ttm"],
        ),
        DescriptorSpec(
            "cashflow_to_price",
            "value",
            "n_cashflow_act_ttm / total_mv",
            "Operating cash flow yield proxy.",
            ["n_cashflow_act_ttm", "total_mv"],
            availability="proxy",
        ),
        DescriptorSpec(
            "revenue_yoy",
            "growth",
            "revenue_yoy",
            "Revenue growth exposure.",
            ["revenue_yoy"],
        ),
        DescriptorSpec(
            "net_profit_yoy",
            "growth",
            "net_profit_yoy",
            "Net profit growth exposure.",
            ["net_profit_yoy"],
        ),
        DescriptorSpec(
            "roe_growth",
            "growth",
            "roe_ttm - roe_ttm.shift(252)",
            "ROE improvement proxy.",
            ["roe_ttm"],
            availability="proxy",
        ),
        DescriptorSpec(
            "asset_turnover_yoy",
            "growth",
            "asset_turnover_yoy",
            "Asset turnover improvement exposure.",
            ["asset_turnover_yoy"],
        ),
        DescriptorSpec(
            "roe_ttm",
            "quality",
            "roe_ttm",
            "Return on equity.",
            ["roe_ttm"],
        ),
        DescriptorSpec(
            "roa_ttm",
            "quality",
            "roa_ttm",
            "Return on assets.",
            ["roa_ttm"],
        ),
        DescriptorSpec(
            "gross_margin_ttm",
            "quality",
            "gross_margin_ttm",
            "Gross margin level.",
            ["gross_margin_ttm"],
        ),
        DescriptorSpec(
            "cashflow_to_profit",
            "quality",
            "cashflow_to_profit",
            "Cash conversion of accounting profit.",
            ["cashflow_to_profit"],
        ),
        DescriptorSpec(
            "low_leverage",
            "quality",
            "-debt_to_assets",
            "Lower leverage quality proxy.",
            ["debt_to_assets"],
        ),
        DescriptorSpec(
            "accrual_quality",
            "quality",
            "(n_cashflow_act_ttm - net_profit_ttm) / total_assets",
            "Cash earnings relative to accrual earnings.",
            ["n_cashflow_act_ttm", "net_profit_ttm", "total_assets"],
            availability="proxy",
        ),
        DescriptorSpec(
            "dv_ttm",
            "dividend_yield",
            "dv_ttm",
            "TTM dividend yield.",
            ["dv_ttm"],
        ),
        DescriptorSpec(
            "dv_ratio",
            "dividend_yield",
            "dv_ratio",
            "Dividend ratio exposure.",
            ["dv_ratio"],
        ),
        DescriptorSpec(
            "analyst_report_count_90",
            "sentiment",
            "rolling_count(report_rc, 90d)",
            "Sell-side analyst report coverage over the past 90 calendar days.",
            ["analyst_report_count_90"],
            source="tushare.report_rc",
        ),
        DescriptorSpec(
            "analyst_org_count_180",
            "sentiment",
            "rolling_nunique(org_name, 180d)",
            "Number of sell-side institutions covering the stock over the past 180 calendar days.",
            ["analyst_org_count_180"],
            source="tushare.report_rc",
        ),
        DescriptorSpec(
            "analyst_rating_score_180",
            "sentiment",
            "rolling_mean(mapped_rating_score, 180d)",
            "Average mapped analyst rating score over the past 180 calendar days.",
            ["analyst_rating_score_180"],
            source="tushare.report_rc",
        ),
        DescriptorSpec(
            "analyst_target_upside_180",
            "sentiment",
            "mean(target_price, 180d) / close - 1",
            "Mean analyst target-price upside over the past 180 calendar days.",
            ["analyst_target_upside_180"],
            source="tushare.report_rc",
        ),
        DescriptorSpec(
            "analyst_eps_revision_180",
            "sentiment",
            "mean(eps_forecast, 0-180d) / mean(eps_forecast, 180-360d) - 1",
            "Recent sell-side EPS forecast revision relative to the prior 180-day window.",
            ["analyst_eps_revision_180"],
            source="tushare.report_rc",
        ),
    ]


def descriptor_metadata(columns: Sequence[str] | None = None) -> pd.DataFrame:
    rows = []
    present = set([] if columns is None else columns)
    for spec in descriptor_specs():
        available = True if columns is None else spec.is_available(columns)
        rows.append(
            {
                "descriptor": spec.name,
                "style": spec.style,
                "expression": spec.expression,
                "description": spec.description,
                "required_columns": ",".join(spec.required_columns),
                "availability": spec.availability,
                "source": spec.source,
                "is_available": available,
                "missing_columns": "" if columns is None else ",".join([col for col in spec.required_columns if col not in present]),
            }
        )
    return pd.DataFrame(rows)
