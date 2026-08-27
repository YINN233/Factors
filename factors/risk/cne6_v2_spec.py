"""Immutable descriptor and style specification for CNE6 enhanced V2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


STYLE_FACTORS_V2 = (
    "size",
    "nonlinear_size",
    "beta",
    "residual_volatility",
    "liquidity",
    "momentum",
    "short_reversal",
    "book_to_price",
    "earnings_yield",
    "growth",
    "profitability",
    "investment_quality",
    "leverage",
    "dividend_yield",
    "sentiment",
)


@dataclass(frozen=True)
class DescriptorSpecV2:
    name: str
    style: str
    weight: float
    formula: str
    required_columns: tuple[str, ...]
    direction: float = 1.0
    window: int | None = None
    half_life: int | None = None
    availability: str = "direct"


def descriptor_specs_v2() -> tuple[DescriptorSpecV2, ...]:
    specs = (
        DescriptorSpecV2("log_total_mv", "size", 1.00, "log(total_mv_rmb)", ("total_mv",)),
        DescriptorSpecV2("nonlinear_size_residual", "nonlinear_size", 1.00, "residual(z(size)^3 | 1 + z(size))", ("total_mv",), availability="derived"),
        DescriptorSpecV2("beta_252_ewma", "beta", 0.50, "ewma_cov(r,m,252,hl=63)/ewma_var(m)", ("returns_1d", "csi500_return"), window=252, half_life=63),
        DescriptorSpecV2("beta_504_ewma", "beta", 0.30, "ewma_cov(r,m,504,hl=126)/ewma_var(m)", ("returns_1d", "csi500_return"), window=504, half_life=126),
        DescriptorSpecV2("downside_beta_252", "beta", 0.20, "beta_252 on market_return < 0", ("returns_1d", "csi500_return"), window=252, half_life=63),
        DescriptorSpecV2("dastd_252", "residual_volatility", 0.50, "ewma_std(r,252,hl=42)", ("returns_1d",), window=252, half_life=42),
        DescriptorSpecV2("hsigma_252", "residual_volatility", 0.30, "ewma_std(r-beta*m,252,hl=63)", ("returns_1d", "csi500_return"), window=252, half_life=63),
        DescriptorSpecV2("cmra_12m", "residual_volatility", 0.20, "range(cumulative monthly log excess return,12m)", ("returns_1d", "csi500_return"), window=252),
        DescriptorSpecV2("stom_21", "liquidity", 0.30, "log(sum(turnover_rate/100,21))", ("turnover_rate",), window=21),
        DescriptorSpecV2("stoq_63", "liquidity", 0.30, "log(sum(turnover_rate/100,63)/3)", ("turnover_rate",), window=63),
        DescriptorSpecV2("stoa_252", "liquidity", 0.25, "log(sum(turnover_rate/100,252)/12)", ("turnover_rate",), window=252),
        DescriptorSpecV2("amihud_63", "liquidity", 0.10, "log(mean(abs(return)/amount_rmb,63))", ("returns_1d", "amount"), direction=-1.0, window=63),
        DescriptorSpecV2("turnover_stability_63", "liquidity", 0.05, "std(turnover_rate,63)", ("turnover_rate",), direction=-1.0, window=63),
        DescriptorSpecV2("rstr_12m_ex_1m", "momentum", 0.50, "sum(log_excess_return,t-252:t-21)", ("returns_1d", "csi500_return"), window=252, half_life=126),
        DescriptorSpecV2("rstr_6m_ex_1m", "momentum", 0.30, "sum(log_excess_return,t-126:t-21)", ("returns_1d", "csi500_return"), window=126, half_life=63),
        DescriptorSpecV2("momentum_ewma_252", "momentum", 0.20, "ewma_sum(log_excess_return,t-252:t-21,hl=126)", ("returns_1d", "csi500_return"), window=252, half_life=126),
        DescriptorSpecV2("reversal_5d", "short_reversal", 0.40, "-return_5d", ("close_adj",), window=5),
        DescriptorSpecV2("reversal_21d", "short_reversal", 0.60, "-return_21d", ("close_adj",), window=21),
        DescriptorSpecV2("book_to_price", "book_to_price", 1.00, "1/pb", ("pb",)),
        DescriptorSpecV2("earnings_yield", "earnings_yield", 0.40, "1/pe_ttm", ("pe_ttm",)),
        DescriptorSpecV2("cashflow_to_price", "earnings_yield", 0.35, "ocf_ttm/total_mv_rmb", ("n_cashflow_act_ttm", "total_mv")),
        DescriptorSpecV2("forecast_earnings_yield", "earnings_yield", 0.25, "analyst_forward_eps_180/close", ("analyst_forward_eps_180", "close"), availability="proxy"),
        DescriptorSpecV2("revenue_yoy", "growth", 0.25, "revenue_latest/revenue_same_quarter_last_year-1", ("revenue_yoy",)),
        DescriptorSpecV2("net_profit_yoy", "growth", 0.25, "profit_latest/profit_same_quarter_last_year-1", ("net_profit_yoy",)),
        DescriptorSpecV2("eps_growth", "growth", 0.20, "eps_latest/eps_same_quarter_last_year-1", ("eps_growth",)),
        DescriptorSpecV2("roe_growth", "growth", 0.15, "roe_latest-roe_same_quarter_last_year", ("roe_growth",)),
        DescriptorSpecV2("asset_turnover_yoy", "growth", 0.15, "asset_turnover_latest/lag4-1", ("asset_turnover_yoy",)),
        DescriptorSpecV2("roe_ttm", "profitability", 0.20, "net_profit_ttm/equity", ("roe_ttm",)),
        DescriptorSpecV2("roa_ttm", "profitability", 0.15, "net_profit_ttm/total_assets", ("roa_ttm",)),
        DescriptorSpecV2("gross_margin_ttm", "profitability", 0.15, "gross_profit_ttm/revenue_ttm", ("gross_margin_ttm",)),
        DescriptorSpecV2("operating_margin_ttm", "profitability", 0.10, "operate_profit_ttm/revenue_ttm", ("operating_margin_ttm",)),
        DescriptorSpecV2("cashflow_to_profit", "profitability", 0.15, "ocf_ttm/net_profit_ttm", ("cashflow_to_profit",)),
        DescriptorSpecV2("accrual_quality", "profitability", 0.15, "(ocf_ttm-net_profit_ttm)/total_assets", ("n_cashflow_act_ttm", "net_profit_ttm", "total_assets")),
        DescriptorSpecV2("earnings_stability", "profitability", 0.10, "-std(roa_quarterly,8q)", ("earnings_stability",)),
        DescriptorSpecV2("asset_growth", "investment_quality", 0.35, "total_assets/lag4-1", ("asset_growth",), direction=-1.0),
        DescriptorSpecV2("capex_growth", "investment_quality", 0.25, "capex_ttm/lag4-1", ("capex_growth",), direction=-1.0),
        DescriptorSpecV2("inventory_growth", "investment_quality", 0.20, "inventories/lag4-1", ("inventory_growth",), direction=-1.0),
        DescriptorSpecV2("working_capital_growth", "investment_quality", 0.20, "working_capital/lag4-1", ("working_capital_growth",), direction=-1.0),
        DescriptorSpecV2("debt_to_assets", "leverage", 0.35, "total_liab/total_assets", ("debt_to_assets",)),
        DescriptorSpecV2("book_leverage", "leverage", 0.30, "total_assets/equity", ("book_leverage",)),
        DescriptorSpecV2("market_leverage", "leverage", 0.25, "total_liab/(total_liab+total_mv_rmb)", ("total_liab", "total_mv")),
        DescriptorSpecV2("inverse_interest_coverage", "leverage", 0.10, "fin_exp_ttm/operate_profit_ttm", ("inverse_interest_coverage",)),
        DescriptorSpecV2("dv_ttm", "dividend_yield", 0.70, "dv_ttm", ("dv_ttm",)),
        DescriptorSpecV2("dv_ratio", "dividend_yield", 0.30, "dv_ratio", ("dv_ratio",)),
        DescriptorSpecV2("analyst_report_count_90", "sentiment", 0.15, "report_count_90d", ("analyst_report_count_90",)),
        DescriptorSpecV2("analyst_org_count_180", "sentiment", 0.15, "covered_org_count_180d", ("analyst_org_count_180",)),
        DescriptorSpecV2("analyst_rating_score_180", "sentiment", 0.20, "mean_rating_score_180d", ("analyst_rating_score_180",)),
        DescriptorSpecV2("analyst_target_upside_180", "sentiment", 0.25, "mean_target_price_180d/close-1", ("analyst_target_upside_180",)),
        DescriptorSpecV2("analyst_eps_revision_180", "sentiment", 0.25, "recent_eps_consensus/prior_eps_consensus-1", ("analyst_eps_revision_180",)),
    )
    return specs


def validate_v2_spec(specs: tuple[DescriptorSpecV2, ...] | None = None) -> None:
    specs = descriptor_specs_v2() if specs is None else specs
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate CNE6 V2 descriptor names")
    unknown_styles = sorted({spec.style for spec in specs}.difference(STYLE_FACTORS_V2))
    if unknown_styles:
        raise ValueError(f"unknown CNE6 V2 styles: {unknown_styles}")
    for style in {spec.style for spec in specs}:
        total = sum(spec.weight for spec in specs if spec.style == style)
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"descriptor weights for {style} sum to {total}, expected 1")
    if any(spec.direction not in (-1.0, 1.0) for spec in specs):
        raise ValueError("descriptor direction must be -1 or 1")


def descriptor_metadata_v2(columns: Sequence[str] | None = None) -> pd.DataFrame:
    present = None if columns is None else set(columns)
    rows = []
    for spec in descriptor_specs_v2():
        missing = [] if present is None else [column for column in spec.required_columns if column not in present]
        rows.append(
            {
                "descriptor": spec.name,
                "style": spec.style,
                "weight": spec.weight,
                "direction": spec.direction,
                "formula": spec.formula,
                "required_columns": ",".join(spec.required_columns),
                "window": spec.window,
                "half_life": spec.half_life,
                "availability": spec.availability,
                "is_available": not missing,
                "missing_columns": ",".join(missing),
            }
        )
    return pd.DataFrame(rows)


validate_v2_spec()
