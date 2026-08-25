"""
Systematic fundamental operator factory.

The existing hand-written fundamental candidates are intentionally concise. This
module builds a broader, interpretable candidate library from point-in-time
financial statement fields by composing oriented fundamental atoms with simple
operators: rank, inverse-rank, ratio, mean, harmonic mean, industry neutralize,
and size neutralize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from factors.alpha import operators as op
from factors.alpha.candidates import AlphaCandidate


ComputeFunc = Callable[[pd.DataFrame], pd.Series]
_CACHE: dict[int, dict[str, pd.Series]] = {}


@dataclass(frozen=True)
class FundamentalAtom:
    name: str
    expression: str
    description: str
    source: str
    family: str
    required_columns: Sequence[str]
    compute: ComputeFunc


def _cs_rank_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values.groupby(df["trade_date"], sort=False).rank(pct=True)


def clear_fundamental_operator_cache(df: pd.DataFrame | None = None) -> None:
    if df is None:
        _CACHE.clear()
    else:
        _CACHE.pop(id(df), None)


def _cached_series(df: pd.DataFrame, key: str, builder: Callable[[], pd.Series]) -> pd.Series:
    cache = _CACHE.setdefault(id(df), {})
    if key not in cache:
        cache[key] = builder()
    return cache[key]


def _cached_rank_col(df: pd.DataFrame, col: str) -> pd.Series:
    return _cached_series(df, f"rank:{col}", lambda: op.cs_rank(df, col))


def _rank_col(col: str) -> ComputeFunc:
    return lambda df, c=col: _cached_rank_col(df, c)


def _inv_rank_col(col: str) -> ComputeFunc:
    return lambda df, c=col: _cached_series(df, f"inv_rank:{c}", lambda: 1.0 - _cached_rank_col(df, c))


def _rank_ratio(num_col: str, den_col: str, high_good: bool = True) -> ComputeFunc:
    def _compute(df: pd.DataFrame, n: str = num_col, d: str = den_col, good: bool = high_good) -> pd.Series:
        key = f"rank_ratio:{n}:{d}:{good}"

        def _build() -> pd.Series:
            rank = _cs_rank_series(df, op.safe_div(df[n], df[d]))
            return rank if good else 1.0 - rank

        return _cached_series(df, key, _build)

    return _compute


def _rank_expr(required: Sequence[str], func: Callable[[pd.DataFrame], pd.Series], high_good: bool = True) -> ComputeFunc:
    def _compute(df: pd.DataFrame, f: Callable[[pd.DataFrame], pd.Series] = func, good: bool = high_good) -> pd.Series:
        key = f"rank_expr:{','.join(required)}:{id(f)}:{good}"

        def _build() -> pd.Series:
            rank = _cs_rank_series(df, f(df))
            return rank if good else 1.0 - rank

        return _cached_series(df, key, _build)

    return _compute


def _atom_candidate(atom: FundamentalAtom) -> AlphaCandidate:
    return AlphaCandidate(
        name=f"fund_atom_{atom.name}",
        expression=atom.expression,
        description=f"{atom.description} Source: {atom.source}.",
        required_columns=list(atom.required_columns),
        window=1,
        complexity=2,
        family=atom.family,
        compute=atom.compute,
    )


def fundamental_atoms() -> list[FundamentalAtom]:
    """Return oriented base metrics; larger values always mean better quality/value."""
    atoms: list[FundamentalAtom] = [
        FundamentalAtom("roe", "rank(roe_ttm)", "Higher return on equity.", "income + balance", "fundamental_quality", ["roe_ttm"], _rank_col("roe_ttm")),
        FundamentalAtom("roa", "rank(roa_ttm)", "Higher return on assets.", "income + balance", "fundamental_quality", ["roa_ttm"], _rank_col("roa_ttm")),
        FundamentalAtom("gross_margin", "rank(gross_margin_ttm)", "Higher gross margin, usually reflecting pricing power.", "income", "fundamental_quality", ["gross_margin_ttm"], _rank_col("gross_margin_ttm")),
        FundamentalAtom("net_margin", "rank(net_margin_ttm)", "Higher net margin after expenses.", "income", "fundamental_quality", ["net_margin_ttm"], _rank_col("net_margin_ttm")),
        FundamentalAtom("operating_cf_margin", "rank(operating_cf_margin_ttm)", "Higher operating cash flow relative to revenue.", "cashflow + income", "fundamental_quality", ["operating_cf_margin_ttm"], _rank_col("operating_cf_margin_ttm")),
        FundamentalAtom("cashflow_to_profit", "rank(cashflow_to_profit)", "Profit confirmed by operating cash flow.", "cashflow + income", "fundamental_quality", ["cashflow_to_profit"], _rank_col("cashflow_to_profit")),
        FundamentalAtom("reported_roe", "rank(roe)", "Reported return on equity from financial indicators.", "fina_indicator", "fundamental_quality", ["roe"], _rank_col("roe")),
        FundamentalAtom("reported_roa", "rank(roa)", "Reported return on assets from financial indicators.", "fina_indicator", "fundamental_quality", ["roa"], _rank_col("roa")),
        FundamentalAtom("reported_gross_margin", "rank(grossprofit_margin)", "Reported gross margin from financial indicators.", "fina_indicator", "fundamental_quality", ["grossprofit_margin"], _rank_col("grossprofit_margin")),
        FundamentalAtom("reported_net_margin", "rank(netprofit_margin)", "Reported net margin from financial indicators.", "fina_indicator", "fundamental_quality", ["netprofit_margin"], _rank_col("netprofit_margin")),
        FundamentalAtom("reported_ocf_to_or", "rank(ocf_to_or)", "Reported operating cash flow relative to revenue.", "fina_indicator", "fundamental_quality", ["ocf_to_or"], _rank_col("ocf_to_or")),
        FundamentalAtom("reported_ocf_to_profit", "rank(ocf_to_profit)", "Reported operating cash flow relative to profit.", "fina_indicator", "fundamental_quality", ["ocf_to_profit"], _rank_col("ocf_to_profit")),
        FundamentalAtom("eps", "rank(eps)", "Higher earnings per share.", "fina_indicator", "fundamental_quality", ["eps"], _rank_col("eps")),
        FundamentalAtom("bps", "rank(bps)", "Higher book value per share.", "fina_indicator", "fundamental_quality", ["bps"], _rank_col("bps")),
        FundamentalAtom("cfps", "rank(cfps)", "Higher cash flow per share.", "fina_indicator", "fundamental_quality", ["cfps"], _rank_col("cfps")),
        FundamentalAtom("revenue_yoy", "rank(revenue_yoy)", "Revenue growth.", "income", "fundamental_growth", ["revenue_yoy"], _rank_col("revenue_yoy")),
        FundamentalAtom("net_profit_yoy", "rank(net_profit_yoy)", "Net profit growth.", "income", "fundamental_growth", ["net_profit_yoy"], _rank_col("net_profit_yoy")),
        FundamentalAtom("ocf_yoy", "rank(ocf_yoy)", "Operating cash flow growth.", "cashflow", "fundamental_growth", ["ocf_yoy"], _rank_col("ocf_yoy")),
        FundamentalAtom("gross_margin_yoy", "rank(gross_margin_yoy)", "Gross margin improvement.", "income", "fundamental_growth", ["gross_margin_yoy"], _rank_col("gross_margin_yoy")),
        FundamentalAtom("asset_turnover_yoy", "rank(asset_turnover_yoy)", "Asset turnover improvement.", "income + balance", "fundamental_growth", ["asset_turnover_yoy"], _rank_col("asset_turnover_yoy")),
        FundamentalAtom("op_yoy", "rank(op_yoy)", "Operating profit growth from reported indicators.", "fina_indicator", "fundamental_growth", ["op_yoy"], _rank_col("op_yoy")),
        FundamentalAtom("or_yoy", "rank(or_yoy)", "Operating revenue growth from reported indicators.", "fina_indicator", "fundamental_growth", ["or_yoy"], _rank_col("or_yoy")),
        FundamentalAtom("netprofit_yoy", "rank(netprofit_yoy)", "Reported net profit growth.", "fina_indicator", "fundamental_growth", ["netprofit_yoy"], _rank_col("netprofit_yoy")),
        FundamentalAtom("low_pe", "1 - rank(pe_ttm)", "Lower earnings valuation.", "valuation overlay", "fundamental_value", ["pe_ttm"], _inv_rank_col("pe_ttm")),
        FundamentalAtom("earnings_yield", "rank(1 / pe_ttm)", "Higher earnings yield, with negative PE naturally penalized.", "valuation overlay", "fundamental_value", ["pe_ttm"], _rank_expr(["pe_ttm"], lambda df: op.safe_div(pd.Series(1.0, index=df.index), df["pe_ttm"]))),
        FundamentalAtom("low_pb", "1 - rank(pb)", "Lower book valuation.", "valuation overlay", "fundamental_value", ["pb"], _inv_rank_col("pb")),
        FundamentalAtom("low_ps", "1 - rank(ps_ttm)", "Lower sales valuation.", "valuation overlay", "fundamental_value", ["ps_ttm"], _inv_rank_col("ps_ttm")),
        FundamentalAtom("dividend_yield", "rank(dv_ttm)", "Higher dividend yield.", "valuation overlay", "fundamental_value", ["dv_ttm"], _rank_col("dv_ttm")),
        FundamentalAtom("current_ratio", "rank(current_ratio)", "Higher current ratio.", "fina_indicator + balance", "fundamental_safety", ["current_ratio"], _rank_col("current_ratio")),
        FundamentalAtom("quick_ratio", "rank(quick_ratio)", "Higher quick ratio.", "fina_indicator + balance", "fundamental_safety", ["quick_ratio"], _rank_col("quick_ratio")),
        FundamentalAtom("cash_to_liab", "rank(cash_to_liab)", "Cash buffer relative to liabilities.", "balance", "fundamental_safety", ["cash_to_liab"], _rank_col("cash_to_liab")),
        FundamentalAtom("cashflow_to_liab", "rank(n_cashflow_act_ttm / total_liab)", "Operating cash flow relative to total liabilities.", "cashflow + balance", "fundamental_safety", ["n_cashflow_act_ttm", "total_liab"], _rank_ratio("n_cashflow_act_ttm", "total_liab")),
        FundamentalAtom("profit_to_liab", "rank(net_profit_ttm / total_liab)", "Profit relative to total liabilities.", "income + balance", "fundamental_safety", ["net_profit_ttm", "total_liab"], _rank_ratio("net_profit_ttm", "total_liab")),
        FundamentalAtom("revenue_to_liab", "rank(total_revenue_ttm / total_liab)", "Revenue scale relative to total liabilities.", "income + balance", "fundamental_safety", ["total_revenue_ttm", "total_liab"], _rank_ratio("total_revenue_ttm", "total_liab")),
        FundamentalAtom("net_cash_to_assets", "rank((money_cap - total_liab) / total_assets)", "Net cash buffer after liabilities relative to assets.", "balance", "fundamental_safety", ["money_cap", "total_liab", "total_assets"], _rank_expr(["money_cap", "total_liab", "total_assets"], lambda df: op.safe_div(df["money_cap"] - df["total_liab"], df["total_assets"]))),
        FundamentalAtom("leverage_adjusted_roe", "rank(roe_ttm - debt_to_assets)", "Profitability after penalizing leverage burden.", "income + balance", "fundamental_safety", ["roe_ttm", "debt_to_assets"], _rank_expr(["roe_ttm", "debt_to_assets"], lambda df: df["roe_ttm"] - df["debt_to_assets"])),
        FundamentalAtom("low_debt_assets", "1 - rank(debt_to_assets)", "Lower liability burden.", "balance", "fundamental_safety", ["debt_to_assets"], _inv_rank_col("debt_to_assets")),
        FundamentalAtom("low_working_capital_pressure", "1 - rank(working_capital_pressure)", "Lower receivable and inventory occupation net of payables.", "balance", "fundamental_efficiency", ["working_capital_pressure"], _inv_rank_col("working_capital_pressure")),
        FundamentalAtom("asset_turnover", "rank(asset_turnover_ttm)", "Higher revenue generated per unit assets.", "income + balance", "fundamental_efficiency", ["asset_turnover_ttm"], _rank_col("asset_turnover_ttm")),
        FundamentalAtom("inventory_turnover", "rank(inv_turn)", "Higher inventory turnover.", "fina_indicator", "fundamental_efficiency", ["inv_turn"], _rank_col("inv_turn")),
        FundamentalAtom("receivable_turnover", "rank(ar_turn)", "Higher accounts receivable turnover.", "fina_indicator", "fundamental_efficiency", ["ar_turn"], _rank_col("ar_turn")),
        FundamentalAtom("low_capex_assets", "1 - rank(capex_to_assets)", "Lower capital expenditure intensity for a given operation base.", "cashflow + balance", "fundamental_efficiency", ["capex_to_assets"], _inv_rank_col("capex_to_assets")),
        FundamentalAtom("rd_intensity", "rank(rd_expense_intensity)", "Higher R&D intensity, used as a long-term investment proxy.", "income", "fundamental_growth", ["rd_expense_intensity"], _rank_col("rd_expense_intensity")),
        FundamentalAtom("fcf_yield", "rank(free_cashflow_ttm / total_mv)", "Free cash flow yield.", "cashflow + valuation overlay", "fundamental_value", ["free_cashflow_ttm", "total_mv"], _rank_ratio("free_cashflow_ttm", "total_mv")),
        FundamentalAtom("fcf_to_assets", "rank(free_cashflow_ttm / total_assets)", "Free cash flow generated per unit assets.", "cashflow + balance", "fundamental_quality", ["free_cashflow_ttm", "total_assets"], _rank_ratio("free_cashflow_ttm", "total_assets")),
        FundamentalAtom("fcf_margin", "rank(free_cashflow_ttm / total_revenue_ttm)", "Free cash flow relative to revenue.", "cashflow + income", "fundamental_quality", ["free_cashflow_ttm", "total_revenue_ttm"], _rank_ratio("free_cashflow_ttm", "total_revenue_ttm")),
        FundamentalAtom("fcf_to_profit", "rank(free_cashflow_ttm / net_profit_ttm)", "Free cash flow coverage of accounting profit.", "cashflow + income", "fundamental_quality", ["free_cashflow_ttm", "net_profit_ttm"], _rank_ratio("free_cashflow_ttm", "net_profit_ttm")),
        FundamentalAtom("profit_to_assets", "rank(net_profit_ttm / total_assets)", "Net profit relative to total assets.", "income + balance", "fundamental_quality", ["net_profit_ttm", "total_assets"], _rank_ratio("net_profit_ttm", "total_assets")),
        FundamentalAtom("ocf_to_assets", "rank(n_cashflow_act_ttm / total_assets)", "Operating cash flow relative to total assets.", "cashflow + balance", "fundamental_quality", ["n_cashflow_act_ttm", "total_assets"], _rank_ratio("n_cashflow_act_ttm", "total_assets")),
        FundamentalAtom("revenue_to_assets", "rank(total_revenue_ttm / total_assets)", "Revenue generated per unit assets.", "income + balance", "fundamental_efficiency", ["total_revenue_ttm", "total_assets"], _rank_ratio("total_revenue_ttm", "total_assets")),
        FundamentalAtom("cash_to_assets", "rank(money_cap / total_assets)", "Cash reserves relative to assets.", "balance", "fundamental_safety", ["money_cap", "total_assets"], _rank_ratio("money_cap", "total_assets")),
        FundamentalAtom("accrual_quality", "rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)", "Cash earnings exceed accrual earnings.", "cashflow + income + balance", "fundamental_quality", ["n_cashflow_act_ttm", "net_profit_ttm", "total_assets"], _rank_expr(["n_cashflow_act_ttm", "net_profit_ttm", "total_assets"], lambda df: op.safe_div(df["n_cashflow_act_ttm"] - df["net_profit_ttm"], df["total_assets"]))),
        FundamentalAtom("cash_conversion_spread", "rank((n_cashflow_act_ttm - net_profit_ttm) / total_revenue_ttm)", "Operating cash flow exceeds accounting profit relative to revenue.", "cashflow + income", "fundamental_quality", ["n_cashflow_act_ttm", "net_profit_ttm", "total_revenue_ttm"], _rank_expr(["n_cashflow_act_ttm", "net_profit_ttm", "total_revenue_ttm"], lambda df: op.safe_div(df["n_cashflow_act_ttm"] - df["net_profit_ttm"], df["total_revenue_ttm"]))),
        FundamentalAtom("gross_to_net_margin", "rank(net_margin_ttm / gross_margin_ttm)", "Net margin retained from gross margin after expenses.", "income", "fundamental_efficiency", ["net_margin_ttm", "gross_margin_ttm"], _rank_ratio("net_margin_ttm", "gross_margin_ttm")),
        FundamentalAtom("low_inventory_assets", "1 - rank(inventories / total_assets)", "Lower inventory occupation.", "balance", "fundamental_efficiency", ["inventories", "total_assets"], _rank_ratio("inventories", "total_assets", high_good=False)),
        FundamentalAtom("low_receivable_assets", "1 - rank(accounts_receiv / total_assets)", "Lower receivable occupation.", "balance", "fundamental_efficiency", ["accounts_receiv", "total_assets"], _rank_ratio("accounts_receiv", "total_assets", high_good=False)),
        FundamentalAtom("low_receivable_inventory_assets", "1 - rank((accounts_receiv + inventories) / total_assets)", "Lower combined receivable and inventory occupation.", "balance", "fundamental_efficiency", ["accounts_receiv", "inventories", "total_assets"], _rank_expr(["accounts_receiv", "inventories", "total_assets"], lambda df: op.safe_div(df["accounts_receiv"] + df["inventories"], df["total_assets"]), high_good=False)),
    ]
    return atoms


def _combine_series(series: list[pd.Series], mode: str) -> pd.Series:
    if not series:
        raise ValueError("empty series list")
    if mode == "mean":
        return pd.concat(series, axis=1).mean(axis=1, skipna=False)
    if mode == "sum":
        return pd.concat(series, axis=1).sum(axis=1, skipna=False)
    if mode == "hmean2":
        if len(series) != 2:
            raise ValueError("hmean2 requires exactly two series")
        return op.harm_mean(series[0], series[1])
    if mode == "hmean2_plus":
        if len(series) < 3:
            raise ValueError("hmean2_plus requires at least three series")
        return op.harm_mean(series[0], series[1]) + pd.concat(series[2:], axis=1).mean(axis=1, skipna=False)
    raise ValueError(f"unknown combine mode: {mode}")


def _make_combo(
    short_name: str,
    atom_names: Sequence[str],
    atom_map: dict[str, FundamentalAtom],
    family: str,
    description: str,
    mode: str = "mean",
    complexity: int | None = None,
) -> AlphaCandidate:
    atoms = [atom_map[name] for name in atom_names]
    required = sorted({col for atom in atoms for col in atom.required_columns})
    expression = f"{mode}(" + ", ".join(atom.expression for atom in atoms) + ")"

    def _compute(df: pd.DataFrame, names: tuple[str, ...] = tuple(atom_names), m: str = mode, s: str = short_name) -> pd.Series:
        return _cached_series(
            df,
            f"combo:{s}",
            lambda: _combine_series([atom_map[name].compute(df) for name in names], m),
        )

    return AlphaCandidate(
        name=f"fund_combo_{short_name}",
        expression=expression,
        description=description,
        required_columns=required,
        window=1,
        complexity=complexity or max(3, 2 + len(atom_names)),
        family=family,
        compute=_compute,
    )


def _neutral_variant(base: AlphaCandidate, kind: str) -> AlphaCandidate:
    if base.name.startswith("fund_combo_"):
        short = base.name.replace("fund_combo_", "")
    elif base.name.startswith("fund_atom_"):
        short = base.name.replace("fund_atom_", "")
    else:
        short = base.name
    if kind == "industry":
        required = sorted(set(base.required_columns) | {"industry"})
        name = f"fund_ind_neu_{short}"
        expression = f"industry_neutralize({base.expression})"
        description = base.description + " Industry-neutralized to reduce sector structure bias."

        def _compute(df: pd.DataFrame, b: AlphaCandidate = base) -> pd.Series:
            score = b.compute(df)
            return op.industry_neutralize(df.assign(_fund_score=score), "_fund_score")

        return AlphaCandidate(name, expression, description, required, _compute, window=base.window, complexity=base.complexity + 2, family=base.family)
    if kind == "size":
        required = sorted(set(base.required_columns) | {"log_mv"})
        name = f"fund_size_neu_{short}"
        expression = f"size_neutralize({base.expression})"
        description = base.description + " Size-neutralized to reduce market-cap style exposure."

        def _compute(df: pd.DataFrame, b: AlphaCandidate = base) -> pd.Series:
            score = b.compute(df)
            return op.size_neutralize(df.assign(_fund_score=score), "_fund_score")

        return AlphaCandidate(name, expression, description, required, _compute, window=base.window, complexity=base.complexity + 2, family=base.family)
    raise ValueError("kind must be industry or size")


def fundamental_operator_candidates(
    include_atoms: bool = True,
    include_neutralized: bool = True,
) -> list[AlphaCandidate]:
    """Build a deterministic candidate set from fundamental atoms and combinations."""
    atoms = fundamental_atoms()
    atom_map = {atom.name: atom for atom in atoms}
    candidates: list[AlphaCandidate] = []
    if include_atoms:
        candidates.extend(_atom_candidate(atom) for atom in atoms)

    quality = [
        "roe", "roa", "net_margin", "gross_margin", "operating_cf_margin", "cashflow_to_profit",
        "reported_ocf_to_or", "reported_ocf_to_profit", "reported_net_margin",
        "eps", "profit_to_assets", "ocf_to_assets", "fcf_to_assets", "fcf_margin",
        "fcf_to_profit", "accrual_quality", "cash_conversion_spread",
    ]
    value = ["low_pb", "low_pe", "earnings_yield", "low_ps", "fcf_yield", "dividend_yield"]
    growth = ["revenue_yoy", "net_profit_yoy", "ocf_yoy", "gross_margin_yoy", "asset_turnover_yoy", "op_yoy", "or_yoy", "netprofit_yoy"]
    safety = [
        "cash_to_liab", "cashflow_to_liab", "profit_to_liab", "revenue_to_liab", "net_cash_to_assets",
        "cash_to_assets", "current_ratio", "quick_ratio", "leverage_adjusted_roe", "low_debt_assets",
    ]
    efficiency = [
        "asset_turnover", "revenue_to_assets", "inventory_turnover", "receivable_turnover",
        "low_working_capital_pressure", "low_inventory_assets", "low_receivable_assets",
        "low_receivable_inventory_assets", "gross_to_net_margin", "low_capex_assets",
    ]

    for q in [
        "roe", "roa", "net_margin", "operating_cf_margin", "cashflow_to_profit",
        "reported_ocf_to_or", "reported_ocf_to_profit", "profit_to_assets",
        "ocf_to_assets", "fcf_to_assets", "fcf_margin", "fcf_to_profit",
        "cash_conversion_spread", "leverage_adjusted_roe", "eps",
    ]:
        for v in ["low_pb", "low_pe", "earnings_yield", "low_ps", "fcf_yield"]:
            candidates.append(_make_combo(f"{q}_{v}", [q, v], atom_map, "fundamental_value", f"{q} quality combined with {v} valuation."))

    for g in ["revenue_yoy", "net_profit_yoy", "ocf_yoy", "gross_margin_yoy", "asset_turnover_yoy"]:
        for q in [
            "cashflow_to_profit", "operating_cf_margin", "reported_ocf_to_or",
            "reported_ocf_to_profit", "roe", "net_margin", "fcf_margin",
            "accrual_quality", "cash_conversion_spread",
        ]:
            candidates.append(_make_combo(f"{g}_{q}", [g, q], atom_map, "fundamental_growth", f"{g} growth confirmed by {q} quality."))

    for s in safety:
        for v in ["low_pb", "low_pe", "earnings_yield", "fcf_yield"]:
            candidates.append(_make_combo(f"{s}_{v}", [s, v], atom_map, "fundamental_safety", f"{s} balance-sheet safety combined with {v} valuation."))

    for e in efficiency:
        for q in ["net_margin", "cashflow_to_profit", "reported_ocf_to_profit", "fcf_margin", "low_pb"]:
            candidates.append(_make_combo(f"{e}_{q}", [e, q], atom_map, "fundamental_efficiency", f"{e} operating efficiency combined with {q}."))

    triple_specs = [
        ("revenue_profit_cash_hmean", ["revenue_yoy", "net_profit_yoy", "cashflow_to_profit"], "fundamental_growth", "Revenue and profit grow together and are confirmed by cash conversion.", "hmean2_plus"),
        ("revenue_profit_low_pe_hmean", ["revenue_yoy", "net_profit_yoy", "low_pe"], "fundamental_growth", "Synchronous growth bought at lower earnings valuation.", "hmean2_plus"),
        ("roe_cash_low_pb", ["roe", "cashflow_to_profit", "low_pb"], "fundamental_value", "ROE and cash conversion at lower book valuation.", "mean"),
        ("roa_ocf_low_pb", ["roa", "ocf_to_assets", "low_pb"], "fundamental_value", "Asset-level profitability and cash generation at lower book valuation.", "mean"),
        ("dupont_low_pb", ["net_margin", "asset_turnover", "low_pb"], "fundamental_value", "DuPont margin and turnover quality adjusted by book valuation.", "mean"),
        ("cash_safety_low_pb", ["cash_to_liab", "low_debt_assets", "low_pb"], "fundamental_safety", "Cash buffer, lower leverage, and lower book valuation.", "mean"),
        ("working_capital_cash_growth", ["low_working_capital_pressure", "cashflow_to_profit", "net_profit_yoy"], "fundamental_growth", "Clean growth with cash conversion and low working-capital pressure.", "mean"),
        ("inventory_receivable_efficiency", ["low_inventory_assets", "low_receivable_assets", "receivable_turnover"], "fundamental_efficiency", "Lower inventory/receivable occupation with stronger receivable turnover.", "mean"),
        ("ocf_assets_low_debt", ["ocf_to_assets", "cash_to_assets", "low_debt_assets"], "fundamental_safety", "Cash generation and cash reserves without high leverage.", "mean"),
        ("rd_growth_cash", ["rd_intensity", "revenue_yoy", "cashflow_to_profit"], "fundamental_growth", "R&D intensity with revenue growth and cash confirmation.", "mean"),
        ("margin_growth_low_ps", ["gross_margin_yoy", "net_margin", "low_ps"], "fundamental_value", "Improving margin and margin level at lower sales valuation.", "mean"),
        ("eps_bps_low_pb", ["eps", "bps", "low_pb"], "fundamental_value", "Per-share earnings and book value at lower book valuation.", "mean"),
        ("cfps_cash_low_pe", ["cfps", "cashflow_to_profit", "low_pe"], "fundamental_value", "Per-share cash flow and cash conversion at lower earnings valuation.", "mean"),
        ("cash_profit_debt_cover", ["cashflow_to_liab", "profit_to_liab", "low_debt_assets"], "fundamental_safety", "Cash flow and profit cover liabilities without high leverage.", "mean"),
        ("fcf_margin_low_ps", ["fcf_margin", "reported_ocf_to_or", "low_ps"], "fundamental_value", "Free cash flow margin and reported cash revenue quality at lower sales valuation.", "mean"),
        ("accrual_cash_low_pb", ["accrual_quality", "cash_conversion_spread", "low_pb"], "fundamental_value", "Cash earnings exceed accrual earnings at lower book valuation.", "mean"),
        ("working_capital_fcf_value", ["low_receivable_inventory_assets", "fcf_margin", "low_pb"], "fundamental_efficiency", "Low receivable and inventory occupation with free cash flow and lower valuation.", "mean"),
        ("reported_cash_margin_low_pb", ["reported_ocf_to_profit", "reported_net_margin", "low_pb"], "fundamental_value", "Reported cash conversion and net margin at lower book valuation.", "mean"),
        ("leverage_adjusted_quality_value", ["leverage_adjusted_roe", "cashflow_to_profit", "low_pb"], "fundamental_safety", "Profitability not driven by leverage, confirmed by cash conversion and valuation.", "mean"),
        ("fcf_assets_low_pb", ["fcf_to_assets", "low_debt_assets", "low_pb"], "fundamental_value", "Free cash flow generated by assets with lower leverage and lower book valuation.", "mean"),
        ("revenue_liab_efficiency_growth", ["revenue_to_liab", "asset_turnover_yoy", "revenue_yoy"], "fundamental_growth", "Revenue scale covers liabilities while operating efficiency and revenue improve.", "mean"),
        ("earnings_yield_quality_cash", ["earnings_yield", "cashflow_to_profit", "reported_ocf_to_profit"], "fundamental_value", "Earnings yield confirmed by derived and reported cash conversion.", "mean"),
        ("gross_net_cash_value", ["gross_to_net_margin", "operating_cf_margin", "low_ps"], "fundamental_value", "Gross margin retained as net margin and cash flow at lower sales valuation.", "mean"),
    ]
    base_for_neutral: list[AlphaCandidate] = []
    for short, atoms_, family, desc, mode in triple_specs:
        candidate = _make_combo(short, atoms_, atom_map, family, desc, mode=mode, complexity=6)
        candidates.append(candidate)
        base_for_neutral.append(candidate)

    if include_neutralized:
        neutral_shorts = {
            "roe_low_pb", "roe_low_pe", "roa_low_pb", "net_margin_low_ps", "operating_cf_margin_low_pb",
            "cashflow_to_profit_low_pb", "profit_to_assets_low_pb", "ocf_to_assets_low_pb",
            "revenue_profit_cash_hmean", "roe_cash_low_pb", "roa_ocf_low_pb", "dupont_low_pb",
            "cash_safety_low_pb", "working_capital_cash_growth", "eps_bps_low_pb",
            "reported_ocf_to_profit_low_pb", "fcf_margin_low_ps", "accrual_cash_low_pb",
            "working_capital_fcf_value", "reported_cash_margin_low_pb", "leverage_adjusted_quality_value",
            "fcf_assets_low_pb", "earnings_yield_quality_cash", "gross_net_cash_value",
        }
        by_short = {}
        for candidate in candidates:
            if candidate.name.startswith("fund_combo_"):
                by_short[candidate.name.replace("fund_combo_", "")] = candidate
        for short in sorted(neutral_shorts):
            base = by_short.get(short)
            if base is None:
                continue
            candidates.append(_neutral_variant(base, "industry"))
            if any(token in short for token in ["low_pb", "low_pe", "low_ps", "fcf", "value", "dupont"]):
                candidates.append(_neutral_variant(base, "size"))

    # Stable deterministic de-duplication by name.
    deduped: dict[str, AlphaCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.name, candidate)
    return list(deduped.values())


def available_fundamental_operator_candidates(df: pd.DataFrame) -> list[AlphaCandidate]:
    return [candidate for candidate in fundamental_operator_candidates() if candidate.is_available(df)]
