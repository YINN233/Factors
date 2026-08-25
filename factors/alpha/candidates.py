"""
Candidate alpha definitions inspired by the CSJC Alpha report summaries.

The library starts with interpretable daily price/volume candidates and leaves
hooks for point-in-time fundamental fields when they are available.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import pandas as pd

from factors.alpha import operators as op


ComputeFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class AlphaCandidate:
    name: str
    expression: str
    description: str
    required_columns: Sequence[str]
    compute: ComputeFunc
    window: int = 1
    complexity: int = 1
    family: str = "price_volume"

    def is_available(self, df: pd.DataFrame) -> bool:
        return set(self.required_columns).issubset(df.columns)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        missing = sorted(set(self.required_columns) - set(df.columns))
        if missing:
            raise KeyError(f"{self.name} missing required columns: {missing}")
        values = self.compute(df)
        return values.replace([np.inf, -np.inf], np.nan)


def _price_cols(df: pd.DataFrame) -> tuple[str, str, str, str]:
    open_col = "open_adj" if "open_adj" in df.columns else "open"
    high_col = "high_adj" if "high_adj" in df.columns else "high"
    low_col = "low_adj" if "low_adj" in df.columns else "low"
    close_col = "close_adj" if "close_adj" in df.columns else "close"
    return open_col, high_col, low_col, close_col


def _required_price_cols() -> list[str]:
    return ["open_adj", "high_adj", "low_adj", "close_adj"]


def _with_temp(df: pd.DataFrame, **cols: pd.Series) -> pd.DataFrame:
    tmp = df.copy()
    for key, value in cols.items():
        tmp[key] = value
    return tmp


def _ret_1d(df: pd.DataFrame) -> pd.Series:
    return op.ts_pct(df, _price_cols(df)[3], 1)


def _volume_change(df: pd.DataFrame) -> pd.Series:
    return op.ts_pct(df, "volume", 1)


def _intraday_return(df: pd.DataFrame) -> pd.Series:
    open_col, _, _, close_col = _price_cols(df)
    return op.safe_div(df[close_col], df[open_col]) - 1.0


def _overnight_return(df: pd.DataFrame) -> pd.Series:
    open_col, _, _, close_col = _price_cols(df)
    return op.safe_div(df[open_col], op.ts_delay(df, close_col, 1)) - 1.0


def _oi_spread(df: pd.DataFrame) -> pd.Series:
    return _intraday_return(df) - _overnight_return(df)


def _low_close_support(df: pd.DataFrame) -> pd.Series:
    _, _, low_col, close_col = _price_cols(df)
    return op.safe_div(df[close_col] - df[low_col], df[close_col])


def _close_location(df: pd.DataFrame) -> pd.Series:
    _, high_col, low_col, close_col = _price_cols(df)
    return op.safe_div(df[close_col] - df[low_col], df[high_col] - df[low_col])


def _range_return(df: pd.DataFrame) -> pd.Series:
    _, high_col, low_col, _ = _price_cols(df)
    return op.safe_div(df[high_col], df[low_col]) - 1.0


def _amount_expansion(df: pd.DataFrame, window: int) -> pd.Series:
    return op.safe_div(df["amount"], op.ts_mean(df, "amount", window)) - 1.0


def _turnover_expansion(df: pd.DataFrame, window: int) -> pd.Series:
    return op.safe_div(df["turnover_rate"], op.ts_mean(df, "turnover_rate", window)) - 1.0


def _rank(df: pd.DataFrame, col: str) -> pd.Series:
    return op.cs_rank(df, col)


def _inv_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return 1.0 - op.cs_rank(df, col)


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    return op.cs_zscore(df, col)


def _fund_ratio(df: pd.DataFrame, num_col: str, den_col: str) -> pd.Series:
    return op.safe_div(df[num_col], df[den_col])


def default_daily_alpha_candidates(windows: Iterable[int] = (5, 20)) -> List[AlphaCandidate]:
    candidates: List[AlphaCandidate] = []

    for window in windows:
        candidates.append(
            AlphaCandidate(
                name=f"mom_close_{window}",
                expression=f"close / delay(close,{window}) - 1",
                description="Close-to-close momentum.",
                required_columns=["close_adj"],
                window=window,
                complexity=2,
                compute=lambda df, w=window: op.ts_pct(df, _price_cols(df)[3], w),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"volatility_{window}",
                expression=f"ts_std(close / delay(close,1) - 1,{window})",
                description="Recent return volatility; useful as a risk or reversal signal.",
                required_columns=["close_adj"],
                window=window,
                complexity=3,
                compute=lambda df, w=window: op.ts_std(
                    _with_temp(df, _ret_1d=_ret_1d(df)),
                    "_ret_1d",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"intraday_strength_{window}",
                expression=f"ts_mean(close / open - 1,{window})",
                description="Average intraday O2C strength, matching the overnight/intraday split idea.",
                required_columns=["open_adj", "close_adj"],
                window=window,
                complexity=3,
                family="overnight_intraday",
                compute=lambda df, w=window: op.ts_mean(
                    _with_temp(df, _intraday=_intraday_return(df)),
                    "_intraday",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"overnight_reversal_{window}",
                expression=f"-ts_mean(open / delay(close,1) - 1,{window})",
                description="Negative overnight return pressure, based on the A-share overnight anomaly.",
                required_columns=["open_adj", "close_adj"],
                window=window,
                complexity=4,
                family="overnight_intraday",
                compute=lambda df, w=window: -op.ts_mean(
                    _with_temp(df, _overnight=_overnight_return(df)),
                    "_overnight",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"oi_spread_{window}",
                expression=f"ts_mean(intraday_return - overnight_return,{window})",
                description="Spread between intraday compensation and overnight pressure.",
                required_columns=["open_adj", "close_adj"],
                window=window,
                complexity=5,
                family="overnight_intraday",
                compute=lambda df, w=window: op.ts_mean(
                    _with_temp(df, _oi_spread=_oi_spread(df)),
                    "_oi_spread",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"amount_expansion_{window}",
                expression=f"amount / ts_mean(amount,{window}) - 1",
                description="Trading amount expansion as market confirmation.",
                required_columns=["amount"],
                window=window,
                complexity=3,
                compute=lambda df, w=window: _amount_expansion(df, w),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"turnover_expansion_{window}",
                expression=f"turnover_rate / ts_mean(turnover_rate,{window}) - 1",
                description="Turnover expansion as a lower-noise attention confirmation leg.",
                required_columns=["turnover_rate"],
                window=window,
                complexity=3,
                compute=lambda df, w=window: _turnover_expansion(df, w),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"turnover_stability_{window}",
                expression=f"-ts_std(turnover_rate,{window})",
                description="Stable turnover; used as a liquidity and crowding-risk penalty.",
                required_columns=["turnover_rate"],
                window=window,
                complexity=3,
                compute=lambda df, w=window: -op.ts_std(df, "turnover_rate", w),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"price_volume_corr_{window}",
                expression=f"ts_corr(return_1d, volume_change,{window})",
                description="Rolling relation between price movement and volume participation.",
                required_columns=["close_adj", "volume"],
                window=window,
                complexity=5,
                compute=lambda df, w=window: op.ts_corr(
                    _with_temp(
                        df,
                        _ret_1d=_ret_1d(df),
                        _volume_change=_volume_change(df),
                    ),
                    "_ret_1d",
                    "_volume_change",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"intraday_amount_confirm_{window}",
                expression=f"ts_mean(intraday_return * (amount / ts_mean(amount,{window})),{window})",
                description="Intraday O2C strength confirmed by relative trading amount.",
                required_columns=["open_adj", "close_adj", "amount"],
                window=window,
                complexity=6,
                family="overnight_intraday",
                compute=lambda df, w=window: op.ts_mean(
                    _with_temp(
                        df,
                        _intraday_amount_confirm=_intraday_return(df) * (1.0 + _amount_expansion(df, w)),
                    ),
                    "_intraday_amount_confirm",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"overnight_amount_corr_{window}",
                expression=f"ts_corr(overnight_return, amount / ts_mean(amount,{window}) - 1,{window})",
                description="Correlation between overnight pressure and amount expansion.",
                required_columns=["open_adj", "close_adj", "amount"],
                window=window,
                complexity=6,
                family="overnight_intraday",
                compute=lambda df, w=window: op.ts_corr(
                    _with_temp(
                        df,
                        _overnight=_overnight_return(df),
                        _amount_expansion=_amount_expansion(df, w),
                    ),
                    "_overnight",
                    "_amount_expansion",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"close_location_{window}",
                expression=f"ts_mean((close - low) / (high - low),{window})",
                description="Close position inside the daily range; captures late-session price support.",
                required_columns=["high_adj", "low_adj", "close_adj"],
                window=window,
                complexity=4,
                compute=lambda df, w=window: op.ts_mean(
                    _with_temp(df, _close_location=_close_location(df)),
                    "_close_location",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"range_pressure_{window}",
                expression=f"-ts_mean(high / low - 1,{window})",
                description="Low intraday range as a stability preference and volatility penalty.",
                required_columns=["high_adj", "low_adj"],
                window=window,
                complexity=4,
                compute=lambda df, w=window: -op.ts_mean(
                    _with_temp(df, _range_return=_range_return(df)),
                    "_range_return",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"liquidity_preference_{window}",
                expression=f"-ts_mean(abs(return_1d) / amount,{window})",
                description="Negative Amihud-style illiquidity; prefers tradable names with lower price impact.",
                required_columns=["close_adj", "amount"],
                window=window,
                complexity=5,
                compute=lambda df, w=window: -op.ts_mean(
                    _with_temp(df, _illiq=op.safe_div(_ret_1d(df).abs(), df["amount"])),
                    "_illiq",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"low_close_support_{window}",
                expression=f"ts_mean((close - low) / close,{window})",
                description="Price support from the daily low to close; a simple market-confirmation leg.",
                required_columns=["low_adj", "close_adj"],
                window=window,
                complexity=3,
                compute=lambda df, w=window: op.ts_mean(
                    _with_temp(df, _support=_low_close_support(df)),
                    "_support",
                    w,
                ),
            )
        )

    candidates.append(
        AlphaCandidate(
            name="reversal_1d",
            expression="-(close / delay(close,1) - 1)",
            description="One-day reversal baseline.",
            required_columns=["close_adj"],
            window=1,
            complexity=2,
            compute=lambda df: -op.ts_pct(df, _price_cols(df)[3], 1),
        )
    )

    return candidates


def optional_fundamental_candidates(windows: Iterable[int] = (20,)) -> List[AlphaCandidate]:
    candidates: List[AlphaCandidate] = []

    candidates.extend(
        [
            AlphaCandidate(
                name="quality_roe_ocf",
                expression="rank(roe_ttm) + rank(cashflow_to_profit)",
                description="Profitability backed by operating cash conversion.",
                required_columns=["roe_ttm", "cashflow_to_profit"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: _rank(df, "roe_ttm") + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="quality_roa_ocf",
                expression="rank(roa_ttm) + rank(cashflow_to_profit)",
                description="Asset-level profitability with cash-flow confirmation.",
                required_columns=["roa_ttm", "cashflow_to_profit"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: _rank(df, "roa_ttm") + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="cash_profit_cover",
                expression="rank(n_cashflow_act_ttm / net_profit_ttm)",
                description="Operating cash flow coverage of accounting earnings.",
                required_columns=["n_cashflow_act_ttm", "net_profit_ttm"],
                window=1,
                complexity=3,
                family="fundamental_quality",
                compute=lambda df: op.cs_rank(
                    _with_temp(df, _cash_profit_cover=_fund_ratio(df, "n_cashflow_act_ttm", "net_profit_ttm")),
                    "_cash_profit_cover",
                ),
            ),
            AlphaCandidate(
                name="low_accrual_to_assets",
                expression="rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)",
                description="Lower accrual pressure; cash earnings are preferred to accrual earnings.",
                required_columns=["n_cashflow_act_ttm", "net_profit_ttm", "total_assets"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: op.cs_rank(
                    _with_temp(
                        df,
                        _low_accrual=op.safe_div(df["n_cashflow_act_ttm"] - df["net_profit_ttm"], df["total_assets"]),
                    ),
                    "_low_accrual",
                ),
            ),
            AlphaCandidate(
                name="gross_margin_quality",
                expression="rank(gross_margin_ttm) + rank(gross_margin_yoy)",
                description="High and improving gross margin as pricing-power signal.",
                required_columns=["gross_margin_ttm", "gross_margin_yoy"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: _rank(df, "gross_margin_ttm") + _rank(df, "gross_margin_yoy"),
            ),
            AlphaCandidate(
                name="quality_growth_hmean",
                expression="harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit)",
                description="Growth that is simultaneous in revenue, profit, and cash conversion.",
                required_columns=["revenue_yoy", "net_profit_yoy", "cashflow_to_profit"],
                window=1,
                complexity=6,
                family="fundamental_growth",
                compute=lambda df: op.harm_mean(_rank(df, "revenue_yoy"), _rank(df, "net_profit_yoy"))
                + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="ocf_growth_quality",
                expression="rank(ocf_yoy) + rank(cashflow_to_profit)",
                description="Operating cash-flow growth with earnings-quality confirmation.",
                required_columns=["ocf_yoy", "cashflow_to_profit"],
                window=1,
                complexity=4,
                family="fundamental_growth",
                compute=lambda df: _rank(df, "ocf_yoy") + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="asset_turnover_improve",
                expression="rank(asset_turnover_yoy)",
                description="Improving operating efficiency measured by asset turnover.",
                required_columns=["asset_turnover_yoy"],
                window=1,
                complexity=2,
                family="fundamental_efficiency",
                compute=lambda df: _rank(df, "asset_turnover_yoy"),
            ),
            AlphaCandidate(
                name="debt_cash_safety",
                expression="rank(n_cashflow_act_ttm / total_liab) - rank(debt_to_assets)",
                description="Balance-sheet safety: cash generation relative to debt minus leverage burden.",
                required_columns=["n_cashflow_act_ttm", "total_liab", "debt_to_assets"],
                window=1,
                complexity=5,
                family="fundamental_safety",
                compute=lambda df: op.cs_rank(
                    _with_temp(df, _cash_to_debt=_fund_ratio(df, "n_cashflow_act_ttm", "total_liab")),
                    "_cash_to_debt",
                )
                - _rank(df, "debt_to_assets"),
            ),
            AlphaCandidate(
                name="low_leverage_quality",
                expression="rank(roe_ttm) - rank(debt_to_assets)",
                description="Profitability that does not rely on excessive leverage.",
                required_columns=["roe_ttm", "debt_to_assets"],
                window=1,
                complexity=4,
                family="fundamental_safety",
                compute=lambda df: _rank(df, "roe_ttm") - _rank(df, "debt_to_assets"),
            ),
            AlphaCandidate(
                name="capex_efficiency",
                expression="rank(revenue_yoy) - rank(capex_to_assets)",
                description="Growth delivered with lower reinvestment intensity.",
                required_columns=["revenue_yoy", "capex_to_assets"],
                window=1,
                complexity=4,
                family="fundamental_efficiency",
                compute=lambda df: _rank(df, "revenue_yoy") - _rank(df, "capex_to_assets"),
            ),
            AlphaCandidate(
                name="roe_value_pb",
                expression="rank(roe_ttm) - rank(pb)",
                description="Quality at a reasonable book valuation.",
                required_columns=["roe_ttm", "pb"],
                window=1,
                complexity=4,
                family="fundamental_value",
                compute=lambda df: _rank(df, "roe_ttm") - _rank(df, "pb"),
            ),
            AlphaCandidate(
                name="earnings_yield_quality",
                expression="rank(1 / pe_ttm) + rank(cashflow_to_profit)",
                description="Cheap earnings yield combined with cash-flow quality.",
                required_columns=["pe_ttm", "cashflow_to_profit"],
                window=1,
                complexity=5,
                family="fundamental_value",
                compute=lambda df: op.cs_rank(
                    _with_temp(df, _earnings_yield=op.safe_div(pd.Series(1.0, index=df.index), df["pe_ttm"])),
                    "_earnings_yield",
                )
                + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="shareholder_yield_quality",
                expression="rank(dv_ttm) + rank(roe_ttm)",
                description="Dividend yield plus profitability, preferring shareholder-return quality.",
                required_columns=["dv_ttm", "roe_ttm"],
                window=1,
                complexity=4,
                family="fundamental_value",
                compute=lambda df: _rank(df, "dv_ttm") + _rank(df, "roe_ttm"),
            ),
            AlphaCandidate(
                name="free_cashflow_yield_quality",
                expression="rank(free_cashflow_ttm / total_mv) + rank(cashflow_to_profit)",
                description="Free-cash-flow yield with accounting-profit cash conversion.",
                required_columns=["free_cashflow_ttm", "total_mv", "cashflow_to_profit"],
                window=1,
                complexity=5,
                family="fundamental_value",
                compute=lambda df: op.cs_rank(
                    _with_temp(df, _fcf_yield=_fund_ratio(df, "free_cashflow_ttm", "total_mv")),
                    "_fcf_yield",
                )
                + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="cash_revenue_quality",
                expression="rank(operating_cf_margin_ttm) + rank(ocf_to_or)",
                description="Operating cash flow relative to revenue, combining derived TTM and reported indicators.",
                required_columns=["operating_cf_margin_ttm", "ocf_to_or"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: _rank(df, "operating_cf_margin_ttm") + _rank(df, "ocf_to_or"),
            ),
            AlphaCandidate(
                name="margin_value_ps",
                expression="rank(net_margin_ttm) - rank(ps_ttm)",
                description="Net margin bought at a reasonable sales valuation.",
                required_columns=["net_margin_ttm", "ps_ttm"],
                window=1,
                complexity=4,
                family="fundamental_value",
                compute=lambda df: _rank(df, "net_margin_ttm") - _rank(df, "ps_ttm"),
            ),
            AlphaCandidate(
                name="solvency_liquidity_quality",
                expression="rank(current_ratio) + rank(quick_ratio) - rank(debt_to_assets)",
                description="Short-term solvency and low leverage balance-sheet quality.",
                required_columns=["current_ratio", "quick_ratio", "debt_to_assets"],
                window=1,
                complexity=5,
                family="fundamental_safety",
                compute=lambda df: _rank(df, "current_ratio") + _rank(df, "quick_ratio") - _rank(df, "debt_to_assets"),
            ),
            AlphaCandidate(
                name="working_capital_light",
                expression="-rank(working_capital_pressure)",
                description="Lower receivable and inventory pressure relative to assets.",
                required_columns=["working_capital_pressure"],
                window=1,
                complexity=2,
                family="fundamental_efficiency",
                compute=lambda df: -_rank(df, "working_capital_pressure"),
            ),
            AlphaCandidate(
                name="turnover_efficiency_combo",
                expression="rank(inv_turn) + rank(ar_turn)",
                description="Inventory and receivable turnover efficiency.",
                required_columns=["inv_turn", "ar_turn"],
                window=1,
                complexity=4,
                family="fundamental_efficiency",
                compute=lambda df: _rank(df, "inv_turn") + _rank(df, "ar_turn"),
            ),
            AlphaCandidate(
                name="reported_margin_quality",
                expression="rank(grossprofit_margin) + rank(netprofit_margin)",
                description="Reported gross and net margin quality from fina_indicator.",
                required_columns=["grossprofit_margin", "netprofit_margin"],
                window=1,
                complexity=4,
                family="fundamental_quality",
                compute=lambda df: _rank(df, "grossprofit_margin") + _rank(df, "netprofit_margin"),
            ),
            AlphaCandidate(
                name="eps_bps_value_quality",
                expression="rank(eps) + rank(bps) - rank(pb)",
                description="Per-share earnings and book value quality adjusted by book valuation.",
                required_columns=["eps", "bps", "pb"],
                window=1,
                complexity=5,
                family="fundamental_value",
                compute=lambda df: _rank(df, "eps") + _rank(df, "bps") - _rank(df, "pb"),
            ),
            AlphaCandidate(
                name="cash_buffer_value",
                expression="rank(cash_to_liab) - rank(pb)",
                description="Cash buffer relative to liabilities, adjusted for book valuation.",
                required_columns=["cash_to_liab", "pb"],
                window=1,
                complexity=4,
                family="fundamental_safety",
                compute=lambda df: _rank(df, "cash_to_liab") - _rank(df, "pb"),
            ),
            AlphaCandidate(
                name="growth_value_balance",
                expression="harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) - rank(pe_ttm)",
                description="Fundamental growth adjusted for earnings valuation.",
                required_columns=["revenue_yoy", "net_profit_yoy", "pe_ttm"],
                window=1,
                complexity=5,
                family="fundamental_growth",
                compute=lambda df: op.harm_mean(_rank(df, "revenue_yoy"), _rank(df, "net_profit_yoy"))
                - _rank(df, "pe_ttm"),
            ),
            AlphaCandidate(
                name="operating_efficiency_quality",
                expression="rank(asset_turnover_ttm) + rank(net_margin_ttm)",
                description="DuPont-style operating efficiency and margin quality.",
                required_columns=["asset_turnover_ttm", "net_margin_ttm"],
                window=1,
                complexity=4,
                family="fundamental_efficiency",
                compute=lambda df: _rank(df, "asset_turnover_ttm") + _rank(df, "net_margin_ttm"),
            ),
            AlphaCandidate(
                name="clean_growth_quality",
                expression="rank(cashflow_to_profit) + rank(net_profit_yoy) - rank(working_capital_pressure)",
                description="Profit growth with cash conversion and low working-capital pressure.",
                required_columns=["cashflow_to_profit", "net_profit_yoy", "working_capital_pressure"],
                window=1,
                complexity=5,
                family="fundamental_growth",
                compute=lambda df: _rank(df, "cashflow_to_profit")
                + _rank(df, "net_profit_yoy")
                - _rank(df, "working_capital_pressure"),
            ),
            AlphaCandidate(
                name="robust_margin_value_ps",
                expression="robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm)",
                description="Robust margin quality adjusted by sales valuation.",
                required_columns=["net_margin_ttm", "ps_ttm"],
                window=1,
                complexity=4,
                family="fundamental_value",
                compute=lambda df: op.cs_robust_zscore(df, "net_margin_ttm") - op.cs_robust_zscore(df, "ps_ttm"),
            ),
            AlphaCandidate(
                name="fcf_cash_conversion",
                expression="rank(free_cashflow_ttm / n_cashflow_act_ttm) + rank(cashflow_to_profit)",
                description="Free cash flow conversion backed by operating cash-flow coverage of profit.",
                required_columns=["free_cashflow_ttm", "n_cashflow_act_ttm", "cashflow_to_profit"],
                window=1,
                complexity=5,
                family="fundamental_quality",
                compute=lambda df: op.cs_rank(
                    _with_temp(df, _fcf_ocf=_fund_ratio(df, "free_cashflow_ttm", "n_cashflow_act_ttm")),
                    "_fcf_ocf",
                )
                + _rank(df, "cashflow_to_profit"),
            ),
            AlphaCandidate(
                name="liquidity_solvency_value",
                expression="rank(cash_to_liab) + rank(current_ratio) - rank(pb)",
                description="Cash and current-asset solvency bought at lower book valuation.",
                required_columns=["cash_to_liab", "current_ratio", "pb"],
                window=1,
                complexity=5,
                family="fundamental_safety",
                compute=lambda df: _rank(df, "cash_to_liab") + _rank(df, "current_ratio") - _rank(df, "pb"),
            ),
            AlphaCandidate(
                name="dupont_value_quality",
                expression="rank(net_margin_ttm) + rank(asset_turnover_ttm) - rank(pb)",
                description="DuPont operating quality adjusted by book valuation.",
                required_columns=["net_margin_ttm", "asset_turnover_ttm", "pb"],
                window=1,
                complexity=5,
                family="fundamental_value",
                compute=lambda df: _rank(df, "net_margin_ttm") + _rank(df, "asset_turnover_ttm") - _rank(df, "pb"),
            ),
            AlphaCandidate(
                name="inventory_receivable_light",
                expression="-rank(inventories / total_assets) - rank(accounts_receiv / total_assets) + rank(ar_turn)",
                description="Lower inventory and receivable occupation with stronger receivable turnover.",
                required_columns=["inventories", "accounts_receiv", "total_assets", "ar_turn"],
                window=1,
                complexity=6,
                family="fundamental_efficiency",
                compute=lambda df: -op.cs_rank(
                    _with_temp(df, _inventory_assets=_fund_ratio(df, "inventories", "total_assets")),
                    "_inventory_assets",
                )
                - op.cs_rank(
                    _with_temp(df, _receivable_assets=_fund_ratio(df, "accounts_receiv", "total_assets")),
                    "_receivable_assets",
                )
                + _rank(df, "ar_turn"),
            ),
            AlphaCandidate(
                name="industry_neutral_roe_value_pb",
                expression="industry_neutralize(rank(roe_ttm) - rank(pb))",
                description="Quality-value score after removing same-industry average exposure.",
                required_columns=["roe_ttm", "pb", "industry"],
                window=1,
                complexity=6,
                family="fundamental_value",
                compute=lambda df: op.industry_neutralize(
                    _with_temp(df, _roe_value=_rank(df, "roe_ttm") - _rank(df, "pb")),
                    "_roe_value",
                ),
            ),
            AlphaCandidate(
                name="industry_rank_cash_profit_cover",
                expression="group_rank(n_cashflow_act_ttm / net_profit_ttm, industry)",
                description="Profit cash coverage ranked within each industry.",
                required_columns=["n_cashflow_act_ttm", "net_profit_ttm", "industry"],
                window=1,
                complexity=5,
                family="fundamental_quality",
                compute=lambda df: op.group_rank(
                    _with_temp(df, _cash_profit_cover=_fund_ratio(df, "n_cashflow_act_ttm", "net_profit_ttm")),
                    "_cash_profit_cover",
                ),
            ),
            AlphaCandidate(
                name="size_neutral_earnings_yield_quality",
                expression="size_neutralize(rank(1 / pe_ttm) + rank(cashflow_to_profit), log_mv)",
                description="Cheap earnings yield and cash-flow quality after removing market-cap exposure.",
                required_columns=["pe_ttm", "cashflow_to_profit", "log_mv"],
                window=1,
                complexity=7,
                family="fundamental_value",
                compute=lambda df: op.size_neutralize(
                    _with_temp(
                        df,
                        _earnings_quality=op.cs_rank(
                            _with_temp(df, _earnings_yield=op.safe_div(pd.Series(1.0, index=df.index), df["pe_ttm"])),
                            "_earnings_yield",
                        )
                        + _rank(df, "cashflow_to_profit"),
                    ),
                    "_earnings_quality",
                ),
            ),
        ]
    )

    for window in windows:
        candidates.append(
            AlphaCandidate(
                name=f"quality_liquidity_confirm_{window}",
                expression=f"zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,{window})))",
                description="Cash-flow quality confirmed by trading amount expansion.",
                required_columns=["operating_cf_margin_ttm", "amount"],
                window=window,
                complexity=6,
                family="mixed",
                compute=lambda df, w=window: op.cs_zscore(
                    _with_temp(
                        df,
                        _mixed=(
                            op.cs_rank(df, "operating_cf_margin_ttm")
                            + op.cs_rank(
                                _with_temp(df, _amount_expansion=op.safe_div(df["amount"], op.ts_mean(df, "amount", w))),
                                "_amount_expansion",
                            )
                        ),
                    ),
                    "_mixed",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"decayed_quality_growth_{window}",
                expression=f"ts_decay_linear(rank(revenue_yoy) + rank(net_profit_yoy) + rank(cashflow_to_profit),{window})",
                description="Recent quality-growth signal with linear decay toward the latest observations.",
                required_columns=["revenue_yoy", "net_profit_yoy", "cashflow_to_profit"],
                window=window,
                complexity=7,
                family="fundamental_growth",
                compute=lambda df, w=window: op.ts_decay_linear(
                    _with_temp(
                        df,
                        _quality_growth=_rank(df, "revenue_yoy")
                        + _rank(df, "net_profit_yoy")
                        + _rank(df, "cashflow_to_profit"),
                    ),
                    "_quality_growth",
                    w,
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"stable_cash_conversion_{window}",
                expression=f"zscore(cashflow_to_profit) - zscore(ts_coef_var(cashflow_to_profit,{window}))",
                description="High and stable cash conversion; penalizes volatile cash-profit coverage.",
                required_columns=["cashflow_to_profit"],
                window=window,
                complexity=7,
                family="fundamental_quality",
                compute=lambda df, w=window: op.cs_zscore(df, "cashflow_to_profit")
                - op.cs_zscore(
                    _with_temp(df, _cash_cv=op.ts_coef_var(df, "cashflow_to_profit", w)),
                    "_cash_cv",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"margin_trend_value_{window}",
                expression=f"zscore(ts_zscore(net_margin_ttm,{window})) - zscore(ps_ttm)",
                description="Improving net-margin trend adjusted by sales valuation.",
                required_columns=["net_margin_ttm", "ps_ttm"],
                window=window,
                complexity=7,
                family="fundamental_value",
                compute=lambda df, w=window: op.cs_zscore(
                    _with_temp(df, _margin_trend=op.ts_zscore(df, "net_margin_ttm", w)),
                    "_margin_trend",
                )
                - op.cs_zscore(df, "ps_ttm"),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"cash_value_attention_gap_{window}",
                expression=f"robust_zscore(rank(cash_to_liab) - rank(pb)) - robust_zscore(turnover_rate / ts_mean(turnover_rate,{window}))",
                description="Cash-buffer value score with a penalty for short-term attention crowding.",
                required_columns=["cash_to_liab", "pb", "turnover_rate"],
                window=window,
                complexity=7,
                family="mixed",
                compute=lambda df, w=window: op.cs_robust_zscore(
                    _with_temp(df, _cash_value=_rank(df, "cash_to_liab") - _rank(df, "pb")),
                    "_cash_value",
                )
                - op.cs_robust_zscore(
                    _with_temp(
                        df,
                        _turnover_expansion=op.safe_div(df["turnover_rate"], op.ts_mean(df, "turnover_rate", w)),
                    ),
                    "_turnover_expansion",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"rd_neglect_{window}",
                expression=f"zscore(rd_expense_intensity) - zscore(amount / ts_mean(amount,{window}))",
                description="High R&D intensity with low market attention.",
                required_columns=["rd_expense_intensity", "amount"],
                window=window,
                complexity=5,
                family="mixed",
                compute=lambda df, w=window: op.cs_zscore(df, "rd_expense_intensity") - op.cs_zscore(
                    _with_temp(df, _amount_expansion=op.safe_div(df["amount"], op.ts_mean(df, "amount", w))),
                    "_amount_expansion",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"value_attention_gap_{window}",
                expression=f"zscore(rank(1 / pe_ttm) + rank(roe_ttm)) - zscore(amount / ts_mean(amount,{window}))",
                description="Cheap quality names that have not yet attracted heavy trading attention.",
                required_columns=["pe_ttm", "roe_ttm", "amount"],
                window=window,
                complexity=6,
                family="mixed",
                compute=lambda df, w=window: op.cs_zscore(
                    _with_temp(
                        df,
                        _value_quality=op.cs_rank(
                            _with_temp(df, _earnings_yield=op.safe_div(pd.Series(1.0, index=df.index), df["pe_ttm"])),
                            "_earnings_yield",
                        )
                        + _rank(df, "roe_ttm"),
                    ),
                    "_value_quality",
                )
                - op.cs_zscore(
                    _with_temp(df, _amount_expansion=op.safe_div(df["amount"], op.ts_mean(df, "amount", w))),
                    "_amount_expansion",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"growth_turnover_confirm_{window}",
                expression=f"zscore(rank(revenue_yoy) + rank(net_profit_yoy)) + zscore(turnover_rate / ts_mean(turnover_rate,{window}))",
                description="Fundamental growth confirmed by rising turnover participation.",
                required_columns=["revenue_yoy", "net_profit_yoy", "turnover_rate"],
                window=window,
                complexity=6,
                family="mixed",
                compute=lambda df, w=window: op.cs_zscore(
                    _with_temp(df, _growth=_rank(df, "revenue_yoy") + _rank(df, "net_profit_yoy")),
                    "_growth",
                )
                + op.cs_zscore(
                    _with_temp(df, _turnover_expansion=op.safe_div(df["turnover_rate"], op.ts_mean(df, "turnover_rate", w))),
                    "_turnover_expansion",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"quality_attention_gap_robust_{window}",
                expression=f"robust_zscore(rank(cashflow_to_profit) + rank(roe_ttm)) - robust_zscore(amount / ts_mean(amount,{window}))",
                description="Robust quality score minus short-term attention crowding.",
                required_columns=["cashflow_to_profit", "roe_ttm", "amount"],
                window=window,
                complexity=7,
                family="mixed",
                compute=lambda df, w=window: op.cs_robust_zscore(
                    _with_temp(df, _quality=_rank(df, "cashflow_to_profit") + _rank(df, "roe_ttm")),
                    "_quality",
                )
                - op.cs_robust_zscore(
                    _with_temp(df, _amount_expansion=op.safe_div(df["amount"], op.ts_mean(df, "amount", w))),
                    "_amount_expansion",
                ),
            )
        )
        candidates.append(
            AlphaCandidate(
                name=f"fcf_turnover_confirm_{window}",
                expression=f"zscore(rank(free_cashflow_ttm / total_mv)) + zscore(turnover_rate / ts_mean(turnover_rate,{window}))",
                description="Free-cash-flow yield confirmed by rising turnover participation.",
                required_columns=["free_cashflow_ttm", "total_mv", "turnover_rate"],
                window=window,
                complexity=7,
                family="mixed",
                compute=lambda df, w=window: op.cs_zscore(
                    _with_temp(
                        df,
                        _fcf_yield=op.cs_rank(
                            _with_temp(df, _raw_fcf_yield=_fund_ratio(df, "free_cashflow_ttm", "total_mv")),
                            "_raw_fcf_yield",
                        ),
                    ),
                    "_fcf_yield",
                )
                + op.cs_zscore(
                    _with_temp(df, _turnover_expansion=op.safe_div(df["turnover_rate"], op.ts_mean(df, "turnover_rate", w))),
                    "_turnover_expansion",
                ),
            )
        )
    return candidates


def available_candidates(
    df: pd.DataFrame,
    include_fundamental: bool = True,
    factor_set: str = "all",
    windows: Iterable[int] = (5, 20),
) -> List[AlphaCandidate]:
    if factor_set not in {"all", "daily", "fundamental"}:
        raise ValueError("factor_set must be one of: all, daily, fundamental")

    candidates: List[AlphaCandidate] = []
    if factor_set in {"all", "daily"}:
        candidates.extend(default_daily_alpha_candidates(windows=windows))
    if include_fundamental and factor_set in {"all", "fundamental"}:
        candidates.extend(optional_fundamental_candidates(windows=(20,)))
    return [candidate for candidate in candidates if candidate.is_available(df)]
