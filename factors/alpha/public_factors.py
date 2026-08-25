"""
Locally reproducible versions of the Rongliang public alpha factors.

The source file contains public expressions published in 2026.  This module
does not trust their reported IC/Sharpe values.  It only provides local direct
or proxy implementations that can be re-tested on the CSI500 panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from factors.alpha import operators as op
from factors.alpha.external_alpha import (
    AlphaAvailability,
    AlphaValidationStatus,
    ExternalAlphaMetadata,
    metadata_frame,
)


ComputeFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class PublicAlphaSpec:
    metadata: ExternalAlphaMetadata
    compute: ComputeFunc | None = None

    @property
    def name(self) -> str:
        return self.metadata.factor_name

    @property
    def required_columns(self) -> Sequence[str]:
        return self.metadata.required_columns

    def is_available(self, df: pd.DataFrame) -> bool:
        return set(self.required_columns).issubset(df.columns)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        missing = sorted(set(self.required_columns) - set(df.columns))
        if missing:
            raise KeyError(f"{self.name} missing required columns: {missing}")
        if self.compute is None:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return self.compute(df).replace([np.inf, -np.inf], np.nan)


def _meta(
    source_factor_id: str,
    factor_name: str,
    expression: str,
    description: str,
    required_columns: Sequence[str],
    availability: AlphaAvailability,
    source: str = "rongliang_public",
    version: str = "2026_ytd_extract",
    release_date: str = "2026-07-08",
    local_expression: str = "",
    proxy_reason: str = "",
    skip_reason: str = "",
    missing_columns: Sequence[str] = (),
) -> ExternalAlphaMetadata:
    status = AlphaValidationStatus.PENDING
    if availability == AlphaAvailability.SKIPPED:
        status = AlphaValidationStatus.SKIPPED
    return ExternalAlphaMetadata(
        source=source,
        source_factor_id=source_factor_id,
        factor_name=factor_name,
        version=version,
        release_date=release_date,
        expression=expression,
        local_expression=local_expression,
        description=description,
        required_columns=tuple(required_columns),
        missing_columns=tuple(missing_columns),
        availability=availability,
        validation_status=status,
        proxy_reason=proxy_reason,
        skip_reason=skip_reason,
        used_in_model=False,
    )


def _base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_ret_1d" not in out:
        out["_ret_1d"] = op.ts_pct(out, "close_adj", 1)
    if "_ret_5d" not in out:
        out["_ret_5d"] = op.ts_pct(out, "close_adj", 5)
    if "_vwap" not in out:
        out["_vwap"] = op.safe_div(out["amount"], out["volume"])
    if "_range" not in out:
        out["_range"] = op.safe_div(out["high_adj"], out["low_adj"]) - 1.0
    if "_low_shadow" not in out:
        out["_low_shadow"] = op.safe_div(out["close_adj"] - out["low_adj"], out["high_adj"] - out["low_adj"])
    return out


def _cs_rank_series(df: pd.DataFrame, values: pd.Series, name: str) -> pd.Series:
    tmp = df[["trade_date"]].copy()
    tmp[name] = values
    return op.cs_rank(tmp, name)


def _cs_z_series(df: pd.DataFrame, values: pd.Series, name: str) -> pd.Series:
    tmp = df[["trade_date"]].copy()
    tmp[name] = values
    return op.cs_zscore(tmp, name)


def _group_ewm(df: pd.DataFrame, col: str, span: int) -> pd.Series:
    return df.groupby("ts_code", sort=False)[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())


def _ts_ir(df: pd.DataFrame, col: str, window: int, min_periods: int | None = None) -> pd.Series:
    return op.safe_div(op.ts_mean(df, col, window, min_periods=min_periods), op.ts_std(df, col, window, min_periods=min_periods))


def _cs_pct_rank(df: pd.DataFrame, values: pd.Series, name: str) -> pd.Series:
    return _cs_rank_series(df, values, name)


def _load_moneyflow_features(
    raw_dir: Path,
    start: str,
    end: str,
    ts_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = raw_dir / f"moneyflow_{start}_{end}.parquet"
    if not path.exists():
        return pd.DataFrame()
    money = pd.read_parquet(path)
    if money.empty:
        return pd.DataFrame()
    money["trade_date"] = pd.to_datetime(money["trade_date"])
    if ts_codes is not None:
        codes = set(ts_codes)
        money = money[money["ts_code"].isin(codes)]
    for col in ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]:
        if col not in money:
            money[col] = np.nan
        money[col] = pd.to_numeric(money[col], errors="coerce")
    money = money.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    large_net = money["buy_lg_amount"] - money["sell_lg_amount"]
    extra_large_net = money["buy_elg_amount"] - money["sell_elg_amount"]
    money["MAIN_IN_FLOW_V2"] = large_net + extra_large_net
    money["SLARGE_IN_FLOW_V2"] = extra_large_net
    money["LARGE_OUT_FLOW_V2"] = money["sell_lg_amount"] + money["sell_elg_amount"]
    money["NET_MF_AMOUNT_V2"] = money["net_mf_amount"].fillna(money["MAIN_IN_FLOW_V2"])
    money["MAIN_IN_FLOW_20D_V2"] = op.ts_sum(money, "MAIN_IN_FLOW_V2", 20, min_periods=10)
    money["_main_inflow_day"] = (money["MAIN_IN_FLOW_V2"] > 0).astype(float)
    money["_slarge_inflow_day"] = (money["SLARGE_IN_FLOW_V2"] > 0).astype(float)
    money["MAIN_IN_FLOW_DAYS_10D_V2"] = op.ts_sum(money, "_main_inflow_day", 10, min_periods=5)
    money["MAIN_IN_FLOW_DAYS_20D_V2"] = op.ts_sum(money, "_main_inflow_day", 20, min_periods=10)
    money["CON_FUND_DAY_IN_10D_V2"] = op.ts_sum(money, "_slarge_inflow_day", 10, min_periods=5)
    cols = [
        "trade_date",
        "ts_code",
        "MAIN_IN_FLOW_V2",
        "MAIN_IN_FLOW_20D_V2",
        "MAIN_IN_FLOW_DAYS_10D_V2",
        "MAIN_IN_FLOW_DAYS_20D_V2",
        "SLARGE_IN_FLOW_V2",
        "LARGE_OUT_FLOW_V2",
        "NET_MF_AMOUNT_V2",
        "CON_FUND_DAY_IN_10D_V2",
    ]
    return money[cols]


def augment_public_factor_fields(
    panel: pd.DataFrame,
    raw_dir: Path | None = None,
    start: str = "20180101",
    end: str = "20260706",
) -> pd.DataFrame:
    """Add auditable local field mappings used only by the 65 public factors."""
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    alias_map = {
        "AF_CLOSE": "close_adj",
        "AF_HIGH": "high_adj",
        "AF_LOW": "low_adj",
        "AF_OPEN": "open_adj",
    }
    for alias, source in alias_map.items():
        if alias not in out and source in out:
            out[alias] = out[source]
    if "AF_VWAP" not in out and {"amount", "volume"}.issubset(out.columns):
        out["AF_VWAP"] = op.safe_div(out["amount"], out["volume"])
    if "REINSTATEMENT_CHG_60D" not in out and "close_adj" in out:
        out["REINSTATEMENT_CHG_60D"] = op.ts_pct(out, "close_adj", 60)
    if "FACTOR_CNE5_SIZE" not in out:
        if "log_mv" in out:
            out["FACTOR_CNE5_SIZE"] = _cs_z_series(out, out["log_mv"], "_local_size_proxy")
        elif "total_mv" in out:
            out["FACTOR_CNE5_SIZE"] = _cs_z_series(out, np.log(out["total_mv"].where(out["total_mv"] > 0)), "_local_size_proxy")
    if "FACTOR_CNE5_BETA" not in out and "close_adj" in out:
        out["_local_ret1_for_beta"] = op.ts_pct(out, "close_adj", 1)
        vol_proxy = 0.5 * op.ts_std(out, "_local_ret1_for_beta", 20, min_periods=10) + 0.5 * op.ts_std(
            out, "_local_ret1_for_beta", 60, min_periods=30
        )
        out["FACTOR_CNE5_BETA"] = _cs_z_series(out, vol_proxy, "_local_beta_proxy")
        out = out.drop(columns=["_local_ret1_for_beta"])
    if raw_dir is not None:
        money = _load_moneyflow_features(raw_dir, start, end, out["ts_code"].dropna().astype(str).unique())
        if not money.empty:
            out = out.merge(money, on=["trade_date", "ts_code"], how="left")
    return out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _rolling_sum_max(df: pd.DataFrame, col: str, outer: int, inner: int) -> pd.Series:
    tmp = df.copy()
    tmp["_inner_sum"] = op.ts_sum(tmp, col, inner)
    return op.ts_max(tmp, "_inner_sum", outer)


def _mfi_proxy(df: pd.DataFrame, window: int = 21) -> pd.Series:
    tmp = _base(df)
    typical = (tmp["high_adj"] + tmp["low_adj"] + tmp["close_adj"]) / 3.0
    raw_flow = typical * tmp["volume"]
    direction = np.sign(typical - op.ts_delay(tmp.assign(_typical=typical), "_typical", 1)).fillna(0.0)
    mf = raw_flow * direction
    tmp["_pos_mf"] = mf.clip(lower=0)
    tmp["_neg_mf"] = (-mf.clip(upper=0)).abs()
    ratio = op.safe_div(op.ts_sum(tmp, "_pos_mf", window), op.ts_sum(tmp, "_neg_mf", window))
    return 100.0 - 100.0 / (1.0 + ratio)


def _factor_01_tail_risk(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_kurt5"] = op.ts_kurt(tmp, "close_adj", 5)
    tmp["_vol_skew5"] = op.ts_skew(tmp, "volume", 5)
    tmp["_vol_skew20"] = op.ts_skew(tmp, "volume", 20)
    component = (
        op.signed_log1p(tmp["_close_kurt5"])
        + op.ts_max(tmp, "_vol_skew5", 3)
        - op.ts_min(tmp, "_vol_skew20", 3)
        + _rolling_sum_max(tmp, "_ret_1d", 20, 5)
    )
    return -_cs_z_series(tmp, component, "_factor_01")


def _factor_02_projection_support_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_ret5_rank"] = _cs_rank_series(tmp, tmp["_ret_5d"], "_ret5")
    tmp["_vol_mean15_rank"] = _cs_rank_series(tmp, op.ts_mean(tmp, "volume", 15), "_vol_mean15")
    tmp["_proj_resid"] = op.cs_regression_resid(tmp, "_ret5_rank", ["_vol_mean15_rank"])
    trend = op.ts_mean(tmp, "_ret_1d", 20)
    support = _cs_rank_series(tmp, op.ts_mean(tmp.assign(_support=tmp["_low_shadow"]), "_support", 5), "_support5")
    money_leg = pd.Series(0.0, index=tmp.index)
    if {"MAIN_IN_FLOW_20D_V2", "total_mv"}.issubset(tmp.columns):
        tmp["_mainflow_to_mv"] = op.safe_div(tmp["MAIN_IN_FLOW_20D_V2"], tmp["total_mv"])
        money_leg = op.ts_sum(tmp, "_mainflow_to_mv", 10, min_periods=5) * support
    return -(tmp["_proj_resid"] * trend + money_leg)


def _factor_03_pvt_covariance(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_pvt1d"] = tmp["_ret_1d"] * tmp["volume"]
    cov = op.ts_covariance(tmp, "_ret_1d", "_pvt1d", 20)
    return -_cs_z_series(tmp, op.signed_log1p(cov), "_pvt_cov")


def _factor_04_valuation_price_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_1y_dev"] = op.safe_div(tmp["close_adj"], op.ts_mean(tmp, "close_adj", 250, min_periods=120)) - 1.0
    if "pb" in tmp:
        tmp["_neg_pb"] = -tmp["pb"]
        tmp["_industry_bm"] = op.industry_neutralize(tmp, "_neg_pb", industry_col="industry")
    else:
        tmp["_industry_bm"] = np.nan
    corr = op.ts_corr(tmp, "_close_1y_dev", "_industry_bm", 90, min_periods=45)
    tmp["_vwap_volume"] = tmp["_vwap"] + _cs_z_series(tmp, tmp["volume"], "_volume_z")
    resid = op.cs_regression_resid(tmp, "close_adj", ["_vwap_volume"])
    tmp["_ret_mean5"] = op.ts_mean(tmp, "_ret_1d", 5)
    momentum = op.ts_max(tmp, "_ret_mean5", 25)
    inv_value = op.safe_div(pd.Series(1.0, index=tmp.index), tmp["pe_ttm"]) if "pe_ttm" in tmp else 0.0
    return corr * resid - momentum + inv_value


def _factor_05_ema_midprice_divergence(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_ema20"] = _group_ewm(tmp, "close_adj", 20)
    state = (tmp.groupby("ts_code", sort=False)["_ema20"].pct_change().abs() < 0.02).astype(float)
    tmp["_state"] = state
    tmp["_midprice"] = (tmp["high_adj"] + tmp["low_adj"]) / 2.0
    tmp["_mid10"] = op.ts_mean(tmp, "_midprice", 10)
    value = op.ts_sum(tmp, "_state", 60) - op.ts_max(tmp, "_mid10", 30)
    return _cs_rank_series(tmp, value, "_ema_mid")


def _factor_06_vol_compression_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    price_range = op.safe_div(op.ts_max(tmp, "close_adj", 20) - op.ts_min(tmp, "close_adj", 20), op.ts_mean(tmp, "close_adj", 20))
    moment = _cs_rank_series(tmp, op.ts_skew(tmp, "close_adj", 30), "_moment30")
    return -(price_range * moment)


def _factor_07_log_volume_price_trend(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    delta5 = op.ts_delta(tmp, "close_adj", 5)
    tmp["_vp"] = delta5 * np.log10(1.0 + op.ts_sum(tmp, "volume", 10).clip(lower=0))
    summed = op.ts_sum(tmp, "_vp", 10)
    return -op.signed_log1p(summed)


def _factor_08_variance_ratio_divergence(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    gain = tmp["_ret_1d"].where(tmp["_ret_1d"] > 0)
    loss = tmp["_ret_1d"].where(tmp["_ret_1d"] < 0)
    tmp["_gain"] = gain
    tmp["_loss"] = loss
    gain_var120 = op.ts_var(tmp, "_gain", 120, min_periods=60)
    loss_var120 = op.ts_var(tmp, "_loss", 120, min_periods=60)
    gain_loss_ratio = op.safe_div(gain_var120, loss_var120)
    gain_var60 = op.ts_var(tmp, "_gain", 60, min_periods=30)
    return -_cs_rank_series(tmp, (gain_loss_ratio - gain_var60).abs(), "_variance_ratio")


def _factor_09_filtered_momentum_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    earnings_yield = op.safe_div(pd.Series(1.0, index=tmp.index), tmp["pe_ttm"]) if "pe_ttm" in tmp else 0.0
    value = op.ts_delta(tmp, "close_adj", 5) + op.signed_log1p(tmp["volume"] + earnings_yield)
    out = -_cs_rank_series(tmp, value, "_filtered_mom")
    if "pe_ttm" in tmp:
        out = out.where(tmp["pe_ttm"] > 0)
    return out


def _factor_10_cashflow_price_trend_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    mfi_rank = _cs_rank_series(tmp, _mfi_proxy(tmp), "_mfi21")
    cash_sales = tmp["operating_cf_margin_ttm"] if "operating_cf_margin_ttm" in tmp else tmp.get("ocf_to_or", np.nan)
    tmp["_range_std"] = op.ts_std(tmp, "close_adj", 60, min_periods=30)
    return _cs_rank_series(tmp, mfi_rank * (cash_sales - tmp["_range_std"]), "_cash_price")


def _factor_11_industry_ma_valuation_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    ma3 = op.ts_mean(tmp, "close_adj", 3, min_periods=2)
    ma6 = op.ts_mean(tmp, "close_adj", 6, min_periods=3)
    ma12 = op.ts_mean(tmp, "close_adj", 12, min_periods=6)
    ma24 = op.ts_mean(tmp, "close_adj", 24, min_periods=12)
    tmp["_bbi"] = (ma3 + ma6 + ma12 + ma24) / 4.0
    tmp["_ema60"] = _group_ewm(tmp, "close_adj", 60)
    tmp["_bbi_ema_spread"] = tmp["_bbi"] - tmp["_ema60"]
    group_std = tmp.groupby(["trade_date", "industry"], sort=False)["_bbi_ema_spread"].transform("std")
    tmp["_chg_3m_avg"] = op.safe_div(tmp["close_adj"], op.ts_mean(tmp, "close_adj", 60, min_periods=30)) - 1.0
    tmp["_chg_3m_winsor"] = op.cs_winsorize(tmp, "_chg_3m_avg", 0.01, 0.99)
    valuation = np.log1p(tmp["pe_ttm"].where(tmp["pe_ttm"] > 0)) if "pe_ttm" in tmp else 0.0
    value = group_std.fillna(0.0) * tmp["_chg_3m_winsor"] + valuation
    return -_cs_rank_series(tmp, value, "_industry_ma_valuation_11")


def _factor_12_industry_mainflow_quality_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_mainflow_group_rank"] = op.group_rank(tmp, "MAIN_IN_FLOW_20D_V2")
    tmp["_roe_change"] = op.ts_delta(tmp, "roe_ttm", 20) if "roe_ttm" in tmp else np.nan
    tmp["_quality_proxy"] = tmp["roe_ttm"] - op.ts_min(tmp.assign(_roe_change=tmp["_roe_change"]), "_roe_change", 250, min_periods=60)
    tmp["_quality_mid"] = tmp.groupby("trade_date", sort=False)["_quality_proxy"].transform("median")
    return -_cs_rank_series(tmp, tmp["_mainflow_group_rank"] - tmp["_quality_mid"], "_industry_flow_quality_12")


def _factor_13_fundflow_cross_section_pct(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_high_diff10"] = op.ts_delta(tmp, "AF_HIGH", 10)
    tmp["_vwap_high_mix"] = tmp["AF_VWAP"] + tmp["_high_diff10"]
    price_leg = op.ts_delta(tmp, "_vwap_high_mix", 15)
    flow_strength = tmp["MAIN_IN_FLOW_DAYS_20D_V2"] + tmp["CON_FUND_DAY_IN_10D_V2"]
    flow_rank = _cs_pct_rank(tmp, flow_strength, "_flow_days_rank_13")
    value = np.round(price_leg * flow_rank.where(flow_rank >= 0.8, 0.0))
    return -_cs_rank_series(tmp, value, "_fundflow_pct_13")


def _factor_14_large_outflow_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_large_outflow_moment2"] = op.ts_var(tmp, "LARGE_OUT_FLOW_V2", 60, min_periods=30)
    return -(tmp["turnover_rate"] + tmp["_large_outflow_moment2"])


def _factor_15_nonlinear_price_volume_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    ret_sin = np.sin(op.ts_mean(tmp, "_ret_1d", 5))
    if "MAIN_IN_FLOW_DAYS_10D_V2" in tmp:
        cs_vwap_skew = tmp.groupby("trade_date", sort=False)["_vwap"].transform("skew").fillna(0.0)
        flow_median = op.ts_median(tmp, "MAIN_IN_FLOW_DAYS_10D_V2", 20, min_periods=10).fillna(0.0)
        nonlinear_flow = ret_sin * cs_vwap_skew * flow_median
    else:
        vwap_rank = _cs_rank_series(tmp, tmp["_vwap"], "_vwap_rank") - 0.5
        nonlinear_flow = ret_sin * vwap_rank
    range_vol = op.signed_log1p(op.ts_max(tmp.assign(_hl=tmp["high_adj"] - tmp["low_adj"]), "_hl", 60, min_periods=30))
    vol_ratio = op.safe_div(op.ts_max(tmp, "volume", 20, min_periods=10), op.ts_mean(tmp, "volume", 20, min_periods=10))
    return -(nonlinear_flow + range_vol * vol_ratio)


def _factor_16_decay_volume_trend(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_decay_ret20"] = op.ts_decay_linear(tmp, "_ret_1d", 20, min_periods=10)
    value = op.ts_max(tmp.assign(_decay_vol=tmp["_decay_ret20"] * np.log1p(tmp["volume"].clip(lower=0))), "_decay_vol", 3)
    return -_cs_z_series(tmp, value, "_decay_vol")


def _factor_17_vol_trend_structure(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    ema10 = _group_ewm(tmp, "close_adj", 10)
    ma5 = op.ts_mean(tmp, "close_adj", 5)
    trend = np.round(op.safe_div(ema10, ma5))
    vol120 = op.ts_std(tmp, "_ret_1d", 120, min_periods=60)
    vol20 = op.ts_std(tmp, "_ret_1d", 20, min_periods=10)
    sharpe120 = op.safe_div(op.ts_mean(tmp, "_ret_1d", 120, min_periods=60), vol120)
    sharpe20 = op.safe_div(op.ts_mean(tmp, "_ret_1d", 20, min_periods=10), vol20)
    return trend * (vol120 - vol20) + (sharpe120 - sharpe20)


def _factor_18_turnover_adjusted_price(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_inv_turnover"] = -op.safe_div(pd.Series(1.0, index=tmp.index), tmp["turnover_rate"])
    resid = op.cs_regression_resid(tmp, "close_adj", ["_inv_turnover"])
    return -_cs_rank_series(tmp, op.signed_log1p(resid), "_turnover_resid")


def _factor_19_poly_volume_price_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    slope = op.ts_slope(tmp, "close_adj", "volume", 30, min_periods=20)
    mean_close = op.ts_mean(tmp, "close_adj", 30, min_periods=20)
    mean_volume = op.ts_mean(tmp, "volume", 30, min_periods=20)
    tmp["_poly_resid"] = tmp["close_adj"] - (mean_close + slope * (tmp["volume"] - mean_volume))
    tmp["_kurt20"] = op.ts_kurt(tmp, "_ret_1d", 20, min_periods=10)
    return -((tmp["_poly_resid"] + tmp["_kurt20"]) * np.sqrt(tmp["volume"].abs()))


def _factor_20_price_exhaustion(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    rank20 = op.ts_rank(tmp, "close_adj", 20)
    high_sum_max = _rolling_sum_max(tmp, "high_adj", 20, 5)
    median20 = op.ts_quantile(tmp, "close_adj", 20, 0.5)
    av_diff = tmp["close_adj"] - op.ts_mean(tmp, "close_adj", 20)
    return -(rank20 * ((high_sum_max - median20) + av_diff))


def _factor_21_log_momentum_reversal_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    roc = op.ts_pct(tmp, "close_adj", 60)
    value = op.signed_log1p(op.ts_pct(tmp, "close_adj", 15) + roc)
    return -_cs_rank_series(tmp, _cs_z_series(tmp, value, "_log_mom"), "_log_mom_rank")


def _factor_22_volume_divergence_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_mean15_rank"] = _cs_rank_series(tmp, op.ts_mean(tmp, "close_adj", 15), "_close_mean15")
    tmp["_volume_mean15_rank"] = _cs_rank_series(tmp, op.ts_mean(tmp, "volume", 15), "_volume_mean15")
    corr = op.ts_corr(tmp, "_close_mean15_rank", "_volume_mean15_rank", 10)
    ret_rank = _cs_rank_series(tmp, op.ts_mean(tmp, "_ret_1d", 15), "_ret_mean15")
    turn_rank = _cs_rank_series(tmp, op.ts_mean(tmp, "turnover_rate", 15), "_turn_mean15")
    vol_rank = _cs_rank_series(tmp, op.ts_mean(tmp, "volume", 15), "_vol_mean15")
    return -_cs_rank_series(tmp, corr, "_corr_rank") * ret_rank * turn_rank * vol_rank


def _factor_23_turnover_relative_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return -op.safe_div(op.ts_mean(tmp, "turnover_rate", 20), op.ts_mean(tmp, "turnover_rate", 120, min_periods=60))


def _factor_24_price_reversal_30d(df: pd.DataFrame) -> pd.Series:
    return -op.ts_pct(df, "close_adj", 30)


def _factor_25_high_open_decay(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_high_open"] = tmp["high_adj"] - tmp["open_adj"]
    return -op.ts_wma(tmp, "_high_open", 5)


def _factor_26_moneyflow_drawdown(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return op.ts_max_drawdown(tmp, "NET_MF_AMOUNT_V2", 15, min_periods=8)


def _pack2_meta(
    source_factor_id: str,
    factor_name: str,
    expression: str,
    description: str,
    required_columns: Sequence[str],
    availability: AlphaAvailability,
    local_expression: str = "",
    proxy_reason: str = "",
    skip_reason: str = "",
    missing_columns: Sequence[str] = (),
) -> ExternalAlphaMetadata:
    return _meta(
        source_factor_id=source_factor_id,
        factor_name=factor_name,
        expression=expression,
        description=description,
        required_columns=required_columns,
        availability=availability,
        source="rongliang_public_pack2",
        version="2026_pack2_extract",
        release_date="2026-07-09",
        local_expression=local_expression,
        proxy_reason=proxy_reason,
        skip_reason=skip_reason,
        missing_columns=missing_columns,
    )


def _factor_vroc12_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return op.ts_pct(tmp.assign(_volume=tmp["volume"].where(tmp["volume"] > 0)), "_volume", 12)


def _factor_tvsd20_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_volume_resid"] = op.cs_regression_resid(tmp, "close_adj", ["volume"])
    return op.ts_std(tmp, "_close_volume_resid", 20, min_periods=10)


def _factor_vol60_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_log_volume"] = np.log1p(tmp["volume"].clip(lower=0))
    return op.ts_std(tmp, "_log_volume", 60, min_periods=30)


def _factor_27_reverse_price_volume_rank(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return -_cs_rank_series(tmp, tmp["close_adj"], "_close_rank_27") * _cs_rank_series(tmp, tmp["volume"], "_volume_rank_27")


def _factor_28_volatility_spread_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vol60d"] = _factor_vol60_proxy(tmp)
    tmp["_factor_tvsd20d"] = _factor_tvsd20_proxy(tmp)
    left = op.ts_percentage(tmp, "_factor_vol60d", 5, min_periods=3)
    right = op.ts_percentage(tmp, "_factor_tvsd20d", 5, min_periods=3)
    return left - right


def _factor_29_volume_stable_close(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    vol_rank = _cs_rank_series(tmp, tmp["volume"], "_volume_rank_29")
    close_std_rank = _cs_rank_series(tmp, op.ts_std(tmp, "close_adj", 10, min_periods=5), "_close_std10_rank_29")
    return vol_rank * (1.0 - close_std_rank)


def _factor_30_short_term_vol_adjusted_return(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_open_spread"] = tmp["close_adj"] - tmp["open_adj"]
    spread_rank = _cs_rank_series(tmp, op.ts_sum(tmp, "_close_open_spread", 15, min_periods=8), "_spread_rank_30")
    vol_rank = _cs_rank_series(tmp, op.ts_std(tmp, "close_adj", 15, min_periods=8), "_close_std15_rank_30")
    return -(spread_rank * vol_rank)


def _factor_31_gap_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    gap = op.ts_mean(tmp, "open_adj", 10, min_periods=5) - op.ts_mean(tmp, "close_adj", 10, min_periods=5)
    momentum_rank = _cs_rank_series(tmp, op.ts_delta(tmp, "close_adj", 10), "_close_delta10_rank_31")
    return gap * momentum_rank


def _factor_32_reverse_volatility_covariance_proxy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vroc12d"] = _factor_vroc12_proxy(tmp)
    tmp["_close_rank_32"] = _cs_rank_series(tmp, tmp["close_adj"], "_close_rank_32_raw")
    tmp["_close_rank_std15"] = op.ts_std(tmp, "_close_rank_32", 15, min_periods=8)
    return -op.ts_covariance(tmp, "_factor_vroc12d", "_close_rank_std15", 20, min_periods=10)


def _factor_33_reinstatement_vol_ratio(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_tvsd20d"] = _factor_tvsd20_proxy(tmp)
    numerator = op.ts_std(tmp, "REINSTATEMENT_CHG_60D", 35, min_periods=18)
    denominator = op.ts_std(tmp, "_factor_tvsd20d", 35, min_periods=18)
    return op.safe_div(numerator, denominator)


def _factor_34_composite_momentum_flow(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vroc12d"] = _factor_vroc12_proxy(tmp)
    return op.ts_decay_linear(tmp, "_factor_vroc12d", 10, min_periods=5) + op.ts_decay_linear(
        tmp, "MAIN_IN_FLOW_20D_V2", 10, min_periods=5
    )


def _factor_35_flow_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    main_rank = _cs_rank_series(tmp, tmp["MAIN_IN_FLOW_20D_V2"], "_main_rank_35")
    slarge_rank = _cs_rank_series(tmp, tmp["SLARGE_IN_FLOW_V2"], "_slarge_rank_35")
    tmp["_flow_rank_spread"] = main_rank.abs() - slarge_rank.abs()
    return op.ts_decay_linear(tmp, "_flow_rank_spread", 15, min_periods=8)


def _factor_36_momentum_flow_volatility(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vol60d"] = _factor_vol60_proxy(tmp)
    return op.ts_decay_linear(tmp, "_factor_vol60d", 10, min_periods=5) * op.ts_percentage(
        tmp, "MAIN_IN_FLOW_20D_V2", 10, min_periods=5
    )


def _factor_37_flow_volatility_synergy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vol60d"] = _factor_vol60_proxy(tmp)
    return op.ts_decay_linear(tmp, "MAIN_IN_FLOW_20D_V2", 10, min_periods=5) * tmp["_factor_vol60d"]


def _factor_38_main_superlarge_flow_synergy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    main_pct = op.ts_percentage(tmp, "MAIN_IN_FLOW_20D_V2", 30, min_periods=15)
    slarge_decay = op.ts_decay_linear(tmp, "SLARGE_IN_FLOW_V2", 15, min_periods=8)
    return _cs_rank_series(tmp, main_pct * slarge_decay, "_flow_synergy_38")


def _factor_39_main_superlarge_flow_spread(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return op.ts_decay_linear(tmp, "MAIN_IN_FLOW_20D_V2", 5, min_periods=3) - op.ts_decay_linear(
        tmp, "SLARGE_IN_FLOW_V2", 5, min_periods=3
    )


def _factor_40_style_ir_spread(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return _ts_ir(tmp, "FACTOR_CNE5_BETA", 20, min_periods=10) - _ts_ir(tmp, "FACTOR_CNE5_SIZE", 20, min_periods=10)


def _factor_41_flow_volatility_factor(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_factor_vol60d"] = _factor_vol60_proxy(tmp)
    return op.ts_percentage(tmp, "MAIN_IN_FLOW_20D_V2", 10, min_periods=5) * tmp["_factor_vol60d"]


def _factor_42_flow_linear_decay(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_flow_spread_rank"] = _cs_rank_series(tmp, tmp["MAIN_IN_FLOW_20D_V2"] - tmp["SLARGE_IN_FLOW_V2"], "_flow_spread_42")
    return op.ts_decay_linear(tmp, "_flow_spread_rank", 15, min_periods=8)


def _factor_43_main_flow_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_decay_mainflow6"] = op.ts_decay_linear(tmp, "MAIN_IN_FLOW_20D_V2", 6, min_periods=3)
    return op.ts_percentage(tmp, "_decay_mainflow6", 3, min_periods=2)


def _factor_44_momentum_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_af_close_rank"] = _cs_rank_series(tmp, tmp["AF_CLOSE"], "_af_close_rank_44")
    tmp["_rank_delta"] = op.ts_delta(tmp, "_af_close_rank", 1)
    return -op.ts_sum(tmp, "_rank_delta", 40, min_periods=20)


def _factor_45_flow_momentum_reinstatement(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return tmp["REINSTATEMENT_CHG_60D"].abs() * op.ts_decay_linear(tmp, "MAIN_IN_FLOW_20D_V2", 15, min_periods=8)


def _factor_46_main_flow_linear_decay(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return op.ts_decay_linear(tmp, "MAIN_IN_FLOW_20D_V2", 10, min_periods=5)


def _factor_47_price_flow_volatility_coupling(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    price_rank = _cs_rank_series(tmp, op.safe_div(tmp["AF_CLOSE"], op.ts_delay(tmp, "AF_CLOSE", 15)), "_price_rank_47")
    tmp["_mainflow_std15"] = op.ts_std(tmp, "MAIN_IN_FLOW_V2", 15, min_periods=8)
    flow_vol_rank = _cs_rank_series(tmp, tmp["_mainflow_std15"], "_flow_vol_rank_47")
    return -(price_rank * flow_vol_rank)


def _factor_48_price_volume_decay_synergy(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_rank_48"] = _cs_rank_series(tmp, tmp["close_adj"], "_close_rank_48_raw")
    price_component = op.ts_percentage(tmp, "_close_rank_48", 10, min_periods=5)
    tmp["_factor_vroc12d"] = _factor_vroc12_proxy(tmp)
    volume_component = op.ts_decay_linear(tmp, "_factor_vroc12d", 60, min_periods=30)
    return -(price_component * volume_component)


def _factor_50_price_volume_volatility_inverse(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    close_std_rank = _cs_rank_series(tmp, op.ts_std(tmp, "close_adj", 10, min_periods=5), "_close_std10_rank_50")
    volume_std_rank = _cs_rank_series(tmp, op.ts_std(tmp, "volume", 10, min_periods=5), "_volume_std10_rank_50")
    return -(close_std_rank * volume_std_rank)


def _factor_51_multidim_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    price_rank = _cs_rank_series(
        tmp,
        op.safe_div(tmp["close_adj"], op.ts_mean(tmp, "close_adj", 20, min_periods=10)),
        "_price_strength_rank_51",
    )
    volume_rank = _cs_rank_series(tmp, op.ts_mean(tmp, "volume", 20, min_periods=10), "_volume_mean20_rank_51")
    turnover_rank = _cs_rank_series(tmp, op.ts_mean(tmp, "turnover_rate", 20, min_periods=10), "_turnover_mean20_rank_51")
    return -(price_rank * volume_rank * turnover_rank)


def _factor_49_dual_volatility_flow_inverse(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    close_std_rank = _cs_rank_series(tmp, op.ts_std(tmp, "close_adj", 10, min_periods=5), "_close_std10_rank_49")
    tmp["_mainflow_std10"] = op.ts_std(tmp, "MAIN_IN_FLOW_V2", 10, min_periods=5)
    flow_std_rank = _cs_rank_series(tmp, tmp["_mainflow_std10"], "_flow_std10_rank_49")
    return -(close_std_rank * flow_std_rank)


def _factor_52_main_flow_peak_inverse(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return -_cs_rank_series(tmp, op.ts_max(tmp, "MAIN_IN_FLOW_V2", 10, min_periods=5), "_mainflow_peak_rank_52")


def _factor_53_main_flow_stability(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return -op.ts_std(tmp, "MAIN_IN_FLOW_V2", 10, min_periods=5)


def _factor_54_turnover_volatility_inverse(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    return -op.ts_std(tmp, "turnover_rate", 15, min_periods=8)


def _factor_55_vol_turnover_coupling(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    vol_rank = _cs_rank_series(tmp, op.ts_std(tmp, "close_adj", 15, min_periods=8), "_close_std15_rank_55")
    scaled_rank = _cs_z_series(tmp, vol_rank, "_close_std15_scale_55")
    return scaled_rank * (-tmp["turnover_rate"])


def _factor_56_ma_deviation_volume_weighted(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    deviation = op.ts_mean(tmp, "close_adj", 20, min_periods=10) - tmp["close_adj"]
    volume_rank = _cs_rank_series(tmp, tmp["volume"], "_volume_rank_56")
    return deviation * volume_rank


def _factor_57_vol_adjusted_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_change_pct"] = tmp["_ret_1d"]
    tmp["_signed_sq_ret"] = op.signed_power(tmp["_change_pct"], 2)
    return -op.ts_mean(tmp, "_signed_sq_ret", 30, min_periods=15)


def _factor_58_liquidity_stability(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_amount_rank_58"] = _cs_rank_series(tmp, tmp["amount"], "_amount_rank_58_raw")
    return -op.ts_mean(tmp, "_amount_rank_58", 10, min_periods=5)


def _factor_59_price_vwap_volume_volatility(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_close_vwap_volume"] = (tmp["close_adj"] - tmp["_vwap"]) * tmp["volume"]
    return -op.ts_std(tmp, "_close_vwap_volume", 10, min_periods=5)


def _factor_60_ma_filter_reversal(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    mean35 = op.ts_mean(tmp, "close_adj", 35, min_periods=18)
    delta10 = op.ts_delta(tmp, "close_adj", 10)
    return pd.Series(np.where(tmp["close_adj"] > mean35, -delta10, 0.0), index=tmp.index, dtype=float)


def _factor_61_price_volume_divergence_cov(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    tmp["_delta_volume_61"] = op.ts_delta(tmp, "volume", 1)
    tmp["_delta_close_61"] = op.ts_delta(tmp, "close_adj", 1)
    return -op.ts_covariance(tmp, "_delta_volume_61", "_delta_close_61", 30, min_periods=15)


def _factor_62_reversal_turnover_boost(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    ratio = op.safe_div(tmp["close_adj"], op.ts_delay(tmp, "close_adj", 5))
    return -(ratio * tmp["turnover_rate"])


def _factor_63_price_volume_efficiency(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    volume_rank = _cs_rank_series(tmp, op.ts_sum(tmp, "volume", 30, min_periods=15), "_volume_sum30_rank_63")
    amount_rank = _cs_rank_series(tmp, op.ts_sum(tmp, "amount", 30, min_periods=15), "_amount_sum30_rank_63")
    return op.safe_div(volume_rank, amount_rank)


def _factor_64_residual_volatility(df: pd.DataFrame) -> pd.Series:
    return _factor_tvsd20_proxy(df)


def _factor_65_turnover_volatility_momentum(df: pd.DataFrame) -> pd.Series:
    tmp = _base(df)
    turnover_std_rank = _cs_rank_series(tmp, op.ts_std(tmp, "turnover_rate", 14, min_periods=7), "_turnover_std14_rank_65")
    penalty = -op.signed_power(turnover_std_rank, 2)
    return penalty * op.ts_rank(tmp, "close_adj", 30, min_periods=15)


def _pack2_public_alpha_specs() -> list[PublicAlphaSpec]:
    specs: list[PublicAlphaSpec] = [
        PublicAlphaSpec(
            _pack2_meta(
                "27",
                "rl2_27_reverse_price_volume_rank",
                "-1 * RANK(CLOSE) * RANK(VOLUME)",
                "Reverse price-volume rank product factor.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.DIRECT,
                local_expression="-cs_rank(close_adj) * cs_rank(volume)",
            ),
            _factor_27_reverse_price_volume_rank,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "28",
                "rl2_28_volatility_spread_proxy",
                "TS_PERCENTAGE(FACTOR_VOL60D,5) - TS_PERCENTAGE(FACTOR_TVSD20D,5)",
                "Spread between long-window volume volatility proxy and residual volatility proxy.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.PROXY,
                local_expression="ts_percentage(volume_vol_60_proxy,5) - ts_percentage(tvsd20_proxy,5)",
                proxy_reason="FACTOR_VOL60D and FACTOR_TVSD20D mapped to local volume-volatility and residual-volatility proxies.",
            ),
            _factor_28_volatility_spread_proxy,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "29",
                "rl2_29_volume_stable_close",
                "RANK(VOLUME) * (1 - RANK(TS_STDDEV(CLOSE,10)))",
                "High-liquidity and low-close-volatility composite.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.DIRECT,
                local_expression="cs_rank(volume) * (1 - cs_rank(ts_std(close_adj,10)))",
            ),
            _factor_29_volume_stable_close,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "30",
                "rl2_30_short_term_vol_adjusted_return",
                "-1 * RANK(TS_SUM(CLOSE-OPEN,15)) * RANK(TS_STDDEV(CLOSE,15))",
                "Short-term return strength adjusted by close volatility.",
                ["trade_date", "ts_code", "open_adj", "close_adj"],
                AlphaAvailability.DIRECT,
                local_expression="-cs_rank(ts_sum(close_adj-open_adj,15)) * cs_rank(ts_std(close_adj,15))",
            ),
            _factor_30_short_term_vol_adjusted_return,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "31",
                "rl2_31_gap_momentum",
                "(TS_MEAN(OPEN,10)-TS_MEAN(CLOSE,10)) * RANK(TS_DELTA(CLOSE,10))",
                "Gap-like mean spread times 10-day close momentum rank.",
                ["trade_date", "ts_code", "open_adj", "close_adj"],
                AlphaAvailability.DIRECT,
                local_expression="(ts_mean(open_adj,10)-ts_mean(close_adj,10)) * cs_rank(ts_delta(close_adj,10))",
            ),
            _factor_31_gap_momentum,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "32",
                "rl2_32_reverse_volatility_covariance_proxy",
                "-TS_COVARIANCE(FACTOR_VROC12D, TS_STDDEV(RANK(CLOSE),15),20)",
                "Negative covariance between volume-rate-of-change proxy and ranked-price volatility.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.PROXY,
                local_expression="-ts_covariance(vroc12_proxy, ts_std(cs_rank(close_adj),15),20)",
                proxy_reason="FACTOR_VROC12D mapped to 12-day volume ROC proxy.",
            ),
            _factor_32_reverse_volatility_covariance_proxy,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "33",
                "rl2_33_reinstatement_vol_ratio",
                "TS_STDDEV(REINSTATEMENT_CHG_60D,35) / TS_STDDEV(FACTOR_TVSD20D,35)",
                "Reinstatement-volatility ratio.",
                ["trade_date", "ts_code", "close_adj", "volume", "REINSTATEMENT_CHG_60D"],
                AlphaAvailability.PROXY,
                local_expression="ts_std(reinstatement_chg_60d_proxy,35) / ts_std(tvsd20_proxy,35)",
                proxy_reason="REINSTATEMENT_CHG_60D mapped to 60-day adjusted-close return; FACTOR_TVSD20D mapped to local residual-volatility proxy.",
            ),
            _factor_33_reinstatement_vol_ratio,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "34",
                "rl2_34_composite_momentum_flow",
                "TS_DECAY_LINEAR(FACTOR_VROC12D,10) + TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,10)",
                "Composite momentum and main-flow factor.",
                ["trade_date", "ts_code", "volume", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(vroc12_proxy,10) + ts_decay_linear(main_in_flow_20d_proxy,10)",
                proxy_reason="FACTOR_VROC12D and MAIN_IN_FLOW_20D_V2 mapped to local volume ROC and Tushare moneyflow proxies.",
            ),
            _factor_34_composite_momentum_flow,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "35",
                "rl2_35_flow_momentum",
                "TS_DECAY_LINEAR(ABS(RANK(MAIN_IN_FLOW_20D_V2))-ABS(RANK(SLARGE_IN_FLOW_V2)),15)",
                "Main-flow versus super-large-flow momentum.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2", "SLARGE_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(abs(cs_rank(main_in_flow_20d))-abs(cs_rank(slarge_in_flow)),15)",
                proxy_reason="Flow fields mapped from Tushare large and extra-large order moneyflow.",
            ),
            _factor_35_flow_momentum,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "36",
                "rl2_36_momentum_flow_volatility",
                "TS_DECAY_LINEAR(FACTOR_VOL60D,10) * TS_PERCENTAGE(MAIN_IN_FLOW_20D_V2,10)",
                "Volume-volatility and main-flow interaction.",
                ["trade_date", "ts_code", "volume", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(vol60_proxy,10) * ts_percentage(main_in_flow_20d_proxy,10)",
                proxy_reason="FACTOR_VOL60D mapped to local volume-volatility proxy; moneyflow mapped from Tushare.",
            ),
            _factor_36_momentum_flow_volatility,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "37",
                "rl2_37_flow_volatility_synergy",
                "TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,10) * FACTOR_VOL60D",
                "Decayed main-flow times volume volatility.",
                ["trade_date", "ts_code", "volume", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(main_in_flow_20d_proxy,10) * vol60_proxy",
                proxy_reason="FACTOR_VOL60D mapped to local volume-volatility proxy; moneyflow mapped from Tushare.",
            ),
            _factor_37_flow_volatility_synergy,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "38",
                "rl2_38_main_superlarge_flow_synergy",
                "RANK(TS_PERCENTAGE(MAIN_IN_FLOW_20D_V2,30) * TS_DECAY_LINEAR(SLARGE_IN_FLOW_V2,15))",
                "Main and super-large flow synergy.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2", "SLARGE_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="cs_rank(ts_percentage(main_in_flow_20d_proxy,30) * ts_decay_linear(slarge_in_flow_proxy,15))",
                proxy_reason="Flow fields mapped from Tushare large and extra-large order moneyflow.",
            ),
            _factor_38_main_superlarge_flow_synergy,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "39",
                "rl2_39_main_superlarge_flow_spread",
                "TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,5) - TS_DECAY_LINEAR(SLARGE_IN_FLOW_V2,5)",
                "Main minus super-large flow spread.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2", "SLARGE_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(main_in_flow_20d_proxy,5) - ts_decay_linear(slarge_in_flow_proxy,5)",
                proxy_reason="Flow fields mapped from Tushare large and extra-large order moneyflow.",
            ),
            _factor_39_main_superlarge_flow_spread,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "40",
                "rl2_40_style_ir_spread",
                "TS_IR(FACTOR_CNE5_BETA,20) - TS_IR(FACTOR_CNE5_SIZE,20)",
                "Style IR spread between beta and size.",
                ["trade_date", "ts_code", "FACTOR_CNE5_BETA", "FACTOR_CNE5_SIZE"],
                AlphaAvailability.PROXY,
                local_expression="ts_ir(local_volatility_style_proxy,20) - ts_ir(local_size_style_proxy,20)",
                proxy_reason="CNE5 fields unavailable; mapped to local volatility and size style proxies, not original CNE5 exposures.",
            ),
            _factor_40_style_ir_spread,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "41",
                "rl2_41_flow_volatility_factor",
                "TS_PERCENTAGE(MAIN_IN_FLOW_20D_V2,10) * FACTOR_VOL60D",
                "Main-flow percentile times volume volatility.",
                ["trade_date", "ts_code", "volume", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_percentage(main_in_flow_20d_proxy,10) * vol60_proxy",
                proxy_reason="FACTOR_VOL60D mapped to local volume-volatility proxy; moneyflow mapped from Tushare.",
            ),
            _factor_41_flow_volatility_factor,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "42",
                "rl2_42_flow_linear_decay",
                "TS_DECAY_LINEAR(RANK(MAIN_IN_FLOW_20D_V2-SLARGE_IN_FLOW_V2),15)",
                "Linear decay of ranked flow spread.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2", "SLARGE_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(cs_rank(main_in_flow_20d_proxy-slarge_in_flow_proxy),15)",
                proxy_reason="Flow fields mapped from Tushare large and extra-large order moneyflow.",
            ),
            _factor_42_flow_linear_decay,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "43",
                "rl2_43_main_flow_momentum",
                "TS_PERCENTAGE(TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,6),3)",
                "Main-flow momentum after decay.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_percentage(ts_decay_linear(main_in_flow_20d_proxy,6),3)",
                proxy_reason="MAIN_IN_FLOW_20D_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_43_main_flow_momentum,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "44",
                "rl2_44_momentum_reversal",
                "-TS_SUM(TS_DELTA(RANK(AF_CLOSE),1),40)",
                "40-day rank-change reversal.",
                ["trade_date", "ts_code", "AF_CLOSE"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_sum(ts_delta(cs_rank(AF_CLOSE),1),40)",
            ),
            _factor_44_momentum_reversal,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "45",
                "rl2_45_flow_momentum_reinstatement",
                "ABS(REINSTATEMENT_CHG_60D) * TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,15)",
                "Reinstatement shock times decayed main flow.",
                ["trade_date", "ts_code", "REINSTATEMENT_CHG_60D", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="abs(reinstatement_chg_60d_proxy) * ts_decay_linear(main_in_flow_20d_proxy,15)",
                proxy_reason="REINSTATEMENT_CHG_60D mapped to 60-day adjusted-close return; moneyflow mapped from Tushare.",
            ),
            _factor_45_flow_momentum_reinstatement,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "46",
                "rl2_46_main_flow_linear_decay",
                "TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,10)",
                "Main-flow linear decay.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_20D_V2"],
                AlphaAvailability.PROXY,
                local_expression="ts_decay_linear(main_in_flow_20d_proxy,10)",
                proxy_reason="MAIN_IN_FLOW_20D_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_46_main_flow_linear_decay,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "47",
                "rl2_47_price_flow_volatility_coupling",
                "-RANK(AF_CLOSE/DELAY(AF_CLOSE,15)) * RANK(TS_STDDEV(MAIN_IN_FLOW_V2,15))",
                "Price momentum and main-flow volatility inverse coupling.",
                ["trade_date", "ts_code", "AF_CLOSE", "MAIN_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="-cs_rank(AF_CLOSE/delay(AF_CLOSE,15)) * cs_rank(ts_std(main_in_flow_proxy,15))",
                proxy_reason="MAIN_IN_FLOW_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_47_price_flow_volatility_coupling,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "48",
                "rl2_48_price_volume_decay_synergy",
                "-TS_PERCENTAGE(RANK(CLOSE),10) * TS_DECAY_LINEAR(FACTOR_VROC12D,60)",
                "Negative synergy of ranked-price time-series percentile and decayed volume ROC proxy.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.PROXY,
                local_expression="-ts_percentage(cs_rank(close_adj),10) * ts_decay_linear(vroc12_proxy,60)",
                proxy_reason="FACTOR_VROC12D mapped to 12-day volume ROC proxy.",
            ),
            _factor_48_price_volume_decay_synergy,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "49",
                "rl2_49_dual_volatility_flow_inverse",
                "-RANK(TS_STDDEV(CLOSE,10)) * RANK(TS_STDDEV(MAIN_IN_FLOW_V2,10))",
                "Close volatility and main-flow volatility inverse interaction.",
                ["trade_date", "ts_code", "close_adj", "MAIN_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="-cs_rank(ts_std(close_adj,10)) * cs_rank(ts_std(main_in_flow_proxy,10))",
                proxy_reason="MAIN_IN_FLOW_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_49_dual_volatility_flow_inverse,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "50",
                "rl2_50_price_volume_volatility_inverse",
                "-1 * RANK(TS_STDDEV(CLOSE,10)) * RANK(TS_STDDEV(VOLUME,10))",
                "Negative interaction of close and volume volatility ranks.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.DIRECT,
                local_expression="-cs_rank(ts_std(close_adj,10)) * cs_rank(ts_std(volume,10))",
            ),
            _factor_50_price_volume_volatility_inverse,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "51",
                "rl2_51_multidim_reversal",
                "-RANK(CLOSE/TS_MEAN(CLOSE,20)) * RANK(TS_MEAN(VOLUME,20)) * RANK(TS_MEAN(TURN_RATE,20))",
                "Three-leg reversal using price-to-mean, volume mean and turnover mean ranks.",
                ["trade_date", "ts_code", "close_adj", "volume", "turnover_rate"],
                AlphaAvailability.DIRECT,
                local_expression="-cs_rank(close_adj/ts_mean(close_adj,20)) * cs_rank(ts_mean(volume,20)) * cs_rank(ts_mean(turnover_rate,20))",
            ),
            _factor_51_multidim_reversal,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "52",
                "rl2_52_main_flow_peak_inverse",
                "-RANK(TS_MAX(MAIN_IN_FLOW_V2,10))",
                "Inverse rank of main-flow peak.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="-cs_rank(ts_max(main_in_flow_proxy,10))",
                proxy_reason="MAIN_IN_FLOW_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_52_main_flow_peak_inverse,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "53",
                "rl2_53_main_flow_stability",
                "-TS_STDDEV(MAIN_IN_FLOW_V2,10)",
                "Inverse main-flow volatility.",
                ["trade_date", "ts_code", "MAIN_IN_FLOW_V2"],
                AlphaAvailability.PROXY,
                local_expression="-ts_std(main_in_flow_proxy,10)",
                proxy_reason="MAIN_IN_FLOW_V2 mapped from Tushare large plus extra-large net moneyflow.",
            ),
            _factor_53_main_flow_stability,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "54",
                "rl2_54_turnover_volatility_inverse",
                "-TS_STDDEV(TURN_RATE,15)",
                "Inverse turnover-rate volatility.",
                ["trade_date", "ts_code", "turnover_rate"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_std(turnover_rate,15)",
            ),
            _factor_54_turnover_volatility_inverse,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "55",
                "rl2_55_vol_turnover_coupling",
                "SCALE(RANK(TS_STDDEV(CLOSE,15))) * -1 * TURN_RATE",
                "Scaled close-volatility rank coupled with negative turnover.",
                ["trade_date", "ts_code", "close_adj", "turnover_rate"],
                AlphaAvailability.DIRECT,
                local_expression="cs_zscore(cs_rank(ts_std(close_adj,15))) * (-turnover_rate)",
            ),
            _factor_55_vol_turnover_coupling,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "56",
                "rl2_56_ma_deviation_volume_weighted",
                "(TS_MEAN(CLOSE,20)-CLOSE) * RANK(VOLUME)",
                "20-day moving-average deviation weighted by volume rank.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.DIRECT,
                local_expression="(ts_mean(close_adj,20)-close_adj) * cs_rank(volume)",
            ),
            _factor_56_ma_deviation_volume_weighted,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "57",
                "rl2_57_vol_adjusted_reversal",
                "-TS_MEAN(SIGNEDPOWER(CHANGE_PCT,2),30)",
                "Inverse mean signed-squared return.",
                ["trade_date", "ts_code", "close_adj"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_mean(signed_power(change_pct,2),30)",
            ),
            _factor_57_vol_adjusted_reversal,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "58",
                "rl2_58_liquidity_stability",
                "-TS_MEAN(RANK(AMOUNT),10)",
                "Inverse rolling mean of amount rank.",
                ["trade_date", "ts_code", "amount"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_mean(cs_rank(amount),10)",
            ),
            _factor_58_liquidity_stability,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "59",
                "rl2_59_price_vwap_volume_volatility",
                "-TS_STDDEV((CLOSE-VWAP) * VOLUME,10)",
                "Inverse volatility of close-to-vwap deviation weighted by volume.",
                ["trade_date", "ts_code", "close_adj", "volume", "amount"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_std((close_adj-vwap)*volume,10)",
            ),
            _factor_59_price_vwap_volume_volatility,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "60",
                "rl2_60_ma_filter_reversal",
                "((TS_SUM(CLOSE,35)/35) < CLOSE) ? (-1*DELTA(CLOSE,10)) : 0",
                "Negative 10-day close delta only when close breaks above 35-day mean.",
                ["trade_date", "ts_code", "close_adj"],
                AlphaAvailability.DIRECT,
                local_expression="where(close_adj > ts_mean(close_adj,35), -ts_delta(close_adj,10), 0)",
            ),
            _factor_60_ma_filter_reversal,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "61",
                "rl2_61_price_volume_divergence_cov",
                "-TS_COVARIANCE(DELTA(VOLUME,1), DELTA(CLOSE,1),30)",
                "Negative covariance of daily volume delta and close delta.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.DIRECT,
                local_expression="-ts_covariance(ts_delta(volume,1), ts_delta(close_adj,1), 30)",
            ),
            _factor_61_price_volume_divergence_cov,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "62",
                "rl2_62_reversal_turnover_boost",
                "-(AF_CLOSE/DELAY(AF_CLOSE,5) * TURN_RATE)",
                "Negative 5-day adjusted-close ratio times turnover rate.",
                ["trade_date", "ts_code", "close_adj", "turnover_rate"],
                AlphaAvailability.DIRECT,
                local_expression="-(close_adj / delay(close_adj,5) * turnover_rate)",
            ),
            _factor_62_reversal_turnover_boost,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "63",
                "rl2_63_price_volume_efficiency",
                "RANK(TS_SUM(VOLUME,30)) / RANK(TS_SUM(AMOUNT,30))",
                "Ratio of 30-day volume-sum rank to amount-sum rank.",
                ["trade_date", "ts_code", "volume", "amount"],
                AlphaAvailability.DIRECT,
                local_expression="cs_rank(ts_sum(volume,30)) / cs_rank(ts_sum(amount,30))",
            ),
            _factor_63_price_volume_efficiency,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "64",
                "rl2_64_residual_volatility",
                "TS_STDDEV(CS_REGRESSION(CLOSE, VOLUME, OUT_TYPE=0),20)",
                "20-day volatility of cross-sectional close-on-volume residuals.",
                ["trade_date", "ts_code", "close_adj", "volume"],
                AlphaAvailability.PROXY,
                local_expression="ts_std(cs_regression_resid(close_adj ~ volume),20)",
                proxy_reason="CLOSE mapped to close_adj and CS_REGRESSION semantics matched with local residual regression helper.",
            ),
            _factor_64_residual_volatility,
        ),
        PublicAlphaSpec(
            _pack2_meta(
                "65",
                "rl2_65_turnover_volatility_momentum",
                "-SIGNED_POWER(RANK(TS_STDDEV(TURN_RATE,14)),2) * TS_RANK(CLOSE,30)",
                "Negative squared rank of turnover volatility times 30-day time-series close rank.",
                ["trade_date", "ts_code", "close_adj", "turnover_rate"],
                AlphaAvailability.DIRECT,
                local_expression="-signed_power(cs_rank(ts_std(turnover_rate,14)),2) * ts_rank(close_adj,30)",
            ),
            _factor_65_turnover_volatility_momentum,
        ),
    ]
    return specs


def public_alpha_specs() -> list[PublicAlphaSpec]:
    specs: list[PublicAlphaSpec] = [
        PublicAlphaSpec(_meta("01", "rl_01_tail_risk_reversal", "negative price-volume skew/kurtosis composite", "Tail-risk and crowded-volume reversal proxy.", ["trade_date", "ts_code", "close_adj", "volume"], AlphaAvailability.PROXY, proxy_reason="TS_MAX_SKEW/TS_MIN_SKEW approximated with rolling skew extrema."), _factor_01_tail_risk),
        PublicAlphaSpec(_meta("02", "rl_02_projection_support_proxy", "volume-rank projection plus main-flow low-shadow leg", "Volume-projection and main-flow low-shadow factor.", ["trade_date", "ts_code", "close_adj", "volume", "high_adj", "low_adj", "MAIN_IN_FLOW_20D_V2", "total_mv"], AlphaAvailability.PROXY, proxy_reason="MAIN_IN_FLOW_20D_V2 mapped from Tushare large plus extra-large net moneyflow."), _factor_02_projection_support_proxy),
        PublicAlphaSpec(_meta("03", "rl_03_pvt_covariance_reversal", "negative covariance of return and PVT", "PVT covariance reversal.", ["trade_date", "ts_code", "close_adj", "volume"], AlphaAvailability.PROXY, proxy_reason="FACTOR_PVT1D reconstructed from return * volume."), _factor_03_pvt_covariance),
        PublicAlphaSpec(_meta("04", "rl_04_valuation_price_resid_proxy", "valuation-price residual composite", "Approximates proprietary valuation fields with PB/PE and price residual.", ["trade_date", "ts_code", "close_adj", "volume", "amount", "industry", "pb", "pe_ttm"], AlphaAvailability.PROXY, proxy_reason="Industry BM and EV/EBITDA fields unavailable."), _factor_04_valuation_price_proxy),
        PublicAlphaSpec(_meta("05", "rl_05_ema_midprice_divergence", "EMA state count minus midprice buildup", "EMA and midprice trend divergence.", ["trade_date", "ts_code", "close_adj", "high_adj", "low_adj"], AlphaAvailability.PROXY, proxy_reason="DIGITAL_COUNT and TS_MAX_BUILDUP approximated with state counts and rolling extrema."), _factor_05_ema_midprice_divergence),
        PublicAlphaSpec(_meta("06", "rl_06_vol_compression_momentum", "negative volatility-compression momentum", "Direct OHLCV implementation.", ["trade_date", "ts_code", "close_adj"], AlphaAvailability.DIRECT), _factor_06_vol_compression_momentum),
        PublicAlphaSpec(_meta("07", "rl_07_log_volume_price_trend", "negative log volume-price trend", "Direct price/volume implementation.", ["trade_date", "ts_code", "close_adj", "volume"], AlphaAvailability.DIRECT), _factor_07_log_volume_price_trend),
        PublicAlphaSpec(_meta("08", "rl_08_variance_ratio_divergence", "negative gain/loss variance ratio divergence", "Local volatility-structure proxy.", ["trade_date", "ts_code", "close_adj"], AlphaAvailability.PROXY, proxy_reason="FACTOR_GAINLOSSVARIANCERATIO120D reconstructed from positive/negative return variance."), _factor_08_variance_ratio_divergence),
        PublicAlphaSpec(_meta("09", "rl_09_filtered_momentum_proxy", "filtered composite momentum", "Uses PE filter and earnings-yield proxy.", ["trade_date", "ts_code", "close_adj", "volume", "pe_ttm"], AlphaAvailability.PROXY, proxy_reason="CETOPTTM/PE_LFY unavailable; uses earnings-yield and PE_TTM."), _factor_09_filtered_momentum_proxy),
        PublicAlphaSpec(_meta("10", "rl_10_cashflow_price_trend_proxy", "cashflow price-volume trend", "Uses MFI proxy and operating cash-flow margin.", ["trade_date", "ts_code", "high_adj", "low_adj", "close_adj", "volume", "operating_cf_margin_ttm"], AlphaAvailability.PROXY, proxy_reason="FACTOR_MFI21D reconstructed; FACTOR_CASHOFSALES mapped to operating_cf_margin_ttm."), _factor_10_cashflow_price_trend_proxy),
        PublicAlphaSpec(_meta("11", "rl_11_large_outflow_reversal", "industry BBI/EMA spread valuation proxy", "Industry moving-average dispersion and valuation proxy.", ["trade_date", "ts_code", "close_adj", "industry", "pe_ttm"], AlphaAvailability.PROXY, proxy_reason="FACTOR_BBI, FACTOR_EMA60D and EV/EBITDA mapped to local BBI/EMA60 and PE_TTM proxy."), _factor_11_industry_ma_valuation_proxy),
        PublicAlphaSpec(_meta("12", "rl_12_industry_mainflow_profit_quality", "industry main-flow profit-quality adjustment", "Industry main-flow rank adjusted by local profitability proxy.", ["trade_date", "ts_code", "industry", "MAIN_IN_FLOW_20D_V2", "roe_ttm"], AlphaAvailability.PROXY, proxy_reason="MAIN_FUND_IN_20D_V2 mapped to local MAIN_IN_FLOW_20D_V2; 5Y ROE/ROA-change mapped to ROE_TTM stability proxy."), _factor_12_industry_mainflow_quality_proxy),
        PublicAlphaSpec(_meta("13", "rl_13_fundflow_cross_section_pct", "price-volume fund-flow cross-section percentile", "Price-volume change gated by top moneyflow-day percentile.", ["trade_date", "ts_code", "AF_HIGH", "AF_VWAP", "MAIN_IN_FLOW_DAYS_20D_V2", "CON_FUND_DAY_IN_10D_V2"], AlphaAvailability.PROXY, proxy_reason="Institutional flow day field mapped to extra-large moneyflow positive-day proxy."), _factor_13_fundflow_cross_section_pct),
        PublicAlphaSpec(_meta("14", "rl_14_large_outflow_reversal_dup", "large outflow momentum reversal duplicate", "Large outflow pressure reversal; duplicate economic theme with public factor 14 text.", ["trade_date", "ts_code", "turnover_rate", "LARGE_OUT_FLOW_V2"], AlphaAvailability.PROXY, proxy_reason="LARGE_OUT_FLOW_V2 mapped from Tushare large plus extra-large sell amount; TS_MOMENT(...,2) mapped to 60-day variance."), _factor_14_large_outflow_reversal),
        PublicAlphaSpec(_meta("15", "rl_15_nonlinear_price_volume_proxy", "nonlinear price-volume fund-flow factor", "Nonlinear price-volume factor with moneyflow-day leg restored.", ["trade_date", "ts_code", "open_adj", "high_adj", "low_adj", "close_adj", "volume", "amount", "MAIN_IN_FLOW_DAYS_10D_V2"], AlphaAvailability.PROXY, proxy_reason="MAIN_IN_FLOW_DAYS_10D_V2 mapped from Tushare large plus extra-large positive moneyflow days."), _factor_15_nonlinear_price_volume_proxy),
        PublicAlphaSpec(_meta("16", "rl_16_decay_volume_trend", "negative decay volume-price composite", "Direct implementation.", ["trade_date", "ts_code", "close_adj", "volume"], AlphaAvailability.DIRECT), _factor_16_decay_volume_trend),
        PublicAlphaSpec(_meta("17", "rl_17_vol_trend_structure", "volatility trend structure", "Direct implementation from EMA/MA/VOL/Sharpe.", ["trade_date", "ts_code", "close_adj"], AlphaAvailability.DIRECT), _factor_17_vol_trend_structure),
        PublicAlphaSpec(_meta("18", "rl_18_turnover_adjusted_price", "turnover-adjusted price residual", "Direct implementation with cross-sectional regression residual.", ["trade_date", "ts_code", "close_adj", "turnover_rate"], AlphaAvailability.DIRECT), _factor_18_turnover_adjusted_price),
        PublicAlphaSpec(_meta("19", "rl_19_poly_volume_price_reversal", "nonlinear volume-price extreme reversal", "Lightweight volume-price residual proxy.", ["trade_date", "ts_code", "close_adj", "volume"], AlphaAvailability.PROXY, proxy_reason="TS_POLY_REGRESSION is too costly for A-stage full panel; mapped to rolling linear residual plus kurtosis."), _factor_19_poly_volume_price_reversal),
        PublicAlphaSpec(_meta("20", "rl_20_price_exhaustion_reversal", "price momentum decay and reversal", "Direct/proxy implementation from OHLC.", ["trade_date", "ts_code", "close_adj", "high_adj"], AlphaAvailability.PROXY, proxy_reason="TS_MAX_SUM/TS_AV_DIFF mapped to rolling sum maxima and mean deviation."), _factor_20_price_exhaustion),
        PublicAlphaSpec(_meta("21", "rl_21_log_momentum_reversal_proxy", "log momentum inverse rank", "Uses 60-day ROC proxy for FACTOR_ROCTTM.", ["trade_date", "ts_code", "close_adj"], AlphaAvailability.PROXY, proxy_reason="FACTOR_ROCTTM unavailable."), _factor_21_log_momentum_reversal_proxy),
        PublicAlphaSpec(_meta("22", "rl_22_volume_divergence_momentum", "volume divergence composite momentum", "Direct implementation.", ["trade_date", "ts_code", "close_adj", "volume", "turnover_rate"], AlphaAvailability.DIRECT), _factor_22_volume_divergence_momentum),
        PublicAlphaSpec(_meta("23", "rl_23_turnover_relative_reversal", "turnover relative strength reversal", "Direct implementation.", ["trade_date", "ts_code", "turnover_rate"], AlphaAvailability.DIRECT), _factor_23_turnover_relative_reversal),
        PublicAlphaSpec(_meta("24", "rl_24_price_reversal_30d", "30-day adjusted price reversal", "Direct implementation.", ["trade_date", "ts_code", "close_adj"], AlphaAvailability.DIRECT), _factor_24_price_reversal_30d),
        PublicAlphaSpec(_meta("25", "rl_25_high_open_decay", "high-open momentum decay", "Direct implementation.", ["trade_date", "ts_code", "high_adj", "open_adj"], AlphaAvailability.DIRECT), _factor_25_high_open_decay),
        PublicAlphaSpec(_meta("26", "rl_26_moneyflow_drawdown", "money-flow max drawdown", "15-day net moneyflow drawdown stability.", ["trade_date", "ts_code", "NET_MF_AMOUNT_V2"], AlphaAvailability.PROXY, proxy_reason="NET_MF_AMOUNT_V2 mapped to Tushare net_mf_amount, falling back to large plus extra-large net moneyflow."), _factor_26_moneyflow_drawdown),
    ]
    specs.extend(_pack2_public_alpha_specs())
    return specs


def public_factor_availability(df: pd.DataFrame | None = None) -> pd.DataFrame:
    records = []
    columns = set(df.columns) if df is not None else set()
    for spec in public_alpha_specs():
        meta = spec.metadata
        missing = sorted(set(meta.required_columns) - columns) if df is not None else list(meta.missing_columns)
        record = meta.to_record()
        if df is not None and missing:
            record["missing_columns"] = ",".join(missing)
            record["validation_status"] = AlphaValidationStatus.SKIPPED.value
            record["skip_reason"] = record.get("skip_reason") or f"missing columns: {','.join(missing)}"
        records.append(record)
    return pd.DataFrame(records)


def calculate_public_factors(df: pd.DataFrame, include_skipped: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df[["trade_date", "ts_code"]].copy()
    metadata = []
    for spec in public_alpha_specs():
        record = spec.metadata
        if record.availability == AlphaAvailability.SKIPPED and not include_skipped:
            metadata.append(record)
            continue
        missing = sorted(set(spec.required_columns) - set(df.columns))
        if missing:
            metadata.append(
                _meta(
                    record.source_factor_id,
                    record.factor_name,
                    record.expression,
                    record.description,
                    record.required_columns,
                    AlphaAvailability.SKIPPED,
                    source=record.source,
                    version=record.version,
                    release_date=record.release_date or "",
                    local_expression=record.local_expression,
                    proxy_reason=record.proxy_reason,
                    skip_reason=f"missing columns: {','.join(missing)}",
                    missing_columns=missing,
                )
            )
            continue
        out[spec.name] = spec.calculate(df)
        metadata.append(record)
    return out, metadata_frame(metadata)


def load_adjusted_price_panel(
    raw_dir: Path,
    start: str = "20180101",
    end: str = "20260706",
    ts_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    daily = pd.read_parquet(raw_dir / f"daily_{start}_{end}.parquet")
    adj = pd.read_parquet(raw_dir / f"adj_factor_{start}_{end}.parquet")
    basic = pd.read_parquet(raw_dir / f"daily_basic_{start}_{end}.parquet")
    if ts_codes is not None:
        codes = set(ts_codes)
        daily = daily[daily["ts_code"].isin(codes)]
        adj = adj[adj["ts_code"].isin(codes)]
        basic = basic[basic["ts_code"].isin(codes)]
    panel = daily.merge(adj, on=["trade_date", "ts_code"], how="left")
    panel = panel.merge(
        basic[["trade_date", "ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "total_mv", "circ_mv"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    for col in ["open", "high", "low", "close"]:
        panel[f"{col}_adj"] = panel[col] * panel["adj_factor"]
    panel["log_mv"] = np.log(panel["total_mv"].where(panel["total_mv"] > 0))
    return panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
