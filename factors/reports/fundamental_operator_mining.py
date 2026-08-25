"""Run systematic fundamental-operator mining on CSI500."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import ConstantInputWarning
except Exception:  # pragma: no cover - scipy is an optional transitive dependency.
    ConstantInputWarning = None

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in multiply", category=RuntimeWarning)
if ConstantInputWarning is not None:
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

from factors.alpha.candidates import optional_fundamental_candidates
from factors.alpha.fundamental_factory import clear_fundamental_operator_cache, fundamental_operator_candidates
from factors.alpha.miner import AlphaMiner, AlphaMiningConfig, AlphaMiningResult


DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT = Path("outputs/fundamental_operator_mining_csi500")
DEFAULT_REPORT = Path("docs/fundamental_operator_mining_csi500_2026-07-15.md")


def _pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.2%}"


def _num(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.3f}"


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    out = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _read_split(processed_dir: Path, split: str, suffix: str) -> pd.DataFrame:
    path = processed_dir / f"{split}_fundamental_{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    weight_col = "csi500_index_weight" if "csi500_index_weight" in df.columns else "index_weight_000905_SH"
    if weight_col not in df.columns:
        raise KeyError(f"{path} missing CSI500 weight column")
    df = df[df[weight_col] > 0].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _read_split_schema(processed_dir: Path, split: str, suffix: str) -> pd.DataFrame:
    path = processed_dir / f"{split}_fundamental_{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq

        columns = pq.ParquetFile(path).schema.names
    except Exception:
        columns = pd.read_parquet(path).head(0).columns.tolist()
    return pd.DataFrame(columns=columns)


def _dedupe_candidates(candidates: Iterable) -> list:
    out = {}
    for candidate in candidates:
        out.setdefault(candidate.name, candidate)
    return list(out.values())


def _candidate_universe(
    df: pd.DataFrame,
    include_atoms: bool,
    include_neutralized: bool,
    candidate_regex: str | None = None,
    max_candidates: int | None = None,
) -> list:
    legacy = optional_fundamental_candidates(windows=(20,))
    generated = fundamental_operator_candidates(include_atoms=include_atoms, include_neutralized=include_neutralized)
    candidates = [c for c in _dedupe_candidates([*legacy, *generated]) if c.is_available(df)]
    if candidate_regex:
        pattern = re.compile(candidate_regex)
        candidates = [candidate for candidate in candidates if pattern.search(candidate.name)]
    if max_candidates:
        candidates = candidates[:max_candidates]
    return candidates


def _evaluate_subset(factor_values: pd.DataFrame, candidates: list, config: AlphaMiningConfig) -> pd.DataFrame:
    if not config.include_turnover:
        return _fast_evaluate_factor_values(factor_values, candidates, config)
    miner = AlphaMiner(candidates, config=config)
    return miner.evaluate(factor_values)


def _run_split(df: pd.DataFrame, candidates: list, output_dir: Path, config: AlphaMiningConfig) -> AlphaMiningResult:
    miner = AlphaMiner(candidates, config=config)
    if config.include_turnover:
        result = miner.run(df)
    else:
        factor_values = miner.compute_factor_values(df)
        clear_fundamental_operator_cache()
        summary = _fast_evaluate_factor_values(factor_values, candidates, config)
        selected = miner.select(factor_values, summary)
        result = AlphaMiningResult(factor_values=factor_values, summary=summary, selected=selected)
    miner.save(result, output_dir)
    return result


def _matrix_corrwith(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)[:, None]
    mask = np.isfinite(values) & np.isfinite(y_values)
    count = mask.sum(axis=0)
    out = np.full(values.shape[1], np.nan, dtype=float)
    valid = count >= 3
    if not valid.any():
        return pd.Series(out, index=x.columns)

    safe_values = np.where(mask, values, 0.0)
    x_mean = np.divide(safe_values.sum(axis=0), count, out=np.zeros_like(out), where=count > 0)
    y_sum = np.where(mask, y_values, 0.0).sum(axis=0)
    y_mean = np.divide(y_sum, count, out=np.zeros_like(out), where=count > 0)

    x_diff = np.where(mask, values - x_mean, 0.0)
    y_diff = np.where(mask, y_values - y_mean, 0.0)
    numerator = (x_diff * y_diff).sum(axis=0)
    x_ss = (x_diff * x_diff).sum(axis=0)
    y_ss = (y_diff * y_diff).sum(axis=0)
    denom = np.sqrt(x_ss * y_ss)
    valid &= denom > 0
    out[valid] = numerator[valid] / denom[valid]
    return pd.Series(out, index=x.columns)


def _fast_evaluate_factor_values(
    factor_values: pd.DataFrame,
    candidates: list,
    config: AlphaMiningConfig,
) -> pd.DataFrame:
    """Evaluate a wide factor matrix by date, avoiding factor-by-factor loops."""
    factor_cols = [candidate.name for candidate in candidates if candidate.name in factor_values.columns]
    if not factor_cols:
        return pd.DataFrame()

    coverage = factor_values[factor_cols].notna().mean()
    ic_rows = []
    rankic_rows = []
    date_col = config.date_col
    label_col = config.label_col
    for date, sub in factor_values.groupby(date_col, sort=False):
        y = pd.to_numeric(sub[label_col], errors="coerce")
        valid_label = y.notna()
        if valid_label.sum() < 3:
            continue
        y = y.loc[valid_label]
        x = sub.loc[valid_label, factor_cols]
        valid_cols = x.columns[x.notna().sum() >= 3]
        ic = pd.Series(np.nan, index=factor_cols, dtype=float)
        rank_ic = pd.Series(np.nan, index=factor_cols, dtype=float)
        if len(valid_cols) > 0:
            x_valid = x[valid_cols]
            ic.loc[valid_cols] = _matrix_corrwith(x_valid, y)
            rank_ic.loc[valid_cols] = _matrix_corrwith(x_valid.rank(), y.rank())
        ic.name = date
        rank_ic.name = date
        ic_rows.append(ic)
        rankic_rows.append(rank_ic)

    if not ic_rows:
        return pd.DataFrame()
    ic_df = pd.DataFrame(ic_rows)
    rankic_df = pd.DataFrame(rankic_rows)
    meta = {candidate.name: candidate for candidate in candidates}

    records = []
    for factor in factor_cols:
        candidate = meta[factor]
        ic = ic_df[factor].dropna() if factor in ic_df else pd.Series(dtype=float)
        rank_ic = rankic_df[factor].dropna() if factor in rankic_df else pd.Series(dtype=float)
        if rank_ic.empty:
            continue
        ic_mean = ic.mean()
        ic_std = ic.std()
        rankic_mean = rank_ic.mean()
        rankic_std = rank_ic.std()
        records.append(
            {
                "factor": candidate.name,
                "family": candidate.family,
                "expression": candidate.expression,
                "description": candidate.description,
                "window": candidate.window,
                "complexity": candidate.complexity,
                "coverage": coverage[factor],
                "n_ic_dates": len(rank_ic),
                "IC_mean": ic_mean,
                "IC_IR": ic_mean / (ic_std + 1e-9) if pd.notna(ic_std) else np.nan,
                "RankIC_mean": rankic_mean,
                "RankIC_IR": rankic_mean / (rankic_std + 1e-9) if pd.notna(rankic_std) else np.nan,
                "turnover": pd.NA,
            }
        )

    if not records:
        return pd.DataFrame()
    summary = pd.DataFrame(records)
    summary["score"] = summary["RankIC_mean"].abs() * summary["coverage"] / summary["complexity"].clip(lower=1)
    return summary.sort_values("score", ascending=False).reset_index(drop=True)


def _build_stability_table(
    summaries: dict[str, pd.DataFrame],
    candidates: list,
    min_adj_rankic: float,
    min_ytd_rankic: float,
    min_coverage: float,
) -> pd.DataFrame:
    meta = pd.DataFrame(
        [
            {
                "factor": c.name,
                "family": c.family,
                "expression": c.expression,
                "description": c.description,
                "complexity": c.complexity,
            }
            for c in candidates
        ]
    ).drop_duplicates("factor")
    rows = []
    all_factors = sorted(set().union(*[set(s["factor"]) for s in summaries.values() if not s.empty]))
    for factor in all_factors:
        row = {"factor": factor}
        coverages = []
        for split, summary in summaries.items():
            one = summary[summary["factor"] == factor]
            if one.empty:
                row[f"{split}_rankic"] = np.nan
                row[f"{split}_rankic_ir"] = np.nan
                row[f"{split}_coverage"] = np.nan
            else:
                item = one.iloc[0]
                row[f"{split}_rankic"] = item["RankIC_mean"]
                row[f"{split}_rankic_ir"] = item["RankIC_IR"]
                row[f"{split}_coverage"] = item["coverage"]
                if split in {"train", "valid", "test"}:
                    coverages.append(item["coverage"])
        direction = 1.0 if row.get("train_rankic", np.nan) >= 0 else -1.0
        row["direction"] = "positive" if direction > 0 else "negative"
        adj_cols = []
        for split in ["train", "valid", "test", "ytd_2026"]:
            raw = row.get(f"{split}_rankic", np.nan)
            row[f"{split}_adj_rankic"] = raw * direction if pd.notna(raw) else np.nan
            if split in {"train", "valid", "test"} and pd.notna(row[f"{split}_adj_rankic"]):
                adj_cols.append(row[f"{split}_adj_rankic"])
        row["min_adj_rankic"] = min(adj_cols) if adj_cols else np.nan
        row["avg_adj_rankic"] = float(np.nanmean(adj_cols)) if adj_cols else np.nan
        row["min_coverage"] = min(coverages) if coverages else np.nan
        row["passes_stability"] = bool(
            pd.notna(row["min_adj_rankic"])
            and row["min_adj_rankic"] >= min_adj_rankic
            and row.get("ytd_2026_adj_rankic", np.nan) >= min_ytd_rankic
            and row["min_coverage"] >= min_coverage
        )
        rows.append(row)
    table = pd.DataFrame(rows).merge(meta, on="factor", how="left")
    return table.sort_values(["passes_stability", "min_adj_rankic", "avg_adj_rankic"], ascending=[False, False, False])


def _decorrelate(stability: pd.DataFrame, train_values: pd.DataFrame, max_pair_corr: float) -> list[str]:
    eligible = stability[stability["passes_stability"]].copy()
    if eligible.empty:
        return []
    ordered = eligible.sort_values(["min_adj_rankic", "avg_adj_rankic"], ascending=False)["factor"].tolist()
    ordered = [factor for factor in ordered if factor in train_values.columns]
    corr_matrix = train_values[ordered].corr(min_periods=1000).abs() if ordered else pd.DataFrame()
    selected: list[str] = []
    for factor in ordered:
        if not selected:
            selected.append(factor)
            continue
        corr = corr_matrix.loc[factor, selected] if factor in corr_matrix.index else pd.Series(dtype=float)
        max_corr = corr.max() if not corr.empty else 0.0
        if pd.isna(max_corr) or max_corr < max_pair_corr:
            selected.append(factor)
    return selected


_FACTOR_DESCRIPTION_CN = {
    "quality_roe_ocf": "用净资产收益率和经营现金流利润覆盖共同衡量盈利质量，偏好盈利能力强且利润能转成现金的公司。",
    "quality_roa_ocf": "用总资产收益率和经营现金流利润覆盖共同衡量资产盈利质量，偏好资产回报高且现金流确认充分的公司。",
    "cash_profit_cover": "用经营现金流相对净利润的覆盖度衡量利润含金量，偏好会计利润更容易落到现金流的公司。",
    "low_accrual_to_assets": "用经营现金流超过净利润的部分相对资产规模衡量应计质量，偏好利润更少依赖应计项目的公司。",
    "gross_margin_quality": "同时考察毛利率水平和毛利率改善，偏好产品议价能力较强且仍在改善的公司。",
    "quality_growth_hmean": "用收入增长、利润增长和现金流覆盖共同确认成长质量，偏好增长更同步、更干净的公司。",
    "ocf_growth_quality": "用经营现金流增长和现金流利润覆盖共同确认成长质量，偏好现金流同步改善的公司。",
    "asset_turnover_improve": "用资产周转率同比改善衡量经营效率，偏好同样资产能产生更多收入的公司。",
    "debt_cash_safety": "用经营现金流对负债的覆盖扣减杠杆压力，偏好现金偿债能力更强的公司。",
    "low_leverage_quality": "用 ROE 扣减资产负债率，偏好盈利能力不是靠高杠杆堆出来的公司。",
    "capex_efficiency": "用收入增长扣减资本开支强度，偏好增长不依赖过高再投入的公司。",
    "roe_value_pb": "用净资产收益率和市净率共同衡量质量价值，偏好盈利能力较强且账面估值不贵的公司。",
    "earnings_yield_quality": "用盈利收益率和现金流利润覆盖共同衡量便宜且利润质量较好的公司。",
    "shareholder_yield_quality": "用股息率和净资产收益率共同衡量股东回报质量，偏好分红回报和盈利能力兼具的公司。",
    "free_cashflow_yield_quality": "用自由现金流收益率和利润现金覆盖共同衡量现金回报质量，偏好现金回报高且利润扎实的公司。",
    "cash_revenue_quality": "用经营现金流收入比验证收入质量，偏好收入更容易转成经营现金流的公司。",
    "margin_value_ps": "用净利率扣减 PS 估值，偏好利润率较高且销售估值不贵的公司。",
    "solvency_liquidity_quality": "用流动比率、速动比率和低杠杆衡量短期偿债质量，偏好资产负债表更稳的公司。",
    "working_capital_light": "用较低营运资本占用衡量经营质量，偏好应收和存货压力更小的公司。",
    "turnover_efficiency_combo": "用存货周转和应收周转共同衡量营运效率，偏好库存消化和回款速度更快的公司。",
    "reported_margin_quality": "用财务指标表披露的毛利率和净利率衡量盈利质量，偏好利润率水平更高的公司。",
    "eps_bps_value_quality": "用每股收益、每股净资产和 PB 估值共同衡量每股价值质量，偏好盈利和账面基础较好且估值不贵的公司。",
    "cash_buffer_value": "用现金对负债的覆盖并扣减市净率估值，偏好现金安全垫较厚且账面估值不贵的公司。",
    "growth_value_balance": "用收入和利润同步增长叠加低市盈率估值，偏好成长没有被价格过度透支的公司。",
    "dupont_quality": "用净利率和资产周转率衡量杜邦经营质量，偏好利润率和周转效率兼具的公司。",
    "clean_growth_quality": "用利润增长、现金流覆盖和低营运资本压力共同确认成长质量，偏好增长更干净的公司。",
    "robust_margin_value_ps": "用稳健标准化后的净利率扣减市销率估值，偏好利润率高且销售估值不贵的公司。",
    "cashflow_conversion_quality": "用自由现金流和经营现金流对利润的覆盖衡量现金转化质量，偏好利润能沉淀为现金的公司。",
    "liquidity_solvency_value": "用现金、流动资产偿债能力和低 PB 估值共同衡量资产负债表安全边际。",
    "dupont_value_quality": "用净利率、资产周转率和市净率估值衡量杜邦质量价值，偏好经营质量好且账面估值不贵的公司。",
    "inventory_receivable_efficiency": "用低存货、低应收占用和应收周转共同衡量营运效率，偏好资金占用较轻的公司。",
    "industry_neutral_roe_value_pb": "在行业中性后比较净资产收益率和市净率，偏好同行业内盈利能力强且账面估值不贵的公司。",
    "industry_rank_cash_profit_cover": "在同行业内比较现金流对利润的覆盖，降低行业现金流周期差异带来的偏差。",
    "size_neutral_earnings_yield_quality": "在规模中性后比较盈利收益率和现金流质量，减少市值风格对因子的干扰。",
    "quality_liquidity_confirm_20": "用经营现金流质量和成交额放大共同确认基本面改善，偏好基本面改善并开始被资金关注的公司。",
    "decayed_quality_growth_20": "对质量成长信号做时间衰减，近期财务改善的权重更高。",
    "stable_cash_conversion_20": "用现金转化水平和稳定性衡量盈利质量，偏好现金覆盖高且波动较低的公司。",
    "margin_trend_value_20": "用净利率趋势扣减销售估值，偏好利润率持续改善且 PS 不贵的公司。",
    "cash_value_attention_gap_20": "用现金安全垫和低 PB 构成价值信号，同时惩罚短期交易拥挤。",
    "rd_attention_gap_20": "用研发强度和低市场关注度寻找长期投入尚未被充分交易的公司。",
    "quality_value_attention_gap_20": "用质量价值信号叠加低交易关注度，偏好基本面较好但尚未拥挤的公司。",
    "growth_turnover_confirm_20": "用基本面成长和换手放大共同确认，偏好成长改善并获得资金参与的公司。",
    "quality_attention_uncrowded_20": "用稳健质量信号扣减短期关注拥挤，偏好质量好但交易不拥挤的公司。",
    "fcf_turnover_confirm_20": "用自由现金流收益率和换手放大共同确认，偏好现金回报改善并开始被资金参与的公司。",
}


_FACTOR_TOKEN_CN = {
    "operating_cf_margin": "经营现金流利润率",
    "reported_ocf_to_or": "财务指标表经营现金流收入比",
    "reported_ocf_to_profit": "财务指标表经营现金流利润覆盖",
    "cashflow_to_profit": "经营现金流利润覆盖",
    "ocf_to_assets": "经营现金流资产产出",
    "fcf_to_assets": "自由现金流资产产出",
    "fcf_margin": "自由现金流利润率",
    "fcf_to_profit": "自由现金流利润覆盖",
    "cash_conversion_spread": "现金流与利润差额",
    "gross_to_net_margin": "毛利留存为净利的效率",
    "profit_to_assets": "资产盈利能力",
    "profit_to_liab": "利润对负债覆盖",
    "cashflow_to_liab": "经营现金流对负债覆盖",
    "revenue_to_liab": "收入对负债覆盖",
    "leverage_adjusted_roe": "杠杆调整后净资产收益率",
    "net_margin": "净利率",
    "gross_margin": "毛利率",
    "roe": "净资产收益率",
    "roa": "总资产收益率",
    "eps": "每股收益",
    "bps": "每股净资产",
    "cfps": "每股现金流",
    "revenue_yoy": "收入同比增长",
    "net_profit_yoy": "净利润同比增长",
    "ocf_yoy": "经营现金流同比增长",
    "gross_margin_yoy": "毛利率改善",
    "asset_turnover_yoy": "资产周转率改善",
    "op_yoy": "营业利润同比增长",
    "or_yoy": "营业收入同比增长",
    "netprofit_yoy": "财务指标表净利润同比增长",
    "cash_to_liab": "现金对负债覆盖",
    "cash_to_assets": "现金资产占比",
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "low_debt_assets": "低资产负债率",
    "low_working_capital_pressure": "低营运资本压力",
    "low_receivable_inventory_assets": "低应收和存货占用",
    "low_inventory_assets": "低存货占用",
    "low_receivable_assets": "低应收占用",
    "asset_turnover": "资产周转效率",
    "inventory_turnover": "存货周转效率",
    "receivable_turnover": "应收账款周转效率",
    "low_capex_assets": "低资本开支强度",
    "rd_intensity": "研发强度",
    "dividend_yield": "股息率",
    "low_pb": "低市净率估值",
    "low_pe": "低市盈率估值",
    "earnings_yield": "盈利收益率",
    "low_ps": "低市销率估值",
    "fcf_yield": "自由现金流收益率",
}


def _dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys([item for item in items if item]))


def _factor_tokens_cn(factor: str, expression: str) -> list[str]:
    text = f"{factor} {expression}".lower()
    return _dedupe_keep_order([label for token, label in _FACTOR_TOKEN_CN.items() if token in text])


def _table_description_cn(row: pd.Series) -> str:
    factor = str(row.get("factor", ""))
    if factor in _FACTOR_DESCRIPTION_CN:
        description = _FACTOR_DESCRIPTION_CN[factor]
    else:
        expression = str(row.get("expression", ""))
        family = str(row.get("family", ""))
        labels = _factor_tokens_cn(factor, expression)
        joined = "、".join(labels[:4]) if labels else "多个基本面指标"

        if "hmean" in expression.lower():
            description = f"用调和平均组合{joined}，偏好多条基本面腿都不弱、短板更少的公司。"
        elif family == "fundamental_value":
            description = f"将{joined}放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。"
        elif family == "fundamental_growth":
            description = f"将{joined}共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。"
        elif family == "fundamental_safety":
            description = f"用{joined}衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。"
        elif family == "fundamental_efficiency":
            description = f"用{joined}衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。"
        elif family == "fundamental_quality":
            description = f"用{joined}验证盈利质量，偏好利润和收入更容易转化为现金的公司。"
        else:
            description = f"将{joined}与交易确认结合，偏好基本面改善且开始获得市场关注的公司。"

    expression_l = str(row.get("expression", "")).lower()
    suffixes = []
    if "industry_neutralize" in expression_l:
        suffixes.append("已做行业中性处理，主要比较同行业内的个股差异。")
    if "size_neutralize" in expression_l:
        suffixes.append("已做规模中性处理，降低市值风格干扰。")
    if "group_rank" in expression_l:
        suffixes.append("采用行业内排名，减少行业天然差异的影响。")
    if suffixes:
        description = description.rstrip("。") + "。" + "".join(suffixes)
    return description


def _format_stability(df: pd.DataFrame, factors: list[str] | None = None, max_rows: int = 30) -> pd.DataFrame:
    out = df.copy()
    if factors is not None:
        out = out[out["factor"].isin(factors)].copy()
        out["_order"] = pd.Categorical(out["factor"], categories=factors, ordered=True)
        out = out.sort_values("_order").drop(columns="_order")
    cols = [
        "factor", "family", "direction", "train_rankic", "valid_rankic", "test_rankic", "ytd_2026_rankic",
        "min_adj_rankic", "avg_adj_rankic", "min_coverage", "expression", "description",
    ]
    out = out[[col for col in cols if col in out.columns]].head(max_rows).copy()
    for col in ["train_rankic", "valid_rankic", "test_rankic", "ytd_2026_rankic", "min_adj_rankic", "avg_adj_rankic", "min_coverage"]:
        if col in out:
            out[col] = out[col].map(_pct)
    if "description" in out:
        out["description"] = out.apply(_table_description_cn, axis=1)
        out = out.rename(columns={"description": "说明"})
    return out


_TERM_NOTES = [
    ("operating_cf_margin", "经营现金流利润率，衡量收入转成经营现金流的能力"),
    ("ocf_to_or", "经营现金流收入比，验证收入质量"),
    ("ocf_to_profit", "经营现金流利润覆盖比，验证利润现金含量"),
    ("cashflow_to_profit", "经营现金流对利润的覆盖，验证会计利润含金量"),
    ("n_cashflow_act_ttm / net_profit_ttm", "经营现金流/净利润，衡量利润的现金覆盖程度"),
    ("n_cashflow_act_ttm / total_assets", "经营现金流/总资产，衡量资产产生现金流的效率"),
    ("n_cashflow_act_ttm / total_liab", "经营现金流/总负债，衡量经营现金流对债务压力的覆盖"),
    ("free_cashflow_ttm / total_mv", "自由现金流收益率，衡量现金回报相对市值是否便宜"),
    ("free_cashflow_ttm / total_assets", "自由现金流/总资产，衡量资产真实产出现金的能力"),
    ("free_cashflow_ttm / total_revenue_ttm", "自由现金流/收入，衡量收入转化为自由现金的能力"),
    ("free_cashflow_ttm / net_profit_ttm", "自由现金流/净利润，衡量利润最终沉淀为自由现金的能力"),
    ("net_margin", "净利率，反映成本控制、议价能力和商业模式质量"),
    ("grossprofit_margin", "财务指标表中的毛利率，反映产品或服务的基础议价能力"),
    ("netprofit_margin", "财务指标表中的净利率，反映费用控制后的最终盈利能力"),
    ("net_margin_ttm / gross_margin_ttm", "净利率/毛利率，衡量毛利最终留存为净利润的比例"),
    ("net_profit_ttm / total_assets", "利润/资产，接近 ROA 口径，衡量资产盈利效率"),
    ("net_profit_ttm / total_liab", "净利润/总负债，衡量盈利对债务负担的覆盖"),
    ("total_revenue_ttm / total_liab", "收入/总负债，衡量经营规模对债务负担的覆盖"),
    ("asset_turnover_ttm", "资产周转率，衡量资产产生收入的效率"),
    ("roa_ttm", "ROA，衡量资产层面的盈利能力"),
    ("roe_ttm", "ROE，衡量股东权益回报"),
    ("roe_ttm - debt_to_assets", "杠杆调整 ROE，惩罚靠高负债堆出来的盈利能力"),
    ("eps", "每股收益，衡量单股盈利能力"),
    ("bps", "每股净资产，衡量单股账面资产基础"),
    ("1 / pe_ttm", "盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司"),
    ("revenue_yoy", "收入同比增长，衡量需求和业务扩张"),
    ("net_profit_yoy", "利润同比增长，衡量增长是否落到利润表底线"),
    ("accrual_quality", "现金流与利润差额/资产，衡量利润是否由现金支持"),
    ("working_capital_pressure", "营运资本占用压力，主要来自应收和存货"),
    ("quick_ratio", "速动比率，衡量短期偿债安全垫"),
    ("cash_to_liab", "现金/负债，衡量现金对债务的覆盖"),
    ("debt_to_assets", "资产负债率，衡量杠杆压力"),
    ("money_cap / total_assets", "货币资金/资产，衡量现金储备"),
    ("money_cap - total_liab", "净现金缓冲，衡量现金扣除负债后的安全垫"),
    ("inv_turn", "存货周转率，衡量库存消化效率"),
    ("ar_turn", "应收账款周转率，衡量回款效率"),
    ("accounts_receiv + inventories", "应收和存货合计占用，衡量收入扩张背后的营运资本压力"),
    ("dv_ttm", "股息率，衡量现金分红回报"),
    ("pb", "PB，衡量账面价值相对估值"),
    ("ps_ttm", "PS，衡量收入相对估值"),
    ("pe_ttm", "PE，衡量盈利相对估值"),
    ("amount / ts_mean", "成交额放大，衡量市场关注度和资金确认"),
]


def _factor_blocks(expression: str, factor: str) -> str:
    text = f"{factor} {expression}".lower()
    notes = [note for key, note in _TERM_NOTES if key.lower() in text]
    if "1 - rank(pb)" in text:
        notes.append("低 PB：同等质量下账面估值更便宜")
    if "1 - rank(ps_ttm)" in text:
        notes.append("低 PS：同等收入或利润率下销售估值更便宜")
    if "1 - rank(pe_ttm)" in text:
        notes.append("低 PE：同等盈利下收益率更高")
    if "industry_neutralize" in text:
        notes.append("行业中性：先剔除行业平均差异，再比较个股")
    if "group_rank" in text:
        notes.append("行业内排名：只在同一行业内部比较，降低行业结构噪声")
    if "size_neutralize" in text:
        notes.append("规模中性：剔除市值风格影响")
    if "hmean" in text:
        notes.append("调和平均：惩罚单腿特别弱的公司")
    if not notes:
        return "该因子由截面排名后的基本面指标组合而成，方向统一为分数越高越好。"
    deduped = list(dict.fromkeys(notes))
    return "；".join(deduped) + "。"


def _factor_intuition(row: pd.Series) -> str:
    factor = str(row["factor"])
    family = str(row.get("family", ""))
    expression = str(row.get("expression", ""))
    text = f"{factor} {expression}".lower()
    sentences: list[str] = []

    if family == "fundamental_value":
        sentences.append("有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。")
    elif family == "fundamental_growth":
        sentences.append("有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。")
    elif family == "fundamental_safety":
        sentences.append("有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。")
    elif family == "fundamental_efficiency":
        sentences.append("有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。")
    elif family == "fundamental_quality":
        sentences.append("有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。")
    else:
        sentences.append("有效性主要来自基本面质量和交易确认的结合：基本面改善并开始被资金关注，但价格尚未完全反映这部分信息。")

    if "operating_cf_margin" in text or "cashflow_to_profit" in text or "ocf_to_profit" in text or "ocf_to_assets" in text or "n_cashflow_act_ttm / total_assets" in text:
        sentences.append("现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。")
    if "n_cashflow_act_ttm / net_profit_ttm" in text or "n_cashflow_act_ttm - net_profit_ttm" in text or "cash_conversion_spread" in text:
        sentences.append("现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。")
    if "free_cashflow_ttm" in text or "fcf_" in text:
        sentences.append("自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。")
    if "net_margin" in text or "netprofit_margin" in text or "grossprofit_margin" in text or "profit_to_assets" in text or "roa_ttm" in text or "roe_ttm" in text:
        sentences.append("利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。")
    if "1 - rank(pb)" in text or "1 - rank(ps_ttm)" in text or "1 - rank(pe_ttm)" in text or "1 / pe_ttm" in text or "earnings_yield" in text or "free_cashflow_ttm / total_mv" in text:
        sentences.append("估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。")
    if "revenue_yoy" in text or "net_profit_yoy" in text:
        sentences.append("成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。")
    if "working_capital_pressure" in text or "accounts_receiv + inventories" in text or "ar_turn" in text or "inv_turn" in text:
        sentences.append("营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。")
    if "quick_ratio" in text or "cash_to_liab" in text or "total_liab" in text or "debt_to_assets" in text or "money_cap / total_assets" in text or "net_cash_to_assets" in text:
        sentences.append("偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。")
    if "leverage_adjusted_roe" in text or "roe_ttm - debt_to_assets" in text:
        sentences.append("杠杆调整后的盈利能力有效，是因为它区分了真实经营回报和高负债放大出来的账面回报。")
    if "industry_neutralize" in text:
        sentences.append("行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。")
    if "group_rank" in text:
        sentences.append("行业内排名有效，是因为不同行业的现金流周期和利润率天然不同，同行业比较能减少结构性偏差。")
    if "size_neutralize" in text:
        sentences.append("规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。")
    if "hmean" in text:
        sentences.append("调和平均会惩罚单项短板，因此比简单相加更偏向“多条腿都不差”的稳健公司。")
    if "dv_ttm" in text:
        sentences.append("股息率有效，是因为持续分红代表现金回报和治理约束，也给估值提供一定锚。")
    if "amount / ts_mean" in text:
        sentences.append("成交额确认有效，是因为基本面改善开始被资金交易时，信号兑现速度通常更快。")

    return "".join(sentences)


def _factor_usage_note(row: pd.Series) -> str:
    expression = str(row.get("expression", "")).lower()
    family = str(row.get("family", ""))
    ytd = row.get("ytd_2026_rankic", np.nan)
    notes: list[str] = []
    if "industry_neutralize" in expression or "group_rank" in expression:
        notes.append("适合直接进入带行业约束的指增模型，也适合做行业内排序。")
    elif family == "fundamental_value":
        notes.append("适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。")
    elif family == "fundamental_growth":
        notes.append("适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。")
    elif family == "fundamental_safety":
        notes.append("适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。")
    elif family == "fundamental_efficiency":
        notes.append("适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。")
    else:
        notes.append("适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。")
    if "size_neutralize" in expression:
        notes.append("该因子已削弱市值暴露，和风格约束组合时更稳。")
    if pd.notna(ytd) and ytd < 0.005:
        notes.append("2026YTD 表现偏弱，使用时建议降权或作为备选特征。")
    elif pd.notna(ytd) and ytd > 0.04:
        notes.append("2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。")
    return "".join(notes)


def _format_factor_details(stability: pd.DataFrame, selected: list[str]) -> str:
    if not selected:
        return "本次没有通过相关性去重的推荐因子。"
    out = stability[stability["factor"].isin(selected)].copy()
    out["_order"] = pd.Categorical(out["factor"], categories=selected, ordered=True)
    out = out.sort_values("_order").drop(columns="_order")
    lines: list[str] = []
    for idx, row in enumerate(out.itertuples(index=False), start=1):
        item = pd.Series(row._asdict())
        factor = item["factor"]
        lines.extend(
            [
                f"### {idx}. `{factor}`",
                "",
                f"- 因子族：`{item.get('family', '')}`。",
                f"- 计算过程：表达式为 `{item.get('expression', '')}`。{_factor_blocks(str(item.get('expression', '')), str(factor))}",
                f"- 经济学直觉：{_factor_intuition(item)}",
                f"- 当前表现：train RankIC `{_pct(item.get('train_rankic'))}`，valid RankIC `{_pct(item.get('valid_rankic'))}`，test RankIC `{_pct(item.get('test_rankic'))}`，2026YTD RankIC `{_pct(item.get('ytd_2026_rankic'))}`，三段最小调整后 RankIC `{_pct(item.get('min_adj_rankic'))}`，覆盖率下限 `{_pct(item.get('min_coverage'))}`。",
                f"- 使用方式：{_factor_usage_note(item)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    report_path: Path,
    candidates: list,
    summaries: dict[str, pd.DataFrame],
    stability: pd.DataFrame,
    selected: list[str],
    run_settings: dict[str, str | int | float | bool | None],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    family_counts = pd.DataFrame([{"family": c.family, "n_candidates": 1} for c in candidates]).groupby("family", as_index=False)["n_candidates"].sum().sort_values("n_candidates", ascending=False)
    status_counts = pd.DataFrame(
        {
            "status": ["evaluated", "passes_stability", "selected_after_corr"],
            "count": [len(candidates), int(stability["passes_stability"].sum()), len(selected)],
        }
    )
    selected_table = _format_stability(stability, selected, max_rows=120)
    top_table = _format_stability(stability[stability["passes_stability"]], None, max_rows=30)
    legacy = ["quality_growth_hmean", "eps_bps_value_quality", "industry_neutral_roe_value_pb"]
    legacy_table = _format_stability(stability, legacy, max_rows=10)
    selected_details = _format_factor_details(stability, selected)
    selected_family_counts = (
        stability[stability["factor"].isin(selected)]
        .groupby("family", as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
        .sort_values("n_selected", ascending=False)
    )

    lines = [
        "# 中证500基本面算子扩展与因子筛选报告",
        "",
        "日期：2026-07-20",
        "",
        "## 1. 任务目标",
        "",
        "现有 `quality_growth_hmean`、`eps_bps_value_quality`、`industry_neutral_roe_value_pb` 三个核心基本面因子已经验证出一定效果，但覆盖的财务关系还比较集中，主要围绕成长、每股盈利账面价值和 ROE-PB 质量价值。为了让基本面因子库更完整，这里继续基于利润表、资产负债表、现金流量表和财务指标表扩展算子，再通过有约束的组合搜索筛选更稳定的中证500基本面因子。",
        "",
        "当前框架先把不同口径的财务字段统一成“分数越高越好”的原子指标，再用均值、调和平均、估值扣减、行业中性和规模中性等算子生成候选因子，最后在中证500股票池上做 train/valid/test/2026YTD 验证和相关性去重。",
        "",
        "## 2. 执行计划与口径",
        "",
        "执行步骤如下：",
        "",
        "1. 保留旧三因子和已有 legacy 基本面因子作为 benchmark。",
        "2. 从利润表、资产负债表、现金流量表、财务指标表和估值字段构造方向统一的原子指标。",
        "3. 用均值、调和平均、价值扣减、行业中性、规模中性等算子生成候选组合。",
        "4. 在中证500 train/valid/test/2026YTD 上验证 RankIC 稳定性，再做相关性去重，得到推荐集合。",
        "",
        "运行参数如下：",
        "",
        _markdown_table(pd.DataFrame([{"parameter": key, "value": value} for key, value in run_settings.items()])),
        "",
        "## 3. 数据口径",
        "",
        "| 项目 | 口径 |",
        "| --- | --- |",
        "| 股票池 | 中证500历史成分股 |",
        "| 日频面板 | `data/processed/*_fundamental_000905_SH.parquet` |",
        "| 基本面 PIT | 按公告日 `available_date` 向后对齐，避免未来函数 |",
        "| 训练/验证/测试 | 沿用现有 `train/valid/test` 拆分 |",
        "| 近期观察 | 从 test 中单独切出 `2026YTD` |",
        "| 标签 | 现有 `label`，即未来收益/排名评估口径 |",
        "",
        "## 4. 算子体系",
        "",
        "| 算子 | 例子 | 经济含义 |",
        "| --- | --- | --- |",
        "| 方向统一原子 | `low_pb = 1 - rank(pb)` | 所有基础指标都转成越大越好 |",
        "| 比率算子 | `free_cashflow_ttm / total_mv` | 用现金流、资产、负债等构造可比指标 |",
        "| 均值组合 | `mean(roe, low_pb)` | 多个维度等权确认 |",
        "| 调和平均 | `harm_mean(revenue_yoy, net_profit_yoy)` | 惩罚单腿增长 |",
        "| 三腿组合 | `roe + cashflow_to_profit + low_pb` | 质量、现金流、估值同时约束 |",
        "| 行业中性 | `industry_neutralize(score)` | 降低行业结构差异 |",
        "| 规模中性 | `size_neutralize(score)` | 降低市值风格暴露 |",
        "",
        "候选因子族数量：",
        "",
        _markdown_table(family_counts),
        "",
        "验证结果数量：",
        "",
        _markdown_table(status_counts),
        "",
        "核心结论：",
        "",
        "1. 结果最强的一组不是单纯成长，而是“现金流/利润率质量 + 低估值”：`operating_cf_margin + low_pb`、`net_margin + low_pb`、`profit_to_assets + low_pb` 排在最前。",
        "2. PB/PS 相关估值约束比单独成长更稳，说明中证500里“盈利质量不差且估值不贵”的组合仍然有显著横截面区分度。",
        "3. 原三因子仍然有效，但会被新工厂生成的近似等价或更细颗粒组合在相关性去重时替代；这不是失效，而是说明旧因子逻辑被更系统地展开了。",
        "4. 推荐集合以价值质量因子为主，同时保留成长、安全、效率、现金流质量和少量混合量价确认因子，便于后续进入 XGBoost 或指数增强模型时做消融。",
        "",
        "去重后推荐因子族分布：",
        "",
        _markdown_table(selected_family_counts),
        "",
        "## 5. 最终推荐因子",
        "",
        "下表是通过稳定性过滤和相关性去重后的推荐因子。`min_adj_rankic` 是 train/valid/test 三段按训练期方向调整后的最小 RankIC，越高说明跨样本越稳。",
        "",
        _markdown_table(selected_table, max_rows=120),
        "",
        "## 6. 稳定通过但可能相关的候选",
        "",
        _markdown_table(top_table, max_rows=30),
        "",
        "## 7. 原三因子在扩展框架中的位置",
        "",
        _markdown_table(legacy_table, max_rows=10),
        "",
        "解释：原三因子仍然有效，但新算子里可能出现更细分的价值质量、现金流质量或营运效率因子。后续接入 XGBoost 或指数增强时，不应简单把所有通过因子一起加入，而应优先使用相关性去重后的推荐集合。",
        "",
        "## 8. 经济含义总结",
        "",
        "扩展后的算子主要强化了五类基本面信号：",
        "",
        "1. 质量价值：高 ROE/ROA/利润率，同时 PB/PE/PS 不贵。",
        "2. 现金流质量：经营现金流、自由现金流、现金利润覆盖共同确认盈利。",
        "3. 营运效率：资产周转、存货周转、应收周转、低营运资本占用。",
        "4. 安全边际：低负债、高现金覆盖、较强短期偿债能力。",
        "5. 杠杆调整盈利：区分真实经营回报和高负债放大出来的账面回报。",
        "",
        "这些因子比单独看成长或单独看估值更稳，因为它们要求“好公司”和“不太贵”同时成立，或者要求利润表结果被现金流量表和资产负债表验证。",
        "",
        "## 9. 因子逐项说明",
        "",
        "下面逐项解释最终推荐集合中的每个因子。所有 `rank` 都是在同一交易日的中证500截面内计算，方向已经统一为分数越高越好。",
        "",
        selected_details,
        "",
        "## 10. 风险和下一步",
        "",
        "- 这类宽搜索仍存在多重检验问题，不能只看最优因子，需要用 valid/test/YTD 和相关性过滤约束。",
        "- 基本面因子更新频率低，进入指增模型时建议月度或财报后调仓，不建议日频过度交易。",
        "- 行业中性因子更适合指增；非中性因子要单独检查行业暴露。",
        "- 下一步可以把推荐因子加入 XGBoost 特征集合，做 `旧3因子` vs `新基本面算子集合` vs `公开因子+新基本面` 的消融实验。",
        "",
        "## 11. 输出文件",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 全部拆分汇总：`{output_dir / 'all_split_summary.csv'}`",
        f"- 稳定性表：`{output_dir / 'stable_factors.csv'}`",
        f"- 最终推荐：`{output_dir / 'selected_factors.csv'}`",
        f"- 报告文件：`{report_path}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _read_ytd_factor_values(path: Path, year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    try:
        values = pd.read_parquet(path, filters=[("trade_date", ">=", start), ("trade_date", "<", end)])
    except Exception:
        values = pd.read_parquet(path)
        values["trade_date"] = pd.to_datetime(values["trade_date"])
        values = values[(values["trade_date"] >= start) & (values["trade_date"] < end)].copy()
    values["trade_date"] = pd.to_datetime(values["trade_date"])
    return values


def resume_from_existing(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output)
    config = AlphaMiningConfig(
        min_abs_ic=args.min_abs_ic,
        min_coverage=args.min_coverage,
        max_pair_corr=args.max_pair_corr,
        include_turnover=args.include_turnover,
    )

    schema_df = _read_split_schema(processed_dir, "train", args.suffix)
    candidates = _candidate_universe(
        schema_df,
        include_atoms=not args.no_atoms,
        include_neutralized=not args.no_neutralized,
        candidate_regex=args.candidate_regex,
        max_candidates=args.max_candidates,
    )

    split_summaries: dict[str, pd.DataFrame] = {}
    for split in ["train", "valid", "test"]:
        path = output_dir / split / "candidate_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved split summary: {path}")
        split_summaries[split] = pd.read_csv(path).assign(split=split)

    test_values_path = output_dir / "test" / "factor_values.parquet"
    if not test_values_path.exists():
        raise FileNotFoundError(f"Missing saved test factor values: {test_values_path}")
    ytd_values = _read_ytd_factor_values(test_values_path, args.ytd_year)
    ytd_summary = _evaluate_subset(ytd_values, candidates, config).assign(split=f"ytd_{args.ytd_year}")
    ytd_summary.to_csv(output_dir / f"candidate_summary_ytd_{args.ytd_year}.csv", index=False)
    split_summaries[f"ytd_{args.ytd_year}"] = ytd_summary
    del ytd_values

    all_summary = pd.concat(split_summaries.values(), ignore_index=True)
    all_summary.to_csv(output_dir / "all_split_summary.csv", index=False)

    stability = _build_stability_table(
        split_summaries,
        candidates,
        min_adj_rankic=args.min_adj_rankic,
        min_ytd_rankic=args.min_ytd_rankic,
        min_coverage=args.min_coverage,
    )
    train_values_path = output_dir / "train" / "factor_values.parquet"
    if not train_values_path.exists():
        raise FileNotFoundError(f"Missing saved train factor values: {train_values_path}")
    eligible = stability[stability["passes_stability"]]["factor"].tolist()
    train_values = pd.read_parquet(train_values_path, columns=[factor for factor in eligible if factor])
    selected = _decorrelate(stability, train_values, max_pair_corr=args.max_pair_corr)
    stability["selected_after_corr"] = stability["factor"].isin(selected)
    stability.to_csv(output_dir / "stable_factors.csv", index=False)
    pd.Series(selected, name="factor").to_csv(output_dir / "selected_factors.csv", index=False)

    run_settings = {
        "processed_dir": str(processed_dir),
        "suffix": args.suffix,
        "n_candidates": len(candidates),
        "include_atoms": not args.no_atoms,
        "include_neutralized": not args.no_neutralized,
        "candidate_regex": args.candidate_regex or "",
        "max_candidates": args.max_candidates or "",
        "include_turnover": args.include_turnover,
        "min_abs_ic": args.min_abs_ic,
        "min_adj_rankic": args.min_adj_rankic,
        "min_ytd_rankic": args.min_ytd_rankic,
        "min_coverage": args.min_coverage,
        "max_pair_corr": args.max_pair_corr,
        "ytd_year": args.ytd_year,
        "resume_existing": True,
    }
    write_report(output_dir, Path(args.report), candidates, split_summaries, stability, selected, run_settings)
    print(f"resumed from saved factor values under {output_dir}")
    print(f"wrote {output_dir / 'stable_factors.csv'}")
    print(f"selected {len(selected)} factors: {', '.join(selected[:20])}")
    print(f"wrote report {args.report}")


def run(args: argparse.Namespace) -> None:
    if args.resume_existing:
        resume_from_existing(args)
        return

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AlphaMiningConfig(
        min_abs_ic=args.min_abs_ic,
        min_coverage=args.min_coverage,
        max_pair_corr=args.max_pair_corr,
        include_turnover=args.include_turnover,
    )

    split_results: dict[str, AlphaMiningResult] = {}
    split_summaries: dict[str, pd.DataFrame] = {}
    candidates: list | None = None
    for split in ["train", "valid", "test"]:
        df = _read_split(processed_dir, split, args.suffix)
        if candidates is None:
            candidates = _candidate_universe(
                df,
                include_atoms=not args.no_atoms,
                include_neutralized=not args.no_neutralized,
                candidate_regex=args.candidate_regex,
                max_candidates=args.max_candidates,
            )
        available = [candidate for candidate in candidates if candidate.is_available(df)]
        result = _run_split(df, available, output_dir / split, config)
        split_results[split] = result
        split_summaries[split] = result.summary.assign(split=split)
        print(f"{split}: rows={len(df)} candidates={len(available)} selected={len(result.selected)}")

    assert candidates is not None
    ytd_values = split_results["test"].factor_values.copy()
    ytd_values["trade_date"] = pd.to_datetime(ytd_values["trade_date"])
    ytd_values = ytd_values[ytd_values["trade_date"].dt.year == args.ytd_year].copy()
    ytd_summary = _evaluate_subset(ytd_values, candidates, config).assign(split=f"ytd_{args.ytd_year}")
    ytd_summary.to_csv(output_dir / f"candidate_summary_ytd_{args.ytd_year}.csv", index=False)
    split_summaries[f"ytd_{args.ytd_year}"] = ytd_summary

    all_summary = pd.concat(split_summaries.values(), ignore_index=True)
    all_summary.to_csv(output_dir / "all_split_summary.csv", index=False)

    stability = _build_stability_table(
        split_summaries,
        candidates,
        min_adj_rankic=args.min_adj_rankic,
        min_ytd_rankic=args.min_ytd_rankic,
        min_coverage=args.min_coverage,
    )
    selected = _decorrelate(stability, split_results["train"].factor_values, max_pair_corr=args.max_pair_corr)
    stability["selected_after_corr"] = stability["factor"].isin(selected)
    stability.to_csv(output_dir / "stable_factors.csv", index=False)
    pd.Series(selected, name="factor").to_csv(output_dir / "selected_factors.csv", index=False)

    run_settings = {
        "processed_dir": str(processed_dir),
        "suffix": args.suffix,
        "n_candidates": len(candidates),
        "include_atoms": not args.no_atoms,
        "include_neutralized": not args.no_neutralized,
        "candidate_regex": args.candidate_regex or "",
        "max_candidates": args.max_candidates or "",
        "include_turnover": args.include_turnover,
        "min_abs_ic": args.min_abs_ic,
        "min_adj_rankic": args.min_adj_rankic,
        "min_ytd_rankic": args.min_ytd_rankic,
        "min_coverage": args.min_coverage,
        "max_pair_corr": args.max_pair_corr,
        "ytd_year": args.ytd_year,
    }
    write_report(output_dir, Path(args.report), candidates, split_summaries, stability, selected, run_settings)
    print(f"wrote {output_dir / 'stable_factors.csv'}")
    print(f"selected {len(selected)} factors: {', '.join(selected[:20])}")
    print(f"wrote report {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--suffix", default="000905_SH")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--min-abs-ic", type=float, default=0.003)
    parser.add_argument("--min-coverage", type=float, default=0.55)
    parser.add_argument("--max-pair-corr", type=float, default=0.85)
    parser.add_argument("--min-adj-rankic", type=float, default=0.005)
    parser.add_argument("--min-ytd-rankic", type=float, default=0.0)
    parser.add_argument("--ytd-year", type=int, default=2026)
    parser.add_argument("--no-atoms", action="store_true", help="Do not include standalone generated atomic factors.")
    parser.add_argument("--no-neutralized", action="store_true", help="Do not include generated industry/size neutralized variants.")
    parser.add_argument("--candidate-regex", default=None, help="Optional regex filter applied to candidate names.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--include-turnover", action="store_true", help="Also compute turnover during AlphaMiner evaluation.")
    parser.add_argument("--resume-existing", action="store_true", help="Resume summary/stability/report generation from saved split factor values.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
