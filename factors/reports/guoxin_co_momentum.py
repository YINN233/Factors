"""Reproduce Guosen co-momentum factors with CITIC industry index returns."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors.data.fetcher import (
    RAW_DIR,
    fetch_adj_factor,
    fetch_daily,
    fetch_index_weight,
    get_tushare_pro,
)


CITIC_LEVEL1_CODES = {f"CI005{i:03d}.CI" for i in range(1, 31)}
DEFAULT_OUTPUT = Path("outputs/guoxin_co_momentum_reproduction")
DEFAULT_REPORT = Path("docs/guoxin_co_momentum_reproduction_2026-07-13.md")

PDF_REFERENCE = pd.DataFrame(
    [
        {"factor": "IMAX20", "rankic_mean": 0.0376, "icir_annual": 3.23, "top_excess": 0.0061, "bottom_excess": -0.0046},
        {"factor": "ICM", "rankic_mean": 0.0559, "icir_annual": 3.85, "top_excess": 0.0063, "bottom_excess": -0.0102},
        {"factor": "VICM", "rankic_mean": 0.0590, "icir_annual": 3.90, "top_excess": 0.0068, "bottom_excess": -0.0109},
        {"factor": "ICR", "rankic_mean": -0.0560, "icir_annual": -3.39, "top_excess": 0.0060, "bottom_excess": -0.0113},
        {"factor": "VICR", "rankic_mean": -0.0593, "icir_annual": -3.55, "top_excess": 0.0059, "bottom_excess": -0.0123},
        {"factor": "CMC", "rankic_mean": 0.0602, "icir_annual": 3.86, "top_excess": 0.0066, "bottom_excess": -0.0115},
        {"factor": "MCMC", "rankic_mean": 0.0640, "icir_annual": 4.00, "top_excess": 0.0053, "bottom_excess": -0.0122},
    ]
)


@dataclass(frozen=True)
class CoMomentumConfig:
    start: str = "20100101"
    end: str = "20231231"
    output: Path = DEFAULT_OUTPUT
    report: Path = DEFAULT_REPORT
    universe: str = "all"
    window: int = 20
    momentum_n: int = 5
    reversal_n: int = 15
    fetch: bool = True
    smoke: bool = False


def _log(message: str) -> None:
    print(f"[guoxin_co_momentum] {message}", flush=True)


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def _num(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    table = df.copy().fillna("")
    lines = [
        "| " + " | ".join(str(col) for col in table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in table.columns) + " |")
    return "\n".join(lines)


def _read_or_fetch(path: Path, fetch_fn: Callable[[], pd.DataFrame], fetch: bool) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    if not fetch:
        raise FileNotFoundError(f"Missing cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df


def _read_covering_raw_cache(prefix: str, start: str, end: str) -> pd.DataFrame | None:
    request_start = pd.Timestamp(start)
    request_end = pd.Timestamp(end)
    for path in RAW_DIR.glob(f"{prefix}_*.parquet"):
        tail = path.stem[len(prefix) + 1 :].split("_")
        if len(tail) < 2:
            continue
        try:
            cache_start = pd.Timestamp(tail[-2])
            cache_end = pd.Timestamp(tail[-1])
        except ValueError:
            continue
        if cache_start <= request_start and cache_end >= request_end:
            df = pd.read_parquet(path)
            if "trade_date" in df:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
                df = df[(df["trade_date"] >= request_start) & (df["trade_date"] <= request_end)].copy()
            return df
    return None


def fetch_stock_basic_extended(pro, output_dir: Path, fetch: bool = True) -> pd.DataFrame:
    path = output_dir / "stock_basic_extended.parquet"

    def _fetch() -> pd.DataFrame:
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,list_date,list_status")
        for col in ["ts_code", "name", "industry", "list_date", "list_status"]:
            if col not in df:
                df[col] = pd.NA
        df["industry"] = df["industry"].fillna("未知")
        return df[["ts_code", "name", "industry", "list_date", "list_status"]].dropna(subset=["ts_code"])

    out = _read_or_fetch(path, _fetch, fetch)
    if "list_date" in out:
        out["list_date"] = pd.to_datetime(out["list_date"], format="%Y%m%d", errors="coerce")
    return out


def fetch_citic_index_basic(pro, output_dir: Path, fetch: bool = True) -> pd.DataFrame:
    path = output_dir / "citic_industry_index_basic.parquet"

    def _fetch() -> pd.DataFrame:
        df = pro.index_basic(market="CI")
        df = df[df["ts_code"].isin(CITIC_LEVEL1_CODES)].copy()
        df["citic_level1"] = df["name"].str.replace("^中信", "", regex=True)
        return df.sort_values("ts_code").reset_index(drop=True)

    return _read_or_fetch(path, _fetch, fetch)


def fetch_index_daily_cache(
    pro,
    ts_codes: list[str],
    start: str,
    end: str,
    output_dir: Path,
    name: str,
    fetch: bool = True,
) -> pd.DataFrame:
    path = output_dir / f"{name}_{start}_{end}.parquet"

    def _fetch() -> pd.DataFrame:
        frames = []
        for code in ts_codes:
            df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        if "pct_chg" in out:
            out["ret"] = out["pct_chg"].astype(float) / 100.0
        else:
            out["ret"] = out["close"].astype(float) / out["pre_close"].astype(float) - 1.0
        return out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    return _read_or_fetch(path, _fetch, fetch)


def _map_tushare_industry_to_citic(source: str) -> str:
    text = "" if pd.isna(source) else str(source)
    if not text or text == "未知":
        return "unknown"
    rules = [
        ("石油石化", ["石油", "油气", "石化"]),
        ("煤炭", ["煤"]),
        ("有色金属", ["有色", "小金属", "稀土", "黄金", "铜", "铝", "铅锌"]),
        ("电力及公用事业", ["供气", "供热", "火力", "水力", "电力", "环境保护", "公用"]),
        ("钢铁", ["钢", "普钢", "特种钢"]),
        ("医药", ["医疗", "化学制药", "中成药", "生物制药", "医药"]),
        ("基础化工", ["化工", "化纤", "农药", "塑料", "橡胶", "染料", "涂料", "日用化工"]),
        ("建筑", ["建筑", "装修", "路桥"]),
        ("建材", ["建材", "水泥", "玻璃", "陶瓷"]),
        ("轻工制造", ["造纸", "包装", "家居", "文教", "轻工"]),
        ("电力设备及新能源", ["电气设备", "新能源"]),
        ("国防军工", ["航空", "航天", "船舶", "军工"]),
        ("汽车", ["汽车", "摩托车"]),
        ("商贸零售", ["百货", "超市", "商业", "商品城", "批发"]),
        ("消费者服务", ["酒店", "餐饮", "旅游", "休闲"]),
        ("家电", ["家用电器", "家电"]),
        ("纺织服装", ["纺织", "服饰", "服装"]),
        ("食品饮料", ["食品", "白酒", "啤酒", "软饮料", "乳制品", "红黄酒"]),
        ("农林牧渔", ["农业", "种植", "渔", "饲料", "林", "牧"]),
        ("银行", ["银行"]),
        ("非银行金融", ["证券", "保险", "多元金融", "金融"]),
        ("房地产", ["地产", "房产", "园区"]),
        ("交通运输", ["机场", "港口", "公路", "水运", "铁路", "仓储", "物流", "运输服务"]),
        ("电子", ["元器件", "半导体", "电子"]),
        ("通信", ["通信", "电信"]),
        ("计算机", ["软件", "IT设备", "互联网", "计算机"]),
        ("传媒", ["传媒", "影视", "出版", "广告"]),
        ("机械", ["机械", "电器仪表", "工程机械", "运输设备"]),
        ("综合", ["综合"]),
    ]
    for citic, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return citic
    return "unknown"


def build_industry_mapping(stock_basic: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = (
        stock_basic[["industry"]]
        .drop_duplicates()
        .rename(columns={"industry": "source_industry"})
        .sort_values("source_industry")
        .reset_index(drop=True)
    )
    mapping["citic_level1"] = mapping["source_industry"].map(_map_tushare_industry_to_citic)
    mapping["mapping_status"] = np.where(mapping["citic_level1"].eq("unknown"), "unknown", "mapped")
    mapping["mapping_note"] = "keyword_proxy_from_tushare_industry"
    coverage = stock_basic.merge(mapping, left_on="industry", right_on="source_industry", how="left")
    summary = (
        coverage.groupby(["source_industry", "citic_level1", "mapping_status"], dropna=False)
        .agg(n_stocks=("ts_code", "nunique"))
        .reset_index()
        .sort_values(["mapping_status", "n_stocks"], ascending=[True, False])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_dir / "citic_level1_industry_mapping.csv", index=False)
    summary.to_csv(output_dir / "industry_mapping_coverage.csv", index=False)
    return mapping, summary


def load_equity_panel(pro, cfg: CoMomentumConfig) -> pd.DataFrame:
    _log("loading equity raw inputs")
    daily = _read_covering_raw_cache("daily", cfg.start, cfg.end)
    if daily is None:
        daily = fetch_daily(pro, start_date=cfg.start, end_date=cfg.end, cache=cfg.fetch)
    adj = _read_covering_raw_cache("adj_factor", cfg.start, cfg.end)
    if adj is None:
        adj = fetch_adj_factor(pro, start_date=cfg.start, end_date=cfg.end, cache=cfg.fetch)
    stock_basic = fetch_stock_basic_extended(pro, cfg.output, fetch=cfg.fetch)
    mapping, _ = build_industry_mapping(stock_basic, cfg.output)

    daily = daily.rename(columns={"vol": "volume"}) if "vol" in daily.columns else daily.copy()
    for df in [daily, adj]:
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    daily_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "volume", "amount"]
    daily = daily[[col for col in daily_cols if col in daily.columns]].copy()
    panel = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    for col in ["open", "high", "low", "close"]:
        if col in panel:
            panel[f"{col}_adj"] = panel[col].astype(float) * panel["adj_factor"].astype(float)
    panel = panel.merge(stock_basic[["ts_code", "name", "industry", "list_date"]], on="ts_code", how="left")
    panel = panel.merge(mapping[["source_industry", "citic_level1"]], left_on="industry", right_on="source_industry", how="left")
    panel["citic_level1"] = panel["citic_level1"].fillna("unknown")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel["stock_ret"] = panel.groupby("ts_code", sort=False)["close_adj"].pct_change()
    panel["volume_price"] = panel["stock_ret"] * panel["volume"].astype(float)
    panel["amount_price"] = panel["stock_ret"] * panel["amount"].astype(float)
    list_age = panel["trade_date"] - panel["list_date"]
    panel["is_new_stock"] = list_age.dt.days < 183
    panel["is_current_st"] = panel["name"].fillna("").astype(str).str.contains("ST", case=False, regex=False)
    return panel


def add_index_weight_filter(pro, panel: pd.DataFrame, cfg: CoMomentumConfig, index_code: str) -> pd.DataFrame:
    weights = fetch_index_weight(pro, index_code=index_code, start_date=cfg.start, end_date=cfg.end, cache=cfg.fetch)
    if weights.empty:
        out = panel.copy()
        out["index_weight"] = np.nan
        return out.iloc[0:0]
    weights = weights.rename(columns={"con_code": "ts_code", "weight": "index_weight"}).copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"], format="%Y%m%d", errors="coerce")
    weights = weights[["ts_code", "trade_date", "index_weight"]].sort_values(["trade_date", "ts_code"])
    work = panel.sort_values(["trade_date", "ts_code"]).copy()
    work = pd.merge_asof(work, weights, on="trade_date", by="ts_code", direction="backward")
    work["index_weight"] = work["index_weight"].fillna(0.0)
    return work[work["index_weight"] > 0].copy()


def add_industry_and_market_returns(
    panel: pd.DataFrame,
    citic_basic: pd.DataFrame,
    citic_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> pd.DataFrame:
    index_map = citic_basic[["ts_code", "citic_level1"]].copy()
    ind = citic_daily.merge(index_map, on="ts_code", how="left")
    ind_ret = ind[["trade_date", "citic_level1", "ret"]].rename(columns={"ret": "industry_ret"})
    market_ret = market_daily[["trade_date", "ret"]].rename(columns={"ret": "market_ret"})
    out = panel.merge(ind_ret, on=["trade_date", "citic_level1"], how="left")
    out = out.merge(market_ret, on="trade_date", how="left")
    return out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _weighted_extreme(values: np.ndarray, sort_key: np.ndarray, n: int, largest: bool) -> float:
    mask = np.isfinite(values) & np.isfinite(sort_key)
    if mask.sum() < n:
        return np.nan
    key = sort_key.copy()
    key[~mask] = -np.inf if largest else np.inf
    order = np.argsort(-key if largest else key)[:n]
    weights = 2.0 ** (-(np.arange(n, dtype=float)) / float(n))
    return float(np.sum(weights * values[order]))


def compute_rolling_comomentum(
    panel: pd.DataFrame,
    window: int = 20,
    momentum_n: int = 5,
    reversal_n: int = 15,
) -> pd.DataFrame:
    factor_cols = ["imax20", "icm", "vicm", "icr", "vicr", "cmc", "mcmc"]
    frames = []
    for _, sub in panel.groupby("ts_code", sort=False):
        sub = sub.sort_values("trade_date").copy()
        stock_ret = sub["stock_ret"].astype(float).to_numpy()
        volume_key = sub["volume_price"].astype(float).to_numpy()
        industry_ret = sub["industry_ret"].astype(float).to_numpy()
        market_ret = sub["market_ret"].astype(float).to_numpy()
        out = {col: np.full(len(sub), np.nan, dtype=float) for col in factor_cols}
        for idx in range(window - 1, len(sub)):
            sl = slice(idx - window + 1, idx + 1)
            ret_win = stock_ret[sl]
            vol_key_win = volume_key[sl]
            ind_win = industry_ret[sl]
            mkt_win = market_ret[sl]
            out["imax20"][idx] = _weighted_extreme(ind_win, ret_win, 1, largest=True)
            out["icm"][idx] = _weighted_extreme(ind_win, ret_win, momentum_n, largest=True)
            out["vicm"][idx] = _weighted_extreme(ind_win, vol_key_win, momentum_n, largest=True)
            out["icr"][idx] = _weighted_extreme(ind_win, ret_win, reversal_n, largest=False)
            out["vicr"][idx] = _weighted_extreme(ind_win, vol_key_win, reversal_n, largest=False)
            vmcm = _weighted_extreme(mkt_win, vol_key_win, momentum_n, largest=True)
            vmcr = _weighted_extreme(mkt_win, vol_key_win, reversal_n, largest=False)
            out["mcmc"][idx] = vmcm - vmcr if np.isfinite(vmcm) and np.isfinite(vmcr) else np.nan
        out["cmc"] = out["vicm"] - out["vicr"]
        factors = pd.DataFrame(out)
        factors.insert(0, "trade_date", sub["trade_date"].to_numpy())
        factors.insert(1, "ts_code", sub["ts_code"].to_numpy())
        frames.append(factors)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["trade_date", "ts_code", *factor_cols])


def build_monthly_evaluation_panel(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["trade_date", "ts_code", "close_adj", "is_new_stock", "is_current_st", "citic_level1"]
    work = panel[[col for col in base_cols if col in panel.columns]].copy()
    work = work[~work["is_new_stock"].fillna(False) & ~work["is_current_st"].fillna(False)].copy()
    work = work[work["citic_level1"].ne("unknown")].copy()
    work["month"] = work["trade_date"].dt.to_period("M")
    month_end = work.groupby("month", sort=False)["trade_date"].transform("max")
    monthly = work[work["trade_date"].eq(month_end)].copy()
    monthly = monthly.sort_values(["ts_code", "trade_date"])
    monthly["fwd_1m_return"] = monthly.groupby("ts_code", sort=False)["close_adj"].shift(-1) / monthly["close_adj"] - 1.0
    monthly["fwd_1m_rank"] = monthly.groupby("trade_date", sort=False)["fwd_1m_return"].rank(pct=True)
    factor_cols = [col for col in factors.columns if col not in {"trade_date", "ts_code"}]
    factors = factors.copy()
    factors["trade_date"] = pd.to_datetime(factors["trade_date"])
    monthly = monthly.merge(factors[["trade_date", "ts_code"] + factor_cols], on=["trade_date", "ts_code"], how="left")
    return monthly.dropna(subset=["fwd_1m_return"]).reset_index(drop=True)


def monthly_rankic(eval_panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    valid = eval_panel[["trade_date", factor, "fwd_1m_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for date, sub in valid.groupby("trade_date", sort=False):
        if len(sub) < 30:
            continue
        rank_x = sub[factor].rank(pct=True)
        rank_y = sub["fwd_1m_return"].rank(pct=True)
        rows.append({"trade_date": date, "RankIC": rank_x.corr(rank_y), "n_stocks": len(sub)})
    return pd.DataFrame(rows)


def decile_returns(eval_panel: pd.DataFrame, factor: str, n_groups: int = 10) -> pd.DataFrame:
    valid = eval_panel[["trade_date", factor, "fwd_1m_return"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    frames = []
    for date, sub in valid.groupby("trade_date", sort=False):
        if len(sub) < n_groups * 10:
            continue
        sub = sub.copy()
        sub["group"] = pd.qcut(sub[factor].rank(method="first"), n_groups, labels=False, duplicates="drop") + 1
        base_ret = float(sub["fwd_1m_return"].mean())
        grouped = sub.groupby("group", sort=False)["fwd_1m_return"].mean().reset_index()
        grouped["trade_date"] = date
        grouped["factor"] = factor
        grouped["excess_return"] = grouped["fwd_1m_return"] - base_ret
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_factor(eval_panel: pd.DataFrame, factor_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    ic_frames = []
    decile_frames = []
    for factor in factor_cols:
        ic = monthly_rankic(eval_panel, factor)
        dec = decile_returns(eval_panel, factor)
        ic_frames.append(ic.assign(factor=factor))
        decile_frames.append(dec)
        rankic = ic["RankIC"].dropna()
        dec_mean = dec.groupby("group")["excess_return"].mean() if not dec.empty else pd.Series(dtype=float)
        direction = 1.0 if rankic.mean() >= 0 else -1.0
        top = dec_mean.get(10 if direction > 0 else 1, np.nan)
        bottom = dec_mean.get(1 if direction > 0 else 10, np.nan)
        rows.append(
            {
                "factor": factor.upper(),
                "rankic_mean": float(rankic.mean()) if not rankic.empty else np.nan,
                "rankic_std": float(rankic.std(ddof=1)) if len(rankic) > 1 else np.nan,
                "icir_annual": float(rankic.mean() / (rankic.std(ddof=1) + 1e-12) * np.sqrt(12.0)) if len(rankic) > 1 else np.nan,
                "rankic_positive_ratio": float((rankic > 0).mean()) if not rankic.empty else np.nan,
                "n_months": int(len(rankic)),
                "direction": direction,
                "top_excess": float(top) if pd.notna(top) else np.nan,
                "bottom_excess": float(bottom) if pd.notna(bottom) else np.nan,
                "long_short_excess": float(top - bottom) if pd.notna(top) and pd.notna(bottom) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    ic_all = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    dec_all = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    return summary, ic_all, dec_all


def _plot_deciles(decile: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "decile_monthly_excess.png"
    if decile.empty:
        return ""
    pivot = decile.groupby(["factor", "group"])["excess_return"].mean().reset_index()
    factors = ["imax20", "icm", "vicm", "vicr", "cmc", "mcmc"]
    sub = pivot[pivot["factor"].isin(factors)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, factor in zip(axes.ravel(), factors):
        one = sub[sub["factor"] == factor]
        ax.bar(one["group"].astype(str), one["excess_return"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(factor.upper())
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_rankic(ic: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "monthly_rankic.png"
    if ic.empty:
        return ""
    plt.figure(figsize=(11, 5))
    for factor, sub in ic.groupby("factor", sort=False):
        sub = sub.sort_values("trade_date")
        plt.plot(sub["trade_date"], sub["RankIC"].rolling(12, min_periods=3).mean(), label=factor.upper(), linewidth=1.0)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend(ncol=3, fontsize=8)
    plt.title("Rolling 12M RankIC")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _format_summary_for_report(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["rankic_mean", "rankic_std", "top_excess", "bottom_excess", "long_short_excess"]:
        if col in out:
            out[col] = out[col].map(_pct)
    if "icir_annual" in out:
        out["icir_annual"] = out["icir_annual"].map(_num)
    if "rankic_positive_ratio" in out:
        out["rankic_positive_ratio"] = out["rankic_positive_ratio"].map(_pct)
    return out


def write_report(
    cfg: CoMomentumConfig,
    summary: pd.DataFrame,
    decile: pd.DataFrame,
    ic: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    decile_png = _plot_deciles(decile, cfg.output)
    rankic_png = _plot_rankic(ic, cfg.output)
    comparison = summary.merge(PDF_REFERENCE, on="factor", how="left", suffixes=("_local", "_pdf"))
    for col in ["rankic_mean_local", "rankic_mean_pdf", "top_excess_local", "top_excess_pdf", "bottom_excess_local", "bottom_excess_pdf"]:
        if col in comparison:
            comparison[col] = comparison[col].map(_pct)
    for col in ["icir_annual_local", "icir_annual_pdf"]:
        if col in comparison:
            comparison[col] = comparison[col].map(_num)
    keep_compare = [
        "factor",
        "rankic_mean_local",
        "rankic_mean_pdf",
        "icir_annual_local",
        "icir_annual_pdf",
        "top_excess_local",
        "top_excess_pdf",
        "bottom_excess_local",
        "bottom_excess_pdf",
    ]
    lines = [
        "# 国信联合动量因子复现报告",
        "",
        "日期：2026-07-13",
        "",
        "## 结论摘要",
        "",
        f"- 样本期：`{cfg.start}` 至 `{cfg.end}`；股票池：`{cfg.universe}`。",
        "- 行业收益使用 Tushare 中信一级 `.CI` 行业指数；股票到中信一级行业的归属使用 Tushare 行业字段映射 proxy。",
        "- 因子方向和数量级需要与 PDF 对照，但不能解释为 Wind 原始口径逐点复现。",
        "",
        "## PDF 原始结果",
        "",
        _markdown_table(_format_summary_for_report(PDF_REFERENCE)),
        "",
        "## 本地复现结果",
        "",
        _markdown_table(_format_summary_for_report(summary)),
        "",
        "## 与 PDF 对照",
        "",
        _markdown_table(comparison[[col for col in keep_compare if col in comparison.columns]]),
        "",
        f"![](../{cfg.output}/{rankic_png})" if rankic_png else "",
        "",
        f"![](../{cfg.output}/{decile_png})" if decile_png else "",
        "",
        "## 行业映射覆盖",
        "",
        _markdown_table(coverage.head(40)),
        "",
        "## 口径差异",
        "",
        "- PDF 使用 Wind 和中信行业历史成分；本地当前无法从 Tushare 取得中信行业历史成分，使用 proxy 映射。",
        "- 本地按可用字段剔除当前 ST 和上市未满 6 个月股票，不能完全复刻 ST 摘帽 3 个月过滤。",
        "- `VICM/VICR` 默认使用 `return * volume` 排序，后续需要保留 `return * amount` 做敏感性检查。",
        "",
        "## 输出文件",
        "",
        f"- 输出目录：`{cfg.output}`",
        f"- 因子值：`{cfg.output / 'factor_values.parquet'}`",
        f"- 汇总表：`{cfg.output / 'factor_summary.csv'}`",
        f"- 月度 RankIC：`{cfg.output / 'monthly_rankic.csv'}`",
        f"- 分组收益：`{cfg.output / 'decile_returns.csv'}`",
    ]
    cfg.report.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def run(cfg: CoMomentumConfig) -> None:
    cfg.output.mkdir(parents=True, exist_ok=True)
    pro = get_tushare_pro() if cfg.fetch else None
    _log("fetching CITIC industry and market index data")
    citic_basic = fetch_citic_index_basic(pro, cfg.output, fetch=cfg.fetch)
    citic_daily = fetch_index_daily_cache(
        pro,
        citic_basic["ts_code"].tolist(),
        cfg.start,
        cfg.end,
        cfg.output,
        "citic_industry_daily",
        fetch=cfg.fetch,
    )
    market_daily = fetch_index_daily_cache(
        pro,
        ["000985.CSI"],
        cfg.start,
        cfg.end,
        cfg.output,
        "market_index_daily",
        fetch=cfg.fetch,
    )
    panel = load_equity_panel(pro, cfg)
    if cfg.universe == "csi500":
        panel = add_index_weight_filter(pro, panel, cfg, "000905.SH")
    elif cfg.universe == "hs300":
        panel = add_index_weight_filter(pro, panel, cfg, "000300.SH")
    elif cfg.universe != "all":
        raise ValueError(f"unsupported universe: {cfg.universe}")

    _log(f"panel rows={len(panel)}, dates={panel['trade_date'].nunique()}, stocks={panel['ts_code'].nunique()}")
    factor_path = cfg.output / "factor_values.parquet"
    if factor_path.exists() and not cfg.fetch:
        _log("loading cached co-momentum factors")
        factors = pd.read_parquet(factor_path)
    else:
        panel = add_industry_and_market_returns(panel, citic_basic, citic_daily, market_daily)
        _log("computing rolling co-momentum factors")
        factors = compute_rolling_comomentum(panel, window=cfg.window, momentum_n=cfg.momentum_n, reversal_n=cfg.reversal_n)
        factors.to_parquet(factor_path, index=False)

    _log("building monthly evaluation panel")
    monthly = build_monthly_evaluation_panel(panel, factors)
    monthly.to_parquet(cfg.output / "monthly_eval_panel.parquet", index=False)
    factor_cols = ["imax20", "icm", "vicm", "icr", "vicr", "cmc", "mcmc"]
    summary, ic, decile = summarize_factor(monthly, factor_cols)
    summary.to_csv(cfg.output / "factor_summary.csv", index=False)
    ic.to_csv(cfg.output / "monthly_rankic.csv", index=False)
    decile.to_csv(cfg.output / "decile_returns.csv", index=False)

    coverage = pd.read_csv(cfg.output / "industry_mapping_coverage.csv")
    write_report(cfg, summary, decile, ic, coverage)
    print(summary.to_string(index=False))
    print(f"wrote outputs to {cfg.output}")
    print(f"wrote report to {cfg.report}")


def parse_args() -> CoMomentumConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="20231231")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--universe", default="all", choices=["all", "csi500", "hs300"])
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--momentum-n", type=int, default=5)
    parser.add_argument("--reversal-n", type=int, default=15)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = args.start
    end = args.end
    if args.smoke and (start == "20100101" and end == "20231231"):
        start, end = "20190101", "20201231"
    return CoMomentumConfig(
        start=start,
        end=end,
        output=Path(args.output),
        report=Path(args.report),
        universe=args.universe,
        window=args.window,
        momentum_n=args.momentum_n,
        reversal_n=args.reversal_n,
        fetch=not args.no_fetch,
        smoke=args.smoke,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
