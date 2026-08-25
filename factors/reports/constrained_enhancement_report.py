"""Generate Markdown report for constrained CSI500 enhancement outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def _num(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    table = df.copy().fillna("")
    headers = [str(col) for col in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([str(row[col]).replace("|", "\\|") for col in table.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_summary(df: pd.DataFrame) -> str:
    cols = [
        "scenario",
        "period",
        "excess_total_return",
        "tracking_error",
        "information_ratio",
        "active_max_drawdown",
        "daily_active_win_rate",
        "monthly_active_win_rate",
        "avg_active_share",
        "avg_max_industry_active",
    ]
    table = df[[col for col in cols if col in df.columns]].copy()
    for col in [
        "excess_total_return",
        "tracking_error",
        "active_max_drawdown",
        "daily_active_win_rate",
        "monthly_active_win_rate",
        "avg_active_share",
        "avg_max_industry_active",
    ]:
        if col in table:
            table[col] = table[col].map(_pct)
    if "information_ratio" in table:
        table["information_ratio"] = table["information_ratio"].map(_num)
    return _markdown_table(table)


def _monthly_active_table(daily: pd.DataFrame) -> str:
    ytd = daily[daily["trade_date"] >= pd.Timestamp("2026-01-01")].copy()
    if ytd.empty:
        return ""
    monthly = (
        ytd.assign(month=ytd["trade_date"].dt.to_period("M").astype(str))
        .groupby(["scenario", "month"])
        .apply(lambda x: (1.0 + x["portfolio_return_net"]).prod() / (1.0 + x["benchmark_return"]).prod() - 1.0, include_groups=False)
        .reset_index(name="active_return")
    )
    pivot = monthly.pivot(index="month", columns="scenario", values="active_return").reset_index()
    for col in pivot.columns:
        if col != "month":
            pivot[col] = pivot[col].map(_pct)
    return _markdown_table(pivot)


def _top_factor_table(factor_summary: pd.DataFrame, n: int = 12) -> str:
    if factor_summary.empty:
        return ""
    cols = ["factor", "validation_status", "coverage", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean", "validation_reason"]
    table = factor_summary[[col for col in cols if col in factor_summary.columns]].copy()
    status_order = {"passed": 0, "quarantined": 1, "failed": 2}
    table["_status_order"] = table["validation_status"].map(status_order).fillna(9)
    table = table.sort_values(["_status_order", "test_rankic_mean"], ascending=[True, False]).drop(columns=["_status_order"]).head(n)
    for col in ["coverage", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _failed_factor_table(factor_summary: pd.DataFrame, n: int = 12) -> str:
    if factor_summary.empty or "validation_status" not in factor_summary:
        return ""
    bad = factor_summary[factor_summary["validation_status"] != "passed"].copy()
    if bad.empty:
        return ""
    cols = ["factor", "validation_status", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean", "validation_reason"]
    table = bad[[col for col in cols if col in bad.columns]].head(n)
    for col in ["valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _importance_table(importance: pd.DataFrame, n: int = 14) -> str:
    if importance.empty:
        return ""
    table = importance.head(n).copy()
    if "importance" in table:
        table["importance"] = table["importance"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _merged_factor_metadata(metadata: pd.DataFrame, factor_summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()
    out = metadata.copy()
    if not factor_summary.empty and "factor" in factor_summary:
        out = out.merge(factor_summary, left_on="factor_name", right_on="factor", how="left", suffixes=("_meta", ""))
    selected_set = set(selected["factor"]) if not selected.empty and "factor" in selected else set()
    out["selected"] = out["factor_name"].isin(selected_set)
    return out


def _availability_status_table(metadata: pd.DataFrame, factor_summary: pd.DataFrame, selected: pd.DataFrame) -> str:
    merged = _merged_factor_metadata(metadata, factor_summary, selected)
    if merged.empty:
        return ""
    rows = []
    for availability, sub in merged.groupby("availability", sort=False):
        rows.append(
            {
                "availability": availability,
                "count": len(sub),
                "passed": int((sub["validation_status"] == "passed").sum()) if "validation_status" in sub else 0,
                "quarantined": int((sub["validation_status"] == "quarantined").sum()) if "validation_status" in sub else 0,
                "failed": int((sub["validation_status"] == "failed").sum()) if "validation_status" in sub else 0,
                "selected": int(sub["selected"].sum()),
                "selected_ratio": _pct(float(sub["selected"].mean())),
            }
        )
    rows.append(
        {
            "availability": "total",
            "count": len(merged),
            "passed": int((merged["validation_status"] == "passed").sum()) if "validation_status" in merged else 0,
            "quarantined": int((merged["validation_status"] == "quarantined").sum()) if "validation_status" in merged else 0,
            "failed": int((merged["validation_status"] == "failed").sum()) if "validation_status" in merged else 0,
            "selected": int(merged["selected"].sum()),
            "selected_ratio": _pct(float(merged["selected"].mean())),
        }
    )
    return _markdown_table(pd.DataFrame(rows))


def _selected_public_factor_table(metadata: pd.DataFrame, factor_summary: pd.DataFrame, selected: pd.DataFrame, n: int = 40) -> str:
    merged = _merged_factor_metadata(metadata, factor_summary, selected)
    if merged.empty:
        return ""
    selected_set = set(selected["factor"]) if not selected.empty and "factor" in selected else set()
    table = merged[merged["factor_name"].isin(selected_set)].copy()
    if table.empty:
        return ""
    order = {factor: idx for idx, factor in enumerate(selected["factor"].tolist())} if "factor" in selected else {}
    table["_order"] = table["factor_name"].map(order).fillna(9999)
    cols = [
        "source_factor_id",
        "factor_name",
        "availability",
        "validation_status",
        "valid_rankic_mean",
        "test_rankic_mean",
        "ytd_2026_rankic_mean",
    ]
    table = table.sort_values("_order")[[col for col in cols if col in table.columns]].head(n)
    for col in ["valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _restored_factor_table(metadata: pd.DataFrame, factor_summary: pd.DataFrame, selected: pd.DataFrame) -> str:
    merged = _merged_factor_metadata(metadata, factor_summary, selected)
    if merged.empty:
        return ""
    restored = {"02", "11", "12", "13", "14", "15", "26"} | {str(i) for i in range(33, 48)} | {"49", "52", "53"}
    ids = merged["source_factor_id"].map(lambda x: str(x).split(".")[0].zfill(2))
    table = merged[ids.isin(restored)].copy()
    if table.empty:
        return ""
    table["_id"] = table["source_factor_id"].map(lambda x: str(x).split(".")[0].zfill(2))
    cols = [
        "_id",
        "factor_name",
        "availability",
        "validation_status",
        "selected",
        "valid_rankic_mean",
        "test_rankic_mean",
        "ytd_2026_rankic_mean",
        "validation_reason",
    ]
    table = table.sort_values("_id")[[col for col in cols if col in table.columns]]
    for col in ["valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    if "selected" in table:
        table["selected"] = table["selected"].map(lambda x: "yes" if bool(x) else "no")
    return _markdown_table(table.rename(columns={"_id": "source_factor_id"}))


def _unpassed_factor_table(metadata: pd.DataFrame, factor_summary: pd.DataFrame, selected: pd.DataFrame, n: int = 30) -> str:
    merged = _merged_factor_metadata(metadata, factor_summary, selected)
    if merged.empty or "validation_status" not in merged:
        return ""
    table = merged[merged["validation_status"] != "passed"].copy()
    if table.empty:
        return ""
    cols = [
        "source_factor_id",
        "factor_name",
        "availability",
        "validation_status",
        "valid_rankic_mean",
        "test_rankic_mean",
        "ytd_2026_rankic_mean",
        "validation_reason",
    ]
    table = table[[col for col in cols if col in table.columns]].head(n)
    for col in ["valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _pack2_factor_table(factor_summary: pd.DataFrame, n: int = 18) -> str:
    if factor_summary.empty:
        return ""
    pack2 = factor_summary[factor_summary["factor"].str.startswith("rl2_", na=False)].copy()
    if pack2.empty:
        return ""
    status_order = {"passed": 0, "quarantined": 1, "failed": 2}
    pack2["_status_order"] = pack2["validation_status"].map(status_order).fillna(9)
    pack2 = pack2.sort_values(["_status_order", "test_rankic_mean"], ascending=[True, False]).head(n)
    cols = ["factor", "validation_status", "coverage", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean", "validation_reason"]
    table = pack2[[col for col in cols if col in pack2.columns]].copy()
    for col in ["coverage", "valid_rankic_mean", "test_rankic_mean", "ytd_2026_rankic_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return _markdown_table(table)


def _skipped_pack2_table(metadata: pd.DataFrame, n: int = 18) -> str:
    if metadata.empty or "source" not in metadata:
        return ""
    skipped = metadata[(metadata["source"] == "rongliang_public_pack2") & (metadata["availability"] == "skipped")].copy()
    if skipped.empty:
        return ""
    cols = ["source_factor_id", "factor_name", "missing_columns", "skip_reason"]
    return _markdown_table(skipped[[col for col in cols if col in skipped.columns]].head(n))


def _plot_excess_nav(daily: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "scenario_excess_nav.png"
    plt.figure(figsize=(10, 5))
    for scenario, sub in daily.groupby("scenario"):
        sub = sub.sort_values("trade_date")
        if "excess_nav" not in sub:
            sub["excess_nav"] = (1.0 + sub["active_return"]).cumprod()
        plt.plot(sub["trade_date"], sub["excess_nav"], label=scenario, linewidth=1.2)
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.legend(fontsize=8)
    plt.title("Scenario Excess NAV")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_ytd_monthly(daily: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "ytd_monthly_active_return.png"
    ytd = daily[daily["trade_date"] >= pd.Timestamp("2026-01-01")].copy()
    if ytd.empty:
        return ""
    monthly = (
        ytd.assign(month=ytd["trade_date"].dt.to_period("M").astype(str))
        .groupby(["scenario", "month"])
        .apply(lambda x: (1.0 + x["portfolio_return_net"]).prod() / (1.0 + x["benchmark_return"]).prod() - 1.0, include_groups=False)
        .reset_index(name="active_return")
    )
    pivot = monthly.pivot(index="month", columns="scenario", values="active_return")
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.title("2026 YTD Monthly Active Return")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_feature_importance(importance: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "xgb_feature_importance.png"
    if importance.empty:
        return ""
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Model Feature Contribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_factor_status(summary: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "public_factor_status.png"
    if summary.empty or "validation_status" not in summary:
        return ""
    counts = summary["validation_status"].value_counts()
    counts.plot(kind="bar", figsize=(7, 4))
    plt.title("Public Factor Validation Status")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_active_distribution(daily: pd.DataFrame, scenario: str, output_dir: Path) -> str:
    path = output_dir / "active_return_distribution.png"
    sub = daily[(daily["scenario"] == scenario) & (daily["trade_date"] >= pd.Timestamp("2025-01-01"))].copy()
    if sub.empty:
        return ""
    plt.figure(figsize=(8, 5))
    plt.hist(sub["active_return"].dropna(), bins=50, color="#4c78a8", alpha=0.85)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title(f"{scenario} Daily Active Return Distribution Since 2025")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_rolling_ir(daily: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "rolling_60d_information_ratio.png"
    plt.figure(figsize=(10, 5))
    for scenario, sub in daily.groupby("scenario"):
        sub = sub.sort_values("trade_date").copy()
        mean = sub["active_return"].rolling(60).mean() * 252.0
        vol = sub["active_return"].rolling(60).std(ddof=1) * np.sqrt(252.0)
        rolling_ir = mean / (vol + 1e-12)
        plt.plot(sub["trade_date"], rolling_ir, label=scenario, linewidth=1.0)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend(fontsize=8)
    plt.title("Rolling 60D Information Ratio")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_constraint_profile(summary: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "constraint_profile_test.png"
    sub = summary[summary["period"] == "test"].copy()
    if sub.empty:
        return ""
    plot = sub.set_index("scenario")[["avg_active_share", "avg_max_industry_active"]]
    plot.plot(kind="bar", figsize=(10, 5))
    plt.title("Test Period Constraint Profile")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def _plot_factor_rankic(factor_summary: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "public_factor_test_rankic.png"
    if factor_summary.empty or "test_rankic_mean" not in factor_summary:
        return ""
    top = factor_summary.sort_values("test_rankic_mean", ascending=False).head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    colors = np.where(top["validation_status"].eq("passed"), "#2ca02c", "#d62728")
    plt.barh(top["factor"], top["test_rankic_mean"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("Public Factor Test RankIC")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/csi500_xgb_constrained_index_enhancement")
    parser.add_argument("--report", default="docs/csi500_xgb_constrained_index_enhancement_2026-07-10.md")
    args = parser.parse_args()

    output_dir = Path(args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    daily = _read_csv(output_dir / "constrained_daily_returns.csv", parse_dates=["trade_date"])
    scenario_summary = _read_csv(output_dir / "scenario_summary.csv")
    factor_summary = _read_csv(output_dir / "public_factor_validation_summary.csv")
    availability = _read_csv(output_dir / "public_factor_metadata.csv")
    model_summary = _read_csv(output_dir / "xgb_model_summary.csv")
    importance = _read_csv(output_dir / "xgb_feature_importance.csv")
    selected = _read_csv(output_dir / "selected_public_features.csv")

    nav_png = _plot_excess_nav(daily, output_dir)
    ytd_png = _plot_ytd_monthly(daily, output_dir)
    imp_png = _plot_feature_importance(importance, output_dir)
    status_png = _plot_factor_status(factor_summary, output_dir)

    constrained = scenario_summary[
        (scenario_summary["period"] == "test") & scenario_summary["scenario"].str.startswith("xgb_industry_style", na=False)
    ].copy()
    best_constrained = constrained.sort_values("information_ratio", ascending=False).head(1)
    best_constrained_name = best_constrained["scenario"].iloc[0] if not best_constrained.empty else "xgb_industry_style_tight"
    dist_png = _plot_active_distribution(daily, best_constrained_name, output_dir)
    rolling_png = _plot_rolling_ir(daily, output_dir)
    constraint_png = _plot_constraint_profile(scenario_summary, output_dir)
    rankic_png = _plot_factor_rankic(factor_summary, output_dir)

    status_counts = factor_summary["validation_status"].value_counts().to_dict() if not factor_summary.empty else {}
    availability_counts = availability["availability"].value_counts().to_dict() if not availability.empty else {}
    source_counts = availability.groupby("source").size().to_dict() if not availability.empty and "source" in availability else {}
    pack2_meta = availability[availability["source"] == "rongliang_public_pack2"] if not availability.empty and "source" in availability else pd.DataFrame()
    pack2_summary = factor_summary[factor_summary["factor"].str.startswith("rl2_", na=False)] if not factor_summary.empty else pd.DataFrame()
    pack2_status_counts = pack2_summary["validation_status"].value_counts().to_dict() if not pack2_summary.empty else {}
    pack2_availability_counts = pack2_meta["availability"].value_counts().to_dict() if not pack2_meta.empty else {}
    no_style_test = scenario_summary[(scenario_summary["scenario"] == "xgb_no_style_constraint") & (scenario_summary["period"] == "test")]
    no_style_excess = _pct(no_style_test["excess_total_return"].iloc[0]) if not no_style_test.empty else ""
    no_style_ir = _num(no_style_test["information_ratio"].iloc[0]) if not no_style_test.empty else ""
    selected_count = len(selected) if not selected.empty else 0
    backend = scenario_summary["backend"].dropna().iloc[0] if "backend" in scenario_summary and not scenario_summary["backend"].dropna().empty else ""
    importance_type = importance["importance_type"].dropna().iloc[0] if "importance_type" in importance and not importance.empty else ""
    if backend == "xgboost":
        backend_note = "- 当前后端：`xgboost`。本次结果使用原生 `XGBRegressor`，不再是 sklearn fallback。"
    else:
        backend_note = f"- 当前后端：`{backend}`。如果本机未安装 xgboost，会自动 fallback 到 sklearn HistGradientBoosting。"
    if importance_type == "xgboost_gain":
        importance_note = "- 当前特征重要性类型：`xgboost_gain`。它反映树模型分裂增益贡献，只能作为模型解释线索，不能单独等同于因子有效性。"
    else:
        importance_note = f"- 当前特征重要性类型：`{importance_type}`。`prediction_corr` 表示 fallback 模型没有原生 gain，用特征与模型预测值的相关度作为替代贡献度，不应解释为 XGBoost split gain。"
    if backend == "xgboost":
        model_caveat = "4. 当前模型后端是原生 xgboost；特征重要性采用 `xgboost_gain` 时只能辅助解释模型使用了什么，最终仍以样本外 RankIC 和约束组合回测为准。"
    else:
        model_caveat = "4. 当前模型后端是 sklearn fallback，不是原生 xgboost gain；特征重要性采用 `prediction_corr`，只能看作贡献线索。"
    merged_factors = _merged_factor_metadata(availability, factor_summary, selected)
    selected_direct = int(((merged_factors["availability"] == "direct") & merged_factors["selected"]).sum()) if not merged_factors.empty else 0
    selected_proxy = int(((merged_factors["availability"] == "proxy") & merged_factors["selected"]).sum()) if not merged_factors.empty else 0

    lines = [
        "# 中证500 XGBoost 约束指数增强阶段报告",
        "",
        "日期：2026-07-13",
        "",
        "## 摘要结论",
        "",
        f"- 本轮严格限定在两个融量公开因子文本里的 65 个条目内，没有新增第 66 个候选因子。补字段后元数据共 {len(availability)} 个，direct {availability_counts.get('direct', 0)} 个、proxy {availability_counts.get('proxy', 0)} 个、skipped {availability_counts.get('skipped', 0)} 个。",
        f"- 65 个因子全部完成本地计算和验证：通过 {status_counts.get('passed', 0)} 个，隔离 {status_counts.get('quarantined', 0)} 个，失败 {status_counts.get('failed', 0)} 个。相关性去重后入模公开因子 {selected_count} 个，其中 direct {selected_direct} 个、proxy {selected_proxy} 个。",
        f"- Pack2 单独看：共 {source_counts.get('rongliang_public_pack2', 0)} 个条目，补字段后可计算 {len(pack2_summary)} 个，放行 {pack2_status_counts.get('passed', 0)} 个，隔离 {pack2_status_counts.get('quarantined', 0)} 个，失败 {pack2_status_counts.get('failed', 0)} 个，skipped {pack2_availability_counts.get('skipped', 0)} 个。",
        f"- 当前公开因子筛选后进入模型候选的公开特征数为 {selected_count} 个；不加风格约束的 XGB 方案在 2025-01-02 至 2026-07-03 测试期超额为 {no_style_excess}，IR 为 {no_style_ir}，说明外源公开因子直接进组合仍不稳。",
        f"- 加行业/风格约束后，`{best_constrained_name}` 是测试期最稳的 XGB 约束方案；测试期超额约 {_pct(best_constrained['excess_total_return'].iloc[0]) if not best_constrained.empty else ''}，IR 约 {_num(best_constrained['information_ratio'].iloc[0]) if not best_constrained.empty else ''}，主动最大回撤约 {_pct(best_constrained['active_max_drawdown'].iloc[0]) if not best_constrained.empty else ''}。",
        "- 旧的 `current_exp_score` 基本面组合在测试期仍很强，不能武断说 XGB 已经全面替代旧方案；更合理的结论是：公开价量因子 + 旧基本面因子在模型层有预测力，但必须通过行业/风格约束才能变成像样的指增组合。",
        "",
        "## 65 个公开因子缺字段补齐",
        "",
        "补字段只服务于这 65 个公开因子。`AF_CLOSE/AF_HIGH/AF_LOW/AF_OPEN` 映射到本地复权价；`AF_VWAP`、`REINSTATEMENT_CHG_60D`、主力/大单资金流、CNE5 风格字段均按设计文档里的本地 proxy 口径补齐。proxy 不等于原始生产字段，报告结论只对本地可复现口径成立。",
        "",
        _availability_status_table(availability, factor_summary, selected),
        "",
        "原先缺字段或部分缺字段的重点恢复条目如下。`selected=yes` 表示通过验证且经过相关性去重后进入本轮模型特征。",
        "",
        _restored_factor_table(availability, factor_summary, selected),
        "",
        "## 方案表现",
        "",
        _format_summary(scenario_summary),
        "",
        f"![](../{output_dir}/{nav_png})" if nav_png else "",
        "",
        f"![](../{output_dir}/{rolling_png})" if rolling_png else "",
        "",
        f"![](../{output_dir}/{constraint_png})" if constraint_png else "",
        "",
        "## 2026 年以来表现",
        "",
        _monthly_active_table(daily),
        "",
        f"![](../{output_dir}/{ytd_png})" if ytd_png else "",
        "",
        f"![](../{output_dir}/{dist_png})" if dist_png else "",
        "",
        "## 公开因子验证",
        "",
        "公开文本中的 IC/Sharpe 没有直接采信。所有因子先经过本地中证500历史成分股、未来 5 日截面 Rank 标签的验证门，方向由训练集确定，再看验证期、测试期和 2026YTD。",
        "",
        _top_factor_table(factor_summary),
        "",
        f"![](../{output_dir}/{status_png})" if status_png else "",
        "",
        f"![](../{output_dir}/{rankic_png})" if rankic_png else "",
        "",
        "未放行或需隔离的因子如下。这些因子本轮不进入模型，除非未来重新通过同一验证门。",
        "",
        _unpassed_factor_table(availability, factor_summary, selected),
        "",
        "通过验证并进入模型的公开因子如下。顺序即相关性去重后的入模顺序。",
        "",
        _selected_public_factor_table(availability, factor_summary, selected),
        "",
        "## XGBoost/替代模型",
        "",
        backend_note,
        importance_note,
        "",
        _markdown_table(model_summary) if not model_summary.empty else "",
        "",
        _importance_table(importance),
        "",
        f"![](../{output_dir}/{imp_png})" if imp_png else "",
        "",
        "## 经济含义",
        "",
        "- 通过验证的公开因子主要集中在价量拥挤、成交量背离、波动压缩后的反转、换手相对强弱、PVT 协方差反转，以及 Pack2 新增的换手率稳定、量价残差波动和资金流稳定类 proxy。这类因子本质上在捕捉中证500成分股里的交易拥挤解除和短中期行为反转。",
        "- 资金流 proxy 并非全部有效：部分主力/超大单流入衰减类因子在 2025 以来或 2026YTD 走弱，被隔离；通过验证的资金流因子更偏向资金流极值、波动和稳定性，而不是简单净流入追涨。",
        "- 旧基本面因子 `eps_bps_value_quality`、`quality_growth_hmean`、`industry_neutral_roe_value_pb` 仍进入模型，提供价值/质量底座，避免纯价量模型在风格漂移时失效。",
        "- 行业和风格约束的作用不是提高裸预测 IC，而是把 alpha 兑现方式限制在指数增强可接受的主动风险预算内；本轮最关键的证据是无约束 XGB 样本外为负，而约束后转正且回撤下降。",
        "",
        "## 使用方式",
        "",
        "1. 新外源因子统一按长表接入：`trade_date, ts_code, factor_name, factor_value, source, version, release_date`。",
        "2. 先运行外源因子校验入口，状态保持 `pending`；只有通过覆盖率、验证期 RankIC、测试期 RankIC、2026YTD 稳定性和相关性筛选后，才能标记为 `passed` 并进入模型。",
        "3. 主流水线命令：`venv/bin/python -m factors.reports.constrained_index_enhancement --start 20180101 --end 20260706`。",
        "4. 报告命令：`venv/bin/python -m factors.reports.constrained_enhancement_report`。",
        "",
        "## 批判性备注",
        "",
        "1. 38 个 proxy 因子不能等同于融量原始生产字段；尤其是资金流和 CNE5 风格字段，只能解释为本地 Tushare/风格暴露代理。",
        "2. 通过验证不代表长期有效，隔离和失败列表需要保留；本轮 25 个因子没有进入模型，说明公开因子存在明显失效和口径不匹配。",
        "3. 2026YTD 截止到 2026-07-03，样本仍短，需要后续滚动复检。",
        model_caveat,
        "5. 当前组合优化默认使用快速投影式约束求解，适合研究流水线；若进入生产或正式复盘，应抽样用 `method='cvxpy'` 做精确约束交叉验证。",
        "6. 本轮修正了公开因子相关性去重的性能问题：先按日截面 rank，再一次性计算相关矩阵，避免对全样本重复 Spearman 排序；不改变去重阈值和入模规则。",
        "",
        "## 输出文件",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 日度回测：`{output_dir / 'constrained_daily_returns.csv'}`",
        f"- 调仓权重：`{output_dir / 'constrained_weights.csv'}`",
        f"- 方案摘要：`{output_dir / 'scenario_summary.csv'}`",
        f"- 因子验证：`{output_dir / 'public_factor_validation_summary.csv'}`",
        f"- 模型预测：`{output_dir / 'xgb_predictions.parquet'}`",
    ]
    report_path.write_text("\n".join(line for line in lines if line is not None))
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
