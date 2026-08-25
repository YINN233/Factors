"""Compare CNE6 cross-sectional regression R2 under different return definitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from factors.risk.cne6_regression import run_factor_return_regression


RETURN_MODE_LABELS = {
    "forward_1d": "今天暴露解释下一日收益",
    "same_day": "当天暴露解释当天收益",
    "lagged_exposure_1d": "昨日暴露解释今日收益",
}

RETURN_MODE_PLOT_LABELS = {
    "forward_1d": "Exposure t -> return t+1",
    "same_day": "Exposure t -> return t",
    "lagged_exposure_1d": "Exposure t-1 -> return t",
}


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "无数据。"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    columns = [str(col) for col in show.columns]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in show.itertuples(index=False):
        values = [str(item).replace("|", "/") if not pd.isna(item) else "" for item in row]
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def _summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    ok = diagnostics[diagnostics["regression_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for mode, sub in ok.groupby("return_mode", sort=False):
        rows.append(
            {
                "return_mode": mode,
                "口径说明": RETURN_MODE_LABELS.get(mode, mode),
                "起始日期": pd.to_datetime(sub["trade_date"]).min().date().isoformat(),
                "结束日期": pd.to_datetime(sub["trade_date"]).max().date().isoformat(),
                "成功回归天数": int(len(sub)),
                "平均样本股票数": float(sub["n_obs"].mean()),
                "平均因子数": float(sub["n_factors"].mean()),
                "平均行业数": float(sub["n_industries"].mean()),
                "平均R2": float(sub["r2"].mean()),
                "R2中位数": float(sub["r2"].median()),
                "平均调整R2": float(sub["adj_r2"].mean()),
                "R2超过0.5的天数占比": float((sub["r2"] >= 0.5).mean()),
                "平均条件数": float(sub["condition_number"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_by_year(diagnostics: pd.DataFrame) -> pd.DataFrame:
    ok = diagnostics[diagnostics["regression_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["year"] = pd.to_datetime(ok["trade_date"]).dt.year
    out = (
        ok.groupby(["return_mode", "year"], sort=True)
        .agg(
            成功回归天数=("r2", "count"),
            平均R2=("r2", "mean"),
            R2中位数=("r2", "median"),
            R2超过0_5占比=("r2", lambda s: float((s >= 0.5).mean())),
            平均样本股票数=("n_obs", "mean"),
            平均行业数=("n_industries", "mean"),
        )
        .reset_index()
    )
    out["口径说明"] = out["return_mode"].map(RETURN_MODE_LABELS).fillna(out["return_mode"])
    return out[["return_mode", "口径说明", "year", "成功回归天数", "平均R2", "R2中位数", "R2超过0_5占比", "平均样本股票数", "平均行业数"]]


def _save_timeseries_plot(diagnostics: pd.DataFrame, output: Path) -> str | None:
    ok = diagnostics[diagnostics["regression_status"] == "ok"].copy()
    if ok.empty:
        return None
    ok["trade_date"] = pd.to_datetime(ok["trade_date"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for mode, sub in ok.groupby("return_mode", sort=False):
        sub = sub.sort_values("trade_date")
        ax.plot(sub["trade_date"], sub["r2"].rolling(60, min_periods=20).mean(), linewidth=1.2, label=RETURN_MODE_PLOT_LABELS.get(mode, mode))
    ax.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.0, label="0.5 reference")
    ax.set_title("CNE6 cross-sectional regression R2 by return definition, 60d rolling mean")
    ax.set_ylabel("R2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = output / "r2_return_definition_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _write_report(doc: Path, output: Path, summary: pd.DataFrame, by_year: pd.DataFrame, fig_name: str | None) -> None:
    same_day_r2 = summary.loc[summary["return_mode"] == "same_day", "平均R2"]
    forward_r2 = summary.loc[summary["return_mode"] == "forward_1d", "平均R2"]
    lagged_r2 = summary.loc[summary["return_mode"] == "lagged_exposure_1d", "平均R2"]
    same_day_value = float(same_day_r2.iloc[0]) if not same_day_r2.empty else float("nan")
    forward_value = float(forward_r2.iloc[0]) if not forward_r2.empty else float("nan")
    lagged_value = float(lagged_r2.iloc[0]) if not lagged_r2.empty else float("nan")
    same_forward_diff = same_day_value - forward_value
    lagged_forward_diff = lagged_value - forward_value

    conclusion = (
        "这次对照说明，之前报告里的 0.3693 不能直接拿去和导师说的 0.5 风险模型标准比较，"
        "因为它是预测下一日收益的口径。"
    )
    if pd.notna(same_day_value) and same_day_value >= 0.5:
        conclusion += (
            f"换成当天收益分解口径后，平均 R2 提高到 {same_day_value:.4f}，已经达到 0.5 参考线附近或以上。"
            "所以主要问题是口径，而不是共同因子完全没有解释力。"
        )
    elif pd.notna(same_day_value):
        conclusion += (
            f"换成当天收益分解口径后，平均 R2 是 {same_day_value:.4f}，只比原来的预测口径高 {same_forward_diff:.4f}，仍然没有达到 0.5。"
            "这说明收益口径确实有影响，但影响幅度很小，不能把低 R2 主要归因于口径问题；当前公开数据版 CNE6-style 模型本身还需要继续改进。"
        )

    lines = [
        "# CNE6 横截面回归 R2 口径对照实验",
        "",
        "## 1. 为什么要做这个实验",
        "",
        "导师提到 R2 到 0.5 才算比较可用，我理解这个判断主要针对风险模型里的收益分解口径。前一版报告里的平均 R2 是 0.3693，但代码实际用的是 `fwd_1d_return`，也就是用今天的暴露解释下一天收益。这个更像短期预测，不是标准风险模型事后分解，所以我先把口径拆开看。",
        "",
        "这次我没有直接改掉正式风险模型输出，而是单独做一个对照实验。这样可以先判断问题到底出在收益口径，还是出在行业分类、风格描述子和回归方法本身。",
        "",
        "## 2. 三种口径",
        "",
        "|return_mode|含义|我怎么看这个口径|",
        "|---|---|---|",
        "|forward_1d|今天暴露解释下一日收益|这是前一版报告口径，更偏预测，所以 R2 天然低。|",
        "|same_day|当天暴露解释当天收益|这是更接近 Barra/CNE 风险模型的事后收益分解口径，适合和 0.5 参考线比较。|",
        "|lagged_exposure_1d|昨日暴露解释今日收益|这是更严格的可交易近似口径，避免使用当天收盘后才完全知道的暴露解释当天收益。|",
        "",
        "## 3. 总体结果",
        "",
        _md_table(summary, max_rows=10),
        "",
        "## 4. 分年度结果",
        "",
        _md_table(by_year, max_rows=80),
        "",
    ]
    if fig_name:
        lines.extend(["## 5. R2 时间序列", "", f"![](../outputs/cne6_r2_return_definition_comparison/{fig_name})", ""])
    lines.extend(
        [
            "## 6. 我的理解",
            "",
            conclusion,
            "",
            f"具体看数值，`forward_1d` 的平均 R2 是 {forward_value:.4f}，`same_day` 的平均 R2 是 {same_day_value:.4f}，`lagged_exposure_1d` 的平均 R2 是 {lagged_value:.4f}。`same_day` 比 `forward_1d` 高 {same_forward_diff:.4f}，`lagged_exposure_1d` 比 `forward_1d` 高 {lagged_forward_diff:.4f}。这个结果说明，当前模型不是因为用了预测口径才大幅低估 R2；即使用更接近风险分解的当天收益口径，解释力也还是偏弱。",
            "",
            "我后面跟导师沟通时，会把这两个问题分开：风险模型的横截面 R2 主要看收益分解能力；选股模型的有效性应该看 IC、分组收益和回测超额，不能只看风险模型回归 R2。",
            "",
            "## 7. 还需要注意的地方",
            "",
            "1. `same_day` 是事后分解口径，不等于可交易预测能力。",
            "2. 当前仍是 CNE6-style 公开数据代理版，不是商业 Barra CNE6 的精确复现。",
            "3. 如果 `same_day` 仍没有稳定超过 0.5，后续优先检查行业分类粒度、风格因子正交化、极值处理和回归约束。",
            "",
        ]
    )
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def run(panel_path: Path, style_path: Path, output: Path, doc: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    style = pd.read_parquet(style_path)

    diagnostics_frames = []
    for mode in ["forward_1d", "same_day", "lagged_exposure_1d"]:
        _, _, diagnostics = run_factor_return_regression(panel, style, return_mode=mode)
        diagnostics_frames.append(diagnostics)

    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    summary = _summarize_diagnostics(diagnostics)
    by_year = _summarize_by_year(diagnostics)
    fig_name = _save_timeseries_plot(diagnostics, output)

    diagnostics.to_csv(output / "r2_return_definition_diagnostics.csv", index=False)
    summary.to_csv(output / "r2_return_definition_summary.csv", index=False)
    by_year.to_csv(output / "r2_return_definition_by_year.csv", index=False)
    _write_report(doc, output, summary, by_year, fig_name)
    print(f"wrote CNE6 R2 return-definition comparison to {output} and {doc}")


def write_report_from_existing(output: Path, doc: Path) -> None:
    diagnostics = pd.read_csv(output / "r2_return_definition_diagnostics.csv", parse_dates=["trade_date"])
    summary = pd.read_csv(output / "r2_return_definition_summary.csv")
    by_year = pd.read_csv(output / "r2_return_definition_by_year.csv")
    fig_name = _save_timeseries_plot(diagnostics, output)
    _write_report(doc, output, summary, by_year, fig_name)
    print(f"wrote CNE6 R2 return-definition report to {doc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--style", default="outputs/cne6_reproduction/style_exposures.parquet")
    parser.add_argument("--output", default="outputs/cne6_r2_return_definition_comparison")
    parser.add_argument("--doc", default="docs/cne6_regression_r2_return_definition_comparison_2026-07-29.md")
    parser.add_argument("--reuse-existing", action="store_true", help="only rewrite the Markdown report from existing CSV outputs")
    args = parser.parse_args()
    if args.reuse_existing:
        write_report_from_existing(Path(args.output), Path(args.doc))
    else:
        run(Path(args.panel), Path(args.style), Path(args.output), Path(args.doc))


if __name__ == "__main__":
    main()
