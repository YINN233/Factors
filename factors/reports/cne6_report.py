"""Generate a Markdown report for the local CNE6-style risk model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE_CN = {
    "style_size": "规模",
    "style_volatility": "波动率",
    "style_liquidity": "流动性",
    "style_momentum": "动量",
    "style_value": "价值",
    "style_growth": "成长",
    "style_quality": "质量",
    "style_dividend_yield": "股息率",
    "style_sentiment": "分析师预期情绪",
}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "\n无可用数据。\n"
    out = df.head(max_rows).copy()
    cols = out.columns.tolist()
    lines = ["|" + "|".join(map(str, cols)) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in out.iterrows():
        lines.append("|" + "|".join("" if pd.isna(row[col]) else str(row[col]) for col in cols) + "|")
    return "\n".join(lines) + "\n"


def _save_style_corr(output: Path) -> str | None:
    path = output / "style_correlation.csv"
    if not path.exists():
        return None
    corr = pd.read_csv(path, index_col=0)
    if corr.empty:
        return None
    labels = [c.replace("style_", "") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.fillna(0.0).to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = output / "style_correlation_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _save_factor_cum(output: Path) -> str | None:
    path = output / "factor_returns.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["trade_date"])
    style_cols = [c for c in df.columns if c.startswith("style_")]
    if not style_cols:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in style_cols:
        nav = pd.to_numeric(df[col], errors="coerce").fillna(0.0).cumsum()
        ax.plot(df["trade_date"], nav, label=col.replace("style_", ""), linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("CNE6-style style factor cumulative returns")
    fig.tight_layout()
    out = output / "factor_returns_cum.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _save_regression_r2(output: Path) -> str | None:
    path = output / "regression_diagnostics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["trade_date"])
    if "r2" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["trade_date"], df["r2"], linewidth=0.8)
    ax.plot(df["trade_date"], df["r2"].rolling(60, min_periods=20).mean(), linewidth=1.5, label="60d mean")
    ax.legend()
    ax.set_title("Daily cross-sectional regression R2")
    fig.tight_layout()
    out = output / "regression_r2_timeseries.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _save_specific_distribution(output: Path) -> str | None:
    path = output / "specific_risk.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["trade_date", "specific_risk_252"])
    df = df.dropna(subset=["specific_risk_252"])
    if df.empty:
        return None
    latest = pd.to_datetime(df["trade_date"]).max()
    sub = df[pd.to_datetime(df["trade_date"]) == latest]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub["specific_risk_252"] * np.sqrt(252), bins=40, color="#4C78A8", alpha=0.85)
    ax.set_title("Annualized specific risk distribution")
    fig.tight_layout()
    out = output / "specific_risk_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _save_te(output: Path) -> str | None:
    path = output / "predicted_vs_realized_te.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["trade_date"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, sub in df.groupby("scenario", sort=False):
        ax.plot(sub["trade_date"], sub["predicted_te_annual"], linewidth=1.0, label=f"{scenario} predicted")
    ax.set_title("Predicted tracking error by scenario")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = output / "predicted_vs_realized_te.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _save_risk_stack(output: Path) -> str | None:
    path = output / "portfolio_risk_attribution.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["trade_date"])
    if df.empty:
        return None
    scenario = "xgb_industry_style_mid" if "xgb_industry_style_mid" in set(df["scenario"]) else df["scenario"].iloc[-1]
    sub = df[df["scenario"] == scenario].copy()
    cols = ["style_var_daily", "industry_var_daily", "specific_var_daily"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(sub["trade_date"], *[sub[c].clip(lower=0) for c in cols], labels=["style", "industry", "specific"], alpha=0.85)
    ax.set_title(f"Risk attribution variance stack: {scenario}")
    ax.legend()
    fig.tight_layout()
    out = output / "portfolio_risk_attribution_stack.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def generate_report(output: Path, doc: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    doc.parent.mkdir(parents=True, exist_ok=True)
    figs = [
        _save_style_corr(output),
        _save_factor_cum(output),
        _save_regression_r2(output),
        _save_specific_distribution(output),
        _save_te(output),
        _save_risk_stack(output),
    ]
    figs = [f for f in figs if f]

    panel_avail = _read_csv(output / "panel_availability.csv")
    descriptor_meta = _read_csv(output / "descriptor_metadata.csv")
    reg = _read_csv(output / "regression_diagnostics.csv")
    risk_diag = _read_csv(output / "risk_model_diagnostics.csv")
    port_summary = _read_csv(output / "portfolio_risk_summary.csv")
    style_cov = _read_csv(output / "style_coverage_by_year.csv")

    panel_summary = pd.DataFrame()
    if not panel_avail.empty:
        table_row = panel_avail[(panel_avail["field"] == "__table__") & (panel_avail["year"].astype(str) == "all")]
        if not table_row.empty:
            row = table_row.iloc[0]
            panel_summary = pd.DataFrame(
                [
                    {
                        "样本起点": row.get("start", ""),
                        "样本终点": row.get("end", ""),
                        "面板行数": int(row.get("rows", 0)),
                        "历史股票数": int(row.get("n_codes", 0)),
                    }
                ]
            )

    core_coverage = pd.DataFrame()
    if not panel_avail.empty:
        core_fields = ["amount", "analyst_report_count_90", "analyst_rating_score_180", "total_mv", "roe_ttm", "revenue_yoy", "n_cashflow_act_ttm"]
        core = panel_avail[panel_avail["field"].isin(core_fields)].copy()
        core = core[core["year"].astype(str) != "all"]
        if not core.empty:
            core_coverage = core.pivot_table(index="year", columns="field", values="coverage", aggfunc="first").reset_index()
            core_coverage = core_coverage[["year"] + [c for c in core_fields if c in core_coverage.columns]]
            for col in core_coverage.columns:
                if col != "year":
                    core_coverage[col] = pd.to_numeric(core_coverage[col], errors="coerce").round(3)

    descriptor_summary = pd.DataFrame()
    if not descriptor_meta.empty:
        descriptor_summary = (
            descriptor_meta.pivot_table(index="style", columns="availability", values="descriptor", aggfunc="count", fill_value=0)
            .reset_index()
        )

    style_cov_matrix = pd.DataFrame()
    if not style_cov.empty:
        style_cov_matrix = style_cov.pivot_table(index="year", columns="style", values="coverage", aggfunc="first").reset_index()
        style_cov_matrix["year"] = pd.to_numeric(style_cov_matrix["year"], errors="coerce").astype("Int64")
        rename = {col: STYLE_CN.get(col, col) for col in style_cov_matrix.columns if col != "year"}
        style_cov_matrix = style_cov_matrix.rename(columns=rename)
        for col in style_cov_matrix.columns:
            if col != "year":
                style_cov_matrix[col] = pd.to_numeric(style_cov_matrix[col], errors="coerce").round(3)

    universe_note = ""
    if not port_summary.empty and "out_of_model_weight_latest" in port_summary.columns:
        latest_out = pd.to_numeric(port_summary["out_of_model_weight_latest"], errors="coerce").max()
        if pd.notna(latest_out) and latest_out > 0.05:
            universe_note = (
                f"这次归因里最需要单独说明的是股票池问题：当前权重文件在最新调仓日仍有约 {latest_out:.1%} "
                "的组合权重不在当日中证500风险模型面板里。也就是说，前面那版 XGB 权重并没有严格落在当日中证500成分股内，"
                "这不应该被解释成行业 unknown，而应该被看作组合构造层面的约束缺口。下面的 TE 和风险占比可以作为诊断，"
                "但不能直接当作一个合格中证500指数增强组合的最终风险验收结果。"
            )

    reg_summary = pd.DataFrame()
    if not reg.empty and "r2" in reg.columns:
        ok = reg[reg["regression_status"] == "ok"]
        reg_summary = pd.DataFrame(
            [
                {
                    "成功回归天数": len(ok),
                    "总交易日": len(reg),
                    "平均样本股票数": round(ok["n_obs"].mean(), 2),
                    "平均R2": round(ok["r2"].mean(), 4),
                    "R2中位数": round(ok["r2"].median(), 4),
                    "平均条件数": round(ok["condition_number"].mean(), 2),
                }
            ]
        )

    style_explain = pd.DataFrame(
        [
            ["Size", "规模", "用市值和非线性市值刻画大小盘风险。中证500内部也有明显规模差异，组合如果偏向更小或更大的股票，会带来系统性风格暴露。"],
            ["Volatility", "波动率", "用Beta、历史波动和价格区间刻画高波动风险。高波动股票对市场冲击更敏感，也更容易贡献跟踪误差。"],
            ["Liquidity", "流动性", "用换手率和成交额刻画交易活跃度。流动性影响调仓冲击、拥挤交易和交易成本。"],
            ["Momentum", "动量", "用中长期收益和短期反转刻画趋势暴露。它不是单纯 alpha，而是解释组合是否押注趋势风格。"],
            ["Value", "价值", "用BP、EP、SP和现金流收益率刻画估值。价值暴露反映组合是否偏向低估值公司。"],
            ["Growth", "成长", "用收入、利润、ROE和资产周转变化刻画成长属性。成长暴露反映组合是否偏向扩张或效率改善的公司。"],
            ["Quality", "质量", "用ROE、ROA、毛利率、现金流质量和低杠杆刻画经营质量。质量暴露通常和盈利稳定性有关。"],
            ["Dividend Yield", "股息率", "用股息率相关字段刻画现金回报属性。高股息通常偏防御。"],
            ["Sentiment", "分析师预期情绪", "用 Tushare `report_rc` 的卖方研报覆盖、机构覆盖、评级、目标价空间和 EPS 预测修正刻画分析师预期。它比资金流更接近 Barra 语境里的 Sentiment，但也会受到卖方覆盖偏差影响。"],
        ],
        columns=["英文名称", "中文归类", "经济含义"],
    )

    decomposition_map = pd.DataFrame(
        [
            [
                "国家因子",
                "`country` 暴露",
                "单市场 A 股模型里通常近似为所有股票都等于 1，用来吸收当天市场共同涨跌。",
                "`factor_returns.csv` 里的 `country`",
            ],
            [
                "行业因子",
                "`industry_*` 哑变量",
                "股票属于哪个行业就暴露在哪个行业上，用来解释行业层面的共同涨跌。",
                "`factor_returns.csv` 里的行业列；面板中的行业分类字段",
            ],
            [
                "风格因子",
                "`style_size` 等九类风格暴露",
                "用规模、波动、流动性、动量、价值、成长、质量、股息率和分析师预期情绪解释跨行业的共同风格收益。",
                "`style_exposures.parquet` 和 `factor_returns.csv` 里的风格列",
            ],
            [
                "个股特异收益",
                "`specific_return`",
                "行业和风格都解释不了的剩余部分，代表个股自己的公告、交易冲击、定价误差或模型未覆盖信息。",
                "`specific_returns.parquet`",
            ],
            [
                "个股特异风险",
                "`specific_risk_60/120/252`",
                "不是收益分解里的当日收益项，而是用历史特异收益滚动标准差估计出来的个股残差风险。",
                "`specific_risk.parquet`",
            ],
        ],
        columns=["组成部分", "本项目里的变量", "我怎么理解", "落盘位置"],
    )

    lines = [
        "# CNE6 Barra 风险模型复现报告",
        "",
        "日期：2026-07-29",
        "",
        "## 1. 研究目标和边界",
        "",
        "这次复现的目标是搭建一个基于公开资料和 Tushare 可得数据的 CNE6-style 风险模型，用来解释中证500指数增强组合的行业、风格和个股特异风险。这里不把结果表述为 MSCI 商业 Barra CNE6 的精确复制，因为商业模型的描述子权重、部分数据源和协方差处理细节并不公开。",
        "",
        "本轮已经把样本从原来的 2018 年起点前推到 2010 年。行情、估值、资金流、三张财务报表和主要财务指标都尽量使用 Tushare 真实数据；2010 年财务 TTM 指标覆盖偏低，主要是因为刚开始时还没有足够季度公告可以拼出完整滚动指标，这一点在解释早期 Growth 和 Quality 风格时需要保守。",
        "",
        "本次更新主要处理导师指出的 Sentiment 口径问题：旧版用大单、超大单和净资金流作为资金情绪代理；新版已经改用 Tushare `report_rc` 结构化卖方盈利预测数据，基于研报覆盖、覆盖机构数、评级、目标价空间和 EPS 预测修正来刻画分析师预期情绪。资金流数据仍保留在面板里，方便旧公开因子或其他诊断复用，但不再作为 CNE6 Sentiment 的默认合成口径。",
        "",
        _md_table(panel_summary, max_rows=5),
        "",
        "## 2. Barra 收益分解框架",
        "",
        "导师提到的“国家因子 + 行业因子 + 风格因子 + 个股特异收益”，对应的就是 Barra 风险模型最核心的收益拆解。它不是先去预测哪只股票会涨，而是先把一只股票已经发生的收益拆成几类来源：市场整体涨跌解释多少，行业共同涨跌解释多少，风格暴露解释多少，最后还剩多少是这只股票自己的特异部分。",
        "",
        "我这里使用的日度横截面模型可以写成：",
        "",
        "```text",
        "r_i = X_country,i * f_country",
        "    + sum_k X_industry,i,k * f_industry,k",
        "    + sum_m X_style,i,m * f_style,m",
        "    + epsilon_i",
        "```",
        "",
        "其中 `r_i` 是股票 i 的下一日收益，`X` 是股票在国家、行业和风格上的暴露，`f` 是当天回归估计出来的因子收益，`epsilon_i` 就是个股特异收益。因为我们只在 A 股中证500股票池里做复现，所以国家因子在这里更像一个“样本内市场共同项”或截距项；它不等同于全球多国家模型里的中国国家配置因子，但数学位置是一样的。",
        "",
        _md_table(decomposition_map, max_rows=10),
        "",
        "举一个更直观的例子：如果某只半导体股票某天上涨 3%，模型可能把它拆成市场共同项贡献 1.0%、半导体行业贡献 1.2%、成长和动量等风格贡献 0.6%、个股特异收益贡献 0.2%。这样做的意义是，后面看指数增强组合时，我不只看组合有没有跑赢指数，还要看超额收益和跟踪误差到底来自行业押注、风格偏离，还是个股选择。",
        "",
        "## 3. 风格因子归类和经济含义",
        "",
        _md_table(style_explain, max_rows=20),
        "",
        "## 4. 数据覆盖",
        "",
        "下面这张表只挑核心字段看覆盖率。2010 年财务覆盖低，主要是 TTM 和同比指标需要等足够的季度公告；2012 年之后核心财务字段基本进入稳定状态。`analyst_report_count_90` 和 `analyst_rating_score_180` 用来检查本次新接入的分析师预期数据覆盖。这里要注意，研报条数和覆盖机构数把“没有研报”记为 0，所以字段覆盖率会是 1；评级、目标价和 EPS 修正的非空率更能反映真实有效研报信息的覆盖。",
        "",
        _md_table(core_coverage, max_rows=25),
        "",
        "## 5. 描述子可用性",
        "",
        _md_table(descriptor_summary, max_rows=20),
        "",
        _md_table(descriptor_meta[["descriptor", "style", "availability", "is_available", "missing_columns"]], max_rows=40) if not descriptor_meta.empty else "无描述子元数据。",
        "",
        "## 6. 风格暴露覆盖率",
        "",
        _md_table(style_cov_matrix, max_rows=25) if not style_cov_matrix.empty else "无风格覆盖率数据。",
        "",
    ]
    for fig in figs:
        lines.extend([f"![](../outputs/cne6_reproduction/{fig})", ""])
    lines.extend(
        [
            "## 7. 横截面回归结果",
            "",
            "这一节就是上面收益分解公式的实际估计步骤：每日横截面回归使用下一日个股收益作为被解释变量，解释变量包括 `country`、行业哑变量和九类风格暴露。回归权重使用市值平方根，目的是让大市值股票在风险模型估计中更稳定，但又不完全由最大市值股票主导。回归每天输出一组国家、行业、风格因子收益，同时把无法解释的残差落成个股特异收益。",
            "",
            _md_table(reg_summary, max_rows=5),
            "",
            "## 8. 风险模型诊断",
            "",
            "本次按导师意见把正式风险归因使用的 252 日因子协方差改成 Ledoit-Wolf 收缩协方差矩阵，并对 `country + style + industry` 全部共同因子输出完整矩阵。60 日和 120 日窗口仍保留 compact 样本协方差，主要用于轻量诊断。这样做比原来的行业对角方差更接近完整风险模型：行业之间、行业和风格之间的相关性都会进入预测跟踪误差。",
            "",
            _md_table(risk_diag, max_rows=20),
            "",
        "## 9. 指数增强组合风险归因",
        "",
        _md_table(port_summary, max_rows=20),
        "",
        "这里的 `style_var_share`、`industry_var_share` 和 `specific_var_share` 是把各自风险块单独除以总方差后的结果。由于本版已经使用完整共同因子协方差矩阵，风格和行业之间存在交叉协方差项，所以这三列不一定严格加总为 1；它们更适合用来观察主要风险来源，而不是当作互斥分解比例。",
        "",
        universe_note,
        "",
        "如果只看模型内可以解释的部分，预测跟踪误差仍然明显高于用日度超额收益滚动估计的实现跟踪误差。本版已经把 252 日行业协方差从对角近似改成 Ledoit-Wolf 全矩阵，行业风险计算也改成行业主动暴露向量乘行业协方差子矩阵。这个结果说明模型已经能识别风险来源，但 Tushare 细行业数量仍然偏多，后续还需要合并到更稳定的一级行业口径。",
        "",
        "## 10. 主要局限",
        "",
            "1. 这仍然是 CNE6-style 公开可复现版本，不是商业 Barra CNE6 的精确复制。",
            "2. 2010 年早期财务 TTM 和成长类暴露覆盖不足，2011 年开始改善，2012 年后核心财务字段基本稳定。",
            "3. 行业分类使用 Tushare 可得行业字段，粒度偏细，和商业 CNE6 行业口径不同。",
            "4. Sentiment 已改用 Tushare `report_rc` 结构化卖方盈利预测数据，但卖方覆盖本身有市值、行业和机构偏差。",
            "5. 252 日协方差已经使用 Ledoit-Wolf 全因子矩阵，但 60 日和 120 日窗口仍保留 compact 诊断输出；行业口径仍是 Tushare 细行业。",
            "6. 当前 XGB 指数增强权重存在明显股票池外权重，下一步必须先把组合构造约束修正到当日中证500成分股内，再谈正式指增归因。",
        "",
        "## 11. 下一步",
        "",
            "下一步应该先回到组合构造层，把 XGB 权重严格限制在当日中证500成分股内，并补上行业和风格约束的验收表；然后再用这套 2010-2026 风险模型重跑组合归因。风险模型本身后续可以继续做两件事：一是把 Tushare 细行业映射到更稳定的一级行业口径，二是如果 `research_report` 文本权限稳定，再尝试把研报摘要做成 NLP 情绪扩展，而不是只使用结构化预测字段。",
            "",
        ]
    )
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote CNE6 report to {doc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/cne6_reproduction")
    parser.add_argument("--doc", default="docs/cne6_barra_risk_model_reproduction_2026-07-22.md")
    args = parser.parse_args()
    generate_report(Path(args.output), Path(args.doc))


if __name__ == "__main__":
    main()
