# CNE6 Sentiment 改造成分析师预期因子设计

日期：2026-07-29

## 1. 背景

当前 CNE6-style 风险模型里的 Sentiment 风格因子使用的是资金流 proxy，包括大单净流入、特大单净流入和 60 日净资金流动量。这一版虽然能跑通，但导师指出账户里已经有分析师研报相关数据，因此 Sentiment 不应该继续主要依赖资金流。这个判断是合理的：Barra 风格模型里的 Sentiment 更接近“分析师预期、评级变化、盈利预测修正”这一类信息，而不是纯交易资金流。

Tushare 官方文档里有两个相关接口：

1. `report_rc`：券商卖方盈利预测数据，历史从 2010 年开始，字段包括研报日期、机构、作者、预测 EPS、预测 PE、预测股息率、预测 ROE、评级、最高/最低目标价等。
2. `research_report`：券商研究报告文本和下载链接，历史从 2017 年开始，且需要单独权限。

本轮优先使用 `report_rc`，因为它是结构化数据、覆盖期更长，能和 2010 年起的中证500风险模型对齐。`research_report` 暂时作为后续 NLP 扩展，不进入第一版重跑。

## 2. 目标

把 CNE6-style Sentiment 从“资金流代理”改成“分析师预期代理”，并把口径落实到数据、面板、描述子、暴露、回归和报告中。

核心结果应该包括：

1. 能从 Tushare 拉取并缓存 `report_rc` 数据。
2. 能把研报数据按 `report_date` 点位化合并到日频股票面板，避免未来函数。
3. 新 Sentiment 描述子默认来自分析师覆盖、评级、目标价空间和盈利预测修正。
4. 资金流描述子降级为 fallback，不再作为默认 Sentiment 的主口径。
5. 报告中明确说明 Sentiment 已经改成分析师预期代理，并保留覆盖率和局限性说明。

## 3. 数据口径

`report_rc` 原始数据按研报发布日期进入模型。任意交易日只能使用该日之前已经发布的研报信息。为了避免报告日当天盘中不可得的问题，第一版使用 `report_date + 1 个自然日` 作为可用日，再通过 `merge_asof` 合并到日频面板。

需要拉取字段：

| 字段 | 用途 |
|---|---|
| `ts_code` | 股票代码 |
| `report_date` | 研报日期，用于点位化 |
| `report_title` | 保留审计，不直接做数值因子 |
| `report_type` | 保留审计 |
| `classify` | 保留审计 |
| `org_name` | 机构覆盖数 |
| `author_name` | 分析师覆盖数 |
| `quarter` | 预测对应报告期 |
| `op_rt` | 预测营业收入 |
| `op_pr` | 预测营业利润 |
| `tp` | 预测利润总额 |
| `np` | 预测净利润 |
| `eps` | 预测 EPS |
| `pe` | 预测 PE |
| `rd` | 预测股息率 |
| `roe` | 预测 ROE |
| `ev_ebitda` | 预测 EV/EBITDA |
| `rating` | 卖方评级 |
| `max_price` | 最高目标价 |
| `min_price` | 最低目标价 |
| `imp_dg` | 机构关注度 |
| `create_time` | 数据更新时间，保留审计 |

## 4. 特征构造

第一版先构造 5 类日频特征：

| 特征 | 计算口径 | 经济含义 |
|---|---|---|
| `analyst_report_count_90` | 过去 90 天研报条数 | 覆盖度越高，说明卖方关注越充分，信息透明度和机构关注度更高 |
| `analyst_org_count_180` | 过去 180 天覆盖机构数 | 多机构覆盖比单一机构覆盖更稳，代表市场共识更充分 |
| `analyst_rating_score_180` | 过去 180 天评级映射均值 | 买入/增持类评级多，代表卖方主观态度更乐观 |
| `analyst_target_upside_180` | 过去 180 天目标价均值相对当前收盘价的上行空间 | 目标价越高，代表卖方对未来价格空间越乐观 |
| `analyst_eps_revision_180` | 最近 180 天预测 EPS 均值相对前 180 天的变化 | 盈利预测被上调，通常反映基本面预期改善 |

如果目标价或 EPS 字段覆盖不足，对应描述子自然变成缺失；暴露合成时只对当日可用描述子求均值，不用资金流硬填。这样报告中能真实反映分析师数据覆盖情况。

## 5. 代码改动

| 模块 | 改动 |
|---|---|
| `factors/data/cne6_fetcher.py` | 增加 `report_rc` 表字段和按日期拉取逻辑 |
| `factors/data/cne6_builder.py` | 加载 `report_rc` 缓存，构造点位化分析师特征，并合并到 CNE6 日频面板 |
| `factors/risk/cne6_descriptors.py` | 把 Sentiment 描述子从资金流 proxy 改成分析师预期 proxy |
| `factors/risk/cne6_exposures.py` | 计算新的分析师描述子暴露 |
| `factors/reports/cne6_report.py` | 修改 Sentiment 的中文解释、数据覆盖字段和局限性表述 |
| `test_cne6_risk_model.py` | 增加分析师特征点位化和描述子可用性 smoke test |

## 6. 验证方式

1. 先用小样本 `report_rc` 拉取测试接口权限和字段可用性。
2. 拉取 2010-2026 中证500历史成分股对应 `report_rc` 数据。
3. 重建 `data/processed/cne6_csi500_daily_panel.parquet`。
4. 重跑 `factors.risk.cne6_exposures`、`factors.risk.cne6_regression`、`factors.risk.cne6_risk_model` 和 `factors.reports.cne6_report`。
5. 检查 `descriptor_metadata.csv` 中 Sentiment 描述子是否来自分析师数据。
6. 检查 `style_coverage_by_year.csv` 中 `style_sentiment` 的覆盖率，重点确认 2010 年早期覆盖是否偏低。
7. 跑 CNE6 smoke tests，确认点位化没有未来函数。

## 7. 风险和边界

这次仍然不是商业 Barra CNE6 的精确复制，而是在公开接口下把 Sentiment 口径尽量贴近“分析师情绪/预期修正”。主要风险有三点：

1. `report_rc` 权限或历史数据量可能受账户权限限制，如果接口不可用，需要在报告里如实说明。
2. A 股卖方研报覆盖有明显市值和行业偏差，Sentiment 覆盖率本身也是一种机构关注度偏差。
3. 评级文字不完全标准化，需要做保守映射，无法识别的评级不强行赋值。
