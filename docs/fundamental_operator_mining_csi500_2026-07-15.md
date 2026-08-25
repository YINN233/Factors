# 中证500基本面算子组合挖掘报告

日期：2026-07-15

## 1. 任务目标

这次导师的要求是：不要只停留在 `quality_growth_hmean`、`eps_bps_value_quality`、`industry_neutral_roe_value_pb` 这三个已有基本面因子上，而是继续利用利润表、资产负债表、现金流量表和财务指标表，把更多基本面信息拆成算子，再通过组合去找新的有效因子。

所以我这次不是简单把旧因子换个名字，而是单独做了一套基本面算子工厂：先把基础指标都处理成“越大越好”的原子信号，再用均值、调和平均、估值扣减、行业中性、规模中性这些算子去组合，最后放到中证500股票池里做 train/valid/test/2026YTD 验证和相关性去重。

## 2. 我这次怎么做

我把这件事拆成四步：

1. 先保留旧三因子和已有 legacy 基本面因子，把它们当成对照组。
2. 再从利润表、资产负债表、现金流量表、财务指标表和估值字段里拆出一批方向统一的原子指标。
3. 然后用均值、调和平均、估值扣减、行业中性、规模中性等算子去做组合。
4. 最后在中证500的 train/valid/test/2026YTD 上看 RankIC 稳定性，并用相关性去重得到推荐集合。

这次实际跑的参数如下：

| parameter | value |
| --- | --- |
| processed_dir | data/processed |
| suffix | 000905_SH |
| n_candidates | 220 |
| include_atoms | True |
| include_neutralized | True |
| candidate_regex |  |
| max_candidates |  |
| include_turnover | False |
| min_abs_ic | 0.003 |
| min_adj_rankic | 0.005 |
| min_ytd_rankic | 0.0 |
| min_coverage | 0.55 |
| max_pair_corr | 0.85 |
| ytd_year | 2026 |

## 3. 数据口径

| 项目 | 口径 |
| --- | --- |
| 股票池 | 中证500历史成分股 |
| 日频面板 | `data/processed/*_fundamental_000905_SH.parquet` |
| 基本面 PIT | 按公告日 `available_date` 向后对齐，避免未来函数 |
| 训练/验证/测试 | 沿用现有 `train/valid/test` 拆分 |
| 近期观察 | 从 test 中单独切出 `2026YTD` |
| 标签 | 现有 `label`，即未来收益/排名评估口径 |

## 4. 我搭的算子体系

| 算子 | 例子 | 经济含义 |
| --- | --- | --- |
| 方向统一原子 | `low_pb = 1 - rank(pb)` | 所有基础指标都转成越大越好 |
| 比率算子 | `free_cashflow_ttm / total_mv` | 用现金流、资产、负债等构造可比指标 |
| 均值组合 | `mean(roe, low_pb)` | 多个维度等权确认 |
| 调和平均 | `harm_mean(revenue_yoy, net_profit_yoy)` | 惩罚单腿增长 |
| 三腿组合 | `roe + cashflow_to_profit + low_pb` | 质量、现金流、估值同时约束 |
| 行业中性 | `industry_neutralize(score)` | 降低行业结构差异 |
| 规模中性 | `size_neutralize(score)` | 降低市值风格暴露 |

先看候选因子的分布：

| family | n_candidates |
| --- | --- |
| fundamental_value | 78 |
| fundamental_growth | 45 |
| fundamental_efficiency | 39 |
| fundamental_safety | 29 |
| fundamental_quality | 22 |
| mixed | 7 |

再看筛选结果：

| status | count |
| --- | --- |
| evaluated | 220 |
| passes_stability | 90 |
| selected_after_corr | 52 |

我自己的核心判断：

1. 这次最强的一组不是单纯成长，而是“现金流/利润率质量 + 低估值”：`operating_cf_margin + low_pb`、`net_margin + low_pb`、`profit_to_assets + low_pb` 排在最前。
2. PB/PS 相关估值约束比单独成长更稳，说明中证500里“盈利质量不差且估值不贵”的组合仍然有显著横截面区分度。
3. 原三因子仍然有效，但会被新工厂生成的近似等价或更细颗粒组合在相关性去重时替代；这不是失效，而是说明旧因子逻辑被更系统地展开了。
4. 推荐集合以价值质量因子为主，同时保留成长、安全、效率、现金流质量和少量混合量价确认因子，便于后续进入 XGBoost 或指数增强模型时做消融。

去重后推荐因子族分布：

| family | n_selected |
| --- | --- |
| fundamental_value | 25 |
| fundamental_growth | 9 |
| fundamental_safety | 8 |
| fundamental_efficiency | 7 |
| fundamental_quality | 2 |
| mixed | 1 |

## 5. 最后我会重点看的因子

下面这张表是我做完稳定性过滤和相关性去重后留下来的推荐因子。`min_adj_rankic` 可以理解成 train/valid/test 三段里最差的一段表现，越高说明这个因子不是只在某一段偶然有效。

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_operating_cf_margin_low_pb | fundamental_value | positive | 4.08% | 4.05% | 4.11% | 2.94% | 4.05% | 4.08% | 75.16% | mean(rank(operating_cf_margin_ttm), 1 - rank(pb)) | operating_cf_margin quality combined with low_pb valuation. |
| fund_combo_net_margin_low_pb | fundamental_value | positive | 3.53% | 3.18% | 4.23% | 4.49% | 3.18% | 3.65% | 75.16% | mean(rank(net_margin_ttm), 1 - rank(pb)) | net_margin quality combined with low_pb valuation. |
| fund_combo_profit_to_assets_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(net_profit_ttm / total_assets), 1 - rank(pb)) | profit_to_assets quality combined with low_pb valuation. |
| robust_margin_value_ps | fundamental_value | positive | 3.02% | 3.47% | 4.21% | 4.51% | 3.02% | 3.57% | 75.40% | robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm) | Robust margin quality adjusted by sales valuation. |
| fund_size_neu_eps_bps_low_pb | fundamental_value | positive | 2.94% | 3.37% | 3.41% | 5.32% | 2.94% | 3.24% | 95.27% | size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | Per-share earnings and book value at lower book valuation. Size-neutralized to reduce market-cap style exposure. |
| fund_combo_operating_cf_margin_low_ps | fundamental_value | positive | 2.83% | 3.39% | 3.51% | 2.95% | 2.83% | 3.24% | 75.40% | mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)) | operating_cf_margin quality combined with low_ps valuation. |
| fund_combo_revenue_profit_low_pe_hmean | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm)) | Synchronous growth bought at lower earnings valuation. |
| fund_ind_neu_ocf_to_assets_low_pb | fundamental_value | positive | 2.90% | 2.53% | 4.01% | 3.97% | 2.53% | 3.14% | 75.16% | industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | ocf_to_assets quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_operating_cf_margin_low_pe | fundamental_value | positive | 3.57% | 2.88% | 2.51% | 2.64% | 2.51% | 2.99% | 64.82% | mean(rank(operating_cf_margin_ttm), 1 - rank(pe_ttm)) | operating_cf_margin quality combined with low_pe valuation. |
| fund_combo_roa_ocf_low_pb | fundamental_value | positive | 2.37% | 2.32% | 3.63% | 5.84% | 2.32% | 2.78% | 75.16% | mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)) | Asset-level profitability and cash generation at lower book valuation. |
| fund_combo_eps_low_ps | fundamental_value | positive | 2.73% | 2.24% | 3.04% | 4.97% | 2.24% | 2.67% | 95.51% | mean(rank(eps), 1 - rank(ps_ttm)) | eps quality combined with low_ps valuation. |
| fund_ind_neu_roe_cash_low_pb | fundamental_value | positive | 2.88% | 2.24% | 3.58% | 4.41% | 2.24% | 2.90% | 75.16% | industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb))) | ROE and cash conversion at lower book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_net_margin_low_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | mean(rank(net_margin_ttm), 1 - rank(ps_ttm)) | net_margin quality combined with low_ps valuation. |
| fund_combo_revenue_profit_cash_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit)) | Revenue and profit grow together and are confirmed by cash conversion. |
| fund_ind_neu_roe_low_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | roe quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| shareholder_yield_quality | fundamental_value | positive | 1.85% | 2.69% | 2.01% | 3.49% | 1.85% | 2.18% | 56.20% | rank(dv_ttm) + rank(roe_ttm) | Dividend yield plus profitability, preferring shareholder-return quality. |
| fund_atom_dividend_yield | fundamental_value | positive | 1.74% | 5.15% | 3.19% | 1.85% | 1.74% | 3.36% | 73.66% | rank(dv_ttm) | Higher dividend yield. Source: valuation overlay. |
| fund_combo_eps_low_pe | fundamental_value | positive | 2.17% | 1.71% | 1.97% | 4.03% | 1.71% | 1.95% | 80.16% | mean(rank(eps), 1 - rank(pe_ttm)) | eps quality combined with low_pe valuation. |
| fund_size_neu_cash_safety_low_pb | fundamental_safety | positive | 2.60% | 1.66% | 2.20% | 1.11% | 1.66% | 2.15% | 94.99% | size_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb))) | Cash buffer, lower leverage, and lower book valuation. Size-neutralized to reduce market-cap style exposure. |
| fund_combo_quick_ratio_low_pe | fundamental_safety | positive | 2.05% | 1.58% | 1.58% | 0.36% | 1.58% | 1.74% | 76.35% | mean(rank(quick_ratio), 1 - rank(pe_ttm)) | quick_ratio balance-sheet safety combined with low_pe valuation. |
| fund_combo_receivable_turnover_low_pb | fundamental_efficiency | positive | 1.51% | 4.82% | 2.87% | 0.26% | 1.51% | 3.06% | 92.25% | mean(rank(ar_turn), 1 - rank(pb)) | receivable_turnover operating efficiency combined with low_pb. |
| fund_atom_inventory_turnover | fundamental_efficiency | positive | 1.61% | 1.46% | 1.70% | 2.59% | 1.46% | 1.59% | 91.77% | rank(inv_turn) | Higher inventory turnover. Source: fina_indicator. |
| fund_combo_net_margin_low_pe | fundamental_value | positive | 3.48% | 1.46% | 1.86% | 2.80% | 1.46% | 2.27% | 64.82% | mean(rank(net_margin_ttm), 1 - rank(pe_ttm)) | net_margin quality combined with low_pe valuation. |
| fund_combo_profit_to_assets_low_ps | fundamental_value | positive | 1.45% | 1.45% | 2.81% | 3.88% | 1.45% | 1.90% | 75.40% | mean(rank(net_profit_ttm / total_assets), 1 - rank(ps_ttm)) | profit_to_assets quality combined with low_ps valuation. |
| fund_combo_cash_to_assets_low_pe | fundamental_safety | positive | 1.42% | 2.29% | 1.69% | 0.10% | 1.42% | 1.80% | 78.25% | mean(rank(money_cap / total_assets), 1 - rank(pe_ttm)) | cash_to_assets balance-sheet safety combined with low_pe valuation. |
| fund_ind_neu_dupont_low_pb | fundamental_value | positive | 3.28% | 1.36% | 3.88% | 4.81% | 1.36% | 2.84% | 75.16% | industry_neutralize(mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb))) | DuPont margin and turnover quality adjusted by book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_ind_neu_eps_bps_low_pb | fundamental_value | positive | 2.34% | 1.34% | 2.47% | 5.75% | 1.34% | 2.05% | 95.27% | industry_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | Per-share earnings and book value at lower book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_quick_ratio_low_pb | fundamental_safety | positive | 1.33% | 3.17% | 3.24% | 0.04% | 1.33% | 2.58% | 92.96% | mean(rank(quick_ratio), 1 - rank(pb)) | quick_ratio balance-sheet safety combined with low_pb valuation. |
| fund_combo_ocf_to_assets_fcf_yield | fundamental_value | positive | 1.52% | 1.29% | 1.78% | 3.28% | 1.29% | 1.53% | 72.95% | mean(rank(n_cashflow_act_ttm / total_assets), rank(free_cashflow_ttm / total_mv)) | ocf_to_assets quality combined with fcf_yield valuation. |
| fund_combo_inventory_turnover_low_pb | fundamental_efficiency | positive | 1.29% | 4.11% | 3.84% | 1.24% | 1.29% | 3.08% | 91.41% | mean(rank(inv_turn), 1 - rank(pb)) | inventory_turnover operating efficiency combined with low_pb. |
| fund_ind_neu_working_capital_cash_growth | fundamental_growth | positive | 1.27% | 1.29% | 1.34% | 2.35% | 1.27% | 1.30% | 63.49% | industry_neutralize(mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit), rank(net_profit_yoy))) | Clean growth with cash conversion and low working-capital pressure. Industry-neutralized to reduce sector structure bias. |
| fund_combo_low_debt_assets_low_pe | fundamental_safety | positive | 1.68% | 1.22% | 2.01% | 0.65% | 1.22% | 1.64% | 80.16% | mean(1 - rank(debt_to_assets), 1 - rank(pe_ttm)) | low_debt_assets balance-sheet safety combined with low_pe valuation. |
| fund_combo_net_profit_yoy_accrual_quality | fundamental_growth | positive | 1.66% | 1.64% | 1.21% | 1.37% | 1.21% | 1.50% | 74.30% | mean(rank(net_profit_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)) | net_profit_yoy growth confirmed by accrual_quality quality. |
| fund_ind_neu_cash_safety_low_pb | fundamental_safety | positive | 1.62% | 1.16% | 2.34% | 0.60% | 1.16% | 1.71% | 94.99% | industry_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb))) | Cash buffer, lower leverage, and lower book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_ind_neu_roa_ocf_low_pb | fundamental_value | positive | 3.64% | 1.15% | 3.06% | 5.03% | 1.15% | 2.62% | 75.16% | industry_neutralize(mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | Asset-level profitability and cash generation at lower book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_revenue_yoy_accrual_quality | fundamental_growth | positive | 1.14% | 1.76% | 1.70% | 2.26% | 1.14% | 1.54% | 74.30% | mean(rank(revenue_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)) | revenue_yoy growth confirmed by accrual_quality quality. |
| industry_rank_cash_profit_cover | fundamental_quality | positive | 1.21% | 1.76% | 1.13% | 0.81% | 1.13% | 1.36% | 75.52% | group_rank(n_cashflow_act_ttm / net_profit_ttm, industry) | Profit cash coverage ranked within each industry. |
| turnover_efficiency_combo | fundamental_efficiency | positive | 2.08% | 2.14% | 1.05% | 2.00% | 1.05% | 1.76% | 91.35% | rank(inv_turn) + rank(ar_turn) | Inventory and receivable turnover efficiency. |
| fund_combo_cash_to_liab_fcf_yield | fundamental_safety | positive | 1.68% | 1.00% | 1.02% | 0.37% | 1.00% | 1.24% | 72.95% | mean(rank(cash_to_liab), rank(free_cashflow_ttm / total_mv)) | cash_to_liab balance-sheet safety combined with fcf_yield valuation. |
| cash_revenue_quality | fundamental_quality | positive | 2.37% | 1.00% | 1.43% | 5.24% | 1.00% | 1.60% | 75.52% | rank(operating_cf_margin_ttm) + rank(ocf_to_or) | Operating cash flow relative to revenue, combining derived TTM and reported indicators. |
| clean_growth_quality | fundamental_growth | positive | 0.94% | 2.58% | 2.07% | 3.11% | 0.94% | 1.86% | 63.49% | rank(cashflow_to_profit) + rank(net_profit_yoy) - rank(working_capital_pressure) | Profit growth with cash conversion and low working-capital pressure. |
| fund_combo_low_working_capital_pressure_cashflow_to_profit | fundamental_efficiency | positive | 0.94% | 3.07% | 1.96% | 1.17% | 0.94% | 1.99% | 64.66% | mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit)) | low_working_capital_pressure operating efficiency combined with cashflow_to_profit. |
| fund_combo_eps_fcf_yield | fundamental_value | positive | 2.99% | 0.88% | 1.73% | 5.82% | 0.88% | 1.87% | 72.94% | mean(rank(eps), rank(free_cashflow_ttm / total_mv)) | eps quality combined with fcf_yield valuation. |
| fund_combo_low_working_capital_pressure_net_margin | fundamental_efficiency | positive | 2.56% | 0.85% | 0.92% | 3.57% | 0.85% | 1.44% | 64.66% | mean(1 - rank(working_capital_pressure), rank(net_margin_ttm)) | low_working_capital_pressure operating efficiency combined with net_margin. |
| fund_combo_revenue_yoy_operating_cf_margin | fundamental_growth | positive | 5.26% | 0.83% | 1.95% | 6.31% | 0.83% | 2.68% | 74.30% | mean(rank(revenue_yoy), rank(operating_cf_margin_ttm)) | revenue_yoy growth confirmed by operating_cf_margin quality. |
| fund_combo_net_profit_yoy_operating_cf_margin | fundamental_growth | positive | 3.82% | 0.79% | 1.36% | 5.20% | 0.79% | 1.99% | 74.30% | mean(rank(net_profit_yoy), rank(operating_cf_margin_ttm)) | net_profit_yoy growth confirmed by operating_cf_margin quality. |
| fund_ind_neu_roe_low_pe | fundamental_value | positive | 2.96% | 0.78% | 1.51% | 3.39% | 0.78% | 1.75% | 64.82% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pe_ttm))) | roe quality combined with low_pe valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_low_debt_assets_fcf_yield | fundamental_safety | positive | 1.23% | 0.74% | 1.57% | 1.05% | 0.74% | 1.18% | 72.95% | mean(1 - rank(debt_to_assets), rank(free_cashflow_ttm / total_mv)) | low_debt_assets balance-sheet safety combined with fcf_yield valuation. |
| quality_liquidity_confirm_20 | mixed | positive | 2.84% | 0.71% | 0.64% | 2.84% | 0.64% | 1.40% | 74.84% | zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,20))) | Cash-flow quality confirmed by trading amount expansion. |
| fund_atom_revenue_yoy | fundamental_growth | positive | 3.00% | 0.64% | 1.71% | 5.10% | 0.64% | 1.78% | 74.31% | rank(revenue_yoy) | Revenue growth. Source: income. |
| fund_ind_neu_cashflow_to_profit_low_pb | fundamental_value | positive | 0.60% | 3.01% | 3.72% | 2.12% | 0.60% | 2.44% | 75.16% | industry_neutralize(mean(rank(cashflow_to_profit), 1 - rank(pb))) | cashflow_to_profit quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_receivable_turnover_net_margin | fundamental_efficiency | positive | 3.15% | 0.86% | 0.59% | 4.76% | 0.59% | 1.53% | 73.05% | mean(rank(ar_turn), rank(net_margin_ttm)) | receivable_turnover operating efficiency combined with net_margin. |

## 6. 通过稳定性检验但彼此可能相似的候选

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_operating_cf_margin_low_pb | fundamental_value | positive | 4.08% | 4.05% | 4.11% | 2.94% | 4.05% | 4.08% | 75.16% | mean(rank(operating_cf_margin_ttm), 1 - rank(pb)) | operating_cf_margin quality combined with low_pb valuation. |
| fund_size_neu_operating_cf_margin_low_pb | fundamental_value | positive | 3.99% | 3.38% | 3.92% | 4.14% | 3.38% | 3.76% | 75.16% | size_neutralize(mean(rank(operating_cf_margin_ttm), 1 - rank(pb))) | operating_cf_margin quality combined with low_pb valuation. Size-neutralized to reduce market-cap style exposure. |
| fund_combo_net_margin_low_pb | fundamental_value | positive | 3.53% | 3.18% | 4.23% | 4.49% | 3.18% | 3.65% | 75.16% | mean(rank(net_margin_ttm), 1 - rank(pb)) | net_margin quality combined with low_pb valuation. |
| fund_combo_profit_to_assets_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(net_profit_ttm / total_assets), 1 - rank(pb)) | profit_to_assets quality combined with low_pb valuation. |
| fund_combo_roa_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(roa_ttm), 1 - rank(pb)) | roa quality combined with low_pb valuation. |
| dupont_value_quality | fundamental_value | positive | 3.55% | 3.09% | 4.83% | 5.37% | 3.09% | 3.82% | 75.16% | rank(net_margin_ttm) + rank(asset_turnover_ttm) - rank(pb) | DuPont operating quality adjusted by book valuation. |
| fund_combo_dupont_low_pb | fundamental_value | positive | 3.55% | 3.09% | 4.83% | 5.37% | 3.09% | 3.82% | 75.16% | mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb)) | DuPont margin and turnover quality adjusted by book valuation. |
| robust_margin_value_ps | fundamental_value | positive | 3.02% | 3.47% | 4.21% | 4.51% | 3.02% | 3.57% | 75.40% | robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm) | Robust margin quality adjusted by sales valuation. |
| fund_size_neu_eps_bps_low_pb | fundamental_value | positive | 2.94% | 3.37% | 3.41% | 5.32% | 2.94% | 3.24% | 95.27% | size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | Per-share earnings and book value at lower book valuation. Size-neutralized to reduce market-cap style exposure. |
| fund_combo_operating_cf_margin_low_ps | fundamental_value | positive | 2.83% | 3.39% | 3.51% | 2.95% | 2.83% | 3.24% | 75.40% | mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)) | operating_cf_margin quality combined with low_ps valuation. |
| fund_combo_eps_low_pb | fundamental_value | positive | 2.76% | 3.90% | 4.48% | 5.80% | 2.76% | 3.71% | 95.27% | mean(rank(eps), 1 - rank(pb)) | eps quality combined with low_pb valuation. |
| fund_combo_revenue_profit_low_pe_hmean | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm)) | Synchronous growth bought at lower earnings valuation. |
| growth_value_balance | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) - rank(pe_ttm) | Fundamental growth adjusted for earnings valuation. |
| fund_ind_neu_ocf_to_assets_low_pb | fundamental_value | positive | 2.90% | 2.53% | 4.01% | 3.97% | 2.53% | 3.14% | 75.16% | industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | ocf_to_assets quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_operating_cf_margin_low_pe | fundamental_value | positive | 3.57% | 2.88% | 2.51% | 2.64% | 2.51% | 2.99% | 64.82% | mean(rank(operating_cf_margin_ttm), 1 - rank(pe_ttm)) | operating_cf_margin quality combined with low_pe valuation. |
| fund_combo_eps_bps_low_pb | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | mean(rank(eps), rank(bps), 1 - rank(pb)) | Per-share earnings and book value at lower book valuation. |
| eps_bps_value_quality | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | rank(eps) + rank(bps) - rank(pb) | Per-share earnings and book value quality adjusted by book valuation. |
| fund_combo_roa_ocf_low_pb | fundamental_value | positive | 2.37% | 2.32% | 3.63% | 5.84% | 2.32% | 2.78% | 75.16% | mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)) | Asset-level profitability and cash generation at lower book valuation. |
| fund_combo_eps_low_ps | fundamental_value | positive | 2.73% | 2.24% | 3.04% | 4.97% | 2.24% | 2.67% | 95.51% | mean(rank(eps), 1 - rank(ps_ttm)) | eps quality combined with low_ps valuation. |
| fund_ind_neu_roe_cash_low_pb | fundamental_value | positive | 2.88% | 2.24% | 3.58% | 4.41% | 2.24% | 2.90% | 75.16% | industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb))) | ROE and cash conversion at lower book valuation. Industry-neutralized to reduce sector structure bias. |
| fund_combo_net_margin_low_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | mean(rank(net_margin_ttm), 1 - rank(ps_ttm)) | net_margin quality combined with low_ps valuation. |
| margin_value_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | rank(net_margin_ttm) - rank(ps_ttm) | Net margin bought at a reasonable sales valuation. |
| fund_combo_roe_low_pb | fundamental_value | positive | 1.98% | 3.64% | 4.58% | 5.39% | 1.98% | 3.40% | 75.16% | mean(rank(roe_ttm), 1 - rank(pb)) | roe quality combined with low_pb valuation. |
| roe_value_pb | fundamental_value | positive | 1.98% | 3.64% | 4.58% | 5.39% | 1.98% | 3.40% | 75.16% | rank(roe_ttm) - rank(pb) | Quality at a reasonable book valuation. |
| fund_combo_revenue_profit_cash_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit)) | Revenue and profit grow together and are confirmed by cash conversion. |
| quality_growth_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit) | Growth that is simultaneous in revenue, profit, and cash conversion. |
| fund_ind_neu_roe_low_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | roe quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| industry_neutral_roe_value_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(rank(roe_ttm) - rank(pb)) | Quality-value score after removing same-industry average exposure. |
| fund_size_neu_profit_to_assets_low_pb | fundamental_value | positive | 1.86% | 3.34% | 4.81% | 5.01% | 1.86% | 3.34% | 75.16% | size_neutralize(mean(rank(net_profit_ttm / total_assets), 1 - rank(pb))) | profit_to_assets quality combined with low_pb valuation. Size-neutralized to reduce market-cap style exposure. |
| fund_size_neu_roa_low_pb | fundamental_value | positive | 1.86% | 3.34% | 4.81% | 5.01% | 1.86% | 3.34% | 75.16% | size_neutralize(mean(rank(roa_ttm), 1 - rank(pb))) | roa quality combined with low_pb valuation. Size-neutralized to reduce market-cap style exposure. |

## 7. 原来三个因子在这次结果里的位置

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_growth_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit) | Growth that is simultaneous in revenue, profit, and cash conversion. |
| eps_bps_value_quality | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | rank(eps) + rank(bps) - rank(pb) | Per-share earnings and book value quality adjusted by book valuation. |
| industry_neutral_roe_value_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(rank(roe_ttm) - rank(pb)) | Quality-value score after removing same-industry average exposure. |

我的理解是：原来的三个因子并没有失效，只是这次新算子把它们背后的逻辑拆得更细了。比如价值质量、现金流质量、营运效率这些维度里，会出现和旧因子很接近、但表达更细的版本。后面接入 XGBoost 或指数增强时，我不应该把所有通过的因子一股脑塞进去，而是优先用相关性去重后的推荐集合。

## 8. 我对经济含义的理解

我理解这批新算子主要强化了四类基本面信号：

1. 质量价值：高 ROE/ROA/利润率，同时 PB/PE/PS 不贵。
2. 现金流质量：经营现金流、自由现金流、现金利润覆盖共同确认盈利。
3. 营运效率：资产周转、存货周转、应收周转、低营运资本占用。
4. 安全边际：低负债、高现金覆盖、较强短期偿债能力。

这些因子之所以比单独看成长或单独看估值更稳，是因为它们通常要求两件事同时成立：公司本身不能太差，价格也不能太贵；或者至少要求利润表里的结果能被现金流量表和资产负债表验证。

## 9. 我逐个解释这些因子

下面我按最终推荐集合逐个解释。这里所有 `rank` 都是在同一交易日的中证500截面里算的，而且方向已经统一成分数越高越好。

### 1. `fund_combo_operating_cf_margin_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(operating_cf_margin_ttm), 1 - rank(pb))`。经营现金流利润率，衡量收入转成经营现金流的能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `4.08%`，valid RankIC `4.05%`，test RankIC `4.11%`，2026YTD RankIC `2.94%`，三段最小调整后 RankIC `4.05%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 2. `fund_combo_net_margin_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(net_margin_ttm), 1 - rank(pb))`。净利率，反映成本控制、议价能力和商业模式质量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `3.53%`，valid RankIC `3.18%`，test RankIC `4.23%`，2026YTD RankIC `4.49%`，三段最小调整后 RankIC `3.18%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 3. `fund_combo_profit_to_assets_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(net_profit_ttm / total_assets), 1 - rank(pb))`。利润/资产，接近 ROA 口径，衡量资产盈利效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `3.15%`，valid RankIC `3.23%`，test RankIC `4.68%`，2026YTD RankIC `5.04%`，三段最小调整后 RankIC `3.15%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 4. `robust_margin_value_ps`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm)`。净利率，反映成本控制、议价能力和商业模式质量；PS，衡量收入相对估值。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。
- 实证表现：train RankIC `3.02%`，valid RankIC `3.47%`，test RankIC `4.21%`，2026YTD RankIC `4.51%`，三段最小调整后 RankIC `3.02%`，覆盖率下限 `75.40%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 5. `fund_size_neu_eps_bps_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb)))`。每股收益，衡量单股盈利能力；每股净资产，衡量单股账面资产基础；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 实证表现：train RankIC `2.94%`，valid RankIC `3.37%`，test RankIC `3.41%`，2026YTD RankIC `5.32%`，三段最小调整后 RankIC `2.94%`，覆盖率下限 `95.27%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。该因子已削弱市值暴露，和风格约束组合时更稳。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 6. `fund_combo_operating_cf_margin_low_ps`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.83%`，valid RankIC `3.39%`，test RankIC `3.51%`，2026YTD RankIC `2.95%`，三段最小调整后 RankIC `2.83%`，覆盖率下限 `75.40%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 7. `fund_combo_revenue_profit_low_pe_hmean`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm))`。收入同比增长，衡量需求和业务扩张；利润同比增长，衡量增长是否落到利润表底线；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高；调和平均：惩罚单腿特别弱的公司。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。调和平均会惩罚单项短板，因此比简单相加更偏向“多条腿都不差”的稳健公司。
- 实证表现：train RankIC `2.82%`，valid RankIC `2.68%`，test RankIC `3.17%`，2026YTD RankIC `4.79%`，三段最小调整后 RankIC `2.68%`，覆盖率下限 `63.75%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 8. `fund_ind_neu_ocf_to_assets_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)))`。经营现金流/总资产，衡量资产产生现金流的效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `2.90%`，valid RankIC `2.53%`，test RankIC `4.01%`，2026YTD RankIC `3.97%`，三段最小调整后 RankIC `2.53%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 9. `fund_combo_operating_cf_margin_low_pe`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(operating_cf_margin_ttm), 1 - rank(pe_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `3.57%`，valid RankIC `2.88%`，test RankIC `2.51%`，2026YTD RankIC `2.64%`，三段最小调整后 RankIC `2.51%`，覆盖率下限 `64.82%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 10. `fund_combo_roa_ocf_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))`。经营现金流/总资产，衡量资产产生现金流的效率；ROA，衡量资产层面的盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.37%`，valid RankIC `2.32%`，test RankIC `3.63%`，2026YTD RankIC `5.84%`，三段最小调整后 RankIC `2.32%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 11. `fund_combo_eps_low_ps`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(eps), 1 - rank(ps_ttm))`。每股收益，衡量单股盈利能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.73%`，valid RankIC `2.24%`，test RankIC `3.04%`，2026YTD RankIC `4.97%`，三段最小调整后 RankIC `2.24%`，覆盖率下限 `95.51%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 12. `fund_ind_neu_roe_cash_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb)))`。经营现金流对利润的覆盖，验证会计利润含金量；ROE，衡量股东权益回报；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `2.88%`，valid RankIC `2.24%`，test RankIC `3.58%`，2026YTD RankIC `4.41%`，三段最小调整后 RankIC `2.24%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 13. `fund_combo_net_margin_low_ps`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(net_margin_ttm), 1 - rank(ps_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.12%`，valid RankIC `2.90%`，test RankIC `4.07%`，2026YTD RankIC `5.00%`，三段最小调整后 RankIC `2.12%`，覆盖率下限 `75.40%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 14. `fund_combo_revenue_profit_cash_hmean`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit))`。经营现金流对利润的覆盖，验证会计利润含金量；收入同比增长，衡量需求和业务扩张；利润同比增长，衡量增长是否落到利润表底线；调和平均：惩罚单腿特别弱的公司。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。调和平均会惩罚单项短板，因此比简单相加更偏向“多条腿都不差”的稳健公司。
- 实证表现：train RankIC `2.58%`，valid RankIC `1.97%`，test RankIC `2.57%`，2026YTD RankIC `6.04%`，三段最小调整后 RankIC `1.97%`，覆盖率下限 `74.30%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 15. `fund_ind_neu_roe_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb)))`。ROE，衡量股东权益回报；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `3.39%`，valid RankIC `1.88%`，test RankIC `3.82%`，2026YTD RankIC `5.03%`，三段最小调整后 RankIC `1.88%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 16. `shareholder_yield_quality`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `rank(dv_ttm) + rank(roe_ttm)`。ROE，衡量股东权益回报；股息率，衡量现金分红回报。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。股息率有效，是因为持续分红代表现金回报和治理约束，也给估值提供一定锚。
- 实证表现：train RankIC `1.85%`，valid RankIC `2.69%`，test RankIC `2.01%`，2026YTD RankIC `3.49%`，三段最小调整后 RankIC `1.85%`，覆盖率下限 `56.20%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 17. `fund_atom_dividend_yield`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `rank(dv_ttm)`。股息率，衡量现金分红回报。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。股息率有效，是因为持续分红代表现金回报和治理约束，也给估值提供一定锚。
- 实证表现：train RankIC `1.74%`，valid RankIC `5.15%`，test RankIC `3.19%`，2026YTD RankIC `1.85%`，三段最小调整后 RankIC `1.74%`，覆盖率下限 `73.66%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 18. `fund_combo_eps_low_pe`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(eps), 1 - rank(pe_ttm))`。每股收益，衡量单股盈利能力；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.17%`，valid RankIC `1.71%`，test RankIC `1.97%`，2026YTD RankIC `4.03%`，三段最小调整后 RankIC `1.71%`，覆盖率下限 `80.16%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 19. `fund_size_neu_cash_safety_low_pb`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `size_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb)))`。现金/负债，衡量现金对债务的覆盖；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 实证表现：train RankIC `2.60%`，valid RankIC `1.66%`，test RankIC `2.20%`，2026YTD RankIC `1.11%`，三段最小调整后 RankIC `1.66%`，覆盖率下限 `94.99%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。该因子已削弱市值暴露，和风格约束组合时更稳。

### 20. `fund_combo_quick_ratio_low_pe`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(rank(quick_ratio), 1 - rank(pe_ttm))`。速动比率，衡量短期偿债安全垫；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `2.05%`，valid RankIC `1.58%`，test RankIC `1.58%`，2026YTD RankIC `0.36%`，三段最小调整后 RankIC `1.58%`，覆盖率下限 `76.35%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。2026YTD 表现偏弱，我会考虑降权，或者先把它放在备选特征里。

### 21. `fund_combo_receivable_turnover_low_pb`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `mean(rank(ar_turn), 1 - rank(pb))`。应收账款周转率，衡量回款效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `1.51%`，valid RankIC `4.82%`，test RankIC `2.87%`，2026YTD RankIC `0.26%`，三段最小调整后 RankIC `1.51%`，覆盖率下限 `92.25%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。2026YTD 表现偏弱，我会考虑降权，或者先把它放在备选特征里。

### 22. `fund_atom_inventory_turnover`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `rank(inv_turn)`。存货周转率，衡量库存消化效率。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `1.61%`，valid RankIC `1.46%`，test RankIC `1.70%`，2026YTD RankIC `2.59%`，三段最小调整后 RankIC `1.46%`，覆盖率下限 `91.77%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。

### 23. `fund_combo_net_margin_low_pe`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(net_margin_ttm), 1 - rank(pe_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `3.48%`，valid RankIC `1.46%`，test RankIC `1.86%`，2026YTD RankIC `2.80%`，三段最小调整后 RankIC `1.46%`，覆盖率下限 `64.82%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 24. `fund_combo_profit_to_assets_low_ps`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(net_profit_ttm / total_assets), 1 - rank(ps_ttm))`。利润/资产，接近 ROA 口径，衡量资产盈利效率；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `1.45%`，valid RankIC `1.45%`，test RankIC `2.81%`，2026YTD RankIC `3.88%`，三段最小调整后 RankIC `1.45%`，覆盖率下限 `75.40%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 25. `fund_combo_cash_to_assets_low_pe`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(rank(money_cap / total_assets), 1 - rank(pe_ttm))`。货币资金/资产，衡量现金储备；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `1.42%`，valid RankIC `2.29%`，test RankIC `1.69%`，2026YTD RankIC `0.10%`，三段最小调整后 RankIC `1.42%`，覆盖率下限 `78.25%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。2026YTD 表现偏弱，我会考虑降权，或者先把它放在备选特征里。

### 26. `fund_ind_neu_dupont_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb)))`。净利率，反映成本控制、议价能力和商业模式质量；资产周转率，衡量资产产生收入的效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `3.28%`，valid RankIC `1.36%`，test RankIC `3.88%`，2026YTD RankIC `4.81%`，三段最小调整后 RankIC `1.36%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 27. `fund_ind_neu_eps_bps_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb)))`。每股收益，衡量单股盈利能力；每股净资产，衡量单股账面资产基础；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `2.34%`，valid RankIC `1.34%`，test RankIC `2.47%`，2026YTD RankIC `5.75%`，三段最小调整后 RankIC `1.34%`，覆盖率下限 `95.27%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 28. `fund_combo_quick_ratio_low_pb`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(rank(quick_ratio), 1 - rank(pb))`。速动比率，衡量短期偿债安全垫；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `1.33%`，valid RankIC `3.17%`，test RankIC `3.24%`，2026YTD RankIC `0.04%`，三段最小调整后 RankIC `1.33%`，覆盖率下限 `92.96%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。2026YTD 表现偏弱，我会考虑降权，或者先把它放在备选特征里。

### 29. `fund_combo_ocf_to_assets_fcf_yield`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(n_cashflow_act_ttm / total_assets), rank(free_cashflow_ttm / total_mv))`。经营现金流/总资产，衡量资产产生现金流的效率；自由现金流收益率，衡量现金回报相对市值是否便宜。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `1.52%`，valid RankIC `1.29%`，test RankIC `1.78%`，2026YTD RankIC `3.28%`，三段最小调整后 RankIC `1.29%`，覆盖率下限 `72.95%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。

### 30. `fund_combo_inventory_turnover_low_pb`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `mean(rank(inv_turn), 1 - rank(pb))`。存货周转率，衡量库存消化效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `1.29%`，valid RankIC `4.11%`，test RankIC `3.84%`，2026YTD RankIC `1.24%`，三段最小调整后 RankIC `1.29%`，覆盖率下限 `91.41%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。

### 31. `fund_ind_neu_working_capital_cash_growth`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `industry_neutralize(mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit), rank(net_profit_yoy)))`。经营现金流对利润的覆盖，验证会计利润含金量；利润同比增长，衡量增长是否落到利润表底线；营运资本占用压力，主要来自应收和存货；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `1.27%`，valid RankIC `1.29%`，test RankIC `1.34%`，2026YTD RankIC `2.35%`，三段最小调整后 RankIC `1.27%`，覆盖率下限 `63.49%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 32. `fund_combo_low_debt_assets_low_pe`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(1 - rank(debt_to_assets), 1 - rank(pe_ttm))`。资产负债率，衡量杠杆压力；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `1.68%`，valid RankIC `1.22%`，test RankIC `2.01%`，2026YTD RankIC `0.65%`，三段最小调整后 RankIC `1.22%`，覆盖率下限 `80.16%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。

### 33. `fund_combo_net_profit_yoy_accrual_quality`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `mean(rank(net_profit_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets))`。利润同比增长，衡量增长是否落到利润表底线；现金流与利润差额/资产，衡量利润是否由现金支持。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 实证表现：train RankIC `1.66%`，valid RankIC `1.64%`，test RankIC `1.21%`，2026YTD RankIC `1.37%`，三段最小调整后 RankIC `1.21%`，覆盖率下限 `74.30%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。

### 34. `fund_ind_neu_cash_safety_low_pb`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb)))`。现金/负债，衡量现金对债务的覆盖；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `1.62%`，valid RankIC `1.16%`，test RankIC `2.34%`，2026YTD RankIC `0.60%`，三段最小调整后 RankIC `1.16%`，覆盖率下限 `94.99%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 35. `fund_ind_neu_roa_ocf_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)))`。经营现金流/总资产，衡量资产产生现金流的效率；ROA，衡量资产层面的盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `3.64%`，valid RankIC `1.15%`，test RankIC `3.06%`，2026YTD RankIC `5.03%`，三段最小调整后 RankIC `1.15%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 36. `fund_combo_revenue_yoy_accrual_quality`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `mean(rank(revenue_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets))`。收入同比增长，衡量需求和业务扩张；现金流与利润差额/资产，衡量利润是否由现金支持。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 实证表现：train RankIC `1.14%`，valid RankIC `1.76%`，test RankIC `1.70%`，2026YTD RankIC `2.26%`，三段最小调整后 RankIC `1.14%`，覆盖率下限 `74.30%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。

### 37. `industry_rank_cash_profit_cover`

- 归类：`fundamental_quality`。
- 我怎么算：表达式为 `group_rank(n_cashflow_act_ttm / net_profit_ttm, industry)`。经营现金流/净利润，衡量利润的现金覆盖程度；行业内排名：只在同一行业内部比较，降低行业结构噪声。
- 为什么可能有效：我的理解是，这类因子主要看盈利质量：用现金流、行业内排名或收入现金化程度过滤会计利润，可以降低利润虚高和应收堆积带来的误判。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。行业内排名有效，是因为不同行业的现金流周期和利润率天然不同，同行业比较能减少结构性偏差。
- 实证表现：train RankIC `1.21%`，valid RankIC `1.76%`，test RankIC `1.13%`，2026YTD RankIC `0.81%`，三段最小调整后 RankIC `1.13%`，覆盖率下限 `75.52%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 38. `turnover_efficiency_combo`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `rank(inv_turn) + rank(ar_turn)`。存货周转率，衡量库存消化效率；应收账款周转率，衡量回款效率。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `2.08%`，valid RankIC `2.14%`，test RankIC `1.05%`，2026YTD RankIC `2.00%`，三段最小调整后 RankIC `1.05%`，覆盖率下限 `91.35%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。

### 39. `fund_combo_cash_to_liab_fcf_yield`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(rank(cash_to_liab), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；现金/负债，衡量现金对债务的覆盖。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `1.68%`，valid RankIC `1.00%`，test RankIC `1.02%`，2026YTD RankIC `0.37%`，三段最小调整后 RankIC `1.00%`，覆盖率下限 `72.95%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。2026YTD 表现偏弱，我会考虑降权，或者先把它放在备选特征里。

### 40. `cash_revenue_quality`

- 归类：`fundamental_quality`。
- 我怎么算：表达式为 `rank(operating_cf_margin_ttm) + rank(ocf_to_or)`。经营现金流利润率，衡量收入转成经营现金流的能力；经营现金流收入比，验证收入质量。
- 为什么可能有效：我的理解是，这类因子主要看盈利质量：用现金流、行业内排名或收入现金化程度过滤会计利润，可以降低利润虚高和应收堆积带来的误判。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。
- 实证表现：train RankIC `2.37%`，valid RankIC `1.00%`，test RankIC `1.43%`，2026YTD RankIC `5.24%`，三段最小调整后 RankIC `1.00%`，覆盖率下限 `75.52%`。
- 我会怎么用：我会把它作为辅助特征放进多因子或 XGBoost，不会单独拿它决定组合权重。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 41. `clean_growth_quality`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `rank(cashflow_to_profit) + rank(net_profit_yoy) - rank(working_capital_pressure)`。经营现金流对利润的覆盖，验证会计利润含金量；利润同比增长，衡量增长是否落到利润表底线；营运资本占用压力，主要来自应收和存货。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `0.94%`，valid RankIC `2.58%`，test RankIC `2.07%`，2026YTD RankIC `3.11%`，三段最小调整后 RankIC `0.94%`，覆盖率下限 `63.49%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。

### 42. `fund_combo_low_working_capital_pressure_cashflow_to_profit`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit))`。经营现金流对利润的覆盖，验证会计利润含金量；营运资本占用压力，主要来自应收和存货。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `0.94%`，valid RankIC `3.07%`，test RankIC `1.96%`，2026YTD RankIC `1.17%`，三段最小调整后 RankIC `0.94%`，覆盖率下限 `64.66%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。

### 43. `fund_combo_eps_fcf_yield`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `mean(rank(eps), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；每股收益，衡量单股盈利能力。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 实证表现：train RankIC `2.99%`，valid RankIC `0.88%`，test RankIC `1.73%`，2026YTD RankIC `5.82%`，三段最小调整后 RankIC `0.88%`，覆盖率下限 `72.94%`。
- 我会怎么用：我会把它和行业约束、质量过滤一起用，避免只买到便宜但基本面差的价值陷阱。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 44. `fund_combo_low_working_capital_pressure_net_margin`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `mean(1 - rank(working_capital_pressure), rank(net_margin_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；营运资本占用压力，主要来自应收和存货。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `2.56%`，valid RankIC `0.85%`，test RankIC `0.92%`，2026YTD RankIC `3.57%`，三段最小调整后 RankIC `0.85%`，覆盖率下限 `64.66%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。

### 45. `fund_combo_revenue_yoy_operating_cf_margin`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `mean(rank(revenue_yoy), rank(operating_cf_margin_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；收入同比增长，衡量需求和业务扩张。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 实证表现：train RankIC `5.26%`，valid RankIC `0.83%`，test RankIC `1.95%`，2026YTD RankIC `6.31%`，三段最小调整后 RankIC `0.83%`，覆盖率下限 `74.30%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 46. `fund_combo_net_profit_yoy_operating_cf_margin`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `mean(rank(net_profit_yoy), rank(operating_cf_margin_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；利润同比增长，衡量增长是否落到利润表底线。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 实证表现：train RankIC `3.82%`，valid RankIC `0.79%`，test RankIC `1.36%`，2026YTD RankIC `5.20%`，三段最小调整后 RankIC `0.79%`，覆盖率下限 `74.30%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 47. `fund_ind_neu_roe_low_pe`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(roe_ttm), 1 - rank(pe_ttm)))`。ROE，衡量股东权益回报；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `2.96%`，valid RankIC `0.78%`，test RankIC `1.51%`，2026YTD RankIC `3.39%`，三段最小调整后 RankIC `0.78%`，覆盖率下限 `64.82%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 48. `fund_combo_low_debt_assets_fcf_yield`

- 归类：`fundamental_safety`。
- 我怎么算：表达式为 `mean(1 - rank(debt_to_assets), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；资产负债率，衡量杠杆压力。
- 为什么可能有效：我的理解是，这类因子主要看安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 实证表现：train RankIC `1.23%`，valid RankIC `0.74%`，test RankIC `1.57%`，2026YTD RankIC `1.05%`，三段最小调整后 RankIC `0.74%`，覆盖率下限 `72.95%`。
- 我会怎么用：我会更多把它当成风险过滤或防守型 alpha，而不是单独拿它追求最高进攻性。

### 49. `quality_liquidity_confirm_20`

- 归类：`mixed`。
- 我怎么算：表达式为 `zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,20)))`。经营现金流利润率，衡量收入转成经营现金流的能力；成交额放大，衡量市场关注度和资金确认。
- 为什么可能有效：我的理解是，这类因子把基本面质量和市场交易确认放在一起，寻找基本面改善且开始被资金关注、但尚未完全定价的股票。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成交额确认有效，是因为基本面改善开始被资金交易时，信号兑现速度通常更快。
- 实证表现：train RankIC `2.84%`，valid RankIC `0.71%`，test RankIC `0.64%`，2026YTD RankIC `2.84%`，三段最小调整后 RankIC `0.64%`，覆盖率下限 `74.84%`。
- 我会怎么用：我会把它作为辅助特征放进多因子或 XGBoost，不会单独拿它决定组合权重。

### 50. `fund_atom_revenue_yoy`

- 归类：`fundamental_growth`。
- 我怎么算：表达式为 `rank(revenue_yoy)`。收入同比增长，衡量需求和业务扩张。
- 为什么可能有效：我的理解是，这类因子是在筛选更干净的成长：只有收入、利润、现金流或营运效率同时改善的增长，才更可能被市场持续定价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 实证表现：train RankIC `3.00%`，valid RankIC `0.64%`，test RankIC `1.71%`，2026YTD RankIC `5.10%`，三段最小调整后 RankIC `0.64%`，覆盖率下限 `74.31%`。
- 我会怎么用：我会更倾向于在财报更新后做月度或季度调仓，不会把这种低频财务信号拿去日频过度交易。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。

### 51. `fund_ind_neu_cashflow_to_profit_low_pb`

- 归类：`fundamental_value`。
- 我怎么算：表达式为 `industry_neutralize(mean(rank(cashflow_to_profit), 1 - rank(pb)))`。经营现金流对利润的覆盖，验证会计利润含金量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 为什么可能有效：我的理解是，这类因子是在用质量约束估值：市场容易低估盈利质量稳定但估值不高的中盘公司，后续估值修复或盈利确认会带来横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 实证表现：train RankIC `0.60%`，valid RankIC `3.01%`，test RankIC `3.72%`，2026YTD RankIC `2.12%`，三段最小调整后 RankIC `0.60%`，覆盖率下限 `75.16%`。
- 我会怎么用：我会优先把它放进带行业约束的指增模型，也可以直接拿来做行业内排序。

### 52. `fund_combo_receivable_turnover_net_margin`

- 归类：`fundamental_efficiency`。
- 我怎么算：表达式为 `mean(rank(ar_turn), rank(net_margin_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；应收账款周转率，衡量回款效率。
- 为什么可能有效：我的理解是，这类因子主要看营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 实证表现：train RankIC `3.15%`，valid RankIC `0.86%`，test RankIC `0.59%`，2026YTD RankIC `4.76%`，三段最小调整后 RankIC `0.59%`，覆盖率下限 `73.05%`。
- 我会怎么用：我会把它和盈利质量、估值因子一起用，避免只买到短期周转改善但利润率不够的公司。2026YTD 仍有较强正向 RankIC，至少从近期看还没有明显失效。


## 10. 风险和我下一步会怎么用

- 这次本质上还是宽搜索，所以一定有多重检验问题。我不能只盯着最优因子看，必须同时看 valid/test/YTD 和相关性过滤。
- 基本面因子更新频率低，进入指增模型时我更倾向于月度或财报后调仓，不适合日频过度交易。
- 行业中性因子更适合直接放进指增；非中性因子还要单独检查行业暴露，避免变成行业押注。
- 下一步我会把这些推荐因子放进 XGBoost 特征集合，做 `旧3因子`、`新基本面算子集合`、`公开因子+新基本面` 之间的消融实验。

## 11. 输出文件

- 输出目录：`outputs/fundamental_operator_mining_csi500`
- 全部拆分汇总：`outputs/fundamental_operator_mining_csi500/all_split_summary.csv`
- 稳定性表：`outputs/fundamental_operator_mining_csi500/stable_factors.csv`
- 最终推荐：`outputs/fundamental_operator_mining_csi500/selected_factors.csv`
- 这份报告：`docs/fundamental_operator_mining_csi500_2026-07-15.md`