# 中证500基本面算子扩展与因子筛选报告

日期：2026-07-20

## 1. 任务目标

现有 `quality_growth_hmean`、`eps_bps_value_quality`、`industry_neutral_roe_value_pb` 三个核心基本面因子已经验证出一定效果，但覆盖的财务关系还比较集中，主要围绕成长、每股盈利账面价值和 ROE-PB 质量价值。为了让基本面因子库更完整，这里继续基于利润表、资产负债表、现金流量表和财务指标表扩展算子，再通过有约束的组合搜索筛选更稳定的中证500基本面因子。

当前框架先把不同口径的财务字段统一成“分数越高越好”的原子指标，再用均值、调和平均、估值扣减、行业中性和规模中性等算子生成候选因子，最后在中证500股票池上做 train/valid/test/2026YTD 验证和相关性去重。

## 2. 执行计划与口径

执行步骤如下：

1. 保留旧三因子和已有 legacy 基本面因子作为 benchmark。
2. 从利润表、资产负债表、现金流量表、财务指标表和估值字段构造方向统一的原子指标。
3. 用均值、调和平均、价值扣减、行业中性、规模中性等算子生成候选组合。
4. 在中证500 train/valid/test/2026YTD 上验证 RankIC 稳定性，再做相关性去重，得到推荐集合。

运行参数如下：

| parameter | value |
| --- | --- |
| processed_dir | data/processed |
| suffix | 000905_SH |
| n_candidates | 374 |
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
| resume_existing | True |

## 3. 数据口径

| 项目 | 口径 |
| --- | --- |
| 股票池 | 中证500历史成分股 |
| 日频面板 | `data/processed/*_fundamental_000905_SH.parquet` |
| 基本面 PIT | 按公告日 `available_date` 向后对齐，避免未来函数 |
| 训练/验证/测试 | 沿用现有 `train/valid/test` 拆分 |
| 近期观察 | 从 test 中单独切出 `2026YTD` |
| 标签 | 现有 `label`，即未来收益/排名评估口径 |

## 4. 算子体系

| 算子 | 例子 | 经济含义 |
| --- | --- | --- |
| 方向统一原子 | `low_pb = 1 - rank(pb)` | 所有基础指标都转成越大越好 |
| 比率算子 | `free_cashflow_ttm / total_mv` | 用现金流、资产、负债等构造可比指标 |
| 均值组合 | `mean(roe, low_pb)` | 多个维度等权确认 |
| 调和平均 | `harm_mean(revenue_yoy, net_profit_yoy)` | 惩罚单腿增长 |
| 三腿组合 | `roe + cashflow_to_profit + low_pb` | 质量、现金流、估值同时约束 |
| 行业中性 | `industry_neutralize(score)` | 降低行业结构差异 |
| 规模中性 | `size_neutralize(score)` | 降低市值风格暴露 |

候选因子族数量：

| family | n_candidates |
| --- | --- |
| fundamental_value | 140 |
| fundamental_efficiency | 70 |
| fundamental_growth | 66 |
| fundamental_safety | 59 |
| fundamental_quality | 32 |
| mixed | 7 |

验证结果数量：

| status | count |
| --- | --- |
| evaluated | 374 |
| passes_stability | 178 |
| selected_after_corr | 87 |

核心结论：

1. 结果最强的一组不是单纯成长，而是“现金流/利润率质量 + 低估值”：`operating_cf_margin + low_pb`、`net_margin + low_pb`、`profit_to_assets + low_pb` 排在最前。
2. PB/PS 相关估值约束比单独成长更稳，说明中证500里“盈利质量不差且估值不贵”的组合仍然有显著横截面区分度。
3. 原三因子仍然有效，但会被新工厂生成的近似等价或更细颗粒组合在相关性去重时替代；这不是失效，而是说明旧因子逻辑被更系统地展开了。
4. 推荐集合以价值质量因子为主，同时保留成长、安全、效率、现金流质量和少量混合量价确认因子，便于后续进入 XGBoost 或指数增强模型时做消融。

去重后推荐因子族分布：

| family | n_selected |
| --- | --- |
| fundamental_value | 40 |
| fundamental_efficiency | 14 |
| fundamental_growth | 14 |
| fundamental_safety | 13 |
| fundamental_quality | 5 |
| mixed | 1 |

## 5. 最终推荐因子

下表是通过稳定性过滤和相关性去重后的推荐因子。`min_adj_rankic` 是 train/valid/test 三段按训练期方向调整后的最小 RankIC，越高说明跨样本越稳。

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_operating_cf_margin_low_pb | fundamental_value | positive | 4.08% | 4.05% | 4.11% | 2.94% | 4.05% | 4.08% | 75.16% | mean(rank(operating_cf_margin_ttm), 1 - rank(pb)) | 将经营现金流利润率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_net_margin_low_pb | fundamental_value | positive | 3.53% | 3.18% | 4.23% | 4.49% | 3.18% | 3.65% | 75.16% | mean(rank(net_margin_ttm), 1 - rank(pb)) | 将净利率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_profit_to_assets_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(net_profit_ttm / total_assets), 1 - rank(pb)) | 将资产盈利能力、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| robust_margin_value_ps | fundamental_value | positive | 3.02% | 3.47% | 4.21% | 4.51% | 3.02% | 3.57% | 75.40% | robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm) | 用稳健标准化后的净利率扣减市销率估值，偏好利润率高且销售估值不贵的公司。 |
| fund_size_neu_eps_bps_low_pb | fundamental_value | positive | 2.94% | 3.37% | 3.41% | 5.32% | 2.94% | 3.24% | 95.27% | size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | 将每股收益、每股净资产、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_operating_cf_margin_low_ps | fundamental_value | positive | 2.83% | 3.39% | 3.51% | 2.95% | 2.83% | 3.24% | 75.40% | mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)) | 将经营现金流利润率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_gross_to_net_margin_low_pb | fundamental_efficiency | positive | 3.95% | 2.82% | 4.17% | 4.65% | 2.82% | 3.65% | 72.75% | mean(rank(net_margin_ttm / gross_margin_ttm), 1 - rank(pb)) | 用毛利留存为净利的效率、净利率、毛利率、低市净率估值衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_size_neu_gross_net_cash_value | fundamental_value | positive | 5.65% | 2.69% | 3.88% | 5.07% | 2.69% | 4.07% | 72.98% | size_neutralize(mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm))) | 将经营现金流利润率、净利率、毛利率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_revenue_profit_low_pe_hmean | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm)) | 用调和平均组合收入同比增长、净利润同比增长、低市盈率估值，偏好多条基本面腿都不弱、短板更少的公司。 |
| fund_ind_neu_ocf_to_assets_low_pb | fundamental_value | positive | 2.90% | 2.53% | 4.01% | 3.97% | 2.53% | 3.14% | 75.16% | industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | 将经营现金流资产产出、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_leverage_adjusted_roe_low_pb | fundamental_value | positive | 2.44% | 3.09% | 3.84% | 1.49% | 2.44% | 3.12% | 75.16% | mean(rank(roe_ttm - debt_to_assets), 1 - rank(pb)) | 将杠杆调整后净资产收益率、净资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_roa_ocf_low_pb | fundamental_value | positive | 2.37% | 2.32% | 3.63% | 5.84% | 2.32% | 2.78% | 75.16% | mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)) | 将总资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_reported_ocf_to_or_low_pe | fundamental_value | positive | 2.31% | 2.95% | 2.68% | 3.29% | 2.31% | 2.65% | 80.13% | mean(rank(ocf_to_or), 1 - rank(pe_ttm)) | 将财务指标表经营现金流收入比、低市盈率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_eps_low_ps | fundamental_value | positive | 2.73% | 2.24% | 3.04% | 4.97% | 2.24% | 2.67% | 95.51% | mean(rank(eps), 1 - rank(ps_ttm)) | 将每股收益、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_ind_neu_roe_cash_low_pb | fundamental_value | positive | 2.88% | 2.24% | 3.58% | 4.41% | 2.24% | 2.90% | 75.16% | industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb))) | 将经营现金流利润覆盖、净资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_net_margin_low_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | mean(rank(net_margin_ttm), 1 - rank(ps_ttm)) | 将净利率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_reported_ocf_to_or_low_ps | fundamental_value | positive | 2.10% | 3.59% | 3.34% | 3.67% | 2.10% | 3.01% | 95.65% | mean(rank(ocf_to_or), 1 - rank(ps_ttm)) | 将财务指标表经营现金流收入比、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_revenue_profit_cash_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit)) | 用调和平均组合经营现金流利润覆盖、收入同比增长、净利润同比增长，偏好多条基本面腿都不弱、短板更少的公司。 |
| fund_ind_neu_leverage_adjusted_quality_value | fundamental_safety | positive | 2.44% | 1.93% | 3.39% | 2.48% | 1.93% | 2.59% | 75.16% | industry_neutralize(mean(rank(roe_ttm - debt_to_assets), rank(cashflow_to_profit), 1 - rank(pb))) | 用经营现金流利润覆盖、净资产收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_reported_ocf_to_or_low_pb | fundamental_value | positive | 1.91% | 4.43% | 4.27% | 4.01% | 1.91% | 3.54% | 95.41% | mean(rank(ocf_to_or), 1 - rank(pb)) | 将财务指标表经营现金流收入比、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_ind_neu_roe_low_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | 将净资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| shareholder_yield_quality | fundamental_value | positive | 1.85% | 2.69% | 2.01% | 3.49% | 1.85% | 2.18% | 56.20% | rank(dv_ttm) + rank(roe_ttm) | 用股息率和净资产收益率共同衡量股东回报质量，偏好分红回报和盈利能力兼具的公司。 |
| fund_combo_fcf_to_assets_earnings_yield | fundamental_value | positive | 1.80% | 3.22% | 3.04% | 1.20% | 1.80% | 2.68% | 62.40% | mean(rank(free_cashflow_ttm / total_assets), rank(1 / pe_ttm)) | 将自由现金流资产产出、盈利收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_atom_dividend_yield | fundamental_value | positive | 1.74% | 5.15% | 3.19% | 1.85% | 1.74% | 3.36% | 73.66% | rank(dv_ttm) | 将股息率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_eps_earnings_yield | fundamental_value | positive | 2.17% | 1.71% | 1.97% | 4.03% | 1.71% | 1.95% | 80.16% | mean(rank(eps), rank(1 / pe_ttm)) | 将每股收益、盈利收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_size_neu_cash_safety_low_pb | fundamental_safety | positive | 2.60% | 1.66% | 2.20% | 1.11% | 1.66% | 2.15% | 94.99% | size_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb))) | 用现金对负债覆盖、低市净率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_ind_neu_reported_cash_margin_low_pb | fundamental_value | positive | 1.65% | 1.75% | 3.00% | 2.47% | 1.65% | 2.13% | 83.89% | industry_neutralize(mean(rank(ocf_to_profit), rank(netprofit_margin), 1 - rank(pb))) | 将低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_quick_ratio_earnings_yield | fundamental_safety | positive | 2.05% | 1.58% | 1.58% | 0.36% | 1.58% | 1.74% | 76.35% | mean(rank(quick_ratio), rank(1 / pe_ttm)) | 用速动比率、盈利收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_fcf_to_assets_low_ps | fundamental_value | positive | 1.57% | 2.78% | 2.78% | 1.11% | 1.57% | 2.38% | 72.85% | mean(rank(free_cashflow_ttm / total_assets), 1 - rank(ps_ttm)) | 将自由现金流资产产出、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_size_neu_reported_cash_margin_low_pb | fundamental_value | positive | 1.55% | 3.67% | 3.62% | 4.24% | 1.55% | 2.95% | 83.89% | size_neutralize(mean(rank(ocf_to_profit), rank(netprofit_margin), 1 - rank(pb))) | 将低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_revenue_yoy_cash_conversion_spread | fundamental_growth | positive | 2.74% | 1.75% | 1.55% | 2.46% | 1.55% | 2.02% | 74.30% | mean(rank(revenue_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_revenue_ttm)) | 将现金流与利润差额、收入同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_combo_receivable_turnover_low_pb | fundamental_efficiency | positive | 1.51% | 4.82% | 2.87% | 0.26% | 1.51% | 3.06% | 92.25% | mean(rank(ar_turn), 1 - rank(pb)) | 用应收账款周转效率、低市净率估值衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_atom_inventory_turnover | fundamental_efficiency | positive | 1.61% | 1.46% | 1.70% | 2.59% | 1.46% | 1.59% | 91.77% | rank(inv_turn) | 用存货周转效率衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_combo_net_margin_low_pe | fundamental_value | positive | 3.48% | 1.46% | 1.86% | 2.80% | 1.46% | 2.27% | 64.82% | mean(rank(net_margin_ttm), 1 - rank(pe_ttm)) | 将净利率、低市盈率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_profit_to_assets_low_ps | fundamental_value | positive | 1.45% | 1.45% | 2.81% | 3.88% | 1.45% | 1.90% | 75.40% | mean(rank(net_profit_ttm / total_assets), 1 - rank(ps_ttm)) | 将资产盈利能力、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_cash_to_assets_low_pe | fundamental_safety | positive | 1.42% | 2.29% | 1.69% | 0.10% | 1.42% | 1.80% | 78.25% | mean(rank(money_cap / total_assets), 1 - rank(pe_ttm)) | 用现金资产占比、低市盈率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_net_profit_yoy_reported_ocf_to_profit | fundamental_growth | positive | 1.42% | 1.53% | 1.39% | 5.44% | 1.39% | 1.45% | 65.51% | mean(rank(net_profit_yoy), rank(ocf_to_profit)) | 将财务指标表经营现金流利润覆盖、净利润同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_combo_revenue_yoy_reported_ocf_to_profit | fundamental_growth | positive | 1.71% | 1.64% | 1.37% | 5.40% | 1.37% | 1.57% | 65.51% | mean(rank(revenue_yoy), rank(ocf_to_profit)) | 将财务指标表经营现金流利润覆盖、收入同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_ind_neu_dupont_low_pb | fundamental_value | positive | 3.28% | 1.36% | 3.88% | 4.81% | 1.36% | 2.84% | 75.16% | industry_neutralize(mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb))) | 将净利率、资产周转效率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_inventory_turnover_fcf_margin | fundamental_efficiency | positive | 1.35% | 1.44% | 1.61% | 2.12% | 1.35% | 1.46% | 71.98% | mean(rank(inv_turn), rank(free_cashflow_ttm / total_revenue_ttm)) | 用自由现金流利润率、存货周转效率衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_ind_neu_eps_bps_low_pb | fundamental_value | positive | 2.34% | 1.34% | 2.47% | 5.75% | 1.34% | 2.05% | 95.27% | industry_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | 将每股收益、每股净资产、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_quick_ratio_low_pb | fundamental_safety | positive | 1.33% | 3.17% | 3.24% | 0.04% | 1.33% | 2.58% | 92.96% | mean(rank(quick_ratio), 1 - rank(pb)) | 用速动比率、低市净率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_ind_neu_fcf_assets_low_pb | fundamental_value | positive | 1.33% | 1.41% | 3.17% | 1.92% | 1.33% | 1.97% | 72.60% | industry_neutralize(mean(rank(free_cashflow_ttm / total_assets), 1 - rank(debt_to_assets), 1 - rank(pb))) | 将低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_leverage_adjusted_roe_low_ps | fundamental_value | positive | 1.31% | 1.89% | 2.93% | 1.42% | 1.31% | 2.05% | 75.40% | mean(rank(roe_ttm - debt_to_assets), 1 - rank(ps_ttm)) | 将杠杆调整后净资产收益率、净资产收益率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_revenue_to_liab_earnings_yield | fundamental_safety | positive | 1.83% | 1.30% | 1.85% | 0.66% | 1.30% | 1.66% | 64.82% | mean(rank(total_revenue_ttm / total_liab), rank(1 / pe_ttm)) | 用收入对负债覆盖、盈利收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_cashflow_to_liab_fcf_yield | fundamental_safety | positive | 1.73% | 1.30% | 1.83% | 2.97% | 1.30% | 1.62% | 72.95% | mean(rank(n_cashflow_act_ttm / total_liab), rank(free_cashflow_ttm / total_mv)) | 用经营现金流对负债覆盖、自由现金流收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_inventory_turnover_low_pb | fundamental_efficiency | positive | 1.29% | 4.11% | 3.84% | 1.24% | 1.29% | 3.08% | 91.41% | mean(rank(inv_turn), 1 - rank(pb)) | 用存货周转效率、低市净率估值衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_ind_neu_working_capital_cash_growth | fundamental_growth | positive | 1.27% | 1.29% | 1.34% | 2.35% | 1.27% | 1.30% | 63.49% | industry_neutralize(mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit), rank(net_profit_yoy))) | 将经营现金流利润覆盖、净利润同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_low_debt_assets_earnings_yield | fundamental_safety | positive | 1.68% | 1.22% | 2.01% | 0.65% | 1.22% | 1.64% | 80.16% | mean(1 - rank(debt_to_assets), rank(1 / pe_ttm)) | 用低资产负债率、盈利收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_inventory_turnover_reported_ocf_to_profit | fundamental_efficiency | positive | 1.22% | 1.61% | 1.44% | 2.67% | 1.22% | 1.43% | 78.81% | mean(rank(inv_turn), rank(ocf_to_profit)) | 用财务指标表经营现金流利润覆盖、存货周转效率衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_ind_neu_reported_ocf_to_profit_low_pb | fundamental_value | positive | 1.22% | 2.58% | 3.34% | 1.75% | 1.22% | 2.38% | 83.94% | industry_neutralize(mean(rank(ocf_to_profit), 1 - rank(pb))) | 将财务指标表经营现金流利润覆盖、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_revenue_yoy_fcf_margin | fundamental_growth | positive | 3.91% | 1.22% | 1.50% | 3.58% | 1.22% | 2.21% | 71.82% | mean(rank(revenue_yoy), rank(free_cashflow_ttm / total_revenue_ttm)) | 将自由现金流利润率、收入同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_combo_net_profit_yoy_accrual_quality | fundamental_growth | positive | 1.66% | 1.64% | 1.21% | 1.37% | 1.21% | 1.50% | 74.30% | mean(rank(net_profit_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)) | 将净利润同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_atom_reported_ocf_to_or | fundamental_quality | positive | 2.08% | 1.19% | 1.49% | 5.63% | 1.19% | 1.59% | 95.77% | rank(ocf_to_or) | 用财务指标表经营现金流收入比验证盈利质量，偏好利润和收入更容易转化为现金的公司。 |
| fund_atom_reported_ocf_to_profit | fundamental_quality | positive | 1.18% | 1.99% | 1.24% | 3.55% | 1.18% | 1.47% | 84.00% | rank(ocf_to_profit) | 用财务指标表经营现金流利润覆盖验证盈利质量，偏好利润和收入更容易转化为现金的公司。 |
| fund_combo_low_working_capital_pressure_reported_ocf_to_profit | fundamental_efficiency | positive | 2.99% | 2.52% | 1.17% | 1.55% | 1.17% | 2.23% | 65.49% | mean(1 - rank(working_capital_pressure), rank(ocf_to_profit)) | 用财务指标表经营现金流利润覆盖、低营运资本压力衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_ind_neu_cash_safety_low_pb | fundamental_safety | positive | 1.62% | 1.16% | 2.34% | 0.60% | 1.16% | 1.71% | 94.99% | industry_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb))) | 用现金对负债覆盖、低市净率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_ind_neu_gross_net_cash_value | fundamental_value | positive | 2.77% | 1.16% | 2.28% | 4.34% | 1.16% | 2.07% | 72.99% | industry_neutralize(mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm))) | 将经营现金流利润率、净利率、毛利率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_size_neu_fcf_assets_low_pb | fundamental_value | positive | 1.16% | 2.36% | 3.38% | 2.34% | 1.16% | 2.30% | 72.59% | size_neutralize(mean(rank(free_cashflow_ttm / total_assets), 1 - rank(debt_to_assets), 1 - rank(pb))) | 将低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_ind_neu_roa_ocf_low_pb | fundamental_value | positive | 3.64% | 1.15% | 3.06% | 5.03% | 1.15% | 2.62% | 75.16% | industry_neutralize(mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | 将总资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| industry_rank_cash_profit_cover | fundamental_quality | positive | 1.21% | 1.76% | 1.13% | 0.81% | 1.13% | 1.36% | 75.52% | group_rank(n_cashflow_act_ttm / net_profit_ttm, industry) | 在同行业内比较现金流对利润的覆盖，降低行业现金流周期差异带来的偏差。采用行业内排名，减少行业天然差异的影响。 |
| turnover_efficiency_combo | fundamental_efficiency | positive | 2.08% | 2.14% | 1.05% | 2.00% | 1.05% | 1.76% | 91.35% | rank(inv_turn) + rank(ar_turn) | 用存货周转和应收周转共同衡量营运效率，偏好库存消化和回款速度更快的公司。 |
| fund_ind_neu_fcf_margin_low_ps | fundamental_value | positive | 1.03% | 1.54% | 2.48% | 2.22% | 1.03% | 1.68% | 72.85% | industry_neutralize(mean(rank(free_cashflow_ttm / total_revenue_ttm), rank(ocf_to_or), 1 - rank(ps_ttm))) | 将自由现金流利润率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_fcf_to_profit_low_ps | fundamental_value | positive | 1.02% | 3.84% | 3.06% | 0.19% | 1.02% | 2.64% | 72.85% | mean(rank(free_cashflow_ttm / net_profit_ttm), 1 - rank(ps_ttm)) | 将自由现金流利润覆盖、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_cash_to_liab_fcf_yield | fundamental_safety | positive | 1.68% | 1.00% | 1.02% | 0.37% | 1.00% | 1.24% | 72.95% | mean(rank(cash_to_liab), rank(free_cashflow_ttm / total_mv)) | 用现金对负债覆盖、自由现金流收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_net_profit_yoy_fcf_margin | fundamental_growth | positive | 4.12% | 0.96% | 1.01% | 3.11% | 0.96% | 2.03% | 71.82% | mean(rank(net_profit_yoy), rank(free_cashflow_ttm / total_revenue_ttm)) | 将自由现金流利润率、净利润同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_combo_gross_to_net_margin_cashflow_to_profit | fundamental_efficiency | positive | 2.30% | 0.95% | 2.77% | 6.71% | 0.95% | 2.01% | 73.10% | mean(rank(net_margin_ttm / gross_margin_ttm), rank(cashflow_to_profit)) | 用经营现金流利润覆盖、毛利留存为净利的效率、净利率、毛利率衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| clean_growth_quality | fundamental_growth | positive | 0.94% | 2.58% | 2.07% | 3.11% | 0.94% | 1.86% | 63.49% | rank(cashflow_to_profit) + rank(net_profit_yoy) - rank(working_capital_pressure) | 用利润增长、现金流覆盖和低营运资本压力共同确认成长质量，偏好增长更干净的公司。 |
| fund_combo_low_working_capital_pressure_cashflow_to_profit | fundamental_efficiency | positive | 0.94% | 3.07% | 1.96% | 1.17% | 0.94% | 1.99% | 64.66% | mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit)) | 用经营现金流利润覆盖、低营运资本压力衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_ind_neu_working_capital_fcf_value | fundamental_efficiency | positive | 0.91% | 1.98% | 2.84% | 1.00% | 0.91% | 1.91% | 71.52% | industry_neutralize(mean(1 - rank((accounts_receiv + inventories) / total_assets), rank(free_cashflow_ttm / total_revenue_ttm), 1 - rank(pb))) | 用多个基本面指标衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_eps_fcf_yield | fundamental_value | positive | 2.99% | 0.88% | 1.73% | 5.82% | 0.88% | 1.87% | 72.94% | mean(rank(eps), rank(free_cashflow_ttm / total_mv)) | 将每股收益、自由现金流收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_low_working_capital_pressure_net_margin | fundamental_efficiency | positive | 2.56% | 0.85% | 0.92% | 3.57% | 0.85% | 1.44% | 64.66% | mean(1 - rank(working_capital_pressure), rank(net_margin_ttm)) | 用净利率、低营运资本压力衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_combo_working_capital_fcf_value | fundamental_efficiency | positive | 0.85% | 3.83% | 3.04% | 0.04% | 0.85% | 2.57% | 71.52% | mean(1 - rank((accounts_receiv + inventories) / total_assets), rank(free_cashflow_ttm / total_revenue_ttm), 1 - rank(pb)) | 用多个基本面指标衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_combo_revenue_yoy_operating_cf_margin | fundamental_growth | positive | 5.26% | 0.83% | 1.95% | 6.31% | 0.83% | 2.68% | 74.30% | mean(rank(revenue_yoy), rank(operating_cf_margin_ttm)) | 将经营现金流利润率、收入同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_combo_revenue_to_liab_fcf_yield | fundamental_safety | positive | 2.09% | 0.81% | 1.47% | 1.87% | 0.81% | 1.46% | 72.95% | mean(rank(total_revenue_ttm / total_liab), rank(free_cashflow_ttm / total_mv)) | 用收入对负债覆盖、自由现金流收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_combo_net_profit_yoy_operating_cf_margin | fundamental_growth | positive | 3.82% | 0.79% | 1.36% | 5.20% | 0.79% | 1.99% | 74.30% | mean(rank(net_profit_yoy), rank(operating_cf_margin_ttm)) | 将经营现金流利润率、净利润同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_ind_neu_roe_low_pe | fundamental_value | positive | 2.96% | 0.78% | 1.51% | 3.39% | 0.78% | 1.75% | 64.82% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pe_ttm))) | 将净资产收益率、低市盈率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_reported_ocf_to_or_fcf_yield | fundamental_value | positive | 0.76% | 1.70% | 1.83% | 4.02% | 0.76% | 1.43% | 72.95% | mean(rank(ocf_to_or), rank(free_cashflow_ttm / total_mv)) | 将财务指标表经营现金流收入比、自由现金流收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_ocf_yoy_fcf_margin | fundamental_growth | positive | 2.03% | 1.69% | 0.75% | 0.48% | 0.75% | 1.49% | 71.82% | mean(rank(ocf_yoy), rank(free_cashflow_ttm / total_revenue_ttm)) | 将自由现金流利润率、经营现金流同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_atom_operating_cf_margin | fundamental_quality | positive | 2.16% | 0.72% | 1.19% | 3.94% | 0.72% | 1.36% | 75.52% | rank(operating_cf_margin_ttm) | 用经营现金流利润率验证盈利质量，偏好利润和收入更容易转化为现金的公司。 |
| quality_liquidity_confirm_20 | mixed | positive | 2.84% | 0.71% | 0.64% | 2.84% | 0.64% | 1.40% | 74.84% | zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,20))) | 用经营现金流质量和成交额放大共同确认基本面改善，偏好基本面改善并开始被资金关注的公司。 |
| fund_atom_revenue_yoy | fundamental_growth | positive | 3.00% | 0.64% | 1.71% | 5.10% | 0.64% | 1.78% | 74.31% | rank(revenue_yoy) | 将收入同比增长共同验证成长质量，偏好收入、利润或现金流改善更扎实的公司。 |
| fund_ind_neu_cashflow_to_profit_low_pb | fundamental_value | positive | 0.60% | 3.01% | 3.72% | 2.12% | 0.60% | 2.44% | 75.16% | industry_neutralize(mean(rank(cashflow_to_profit), 1 - rank(pb))) | 将经营现金流利润覆盖、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_receivable_turnover_net_margin | fundamental_efficiency | positive | 3.15% | 0.86% | 0.59% | 4.76% | 0.59% | 1.53% | 73.05% | mean(rank(ar_turn), rank(net_margin_ttm)) | 用净利率、应收账款周转效率衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_combo_leverage_adjusted_quality_value | fundamental_safety | positive | 0.59% | 3.59% | 4.22% | 3.35% | 0.59% | 2.80% | 75.16% | mean(rank(roe_ttm - debt_to_assets), rank(cashflow_to_profit), 1 - rank(pb)) | 用经营现金流利润覆盖、净资产收益率衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_atom_fcf_to_assets | fundamental_quality | positive | 0.56% | 0.94% | 0.99% | 0.39% | 0.56% | 0.83% | 72.95% | rank(free_cashflow_ttm / total_assets) | 用自由现金流资产产出验证盈利质量，偏好利润和收入更容易转化为现金的公司。 |
| fund_combo_cashflow_to_liab_low_pb | fundamental_safety | positive | 0.51% | 4.20% | 4.48% | 3.05% | 0.51% | 3.06% | 75.16% | mean(rank(n_cashflow_act_ttm / total_liab), 1 - rank(pb)) | 用经营现金流对负债覆盖、低市净率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |

## 6. 稳定通过但可能相关的候选

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_operating_cf_margin_low_pb | fundamental_value | positive | 4.08% | 4.05% | 4.11% | 2.94% | 4.05% | 4.08% | 75.16% | mean(rank(operating_cf_margin_ttm), 1 - rank(pb)) | 将经营现金流利润率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_size_neu_operating_cf_margin_low_pb | fundamental_value | positive | 3.99% | 3.38% | 3.92% | 4.14% | 3.38% | 3.76% | 75.16% | size_neutralize(mean(rank(operating_cf_margin_ttm), 1 - rank(pb))) | 将经营现金流利润率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_net_margin_low_pb | fundamental_value | positive | 3.53% | 3.18% | 4.23% | 4.49% | 3.18% | 3.65% | 75.16% | mean(rank(net_margin_ttm), 1 - rank(pb)) | 将净利率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_profit_to_assets_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(net_profit_ttm / total_assets), 1 - rank(pb)) | 将资产盈利能力、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_roa_low_pb | fundamental_value | positive | 3.15% | 3.23% | 4.68% | 5.04% | 3.15% | 3.68% | 75.16% | mean(rank(roa_ttm), 1 - rank(pb)) | 将总资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| dupont_value_quality | fundamental_value | positive | 3.55% | 3.09% | 4.83% | 5.37% | 3.09% | 3.82% | 75.16% | rank(net_margin_ttm) + rank(asset_turnover_ttm) - rank(pb) | 用净利率、资产周转率和市净率估值衡量杜邦质量价值，偏好经营质量好且账面估值不贵的公司。 |
| fund_combo_dupont_low_pb | fundamental_value | positive | 3.55% | 3.09% | 4.83% | 5.37% | 3.09% | 3.82% | 75.16% | mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb)) | 将净利率、资产周转效率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| robust_margin_value_ps | fundamental_value | positive | 3.02% | 3.47% | 4.21% | 4.51% | 3.02% | 3.57% | 75.40% | robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm) | 用稳健标准化后的净利率扣减市销率估值，偏好利润率高且销售估值不贵的公司。 |
| fund_combo_profit_to_liab_low_pb | fundamental_safety | positive | 2.99% | 3.15% | 4.58% | 4.49% | 2.99% | 3.57% | 75.16% | mean(rank(net_profit_ttm / total_liab), 1 - rank(pb)) | 用利润对负债覆盖、低市净率估值衡量资产负债表安全边际，偏好偿债压力较低、现金或盈利覆盖更强的公司。 |
| fund_size_neu_eps_bps_low_pb | fundamental_value | positive | 2.94% | 3.37% | 3.41% | 5.32% | 2.94% | 3.24% | 95.27% | size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb))) | 将每股收益、每股净资产、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_operating_cf_margin_low_ps | fundamental_value | positive | 2.83% | 3.39% | 3.51% | 2.95% | 2.83% | 3.24% | 75.40% | mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)) | 将经营现金流利润率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_gross_to_net_margin_low_pb | fundamental_efficiency | positive | 3.95% | 2.82% | 4.17% | 4.65% | 2.82% | 3.65% | 72.75% | mean(rank(net_margin_ttm / gross_margin_ttm), 1 - rank(pb)) | 用毛利留存为净利的效率、净利率、毛利率、低市净率估值衡量营运效率，偏好资产、应收或存货周转更顺畅、资金占用更轻的公司。 |
| fund_combo_eps_low_pb | fundamental_value | positive | 2.76% | 3.90% | 4.48% | 5.80% | 2.76% | 3.71% | 95.27% | mean(rank(eps), 1 - rank(pb)) | 将每股收益、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_size_neu_gross_net_cash_value | fundamental_value | positive | 5.65% | 2.69% | 3.88% | 5.07% | 2.69% | 4.07% | 72.98% | size_neutralize(mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm))) | 将经营现金流利润率、净利率、毛利率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做规模中性处理，降低市值风格干扰。 |
| fund_combo_revenue_profit_low_pe_hmean | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm)) | 用调和平均组合收入同比增长、净利润同比增长、低市盈率估值，偏好多条基本面腿都不弱、短板更少的公司。 |
| growth_value_balance | fundamental_growth | positive | 2.82% | 2.68% | 3.17% | 4.79% | 2.68% | 2.89% | 63.75% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) - rank(pe_ttm) | 用收入和利润同步增长叠加低市盈率估值，偏好成长没有被价格过度透支的公司。 |
| fund_ind_neu_ocf_to_assets_low_pb | fundamental_value | positive | 2.90% | 2.53% | 4.01% | 3.97% | 2.53% | 3.14% | 75.16% | industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))) | 将经营现金流资产产出、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_operating_cf_margin_earnings_yield | fundamental_value | positive | 3.57% | 2.88% | 2.51% | 2.64% | 2.51% | 2.99% | 64.82% | mean(rank(operating_cf_margin_ttm), rank(1 / pe_ttm)) | 将经营现金流利润率、盈利收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_operating_cf_margin_low_pe | fundamental_value | positive | 3.57% | 2.88% | 2.51% | 2.64% | 2.51% | 2.99% | 64.82% | mean(rank(operating_cf_margin_ttm), 1 - rank(pe_ttm)) | 将经营现金流利润率、低市盈率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_leverage_adjusted_roe_low_pb | fundamental_value | positive | 2.44% | 3.09% | 3.84% | 1.49% | 2.44% | 3.12% | 75.16% | mean(rank(roe_ttm - debt_to_assets), 1 - rank(pb)) | 将杠杆调整后净资产收益率、净资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_eps_bps_low_pb | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | mean(rank(eps), rank(bps), 1 - rank(pb)) | 将每股收益、每股净资产、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| eps_bps_value_quality | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | rank(eps) + rank(bps) - rank(pb) | 用每股收益、每股净资产和 PB 估值共同衡量每股价值质量，偏好盈利和账面基础较好且估值不贵的公司。 |
| fund_combo_roa_ocf_low_pb | fundamental_value | positive | 2.37% | 2.32% | 3.63% | 5.84% | 2.32% | 2.78% | 75.16% | mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)) | 将总资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_reported_ocf_to_or_low_pe | fundamental_value | positive | 2.31% | 2.95% | 2.68% | 3.29% | 2.31% | 2.65% | 80.13% | mean(rank(ocf_to_or), 1 - rank(pe_ttm)) | 将财务指标表经营现金流收入比、低市盈率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_reported_ocf_to_or_earnings_yield | fundamental_value | positive | 2.31% | 2.95% | 2.68% | 3.29% | 2.31% | 2.65% | 80.13% | mean(rank(ocf_to_or), rank(1 / pe_ttm)) | 将财务指标表经营现金流收入比、盈利收益率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_eps_low_ps | fundamental_value | positive | 2.73% | 2.24% | 3.04% | 4.97% | 2.24% | 2.67% | 95.51% | mean(rank(eps), 1 - rank(ps_ttm)) | 将每股收益、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_ind_neu_roe_cash_low_pb | fundamental_value | positive | 2.88% | 2.24% | 3.58% | 4.41% | 2.24% | 2.90% | 75.16% | industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb))) | 将经营现金流利润覆盖、净资产收益率、低市净率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。已做行业中性处理，主要比较同行业内的个股差异。 |
| fund_combo_gross_net_cash_value | fundamental_value | positive | 5.10% | 2.19% | 3.41% | 6.11% | 2.19% | 3.57% | 72.99% | mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)) | 将经营现金流利润率、净利率、毛利率放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| fund_combo_net_margin_low_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | mean(rank(net_margin_ttm), 1 - rank(ps_ttm)) | 将净利率、低市销率估值放在同一截面内比较，偏好基本面质量较好且估值没有明显透支的公司。 |
| margin_value_ps | fundamental_value | positive | 2.12% | 2.90% | 4.07% | 5.00% | 2.12% | 3.03% | 75.40% | rank(net_margin_ttm) - rank(ps_ttm) | 用净利率扣减 PS 估值，偏好利润率较高且销售估值不贵的公司。 |

## 7. 原三因子在扩展框架中的位置

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_growth_hmean | fundamental_growth | positive | 2.58% | 1.97% | 2.57% | 6.04% | 1.97% | 2.38% | 74.30% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit) | 用收入增长、利润增长和现金流覆盖共同确认成长质量，偏好增长更同步、更干净的公司。 |
| eps_bps_value_quality | fundamental_value | positive | 2.96% | 2.33% | 2.94% | 6.87% | 2.33% | 2.74% | 95.27% | rank(eps) + rank(bps) - rank(pb) | 用每股收益、每股净资产和 PB 估值共同衡量每股价值质量，偏好盈利和账面基础较好且估值不贵的公司。 |
| industry_neutral_roe_value_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(rank(roe_ttm) - rank(pb)) | 在行业中性后比较净资产收益率和市净率，偏好同行业内盈利能力强且账面估值不贵的公司。已做行业中性处理，主要比较同行业内的个股差异。 |

解释：原三因子仍然有效，但新算子里可能出现更细分的价值质量、现金流质量或营运效率因子。后续接入 XGBoost 或指数增强时，不应简单把所有通过因子一起加入，而应优先使用相关性去重后的推荐集合。

## 8. 经济含义总结

扩展后的算子主要强化了五类基本面信号：

1. 质量价值：高 ROE/ROA/利润率，同时 PB/PE/PS 不贵。
2. 现金流质量：经营现金流、自由现金流、现金利润覆盖共同确认盈利。
3. 营运效率：资产周转、存货周转、应收周转、低营运资本占用。
4. 安全边际：低负债、高现金覆盖、较强短期偿债能力。
5. 杠杆调整盈利：区分真实经营回报和高负债放大出来的账面回报。

这些因子比单独看成长或单独看估值更稳，因为它们要求“好公司”和“不太贵”同时成立，或者要求利润表结果被现金流量表和资产负债表验证。

## 9. 因子逐项说明

下面逐项解释最终推荐集合中的每个因子。所有 `rank` 都是在同一交易日的中证500截面内计算，方向已经统一为分数越高越好。

### 1. `fund_combo_operating_cf_margin_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(operating_cf_margin_ttm), 1 - rank(pb))`。经营现金流利润率，衡量收入转成经营现金流的能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `4.08%`，valid RankIC `4.05%`，test RankIC `4.11%`，2026YTD RankIC `2.94%`，三段最小调整后 RankIC `4.05%`，覆盖率下限 `75.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 2. `fund_combo_net_margin_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(net_margin_ttm), 1 - rank(pb))`。净利率，反映成本控制、议价能力和商业模式质量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `3.53%`，valid RankIC `3.18%`，test RankIC `4.23%`，2026YTD RankIC `4.49%`，三段最小调整后 RankIC `3.18%`，覆盖率下限 `75.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 3. `fund_combo_profit_to_assets_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(net_profit_ttm / total_assets), 1 - rank(pb))`。利润/资产，接近 ROA 口径，衡量资产盈利效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `3.15%`，valid RankIC `3.23%`，test RankIC `4.68%`，2026YTD RankIC `5.04%`，三段最小调整后 RankIC `3.15%`，覆盖率下限 `75.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 4. `robust_margin_value_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm)`。净利率，反映成本控制、议价能力和商业模式质量；PS，衡量收入相对估值。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。
- 当前表现：train RankIC `3.02%`，valid RankIC `3.47%`，test RankIC `4.21%`，2026YTD RankIC `4.51%`，三段最小调整后 RankIC `3.02%`，覆盖率下限 `75.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 5. `fund_size_neu_eps_bps_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `size_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb)))`。每股收益，衡量单股盈利能力；每股净资产，衡量单股账面资产基础；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 当前表现：train RankIC `2.94%`，valid RankIC `3.37%`，test RankIC `3.41%`，2026YTD RankIC `5.32%`，三段最小调整后 RankIC `2.94%`，覆盖率下限 `95.27%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。该因子已削弱市值暴露，和风格约束组合时更稳。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 6. `fund_combo_operating_cf_margin_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(operating_cf_margin_ttm), 1 - rank(ps_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.83%`，valid RankIC `3.39%`，test RankIC `3.51%`，2026YTD RankIC `2.95%`，三段最小调整后 RankIC `2.83%`，覆盖率下限 `75.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 7. `fund_combo_gross_to_net_margin_low_pb`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(net_margin_ttm / gross_margin_ttm), 1 - rank(pb))`。净利率，反映成本控制、议价能力和商业模式质量；净利率/毛利率，衡量毛利最终留存为净利润的比例；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `3.95%`，valid RankIC `2.82%`，test RankIC `4.17%`，2026YTD RankIC `4.65%`，三段最小调整后 RankIC `2.82%`，覆盖率下限 `72.75%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 8. `fund_size_neu_gross_net_cash_value`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `size_neutralize(mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)))`。经营现金流利润率，衡量收入转成经营现金流的能力；净利率，反映成本控制、议价能力和商业模式质量；净利率/毛利率，衡量毛利最终留存为净利润的比例；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜；规模中性：剔除市值风格影响。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 当前表现：train RankIC `5.65%`，valid RankIC `2.69%`，test RankIC `3.88%`，2026YTD RankIC `5.07%`，三段最小调整后 RankIC `2.69%`，覆盖率下限 `72.98%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。该因子已削弱市值暴露，和风格约束组合时更稳。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 9. `fund_combo_revenue_profit_low_pe_hmean`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), 1 - rank(pe_ttm))`。收入同比增长，衡量需求和业务扩张；利润同比增长，衡量增长是否落到利润表底线；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高；调和平均：惩罚单腿特别弱的公司。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。调和平均会惩罚单项短板，因此比简单相加更偏向“多条腿都不差”的稳健公司。
- 当前表现：train RankIC `2.82%`，valid RankIC `2.68%`，test RankIC `3.17%`，2026YTD RankIC `4.79%`，三段最小调整后 RankIC `2.68%`，覆盖率下限 `63.75%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 10. `fund_ind_neu_ocf_to_assets_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)))`。经营现金流/总资产，衡量资产产生现金流的效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.90%`，valid RankIC `2.53%`，test RankIC `4.01%`，2026YTD RankIC `3.97%`，三段最小调整后 RankIC `2.53%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 11. `fund_combo_leverage_adjusted_roe_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(roe_ttm - debt_to_assets), 1 - rank(pb))`。ROE，衡量股东权益回报；杠杆调整 ROE，惩罚靠高负债堆出来的盈利能力；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。杠杆调整后的盈利能力有效，是因为它区分了真实经营回报和高负债放大出来的账面回报。
- 当前表现：train RankIC `2.44%`，valid RankIC `3.09%`，test RankIC `3.84%`，2026YTD RankIC `1.49%`，三段最小调整后 RankIC `2.44%`，覆盖率下限 `75.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 12. `fund_combo_roa_ocf_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb))`。经营现金流/总资产，衡量资产产生现金流的效率；ROA，衡量资产层面的盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.37%`，valid RankIC `2.32%`，test RankIC `3.63%`，2026YTD RankIC `5.84%`，三段最小调整后 RankIC `2.32%`，覆盖率下限 `75.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 13. `fund_combo_reported_ocf_to_or_low_pe`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(ocf_to_or), 1 - rank(pe_ttm))`。经营现金流收入比，验证收入质量；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.31%`，valid RankIC `2.95%`，test RankIC `2.68%`，2026YTD RankIC `3.29%`，三段最小调整后 RankIC `2.31%`，覆盖率下限 `80.13%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 14. `fund_combo_eps_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(eps), 1 - rank(ps_ttm))`。每股收益，衡量单股盈利能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.73%`，valid RankIC `2.24%`，test RankIC `3.04%`，2026YTD RankIC `4.97%`，三段最小调整后 RankIC `2.24%`，覆盖率下限 `95.51%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 15. `fund_ind_neu_roe_cash_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(roe_ttm), rank(cashflow_to_profit), 1 - rank(pb)))`。经营现金流对利润的覆盖，验证会计利润含金量；ROE，衡量股东权益回报；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.88%`，valid RankIC `2.24%`，test RankIC `3.58%`，2026YTD RankIC `4.41%`，三段最小调整后 RankIC `2.24%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 16. `fund_combo_net_margin_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(net_margin_ttm), 1 - rank(ps_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.12%`，valid RankIC `2.90%`，test RankIC `4.07%`，2026YTD RankIC `5.00%`，三段最小调整后 RankIC `2.12%`，覆盖率下限 `75.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 17. `fund_combo_reported_ocf_to_or_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(ocf_to_or), 1 - rank(ps_ttm))`。经营现金流收入比，验证收入质量；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.10%`，valid RankIC `3.59%`，test RankIC `3.34%`，2026YTD RankIC `3.67%`，三段最小调整后 RankIC `2.10%`，覆盖率下限 `95.65%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 18. `fund_combo_revenue_profit_cash_hmean`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit))`。经营现金流对利润的覆盖，验证会计利润含金量；收入同比增长，衡量需求和业务扩张；利润同比增长，衡量增长是否落到利润表底线；调和平均：惩罚单腿特别弱的公司。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。调和平均会惩罚单项短板，因此比简单相加更偏向“多条腿都不差”的稳健公司。
- 当前表现：train RankIC `2.58%`，valid RankIC `1.97%`，test RankIC `2.57%`，2026YTD RankIC `6.04%`，三段最小调整后 RankIC `1.97%`，覆盖率下限 `74.30%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 19. `fund_ind_neu_leverage_adjusted_quality_value`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(roe_ttm - debt_to_assets), rank(cashflow_to_profit), 1 - rank(pb)))`。经营现金流对利润的覆盖，验证会计利润含金量；ROE，衡量股东权益回报；杠杆调整 ROE，惩罚靠高负债堆出来的盈利能力；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。杠杆调整后的盈利能力有效，是因为它区分了真实经营回报和高负债放大出来的账面回报。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.44%`，valid RankIC `1.93%`，test RankIC `3.39%`，2026YTD RankIC `2.48%`，三段最小调整后 RankIC `1.93%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 20. `fund_combo_reported_ocf_to_or_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(ocf_to_or), 1 - rank(pb))`。经营现金流收入比，验证收入质量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `1.91%`，valid RankIC `4.43%`，test RankIC `4.27%`，2026YTD RankIC `4.01%`，三段最小调整后 RankIC `1.91%`，覆盖率下限 `95.41%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 21. `fund_ind_neu_roe_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb)))`。ROE，衡量股东权益回报；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `3.39%`，valid RankIC `1.88%`，test RankIC `3.82%`，2026YTD RankIC `5.03%`，三段最小调整后 RankIC `1.88%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 22. `shareholder_yield_quality`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `rank(dv_ttm) + rank(roe_ttm)`。ROE，衡量股东权益回报；股息率，衡量现金分红回报。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。股息率有效，是因为持续分红代表现金回报和治理约束，也给估值提供一定锚。
- 当前表现：train RankIC `1.85%`，valid RankIC `2.69%`，test RankIC `2.01%`，2026YTD RankIC `3.49%`，三段最小调整后 RankIC `1.85%`，覆盖率下限 `56.20%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 23. `fund_combo_fcf_to_assets_earnings_yield`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(free_cashflow_ttm / total_assets), rank(1 / pe_ttm))`。自由现金流/总资产，衡量资产真实产出现金的能力；盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司；PE，衡量盈利相对估值。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `1.80%`，valid RankIC `3.22%`，test RankIC `3.04%`，2026YTD RankIC `1.20%`，三段最小调整后 RankIC `1.80%`，覆盖率下限 `62.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 24. `fund_atom_dividend_yield`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `rank(dv_ttm)`。股息率，衡量现金分红回报。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。股息率有效，是因为持续分红代表现金回报和治理约束，也给估值提供一定锚。
- 当前表现：train RankIC `1.74%`，valid RankIC `5.15%`，test RankIC `3.19%`，2026YTD RankIC `1.85%`，三段最小调整后 RankIC `1.74%`，覆盖率下限 `73.66%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 25. `fund_combo_eps_earnings_yield`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(eps), rank(1 / pe_ttm))`。每股收益，衡量单股盈利能力；盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司；PE，衡量盈利相对估值。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.17%`，valid RankIC `1.71%`，test RankIC `1.97%`，2026YTD RankIC `4.03%`，三段最小调整后 RankIC `1.71%`，覆盖率下限 `80.16%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 26. `fund_size_neu_cash_safety_low_pb`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `size_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb)))`。现金/负债，衡量现金对债务的覆盖；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 当前表现：train RankIC `2.60%`，valid RankIC `1.66%`，test RankIC `2.20%`，2026YTD RankIC `1.11%`，三段最小调整后 RankIC `1.66%`，覆盖率下限 `94.99%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。该因子已削弱市值暴露，和风格约束组合时更稳。

### 27. `fund_ind_neu_reported_cash_margin_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(ocf_to_profit), rank(netprofit_margin), 1 - rank(pb)))`。经营现金流利润覆盖比，验证利润现金含量；财务指标表中的净利率，反映费用控制后的最终盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.65%`，valid RankIC `1.75%`，test RankIC `3.00%`，2026YTD RankIC `2.47%`，三段最小调整后 RankIC `1.65%`，覆盖率下限 `83.89%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 28. `fund_combo_quick_ratio_earnings_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(quick_ratio), rank(1 / pe_ttm))`。盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司；速动比率，衡量短期偿债安全垫；PE，衡量盈利相对估值。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `2.05%`，valid RankIC `1.58%`，test RankIC `1.58%`，2026YTD RankIC `0.36%`，三段最小调整后 RankIC `1.58%`，覆盖率下限 `76.35%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 29. `fund_combo_fcf_to_assets_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(free_cashflow_ttm / total_assets), 1 - rank(ps_ttm))`。自由现金流/总资产，衡量资产真实产出现金的能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `1.57%`，valid RankIC `2.78%`，test RankIC `2.78%`，2026YTD RankIC `1.11%`，三段最小调整后 RankIC `1.57%`，覆盖率下限 `72.85%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 30. `fund_size_neu_reported_cash_margin_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `size_neutralize(mean(rank(ocf_to_profit), rank(netprofit_margin), 1 - rank(pb)))`。经营现金流利润覆盖比，验证利润现金含量；财务指标表中的净利率，反映费用控制后的最终盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 当前表现：train RankIC `1.55%`，valid RankIC `3.67%`，test RankIC `3.62%`，2026YTD RankIC `4.24%`，三段最小调整后 RankIC `1.55%`，覆盖率下限 `83.89%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。该因子已削弱市值暴露，和风格约束组合时更稳。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 31. `fund_combo_revenue_yoy_cash_conversion_spread`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(revenue_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_revenue_ttm))`。收入同比增长，衡量需求和业务扩张。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `2.74%`，valid RankIC `1.75%`，test RankIC `1.55%`，2026YTD RankIC `2.46%`，三段最小调整后 RankIC `1.55%`，覆盖率下限 `74.30%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。

### 32. `fund_combo_receivable_turnover_low_pb`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(ar_turn), 1 - rank(pb))`。应收账款周转率，衡量回款效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `1.51%`，valid RankIC `4.82%`，test RankIC `2.87%`，2026YTD RankIC `0.26%`，三段最小调整后 RankIC `1.51%`，覆盖率下限 `92.25%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 33. `fund_atom_inventory_turnover`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `rank(inv_turn)`。存货周转率，衡量库存消化效率。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `1.61%`，valid RankIC `1.46%`，test RankIC `1.70%`，2026YTD RankIC `2.59%`，三段最小调整后 RankIC `1.46%`，覆盖率下限 `91.77%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 34. `fund_combo_net_margin_low_pe`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(net_margin_ttm), 1 - rank(pe_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `3.48%`，valid RankIC `1.46%`，test RankIC `1.86%`，2026YTD RankIC `2.80%`，三段最小调整后 RankIC `1.46%`，覆盖率下限 `64.82%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 35. `fund_combo_profit_to_assets_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(net_profit_ttm / total_assets), 1 - rank(ps_ttm))`。利润/资产，接近 ROA 口径，衡量资产盈利效率；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `1.45%`，valid RankIC `1.45%`，test RankIC `2.81%`，2026YTD RankIC `3.88%`，三段最小调整后 RankIC `1.45%`，覆盖率下限 `75.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 36. `fund_combo_cash_to_assets_low_pe`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(money_cap / total_assets), 1 - rank(pe_ttm))`。货币资金/资产，衡量现金储备；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.42%`，valid RankIC `2.29%`，test RankIC `1.69%`，2026YTD RankIC `0.10%`，三段最小调整后 RankIC `1.42%`，覆盖率下限 `78.25%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 37. `fund_combo_net_profit_yoy_reported_ocf_to_profit`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(net_profit_yoy), rank(ocf_to_profit))`。经营现金流利润覆盖比，验证利润现金含量；利润同比增长，衡量增长是否落到利润表底线。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `1.42%`，valid RankIC `1.53%`，test RankIC `1.39%`，2026YTD RankIC `5.44%`，三段最小调整后 RankIC `1.39%`，覆盖率下限 `65.51%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 38. `fund_combo_revenue_yoy_reported_ocf_to_profit`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(revenue_yoy), rank(ocf_to_profit))`。经营现金流利润覆盖比，验证利润现金含量；收入同比增长，衡量需求和业务扩张。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `1.71%`，valid RankIC `1.64%`，test RankIC `1.37%`，2026YTD RankIC `5.40%`，三段最小调整后 RankIC `1.37%`，覆盖率下限 `65.51%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 39. `fund_ind_neu_dupont_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(net_margin_ttm), rank(asset_turnover_ttm), 1 - rank(pb)))`。净利率，反映成本控制、议价能力和商业模式质量；资产周转率，衡量资产产生收入的效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `3.28%`，valid RankIC `1.36%`，test RankIC `3.88%`，2026YTD RankIC `4.81%`，三段最小调整后 RankIC `1.36%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 40. `fund_combo_inventory_turnover_fcf_margin`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(inv_turn), rank(free_cashflow_ttm / total_revenue_ttm))`。自由现金流/收入，衡量收入转化为自由现金的能力；存货周转率，衡量库存消化效率。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `1.35%`，valid RankIC `1.44%`，test RankIC `1.61%`，2026YTD RankIC `2.12%`，三段最小调整后 RankIC `1.35%`，覆盖率下限 `71.98%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 41. `fund_ind_neu_eps_bps_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(eps), rank(bps), 1 - rank(pb)))`。每股收益，衡量单股盈利能力；每股净资产，衡量单股账面资产基础；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.34%`，valid RankIC `1.34%`，test RankIC `2.47%`，2026YTD RankIC `5.75%`，三段最小调整后 RankIC `1.34%`，覆盖率下限 `95.27%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 42. `fund_combo_quick_ratio_low_pb`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(quick_ratio), 1 - rank(pb))`。速动比率，衡量短期偿债安全垫；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.33%`，valid RankIC `3.17%`，test RankIC `3.24%`，2026YTD RankIC `0.04%`，三段最小调整后 RankIC `1.33%`，覆盖率下限 `92.96%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 43. `fund_ind_neu_fcf_assets_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(free_cashflow_ttm / total_assets), 1 - rank(debt_to_assets), 1 - rank(pb)))`。自由现金流/总资产，衡量资产真实产出现金的能力；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.33%`，valid RankIC `1.41%`，test RankIC `3.17%`，2026YTD RankIC `1.92%`，三段最小调整后 RankIC `1.33%`，覆盖率下限 `72.60%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 44. `fund_combo_leverage_adjusted_roe_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(roe_ttm - debt_to_assets), 1 - rank(ps_ttm))`。ROE，衡量股东权益回报；杠杆调整 ROE，惩罚靠高负债堆出来的盈利能力；资产负债率，衡量杠杆压力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。杠杆调整后的盈利能力有效，是因为它区分了真实经营回报和高负债放大出来的账面回报。
- 当前表现：train RankIC `1.31%`，valid RankIC `1.89%`，test RankIC `2.93%`，2026YTD RankIC `1.42%`，三段最小调整后 RankIC `1.31%`，覆盖率下限 `75.40%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。

### 45. `fund_combo_revenue_to_liab_earnings_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(total_revenue_ttm / total_liab), rank(1 / pe_ttm))`。收入/总负债，衡量经营规模对债务负担的覆盖；盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司；PE，衡量盈利相对估值。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.83%`，valid RankIC `1.30%`，test RankIC `1.85%`，2026YTD RankIC `0.66%`，三段最小调整后 RankIC `1.30%`，覆盖率下限 `64.82%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。

### 46. `fund_combo_cashflow_to_liab_fcf_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(n_cashflow_act_ttm / total_liab), rank(free_cashflow_ttm / total_mv))`。经营现金流/总负债，衡量经营现金流对债务压力的覆盖；自由现金流收益率，衡量现金回报相对市值是否便宜。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.73%`，valid RankIC `1.30%`，test RankIC `1.83%`，2026YTD RankIC `2.97%`，三段最小调整后 RankIC `1.30%`，覆盖率下限 `72.95%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。

### 47. `fund_combo_inventory_turnover_low_pb`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(inv_turn), 1 - rank(pb))`。存货周转率，衡量库存消化效率；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `1.29%`，valid RankIC `4.11%`，test RankIC `3.84%`，2026YTD RankIC `1.24%`，三段最小调整后 RankIC `1.29%`，覆盖率下限 `91.41%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 48. `fund_ind_neu_working_capital_cash_growth`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `industry_neutralize(mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit), rank(net_profit_yoy)))`。经营现金流对利润的覆盖，验证会计利润含金量；利润同比增长，衡量增长是否落到利润表底线；营运资本占用压力，主要来自应收和存货；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.27%`，valid RankIC `1.29%`，test RankIC `1.34%`，2026YTD RankIC `2.35%`，三段最小调整后 RankIC `1.27%`，覆盖率下限 `63.49%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 49. `fund_combo_low_debt_assets_earnings_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(1 - rank(debt_to_assets), rank(1 / pe_ttm))`。盈利收益率，相比简单低 PE 能自然惩罚负 PE 公司；资产负债率，衡量杠杆压力；PE，衡量盈利相对估值。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.68%`，valid RankIC `1.22%`，test RankIC `2.01%`，2026YTD RankIC `0.65%`，三段最小调整后 RankIC `1.22%`，覆盖率下限 `80.16%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。

### 50. `fund_combo_inventory_turnover_reported_ocf_to_profit`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(inv_turn), rank(ocf_to_profit))`。经营现金流利润覆盖比，验证利润现金含量；存货周转率，衡量库存消化效率。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `1.22%`，valid RankIC `1.61%`，test RankIC `1.44%`，2026YTD RankIC `2.67%`，三段最小调整后 RankIC `1.22%`，覆盖率下限 `78.81%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 51. `fund_ind_neu_reported_ocf_to_profit_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(ocf_to_profit), 1 - rank(pb)))`。经营现金流利润覆盖比，验证利润现金含量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.22%`，valid RankIC `2.58%`，test RankIC `3.34%`，2026YTD RankIC `1.75%`，三段最小调整后 RankIC `1.22%`，覆盖率下限 `83.94%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 52. `fund_combo_revenue_yoy_fcf_margin`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(revenue_yoy), rank(free_cashflow_ttm / total_revenue_ttm))`。自由现金流/收入，衡量收入转化为自由现金的能力；收入同比增长，衡量需求和业务扩张。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `3.91%`，valid RankIC `1.22%`，test RankIC `1.50%`，2026YTD RankIC `3.58%`，三段最小调整后 RankIC `1.22%`，覆盖率下限 `71.82%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。

### 53. `fund_combo_net_profit_yoy_accrual_quality`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(net_profit_yoy), rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets))`。利润同比增长，衡量增长是否落到利润表底线；现金流与利润差额/资产，衡量利润是否由现金支持。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `1.66%`，valid RankIC `1.64%`，test RankIC `1.21%`，2026YTD RankIC `1.37%`，三段最小调整后 RankIC `1.21%`，覆盖率下限 `74.30%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。

### 54. `fund_atom_reported_ocf_to_or`

- 因子族：`fundamental_quality`。
- 计算过程：表达式为 `rank(ocf_to_or)`。经营现金流收入比，验证收入质量。
- 经济学直觉：有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。
- 当前表现：train RankIC `2.08%`，valid RankIC `1.19%`，test RankIC `1.49%`，2026YTD RankIC `5.63%`，三段最小调整后 RankIC `1.19%`，覆盖率下限 `95.77%`。
- 使用方式：适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 55. `fund_atom_reported_ocf_to_profit`

- 因子族：`fundamental_quality`。
- 计算过程：表达式为 `rank(ocf_to_profit)`。经营现金流利润覆盖比，验证利润现金含量。
- 经济学直觉：有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。
- 当前表现：train RankIC `1.18%`，valid RankIC `1.99%`，test RankIC `1.24%`，2026YTD RankIC `3.55%`，三段最小调整后 RankIC `1.18%`，覆盖率下限 `84.00%`。
- 使用方式：适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。

### 56. `fund_combo_low_working_capital_pressure_reported_ocf_to_profit`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(1 - rank(working_capital_pressure), rank(ocf_to_profit))`。经营现金流利润覆盖比，验证利润现金含量；营运资本占用压力，主要来自应收和存货。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `2.99%`，valid RankIC `2.52%`，test RankIC `1.17%`，2026YTD RankIC `1.55%`，三段最小调整后 RankIC `1.17%`，覆盖率下限 `65.49%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 57. `fund_ind_neu_cash_safety_low_pb`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(cash_to_liab), 1 - rank(debt_to_assets), 1 - rank(pb)))`。现金/负债，衡量现金对债务的覆盖；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.62%`，valid RankIC `1.16%`，test RankIC `2.34%`，2026YTD RankIC `0.60%`，三段最小调整后 RankIC `1.16%`，覆盖率下限 `94.99%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 58. `fund_ind_neu_gross_net_cash_value`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(net_margin_ttm / gross_margin_ttm), rank(operating_cf_margin_ttm), 1 - rank(ps_ttm)))`。经营现金流利润率，衡量收入转成经营现金流的能力；净利率，反映成本控制、议价能力和商业模式质量；净利率/毛利率，衡量毛利最终留存为净利润的比例；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.77%`，valid RankIC `1.16%`，test RankIC `2.28%`，2026YTD RankIC `4.34%`，三段最小调整后 RankIC `1.16%`，覆盖率下限 `72.99%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 59. `fund_size_neu_fcf_assets_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `size_neutralize(mean(rank(free_cashflow_ttm / total_assets), 1 - rank(debt_to_assets), 1 - rank(pb)))`。自由现金流/总资产，衡量资产真实产出现金的能力；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；规模中性：剔除市值风格影响。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。规模中性处理降低了大小盘风格干扰，后续用于指数增强时更容易和市值约束兼容。
- 当前表现：train RankIC `1.16%`，valid RankIC `2.36%`，test RankIC `3.38%`，2026YTD RankIC `2.34%`，三段最小调整后 RankIC `1.16%`，覆盖率下限 `72.59%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。该因子已削弱市值暴露，和风格约束组合时更稳。

### 60. `fund_ind_neu_roa_ocf_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(roa_ttm), rank(n_cashflow_act_ttm / total_assets), 1 - rank(pb)))`。经营现金流/总资产，衡量资产产生现金流的效率；ROA，衡量资产层面的盈利能力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `3.64%`，valid RankIC `1.15%`，test RankIC `3.06%`，2026YTD RankIC `5.03%`，三段最小调整后 RankIC `1.15%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 61. `industry_rank_cash_profit_cover`

- 因子族：`fundamental_quality`。
- 计算过程：表达式为 `group_rank(n_cashflow_act_ttm / net_profit_ttm, industry)`。经营现金流/净利润，衡量利润的现金覆盖程度；行业内排名：只在同一行业内部比较，降低行业结构噪声。
- 经济学直觉：有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。现金覆盖和应计质量有效，是因为利润若长期缺少经营现金流支撑，后续更容易发生盈利回撤或估值折价。行业内排名有效，是因为不同行业的现金流周期和利润率天然不同，同行业比较能减少结构性偏差。
- 当前表现：train RankIC `1.21%`，valid RankIC `1.76%`，test RankIC `1.13%`，2026YTD RankIC `0.81%`，三段最小调整后 RankIC `1.13%`，覆盖率下限 `75.52%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 62. `turnover_efficiency_combo`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `rank(inv_turn) + rank(ar_turn)`。存货周转率，衡量库存消化效率；应收账款周转率，衡量回款效率。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `2.08%`，valid RankIC `2.14%`，test RankIC `1.05%`，2026YTD RankIC `2.00%`，三段最小调整后 RankIC `1.05%`，覆盖率下限 `91.35%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 63. `fund_ind_neu_fcf_margin_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(free_cashflow_ttm / total_revenue_ttm), rank(ocf_to_or), 1 - rank(ps_ttm)))`。经营现金流收入比，验证收入质量；自由现金流/收入，衡量收入转化为自由现金的能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `1.03%`，valid RankIC `1.54%`，test RankIC `2.48%`，2026YTD RankIC `2.22%`，三段最小调整后 RankIC `1.03%`，覆盖率下限 `72.85%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 64. `fund_combo_fcf_to_profit_low_ps`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(free_cashflow_ttm / net_profit_ttm), 1 - rank(ps_ttm))`。自由现金流/净利润，衡量利润最终沉淀为自由现金的能力；PS，衡量收入相对估值；低 PS：同等收入或利润率下销售估值更便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `1.02%`，valid RankIC `3.84%`，test RankIC `3.06%`，2026YTD RankIC `0.19%`，三段最小调整后 RankIC `1.02%`，覆盖率下限 `72.85%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 65. `fund_combo_cash_to_liab_fcf_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(cash_to_liab), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；现金/负债，衡量现金对债务的覆盖。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `1.68%`，valid RankIC `1.00%`，test RankIC `1.02%`，2026YTD RankIC `0.37%`，三段最小调整后 RankIC `1.00%`，覆盖率下限 `72.95%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 66. `fund_combo_net_profit_yoy_fcf_margin`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(net_profit_yoy), rank(free_cashflow_ttm / total_revenue_ttm))`。自由现金流/收入，衡量收入转化为自由现金的能力；利润同比增长，衡量增长是否落到利润表底线。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `4.12%`，valid RankIC `0.96%`，test RankIC `1.01%`，2026YTD RankIC `3.11%`，三段最小调整后 RankIC `0.96%`，覆盖率下限 `71.82%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。

### 67. `fund_combo_gross_to_net_margin_cashflow_to_profit`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(net_margin_ttm / gross_margin_ttm), rank(cashflow_to_profit))`。经营现金流对利润的覆盖，验证会计利润含金量；净利率，反映成本控制、议价能力和商业模式质量；净利率/毛利率，衡量毛利最终留存为净利润的比例。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。
- 当前表现：train RankIC `2.30%`，valid RankIC `0.95%`，test RankIC `2.77%`，2026YTD RankIC `6.71%`，三段最小调整后 RankIC `0.95%`，覆盖率下限 `73.10%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 68. `clean_growth_quality`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `rank(cashflow_to_profit) + rank(net_profit_yoy) - rank(working_capital_pressure)`。经营现金流对利润的覆盖，验证会计利润含金量；利润同比增长，衡量增长是否落到利润表底线；营运资本占用压力，主要来自应收和存货。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `0.94%`，valid RankIC `2.58%`，test RankIC `2.07%`，2026YTD RankIC `3.11%`，三段最小调整后 RankIC `0.94%`，覆盖率下限 `63.49%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。

### 69. `fund_combo_low_working_capital_pressure_cashflow_to_profit`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(1 - rank(working_capital_pressure), rank(cashflow_to_profit))`。经营现金流对利润的覆盖，验证会计利润含金量；营运资本占用压力，主要来自应收和存货。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `0.94%`，valid RankIC `3.07%`，test RankIC `1.96%`，2026YTD RankIC `1.17%`，三段最小调整后 RankIC `0.94%`，覆盖率下限 `64.66%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 70. `fund_ind_neu_working_capital_fcf_value`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `industry_neutralize(mean(1 - rank((accounts_receiv + inventories) / total_assets), rank(free_cashflow_ttm / total_revenue_ttm), 1 - rank(pb)))`。自由现金流/收入，衡量收入转化为自由现金的能力；应收和存货合计占用，衡量收入扩张背后的营运资本压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `0.91%`，valid RankIC `1.98%`，test RankIC `2.84%`，2026YTD RankIC `1.00%`，三段最小调整后 RankIC `0.91%`，覆盖率下限 `71.52%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 71. `fund_combo_eps_fcf_yield`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(eps), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；每股收益，衡量单股盈利能力。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `2.99%`，valid RankIC `0.88%`，test RankIC `1.73%`，2026YTD RankIC `5.82%`，三段最小调整后 RankIC `0.88%`，覆盖率下限 `72.94%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 72. `fund_combo_low_working_capital_pressure_net_margin`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(1 - rank(working_capital_pressure), rank(net_margin_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；营运资本占用压力，主要来自应收和存货。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `2.56%`，valid RankIC `0.85%`，test RankIC `0.92%`，2026YTD RankIC `3.57%`，三段最小调整后 RankIC `0.85%`，覆盖率下限 `64.66%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。

### 73. `fund_combo_working_capital_fcf_value`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(1 - rank((accounts_receiv + inventories) / total_assets), rank(free_cashflow_ttm / total_revenue_ttm), 1 - rank(pb))`。自由现金流/收入，衡量收入转化为自由现金的能力；应收和存货合计占用，衡量收入扩张背后的营运资本压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `0.85%`，valid RankIC `3.83%`，test RankIC `3.04%`，2026YTD RankIC `0.04%`，三段最小调整后 RankIC `0.85%`，覆盖率下限 `71.52%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 74. `fund_combo_revenue_yoy_operating_cf_margin`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(revenue_yoy), rank(operating_cf_margin_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；收入同比增长，衡量需求和业务扩张。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `5.26%`，valid RankIC `0.83%`，test RankIC `1.95%`，2026YTD RankIC `6.31%`，三段最小调整后 RankIC `0.83%`，覆盖率下限 `74.30%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 75. `fund_combo_revenue_to_liab_fcf_yield`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(total_revenue_ttm / total_liab), rank(free_cashflow_ttm / total_mv))`。自由现金流收益率，衡量现金回报相对市值是否便宜；收入/总负债，衡量经营规模对债务负担的覆盖。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `2.09%`，valid RankIC `0.81%`，test RankIC `1.47%`，2026YTD RankIC `1.87%`，三段最小调整后 RankIC `0.81%`，覆盖率下限 `72.95%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。

### 76. `fund_combo_net_profit_yoy_operating_cf_margin`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(net_profit_yoy), rank(operating_cf_margin_ttm))`。经营现金流利润率，衡量收入转成经营现金流的能力；利润同比增长，衡量增长是否落到利润表底线。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `3.82%`，valid RankIC `0.79%`，test RankIC `1.36%`，2026YTD RankIC `5.20%`，三段最小调整后 RankIC `0.79%`，覆盖率下限 `74.30%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 77. `fund_ind_neu_roe_low_pe`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(roe_ttm), 1 - rank(pe_ttm)))`。ROE，衡量股东权益回报；PE，衡量盈利相对估值；低 PE：同等盈利下收益率更高；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `2.96%`，valid RankIC `0.78%`，test RankIC `1.51%`，2026YTD RankIC `3.39%`，三段最小调整后 RankIC `0.78%`，覆盖率下限 `64.82%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 78. `fund_combo_reported_ocf_to_or_fcf_yield`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `mean(rank(ocf_to_or), rank(free_cashflow_ttm / total_mv))`。经营现金流收入比，验证收入质量；自由现金流收益率，衡量现金回报相对市值是否便宜。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。
- 当前表现：train RankIC `0.76%`，valid RankIC `1.70%`，test RankIC `1.83%`，2026YTD RankIC `4.02%`，三段最小调整后 RankIC `0.76%`，覆盖率下限 `72.95%`。
- 使用方式：适合与行业约束、质量过滤一起使用，避免单纯低估值带来的价值陷阱。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 79. `fund_combo_ocf_yoy_fcf_margin`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `mean(rank(ocf_yoy), rank(free_cashflow_ttm / total_revenue_ttm))`。自由现金流/收入，衡量收入转化为自由现金的能力。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。
- 当前表现：train RankIC `2.03%`，valid RankIC `1.69%`，test RankIC `0.75%`，2026YTD RankIC `0.48%`，三段最小调整后 RankIC `0.75%`，覆盖率下限 `71.82%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 80. `fund_atom_operating_cf_margin`

- 因子族：`fundamental_quality`。
- 计算过程：表达式为 `rank(operating_cf_margin_ttm)`。经营现金流利润率，衡量收入转成经营现金流的能力。
- 经济学直觉：有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。
- 当前表现：train RankIC `2.16%`，valid RankIC `0.72%`，test RankIC `1.19%`，2026YTD RankIC `3.94%`，三段最小调整后 RankIC `0.72%`，覆盖率下限 `75.52%`。
- 使用方式：适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。

### 81. `quality_liquidity_confirm_20`

- 因子族：`mixed`。
- 计算过程：表达式为 `zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,20)))`。经营现金流利润率，衡量收入转成经营现金流的能力；成交额放大，衡量市场关注度和资金确认。
- 经济学直觉：有效性主要来自基本面质量和交易确认的结合：基本面改善并开始被资金关注，但价格尚未完全反映这部分信息。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。成交额确认有效，是因为基本面改善开始被资金交易时，信号兑现速度通常更快。
- 当前表现：train RankIC `2.84%`，valid RankIC `0.71%`，test RankIC `0.64%`，2026YTD RankIC `2.84%`，三段最小调整后 RankIC `0.64%`，覆盖率下限 `74.84%`。
- 使用方式：适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。

### 82. `fund_atom_revenue_yoy`

- 因子族：`fundamental_growth`。
- 计算过程：表达式为 `rank(revenue_yoy)`。收入同比增长，衡量需求和业务扩张。
- 经济学直觉：有效性主要来自更干净的成长识别：收入、利润、现金流或营运效率同时改善时，这类增长更容易被市场持续定价。成长腿有效，是因为中证500公司弹性较高，收入和利润同步改善时，市场更容易上修盈利预期。
- 当前表现：train RankIC `3.00%`，valid RankIC `0.64%`，test RankIC `1.71%`，2026YTD RankIC `5.10%`，三段最小调整后 RankIC `0.64%`，覆盖率下限 `74.31%`。
- 使用方式：适合在财报更新后月度或季度调仓，避免把低频财务信号日频过度交易。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 83. `fund_ind_neu_cashflow_to_profit_low_pb`

- 因子族：`fundamental_value`。
- 计算过程：表达式为 `industry_neutralize(mean(rank(cashflow_to_profit), 1 - rank(pb)))`。经营现金流对利润的覆盖，验证会计利润含金量；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜；行业中性：先剔除行业平均差异，再比较个股。
- 经济学直觉：有效性主要来自质量约束下的估值修复：盈利质量稳定但估值不高的中盘公司，后续更容易获得估值修复或盈利确认带来的横截面收益。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。行业中性处理使这个信号更像行业内选股能力，而不是押注某些行业天然高 ROE 或低估值。
- 当前表现：train RankIC `0.60%`，valid RankIC `3.01%`，test RankIC `3.72%`，2026YTD RankIC `2.12%`，三段最小调整后 RankIC `0.60%`，覆盖率下限 `75.16%`。
- 使用方式：适合直接进入带行业约束的指增模型，也适合做行业内排序。

### 84. `fund_combo_receivable_turnover_net_margin`

- 因子族：`fundamental_efficiency`。
- 计算过程：表达式为 `mean(rank(ar_turn), rank(net_margin_ttm))`。净利率，反映成本控制、议价能力和商业模式质量；应收账款周转率，衡量回款效率。
- 经济学直觉：有效性主要来自营运效率：资产、库存和应收周转更快的公司，占用资本更少，盈利释放和现金回收通常更顺畅。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。营运资本和周转指标有效，是因为低应收、低库存占用和高周转能释放现金流，并降低业绩爆雷概率。
- 当前表现：train RankIC `3.15%`，valid RankIC `0.86%`，test RankIC `0.59%`，2026YTD RankIC `4.76%`，三段最小调整后 RankIC `0.59%`，覆盖率下限 `73.05%`。
- 使用方式：适合与盈利质量、估值因子共同使用，避免只买到短期周转改善但利润率不足的公司。2026YTD 仍有较强正向 RankIC，近期没有明显失效迹象。

### 85. `fund_combo_leverage_adjusted_quality_value`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(roe_ttm - debt_to_assets), rank(cashflow_to_profit), 1 - rank(pb))`。经营现金流对利润的覆盖，验证会计利润含金量；ROE，衡量股东权益回报；杠杆调整 ROE，惩罚靠高负债堆出来的盈利能力；资产负债率，衡量杠杆压力；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。现金流指标有效，是因为现金比会计利润更难被短期调节，能确认收入和利润是否真正变成可支配现金。利润率和资产回报有效，是因为它们反映商业模式、成本控制和资产使用效率，高质量公司更容易穿越周期。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。杠杆调整后的盈利能力有效，是因为它区分了真实经营回报和高负债放大出来的账面回报。
- 当前表现：train RankIC `0.59%`，valid RankIC `3.59%`，test RankIC `4.22%`，2026YTD RankIC `3.35%`，三段最小调整后 RankIC `0.59%`，覆盖率下限 `75.16%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。

### 86. `fund_atom_fcf_to_assets`

- 因子族：`fundamental_quality`。
- 计算过程：表达式为 `rank(free_cashflow_ttm / total_assets)`。自由现金流/总资产，衡量资产真实产出现金的能力。
- 经济学直觉：有效性主要来自盈利质量过滤：用现金流、行业内排名或收入现金化程度验证会计利润，可以降低利润虚高和应收堆积带来的误判。自由现金流相关指标有效，是因为它比经营现金流更进一步扣除了资本开支，更接近股东最终可以分配或再投资的现金。
- 当前表现：train RankIC `0.56%`，valid RankIC `0.94%`，test RankIC `0.99%`，2026YTD RankIC `0.39%`，三段最小调整后 RankIC `0.56%`，覆盖率下限 `72.95%`。
- 使用方式：适合作为辅助特征进入多因子或 XGBoost，不建议单独作为组合权重。2026YTD 表现偏弱，使用时建议降权或作为备选特征。

### 87. `fund_combo_cashflow_to_liab_low_pb`

- 因子族：`fundamental_safety`。
- 计算过程：表达式为 `mean(rank(n_cashflow_act_ttm / total_liab), 1 - rank(pb))`。经营现金流/总负债，衡量经营现金流对债务压力的覆盖；PB，衡量账面价值相对估值；低 PB：同等质量下账面估值更便宜。
- 经济学直觉：有效性主要来自资产负债表安全边际：现金、低杠杆和短债偿付能力强的公司，在风险偏好下降或融资环境收紧时更不容易被惩罚。估值腿有效，是因为它要求市场价格没有提前透支基本面，减少买到好公司但价格过贵的风险。偿债和现金安全指标有效，是因为它们降低财务困境概率，尤其适合在中盘股里排除高杠杆尾部风险。
- 当前表现：train RankIC `0.51%`，valid RankIC `4.20%`，test RankIC `4.48%`，2026YTD RankIC `3.05%`，三段最小调整后 RankIC `0.51%`，覆盖率下限 `75.16%`。
- 使用方式：适合做风险过滤或防守型 alpha，不宜单独追求最高进攻性。


## 10. 风险和下一步

- 这类宽搜索仍存在多重检验问题，不能只看最优因子，需要用 valid/test/YTD 和相关性过滤约束。
- 基本面因子更新频率低，进入指增模型时建议月度或财报后调仓，不建议日频过度交易。
- 行业中性因子更适合指增；非中性因子要单独检查行业暴露。
- 下一步可以把推荐因子加入 XGBoost 特征集合，做 `旧3因子` vs `新基本面算子集合` vs `公开因子+新基本面` 的消融实验。

## 11. 输出文件

- 输出目录：`outputs/fundamental_operator_mining_csi500_extended_2026-07-20`
- 全部拆分汇总：`outputs/fundamental_operator_mining_csi500_extended_2026-07-20/all_split_summary.csv`
- 稳定性表：`outputs/fundamental_operator_mining_csi500_extended_2026-07-20/stable_factors.csv`
- 最终推荐：`outputs/fundamental_operator_mining_csi500_extended_2026-07-20/selected_factors.csv`
- 报告文件：`docs/fundamental_operator_mining_csi500_extended_2026-07-20.md`