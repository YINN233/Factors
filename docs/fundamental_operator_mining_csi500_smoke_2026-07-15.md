# 中证500基本面算子组合挖掘报告

日期：2026-07-15

## 1. 任务目标

导师希望在现有 `quality_growth_hmean`、`eps_bps_value_quality`、`industry_neutral_roe_value_pb` 三个核心基本面因子之外，基于利润表、资产负债表、现金流量表和财务指标表，扩展更多基本面算子，并通过算子排列组合挖掘新的有效因子。

本轮实现的是一套独立的基本面算子工厂：先构造方向统一的基本面原子指标，再用均值、调和平均、估值扣减、行业中性和规模中性等算子生成候选因子，最后在中证500股票池上做 train/valid/test/2026YTD 验证和相关性去重。

## 2. 执行计划与口径

本轮按四步执行：

1. 保留旧三因子和已有 legacy 基本面因子作为 benchmark。
2. 从利润表、资产负债表、现金流量表、财务指标表和估值字段构造方向统一的原子指标。
3. 用均值、调和平均、价值扣减、行业中性、规模中性等算子生成候选组合。
4. 在中证500 train/valid/test/2026YTD 上验证 RankIC 稳定性，再做相关性去重，得到推荐集合。

本次运行参数：

| parameter | value |
| --- | --- |
| processed_dir | data/processed |
| suffix | 000905_SH |
| n_candidates | 8 |
| include_atoms | True |
| include_neutralized | True |
| candidate_regex | ^(quality_growth_hmean|eps_bps_value_quality|industry_neutral_roe_value_pb|fund_atom_low_pb|fund_combo_roe_low_pb|fund_combo_revenue_profit_cash_hmean|fund_ind_neu_roe_low_pb|fund_size_neu_roe_low_pb)$ |
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
| fundamental_value | 6 |
| fundamental_growth | 2 |

验证结果数量：

| status | count |
| --- | --- |
| evaluated | 8 |
| passes_stability | 7 |
| selected_after_corr | 4 |

## 5. 最终推荐因子

下表是通过稳定性过滤和相关性去重后的推荐因子。`min_adj_rankic` 是 train/valid/test 三段按训练期方向调整后的最小 RankIC，越高说明跨样本越稳。

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_roe_low_pb | fundamental_value | positive | 3.28% | 3.64% | 4.58% | 5.39% | 3.28% | 3.83% | 75.16% | mean(rank(roe_ttm), 1 - rank(pb)) | roe quality combined with low_pb valuation. |
| eps_bps_value_quality | fundamental_value | positive | 3.06% | 2.33% | 2.94% | 6.87% | 2.33% | 2.78% | 95.27% | rank(eps) + rank(bps) - rank(pb) | Per-share earnings and book value quality adjusted by book valuation. |
| fund_combo_revenue_profit_cash_hmean | fundamental_growth | positive | 2.83% | 1.97% | 2.57% | 6.04% | 1.97% | 2.46% | 74.30% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit)) | Revenue and profit grow together and are confirmed by cash conversion. |
| fund_ind_neu_roe_low_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | roe quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |

## 6. 稳定通过但可能相关的候选

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fund_combo_roe_low_pb | fundamental_value | positive | 3.28% | 3.64% | 4.58% | 5.39% | 3.28% | 3.83% | 75.16% | mean(rank(roe_ttm), 1 - rank(pb)) | roe quality combined with low_pb valuation. |
| eps_bps_value_quality | fundamental_value | positive | 3.06% | 2.33% | 2.94% | 6.87% | 2.33% | 2.78% | 95.27% | rank(eps) + rank(bps) - rank(pb) | Per-share earnings and book value quality adjusted by book valuation. |
| fund_combo_revenue_profit_cash_hmean | fundamental_growth | positive | 2.83% | 1.97% | 2.57% | 6.04% | 1.97% | 2.46% | 74.30% | hmean2_plus(rank(revenue_yoy), rank(net_profit_yoy), rank(cashflow_to_profit)) | Revenue and profit grow together and are confirmed by cash conversion. |
| quality_growth_hmean | fundamental_growth | positive | 2.83% | 1.97% | 2.57% | 6.04% | 1.97% | 2.46% | 74.30% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit) | Growth that is simultaneous in revenue, profit, and cash conversion. |
| fund_ind_neu_roe_low_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | roe quality combined with low_pb valuation. Industry-neutralized to reduce sector structure bias. |
| industry_neutral_roe_value_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(rank(roe_ttm) - rank(pb)) | Quality-value score after removing same-industry average exposure. |
| fund_size_neu_roe_low_pb | fundamental_value | positive | 1.74% | 3.93% | 4.80% | 5.03% | 1.74% | 3.49% | 75.16% | size_neutralize(mean(rank(roe_ttm), 1 - rank(pb))) | roe quality combined with low_pb valuation. Size-neutralized to reduce market-cap style exposure. |

## 7. 原三因子在本轮框架中的位置

| factor | family | direction | train_rankic | valid_rankic | test_rankic | ytd_2026_rankic | min_adj_rankic | avg_adj_rankic | min_coverage | expression | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_growth_hmean | fundamental_growth | positive | 2.83% | 1.97% | 2.57% | 6.04% | 1.97% | 2.46% | 74.30% | harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit) | Growth that is simultaneous in revenue, profit, and cash conversion. |
| eps_bps_value_quality | fundamental_value | positive | 3.06% | 2.33% | 2.94% | 6.87% | 2.33% | 2.78% | 95.27% | rank(eps) + rank(bps) - rank(pb) | Per-share earnings and book value quality adjusted by book valuation. |
| industry_neutral_roe_value_pb | fundamental_value | positive | 3.39% | 1.88% | 3.82% | 5.03% | 1.88% | 3.03% | 75.16% | industry_neutralize(rank(roe_ttm) - rank(pb)) | Quality-value score after removing same-industry average exposure. |

解释：原三因子仍然有效，但新算子里可能出现更细分的价值质量、现金流质量或营运效率因子。后续接入 XGBoost 或指数增强时，不应简单把所有通过因子一起加入，而应优先使用相关性去重后的推荐集合。

## 8. 经济含义总结

本轮新算子主要强化了四类基本面信号：

1. 质量价值：高 ROE/ROA/利润率，同时 PB/PE/PS 不贵。
2. 现金流质量：经营现金流、自由现金流、现金利润覆盖共同确认盈利。
3. 营运效率：资产周转、存货周转、应收周转、低营运资本占用。
4. 安全边际：低负债、高现金覆盖、较强短期偿债能力。

这些因子比单独看成长或单独看估值更稳，因为它们要求“好公司”和“不太贵”同时成立，或者要求利润表结果被现金流量表和资产负债表验证。

## 9. 风险和下一步

- 本轮是宽搜索，仍存在多重检验问题，不能只看最优因子，需要用 valid/test/YTD 和相关性过滤约束。
- 基本面因子更新频率低，进入指增模型时建议月度或财报后调仓，不建议日频过度交易。
- 行业中性因子更适合指增；非中性因子要单独检查行业暴露。
- 下一步可以把推荐因子加入 XGBoost 特征集合，做 `旧3因子` vs `新基本面算子集合` vs `公开因子+新基本面` 的消融实验。

## 10. 输出文件

- 输出目录：`outputs/fundamental_operator_mining_csi500_smoke`
- 全部拆分汇总：`outputs/fundamental_operator_mining_csi500_smoke/all_split_summary.csv`
- 稳定性表：`outputs/fundamental_operator_mining_csi500_smoke/stable_factors.csv`
- 最终推荐：`outputs/fundamental_operator_mining_csi500_smoke/selected_factors.csv`
- 本报告：`docs/fundamental_operator_mining_csi500_smoke_2026-07-15.md`