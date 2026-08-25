# 中证500 XGBoost 约束指数增强阶段报告

日期：2026-07-09

## 摘要结论

- 本轮合并 `融量量化公开因子.txt` 与 `融量公开因子2.txt`，元数据共 65 个条目；A 阶段可计算/近似计算 42 个，验证门最终放行 26 个，隔离 12 个，失败 4 个，另有 23 个缺字段条目留到 B 阶段。
- Pack2 单独看：共 39 个条目，A 阶段可计算 21 个，放行 15 个，隔离 5 个，失败 1 个，挂起 18 个。
- 当前公开因子筛选后进入模型候选的公开特征数为 26 个；不加风格约束的 XGB 方案在 2025-01-02 至 2026-07-03 测试期超额为 -0.03%，IR 为 -0.12，说明外源公开因子直接进组合仍不稳。
- 加行业/风格约束后，`xgb_industry_style_tight` 是测试期最稳的 XGB 约束方案；测试期超额约 3.33%，IR 约 2.49，主动最大回撤约 -0.52%。
- 旧的 `current_exp_score` 基本面组合在测试期仍很强，不能武断说 XGB 已经全面替代旧方案；更合理的结论是：公开价量因子 + 旧基本面因子在模型层有预测力，但必须通过行业/风格约束才能变成像样的指增组合。

## 方案表现

| scenario | period | excess_total_return | tracking_error | information_ratio | active_max_drawdown | daily_active_win_rate | monthly_active_win_rate | avg_active_share | avg_max_industry_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_exp_score | full | 11.35% | 1.15% | 1.10 | -2.73% | 52.79% | 61.17% | 9.86% | 0.89% |
| current_exp_score | test | 4.85% | 1.40% | 2.34 | -0.85% | 54.14% | 73.68% | 10.40% | 1.04% |
| current_exp_score | ytd_2026 | 3.49% | 1.67% | 4.34 | -0.39% | 55.46% | 85.71% | 10.63% | 1.12% |
| xgb_no_style_constraint | full | 32.93% | 1.47% | 2.29 | -2.71% | 56.91% | 71.84% | 19.03% | 0.44% |
| xgb_no_style_constraint | test | -0.03% | 1.79% | -0.12 | -2.33% | 48.62% | 47.37% | 18.70% | 0.50% |
| xgb_no_style_constraint | ytd_2026 | 0.43% | 1.95% | 0.36 | -0.74% | 53.78% | 57.14% | 18.43% | 0.60% |
| xgb_industry_style_tight | full | 20.04% | 0.82% | 2.75 | -0.79% | 57.70% | 73.79% | 12.56% | 0.38% |
| xgb_industry_style_tight | test | 3.33% | 0.92% | 2.49 | -0.52% | 53.31% | 73.68% | 12.46% | 0.48% |
| xgb_industry_style_tight | ytd_2026 | 2.69% | 1.25% | 4.53 | -0.35% | 55.46% | 85.71% | 12.50% | 0.61% |
| xgb_industry_style_mid | full | 26.70% | 0.95% | 3.06 | -0.82% | 58.43% | 79.61% | 17.60% | 0.42% |
| xgb_industry_style_mid | test | 3.42% | 0.99% | 2.36 | -0.50% | 53.59% | 73.68% | 17.40% | 0.50% |
| xgb_industry_style_mid | ytd_2026 | 2.68% | 1.32% | 4.23 | -0.50% | 57.14% | 85.71% | 17.17% | 0.61% |
| xgb_industry_style_loose | full | 33.78% | 1.12% | 3.20 | -1.03% | 58.82% | 79.61% | 22.73% | 0.53% |
| xgb_industry_style_loose | test | 3.21% | 1.09% | 1.99 | -0.69% | 52.49% | 78.95% | 22.45% | 0.57% |
| xgb_industry_style_loose | ytd_2026 | 2.48% | 1.43% | 3.61 | -0.69% | 55.46% | 85.71% | 21.94% | 0.66% |

![](../outputs/csi500_xgb_constrained_index_enhancement/scenario_excess_nav.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/rolling_60d_information_ratio.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/constraint_profile_test.png)

## 2026 年以来表现

| month | current_exp_score | xgb_industry_style_loose | xgb_industry_style_mid | xgb_industry_style_tight | xgb_no_style_constraint |
| --- | --- | --- | --- | --- | --- |
| 2026-01 | -0.06% | 0.16% | 0.15% | 0.14% | 0.05% |
| 2026-02 | 0.25% | -0.33% | -0.24% | -0.17% | -0.36% |
| 2026-03 | 0.24% | 0.20% | 0.12% | 0.07% | 0.45% |
| 2026-04 | 0.31% | 0.26% | 0.34% | 0.36% | -0.53% |
| 2026-05 | 1.19% | 1.23% | 1.12% | 1.00% | 1.15% |
| 2026-06 | 1.28% | 0.81% | 1.03% | 1.14% | -0.45% |
| 2026-07 | 0.24% | 0.13% | 0.13% | 0.13% | 0.13% |

![](../outputs/csi500_xgb_constrained_index_enhancement/ytd_monthly_active_return.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/active_return_distribution.png)

## 公开因子验证

公开文本中的 IC/Sharpe 没有直接采信。所有因子先经过本地中证500历史成分股、未来 5 日截面 Rank 标签的验证门，方向由训练集确定，再看验证期、测试期和 2026YTD。

| factor | validation_status | coverage | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- | --- |
| rl2_48_price_volume_decay_synergy | passed | 0.9617 | 0.0400 | 0.0500 | 0.0244 |  |
| rl2_57_vol_adjusted_reversal | passed | 0.9858 | 0.0392 | 0.0496 | 0.0100 |  |
| rl_03_pvt_covariance_reversal | passed | 0.9812 | 0.0550 | 0.0438 | 0.0382 |  |
| rl2_54_turnover_volatility_inverse | passed | 0.9934 | 0.0691 | 0.0418 | 0.0382 |  |
| rl2_64_residual_volatility | passed | 0.9915 | 0.0485 | 0.0413 | 0.0181 |  |
| rl2_65_turnover_volatility_momentum | passed | 0.9868 | 0.0590 | 0.0398 | 0.0217 |  |
| rl2_51_multidim_reversal | passed | 0.9915 | 0.0362 | 0.0367 | 0.0183 |  |
| rl2_62_reversal_turnover_boost | passed | 0.9953 | 0.0732 | 0.0366 | 0.0218 |  |
| rl_16_decay_volume_trend | passed | 0.9886 | 0.0299 | 0.0362 | -0.0098 |  |
| rl_22_volume_divergence_momentum | passed | 0.9767 | 0.0341 | 0.0359 | 0.0255 |  |
| rl_23_turnover_relative_reversal | passed | 0.9450 | 0.0368 | 0.0352 | -0.0035 |  |
| rl2_61_price_volume_divergence_cov | passed | 0.9858 | 0.0461 | 0.0346 | 0.0284 |  |

![](../outputs/csi500_xgb_constrained_index_enhancement/public_factor_status.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/public_factor_test_rankic.png)

未放行或需隔离的代表性因子如下。这些因子不能直接进入模型，除非未来补字段、重写表达式或重新通过验证门。

| factor | validation_status | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- |
| rl_02_projection_support_proxy | failed | -0.0087 | 0.0209 | 0.0026 | valid_rankic_nonpositive |
| rl_05_ema_midprice_divergence | quarantined | 0.0310 | -0.0123 | -0.0240 | test_rankic_negative,ytd_2026_weak |
| rl_08_variance_ratio_divergence | failed | -0.0259 | -0.0454 | -0.0306 | low_coverage,valid_rankic_nonpositive,test_rankic_negative,ytd_2026_weak |
| rl_10_cashflow_price_trend_proxy | quarantined | 0.0498 | 0.0065 | -0.0075 | ytd_2026_weak |
| rl_17_vol_trend_structure | quarantined | 0.0150 | 0.0205 | -0.0103 | ytd_2026_weak |
| rl_18_turnover_adjusted_price | failed | -0.0175 | 0.0208 | 0.0442 | valid_rankic_nonpositive |
| rl_20_price_exhaustion_reversal | quarantined | 0.0348 | 0.0014 | -0.0167 | ytd_2026_weak |
| rl_21_log_momentum_reversal_proxy | quarantined | 0.0222 | 0.0383 | -0.0060 | ytd_2026_weak |
| rl_24_price_reversal_30d | quarantined | 0.0198 | 0.0472 | -0.0087 | ytd_2026_weak |
| rl_25_high_open_decay | quarantined | 0.0545 | 0.0098 | -0.0115 | ytd_2026_weak |
| rl2_28_volatility_spread_proxy | quarantined | 0.0084 | 0.0023 | -0.0065 | ytd_2026_weak |
| rl2_29_volume_stable_close | failed | -0.0294 | 0.0222 | 0.0377 | valid_rankic_nonpositive |

## Pack2 补充公开因子

`融量公开因子2.txt` 的新增条目已纳入同一验证门。资金流、CNE5 风格和特殊复权字段相关条目没有强行代理，统一留到 B 阶段。

| factor | validation_status | coverage | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- | --- |
| rl2_48_price_volume_decay_synergy | passed | 0.9617 | 0.0400 | 0.0500 | 0.0244 |  |
| rl2_57_vol_adjusted_reversal | passed | 0.9858 | 0.0392 | 0.0496 | 0.0100 |  |
| rl2_54_turnover_volatility_inverse | passed | 0.9934 | 0.0691 | 0.0418 | 0.0382 |  |
| rl2_64_residual_volatility | passed | 0.9915 | 0.0485 | 0.0413 | 0.0181 |  |
| rl2_65_turnover_volatility_momentum | passed | 0.9868 | 0.0590 | 0.0398 | 0.0217 |  |
| rl2_51_multidim_reversal | passed | 0.9915 | 0.0362 | 0.0367 | 0.0183 |  |
| rl2_62_reversal_turnover_boost | passed | 0.9953 | 0.0732 | 0.0366 | 0.0218 |  |
| rl2_61_price_volume_divergence_cov | passed | 0.9858 | 0.0461 | 0.0346 | 0.0284 |  |
| rl2_32_reverse_volatility_covariance_proxy | passed | 0.9802 | 0.0252 | 0.0305 | 0.0056 |  |
| rl2_50_price_volume_volatility_inverse | passed | 0.9962 | 0.0446 | 0.0291 | 0.0148 |  |
| rl2_59_price_vwap_volume_volatility | passed | 0.9962 | 0.0447 | 0.0290 | 0.0126 |  |
| rl2_31_gap_momentum | passed | 0.9905 | 0.0292 | 0.0244 | 0.0021 |  |
| rl2_30_short_term_vol_adjusted_return | passed | 0.9934 | 0.0419 | 0.0219 | 0.0082 |  |
| rl2_58_liquidity_stability | passed | 0.9962 | 0.0707 | 0.0161 | -0.0091 |  |
| rl2_27_reverse_price_volume_rank | passed | 1.0000 | 0.0270 | 0.0150 | 0.0029 |  |
| rl2_56_ma_deviation_volume_weighted | quarantined | 0.9915 | 0.0173 | 0.0292 | -0.0082 | ytd_2026_weak |
| rl2_60_ma_filter_reversal | quarantined | 1.0000 | 0.0272 | 0.0170 | -0.0183 | ytd_2026_weak |
| rl2_28_volatility_spread_proxy | quarantined | 0.9709 | 0.0084 | 0.0023 | -0.0065 | ytd_2026_weak |

Pack2 中本轮挂起的条目如下：

| source_factor_id | factor_name | missing_columns | skip_reason |
| --- | --- | --- | --- |
| 33 | rl2_33_reinstatement_vol_ratio | REINSTATEMENT_CHG_60D | pending_b_stage: REINSTATEMENT_CHG_60D unavailable. |
| 34 | rl2_34_composite_momentum_flow | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 35 | rl2_35_flow_momentum | MAIN_IN_FLOW_20D_V2,SLARGE_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 and SLARGE_IN_FLOW_V2 unavailable. |
| 36 | rl2_36_momentum_flow_volatility | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 37 | rl2_37_flow_volatility_synergy | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 38 | rl2_38_main_superlarge_flow_synergy | MAIN_IN_FLOW_20D_V2,SLARGE_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 and SLARGE_IN_FLOW_V2 unavailable. |
| 39 | rl2_39_main_superlarge_flow_spread | MAIN_IN_FLOW_20D_V2,SLARGE_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 and SLARGE_IN_FLOW_V2 unavailable. |
| 40 | rl2_40_style_ir_spread | FACTOR_CNE5_BETA,FACTOR_CNE5_SIZE | pending_b_stage: CNE5 style fields unavailable. |
| 41 | rl2_41_flow_volatility_factor | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 42 | rl2_42_flow_linear_decay | MAIN_IN_FLOW_20D_V2,SLARGE_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 and SLARGE_IN_FLOW_V2 unavailable. |
| 43 | rl2_43_main_flow_momentum | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 44 | rl2_44_momentum_reversal | AF_CLOSE | pending_b_stage: AF_CLOSE-specific rank-change factor deferred from A-stage to keep scope focused. |
| 45 | rl2_45_flow_momentum_reinstatement | REINSTATEMENT_CHG_60D,MAIN_IN_FLOW_20D_V2 | pending_b_stage: REINSTATEMENT_CHG_60D and MAIN_IN_FLOW_20D_V2 unavailable. |
| 46 | rl2_46_main_flow_linear_decay | MAIN_IN_FLOW_20D_V2 | pending_b_stage: MAIN_IN_FLOW_20D_V2 unavailable. |
| 47 | rl2_47_price_flow_volatility_coupling | MAIN_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_V2 unavailable. |
| 49 | rl2_49_dual_volatility_flow_inverse | MAIN_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_V2 unavailable. |
| 52 | rl2_52_main_flow_peak_inverse | MAIN_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_V2 unavailable. |
| 53 | rl2_53_main_flow_stability | MAIN_IN_FLOW_V2 | pending_b_stage: MAIN_IN_FLOW_V2 unavailable. |

## XGBoost/替代模型

- 当前后端：`sklearn_hist_gradient_boosting`。如果本机未安装 xgboost，会自动 fallback 到 sklearn HistGradientBoosting。
- 当前特征重要性类型：`prediction_corr`。`prediction_corr` 表示 fallback 模型没有原生 gain，用特征与模型预测值的相关度作为替代贡献度，不应解释为 XGBoost split gain。

| period | train_rankic_mean | train_rankic_ir | train_rankic_positive_ratio | train_ic_mean | train_n_days | valid_rankic_mean | valid_rankic_ir | valid_rankic_positive_ratio | valid_ic_mean | valid_n_days | test_rankic_mean | test_rankic_ir | test_rankic_positive_ratio | test_ic_mean | test_n_days | ytd_2026_rankic_mean | ytd_2026_rankic_ir | ytd_2026_rankic_positive_ratio | ytd_2026_ic_mean | ytd_2026_n_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.1445804497477621 | 1.1853526013244848 | 0.8911222780569514 | 0.163013391682771 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| valid |  |  |  |  |  | 0.0776459742240119 | 0.5334715675858396 | 0.7210743801652892 | 0.0913794296361612 | 484.0 |  |  |  |  |  |  |  |  |  |  |
| test |  |  |  |  |  |  |  |  |  |  | 0.0621577768138678 | 0.3743398236220157 | 0.6302521008403361 | 0.0720014211202956 | 357.0 |  |  |  |  |  |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.0473798644324435 | 0.3078256068883087 | 0.5964912280701754 | 0.0535347566498451 | 114.0 |

| feature | importance | importance_type |
| --- | --- | --- |
| rl2_54_turnover_volatility_inverse | 0.7146 | prediction_corr |
| rl2_62_reversal_turnover_boost | 0.6992 | prediction_corr |
| rl_03_pvt_covariance_reversal | 0.6584 | prediction_corr |
| rl2_65_turnover_volatility_momentum | 0.6508 | prediction_corr |
| rl2_51_multidim_reversal | 0.6245 | prediction_corr |
| rl_22_volume_divergence_momentum | 0.5542 | prediction_corr |
| rl2_50_price_volume_volatility_inverse | 0.5329 | prediction_corr |
| rl_06_vol_compression_momentum | 0.5023 | prediction_corr |
| rl2_58_liquidity_stability | 0.4828 | prediction_corr |
| rl_23_turnover_relative_reversal | 0.4637 | prediction_corr |
| rl2_57_vol_adjusted_reversal | 0.4571 | prediction_corr |
| rl_16_decay_volume_trend | 0.4130 | prediction_corr |
| rl2_27_reverse_price_volume_rank | 0.3991 | prediction_corr |
| rl2_30_short_term_vol_adjusted_return | 0.3285 | prediction_corr |

![](../outputs/csi500_xgb_constrained_index_enhancement/xgb_feature_importance.png)

## 经济含义

- 通过验证的公开因子主要集中在价量拥挤、成交量背离、波动压缩后的反转、换手相对强弱、PVT 协方差反转，以及 Pack2 新增的换手率稳定、量价残差波动、量价效率和均线偏离反转。这类因子本质上在捕捉中证500成分股里的交易拥挤解除和短中期行为反转。
- 旧基本面因子 `eps_bps_value_quality`、`quality_growth_hmean`、`industry_neutral_roe_value_pb` 仍进入模型，提供价值/质量底座，避免纯价量模型在风格漂移时失效。
- 行业和风格约束的作用不是提高裸预测 IC，而是把 alpha 兑现方式限制在指数增强可接受的主动风险预算内；本轮最关键的证据是无约束 XGB 样本外为负，而约束后转正且回撤下降。

## 使用方式

1. 新外源因子统一按长表接入：`trade_date, ts_code, factor_name, factor_value, source, version, release_date`。
2. 先运行外源因子校验入口，状态保持 `pending`；只有通过覆盖率、验证期 RankIC、测试期 RankIC、2026YTD 稳定性和相关性筛选后，才能标记为 `passed` 并进入模型。
3. 主流水线命令：`venv/bin/python -m factors.reports.constrained_index_enhancement --start 20180101 --end 20260706`。
4. 报告命令：`venv/bin/python -m factors.reports.constrained_enhancement_report`。

## 批判性备注

1. 本轮 A 阶段没有使用真实主力/大单/机构资金流字段，相关公开因子均未混入主模型。
2. 多个公开因子是 proxy 或 partial 复现，不能等同于原始生产因子；报告结论只针对本地表达式。
3. 2026YTD 截止到 2026-07-03，样本仍短，需要后续滚动复检。
4. 当前组合优化默认使用快速投影式约束求解，适合研究流水线；若进入生产或正式复盘，应抽样用 `method='cvxpy'` 做精确约束交叉验证。

## 输出文件

- 输出目录：`outputs/csi500_xgb_constrained_index_enhancement`
- 日度回测：`outputs/csi500_xgb_constrained_index_enhancement/constrained_daily_returns.csv`
- 调仓权重：`outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv`
- 方案摘要：`outputs/csi500_xgb_constrained_index_enhancement/scenario_summary.csv`
- 因子验证：`outputs/csi500_xgb_constrained_index_enhancement/public_factor_validation_summary.csv`
- 模型预测：`outputs/csi500_xgb_constrained_index_enhancement/xgb_predictions.parquet`