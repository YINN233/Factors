# 中证500 XGBoost 约束指数增强阶段报告

日期：2026-07-13

## 摘要结论

- 本轮严格限定在两个融量公开因子文本里的 65 个条目内，没有新增第 66 个候选因子。补字段后元数据共 65 个，direct 27 个、proxy 38 个、skipped 0 个。
- 65 个因子全部完成本地计算和验证：通过 40 个，隔离 21 个，失败 4 个。相关性去重后入模公开因子 37 个，其中 direct 16 个、proxy 21 个。
- Pack2 单独看：共 39 个条目，补字段后可计算 39 个，放行 24 个，隔离 13 个，失败 2 个，skipped 0 个。
- 当前公开因子筛选后进入模型候选的公开特征数为 37 个；不加风格约束的 XGB 方案在 2025-01-02 至 2026-07-03 测试期超额为 -0.37%，IR 为 -0.25，说明外源公开因子直接进组合仍不稳。
- 加行业/风格约束后，`xgb_industry_style_tight` 是测试期最稳的 XGB 约束方案；测试期超额约 3.32%，IR 约 2.50，主动最大回撤约 -0.61%。
- 旧的 `current_exp_score` 基本面组合在测试期仍很强，不能武断说 XGB 已经全面替代旧方案；更合理的结论是：公开价量因子 + 旧基本面因子在模型层有预测力，但必须通过行业/风格约束才能变成像样的指增组合。

## 65 个公开因子缺字段补齐

补字段只服务于这 65 个公开因子。`AF_CLOSE/AF_HIGH/AF_LOW/AF_OPEN` 映射到本地复权价；`AF_VWAP`、`REINSTATEMENT_CHG_60D`、主力/大单资金流、CNE5 风格字段均按设计文档里的本地 proxy 口径补齐。proxy 不等于原始生产字段，报告结论只对本地可复现口径成立。

| availability | count | passed | quarantined | failed | selected | selected_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| proxy | 38 | 22 | 14 | 2 | 21 | 55.26% |
| direct | 27 | 18 | 7 | 2 | 16 | 59.26% |
| total | 65 | 40 | 21 | 4 | 37 | 56.92% |

原先缺字段或部分缺字段的重点恢复条目如下。`selected=yes` 表示通过验证且经过相关性去重后进入本轮模型特征。

| source_factor_id | factor_name | availability | validation_status | selected | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02 | rl_02_projection_support_proxy | proxy | passed | yes | 0.0040 | 0.0015 | 0.0088 |  |
| 11 | rl_11_large_outflow_reversal | proxy | passed | yes | 0.0533 | 0.0285 | 0.0092 |  |
| 12 | rl_12_industry_mainflow_profit_quality | proxy | quarantined | no | 0.0110 | -0.0022 | -0.0025 | test_rankic_negative |
| 13 | rl_13_fundflow_cross_section_pct | proxy | passed | yes | 0.0095 | 0.0107 | 0.0104 |  |
| 14 | rl_14_large_outflow_reversal_dup | proxy | passed | yes | 0.0692 | 0.0144 | 0.0034 |  |
| 15 | rl_15_nonlinear_price_volume_proxy | proxy | passed | yes | 0.0577 | 0.0165 | 0.0095 |  |
| 26 | rl_26_moneyflow_drawdown | proxy | passed | yes | 0.0070 | 0.0091 | 0.0068 |  |
| 33 | rl2_33_reinstatement_vol_ratio | proxy | failed | no | -0.0031 | 0.0123 | -0.0211 | valid_rankic_nonpositive,ytd_2026_weak |
| 34 | rl2_34_composite_momentum_flow | proxy | quarantined | no | 0.0158 | -0.0029 | 0.0018 | test_rankic_negative |
| 35 | rl2_35_flow_momentum | proxy | passed | yes | 0.0132 | 0.0017 | 0.0070 |  |
| 36 | rl2_36_momentum_flow_volatility | proxy | passed | no | 0.0150 | 0.0224 | 0.0312 |  |
| 37 | rl2_37_flow_volatility_synergy | proxy | passed | yes | 0.0190 | 0.0002 | 0.0055 |  |
| 38 | rl2_38_main_superlarge_flow_synergy | proxy | quarantined | no | 0.0081 | -0.0195 | -0.0013 | test_rankic_negative |
| 39 | rl2_39_main_superlarge_flow_spread | proxy | quarantined | no | 0.0155 | -0.0036 | -0.0012 | test_rankic_negative |
| 40 | rl2_40_style_ir_spread | proxy | passed | yes | 0.0020 | 0.0183 | 0.0534 |  |
| 41 | rl2_41_flow_volatility_factor | proxy | passed | yes | 0.0165 | 0.0234 | 0.0310 |  |
| 42 | rl2_42_flow_linear_decay | proxy | quarantined | no | 0.0182 | -0.0006 | 0.0058 | test_rankic_negative |
| 43 | rl2_43_main_flow_momentum | proxy | quarantined | no | 0.0025 | -0.0083 | -0.0057 | test_rankic_negative,ytd_2026_weak |
| 44 | rl2_44_momentum_reversal | direct | passed | yes | 0.0240 | 0.0441 | 0.0037 |  |
| 45 | rl2_45_flow_momentum_reinstatement | proxy | passed | yes | 0.0211 | 0.0015 | 0.0068 |  |
| 46 | rl2_46_main_flow_linear_decay | proxy | quarantined | no | 0.0157 | -0.0027 | 0.0023 | test_rankic_negative |
| 47 | rl2_47_price_flow_volatility_coupling | proxy | quarantined | no | 0.0470 | 0.0318 | -0.0010 | ytd_2026_weak |
| 49 | rl2_49_dual_volatility_flow_inverse | proxy | quarantined | no | 0.0697 | 0.0151 | -0.0107 | ytd_2026_weak |
| 52 | rl2_52_main_flow_peak_inverse | proxy | passed | yes | 0.0619 | 0.0291 | -0.0012 |  |
| 53 | rl2_53_main_flow_stability | proxy | passed | yes | 0.0694 | 0.0241 | -0.0015 |  |

## 方案表现

| scenario | period | excess_total_return | tracking_error | information_ratio | active_max_drawdown | daily_active_win_rate | monthly_active_win_rate | avg_active_share | avg_max_industry_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_exp_score | full | 11.35% | 1.15% | 1.10 | -2.73% | 52.79% | 61.17% | 9.86% | 0.89% |
| current_exp_score | test | 4.85% | 1.40% | 2.34 | -0.85% | 54.14% | 73.68% | 10.40% | 1.04% |
| current_exp_score | ytd_2026 | 3.49% | 1.67% | 4.34 | -0.39% | 55.46% | 85.71% | 10.63% | 1.12% |
| xgb_no_style_constraint | full | 25.45% | 1.48% | 1.80 | -2.47% | 55.88% | 68.93% | 18.99% | 0.44% |
| xgb_no_style_constraint | test | -0.37% | 1.82% | -0.25 | -2.21% | 48.62% | 47.37% | 18.78% | 0.50% |
| xgb_no_style_constraint | ytd_2026 | 0.04% | 2.05% | -0.07 | -0.89% | 48.74% | 57.14% | 18.48% | 0.60% |
| xgb_industry_style_tight | full | 14.96% | 0.81% | 2.13 | -1.08% | 55.88% | 66.99% | 12.49% | 0.38% |
| xgb_industry_style_tight | test | 3.32% | 0.91% | 2.50 | -0.61% | 55.80% | 68.42% | 12.46% | 0.48% |
| xgb_industry_style_tight | ytd_2026 | 2.70% | 1.26% | 4.49 | -0.35% | 57.14% | 85.71% | 12.50% | 0.62% |
| xgb_industry_style_mid | full | 18.68% | 0.93% | 2.28 | -1.26% | 56.57% | 70.87% | 17.47% | 0.42% |
| xgb_industry_style_mid | test | 3.36% | 0.97% | 2.34 | -0.56% | 55.25% | 78.95% | 17.37% | 0.50% |
| xgb_industry_style_mid | ytd_2026 | 2.66% | 1.33% | 4.15 | -0.49% | 57.14% | 85.71% | 17.12% | 0.62% |
| xgb_industry_style_loose | full | 22.78% | 1.08% | 2.34 | -1.30% | 56.13% | 76.70% | 22.56% | 0.51% |
| xgb_industry_style_loose | test | 3.15% | 1.07% | 1.97 | -0.67% | 51.93% | 73.68% | 22.41% | 0.57% |
| xgb_industry_style_loose | ytd_2026 | 2.45% | 1.45% | 3.49 | -0.67% | 59.66% | 85.71% | 21.86% | 0.67% |

![](../outputs/csi500_xgb_constrained_index_enhancement/scenario_excess_nav.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/rolling_60d_information_ratio.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/constraint_profile_test.png)

## 2026 年以来表现

| month | current_exp_score | xgb_industry_style_loose | xgb_industry_style_mid | xgb_industry_style_tight | xgb_no_style_constraint |
| --- | --- | --- | --- | --- | --- |
| 2026-01 | -0.06% | 0.33% | 0.29% | 0.24% | 0.13% |
| 2026-02 | 0.25% | -0.24% | -0.16% | -0.11% | -0.34% |
| 2026-03 | 0.24% | 0.35% | 0.23% | 0.13% | 0.53% |
| 2026-04 | 0.31% | 0.14% | 0.25% | 0.29% | -0.73% |
| 2026-05 | 1.19% | 0.70% | 0.71% | 0.71% | 0.86% |
| 2026-06 | 1.28% | 1.03% | 1.18% | 1.26% | -0.53% |
| 2026-07 | 0.24% | 0.13% | 0.13% | 0.13% | 0.13% |

![](../outputs/csi500_xgb_constrained_index_enhancement/ytd_monthly_active_return.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/active_return_distribution.png)

## 公开因子验证

公开文本中的 IC/Sharpe 没有直接采信。所有因子先经过本地中证500历史成分股、未来 5 日截面 Rank 标签的验证门，方向由训练集确定，再看验证期、测试期和 2026YTD。

| factor | validation_status | coverage | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- | --- |
| rl2_48_price_volume_decay_synergy | passed | 0.9617 | 0.0400 | 0.0500 | 0.0244 |  |
| rl2_57_vol_adjusted_reversal | passed | 0.9858 | 0.0392 | 0.0496 | 0.0100 |  |
| rl2_44_momentum_reversal | passed | 0.9812 | 0.0240 | 0.0441 | 0.0037 |  |
| rl_03_pvt_covariance_reversal | passed | 0.9812 | 0.0550 | 0.0438 | 0.0382 |  |
| rl2_54_turnover_volatility_inverse | passed | 0.9934 | 0.0691 | 0.0418 | 0.0382 |  |
| rl2_64_residual_volatility | passed | 0.9915 | 0.0485 | 0.0413 | 0.0181 |  |
| rl2_65_turnover_volatility_momentum | passed | 0.9868 | 0.0590 | 0.0398 | 0.0217 |  |
| rl2_51_multidim_reversal | passed | 0.9915 | 0.0362 | 0.0367 | 0.0183 |  |
| rl2_62_reversal_turnover_boost | passed | 0.9953 | 0.0732 | 0.0366 | 0.0218 |  |
| rl_16_decay_volume_trend | passed | 0.9886 | 0.0299 | 0.0362 | -0.0098 |  |
| rl_22_volume_divergence_momentum | passed | 0.9767 | 0.0341 | 0.0359 | 0.0255 |  |
| rl_23_turnover_relative_reversal | passed | 0.9450 | 0.0368 | 0.0352 | -0.0035 |  |

![](../outputs/csi500_xgb_constrained_index_enhancement/public_factor_status.png)

![](../outputs/csi500_xgb_constrained_index_enhancement/public_factor_test_rankic.png)

未放行或需隔离的因子如下。这些因子本轮不进入模型，除非未来重新通过同一验证门。

| source_factor_id | factor_name | availability | validation_status | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean | validation_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | rl_05_ema_midprice_divergence | proxy | quarantined | 0.0310 | -0.0123 | -0.0240 | test_rankic_negative,ytd_2026_weak |
| 8 | rl_08_variance_ratio_divergence | proxy | failed | -0.0259 | -0.0454 | -0.0306 | low_coverage,valid_rankic_nonpositive,test_rankic_negative,ytd_2026_weak |
| 10 | rl_10_cashflow_price_trend_proxy | proxy | quarantined | 0.0498 | 0.0065 | -0.0075 | ytd_2026_weak |
| 12 | rl_12_industry_mainflow_profit_quality | proxy | quarantined | 0.0110 | -0.0022 | -0.0025 | test_rankic_negative |
| 17 | rl_17_vol_trend_structure | direct | quarantined | 0.0150 | 0.0205 | -0.0103 | ytd_2026_weak |
| 18 | rl_18_turnover_adjusted_price | direct | failed | -0.0175 | 0.0208 | 0.0442 | valid_rankic_nonpositive |
| 20 | rl_20_price_exhaustion_reversal | proxy | quarantined | 0.0348 | 0.0014 | -0.0167 | ytd_2026_weak |
| 21 | rl_21_log_momentum_reversal_proxy | proxy | quarantined | 0.0222 | 0.0383 | -0.0060 | ytd_2026_weak |
| 24 | rl_24_price_reversal_30d | direct | quarantined | 0.0198 | 0.0472 | -0.0087 | ytd_2026_weak |
| 25 | rl_25_high_open_decay | direct | quarantined | 0.0545 | 0.0098 | -0.0115 | ytd_2026_weak |
| 28 | rl2_28_volatility_spread_proxy | proxy | quarantined | 0.0084 | 0.0023 | -0.0065 | ytd_2026_weak |
| 29 | rl2_29_volume_stable_close | direct | failed | -0.0294 | 0.0222 | 0.0377 | valid_rankic_nonpositive |
| 33 | rl2_33_reinstatement_vol_ratio | proxy | failed | -0.0031 | 0.0123 | -0.0211 | valid_rankic_nonpositive,ytd_2026_weak |
| 34 | rl2_34_composite_momentum_flow | proxy | quarantined | 0.0158 | -0.0029 | 0.0018 | test_rankic_negative |
| 38 | rl2_38_main_superlarge_flow_synergy | proxy | quarantined | 0.0081 | -0.0195 | -0.0013 | test_rankic_negative |
| 39 | rl2_39_main_superlarge_flow_spread | proxy | quarantined | 0.0155 | -0.0036 | -0.0012 | test_rankic_negative |
| 42 | rl2_42_flow_linear_decay | proxy | quarantined | 0.0182 | -0.0006 | 0.0058 | test_rankic_negative |
| 43 | rl2_43_main_flow_momentum | proxy | quarantined | 0.0025 | -0.0083 | -0.0057 | test_rankic_negative,ytd_2026_weak |
| 46 | rl2_46_main_flow_linear_decay | proxy | quarantined | 0.0157 | -0.0027 | 0.0023 | test_rankic_negative |
| 47 | rl2_47_price_flow_volatility_coupling | proxy | quarantined | 0.0470 | 0.0318 | -0.0010 | ytd_2026_weak |
| 49 | rl2_49_dual_volatility_flow_inverse | proxy | quarantined | 0.0697 | 0.0151 | -0.0107 | ytd_2026_weak |
| 55 | rl2_55_vol_turnover_coupling | direct | quarantined | 0.0457 | -0.0038 | -0.0306 | test_rankic_negative,ytd_2026_weak |
| 56 | rl2_56_ma_deviation_volume_weighted | direct | quarantined | 0.0173 | 0.0292 | -0.0082 | ytd_2026_weak |
| 60 | rl2_60_ma_filter_reversal | direct | quarantined | 0.0272 | 0.0170 | -0.0183 | ytd_2026_weak |
| 63 | rl2_63_price_volume_efficiency | direct | quarantined | 0.0568 | -0.0090 | -0.0178 | test_rankic_negative,ytd_2026_weak |

通过验证并进入模型的公开因子如下。顺序即相关性去重后的入模顺序。

| source_factor_id | factor_name | availability | validation_status | valid_rankic_mean | test_rankic_mean | ytd_2026_rankic_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | rl2_48_price_volume_decay_synergy | proxy | passed | 0.0400 | 0.0500 | 0.0244 |
| 57 | rl2_57_vol_adjusted_reversal | direct | passed | 0.0392 | 0.0496 | 0.0100 |
| 44 | rl2_44_momentum_reversal | direct | passed | 0.0240 | 0.0441 | 0.0037 |
| 3 | rl_03_pvt_covariance_reversal | proxy | passed | 0.0550 | 0.0438 | 0.0382 |
| 54 | rl2_54_turnover_volatility_inverse | direct | passed | 0.0691 | 0.0418 | 0.0382 |
| 64 | rl2_64_residual_volatility | proxy | passed | 0.0485 | 0.0413 | 0.0181 |
| 65 | rl2_65_turnover_volatility_momentum | direct | passed | 0.0590 | 0.0398 | 0.0217 |
| 51 | rl2_51_multidim_reversal | direct | passed | 0.0362 | 0.0367 | 0.0183 |
| 62 | rl2_62_reversal_turnover_boost | direct | passed | 0.0732 | 0.0366 | 0.0218 |
| 16 | rl_16_decay_volume_trend | direct | passed | 0.0299 | 0.0362 | -0.0098 |
| 22 | rl_22_volume_divergence_momentum | direct | passed | 0.0341 | 0.0359 | 0.0255 |
| 23 | rl_23_turnover_relative_reversal | direct | passed | 0.0368 | 0.0352 | -0.0035 |
| 61 | rl2_61_price_volume_divergence_cov | direct | passed | 0.0461 | 0.0346 | 0.0284 |
| 19 | rl_19_poly_volume_price_reversal | proxy | passed | 0.0171 | 0.0305 | -0.0074 |
| 32 | rl2_32_reverse_volatility_covariance_proxy | proxy | passed | 0.0252 | 0.0305 | 0.0056 |
| 52 | rl2_52_main_flow_peak_inverse | proxy | passed | 0.0619 | 0.0291 | -0.0012 |
| 50 | rl2_50_price_volume_volatility_inverse | direct | passed | 0.0446 | 0.0291 | 0.0148 |
| 11 | rl_11_large_outflow_reversal | proxy | passed | 0.0533 | 0.0285 | 0.0092 |
| 9 | rl_09_filtered_momentum_proxy | proxy | passed | 0.0057 | 0.0275 | -0.0008 |
| 31 | rl2_31_gap_momentum | direct | passed | 0.0292 | 0.0244 | 0.0021 |
| 53 | rl2_53_main_flow_stability | proxy | passed | 0.0694 | 0.0241 | -0.0015 |
| 41 | rl2_41_flow_volatility_factor | proxy | passed | 0.0165 | 0.0234 | 0.0310 |
| 30 | rl2_30_short_term_vol_adjusted_return | direct | passed | 0.0419 | 0.0219 | 0.0082 |
| 7 | rl_07_log_volume_price_trend | direct | passed | 0.0236 | 0.0219 | 0.0069 |
| 6 | rl_06_vol_compression_momentum | direct | passed | 0.0478 | 0.0183 | 0.0156 |
| 40 | rl2_40_style_ir_spread | proxy | passed | 0.0020 | 0.0183 | 0.0534 |
| 15 | rl_15_nonlinear_price_volume_proxy | proxy | passed | 0.0577 | 0.0165 | 0.0095 |
| 27 | rl2_27_reverse_price_volume_rank | direct | passed | 0.0270 | 0.0150 | 0.0029 |
| 14 | rl_14_large_outflow_reversal_dup | proxy | passed | 0.0692 | 0.0144 | 0.0034 |
| 4 | rl_04_valuation_price_resid_proxy | proxy | passed | 0.0080 | 0.0132 | 0.0096 |
| 13 | rl_13_fundflow_cross_section_pct | proxy | passed | 0.0095 | 0.0107 | 0.0104 |
| 26 | rl_26_moneyflow_drawdown | proxy | passed | 0.0070 | 0.0091 | 0.0068 |
| 1 | rl_01_tail_risk_reversal | proxy | passed | 0.0017 | 0.0038 | -0.0073 |
| 35 | rl2_35_flow_momentum | proxy | passed | 0.0132 | 0.0017 | 0.0070 |
| 45 | rl2_45_flow_momentum_reinstatement | proxy | passed | 0.0211 | 0.0015 | 0.0068 |
| 2 | rl_02_projection_support_proxy | proxy | passed | 0.0040 | 0.0015 | 0.0088 |
| 37 | rl2_37_flow_volatility_synergy | proxy | passed | 0.0190 | 0.0002 | 0.0055 |

## XGBoost/替代模型

- 当前后端：`xgboost`。本次结果使用原生 `XGBRegressor`，不再是 sklearn fallback。
- 当前特征重要性类型：`xgboost_gain`。它反映树模型分裂增益贡献，只能作为模型解释线索，不能单独等同于因子有效性。

| period | train_rankic_mean | train_rankic_ir | train_rankic_positive_ratio | train_ic_mean | train_n_days | valid_rankic_mean | valid_rankic_ir | valid_rankic_positive_ratio | valid_ic_mean | valid_n_days | test_rankic_mean | test_rankic_ir | test_rankic_positive_ratio | test_ic_mean | test_n_days | ytd_2026_rankic_mean | ytd_2026_rankic_ir | ytd_2026_rankic_positive_ratio | ytd_2026_ic_mean | ytd_2026_n_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.1245235436964764 | 0.988253153666838 | 0.8467336683417085 | 0.1384282469365905 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| valid |  |  |  |  |  | 0.0664081678744083 | 0.4491537088812499 | 0.6735537190082644 | 0.0836432130040917 | 484.0 |  |  |  |  |  |  |  |  |  |  |
| test |  |  |  |  |  |  |  |  |  |  | 0.0593339831806168 | 0.343310451817695 | 0.6106442577030813 | 0.0671066740950527 | 357.0 |  |  |  |  |  |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.0384480236793319 | 0.2270089320023491 | 0.543859649122807 | 0.0404796902107417 | 114.0 |

| feature | importance | importance_type |
| --- | --- | --- |
| rl2_54_turnover_volatility_inverse | 27.9536 | xgboost_gain |
| rl2_62_reversal_turnover_boost | 18.1816 | xgboost_gain |
| rl_03_pvt_covariance_reversal | 15.8656 | xgboost_gain |
| rl2_48_price_volume_decay_synergy | 6.3214 | xgboost_gain |
| rl2_61_price_volume_divergence_cov | 5.7425 | xgboost_gain |
| rl_09_filtered_momentum_proxy | 5.7360 | xgboost_gain |
| rl2_45_flow_momentum_reinstatement | 4.8655 | xgboost_gain |
| rl2_64_residual_volatility | 4.7480 | xgboost_gain |
| rl2_35_flow_momentum | 4.4254 | xgboost_gain |
| rl2_37_flow_volatility_synergy | 4.3242 | xgboost_gain |
| rl_23_turnover_relative_reversal | 4.2454 | xgboost_gain |
| rl_13_fundflow_cross_section_pct | 4.2369 | xgboost_gain |
| rl_16_decay_volume_trend | 4.0502 | xgboost_gain |
| quality_growth_hmean | 3.9965 | xgboost_gain |

![](../outputs/csi500_xgb_constrained_index_enhancement/xgb_feature_importance.png)

## 经济含义

- 通过验证的公开因子主要集中在价量拥挤、成交量背离、波动压缩后的反转、换手相对强弱、PVT 协方差反转，以及 Pack2 新增的换手率稳定、量价残差波动和资金流稳定类 proxy。这类因子本质上在捕捉中证500成分股里的交易拥挤解除和短中期行为反转。
- 资金流 proxy 并非全部有效：部分主力/超大单流入衰减类因子在 2025 以来或 2026YTD 走弱，被隔离；通过验证的资金流因子更偏向资金流极值、波动和稳定性，而不是简单净流入追涨。
- 旧基本面因子 `eps_bps_value_quality`、`quality_growth_hmean`、`industry_neutral_roe_value_pb` 仍进入模型，提供价值/质量底座，避免纯价量模型在风格漂移时失效。
- 行业和风格约束的作用不是提高裸预测 IC，而是把 alpha 兑现方式限制在指数增强可接受的主动风险预算内；本轮最关键的证据是无约束 XGB 样本外为负，而约束后转正且回撤下降。

## 使用方式

1. 新外源因子统一按长表接入：`trade_date, ts_code, factor_name, factor_value, source, version, release_date`。
2. 先运行外源因子校验入口，状态保持 `pending`；只有通过覆盖率、验证期 RankIC、测试期 RankIC、2026YTD 稳定性和相关性筛选后，才能标记为 `passed` 并进入模型。
3. 主流水线命令：`venv/bin/python -m factors.reports.constrained_index_enhancement --start 20180101 --end 20260706`。
4. 报告命令：`venv/bin/python -m factors.reports.constrained_enhancement_report`。

## 批判性备注

1. 38 个 proxy 因子不能等同于融量原始生产字段；尤其是资金流和 CNE5 风格字段，只能解释为本地 Tushare/风格暴露代理。
2. 通过验证不代表长期有效，隔离和失败列表需要保留；本轮 25 个因子没有进入模型，说明公开因子存在明显失效和口径不匹配。
3. 2026YTD 截止到 2026-07-03，样本仍短，需要后续滚动复检。
4. 当前模型后端是原生 xgboost；特征重要性采用 `xgboost_gain` 时只能辅助解释模型使用了什么，最终仍以样本外 RankIC 和约束组合回测为准。
5. 当前组合优化默认使用快速投影式约束求解，适合研究流水线；若进入生产或正式复盘，应抽样用 `method='cvxpy'` 做精确约束交叉验证。
6. 本轮修正了公开因子相关性去重的性能问题：先按日截面 rank，再一次性计算相关矩阵，避免对全样本重复 Spearman 排序；不改变去重阈值和入模规则。

## 输出文件

- 输出目录：`outputs/csi500_xgb_constrained_index_enhancement`
- 日度回测：`outputs/csi500_xgb_constrained_index_enhancement/constrained_daily_returns.csv`
- 调仓权重：`outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv`
- 方案摘要：`outputs/csi500_xgb_constrained_index_enhancement/scenario_summary.csv`
- 因子验证：`outputs/csi500_xgb_constrained_index_enhancement/public_factor_validation_summary.csv`
- 模型预测：`outputs/csi500_xgb_constrained_index_enhancement/xgb_predictions.parquet`