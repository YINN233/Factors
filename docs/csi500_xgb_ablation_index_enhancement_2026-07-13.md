# 中证500指增严格对照实验

日期：2026-07-13

## 结论

- 本实验不新增因子，只改变特征集合和组合构造方式，用来拆解公开因子、旧基本面因子、行业/风格约束分别贡献了什么。
- 测试期 IR 最高的场景是 `xgb_public_plus_fundamental_tight_constraint`，测试期超额 3.32%，IR 2.50。
- 公开因子-only + tight 约束测试期超额 3.11%；基本面-only + tight 约束测试期超额 3.35%；公开+基本面 + tight 约束测试期超额 3.32%。
- 如果 score tilt 强而 tight 约束弱，说明信号更像选股排序；如果 tight 仍稳，才说明它能进入指数增强框架。本轮重点看后者。

## 特征集合

| feature_set | n_features | features |
| --- | --- | --- |
| fundamental_only | 3 | eps_bps_value_quality, quality_growth_hmean, industry_neutral_roe_value_pb |
| public_only | 37 | rl2_48_price_volume_decay_synergy, rl2_57_vol_adjusted_reversal, rl2_44_momentum_reversal, rl_03_pvt_covariance_reversal, rl2_54_turnover_volatility_inverse, rl2_64_residual_volatility, rl2_65_turnover_volatility_momentum, rl2_51_multidim_reversal, rl2_62_reversal_turnover_boost, rl_16_decay_volume_trend, rl_22_volume_divergence_momentum, rl_23_turnover_relative_reversal ... |
| public_plus_fundamental | 40 | rl2_48_price_volume_decay_synergy, rl2_57_vol_adjusted_reversal, rl2_44_momentum_reversal, rl_03_pvt_covariance_reversal, rl2_54_turnover_volatility_inverse, rl2_64_residual_volatility, rl2_65_turnover_volatility_momentum, rl2_51_multidim_reversal, rl2_62_reversal_turnover_boost, rl_16_decay_volume_trend, rl_22_volume_divergence_momentum, rl_23_turnover_relative_reversal ... |

## 场景表现

| scenario | period | feature_set | construction | n_features | excess_total_return | tracking_error | information_ratio | active_max_drawdown | daily_active_win_rate | monthly_active_win_rate | avg_active_share | avg_max_industry_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_core3_score_tilt | full | old_core3_score | score_tilt | 3 | 11.35% | 1.15% | 1.10 | -2.73% | 52.79% | 61.17% | 9.86% | 0.89% |
| old_core3_score_tilt | test | old_core3_score | score_tilt | 3 | 4.85% | 1.40% | 2.34 | -0.85% | 54.14% | 73.68% | 10.40% | 1.04% |
| old_core3_score_tilt | ytd_2026 | old_core3_score | score_tilt | 3 | 3.49% | 1.67% | 4.34 | -0.39% | 55.46% | 85.71% | 10.63% | 1.12% |
| old_core3_tight_constraint | full | old_core3_score | industry_style_tight | 3 | 9.35% | 0.88% | 1.25 | -1.79% | 53.68% | 58.25% | 12.14% | 0.38% |
| old_core3_tight_constraint | test | old_core3_score | industry_style_tight | 3 | 3.53% | 1.05% | 2.29 | -0.92% | 53.59% | 73.68% | 12.38% | 0.49% |
| old_core3_tight_constraint | ytd_2026 | old_core3_score | industry_style_tight | 3 | 2.97% | 1.34% | 4.62 | -0.29% | 57.14% | 100.00% | 12.69% | 0.63% |
| xgb_fundamental_only_score_tilt | full | fundamental_only | score_tilt | 3 | 18.87% | 1.14% | 1.82 | -2.69% | 54.71% | 66.02% | 10.64% | 0.85% |
| xgb_fundamental_only_score_tilt | test | fundamental_only | score_tilt | 3 | 4.50% | 1.28% | 2.39 | -0.75% | 56.08% | 68.42% | 10.92% | 0.91% |
| xgb_fundamental_only_score_tilt | ytd_2026 | fundamental_only | score_tilt | 3 | 3.59% | 1.57% | 4.76 | -0.40% | 58.82% | 85.71% | 10.95% | 0.95% |
| xgb_fundamental_only_tight_constraint | full | fundamental_only | industry_style_tight | 3 | 14.50% | 0.84% | 1.98 | -1.53% | 55.29% | 68.93% | 12.22% | 0.38% |
| xgb_fundamental_only_tight_constraint | test | fundamental_only | industry_style_tight | 3 | 3.35% | 0.98% | 2.35 | -0.95% | 54.70% | 73.68% | 12.37% | 0.48% |
| xgb_fundamental_only_tight_constraint | ytd_2026 | fundamental_only | industry_style_tight | 3 | 2.96% | 1.28% | 4.86 | -0.27% | 59.66% | 100.00% | 12.45% | 0.62% |
| xgb_public_only_score_tilt | full | public_only | score_tilt | 37 | 12.83% | 1.53% | 0.89 | -4.74% | 52.94% | 63.11% | 10.88% | 0.99% |
| xgb_public_only_score_tilt | test | public_only | score_tilt | 37 | -3.32% | 2.04% | -1.27 | -4.59% | 44.48% | 36.84% | 10.80% | 1.18% |
| xgb_public_only_score_tilt | ytd_2026 | public_only | score_tilt | 37 | -1.56% | 2.34% | -1.53 | -2.12% | 42.86% | 42.86% | 10.78% | 1.27% |
| xgb_public_only_tight_constraint | full | public_only | industry_style_tight | 37 | 13.48% | 0.81% | 1.92 | -1.10% | 55.39% | 64.08% | 12.47% | 0.39% |
| xgb_public_only_tight_constraint | test | public_only | industry_style_tight | 37 | 3.11% | 0.91% | 2.34 | -0.55% | 54.70% | 73.68% | 12.45% | 0.48% |
| xgb_public_only_tight_constraint | ytd_2026 | public_only | industry_style_tight | 37 | 2.60% | 1.25% | 4.36 | -0.34% | 58.82% | 85.71% | 12.50% | 0.61% |
| xgb_public_plus_fundamental_score_tilt | full | public_plus_fundamental | score_tilt | 40 | 13.84% | 1.51% | 0.97 | -4.73% | 53.19% | 64.08% | 10.85% | 0.98% |
| xgb_public_plus_fundamental_score_tilt | test | public_plus_fundamental | score_tilt | 40 | -3.11% | 2.03% | -1.20 | -4.38% | 45.03% | 36.84% | 10.78% | 1.17% |
| xgb_public_plus_fundamental_score_tilt | ytd_2026 | public_plus_fundamental | score_tilt | 40 | -1.48% | 2.37% | -1.45 | -2.09% | 42.86% | 42.86% | 10.74% | 1.24% |
| xgb_public_plus_fundamental_tight_constraint | full | public_plus_fundamental | industry_style_tight | 40 | 14.96% | 0.81% | 2.13 | -1.08% | 55.88% | 66.99% | 12.49% | 0.38% |
| xgb_public_plus_fundamental_tight_constraint | test | public_plus_fundamental | industry_style_tight | 40 | 3.32% | 0.91% | 2.50 | -0.61% | 55.80% | 68.42% | 12.46% | 0.48% |
| xgb_public_plus_fundamental_tight_constraint | ytd_2026 | public_plus_fundamental | industry_style_tight | 40 | 2.70% | 1.26% | 4.49 | -0.35% | 57.14% | 85.71% | 12.50% | 0.62% |

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_excess_nav.png)

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_test_excess_return.png)

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_ytd_2026_excess_return.png)

## 模型 RankIC

| period | train_rankic_mean | train_rankic_ir | train_rankic_positive_ratio | train_ic_mean | train_n_days | valid_rankic_mean | valid_rankic_ir | valid_rankic_positive_ratio | valid_ic_mean | valid_n_days | test_rankic_mean | test_rankic_ir | test_rankic_positive_ratio | test_ic_mean | test_n_days | ytd_2026_rankic_mean | ytd_2026_rankic_ir | ytd_2026_rankic_positive_ratio | ytd_2026_ic_mean | ytd_2026_n_days | feature_set | backend | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.05664965824440455 | 0.5510981820347998 | 0.7175043327556326 | 0.06195259520082453 | 1154.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | fundamental_only | xgboost | 3 |
| valid |  |  |  |  |  | 0.0370561751316248 | 0.25739017457421354 | 0.5764462809917356 | 0.03448906646232506 | 484.0 |  |  |  |  |  |  |  |  |  |  | fundamental_only | xgboost | 3 |
| test |  |  |  |  |  |  |  |  |  |  | 0.02986067744215855 | 0.2551103061667811 | 0.6330532212885154 | 0.03128672151300698 | 357.0 |  |  |  |  |  | fundamental_only | xgboost | 3 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.05300033094940831 | 0.5107138999336939 | 0.7631578947368421 | 0.05590680096577195 | 114.0 | fundamental_only | xgboost | 3 |
| train | 0.12190585974144809 | 0.9522944478964934 | 0.8467336683417085 | 0.13612168482765957 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | public_only | xgboost | 37 |
| valid |  |  |  |  |  | 0.06514881998641145 | 0.4507370646649448 | 0.6756198347107438 | 0.08283945097737282 | 484.0 |  |  |  |  |  |  |  |  |  |  | public_only | xgboost | 37 |
| test |  |  |  |  |  |  |  |  |  |  | 0.05598711306551751 | 0.3275758964619754 | 0.6022408963585434 | 0.06439307875344022 | 357.0 |  |  |  |  |  | public_only | xgboost | 37 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.034838957153219344 | 0.21062827477102816 | 0.5087719298245614 | 0.035836382058689734 | 114.0 | public_only | xgboost | 37 |
| train | 0.12452354369647647 | 0.9882531536668381 | 0.8467336683417085 | 0.1384282469365905 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | public_plus_fundamental | xgboost | 40 |
| valid |  |  |  |  |  | 0.06640816787440834 | 0.44915370888124995 | 0.6735537190082644 | 0.08364321300409171 | 484.0 |  |  |  |  |  |  |  |  |  |  | public_plus_fundamental | xgboost | 40 |
| test |  |  |  |  |  |  |  |  |  |  | 0.05933398318061689 | 0.34331045181769504 | 0.6106442577030813 | 0.06710667409505276 | 357.0 |  |  |  |  |  | public_plus_fundamental | xgboost | 40 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.0384480236793319 | 0.22700893200234912 | 0.543859649122807 | 0.04047969021074171 | 114.0 | public_plus_fundamental | xgboost | 40 |

## 批判性解读

- `score_tilt` 只做基准权重上的指数倾斜和个股上限，基本不控制行业/风格，是多因子选股更接近的形态。
- `industry_style_tight` 同时约束行业、风格和 active share，更接近导师要求的指数增强框架。
- 本实验使用原生 `XGBRegressor`，不是 sklearn fallback；模型后端变化会影响与旧报告的数值可比性。

## 输出文件

- 输出目录：`outputs/csi500_xgb_ablation_index_enhancement`
- 日度回测：`outputs/csi500_xgb_ablation_index_enhancement/ablation_daily_returns.csv`
- 方案摘要：`outputs/csi500_xgb_ablation_index_enhancement/ablation_scenario_summary.csv`
- 模型摘要：`outputs/csi500_xgb_ablation_index_enhancement/ablation_model_summary.csv`
- 调仓权重：`outputs/csi500_xgb_ablation_index_enhancement/ablation_weights.csv`