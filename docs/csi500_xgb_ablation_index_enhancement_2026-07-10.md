# 中证500指增严格对照实验

日期：2026-07-10

## 结论

- 本实验不新增因子，只改变特征集合和组合构造方式，用来拆解公开因子、旧基本面因子、行业/风格约束分别贡献了什么。
- 测试期 IR 最高的场景是 `xgb_public_plus_fundamental_tight_constraint`，测试期超额 3.39%，IR 2.62。
- 公开因子-only + tight 约束测试期超额 3.02%；基本面-only + tight 约束测试期超额 3.48%；公开+基本面 + tight 约束测试期超额 3.39%。
- 公开因子-only 模型的测试期 RankIC 为 0.0571，但直接 score tilt 组合测试期超额为 -3.14%；加 tight 行业/风格约束后转为 +3.02%。这说明公开因子的预测排序有信息，但裸组合构造会把信息暴露成不稳定的行业/风格/拥挤风险。
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
| xgb_fundamental_only_score_tilt | full | fundamental_only | score_tilt | 3 | 25.54% | 1.15% | 2.40 | -2.69% | 56.91% | 73.79% | 10.64% | 0.85% |
| xgb_fundamental_only_score_tilt | test | fundamental_only | score_tilt | 3 | 4.74% | 1.25% | 2.57 | -0.72% | 57.18% | 73.68% | 10.80% | 0.85% |
| xgb_fundamental_only_score_tilt | ytd_2026 | fundamental_only | score_tilt | 3 | 3.76% | 1.53% | 5.10 | -0.36% | 59.66% | 100.00% | 10.86% | 0.87% |
| xgb_fundamental_only_tight_constraint | full | fundamental_only | industry_style_tight | 3 | 19.42% | 0.85% | 2.56 | -1.44% | 57.55% | 72.82% | 12.25% | 0.38% |
| xgb_fundamental_only_tight_constraint | test | fundamental_only | industry_style_tight | 3 | 3.48% | 0.98% | 2.45 | -0.83% | 54.97% | 73.68% | 12.40% | 0.48% |
| xgb_fundamental_only_tight_constraint | ytd_2026 | fundamental_only | industry_style_tight | 3 | 3.11% | 1.30% | 5.01 | -0.28% | 62.18% | 100.00% | 12.49% | 0.62% |
| xgb_public_only_score_tilt | full | public_only | score_tilt | 37 | 17.54% | 1.51% | 1.24 | -4.59% | 52.84% | 65.05% | 11.01% | 0.98% |
| xgb_public_only_score_tilt | test | public_only | score_tilt | 37 | -3.14% | 2.00% | -1.22 | -4.35% | 43.92% | 36.84% | 10.87% | 1.16% |
| xgb_public_only_score_tilt | ytd_2026 | public_only | score_tilt | 37 | -1.51% | 2.29% | -1.52 | -1.92% | 42.86% | 42.86% | 10.83% | 1.23% |
| xgb_public_only_tight_constraint | full | public_only | industry_style_tight | 37 | 16.48% | 0.81% | 2.31 | -1.17% | 55.98% | 71.84% | 12.50% | 0.39% |
| xgb_public_only_tight_constraint | test | public_only | industry_style_tight | 37 | 3.02% | 0.90% | 2.31 | -0.54% | 54.14% | 73.68% | 12.46% | 0.48% |
| xgb_public_only_tight_constraint | ytd_2026 | public_only | industry_style_tight | 37 | 2.52% | 1.23% | 4.29 | -0.34% | 58.82% | 85.71% | 12.48% | 0.62% |
| xgb_public_plus_fundamental_score_tilt | full | public_plus_fundamental | score_tilt | 40 | 19.88% | 1.50% | 1.40 | -4.63% | 53.73% | 68.93% | 11.01% | 0.98% |
| xgb_public_plus_fundamental_score_tilt | test | public_plus_fundamental | score_tilt | 40 | -3.03% | 1.98% | -1.20 | -4.23% | 45.03% | 36.84% | 10.85% | 1.16% |
| xgb_public_plus_fundamental_score_tilt | ytd_2026 | public_plus_fundamental | score_tilt | 40 | -1.50% | 2.32% | -1.49 | -2.03% | 42.86% | 42.86% | 10.86% | 1.24% |
| xgb_public_plus_fundamental_tight_constraint | full | public_plus_fundamental | industry_style_tight | 40 | 18.82% | 0.82% | 2.60 | -1.00% | 56.81% | 71.84% | 12.53% | 0.39% |
| xgb_public_plus_fundamental_tight_constraint | test | public_plus_fundamental | industry_style_tight | 40 | 3.39% | 0.89% | 2.62 | -0.54% | 55.25% | 73.68% | 12.44% | 0.48% |
| xgb_public_plus_fundamental_tight_constraint | ytd_2026 | public_plus_fundamental | industry_style_tight | 40 | 2.57% | 1.22% | 4.44 | -0.36% | 57.14% | 85.71% | 12.45% | 0.62% |

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_excess_nav.png)

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_test_excess_return.png)

![](../outputs/csi500_xgb_ablation_index_enhancement/ablation_ytd_2026_excess_return.png)

## 模型 RankIC

| period | train_rankic_mean | train_rankic_ir | train_rankic_positive_ratio | train_ic_mean | train_n_days | valid_rankic_mean | valid_rankic_ir | valid_rankic_positive_ratio | valid_ic_mean | valid_n_days | test_rankic_mean | test_rankic_ir | test_rankic_positive_ratio | test_ic_mean | test_n_days | ytd_2026_rankic_mean | ytd_2026_rankic_ir | ytd_2026_rankic_positive_ratio | ytd_2026_ic_mean | ytd_2026_n_days | feature_set | backend | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.08274937006470189 | 0.793309849329539 | 0.7946273830155979 | 0.09580354821801797 | 1154.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | fundamental_only | sklearn_hist_gradient_boosting | 3 |
| valid |  |  |  |  |  | 0.030665546879988854 | 0.2326352140682177 | 0.5723140495867769 | 0.027831340493934445 | 484.0 |  |  |  |  |  |  |  |  |  |  | fundamental_only | sklearn_hist_gradient_boosting | 3 |
| test |  |  |  |  |  |  |  |  |  |  | 0.029123663193702694 | 0.26361340780506703 | 0.6330532212885154 | 0.028220558353411317 | 357.0 |  |  |  |  |  | fundamental_only | sklearn_hist_gradient_boosting | 3 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.05316476444942132 | 0.5582330418529212 | 0.7807017543859649 | 0.05005165334335303 | 114.0 | fundamental_only | sklearn_hist_gradient_boosting | 3 |
| train | 0.14947043928228101 | 1.2290188983286654 | 0.897822445561139 | 0.1655844239564229 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | public_only | sklearn_hist_gradient_boosting | 37 |
| valid |  |  |  |  |  | 0.06323394861023948 | 0.44901343849299835 | 0.6632231404958677 | 0.08166617466627347 | 484.0 |  |  |  |  |  |  |  |  |  |  | public_only | sklearn_hist_gradient_boosting | 37 |
| test |  |  |  |  |  |  |  |  |  |  | 0.0571409623908715 | 0.34069178688105417 | 0.6162464985994398 | 0.0656806066755726 | 357.0 |  |  |  |  |  | public_only | sklearn_hist_gradient_boosting | 37 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.03390582144530478 | 0.2129059121087449 | 0.5263157894736842 | 0.03590755839764451 | 114.0 | public_only | sklearn_hist_gradient_boosting | 37 |
| train | 0.1525783416394414 | 1.2828465502926896 | 0.9120603015075377 | 0.16870879556808696 | 1194.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | public_plus_fundamental | sklearn_hist_gradient_boosting | 40 |
| valid |  |  |  |  |  | 0.06545832301049802 | 0.45336440602320977 | 0.6797520661157025 | 0.08301449954444336 | 484.0 |  |  |  |  |  |  |  |  |  |  | public_plus_fundamental | sklearn_hist_gradient_boosting | 40 |
| test |  |  |  |  |  |  |  |  |  |  | 0.059924055601594924 | 0.35467731278698345 | 0.6218487394957983 | 0.06733265595725747 | 357.0 |  |  |  |  |  | public_plus_fundamental | sklearn_hist_gradient_boosting | 40 |
| ytd_2026 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.03812146853310613 | 0.23270106566941562 | 0.5526315789473685 | 0.039441264326551796 | 114.0 | public_plus_fundamental | sklearn_hist_gradient_boosting | 40 |

## 批判性解读

- `score_tilt` 只做基准权重上的指数倾斜和个股上限，基本不控制行业/风格，是多因子选股更接近的形态。
- `industry_style_tight` 同时约束行业、风格和 active share，更接近导师要求的指数增强框架。
- 本实验仍使用 sklearn fallback 模型；若后续启用原生 xgboost，需要重跑同一张对照表，避免模型后端变化导致结论混杂。

## 输出文件

- 输出目录：`outputs/csi500_xgb_ablation_index_enhancement`
- 日度回测：`outputs/csi500_xgb_ablation_index_enhancement/ablation_daily_returns.csv`
- 方案摘要：`outputs/csi500_xgb_ablation_index_enhancement/ablation_scenario_summary.csv`
- 模型摘要：`outputs/csi500_xgb_ablation_index_enhancement/ablation_model_summary.csv`
- 调仓权重：`outputs/csi500_xgb_ablation_index_enhancement/ablation_weights.csv`
