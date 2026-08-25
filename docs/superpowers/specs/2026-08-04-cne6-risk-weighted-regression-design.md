# CNE6 风险加权横截面回归实验设计

## 背景

前一轮已经确认，CNE6-style 风险模型的正式横截面回归平均 R2 约为 0.3693；把收益口径从下一日收益换成当天收益后，平均 R2 也只是提高到约 0.3790，没有接近导师提到的 0.5。动态风险校验模块能校准预测风险的偏差，但它不改变每日横截面回归本身，所以不会直接改变回归 R2。

这次实验单独检查另一个可能原因：回归拟合权重。正式复现里使用 `sqrt(total_mv)` 作为 WLS 权重；如果改用低特异风险股票更高权重，模型在“低噪声股票”上的解释力可能明显提高。但这个提高不一定代表全市场横截面解释力真的提高，因为高噪声股票被大幅降权以后，R2 指标本身也变了。

## 实验边界

这次只新增诊断实验，不替换原始 CNE6 输出：

- 不覆盖 `outputs/cne6_reproduction/factor_returns.csv`。
- 不覆盖 `outputs/cne6_reproduction/regression_diagnostics.csv`。
- 不覆盖 `outputs/cne6_reproduction/specific_returns.parquet`。
- 所有结果单独写到 `outputs/cne6_risk_weighted_regression/`。

报告里会把“自权重 R2”和“交叉评价 R2”分开，不用单一 R2 直接宣称模型达到 0.5。

## 输入数据

使用现有文件：

- `data/processed/cne6_csi500_daily_panel.parquet`
- `outputs/cne6_reproduction/style_exposures.parquet`
- `outputs/cne6_reproduction/specific_risk.parquet`

特异风险权重只使用 `t-3` 时点已经存在的 `specific_risk_60/120/252`。例如目标日是 `t`，则回归权重来自 `t-3` 的滚动特异风险，避免使用未来信息。

## 回归口径

默认比较两个收益口径：

1. `same_day`
   - 用当天暴露解释当天收益。
   - 更接近风险模型的事后收益分解。
   - 这是和导师 0.5 参考线更相关的主口径。

2. `forward_1d`
   - 用当天暴露解释下一日收益。
   - 更偏预测口径。
   - 用来和原始正式输出保持联系。

行业哑变量、国家因子、风格因子的构造沿用现有 `cne6_regression.py`，不在这次实验里改变暴露定义。

## 拟合权重

拟合权重包括：

|fit_weight|含义|
|---|---|
|`sqrt_mv`|正式复现基线，权重为 `sqrt(total_mv)`。|
|`equal`|等权回归，观察不强调大市值股票时的解释力。|
|`sqrt_mv_x_low_specific_rank_60_t3`|在 `sqrt(total_mv)` 基础上，用 `t-3` 的 60 日低特异风险排名做 0.75-1.25 倍温和调整。|
|`sqrt_mv_x_low_specific_rank_120_t3`|在 `sqrt(total_mv)` 基础上，用 `t-3` 的 120 日低特异风险排名做 0.75-1.25 倍温和调整。|
|`sqrt_mv_x_low_specific_rank_252_t3`|在 `sqrt(total_mv)` 基础上，用 `t-3` 的 252 日低特异风险排名做 0.75-1.25 倍温和调整。|
|`inv_specific_60_t3`|使用 `t-3` 的 60 日特异风险倒方差，低特异风险股票权重更高。|
|`inv_specific_120_t3`|使用 `t-3` 的 120 日特异风险倒方差。|
|`inv_specific_252_t3`|使用 `t-3` 的 252 日特异风险倒方差。|
|`inv_specific_vol_252_t3`|使用 `t-3` 的 252 日特异风险倒数，比倒方差更温和。|
|`sqrt_mv_div_specific_vol_252_t3`|用 `sqrt(total_mv) / specific_risk_252(t-3)`，同时保留市值权重和风险降权。|

直接倒数和倒方差类权重会做 1%/99% 分位数截尾，避免极少数低风险股票主导回归。rank 类权重本身限制在 0.75-1.25 倍之间，主要用于观察轻微风险降权是否能带来更稳健的改善。

## 评价口径

每一种拟合权重都输出三种 R2：

1. `own`
   - 用拟合时同一套权重评价。
   - 这能说明模型在自己强调的股票集合上解释得怎么样。

2. `sqrt_mv`
   - 用正式基线的市值平方根权重评价。
   - 这更接近原始 CNE6-style 模型的比较口径。

3. `equal`
   - 用等权评价。
   - 观察是否只是大市值或低噪声股票被解释得更好。

如果某个风险加权方案只有 `own` R2 很高，但 `sqrt_mv` 和 `equal` 评价下反而下降，那么我会把它解释为“指标权重变化导致的局部解释力提高”，而不是全市场风险模型变好了。

## 诊断指标

每日和汇总层面输出：

- R2、调整 R2、R2 超过 0.5 的天数占比。
- 成功回归天数、样本股票数、因子数、行业数。
- 条件数，检查回归矩阵是否不稳定。
- 拟合权重有效样本数 `ESS = sum(w)^2 / sum(w^2)`。
- 有效样本占比 `ESS / n_obs`。
- 有效样本数和因子数的相对关系。如果 `ESS` 明显小于因子数，即使自权重 R2 很高也要视为过拟合或数值病态。
- 最大权重占比。
- P99/P50 权重比。

这些权重集中度指标用于判断高 R2 是否来自样本被过度集中。

## 输出

新增输出目录：

- `outputs/cne6_risk_weighted_regression/`

主要文件：

- `risk_weighted_regression_daily.csv`
- `risk_weighted_regression_summary.csv`
- `risk_weighted_regression_comparable_summary.csv`
- `risk_weighted_regression_by_year.csv`
- `risk_weight_diagnostics.csv`
- `risk_weighted_r2_summary.png`
- `risk_weighted_same_day_cross_eval_r2.png`
- `risk_weight_concentration.png`

新增报告：

- `docs/cne6_risk_weighted_regression_experiment_2026-08-04.md`

## 判断方式

这次实验的关键不是找到最高的 `own` R2，而是看三件事：

1. `own` R2 是否明显提高。
2. 换成 `sqrt_mv` 或 `equal` 评价后是否仍然提高。
3. 有效样本数和最大权重占比是否还能接受。

如果倒特异风险方案自权重 R2 超过 0.5 甚至更高，但交叉评价 R2 降到原始基线以下，我会倾向于认为它不能直接作为正式 CNE6 回归替代方案，只能说明低特异风险股票更容易被共同因子解释。

## 验证

1. 单元测试检查 `t-3` 特异风险映射不使用未来数据。
2. 单元测试检查自权重 R2 和交叉评价 R2 可以明显不同。
3. 单元测试检查权重截尾和有效样本数诊断正常。
4. 用真实中证 500 历史面板跑完整实验，并在报告中列出总体和分年度结果。
