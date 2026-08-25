# CNE6 风险模型动态校验与修正设计

## 背景

导师提出希望在 CNE6 Barra 风险模型里加一个校验模块，或者做动态修正。我的理解是：风险模型每天会给出未来风险预测，但等真实收益发生以后，应该回头检查预测偏差，并把近期偏差反馈到后面的风险预测里。

这次模块只处理 CNE6 风险模型本身，不处理 XGB 指数增强选股。它不重写原始 `factor_covariance_rolling.parquet` 和 `specific_risk.parquet`，而是在原始风险预测之上输出一层校准乘数和校准后的预测结果。

## 时间口径

对目标日 `t`：

1. 预测时点使用 `t-3`。
2. 预测输入只使用截至 `t-3` 可见的因子协方差和个股特异风险。
3. 到 `t+1` 时，`t` 日因子收益和个股特异收益已经落盘，再回看 `t` 的预测偏差。
4. 这个偏差进入后续日期的滚动校准窗口，不允许用 `t` 的真实结果修正 `t` 自己。

## 校验对象

1. 共同因子风险
   - 从 `factor_covariance_rolling.parquet` 读取预测日之前最近一期 252 日因子协方差。
   - 用协方差对角线作为每个因子的预测方差。
   - 用 `factor_returns.csv` 中目标日的因子收益平方作为实现方差代理。
   - 分成 `country`、`style`、`industry` 和 `factor_all` 四类做偏差统计。

2. 个股特异风险
   - 从 `specific_risk.parquet` 读取预测日之前最近一期 `specific_risk_252`。
   - 用 `specific_returns.parquet` 中目标日的特异收益平方作为实现方差代理。
   - 每个目标日先在截面上聚合成一个特异风险偏差比率。

## 动态修正

对每个风险块生成偏差比率：

```text
ratio = realized_variance / predicted_variance
```

如果 ratio 长期大于 1，说明原始风险预测偏低；如果 ratio 长期小于 1，说明原始风险预测偏高。

动态修正使用滚动窗口内的聚合偏差比率作为校准乘数：

```text
calibration_multiplier = sum(realized_variance) / sum(predicted_variance)
corrected_predicted_variance = original_predicted_variance * calibration_multiplier
```

本次比较四个校准窗口：

- 20 个交易日
- 40 个交易日
- 60 个交易日
- 126 个交易日，近似六个月

所有单日 ratio 会先做截尾，再还原成用于聚合的 realized_variance，避免极端日把乘数打歪。

## 输出

输出目录：

```text
outputs/cne6_dynamic_risk_calibration/
```

输出文件：

- `daily_block_validation.csv`：目标日、预测时点、风险块、原始预测方差、实现方差、原始偏差比率。
- `daily_calibrated_forecasts.csv`：不同校准窗口下的校准乘数和修正后预测方差。
- `calibration_summary.csv`：不同风险块和校准窗口的整体表现。
- `calibration_by_year.csv`：分年度表现。
- `calibration_bias_by_window.png`：不同窗口的预测偏差对比图。
- `calibration_error_by_window.png`：不同窗口的误差对比图。

报告：

```text
docs/cne6_dynamic_risk_validation_calibration_2026-08-04.md
```

## 判断标准

主要看两个指标：

1. 偏差比率是否更接近 1：
   - `realized / predicted = 1` 表示整体不偏。
   - 大于 1 表示预测偏低。
   - 小于 1 表示预测偏高。

2. log 误差是否下降：
   - 用 `abs(log(corrected_predicted / realized))` 衡量预测和实现的距离。
   - 这个指标比普通差值更适合处理风险这种正数变量。

如果短窗口能显著降低偏差但误差更大，说明它反应快但不稳定；如果 126 日更稳但跟不上近期状态，说明六个月窗口过慢。

## 边界

这不是商业 Barra 的完整动态波动率模型，也不是 GARCH/Kalman 这类状态空间模型。它是一个轻量的、可解释的、不会引入未来函数的校验和乘数修正层。后续如果效果明确，再考虑把乘数接入组合风险归因。
