# CNE6 横截面回归 R2 口径对照实验设计

## 背景

当前 CNE6-style 风险模型报告里的平均 R2 是 0.3693。这个数低于导师提到的 0.5 可用水平，但当前代码里的被解释变量是 `fwd_1d_return`，本质上是用今天的行业和风格暴露去解释下一天收益。这个口径更接近短期收益预测，天然比商业风险模型常见的当期收益分解口径更难。

这次先做口径诊断，不直接改写正式风险模型输出。实验目标是判断 R2 偏低主要来自收益口径，还是来自行业、风格、回归方法本身。

## 对照口径

1. `forward_1d`
   - 暴露日期：t
   - 收益日期：t+1
   - 含义：今天暴露解释下一天收益，偏预测口径。
   - 作用：保留当前报告口径，作为基准。

2. `same_day`
   - 暴露日期：t
   - 收益日期：t
   - 含义：当天暴露解释当天收益，偏事后收益分解口径。
   - 作用：更接近 Barra/CNE 风险模型里横截面因子收益估计的 R2 比较口径。

3. `lagged_exposure_1d`
   - 暴露日期：t-1
   - 收益日期：t
   - 含义：前一交易日暴露解释下一交易日实现收益。
   - 作用：避免把当天收盘后才能知道的暴露用于解释当天收益，是更严格的可交易近似口径。

## 输出

新增一个诊断脚本，读取现有：

- `data/processed/cne6_csi500_daily_panel.parquet`
- `outputs/cne6_reproduction/style_exposures.parquet`

输出到 `outputs/cne6_r2_return_definition_comparison/`：

- `r2_return_definition_summary.csv`
- `r2_return_definition_by_year.csv`
- `r2_return_definition_diagnostics.csv`
- `r2_return_definition_timeseries.png`

新增 Markdown 报告：

- `docs/cne6_regression_r2_return_definition_comparison_2026-07-29.md`

## 判断标准

如果 `same_day` 的平均 R2 明显接近或超过 0.5，而 `forward_1d` 仍在 0.37 左右，说明之前低 R2 主要是口径问题：我们不能拿预测口径的 R2 去和风险分解口径的 0.5 标准直接比较。

如果 `same_day` 仍明显低于 0.5，则说明当前 CNE6-style 复现还有模型层面的不足，重点应继续排查行业分类、风格描述子覆盖、行业样本稳定性、回归约束和异常值处理。

## 验证

1. 用小型合成数据测试三种收益口径的样本对齐。
2. 用现有真实数据跑完整诊断。
3. 报告里明确区分“预测口径 R2”和“风险分解口径 R2”，不把实验结果过度包装成商业 Barra CNE6 精确复现。
