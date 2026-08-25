# 国信联合动量因子复现报告

日期：2026-07-13

## 结论摘要

- 样本期：`20190101` 至 `20201231`；股票池：`all`。
- 行业收益使用 Tushare 中信一级 `.CI` 行业指数；股票到中信一级行业的归属使用 Tushare 行业字段映射 proxy。
- 因子方向和数量级需要与 PDF 对照，但不能解释为 Wind 原始口径逐点复现。

## PDF 原始结果

| factor | rankic_mean | icir_annual | top_excess | bottom_excess |
| --- | --- | --- | --- | --- |
| IMAX20 | 3.76% | 3.23 | 0.61% | -0.46% |
| ICM | 5.59% | 3.85 | 0.63% | -1.02% |
| VICM | 5.90% | 3.90 | 0.68% | -1.09% |
| VICR | -5.93% | -3.55 | 0.59% | -1.23% |
| CMC | 6.02% | 3.86 | 0.66% | -1.15% |
| MCMC | 6.40% | 4.00 | 0.53% | -1.22% |

## 本地复现结果

| factor | rankic_mean | rankic_std | icir_annual | rankic_positive_ratio | n_months | direction | top_excess | bottom_excess | long_short_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMAX20 | 4.83% | 4.87% | 3.44 | 78.26% | 23 | 1.0 | 0.99% | -0.29% | 1.28% |
| ICM | 5.34% | 7.47% | 2.47 | 73.91% | 23 | 1.0 | 0.88% | -0.82% | 1.69% |
| VICM | 5.62% | 7.40% | 2.63 | 73.91% | 23 | 1.0 | 0.81% | -0.66% | 1.47% |
| ICR | -3.39% | 9.04% | -1.30 | 39.13% | 23 | -1.0 | 0.28% | -0.53% | 0.82% |
| VICR | -3.87% | 9.09% | -1.48 | 34.78% | 23 | -1.0 | 0.36% | -0.52% | 0.88% |
| CMC | 5.24% | 8.25% | 2.20 | 73.91% | 23 | 1.0 | 0.59% | -0.71% | 1.30% |
| MCMC | 5.66% | 8.23% | 2.38 | 78.26% | 23 | 1.0 | 0.24% | -0.55% | 0.79% |

## 与 PDF 对照

| factor | rankic_mean_local | rankic_mean_pdf | icir_annual_local | icir_annual_pdf | top_excess_local | top_excess_pdf | bottom_excess_local | bottom_excess_pdf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMAX20 | 4.83% | 3.76% | 3.44 | 3.23 | 0.99% | 0.61% | -0.29% | -0.46% |
| ICM | 5.34% | 5.59% | 2.47 | 3.85 | 0.88% | 0.63% | -0.82% | -1.02% |
| VICM | 5.62% | 5.90% | 2.63 | 3.90 | 0.81% | 0.68% | -0.66% | -1.09% |
| ICR | -3.39% |  | -1.30 |  | 0.28% |  | -0.53% |  |
| VICR | -3.87% | -5.93% | -1.48 | -3.55 | 0.36% | 0.59% | -0.52% | -1.23% |
| CMC | 5.24% | 6.02% | 2.20 | 3.86 | 0.59% | 0.66% | -0.71% | -1.15% |
| MCMC | 5.66% | 6.40% | 2.38 | 4.00 | 0.24% | 0.53% | -0.55% | -1.22% |

![](../outputs/guoxin_co_momentum_reproduction_smoke/monthly_rankic.png)

![](../outputs/guoxin_co_momentum_reproduction_smoke/decile_monthly_excess.png)

## 行业映射覆盖

| source_industry | citic_level1 | mapping_status | n_stocks |
| --- | --- | --- | --- |
| 电气设备 | 电力设备及新能源 | mapped | 347 |
| 元器件 | 电子 | mapped | 309 |
| 专用机械 | 机械 | mapped | 291 |
| 软件服务 | 计算机 | mapped | 279 |
| 汽车配件 | 汽车 | mapped | 266 |
| 化工原料 | 基础化工 | mapped | 257 |
| 半导体 | 电子 | mapped | 195 |
| 医疗保健 | 医药 | mapped | 184 |
| 化学制药 | 医药 | mapped | 148 |
| 机械基件 | 机械 | mapped | 137 |
| 通信设备 | 通信 | mapped | 137 |
| 建筑工程 | 建筑 | mapped | 128 |
| 环境保护 | 电力及公用事业 | mapped | 119 |
| 电器仪表 | 机械 | mapped | 107 |
| 食品 | 食品饮料 | mapped | 93 |
| 家用电器 | 家电 | mapped | 92 |
| IT设备 | 计算机 | mapped | 81 |
| 互联网 | 计算机 | mapped | 80 |
| 塑料 | 基础化工 | mapped | 80 |
| 生物制药 | 医药 | mapped | 79 |
| 家居用品 | 轻工制造 | mapped | 71 |
| 中成药 | 医药 | mapped | 68 |
| 小金属 | 有色金属 | mapped | 67 |
| 服饰 | 纺织服装 | mapped | 65 |
| 农药化肥 | 基础化工 | mapped | 58 |
| 广告包装 | 轻工制造 | mapped | 54 |
| 航空 | 国防军工 | mapped | 53 |
| 文教休闲 | 轻工制造 | mapped | 51 |
| 证券 | 非银行金融 | mapped | 50 |
| 运输设备 | 机械 | mapped | 49 |
| 仓储物流 | 交通运输 | mapped | 47 |
| 供气供热 | 电力及公用事业 | mapped | 45 |
| 纺织 | 纺织服装 | mapped | 44 |
| 区域地产 | 房地产 | mapped | 43 |
| 银行 | 银行 | mapped | 42 |
| 农业综合 | 农林牧渔 | mapped | 40 |
| 工程机械 | 机械 | mapped | 38 |
| 医药商业 | 医药 | mapped | 37 |
| 染料涂料 | 基础化工 | mapped | 36 |
| 影视音像 | 传媒 | mapped | 35 |

## 口径差异

- PDF 使用 Wind 和中信行业历史成分；本地当前无法从 Tushare 取得中信行业历史成分，使用 proxy 映射。
- 本地按可用字段剔除当前 ST 和上市未满 6 个月股票，不能完全复刻 ST 摘帽 3 个月过滤。
- `VICM/VICR` 默认使用 `return * volume` 排序，后续需要保留 `return * amount` 做敏感性检查。

## 输出文件

- 输出目录：`outputs/guoxin_co_momentum_reproduction_smoke`
- 因子值：`outputs/guoxin_co_momentum_reproduction_smoke/factor_values.parquet`
- 汇总表：`outputs/guoxin_co_momentum_reproduction_smoke/factor_summary.csv`
- 月度 RankIC：`outputs/guoxin_co_momentum_reproduction_smoke/monthly_rankic.csv`
- 分组收益：`outputs/guoxin_co_momentum_reproduction_smoke/decile_returns.csv`