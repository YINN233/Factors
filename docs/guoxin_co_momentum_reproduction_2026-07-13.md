# 国信联合动量因子复现报告

日期：2026-07-13

## 结论摘要

- 样本期：`20100101` 至 `20231231`；股票池：`all`。
- 行业收益使用 Tushare 中信一级 `.CI` 行业指数；股票到中信一级行业的归属使用 Tushare 行业字段映射 proxy。
- 因子方向和数量级需要与 PDF 对照，但不能解释为 Wind 原始口径逐点复现。

## PDF 原始结果

| factor | rankic_mean | icir_annual | top_excess | bottom_excess |
| --- | --- | --- | --- | --- |
| IMAX20 | 3.76% | 3.23 | 0.61% | -0.46% |
| ICM | 5.59% | 3.85 | 0.63% | -1.02% |
| VICM | 5.90% | 3.90 | 0.68% | -1.09% |
| ICR | -5.60% | -3.39 | 0.60% | -1.13% |
| VICR | -5.93% | -3.55 | 0.59% | -1.23% |
| CMC | 6.02% | 3.86 | 0.66% | -1.15% |
| MCMC | 6.40% | 4.00 | 0.53% | -1.22% |

## 本地复现结果

| factor | rankic_mean | rankic_std | icir_annual | rankic_positive_ratio | n_months | direction | top_excess | bottom_excess | long_short_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMAX20 | 3.29% | 6.87% | 1.66 | 64.67% | 167 | 1.0 | 0.52% | -0.44% | 0.96% |
| ICM | 4.32% | 9.13% | 1.64 | 66.47% | 167 | 1.0 | 0.65% | -0.82% | 1.47% |
| VICM | 4.77% | 9.14% | 1.81 | 67.07% | 167 | 1.0 | 0.63% | -0.90% | 1.53% |
| ICR | -3.29% | 10.92% | -1.04 | 38.32% | 167 | -1.0 | 0.42% | -0.45% | 0.87% |
| VICR | -3.72% | 10.98% | -1.17 | 34.73% | 167 | -1.0 | 0.44% | -0.54% | 0.99% |
| CMC | 4.69% | 9.35% | 1.74 | 68.86% | 167 | 1.0 | 0.57% | -0.84% | 1.41% |
| MCMC | 6.67% | 8.58% | 2.69 | 80.84% | 167 | 1.0 | 0.59% | -1.04% | 1.63% |

## 与 PDF 对照

| factor | rankic_mean_local | rankic_mean_pdf | icir_annual_local | icir_annual_pdf | top_excess_local | top_excess_pdf | bottom_excess_local | bottom_excess_pdf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMAX20 | 3.29% | 3.76% | 1.66 | 3.23 | 0.52% | 0.61% | -0.44% | -0.46% |
| ICM | 4.32% | 5.59% | 1.64 | 3.85 | 0.65% | 0.63% | -0.82% | -1.02% |
| VICM | 4.77% | 5.90% | 1.81 | 3.90 | 0.63% | 0.68% | -0.90% | -1.09% |
| ICR | -3.29% | -5.60% | -1.04 | -3.39 | 0.42% | 0.60% | -0.45% | -1.13% |
| VICR | -3.72% | -5.93% | -1.17 | -3.55 | 0.44% | 0.59% | -0.54% | -1.23% |
| CMC | 4.69% | 6.02% | 1.74 | 3.86 | 0.57% | 0.66% | -0.84% | -1.15% |
| MCMC | 6.67% | 6.40% | 2.69 | 4.00 | 0.59% | 0.53% | -1.04% | -1.22% |

![](../outputs/guoxin_co_momentum_reproduction/monthly_rankic.png)

![](../outputs/guoxin_co_momentum_reproduction/decile_monthly_excess.png)

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

- 输出目录：`outputs/guoxin_co_momentum_reproduction`
- 因子值：`outputs/guoxin_co_momentum_reproduction/factor_values.parquet`
- 汇总表：`outputs/guoxin_co_momentum_reproduction/factor_summary.csv`
- 月度 RankIC：`outputs/guoxin_co_momentum_reproduction/monthly_rankic.csv`
- 分组收益：`outputs/guoxin_co_momentum_reproduction/decile_returns.csv`
