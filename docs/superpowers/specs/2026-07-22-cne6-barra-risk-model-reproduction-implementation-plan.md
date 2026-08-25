# CNE6 Barra 风险模型复现实现计划

日期：2026-07-22

对应设计文档：`docs/superpowers/specs/2026-07-22-cne6-barra-risk-model-reproduction-design.md`

## 原则

1. 股票池使用中证500历史成分，样本期尽量从 2010 年拉长到当前最新可用交易日。
2. 数据优先使用 Tushare 真实字段；不可得字段才使用 proxy，并在元数据里标明。
3. 财务数据必须按公告日 point-in-time 对齐，不能使用未来财报。
4. CNE6-style 风险因子用于解释和控制风险，不直接当作 alpha 因子筛选“好坏”。
5. 第一版先追求完整闭环：数据、暴露、因子收益、协方差、特异风险、组合归因和报告都要落盘。
6. 不声称精确复刻 MSCI 商业 Barra CNE6。
7. Tushare token 不写进代码、文档或输出文件。

## 阶段 1：数据补齐与可用性审计

目标：补齐 2010 年以来中证500 CNE6-style 风险模型所需数据，并输出覆盖率。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/data/cne6_fetcher.py` | CNE6 专用 Tushare 数据下载、分年缓存和覆盖率审计 |
| `factors/data/cne6_builder.py` | 构建中证500 PIT 日频风险模型面板 |
| `factors/risk/__init__.py` | 风险模型包入口 |

输出文件：

| 文件 | 内容 |
|---|---|
| `data/raw/cne6_daily_<start>_<end>.parquet` | 股票日行情 |
| `data/raw/cne6_adj_factor_<start>_<end>.parquet` | 复权因子 |
| `data/raw/cne6_daily_basic_<start>_<end>.parquet` | 每日估值、换手、市值、股息率 |
| `data/raw/cne6_index_weight_000905_SH_<start>_<end>.parquet` | 中证500历史成分权重 |
| `data/raw/cne6_income_<start>_<end>.parquet` | 利润表 |
| `data/raw/cne6_balancesheet_<start>_<end>.parquet` | 资产负债表 |
| `data/raw/cne6_cashflow_<start>_<end>.parquet` | 现金流量表 |
| `data/raw/cne6_fina_indicator_<start>_<end>.parquet` | 财务指标 |
| `data/raw/cne6_moneyflow_<start>_<end>.parquet` | 资金流 |
| `data/processed/cne6_csi500_daily_panel.parquet` | PIT 对齐后的日频风险模型面板 |
| `outputs/cne6_reproduction/data_availability.csv` | 表级、字段级、年份级覆盖率 |
| `outputs/cne6_reproduction/descriptor_availability.csv` | 描述子 direct/proxy/unavailable 清单 |

实现要点：

1. 下载前检查现有缓存，避免重复拉取。
2. 单表拉取失败时按年度降级重试。
3. 对 Tushare 返回空表、字段缺失、权限不足分别记录原因。
4. `daily + adj_factor` 构造 `open_adj/high_adj/low_adj/close_adj`。
5. `daily_basic` 合并 `total_mv/circ_mv/turnover_rate/pe_ttm/pb/ps_ttm/dv_ttm`。
6. `index_weight(000905.SH)` 按 `con_code + trade_date` 转成历史股票池和基准权重。
7. 财报使用 `ann_date/f_ann_date` 做 as-of merge。
8. 资金流保留 `buy_lg_amount/sell_lg_amount/buy_elg_amount/sell_elg_amount/net_mf_amount` 并派生净流入。

验收：

```bash
venv/bin/python -m factors.data.cne6_fetcher --start 20100101 --end 20260722 --token-env TUSHARE_TOKEN
venv/bin/python -m factors.data.cne6_builder --start 20100101 --end 20260722
```

若 2010-2017 某些表无法完整补齐，第一版不阻塞后续实现，但要在 `data_availability.csv` 和最终报告中说明实际可用样本期。

## 阶段 2：描述子与风格暴露

目标：实现 CNE6-style 底层描述子、九类风格因子暴露和行业暴露。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/risk/cne6_descriptors.py` | 描述子公式、字段依赖、direct/proxy 状态和计算函数 |
| `factors/risk/cne6_exposures.py` | 去极值、标准化、风格合成、行业暴露 |

主要描述子：

| 风格 | 描述子 |
|---|---|
| Size | `log_total_mv`, `mid_cap_proxy` |
| Volatility | `beta_252`, `daily_std_252`, `cumulative_range_252`, `residual_volatility_proxy` |
| Liquidity | `avg_turnover_21`, `avg_turnover_63`, `avg_amount_21`, `turnover_stability_63` |
| Momentum | `ret_252_ex_21`, `ret_126`, `short_reversal_21` |
| Value | `book_to_price`, `earnings_yield`, `sales_to_price`, `cashflow_to_price` |
| Growth | `revenue_yoy`, `net_profit_yoy`, `roe_growth`, `asset_turnover_yoy` |
| Quality | `roe_ttm`, `roa_ttm`, `gross_margin_ttm`, `cashflow_to_profit`, `low_leverage`, `accrual_quality` |
| Dividend Yield | `dv_ttm`, `dv_ratio` |
| Sentiment / Funding Proxy | `large_order_netflow_20`, `extra_large_order_netflow_20`, `moneyflow_momentum_60` |

处理规则：

1. 每个描述子先转成数值，清理无穷值。
2. 同日横截面 winsorize，再 z-score。
3. 方向统一到“暴露越高表示该风格越强”。
4. 同一风格内部先等权合成。
5. 描述子覆盖率不足时保留缺失，不做跨股票静默填零。
6. 风格合成时记录每只股票每个风格的有效描述子数量。
7. 行业暴露用当前面板可得行业字段；若无法获得商业 CNE6 行业口径，标记为行业 proxy。

输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/cne6_reproduction/descriptor_exposures.parquet` | 股票-日期-描述子暴露 |
| `outputs/cne6_reproduction/style_exposures.parquet` | 股票-日期-风格暴露 |
| `outputs/cne6_reproduction/industry_exposures.parquet` | 股票-日期-行业暴露 |
| `outputs/cne6_reproduction/descriptor_metadata.csv` | 公式、字段、direct/proxy、经济含义 |
| `outputs/cne6_reproduction/style_correlation.csv` | 风格相关性 |
| `outputs/cne6_reproduction/style_coverage_by_year.csv` | 年度覆盖率 |

验收：

```bash
venv/bin/python -m factors.risk.cne6_exposures --panel data/processed/cne6_csi500_daily_panel.parquet --output outputs/cne6_reproduction
```

## 阶段 3：横截面因子收益回归

目标：用行业和风格暴露解释下一日股票收益，估计日度行业因子收益、风格因子收益和个股特异收益。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/risk/cne6_regression.py` | 日度横截面 WLS 回归、因子收益和残差输出 |

实现要点：

1. 每个交易日独立回归。
2. `y` 使用下一交易日股票收益。
3. `X` 包含截距、行业哑变量和九类风格暴露。
4. 回归权重默认使用 `sqrt(total_mv)`；如果市值缺失，则降级为等权。
5. 行业哑变量采用“保留截距、删除一个基准行业”的稳定方案。
6. 暴露缺失过多的股票从当日回归样本剔除。
7. 样本数不足、有效行业不足、条件数过高时，该日回归标记失败。

输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/cne6_reproduction/factor_returns.csv` | 日度 country、industry、style 因子收益 |
| `outputs/cne6_reproduction/specific_returns.parquet` | 个股日度特异收益 |
| `outputs/cne6_reproduction/regression_diagnostics.csv` | R²、样本数、行业数、风格数、条件数 |

验收：

```bash
venv/bin/python -m factors.risk.cne6_regression --panel data/processed/cne6_csi500_daily_panel.parquet --exposures outputs/cne6_reproduction/style_exposures.parquet --output outputs/cne6_reproduction
```

## 阶段 4：因子协方差和特异风险

目标：基于因子收益和特异收益估计滚动风险参数。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/risk/cne6_risk_model.py` | 滚动协方差、特异风险、风险预测接口 |

实现要点：

1. 因子协方差默认输出 60 日、120 日、252 日三个窗口。
2. 主报告使用 252 日窗口，短窗口作为敏感性对照。
3. 特异风险使用个股特异收益滚动标准差。
4. 协方差矩阵方差项必须非负；出现非数值时记录并跳过。
5. 第一版不做复杂收缩，若矩阵不稳定再加入 shrinkage。

输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/cne6_reproduction/factor_covariance_rolling.parquet` | 滚动因子协方差 |
| `outputs/cne6_reproduction/specific_risk.parquet` | 个股滚动特异风险 |
| `outputs/cne6_reproduction/risk_model_diagnostics.csv` | 协方差覆盖率、缺失率、异常天数 |

验收：

```bash
venv/bin/python -m factors.risk.cne6_risk_model --factor-returns outputs/cne6_reproduction/factor_returns.csv --specific-returns outputs/cne6_reproduction/specific_returns.parquet --output outputs/cne6_reproduction
```

## 阶段 5：指数增强组合风险归因

目标：把 CNE6-style 风险模型接到已有中证500 XGB 指数增强组合，解释主动风险来源。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/reports/cne6_portfolio_attribution.py` | 读取组合权重，计算主动暴露、风险分解和 TE 对比 |

输入文件：

| 文件 | 内容 |
|---|---|
| `outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv` | 既有 XGB 指数增强组合权重 |
| `data/processed/cne6_csi500_daily_panel.parquet` | 中证500基准权重和收益 |
| `outputs/cne6_reproduction/style_exposures.parquet` | 风格暴露 |
| `outputs/cne6_reproduction/industry_exposures.parquet` | 行业暴露 |
| `outputs/cne6_reproduction/factor_covariance_rolling.parquet` | 因子协方差 |
| `outputs/cne6_reproduction/specific_risk.parquet` | 特异风险 |

输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/cne6_reproduction/portfolio_active_exposures.csv` | 主动行业和风格暴露 |
| `outputs/cne6_reproduction/portfolio_risk_attribution.csv` | 主动风险分解 |
| `outputs/cne6_reproduction/predicted_vs_realized_te.csv` | 预测 TE 和实现 TE 对比 |
| `outputs/cne6_reproduction/portfolio_risk_summary.csv` | 归因摘要 |

验收：

```bash
venv/bin/python -m factors.reports.cne6_portfolio_attribution --weights outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv --output outputs/cne6_reproduction
```

## 阶段 6：图表和 Markdown 报告

目标：生成导师可读报告，包含图片、核心表格、公式、经济含义和局限性。

新增文件：

| 文件 | 内容 |
|---|---|
| `factors/reports/cne6_report.py` | 图表和 Markdown 报告生成 |
| `docs/cne6_barra_risk_model_reproduction_2026-07-22.md` | 最终报告 |

图片：

| 图片 | 内容 |
|---|---|
| `style_correlation_heatmap.png` | 风格暴露相关性 |
| `factor_returns_cum.png` | 风格因子累计收益 |
| `regression_r2_timeseries.png` | 回归解释力时间序列 |
| `specific_risk_distribution.png` | 特异风险分布 |
| `predicted_vs_realized_te.png` | 预测 TE 与实现 TE 对比 |
| `portfolio_risk_attribution_stack.png` | 主动风险来源堆叠图 |

报告章节：

1. 研究目标和模型边界。
2. CNE6-style 风险模型和 alpha 模型的区别。
3. 数据口径、样本期和覆盖率。
4. 风格因子、描述子公式和经济含义。
5. direct/proxy/unavailable 字段说明。
6. 横截面回归结果和模型解释力。
7. 因子协方差和特异风险。
8. 中证500 XGB 指数增强组合风险归因。
9. 预测 TE 与实现 TE。
10. 局限性和下一步改进。

验收：

```bash
venv/bin/python -m factors.reports.cne6_report --output outputs/cne6_reproduction --doc docs/cne6_barra_risk_model_reproduction_2026-07-22.md
```

## 阶段 7：测试

新增文件：

| 文件 | 内容 |
|---|---|
| `test_cne6_risk_model.py` | CNE6 数据、暴露、回归、协方差和归因 smoke tests |

测试内容：

1. PIT 对齐测试：财报可得日不晚于交易日。
2. 股票池测试：样本来自历史中证500权重。
3. 标准化测试：同日风格暴露均值接近 0。
4. 描述子元数据测试：每个输出描述子都有公式和经济含义。
5. 回归测试：玩具面板能正确回归出因子收益。
6. 协方差测试：滚动协方差维度正确，方差非负。
7. 归因测试：主动风险分解输出非空，TE 非负。
8. 报告测试：Markdown 引用图片文件存在。

命令：

```bash
venv/bin/python test_cne6_risk_model.py
venv/bin/python test_alpha_mining.py
```

## 最终流水线命令

完整运行顺序：

```bash
venv/bin/python -m factors.data.cne6_fetcher --start 20100101 --end 20260722 --token-env TUSHARE_TOKEN
venv/bin/python -m factors.data.cne6_builder --start 20100101 --end 20260722
venv/bin/python -m factors.risk.cne6_exposures --panel data/processed/cne6_csi500_daily_panel.parquet --output outputs/cne6_reproduction
venv/bin/python -m factors.risk.cne6_regression --panel data/processed/cne6_csi500_daily_panel.parquet --exposures outputs/cne6_reproduction/style_exposures.parquet --output outputs/cne6_reproduction
venv/bin/python -m factors.risk.cne6_risk_model --factor-returns outputs/cne6_reproduction/factor_returns.csv --specific-returns outputs/cne6_reproduction/specific_returns.parquet --output outputs/cne6_reproduction
venv/bin/python -m factors.reports.cne6_portfolio_attribution --weights outputs/csi500_xgb_constrained_index_enhancement/constrained_weights.csv --output outputs/cne6_reproduction
venv/bin/python -m factors.reports.cne6_report --output outputs/cne6_reproduction --doc docs/cne6_barra_risk_model_reproduction_2026-07-22.md
venv/bin/python test_cne6_risk_model.py
```

## 回退策略

| 问题 | 回退 |
|---|---|
| Tushare 单次下载超限 | 按年度或季度分块 |
| 2010-2017 财报覆盖不足 | 先保留 2010 行情类风险因子，财务类从实际可用年份开始，并在报告说明 |
| 行业字段不稳定 | 使用当前可得行业 proxy，后续再补中信或申万历史行业 |
| 回归矩阵病态 | 删除覆盖率低或高度共线列，必要时使用 ridge 回归作为敏感性 |
| 协方差矩阵不稳定 | 第一版输出原始滚动协方差，后续增加 shrinkage |
| 指增权重日期和风险模型日期不完全重合 | 只在交集日期做归因，并输出覆盖率 |
| 全量运行时间过长 | 先跑 2018-2026 smoke，再跑 2010-2026 完整版 |

## 第一轮实施重点

第一轮先完成阶段 1 到阶段 3 的最小闭环：

1. 数据可用性审计。
2. CNE6 PIT 面板。
3. 描述子和风格暴露。
4. 日度横截面因子收益回归。
5. 基础测试。

这轮跑通后再进入协方差、特异风险、组合归因和正式报告，避免数据补齐问题拖垮后面的实现。
