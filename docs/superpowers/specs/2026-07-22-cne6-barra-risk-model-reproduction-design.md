# CNE6 Barra 风险模型复现设计

日期：2026-07-22

## 1. 背景

前面的中证500指数增强工作已经加入了行业和风格约束，但当前风格暴露仍然是项目内部的简化版本，主要服务组合优化，不是完整的风险模型。导师后续如果继续追问“这个组合的主动风险来自哪里”“风格约束到底控制了什么”“预测跟踪误差和实际跟踪误差能不能对上”，只靠现在的简化暴露不够。

这次任务的目标是复现一个基于公开资料和 Tushare 可得数据的 CNE6-style Barra 风险模型。这里需要先把边界讲清楚：MSCI Barra CNE6 的完整商业方法、描述子权重、协方差估计细节和部分数据源不是公开资料，所以本项目不声称精确复制 MSCI 商业版 CNE6。我的复现目标是尽量沿着 CNE6 的公开结构，用真实可获取的数据搭出可解释、可检验、能服务指数增强的中国股票风险模型。

本次股票池沿用中证500，样本期尽量从 2010 年以后开始，拉到当前可用最新交易日。数据层优先使用 Tushare 真实字段；只有当字段确实不可得时，才使用 proxy，并在元数据和报告中明确标注。

## 2. 目标

第一版复现需要做到以下几点：

1. 补齐中证500自 2010 年以来尽可能完整的日频行情、复权、估值、财报、财务指标、资金流、行业和指数权重数据。
2. 构建 point-in-time 的中证500风险模型研究面板，避免财报未来函数。
3. 计算 CNE6-style 底层描述子暴露和风格因子暴露。
4. 构建行业暴露矩阵，并和风格暴露一起进入横截面回归。
5. 回归得到日度风格因子收益、行业因子收益和个股特异收益。
6. 估计滚动因子协方差、个股特异风险和组合预测风险。
7. 对已有中证500 XGB 指数增强组合做主动风险归因、预测跟踪误差和实现跟踪误差对比。
8. 形成导师可读的 Markdown 报告，解释每个因子的计算过程、经济含义、数据口径、缺陷和用途。

## 3. 非目标

本阶段不做以下事情：

1. 不声称复现结果等同 MSCI 商业 Barra CNE6。
2. 不把缺失字段静默替换成 proxy 后仍然称为 direct。
3. 不用当前中证500成分倒推历史股票池。
4. 不用财报报告期数据直接填充到公告日前的交易日。
5. 不把风险因子当作 alpha 因子直接评价“好坏”。风险模型的核心是解释和控制风险暴露，不是预测收益。
6. 不在第一版引入过度复杂的贝叶斯调整、Newey-West 调整或商业级协方差收缩，除非基础模型已经跑通且诊断显示确实需要。

## 4. 总体架构

整体拆成五层：

| 层级 | 职责 | 主要输出 |
|---|---|---|
| 数据层 | 拉取、缓存、审计 Tushare 数据，构建 PIT 中证500面板 | `cne6_csi500_daily_panel.parquet`, `data_availability.csv` |
| 描述子层 | 计算底层 descriptor，做去极值、标准化、方向处理 | `descriptor_exposures.parquet`, `descriptor_metadata.csv` |
| 暴露层 | 合成风格暴露，构建行业暴露 | `style_exposures.parquet`, `industry_exposures.parquet` |
| 风险模型层 | 横截面回归因子收益，估计协方差和特异风险 | `factor_returns.csv`, `specific_returns.parquet`, `factor_covariance_rolling.parquet`, `specific_risk.parquet` |
| 归因报告层 | 对指数增强组合做主动风险归因并生成报告 | `portfolio_risk_attribution.csv`, `predicted_vs_realized_te.csv`, Markdown 报告 |

数据流如下：

```text
Tushare raw data + existing processed data
    |
    v
2010-2026 中证500 PIT 日频面板
    |
    |-- descriptor exposures
    |-- style exposures
    |-- industry exposures
    v
每日横截面回归
    |
    |-- factor returns
    |-- specific returns
    v
滚动因子协方差 + 特异风险
    |
    v
指数增强组合主动风险归因
```

## 5. 数据补齐设计

优先使用 Tushare 真实字段。当前本地已有 2018-2026 的中证500 PIT 面板，但要拉长到 2010 年以后，需要补齐下列表。

| 数据 | Tushare 表 | 用途 |
|---|---|---|
| 股票日行情 | `daily` | 日收益、动量、波动、回归标签 |
| 复权因子 | `adj_factor` | 构造前复权价格 |
| 每日指标 | `daily_basic` | 市值、估值、换手、股息率 |
| 中证500权重 | `index_weight(000905.SH)` | 历史股票池和基准权重 |
| 财务指标 | `fina_indicator` | ROE、ROA、成长、周转率、杠杆、EPS、BPS、CFPS |
| 利润表 | `income` | 收入、利润、TTM、同比成长 |
| 资产负债表 | `balancesheet` | 总资产、负债、权益、现金、应收、存货 |
| 现金流量表 | `cashflow` | OCF、FCF、现金流质量 |
| 资金流 | `moneyflow` | 大单/超大单资金流代理情绪因子 |
| 股票基础信息 | `stock_basic`, `stock_company` | 行业、上市状态、上市日期 |
| 指数行情 | `index_daily` | 中证500收益、市场收益、回归和归因对照 |

数据补齐规则：

1. 先检测本地已有缓存，缺哪张补哪张，不重复下载已经完整的文件。
2. 默认拉取区间为 `20100101` 到当前日期。若接口单次返回限制明显，则按年度或季度分块。
3. 原始数据保存为 CNE6 专用缓存，例如 `data/raw/cne6_daily_20100101_20260722.parquet`，避免覆盖前面 alpha 挖掘使用的数据。
4. 财报数据统一按 `ann_date` 或 `f_ann_date` 做可得日对齐；若两者同时存在，优先使用更保守的实际公告可得日。
5. 中证500股票池按历史指数权重判断，权重大于 0 的股票进入当日样本。
6. 停牌、上市时间、ST 状态等过滤项优先使用本地已有字段；如果字段不足，报告中披露未完全覆盖的过滤条件。
7. Tushare token 不写入代码、报告或缓存元数据，只从环境变量或命令参数读取。

数据层输出：

| 文件 | 内容 |
|---|---|
| `data/raw/cne6_*.parquet` | CNE6 复现专用原始缓存 |
| `data/processed/cne6_csi500_daily_panel.parquet` | PIT 对齐后的中证500日频研究面板 |
| `outputs/cne6_reproduction/data_availability.csv` | 每张表、每个字段、每年覆盖率 |
| `outputs/cne6_reproduction/descriptor_availability.csv` | 每个描述子的 direct/proxy/unavailable 状态 |

## 6. CNE6-style 风格因子与描述子

每个风格因子由若干底层描述子合成。描述子先单独计算，再按交易日横截面做去极值和标准化。所有描述子都要记录公式、字段来源、是否真实字段、是否 proxy，以及经济含义。

| 风格因子 | 拟实现描述子 | 经济含义 |
|---|---|---|
| Size | `log_total_mv`, `mid_cap_proxy` | 市值不同的股票承担不同系统性风险；小市值往往有更强流动性和交易拥挤风险 |
| Volatility | `beta`, `daily_std`, `cumulative_range`, `residual_volatility_proxy` | 高波动、高 Beta 股票对市场冲击更敏感，组合风险贡献也更高 |
| Liquidity | `avg_turnover`, `avg_amount`, `turnover_stability` | 流动性影响交易冲击、调仓成本和拥挤程度 |
| Momentum | `12m_minus_1m_return`, `6m_return`, `short_reversal` | 中期趋势可能延续，短期过度反应可能反转 |
| Value | `book_to_price`, `earnings_yield`, `sales_to_price`, `cashflow_to_price` | 低估值股票和高估值股票具有不同的风险收益特征 |
| Growth | `revenue_yoy`, `net_profit_yoy`, `roe_growth`, `asset_turnover_yoy` | 成长暴露刻画收入、利润和经营效率改善 |
| Quality | `roe`, `roa`, `gross_margin`, `cashflow_to_profit`, `low_leverage`, `accrual_quality` | 高盈利质量、现金流扎实、负债压力低的公司通常风险更可控 |
| Dividend Yield | `dv_ttm`, `dv_ratio` | 分红收益率暴露反映现金回报和防御属性 |
| Sentiment / Funding Proxy | `large_order_netflow`, `extra_large_order_netflow`, `moneyflow_momentum` | 若分析师预期数据不可得，用资金流刻画市场情绪和交易偏好 |

描述子处理规则：

1. 横截面去极值使用分位数 winsorize 或 MAD robust winsorize。
2. 标准化后同一交易日均值接近 0，标准差接近 1。
3. 财务类描述子使用 PIT 面板，不直接读未来报告期。
4. 方向统一：例如 `debt_to_assets` 越高不代表质量越好，所以 Quality 中使用 `low_leverage = -z(debt_to_assets)`。
5. 同一风格内部描述子暂时等权合成；描述子权重不强行模仿商业 CNE6。
6. 如果某个描述子覆盖率不足，风格合成时按可用描述子均值计算，并记录当日有效描述子数量。
7. 行业暴露单独建模，不混入风格因子。

暴露层输出：

| 文件 | 内容 |
|---|---|
| `descriptor_exposures.parquet` | 每只股票每日所有底层描述子暴露 |
| `style_exposures.parquet` | 每只股票每日九类风格暴露 |
| `industry_exposures.parquet` | 每只股票每日行业哑变量暴露 |
| `descriptor_metadata.csv` | 描述子公式、字段来源、direct/proxy 状态、经济含义 |
| `style_correlation.csv` | 风格因子相关性 |
| `style_coverage_by_year.csv` | 每年每个风格因子的覆盖率 |

## 7. 风险模型估计

风险模型的核心不是预测股票收益，而是把股票收益拆成共同因子收益和个股特异收益。每日横截面回归形式为：

```text
stock_return_i,t = country_factor_t
                 + industry_exposure_i,t * industry_factor_return_t
                 + style_exposure_i,t * style_factor_return_t
                 + specific_return_i,t
```

其中：

| 符号 | 含义 |
|---|---|
| `stock_return_i,t` | 股票 i 在 t 到 t+1 的收益 |
| `country_factor_t` | 全市场共同收益截距项 |
| `industry_exposure_i,t` | 股票 i 的行业哑变量 |
| `industry_factor_return_t` | 当日行业因子收益 |
| `style_exposure_i,t` | 股票 i 的风格暴露 |
| `style_factor_return_t` | 当日风格因子收益 |
| `specific_return_i,t` | 行业和风格无法解释的个股特异收益 |

估计规则：

1. 每个交易日单独做横截面回归。
2. 回归样本为当日中证500历史成分股。
3. `y` 使用下一交易日收益，第一版先用日频；后续可以扩展到周频和月频。
4. 回归权重优先使用市值平方根或中证500权重，降低极小市值和异常收益对结果的影响。
5. 行业哑变量处理共线性：保留截距并丢弃一个基准行业，或使用行业因子收益加权和为零约束。第一版优先采用“保留截距、丢弃基准行业”的稳定实现。
6. 每日输出风格因子收益、行业因子收益、特异收益和回归诊断。
7. 若某日有效股票数过少、有效行业数过少或矩阵病态，该日回归结果标记为失败，不静默填补。

回归诊断：

| 指标 | 用途 |
|---|---|
| 样本股票数 | 检查当日中证500覆盖率 |
| 有效行业数 | 检查行业暴露是否足够 |
| 有效风格数 | 检查风格暴露是否缺失 |
| R² / adjusted R² | 看行业和风格能解释多少截面收益 |
| 残差均值和残差标准差 | 检查回归是否异常 |
| 条件数 | 检查暴露矩阵共线性风险 |

## 8. 因子协方差、特异风险和组合归因

因子收益序列得到后，使用滚动窗口估计因子协方差。第一版使用 60 日、120 日和 252 日窗口，主报告以 252 日为主，60 日和 120 日作为敏感性对照。

组合主动风险近似分解为：

```text
portfolio_var = active_exposure' * factor_cov * active_exposure
              + sum(active_weight_i^2 * specific_var_i)
```

其中：

| 项 | 含义 |
|---|---|
| `active_exposure` | 组合相对中证500基准的主动行业和主动风格暴露 |
| `factor_cov` | 行业和风格因子收益协方差矩阵 |
| `active_weight_i` | 股票 i 的组合权重减基准权重 |
| `specific_var_i` | 股票 i 的特异收益滚动方差 |

风险归因输出：

| 文件 | 内容 |
|---|---|
| `factor_returns.csv` | 日度行业和风格因子收益 |
| `specific_returns.parquet` | 个股日度特异收益 |
| `factor_covariance_rolling.parquet` | 滚动因子协方差 |
| `specific_risk.parquet` | 个股滚动特异风险 |
| `regression_diagnostics.csv` | 每日回归质量 |
| `portfolio_risk_attribution.csv` | 指数增强组合主动风险归因 |
| `predicted_vs_realized_te.csv` | 预测跟踪误差和实现跟踪误差对比 |

归因报告至少回答：

1. 组合主动风险主要来自行业、风格还是个股特异风险。
2. 当前 XGB 指数增强组合是否仍然有明显 Size、Value、Momentum、Volatility、Liquidity、Quality 等主动暴露。
3. 预测跟踪误差和实际跟踪误差是否在同一量级。
4. 哪些时期风险模型解释能力较差，以及可能原因。
5. 风险约束是否应该替换现有简化 style exposure，或者先并行观察。

## 9. 测试设计

测试按数据、暴露、回归、风险和报告五类进行。

| 测试 | 内容 |
|---|---|
| 数据测试 | 日期范围正确，关键字段存在，指数权重非负，财报可得日不晚于使用日 |
| 股票池测试 | 每日样本来自历史中证500权重，不使用当前成分回看历史 |
| 描述子测试 | 去极值和标准化后同日均值接近 0，异常值可控 |
| 暴露测试 | 风格暴露非空，行业暴露每只股票最多对应一个行业 |
| 回归测试 | 暴露矩阵可估计，因子收益非空，失败日期有显式标记 |
| 协方差测试 | 协方差矩阵维度正确、方差非负、窗口长度正确 |
| 特异风险测试 | 个股特异风险非负，覆盖率可解释 |
| 归因测试 | 主动权重、主动暴露、预测风险和实现 TE 能对齐 |
| 报告测试 | Markdown 引用的图片和表格文件存在 |

## 10. 报告设计

最终报告路径：

```text
docs/cne6_barra_risk_model_reproduction_2026-07-22.md
```

报告使用导师可读的研究汇报口吻，不写过程性元话语。报告结构建议如下：

1. 这次复现要解决什么问题。
2. CNE6-style 风险模型和 alpha 因子的区别。
3. 数据来源、样本期、股票池和 PIT 对齐方法。
4. 每个风格因子及其描述子的计算过程和经济含义。
5. 哪些字段是 direct，哪些是 proxy，哪些不可得。
6. 横截面回归方法和因子收益结果。
7. 因子协方差、特异风险和模型解释力。
8. 对现有中证500 XGB 指数增强组合的风险归因。
9. 预测跟踪误差和实现跟踪误差对比。
10. 局限性和后续改进。

报告中尽量包含图片，使用 Markdown 的 `![]()` 引入；核心表格直接写入报告正文，同时保留 CSV 原始输出。

## 11. 实施顺序

第一阶段先完成数据和可用性审计：

1. 检测本地已有 2010-2026 相关缓存。
2. 使用 Tushare 补齐缺失数据。
3. 构建 CNE6 专用中证500 PIT 面板。
4. 输出数据覆盖率和描述子可用性表。

第二阶段完成暴露计算：

1. 实现底层描述子。
2. 实现去极值、标准化和方向处理。
3. 合成九类风格暴露。
4. 输出暴露覆盖率、相关性和描述子元数据。

第三阶段完成风险模型：

1. 实现每日横截面回归。
2. 输出风格因子收益、行业因子收益和特异收益。
3. 估计滚动协方差和个股特异风险。
4. 输出回归诊断。

第四阶段完成组合归因和报告：

1. 读取已有 XGB 指数增强组合权重。
2. 计算主动行业、主动风格和主动个股风险。
3. 对比预测 TE 和实现 TE。
4. 生成图表和 Markdown 报告。
5. 跑测试并修正发现的问题。

## 12. 主要风险和处理

| 风险 | 处理 |
|---|---|
| 2010-2017 财务数据或每日指标拉取不完整 | 分年度拉取，输出覆盖率，必要时报告样本从可用年份开始 |
| Tushare 部分接口权限不足 | 标记 unavailable，不硬造字段 |
| 行业分类和商业 CNE6 口径不一致 | 使用可得行业字段或中信/申万 proxy，并披露口径 |
| 描述子高度相关 | 输出相关性矩阵，第一版等权合成，后续可做去相关或降维 |
| 横截面回归矩阵病态 | 加入列筛选、标准化、基准行业删除和失败日期标记 |
| 预测 TE 和实现 TE 偏离大 | 分解行业、风格、特异项，检查协方差窗口和暴露估计 |
| 运行时间和磁盘占用较大 | 原始缓存按表和年份分块，处理后合并，避免重复下载 |

## 13. 验收标准

第一版完成后应满足：

1. 有 2010 年以来尽可能长的中证500 CNE6-style PIT 面板。
2. 每个风格因子和描述子都有元数据、公式、字段来源和经济含义。
3. 每日风格暴露和行业暴露可落盘。
4. 每日因子收益、特异收益和回归诊断可落盘。
5. 滚动因子协方差和特异风险可落盘。
6. 至少一个已有中证500指数增强组合可以完成主动风险归因。
7. 报告能直接解释模型边界、数据质量、结果和局限性。
8. 核心测试通过；如果某些测试因为数据权限或覆盖不足无法通过，报告中必须说明原因。
