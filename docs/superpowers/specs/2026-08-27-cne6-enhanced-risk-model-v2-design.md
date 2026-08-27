# CNE6 增强风险模型 V2 设计

日期：2026-08-27

## 1. 背景与结论

当前仓库已经实现一套面向历史中证500成分股的 CNE6-style 风险模型，但存在三类结构性问题：

1. 34 个底层描述子直接等权合成为 9 个风格因子。Beta、残差波动、非线性市值和短期反转等经济含义不同的暴露被混在同一风格中，缺失描述子还会使单只股票的实际配比漂移。
2. 行业使用静态 `stock_basic.industry` 细分类。全样本出现 109 个行业，最新截面仍有 85 个行业桶，很多桶只有 1-3 只股票。每日约 486 只股票平均需要估计约 96 个行业参数，解释力、自由度和历史可追溯性不足。
3. 正式共同因子协方差是 252 日 Ledoit-Wolf 收缩矩阵，不是 Eigenfactor 风险调整。特异风险的 60/120/252 日结果全部来自日频残差滚动标准差，并非日、月、季多频率估计。

增强版采用版本化并行升级。现有模型和 `outputs/cne6_reproduction/` 作为 `legacy` 基线保持可重现；新模型标记为 `enhanced_v2`，输出到 `outputs/cne6_enhanced_v2/`。只有端到端验收通过后，才在单独变更中考虑把 V2 提升为默认模型。

## 2. 目标与边界

### 2.1 目标

1. 使用申万 2021 一级行业及历史有效期，替换静态 Tushare 细行业。
2. 把风险暴露重构为 15 个风格因子和 49 个三级描述子，使用显式、可审计的固定配比。
3. 使用约束 WLS 同时估计国家、风格和全部有效行业因子收益，不再依赖每日变化的基准行业。
4. 实现 EWMA、Newey-West 和 Monte Carlo Eigenfactor 组成的共同因子协方差流水线。
5. 用日、月、季非重叠特异收益分别估计方差，再融合并做有限结构化收缩。
6. 在相同历史中证500股票池、日期和收益定义下重算并对比 legacy 与 enhanced_v2。

### 2.2 非目标

1. 不声称复现 MSCI 商业 CNE6 的专有描述子、权重或参数。
2. 本轮不把估计股票池扩展到全 A 股。
3. 不把风险因子按 Alpha IC 优化，也不以收益预测能力选择描述子权重。
4. 不用中信行业或 Tushare 静态细行业填补申万行业缺口。中信一级只做映射覆盖和一致性审计。
5. 不覆盖 legacy 产物，不在本轮自动切换生产默认版本。

## 3. 总体架构

```text
Tushare 申万2021分类和历史成分
    |
    v
按 in_date/out_date 构造 PIT 申万一级行业
    |
    v
历史当日中证500面板 + 申万一级行业
    |
    +--> 49个三级描述子 --> 固定配比 --> 15个风格暴露
    |
    v
国家 + 风格 + 行业约束WLS
    |
    +--> 日度共同因子收益 --> EWMA/NW --> Eigenfactor协方差
    |
    +--> 日度个股特异收益 --> 日/月/季估计 --> 融合特异风险
    |
    v
组合归因 + 动态验证 + legacy/V2对比报告
```

建议模块边界：

| 模块 | 职责 |
|---|---|
| `factors/data/cne6_industry.py` | 拉取、缓存、审计和 PIT 展开申万一级行业；生成中信对照审计 |
| `factors/risk/cne6_v2_spec.py` | 定义 V2 风格、描述子、公式、方向、窗口、配比和正交化规则 |
| `factors/risk/cne6_v2_exposures.py` | 计算、清洗、标准化、正交化并合成 V2 暴露 |
| `factors/risk/cne6_regression.py` | 保留 legacy 默认行为，新增显式的约束 WLS V2 入口和共享数值工具 |
| `factors/risk/eigenfactor_covariance.py` | 估计 EWMA/NW 基础矩阵并执行 Eigenfactor 调整 |
| `factors/risk/multifrequency_specific_risk.py` | 估计日/月/季特异方差、融合和结构化收缩 |
| `factors/risk/cne6_v2_pipeline.py` | 编排 V2 全流程、增量缓存、清单和失败恢复 |
| `factors/reports/cne6_v2_report.py` | 输出 legacy/V2 消融、风险预测和组合归因报告 |

新模块通过 DataFrame 和显式配置对象传递数据，不读取其他模块的内部全局变量。legacy 函数签名和默认参数保持不变。

## 4. 行业数据设计

### 4.1 正式口径

正式口径使用 Tushare `index_classify(level="L1", src="SW2021")` 返回的 31 个申万 2021 一级行业。对每个一级行业调用 `index_member_all` 获取包含 `ts_code`、`in_date`、`out_date` 和 `is_new` 的历史成员记录。

原始数据缓存为：

```text
data/raw/cne6_sw2021_l1_classify.parquet
data/raw/cne6_sw2021_l1_members.parquet
data/raw/cne6_citic_l1_members_audit.parquet
```

token 只从 `TUSHARE_TOKEN` 环境变量读取，不写入命令输出、文件、报告或缓存元数据。

### 4.2 PIT 规则

成员有效区间采用半开区间：

```text
in_date <= trade_date < out_date
```

`out_date` 为空时视为仍然有效。股票收益日 `t` 使用 `t-1` 日收盘后已知的行业和风格暴露，因此解释 `t` 日收益时不会使用当天收盘后才形成的信息。

以下情况直接使正式流水线失败并输出诊断明细：

1. 同一股票同一交易日匹配多个申万一级行业。
2. 同一股票的成员区间发生重叠。
3. 31 个一级行业分类表缺失或代码重复。
4. 有效回归样本的行业覆盖率低于 99%。

未匹配股票保留在缺失清单中，不用中信或静态行业混填。中信一级数据只统计股票覆盖、申万/中信映射交叉表和不一致股票，不进入正式回归。

## 5. 因子层级与描述子

### 5.1 层级

增强模型采用以下三级结构：

1. 一级：国家、行业、风格和特异风险块。
2. 二级：15 个风格因子。
3. 三级：49 个可计算描述子。

### 5.2 初始描述子和配比

所有权重均指三级描述子先完成方向处理、稳健标准化和指定正交化后的组合权重。

| 二级风格 | 三级描述子及初始权重 |
|---|---|
| Size | `log_total_mv` 1.00 |
| Nonlinear Size | `nonlinear_size_residual` 1.00 |
| Beta | `beta_252_ewma` 0.50，`beta_504_ewma` 0.30，`downside_beta_252` 0.20 |
| Residual Volatility | `dastd_252` 0.50，`hsigma_252` 0.30，`cmra_12m` 0.20 |
| Liquidity | `stom_21` 0.30，`stoq_63` 0.30，`stoa_252` 0.25，`amihud_63` 0.10，`turnover_stability_63` 0.05 |
| Momentum | `rstr_12m_ex_1m` 0.50，`rstr_6m_ex_1m` 0.30，`momentum_ewma_252` 0.20 |
| Short Reversal | `reversal_5d` 0.40，`reversal_21d` 0.60 |
| Book-to-Price | `book_to_price` 1.00 |
| Earnings Yield | `earnings_yield` 0.40，`cashflow_to_price` 0.35，`forecast_earnings_yield` 0.25 |
| Growth | `revenue_yoy` 0.25，`net_profit_yoy` 0.25，`eps_growth` 0.20，`roe_growth` 0.15，`asset_turnover_yoy` 0.15 |
| Profitability | `roe_ttm` 0.20，`roa_ttm` 0.15，`gross_margin_ttm` 0.15，`operating_margin_ttm` 0.10，`cashflow_to_profit` 0.15，`accrual_quality` 0.15，`earnings_stability` 0.10 |
| Investment Quality | `asset_growth` 0.35，`capex_growth` 0.25，`inventory_growth` 0.20，`working_capital_growth` 0.20 |
| Leverage | `debt_to_assets` 0.35，`book_leverage` 0.30，`market_leverage` 0.25，`inverse_interest_coverage` 0.10 |
| Dividend Yield | `dv_ttm` 0.70，`dv_ratio` 0.30 |
| Sentiment | `analyst_report_count_90` 0.15，`analyst_org_count_180` 0.15，`analyst_rating_score_180` 0.20，`analyst_target_upside_180` 0.25，`analyst_eps_revision_180` 0.25 |

公式口径固定如下：

1. `log_total_mv` 为总市值自然对数；`nonlinear_size_residual` 为标准化 Size 三次项对 Size 做截面 WLS 后的残差。
2. 252/504 日 Beta 使用个股收益与中证500收益的 EWMA 协方差除以市场 EWMA 方差，半衰期分别为 63/126 日；Downside Beta 仅使用市场收益为负的 252 日样本。
3. DASTD 为 252 日个股收益 EWMA 标准差，半衰期 42 日；HSIGMA 为 252 日 Beta 回归残差标准差；CMRA 为最近 12 个完整自然月累计对数超额收益路径的最大值减最小值。
4. STOM 为 21 日换手率之和的对数，STOQ/STOA 分别为最近 3/12 个完整月月度换手率之和均值的对数；Amihud 为 63 日 `abs(return) / amount_rmb` 均值的对数并使用负方向；换手稳定性为 63 日换手率标准差并使用负方向。
5. 12-1 月、6-1 月动量都排除最近 21 个交易日；`momentum_ewma_252` 对同一区间日对数收益使用 126 日半衰期。5/21 日反转为对应累计收益的负值。
6. Book-to-Price、Earnings Yield 分别使用 `1 / pb`、`1 / pe_ttm`；非正 PB 或非正 PE 记为缺失。现金流收益率使用 TTM 经营现金流除以统一为人民币元的总市值。预测盈利收益率使用截至估计日最近 180 日 FY1 每股盈利一致预期除以前复权收盘价。
7. 收入、净利润、EPS、ROE、资产周转、总资产、资本开支、存货和营运资本增长均按最新已公告财报与上年同季度比较，不用日频行 `shift(252)` 代替财务同比。资本开支、存货和营运资本增长在 Investment Quality 中使用负方向。
8. Profitability 使用最新 PIT TTM 或财务指标：ROE、ROA、毛利率、营业利润率、经营现金流/净利润、经营应计/总资产，以及最近 8 个已公告季度资产收益率标准差的负值。
9. Book Leverage 为总资产/股东权益，Market Leverage 为总负债/(总负债+总市值)，Inverse Interest Coverage 为财务费用/营业利润；分母不为正或接近零时记为缺失。
10. Sentiment 五个描述子沿用当前 PIT 研报口径：报告数 90 日，机构数、评级、目标价空间和 EPS 修正 180 日；报告在 `report_date + 1` 起可用。

金额字段在描述子计算前统一为人民币元，并在元数据记录 Tushare 原始单位和换算倍率。所有滚动交易日描述子都要求连续交易日检查，不跨停牌或缺失行情直接桥接收益。

描述子方向是配置的一部分。风险因子不把“高暴露”等同于“优质股票”，但同一描述子的方向必须跨版本稳定。Investment Quality 中增长越激进的指标使用负方向，使高暴露表示投资更保守；Leverage 中高暴露表示高杠杆。

### 5.3 预处理、正交化和缺失

每个交易日按以下固定顺序处理：

1. 只使用当日可得的 PIT 原始字段计算描述子。
2. 使用截面中位数加减 5 倍 MAD 去极值；MAD 为零时退化为 1%/99% 分位数截断。
3. 使用市值平方根权重中心化，再缩放到等权截面标准差为 1。
4. 按配置执行市值平方根 WLS 正交化：Nonlinear Size 对 Size；Residual Volatility 对 Size 和 Beta；Liquidity 对 Size。
5. 按固定权重合成风格暴露并再次稳健标准化。

单只股票某风格的有效描述子权重不少于计划权重的 60% 时，才对剩余有效权重重新归一化。低于 60% 时该风格记为缺失。输出必须保留有效权重、有效描述子数量和缺失原因。

描述子在日期 `t` 进入风格合成前，还需满足截至 `t-1` 最近 252 个交易日的有效回归样本截面覆盖率不低于 70%；预热期不足 252 日时至少需要 126 日历史。未通过门槛的描述子在 `t` 日视为不可用，再按上面的 60% 计划权重规则判断是否允许对剩余描述子重新归一化。覆盖判断只使用当时已知数据，状态和原因写入每日诊断。

## 6. 日度因子收益回归

### 6.1 收益对齐

正式 V2 使用 `lagged_exposure_1d`：用 `t-1` 收盘后已知的行业、风格和市值解释 `t` 日个股收益。因子收益日期记为 `t`。这避免同日收盘价格同时进入暴露和被解释收益。

### 6.2 约束 WLS

回归包含：

```text
country + 15 styles + all active SW2021 L1 industries
```

样本为历史当日中证500成分股，基础权重为 `sqrt(total_mv)`。行业因子施加当日行业总市值权重和为零约束：

```text
sum_k industry_cap_weight_k * industry_return_k = 0
```

实现使用稳定的零空间或 KKT 约束最小二乘，并对秩、约束残差和条件数做诊断。正式输出展示全部有效行业因子收益，不再把某个行业隐式设为零。没有样本股票的行业当日不可识别，保留为空并从当日独立估计基底中移除，不能填零进入协方差。

## 7. Eigenfactor 共同因子协方差

### 7.1 独立基底

国家因子、全部行业因子和行业加权和为零约束共同存在时，展示基底天然奇异。协方差先在满秩独立基底中估计：国家因子、15 个风格因子和最多 30 个行业对比因子。完成估计后再通过确定性线性变换还原为 31 个行业的展示基底。

### 7.2 基础矩阵

每个估计日使用最近 504 个有效交易日因子收益，最少需要 252 日。基础协方差使用 EWMA，半衰期 90 个交易日。随后用 Newey-West 滞后 2 日修正自相关。基础矩阵不对缺失因子收益填零；因子有效历史不足时不进入当日基底，并记录缺失状态。Newey-West 后先对称化并把负特征值截断到零，形成可采样的 PSD 基础矩阵；截断前最小特征值和被截断的负特征值质量写入诊断。

### 7.3 Monte Carlo Eigenfactor 调整

Eigenfactor 调整每个自然月最后一个交易日收盘后重估，从下一个交易日起生效；月内向前沿用最近已经生效的调整，不能回填未来月末结果。初始模拟次数为 500，随机种子由固定全局种子和估计日期共同生成。

对月末基础矩阵 `C`：

1. 从以 `C` 为真实协方差的多元正态分布生成与有效历史等长的模拟因子收益。
2. 对每组模拟收益使用相同 EWMA/Newey-West 参数重估协方差并排序特征值。
3. 在模拟样本特征向量方向上，比较 `C` 给出的真实方差与模拟估计特征值。
4. 按特征值排序位置汇总方差偏差比，并应用到原始特征值。
5. 方差调整倍率限制在 `[0.5, 2.0]`，超限次数写入诊断。
6. 用原始特征向量和调整后特征值重建矩阵。

最终矩阵做对称化，并把小于数值容差的特征值截断到零。输出同时保留 EWMA/NW 基础矩阵、Eigenfactor 调整矩阵、特征值、调整倍率、模拟参数和基底变换元数据。

Ledoit-Wolf 继续作为 legacy 对比和显式回退。如果 Eigenfactor 输入秩不足、模拟失败或输出不满足 PSD，V2 当日使用最近一期有效 Eigenfactor 矩阵；若不存在历史有效矩阵，才回退到同日 Ledoit-Wolf。所有回退必须带原因字段，不能静默发生。

## 8. 日/月/季多频特异风险

### 8.1 三个频率

V2 使用约束 WLS 的个股日度特异收益作为唯一输入，分别估计：

| 频率 | 聚合与窗口 | 半衰期 | 最低历史 |
|---|---|---|---|
| 日频 | 252 个交易日日收益 | 63 个交易日 | 126 日 |
| 月频 | 非重叠自然月特异收益，36 个月 | 12 个月 | 18 个月 |
| 季频 | 非重叠自然季度特异收益，20 个季度 | 8 个季度 | 8 个季度 |

月度和季度特异收益按日度特异收益求和，保持横截面收益分解的加法口径。月、季方差分别除以该估计窗口内加权平均的实际交易日数，转换为日方差。部分月份或季度只有达到预期交易日的 60% 才进入估计。

### 8.2 融合与结构化收缩

三个日方差的初始融合权重为：

```text
daily 0.50 + monthly 0.30 + quarterly 0.20
```

只在可用频率之间重新归一化，并输出实际权重。融合发生在方差层，不能直接平均波动率。

融合后向“申万一级行业 x 对数市值五分组”的截面中位数先验做有限收缩。收缩可靠度由个股日频等效有效样本数决定，达到 252 个等效交易日时个股权重为 1；历史更短时个股权重线性下降，但最低保留 0.20。若结构化分组少于 5 只股票，依次退化到申万一级行业先验和全样本先验。

输出包含三个频率的日方差、融合前后方差、年化波动率、实际频率权重、等效样本数、结构化先验层级和回退状态。

## 9. 产物与版本清单

V2 主要产物：

```text
data/processed/cne6_csi500_daily_panel_v2.parquet
outputs/cne6_enhanced_v2/industry_mapping_audit.csv
outputs/cne6_enhanced_v2/descriptor_metadata.csv
outputs/cne6_enhanced_v2/descriptor_exposures.parquet
outputs/cne6_enhanced_v2/style_exposures.parquet
outputs/cne6_enhanced_v2/factor_returns.csv
outputs/cne6_enhanced_v2/specific_returns.parquet
outputs/cne6_enhanced_v2/factor_covariance_base.parquet
outputs/cne6_enhanced_v2/factor_covariance_eigenfactor.parquet
outputs/cne6_enhanced_v2/specific_risk_multifrequency.parquet
outputs/cne6_enhanced_v2/model_manifest.json
outputs/cne6_enhanced_v2/validation/
docs/cne6_enhanced_risk_model_v2_2026-08-27.md
```

`model_manifest.json` 记录模型版本、数据日期、行业版本、描述子配置哈希、回归口径、协方差参数、特异风险参数、代码提交和全部产物路径。数据缓存和中间产物均不得包含 token。

## 10. 异常处理

1. 拉取阶段区分认证失败、权限不足、限频、空响应和网络失败。已有完整缓存时允许离线重算；缓存不完整时禁止把旧缓存误报为最新数据。
2. 每个阶段先写临时产物，通过行数、主键唯一性和日期覆盖检查后再发布到正式 V2 路径。
3. PIT 行业冲突、回归秩不足和约束残差超限属于硬失败。Eigenfactor 候选矩阵非 PSD 时按第 7.3 节显式回退；经过回退逻辑后准备正式发布的最终协方差仍非 PSD 才属于硬失败。
4. 单个描述子覆盖不足、个股某频率历史不足属于可诊断降级，不应终止全部流水线。
5. Eigenfactor 回退、结构化先验退化和有效权重重分配必须出现在行级或日期级诊断中。

## 11. 测试设计

### 11.1 单元测试

1. 申万成员 `in_date` 当日生效、`out_date` 当日失效，区间不会重叠。
2. 描述子公式、方向、固定权重和有效权重阈值正确。
3. Nonlinear Size、Residual Volatility 和 Liquidity 正交残差满足加权内积容差。
4. 约束 WLS 的行业收益市值加权和接近零，并能还原拟合值和残差。
5. 月度、季度聚合互不重叠，不使用未完成的未来区间。
6. 三频方差先转换为日方差再融合，缺频时权重正确重归一化。
7. Eigenfactor 在固定种子下可重复，输入和输出矩阵均为对称 PSD。

### 11.2 集成和无未来函数测试

1. 用 `t-1` 行业、风格和市值解释 `t` 收益。
2. 月末 Eigenfactor 调整只能从下一个交易日起被使用。
3. 月频、季频特异风险不能读取目标日所在未完成月或季度的未来残差。
4. legacy 和 V2 使用相同历史中证500股票池、交易日和个股收益定义。
5. 组合归因能消费完整 31 行业展示基底和多频特异方差。

### 11.3 环境

当前 `venv` 有 Tushare 等运行依赖但缺少 pytest，系统 Python 有 pytest 但缺少 Tushare。实现阶段应补齐可重复的测试依赖声明，使同一个解释器能够完成测试收集和执行，不能继续依赖临时 `PYTHONPATH` 拼接。

## 12. 端到端验证与验收

### 12.1 消融顺序

报告至少包含以下同口径版本：

1. legacy 原始模型。
2. legacy 因子体系 + 申万一级行业和约束 WLS。
3. 第 2 项 + V2 描述子和固定配比。
4. 第 3 项 + EWMA/Newey-West/Eigenfactor 协方差。
5. 第 4 项 + 日/月/季多频特异风险，即完整 enhanced_v2。

该顺序用于区分行业、因子层、协方差和特异风险各自的影响。

### 12.2 硬性门槛

1. 有效回归样本申万一级行业覆盖率不低于 99%。
2. 日度回归成功率不低于 99%。
3. V2 预测方差有效率不低于 99%。
4. 协方差最小特征值不小于 `-1e-12`。
5. 月频和季频特异风险在各自预热期后有效覆盖率不低于 90%。
6. 描述子实际权重、缺失频率、结构化先验和协方差回退全部可追溯。

### 12.3 统计比较

在严格时序映射下比较：

1. 每日回归参数数目、调整 R2、条件数和残差分布。
2. 共同因子、风格、行业和特异风险块的预测/实现方差偏差。
3. 绝对 log 误差和 QLIKE。
4. 预测跟踪误差与实现跟踪误差的偏差、相关性和分年度稳定性。
5. 极端市场阶段的风险低估比例和尾部误差。

V2 不要求每个历史子区间都优于 legacy，也不因单一指标改善就自动切换默认版本。默认版本切换至少要求：硬性门槛全部通过；在双方均有效的样本外日期上，总风险 QLIKE、总风险绝对 log 误差和预测/实现跟踪误差绝对 log 误差均不超过 legacy 的 105%；日均行业参数数目和回归条件数中位数均低于 legacy。满足这些非劣与稳定性门槛后，才提交默认版本切换建议。

## 13. 实施顺序

1. 建立申万/中信行业拉取、PIT 映射和覆盖审计。
2. 增加 V2 描述子配置、计算、正交化和固定配比。
3. 实现申万行业约束 WLS，重算共同因子和特异收益。
4. 实现独立基底、EWMA/Newey-West 和 Eigenfactor 调整。
5. 实现日/月/季特异风险、融合和结构化收缩。
6. 串联版本化流水线、产物清单和恢复机制。
7. 完成单元测试、集成测试、历史重算、消融和正式报告。

行业和回归必须先于协方差，因为行业基底变化会使旧因子收益和协方差全部失效；特异风险必须在新回归残差生成后估计。
