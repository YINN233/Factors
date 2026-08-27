# CNE6 增强风险模型 V2 实现计划

日期：2026-08-27

对应设计：`docs/superpowers/specs/2026-08-27-cne6-enhanced-risk-model-v2-design.md`

## 实施原则

1. `outputs/cne6_reproduction/` 和 legacy 函数默认行为保持不变。
2. V2 使用独立入口、配置、面板和输出目录。
3. 每个阶段先增加最小失败测试，再实现，再执行相关回归测试。
4. 行业和回归先完成，之后才能估计共同因子协方差和特异风险。
5. token 只从环境变量或交互进程内存读取，不写入仓库和产物。
6. 全量重算前先用合成数据和短日期窗口通过数值、PIT 和产物合同测试。

## 阶段 0：测试环境与公共合同

修改或新增：

- `requirements-dev.txt`
- `tests/test_cne6_v2_industry.py`
- `tests/test_cne6_v2_exposures.py`
- `tests/test_cne6_v2_regression.py`
- `tests/test_eigenfactor_covariance.py`
- `tests/test_multifrequency_specific_risk.py`
- `tests/test_cne6_v2_pipeline.py`

任务：

1. 为现有虚拟环境补充可重复的 pytest 依赖声明。
2. 建立 V2 测试 fixture：交易日、历史成分、行业切换、财务公告、因子收益和特异收益。
3. 保持 `test_cne6_risk_model.py` 作为 legacy 回归测试，不把 V2 断言混入旧 smoke test。

验收：同一个解释器可以收集 legacy 和 V2 测试；初始 V2 测试因模块尚未实现而失败。

## 阶段 1：申万一级行业 PIT 数据层

新增：

- `factors/data/cne6_industry.py`
- `tests/test_cne6_v2_industry.py`

修改：

- `factors/data/cne6_builder.py`
- `factors/data/cne6_fetcher.py`

任务：

1. 拉取并缓存 `index_classify(level="L1", src="SW2021")`。
2. 逐个一级行业拉取 `index_member_all`，规范代码、名称和日期。
3. 可选拉取中信一级成分，只生成交叉审计，不进入正式字段。
4. 校验分类数、主键、重叠区间和空响应。
5. 实现半开区间 `in_date <= trade_date < out_date` 的 PIT 展开。
6. 为 V2 面板增加 `industry_sw_l1_code`、`industry_sw_l1_name`、`industry_source`。
7. 输出覆盖率、冲突、未匹配股票和申万/中信交叉表。

测试：

- 纳入日生效、剔除日失效。
- 同日多行业和重叠区间报错。
- 未匹配不使用静态行业回填。
- token、请求错误和缓存合同不泄露凭证。

验收：短日期中证500 V2 面板行业覆盖率达到 99%，且行业数量不超过 31。

## 阶段 2：49 个描述子与 15 个风格暴露

新增：

- `factors/risk/cne6_v2_spec.py`
- `factors/risk/cne6_v2_exposures.py`
- `tests/test_cne6_v2_exposures.py`

修改：

- `factors/data/cne6_builder.py`

任务：

1. 用不可变配置对象登记 49 个描述子的风格、公式标识、字段、方向、权重、窗口、半衰期和正交化对象。
2. 在构建器中补齐营业利润率、EPS 同比、资产/资本开支/存货/营运资本同比、账面和市场杠杆、利息保障等 PIT 字段。
3. 实现行情类滚动描述子，禁止跨缺失交易日桥接收益。
4. 实现财务类同季度同比，不使用日频 `shift(252)`。
5. 实现 MAD 去极值、市值平方根中心化、等权标准差缩放。
6. 实现 Nonlinear Size、Residual Volatility 和 Liquidity 的加权正交化。
7. 按固定权重和 60% 有效权重门槛合成风格暴露。
8. 实现截至 `t-1` 的 252 日覆盖门槛和每日诊断。

测试：

- 每组配置权重严格等于 1，描述子总数严格等于 49。
- 财务同比以报告期和公告可得日对齐。
- 金额单位换算一致。
- 正交残差加权内积接近零。
- 缺失权重为 59%/60% 时分别失败/通过。
- 暴露计算不读取目标日后的数据。

验收：合成面板全部 15 个风格列存在，覆盖和实际权重可追溯，legacy 暴露测试继续通过。

## 阶段 3：申万行业约束 WLS

修改：

- `factors/risk/cne6_regression.py`
- `tests/test_cne6_v2_regression.py`

任务：

1. 增加 V2 专用 `run_constrained_factor_return_regression`，不改变 legacy 默认入口。
2. 用 `t-1` 暴露解释 `t` 收益。
3. 构建 country、15 styles、全部当日有效申万一级行业设计矩阵。
4. 使用 `sqrt(total_mv)` 权重和行业市值加权收益和为零约束。
5. 使用零空间或 KKT 求解，输出完整行业收益、特异收益、约束残差、秩和条件数。
6. 不可识别行业保持缺失，不填零。

测试：

- 已知系数的合成面板能恢复因子收益。
- 行业收益市值加权和在容差内为零。
- 因子列顺序变化不改变结果。
- 行业切换和收益对齐无未来函数。
- 秩不足和覆盖不足显式失败。

验收：合成测试通过；短样本日度回归成功率达到 99%，行业参数显著少于 legacy。

## 阶段 4：EWMA/Newey-West/Eigenfactor 协方差

新增：

- `factors/risk/eigenfactor_covariance.py`
- `tests/test_eigenfactor_covariance.py`

任务：

1. 构建满秩行业对比基底及其展示基底变换。
2. 实现 504 日窗口、252 日最低历史、90 日半衰期 EWMA 协方差。
3. 实现 Newey-West lag 2 修正、对称化和基础 PSD 投影。
4. 实现 500 次固定种子的 Monte Carlo Eigenfactor 特征值调整。
5. 月末收盘后重估，从下一交易日起生效。
6. 实现倍率截断、最近有效矩阵和 Ledoit-Wolf 显式回退。
7. 以紧凑长表写出基础矩阵、最终矩阵、特征值和诊断。

测试：

- EWMA 权重和 Newey-West 玩具结果正确。
- 固定种子重复运行输出一致。
- 基础和最终矩阵对称、PSD。
- 月末结果不回填当月。
- 基底往返后的组合方差保持一致。
- 失败回退包含原因且不会静默填零。

验收：合成及短历史滚动测试通过，最终最小特征值不低于 `-1e-12`。

## 阶段 5：日/月/季多频特异风险

新增：

- `factors/risk/multifrequency_specific_risk.py`
- `tests/test_multifrequency_specific_risk.py`

任务：

1. 实现 252 日、半衰期 63 日的日频 EWMA 方差。
2. 聚合完整自然月和季度的非重叠特异收益。
3. 实现 36 月/12 月半衰期和 20 季/8 季半衰期方差。
4. 按实际交易日把月季方差转换为日方差。
5. 按 50%/30%/20% 融合有效频率并记录实际权重。
6. 实现申万一级行业 x 市值五分组先验及行业/全样本退化。
7. 按等效样本数在 0.20--1.00 间确定个股可靠度。

测试：

- 月季区间不重叠且未完成区间不进入估计。
- 频率换算和融合发生在方差层。
- 缺频权重正确重归一化。
- 小分组按规定退化。
- 目标日结果不读取之后的残差。

验收：预热期后月季有效覆盖率达到 90%，全部发布方差非负且有来源诊断。

## 阶段 6：版本化流水线、归因和报告

新增：

- `factors/risk/cne6_v2_pipeline.py`
- `factors/reports/cne6_v2_report.py`
- `tests/test_cne6_v2_pipeline.py`

修改：

- `factors/reports/cne6_portfolio_attribution.py`
- `factors/reports/cne6_dynamic_risk_calibration.py`

任务：

1. 串联 V2 面板、暴露、回归、协方差、特异风险和验证。
2. 每阶段先写临时文件，合同检查通过后再发布。
3. 生成包含配置哈希、数据日期、行业版本、代码提交和产物路径的 manifest。
4. 扩展归因消费者，使其读取完整申万行业展示基底和融合特异方差。
5. 生成五级消融和 legacy/V2 对比指标。
6. 输出正式 Markdown 报告和图表。

测试：

- pipeline 重跑能复用缓存且不会覆盖 legacy。
- 中途失败不会发布半成品。
- manifest 与文件和配置一致。
- 归因方差分解与直接矩阵计算一致。

验收：短日期端到端产物合同完整。

## 阶段 7：历史重算与最终验收

运行顺序：

```bash
venv/bin/python -m factors.data.cne6_industry --token-env TUSHARE_TOKEN
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage panel
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage exposures
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage regression
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage covariance
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage specific-risk
venv/bin/python -m factors.risk.cne6_v2_pipeline --start 20100101 --end latest --stage report
python -m pytest test_cne6_risk_model.py tests/test_cne6_v2_*.py tests/test_eigenfactor_covariance.py tests/test_multifrequency_specific_risk.py -q
```

最终检查：

1. 行业覆盖率、回归成功率和预测方差有效率均达到设计门槛。
2. 最终协方差 PSD，所有回退可追溯。
3. 月季特异风险覆盖率达到门槛。
4. 总风险 QLIKE、总风险绝对 log 误差和 TE 绝对 log 误差满足 105% 非劣标准。
5. 日均行业参数和回归条件数中位数低于 legacy。
6. 报告明确列出没有通过的统计门槛，不自动提升默认版本。

## 提交边界

建议按以下边界提交，便于审查和回退：

1. 申万一级行业 PIT 数据层与测试。
2. V2 描述子、配比和暴露测试。
3. 约束 WLS 与测试。
4. Eigenfactor 协方差与测试。
5. 多频特异风险与测试。
6. 流水线、历史产物合同和对比报告。
