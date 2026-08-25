# 股指期货监测驾驶舱修改记录

本文件记录监测驾驶舱的页面、计算口径、数据回填和验证变化。不得记录 Tushare token、个人账户、Tailnet 地址或其他连接凭据。

## 2026-08-07 - 交互优化实施基线

### 目标

- 将原始表格型页面重构为“今日摘要、配对比较、研究下钻”工作流；
- 保留证据质量边界，不输出自动交易指令或未经校准的概率；
- 建立可独立于聊天记录的修改追溯。

### 设计与计划

- `docs/superpowers/specs/2026-08-07-index-futures-dashboard-ux-optimization-design.md`；
- `docs/superpowers/specs/2026-08-07-index-futures-dashboard-ux-optimization-implementation-plan.md`。

### 实施前数据基线

- 正式缓存区间：2017-01-03 至 2026-08-06；
- 真实期货合约日行情：31,868 行；
- 指数日行情：9,316 行；
- ETF 日行情：23,865 行；
- ETF 份额：23,994 行；
- 基差表：31,868 行；
- ETF 信号：2,329 行；
- 分红事件：65,463 行，覆盖最新四指数 1,800 只成分股；
- IM 首日：2022-07-22。

### 实施前质量状态

- 原始期货、主力映射、指数和 ETF 行情：Tushare 原始字段，B 级；
- 指数权重：月度快照 proxy，C 级；
- 最新含分红基差：部分已披露事件口径，C 级；
- 历史含分红预测：未接入，D 级；
- ETF 分类和信号：人工审核与未披露参数口径，C 级。

### 实施前验证

- 监测相关测试：27 passed；
- Streamlit 两个页面 AppTest：0 exceptions；
- 本地健康检查：HTTP 200。

### 已知限制

- 缺少历史日度精确指数权重；
- 缺少卖方历史分红预测快照；
- ETF 风险分类存在人工维护和存活偏差；
- 尚未接入指数行业、主题或 AI 相对暴露；
- 当前目录没有 Git 元数据，无法以 commit 标识版本。

### 实施进度

- 2026-08-07：完成设计、详细实施计划和回归基线确认；
- 2026-08-07：完成配对基差、今日摘要、派生管线和四页驾驶舱实现；
- 2026-08-07：完成正式回填、数据审计和程序化页面验证。

### 计算口径变化

- 新增 `raw_basis_quality`，将原始基差质量与含分红基差质量分开；
- 新增 `raw_historical_percentile`，历史原始基差使用独立 PIT 分位；
- 新增有方向配对定义：`long_raw_annualized_basis - short_raw_annualized_basis`；
- 配对仅使用相同 `tenor_rank` 且到期日差不超过 7 天的真实合约；
- 配对分位 `<=20` 为基差结构有利，`>=80` 为不利，其余为中性；
- ETF `-1` 映射风险偏好偏强，`+1` 映射偏弱；成交与换手冲突显示混合；
- 新增最近一致信号日期和持续交易日，避免把延续状态误认为当日新信号；
- 页面只输出支持有限、证据混合或证据不足，不输出交易指令和概率。

### 新增文件

- `factors/monitoring/pair_analysis.py`；
- `factors/monitoring/decision_summary.py`；
- `factors/dashboard/components/formatters.py`；
- `factors/dashboard/components/status_badge.py`；
- `factors/dashboard/views/overview.py`；
- `factors/dashboard/views/pair_compare.py`；
- `tests/test_pair_analysis.py`；
- `tests/test_decision_summary.py`；
- `tests/test_dashboard_formatters.py`；
- `tests/test_dashboard_overview.py`；
- `tests/test_dashboard_pair_compare.py`；
- `tests/test_dashboard_research_views.py`。

### 修改文件

- `factors/monitoring/pipeline.py`；
- `factors/monitoring/cli.py`；
- `factors/monitoring/coverage.py`；
- `factors/dashboard/app.py`；
- `factors/dashboard/views/basis.py`；
- `factors/dashboard/views/etf_risk.py`；
- `tests/test_dividend_basis.py`；
- `tests/test_monitoring_backfill.py`；
- `tests/test_monitoring_coverage.py`；
- `tests/test_dashboard_entrypoint.py`；
- `README.md`。

### 最终回填

- 日期探测：2026-08-07 期货 16 行、ETF 行情 12 行、ETF 份额 0 行；
- 统一正式截止日：2026-08-06；
- 历史范围：2017-01-03 至 2026-08-06；
- 期货合约日行情：31,868 行；
- 主力映射：7,967 行；
- 指数日行情：9,316 行；
- 指数权重快照：212,750 行；
- ETF 日行情：23,865 行；
- ETF 份额：23,994 行；
- 今日摘要：7,967 行；
- 有方向配对历史：79,416 行，12 个方向；
- 数据覆盖：8/8 个数据集可用；
- 公司数据库缺口：4 项。

### 最终数据审计

- IM 第一条记录：2022-07-22；
- 无 `IF.CFX`、`IFL1.CFX` 等合成行情；
- 主力映射重复键：0；
- 今日摘要重复键：0；
- 配对历史重复键：0；
- 到期差超过 7 天却输出有效状态：0；
- ETF 风险型和低风险型分组从 2017 年首个交易日起均有记录；
- validation 保留 development、validation、holdout 三个阶段；
- 质量保持：今日摘要 C、配对原始基差 B、权重与 ETF 分类仍为 C。

### 最终验证

- 14 个监测测试文件：62 passed；
- 四个页面使用正式缓存的 Streamlit AppTest：0 exceptions；
- 正式服务健康检查：HTTP 200；
- Playwright Python 包下载速度长期约 44 kB/s，在 40.4/47.7 MB 时终止，因此未完成桌面和移动端浏览器截图；不得将该项记录为已验证。

### 尚需公司数据库补充

- 2017 年以来四指数日度精确权重；
- 历史时点卖方分红预测快照；
- 历史 ETF 分类、转型和退市信息；
- 2026-07-29 Wind/报告同口径验证快照。
