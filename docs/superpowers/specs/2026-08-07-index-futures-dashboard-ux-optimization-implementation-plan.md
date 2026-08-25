# 股指期货监测驾驶舱交互优化实施计划

> 对应设计：[2026-08-07-index-futures-dashboard-ux-optimization-design.md](2026-08-07-index-futures-dashboard-ux-optimization-design.md)

## 0. 实施原则

- 严格按“计划 -> 实现 -> 测试 -> 最终回填 -> 修改记录”执行；
- 金融计算全部位于 `factors/monitoring/`，Streamlit 页面只展示结构化结果；
- 先写失败测试，再实现最小行为；
- 历史比较只使用当日及以前数据，禁止未来信息进入分位数；
- 原始基差、部分含分红基差和不可用分红必须分开；
- ETF 只表示整体风险环境，不解释指数之间的行业或主题差异；
- C/D 级证据不得通过页面组合提升为高置信度；
- 不修改首期两个模块以外的数据范围；
- 每个阶段同步维护 `CHANGELOG_MONITORING.md`。

## 1. 建立修改记录和回归基线

### 文件

- 新增 `CHANGELOG_MONITORING.md`；
- 新增 `tests/test_dashboard_formatters.py`；
- 保留现有 8 个监测测试文件作为回归基线。

### 步骤

1. 在修改记录中写入优化前基线：数据截止日、核心文件行数、27 项既有测试、当前两个页面和已知限制；
2. 记录本轮设计文档、实施计划和需求来源；
3. 运行既有监测测试，确认修改前基线仍为 27 项通过；
4. 不把仓库中依赖 `torch`、`matplotlib` 等未安装包的无关测试计入本模块验收。

### 验收

- 修改记录包含日期、范围、数据状态、测试结果和缺口；
- token、Tailnet 地址和个人账户信息不进入记录；
- 既有监测测试全部通过。

## 2. 实现配对基差纯计算层

### 文件

- 新增 `factors/monitoring/pair_analysis.py`；
- 新增 `tests/test_pair_analysis.py`。

### 数据合同

输入为现有 `basis_table`，至少包含：

```text
trade_date, product, ts_code, expiry_date, days_to_expiry,
tenor_rank, raw_annualized_basis, annualized_basis,
dividend_source, basis_quality, is_main
```

输出至少包含：

```text
trade_date, long_product, short_product, tenor_rank,
long_contract, short_contract, long_expiry_date, short_expiry_date,
expiry_gap_days, long_raw_annualized_basis,
short_raw_annualized_basis, pair_basis_spread,
pair_historical_percentile, pair_structure_status,
pair_quality, point_in_time
```

### 测试先行

1. 多 A/空 B 的差值必须为 `A - B`；
2. 交换方向后差值取反，状态按新方向重新计算；
3. 只匹配同日、同 `tenor_rank` 的不同品种；
4. 到期日差超过 7 天时不生成有效比较；
5. PIT 分位不受未来极值影响；
6. 分位 `<=20` 为有利、`>=80` 为不利，中间为中性；
7. 输入 D 级或缺失基差时输出不足，不填零；
8. 主键 `trade_date + long_product + short_product + tenor_rank` 唯一；
9. 不允许同品种多空组合。

### 实现

- `match_pair_legs()`：匹配同期限真实合约并校验到期差；
- `add_pair_point_in_time_percentile()`：按有方向的品种对和期限计算历史分位；
- `classify_pair_structure()`：映射有利/中性/不利/不足；
- `build_pair_basis_history()`：生成四品种所有有方向组合的完整历史表。

### 验收

- 4 个品种形成 12 个有方向组合；
- 方向交换测试通过；
- 无未来泄漏、无重复主键、无合成合约。

## 3. 实现今日摘要纯计算层

### 文件

- 新增 `factors/monitoring/decision_summary.py`；
- 新增 `tests/test_decision_summary.py`。

### 数据合同

输入为 `basis_table`、`etf_signals` 和可选覆盖率表。输出为每个交易日、每个品种一行的摘要：

```text
trade_date, product, main_contract, expiry_date,
raw_annualized_basis, adjusted_annualized_basis,
basis_percentile, basis_status, dividend_status,
volume_status, turnover_status, risk_appetite_status,
four_factor_status, last_consensus_date, signal_age_days,
concentration_warning, basis_quality, signal_quality,
overall_evidence_status, evidence_reasons
```

### 测试先行

1. ETF `-1` 映射风险偏好偏强，`+1` 映射偏弱；
2. 成交量与换手率冲突时为混合；
3. 两者一致时记录最近一致日期；
4. 冲突期间正确累计信号持续交易日数；
5. 质量 D 或关键字段缺失时为证据不足；
6. C 级信号不能输出高置信度；
7. 部分分红保持 C，未接入分红保持 D；
8. 非交易日解析到不晚于目标日期的最近交易日，并返回实际日期；
9. 没有可用历史日期时显式失败，不使用未来日期。

### 实现

- `resolve_as_of_date()`：确定实际采用交易日；
- `classify_basis_status()`：映射便宜/中性/偏贵/不足；
- `classify_risk_appetite()`：组合成交、换手和份额状态；
- `signal_freshness()`：计算一致信号日期和持续时间；
- `build_decision_summary()`：生成逐日逐品种摘要与可审计原因。

### 验收

- 每日最多四个品种且主力唯一；
- 所有状态都有文字原因和质量字段；
- 不生成买入、卖出或概率字段。

## 4. 接入派生管线和 CLI

### 文件

- 修改 `factors/monitoring/pipeline.py`；
- 修改 `factors/monitoring/cli.py`；
- 修改 `factors/monitoring/coverage.py`；
- 新增或扩展 `tests/test_monitoring_backfill.py`、`tests/test_monitoring_coverage.py`。

### 实现

1. 新增 `build_dashboard_derivatives_from_cache()`；
2. 在基础 `basis_table` 和 `etf_signals` 完成后生成：

   ```text
   decision_summary_<start>_<end>.parquet
   pair_basis_<start>_<end>.parquet
   ```

3. `build --mode all` 和 `backfill --mode all` 自动生成两张派生表；
4. 只运行 futures 或 ETF 时，不把缺少另一输入的摘要标记完整；
5. 覆盖率输出增加两个派生数据集，但质量不得高于其最低输入质量；
6. 构建输出打印行数、日期范围和不足状态数量。

### 验收

- 派生缓存可重复构建且结果稳定；
- 不需要额外 Tushare 接口；
- 缺少输入时显式显示依赖缺失；
- 全区间数据无重复主键。

## 5. 实现统一格式和状态组件

### 文件

- 新增 `factors/dashboard/components/formatters.py`；
- 新增 `factors/dashboard/components/status_badge.py`；
- 扩展 `tests/test_dashboard_formatters.py`。

### 实现

- 中文列名字典；
- 百分比、基差点数、日期、整数和缺失值格式；
- A/B/C/D 质量标签；
- 有利、中性、不利、混合、不足、过期等状态映射；
- 状态同时输出文字和颜色类别，不能只依赖颜色；
- 页面展示使用格式化副本，下载仍保留原字段和精度。

### 验收

- `NaN/NaT` 显示为“不可用”，不显示为 0；
- 红色只用于异常和风险；
- 中文长列名可换行；
- 格式化不改变源 DataFrame。

## 6. 实现今日摘要页

### 文件

- 新增 `factors/dashboard/views/overview.py`；
- 新增 `tests/test_dashboard_overview.py`。

### 实现

1. 紧凑数据状态条：实际日期、回填时间、过期状态和总体质量；
2. 风险偏好状态和主要冲突；
3. 四品种状态表：合约、原始/部分含分红基差、分位、ETF 状态和质量；
4. 证据矩阵：基差、资金、价格修正、未接入暴露；
5. 最近 20 个交易日状态图；
6. 原始字段和质量原因放入折叠区；
7. 非交易日必须显示“选择日期”和“实际采用日期”。

### 验收

- 默认打开首页即可完成当日概览；
- 页面不出现交易建议或高置信度措辞；
- 四品种缺失时页面仍稳定，不产生空白布局；
- Streamlit AppTest 无异常。

## 7. 实现配对比较页

### 文件

- 新增 `factors/dashboard/views/pair_compare.py`；
- 新增 `tests/test_dashboard_pair_compare.py`。

### 实现

1. 使用选择器确定多头、空头、期限和日期；
2. 禁止相同多空品种；
3. 展示两腿真实合约、到期差和基差口径；
4. 展示当前配对差值、PIT 分位和有利/中性/不利状态；
5. 绘制稳定高度的历史差值图及 20/80 分位参考；
6. 展示 ETF 整体环境，但将“相对主题暴露”固定标为未接入；
7. 分栏显示有利、反向和缺失证据；
8. 下载选定配对的完整历史和质量字段。

### 验收

- 默认组合使用不同品种；
- 调换多空腿后数值和状态同步变化；
- 不可比期限显示不足而不是选择其他合约；
- 页面结论明确限定为基差结构，不声称预测相对收益。

## 8. 优化两个研究下钻页

### 文件

- 修改 `factors/dashboard/views/basis.py`；
- 修改 `factors/dashboard/views/etf_risk.py`；
- 新增或扩展页面测试。

### 基差页

- 加入品种和期限筛选；
- 使用中文列名；
- 原始和部分含分红口径分层；
- 增加期限结构图和当前历史分位；
- 原始表、刷新审计和下载置于折叠区；
- 保留分红部分口径警告。

### ETF 页

- 顶部显示风险偏好状态，而非直接显示 `+1/-1`；
- 分开展示成交、换手、份额证据；
- 显示最近一致信号日期及持续天数；
- 展示四品种修正结果和触发原因；
- validation 表按阶段和品种格式化；
- 单 ETF 贡献和分类明细置于折叠区。

### 验收

- 原始字段仍可下载；
- 页面默认视图不需要阅读英文机器字段；
- C/D 和集中度警告仍可见。

## 9. 重构入口和缓存选择

### 文件

- 修改 `factors/dashboard/app.py`；
- 扩展 `tests/test_dashboard_entrypoint.py`。

### 实现

1. 导航顺序固定为“今日摘要、配对比较、基差与期限结构、ETF 风险偏好”；
2. 新增按文件名解析区间的缓存选择器；
3. 优先选择截止日最新、覆盖区间最长且所需列完整的文件；
4. 不再按修改时间选择小窗口文件；
5. 页面载入前检查输入文件并显示具体缺失路径；
6. 保留项目根目录 `sys.path` 启动修复；
7. 数据覆盖和公司数据库缺口下沉至折叠区。

### 验收

- 小窗口文件修改时间较新时仍选择完整正式缓存；
- 从 `/tmp` 运行入口不出现 `ModuleNotFoundError`；
- 四页切换均无异常。

## 10. 全面验证

### 单元和集成测试

运行所有监测相关测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/mnt/d/codefields/Factors/venv/lib/python3.11/site-packages \
python -m pytest -p no:cacheprovider \
  tests/test_monitoring_data_contracts.py \
  tests/test_index_futures_fetcher.py \
  tests/test_dividend_basis.py \
  tests/test_etf_risk_signals.py \
  tests/test_evidence_quality.py \
  tests/test_monitoring_backfill.py \
  tests/test_monitoring_coverage.py \
  tests/test_pair_analysis.py \
  tests/test_decision_summary.py \
  tests/test_dashboard_formatters.py \
  tests/test_dashboard_overview.py \
  tests/test_dashboard_pair_compare.py \
  tests/test_dashboard_entrypoint.py -q
```

### 页面验证

- 使用 Streamlit AppTest 依次打开四页；
- 启动真实服务并检查 `/_stcore/health` 返回 200；
- 使用浏览器分别检查 1440x900 和 390x844；
- 检查非空图表、固定高度、无重叠、无截断、无横向溢出；
- 检查数据缺失、过期、集中度和部分分红警告。

### 数据约束

- IM 第一日仍为 2022-07-22；
- 主力映射每品种每日唯一；
- 无 `IF.CFX`、`IFL1.CFX` 等合成行情；
- 配对表方向和主键唯一；
- ETF 两类风险桶持续可用；
- validation 保留三个阶段；
- 所有 C/D 证据不升级。

## 11. 最终回填和追溯记录

### 步骤

1. 先探测 Tushare 各核心接口的最新共同完整交易日；
2. 将探测结果以 `YYYYMMDD` 格式写入本次 shell 的 `MONITORING_END_DATE`，并在执行前校验变量非空；不得让部分数据集领先后伪装为整体最新；
3. 执行可断点续传的正式回填：

   ```bash
   test -n "$MONITORING_END_DATE"
   venv/bin/python -m factors.monitoring.cli backfill \
     --start 20170101 --end "$MONITORING_END_DATE" \
     --mode all \
     --classification-path factors/monitoring/etf_classification.csv \
     --with-dividends
   ```

4. 重新生成 `basis_table`、ETF 派生表、`decision_summary` 和 `pair_basis`；
5. 输出覆盖率及公司数据库缺口；
6. 复跑全部监测测试和页面验证；
7. 更新 README 的页面使用方式和回填命令；
8. 在 `CHANGELOG_MONITORING.md` 记录：
   - 实际截止交易日；
   - 所有核心表行数；
   - 质量分布；
   - 测试数量和结果；
   - 浏览器验证尺寸；
   - 尚缺公司数据库的数据。

### 最终验收

- 首页、配对页和两个研究页均读取正式全历史缓存；
- 页面显示的数据截止日与文件覆盖一致；
- 修改记录足以在不依赖聊天记录的情况下复盘本轮变更；
- Streamlit 服务可访问且健康检查为 200；
- 页面仍只提供可审计证据，不生成自动交易指令。
