# 股指期货监测驾驶舱实施计划

> 对应设计：[2026-08-06-index-futures-monitoring-dashboard-design.md](2026-08-06-index-futures-monitoring-dashboard-design.md)

## 0. 实施原则

- 首期只实现“分红/基差”和“ETF 风险偏好”两个模块。
- Tushare 是首选数据源；公司数据库通过同一标准化接口作为后备数据源。
- 原始字段、估算字段和 proxy 字段分开保存。
- 历史信号严格按 `as_of_date` 和公告日处理，禁止使用未来修订数据。
- 驾驶舱提供证据和交易表达比较，不自动下单。
- 每一步先加数据/指标测试，再接页面。

## 1. 数据合同和可用性审计

### 目标

建立 Tushare 请求、分块、缓存、覆盖率和字段血缘的共同规范。

### 计划新增

- `factors/data/index_futures_fetcher.py`
- `factors/data/etf_monitor_fetcher.py`
- `factors/monitoring/contracts.py`
- `factors/monitoring/data_audit.py`
- `tests/test_monitoring_data_contracts.py`

复用 `factors/data/fetcher.py` 的 Tushare 初始化、重试、日期分块和 Parquet 缓存模式。

### 要点

1. token 只从 `TUSHARE_TOKEN` 读取，不出现在代码、日志或缓存。
2. 请求按年份/月度/合约分块，避免单次约 2000/7000 行限制。
3. 原始缓存按接口和日期区间保存，重复请求可跳过。
4. 每次请求记录接口、参数、行数、字段、日期范围、耗时、错误类型和覆盖率。
5. 区分网络失败、权限失败、空表和字段缺失。
6. 为 `.CFX` 期货代码建立显示转换，但保留原始代码。

### 验收

- 合同测试识别重复主键、日期类型错误和字段类型错误；
- 一次探查生成数据可用性表；
- token 不出现在测试输出中。

## 2. 股指期货和指数行情层

### 数据接口

- `fut_basic(exchange='CFFEX')`
- `fut_daily(ts_code=...)`
- `fut_mapping(ts_code='IF.CFX'/'IH.CFX'/'IC.CFX'/'IM.CFX')`
- `index_daily`：`000016.SH`、`000300.SH`、`000905.SH`、`000852.SH`
- `trade_cal`

### 计划新增

- `factors/monitoring/futures_normalizer.py`
- `tests/test_futures_normalizer.py`

### 要点

1. 从 `fut_basic`读取到期日、上市日、乘数和品种。
2. 按合约下载日行情，不用连续合约替代真实合约。
3. 使用 `fut_mapping`生成每日主力映射，缺失时显示缺失，不用成交量猜测。
4. 校验行情日期落在上市日和到期日之间。
5. 保存结算价、成交量、成交额、持仓量和持仓变化。

### 验收

- IF/IH/IC 早期合约和 IM 上市后合约均能读回；
- 2026-07-29 主力合约与报告合约月份一致；
- 主力映射不会使用未来合约信息。

## 3. 分红事件和基差指标

### 数据接口

- `dividend`
- `income`、`forecast`、`express`（先探查权限和覆盖率，不假设可用）
- `index_weight`、`index_daily`、`fut_daily`

### 计划新增

- `factors/monitoring/dividend_events.py`
- `factors/monitoring/dividend_basis.py`
- `tests/test_dividend_basis.py`
- `tests/fixtures/guoxin_20260729_expected.csv`

### 要点

1. 以 `ann_date` 建立 point-in-time 分红事件；公告前不能使用最终除息日。
2. 将分红流程规范化为预案、决案、实施、已分红、不分红。
3. 首版用月度/调仓日权重结合个股每日价格重估日权重，并保存 `weight_method=monthly_reweighted_proxy`。
4. 已披露分红和预测分红分列保存。
5. 计算：

   ```text
   raw_basis = futures_close - index_close
   dividend_adjusted_basis = futures_close - index_close + expected_dividend_points
   annualized_basis = dividend_adjusted_basis / index_close * 365 / days_to_expiry
   ```

6. 生成期限结构、历史分位数和预测误差字段。
7. 对 2026-07-29 的 IH/IF/IC/IM 主力表格做容差核对；proxy 差异必须解释。

### 验收

- 公式测试覆盖正负基差、零分红和临近到期；
- 报告日期四个主力年化基差在明确容差内，或输出可解释差异；
- 使用未来除息日的测试失败。

## 4. ETF 资产池和风险偏好指标

### 数据接口

- `fund_basic(market='E')`，包含上市和退市状态；
- `fund_daily`；
- `fund_share`；
- `trade_cal`。

### 计划新增

- `factors/monitoring/etf_universe.py`
- `factors/monitoring/etf_risk_signals.py`
- `factors/monitoring/etf_classification.csv`
- `tests/test_etf_risk_signals.py`

### 要点

1. 过滤交易所上市的股票型 ETF，排除债券、货币、REITs 和非股票基金。
2. 建立规模型、行业型、主题型、策略型、风格型、未分类及有效起止日期。
3. 每条分类保存依据和人工审核状态。
4. 份额和行情按时间分块，处理上市、退市、更名和停牌。
5. 分别计算两组 ETF 的日均成交量、成交量变化率、成交量加权换手率和份额变化率。
6. 计算对应股指期货主力合约滚动收益。
7. 将 `volume_window`、`turnover_window`、`share_window`、`price_window`、`optimization_window` 放入配置；未披露参数不能称为精确复现。
8. 输出基础、份额过滤、行情过滤和四因子信号，不输出自动交易指令。

### 质量控制

- 分类变更必须有版本和生效日期；
- 单个 ETF 不得无提示地主导组信号；
- 成交量、成交额和换手率只计为同一资金证据组。

### 验收

- 510300 行情和份额能从 2017 年开始分块恢复；
- ETF 分组总量、份额和成交量可被单 ETF 明细复核；
- 历史日期的信号方向和计算过程可追溯。

## 5. 证据质量和历史验证

### 计划新增

- `factors/monitoring/evidence_registry.py`
- `factors/monitoring/walk_forward.py`
- `tests/test_point_in_time.py`
- `tests/test_evidence_quality.py`

### 要点

1. 每个指标登记来源、覆盖率、PIT 状态、proxy 状态、历史验证状态和质量等级。
2. 使用滚动样本外或扩展窗口验证，不以全样本回测作为唯一依据。
3. 至少报告上涨、下跌、震荡三类市场状态。
4. 小幅改变参数时结论不应完全反转。
5. 同时输出成本前、成本后、展期前、展期后结果。
6. 按资金、基差、价格、基本面等证据组去重。
7. 输出“支持/混合/不足”，不输出未经校准的上涨概率。

### 默认验证分段

- 2017-2022：开发和参数探索；
- 2023-2024：验证；
- 2025-最新：留出样本外；
- IM：从上市日开始，单独标注样本较短。

留出区间不能反向调参后仍称为样本外。

## 6. Streamlit 驾驶舱

### 计划新增

- `factors/dashboard/app.py`
- `factors/dashboard/pages/basis.py`
- `factors/dashboard/pages/etf_risk.py`
- `factors/dashboard/components/quality_badge.py`
- `tests/test_dashboard_data_contract.py`

### 分红/基差页

- as-of 日期、指数和合约筛选；
- 四个品种卡片；
- 含分红基差和年化贴水表；
- 期限结构曲线和历史分位数；
- 分红状态、预测/已披露拆分；
- 数据过期、覆盖率和 proxy 警示；
- CSV/Parquet 下载。

### ETF 风险偏好页

- ETF 分类覆盖率和审核状态；
- 两组 ETF 的成交、换手、份额和价格趋势；
- 参数面板和信号触发明细；
- 支持/反向证据和历史验证结果；
- 单 ETF 下钻；
- 信号不自动转成下单动作。

首版使用本地 Streamlit 启动，页面从缓存 Parquet 读取；数据刷新和页面查看分为两个命令，避免页面刷新重复调用 Tushare。

## 7. 公司数据库桥接

公司数据适配器必须输出与标准层相同的合同，至少支持：

- `index_weight_snapshot`：指数、成分、权重日期、权重；
- `dividend_event`：公告日、流程、分红金额、除息日；
- `etf_classification`：ETF、分类、生效起止日；
- 期货合约日行情和主力映射。

进入标准层后标记 `source=company_db`，保留原字段名和转换说明。

## 8. 端到端验收

1. Tushare 从 2017 年开始完成原始刷新和覆盖率审计；
2. 2026-07-29 报告关键表格可复核；
3. 基差、分红、ETF 信号可从页面下钻到原始记录；
4. 所有 proxy、缺失和过期数据可见；
5. PIT 测试阻止未来公告、未来权重和未来基金记录进入历史信号；
6. 样本外、成本后和展期后结果单独输出；
7. Tushare 失败时页面显示过期状态，不静默使用 0 或旧结论；
8. 会员持仓、两融、基本面和 AI 暴露仍保持未接入状态。

## 9. 实施顺序

```text
数据合同/审计
      ↓
期货/指数数据 ──→ 分红/基差指标 ──→ 基差页面
      ↓
ETF 资产池 ─────→ ETF 风险指标 ──→ ETF 页面
      ↓
证据质量/样本外验证 ───────────→ 两页统一质量提示
      ↓
公司数据库桥接（按缺口申请）
```

数据合同和 PIT 验收通过前不制作最终页面；证据质量测试通过前不把信号命名为高置信度策略。
