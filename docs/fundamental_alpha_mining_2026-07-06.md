# 基本面 Alpha 挖掘记录

日期：2026-07-06

## 目标

基于根目录四份中信建投“逐鹿”Alpha 报告和 `docs/alpha_reports_summary.md`，把当前日频量价挖掘框架扩展到基本面因子。核心要求是：

1. 基本面数据必须按 `ann_date/f_ann_date` 做点时一致对齐。
2. 财务指标先形成单季度、TTM、YOY、QOQ 等可解释口径。
3. 用受约束算子把盈利、成长、现金流、杠杆、营运效率、估值和市场确认指标批量连接。
4. 在 HS300 股票池上评估 train/valid/test 的 RankIC、覆盖率和去冗余结果。
5. 将可用公式、经济含义和使用方式沉淀为可复现文档。

## 报告框架复核

四份报告对本轮实现最关键的约束如下：

| 报告 | 对本轮实现的约束 |
|---|---|
| 专题二十三：基本面因子挖掘统一框架 | 财报数据必须先从报告期转换为公告日可见的 PIT 数据；基本面算子要控制量纲和语义；GP/LLM 生成的表达式仍需经过 IC、覆盖率、相关性筛选。 |
| 专题二十八：分钟因子模型 | 高频/日频市场确认可以作为基本面信号的辅助腿，但不应替代基本面主逻辑。 |
| 专题二十九：隔夜-日内异象 | A 股日频收益结构中存在隔夜/日内差异，量价确认类混合因子可使用成交额、换手率、日内强弱等日频信息。 |
| 专题三十：量价 X 基本面统一框架 | 因子生产线应覆盖数据、表达、搜索、评估和入库；基本面与量价字段合成前必须先统一到日频 PIT 面板。 |

## 实现计划

| 步骤 | 状态 | 说明 |
|---|---|---|
| 复核报告与现有框架 | 已完成 | 当前框架已有量价候选、IC 评估、去冗余筛选；缺少真实基本面下载、PIT 构建和基本面候选库。 |
| 基本面下载 | 已实现，待运行 | `factors.data.fetcher` 新增 `income/balancesheet/cashflow/fina_indicator` 下载，按公告日期月份分段缓存。 |
| PIT 构建 | 已实现，待运行 | `factors.data.fundamental_builder` 生成 `fundamental_daily.parquet` 和 `train_fundamental/valid_fundamental/test_fundamental`。 |
| 算子扩展 | 已实现，待测试 | 新增 `ts_delta/ts_rank/ts_slope/ts_rsquare/cs_winsorize/size_neutralize`。 |
| 候选扩展 | 已实现，待测试 | 新增质量、成长、现金流、安全、效率、估值、量价确认类基本面候选。 |
| 测试 | 待运行 | 覆盖 PIT 防未来函数、算子和候选因子冒烟测试。 |
| HS300 挖掘 | 待运行 | 使用 `--fundamental --factor-set fundamental` 跑 train/valid/test。 |

## 已实现候选因子

| 因子 | 表达式 | 经济含义 | 使用方式 |
|---|---|---|---|
| `quality_roe_ocf` | `rank(roe_ttm) + rank(cashflow_to_profit)` | 偏好 ROE 高且经营现金流能覆盖利润的公司，避免只看会计利润。 | 作为质量主因子，可与估值或低波动因子组合。 |
| `quality_roa_ocf` | `rank(roa_ttm) + rank(cashflow_to_profit)` | 用资产收益率替代权益收益率，降低高杠杆 ROE 的干扰。 | 适合与 `low_leverage_quality` 对照使用。 |
| `cash_profit_cover` | `rank(n_cashflow_act_ttm / net_profit_ttm)` | 经营现金流对净利润的覆盖度，衡量利润含金量。 | 可作为财报质量过滤器，低值公司降权。 |
| `low_accrual_to_assets` | `rank((n_cashflow_act_ttm - net_profit_ttm) / total_assets)` | 现金利润相对会计利润更强，代表应计压力低。 | 适合在财报披露季控制利润操纵风险。 |
| `gross_margin_quality` | `rank(gross_margin_ttm) + rank(gross_margin_yoy)` | 毛利率水平高且改善，代表定价能力或成本控制改善。 | 用于消费、医药、制造等毛利率有解释力的行业。 |
| `quality_growth_hmean` | `harm_mean(rank(revenue_yoy), rank(net_profit_yoy)) + rank(cashflow_to_profit)` | 收入和利润同步成长，并由现金流确认。 | 成长主因子，避免单边利润增速来自非经常或费用扰动。 |
| `ocf_growth_quality` | `rank(ocf_yoy) + rank(cashflow_to_profit)` | 经营现金流增长且现金转化质量高。 | 对周期和制造企业的订单兑现质量更敏感。 |
| `asset_turnover_improve` | `rank(asset_turnover_yoy)` | 资产周转率改善，代表经营效率提高。 | 可作为效率改善类辅助因子。 |
| `debt_cash_safety` | `rank(n_cashflow_act_ttm / total_liab) - rank(debt_to_assets)` | 现金流偿债能力强且杠杆低。 | 风险控制型基本面因子，适合弱市或信用风险环境。 |
| `low_leverage_quality` | `rank(roe_ttm) - rank(debt_to_assets)` | 盈利能力不依赖高杠杆。 | 与纯 ROE 因子搭配，降低杠杆暴露。 |
| `capex_efficiency` | `rank(revenue_yoy) - rank(capex_to_assets)` | 用较低资本开支取得较高成长。 | 用于识别轻资产或投资效率高的公司。 |
| `roe_value_pb` | `rank(roe_ttm) - rank(pb)` | 高 ROE 且 PB 不高，质量合理估值。 | 价值质量核心因子，可月频调仓。 |
| `earnings_yield_quality` | `rank(1 / pe_ttm) + rank(cashflow_to_profit)` | 盈利收益率便宜且利润现金含量高。 | 适合价值风格增强，注意金融地产行业暴露。 |
| `shareholder_yield_quality` | `rank(dv_ttm) + rank(roe_ttm)` | 股息回报与盈利能力结合。 | 可用于红利质量子策略。 |
| `quality_liquidity_confirm_20` | `zscore(rank(operating_cf_margin_ttm) + rank(amount / ts_mean(amount,20)))` | 现金流质量叠加成交额确认。 | 混合因子，用于基本面改善被市场开始关注的场景。 |
| `value_attention_gap_20` | `zscore(rank(1 / pe_ttm) + rank(roe_ttm)) - zscore(amount / ts_mean(amount,20))` | 便宜且质量好，但短期关注度不高。 | 用于寻找尚未拥挤的价值质量股票。 |
| `growth_turnover_confirm_20` | `zscore(rank(revenue_yoy) + rank(net_profit_yoy)) + zscore(turnover_rate / ts_mean(turnover_rate,20))` | 成长基本面叠加换手确认。 | 用于成长信号和市场参与度共振。 |
| `rd_neglect_20` | `zscore(rd_expense_intensity) - zscore(amount / ts_mean(amount,20))` | 高研发投入但市场关注不足。 | 偏长期主题因子，需结合行业中性化验证。 |

## 当前命令

下载原始数据：

```bash
venv/bin/python -m factors.data.fetcher --start 20180101 --end 20260706 --token "$TUSHARE_TOKEN"
```

构建 PIT 基本面日频数据：

```bash
venv/bin/python -m factors.data.fundamental_builder --start 20180101 --end 20260706
```

挖掘 HS300 纯基本面候选：

```bash
venv/bin/python run_alpha_mining.py --split train --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_train --min-abs-ic 0.003 --min-coverage 0.45
venv/bin/python run_alpha_mining.py --split valid --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_valid --min-abs-ic 0.003 --min-coverage 0.45
venv/bin/python run_alpha_mining.py --split test --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_test --min-abs-ic 0.003 --min-coverage 0.45
```

## 运行结果

### 数据下载与覆盖

本轮已使用 tushare pro 拉取历史 HS300 成分股的四张财报表，不在源码或文档中保存 token。

| 原始表 | 行数 | 股票数 | 报告期范围 | 公告日范围 |
|---|---:|---:|---|---|
| `income` | 18,989 | 564 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `balancesheet` | 18,994 | 564 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `cashflow` | 23,563 | 564 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `fina_indicator` | 18,764 | 564 | 2018-03-31 至 2026-03-31 | 2018-04-10 至 2026-05-15 |

PIT 对齐后生成：

| 文件 | 行数 | 股票数 | 交易日数 |
|---|---:|---:|---:|
| `data/processed/fundamental_daily.parquet` | 9,466,720 | 5,769 | 2,061 |
| `data/processed/train_fundamental.parquet` | 3,768,288 | 4,886 | 952 |
| `data/processed/valid_fundamental.parquet` | 1,179,072 | 5,182 | 242 |
| `data/processed/test_fundamental.parquet` | 4,519,360 | 5,670 | 846 |

HS300 股票池中核心字段覆盖率：

| 字段 | Train | Valid | Test |
|---|---:|---:|---:|
| `roe_ttm` | 74.88% | 100.00% | 99.97% |
| `cashflow_to_profit` | 74.88% | 100.00% | 99.97% |
| `revenue_yoy` | 73.53% | 100.00% | 99.99% |
| `net_profit_yoy` | 73.53% | 100.00% | 99.99% |
| `n_cashflow_act_ttm` | 74.88% | 100.00% | 99.97% |
| `debt_to_assets` | 96.60% | 100.00% | 100.00% |
| `pb` | 99.75% | 99.52% | 99.68% |
| `pe_ttm` | 93.53% | 87.80% | 86.25% |

### HS300 基本面挖掘结果

运行命令：

```bash
venv/bin/python run_alpha_mining.py --split train --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_train --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
venv/bin/python run_alpha_mining.py --split valid --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_valid --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
venv/bin/python run_alpha_mining.py --split test --fundamental --factor-set fundamental --universe hs300 --output outputs/fundamental_alpha_hs300_test --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
```

三段样本：

| Split | 时间 | HS300 行数 | 股票数 | 交易日数 | 候选数 |
|---|---|---:|---:|---:|---:|
| Train | 2018-01-02 至 2021-12-31 | 355,196 | 477 | 952 | 18 |
| Valid | 2022-01-04 至 2022-12-30 | 116,986 | 506 | 242 | 18 |
| Test | 2023-01-03 至 2026-07-03 | 441,863 | 559 | 846 | 18 |

跨样本稳定正向因子如下。`min_abs_rankic` 是三段样本中绝对 RankIC 的最小值，用来衡量最弱样本表现。

| 因子 | 方向 | Train RankIC | Valid RankIC | Test RankIC | Min Abs RankIC | 平均 Abs RankIC | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `roe_value_pb` | 正向 | 0.0284 | 0.0323 | 0.0514 | 0.0284 | 0.0374 | 最稳定的估值质量因子，高 ROE 且 PB 不高。 |
| `shareholder_yield_quality` | 正向 | 0.0306 | 0.0261 | 0.0279 | 0.0261 | 0.0282 | 稳定红利质量，适合做防御型基本面腿。 |
| `cash_profit_cover` | 正向 | 0.0294 | 0.0302 | 0.0253 | 0.0253 | 0.0283 | 利润现金含量稳定有效，推荐入库。 |
| `earnings_yield_quality` | 正向 | 0.0221 | 0.0504 | 0.0481 | 0.0221 | 0.0402 | 样本外更强的便宜质量因子，推荐入库。 |
| `value_attention_gap_20` | 正向 | 0.0305 | 0.0183 | 0.0265 | 0.0183 | 0.0251 | 便宜质量且低关注度，适合和成交拥挤度联用。 |
| `quality_growth_hmean` | 正向 | 0.0144 | 0.0230 | 0.0313 | 0.0144 | 0.0229 | 高质量成长，样本外表现更好。 |
| `quality_roe_ocf` | 正向 | 0.0445 | 0.0115 | 0.0235 | 0.0115 | 0.0265 | 训练期很强，样本外仍为正，适合作质量底仓。 |
| `quality_liquidity_confirm_20` | 正向 | 0.0187 | 0.0146 | 0.0108 | 0.0108 | 0.0147 | 现金流质量加成交确认，稳定但强度中等。 |
| `quality_roa_ocf` | 正向 | 0.0433 | 0.0098 | 0.0147 | 0.0098 | 0.0226 | 与 `quality_roe_ocf` 高相关，优先保留一个。 |
| `low_accrual_to_assets` | 正向 | 0.0238 | 0.0249 | 0.0061 | 0.0061 | 0.0183 | 应计质量有效但测试期减弱，可作为辅助过滤器。 |
| `ocf_growth_quality` | 正向 | 0.0058 | 0.0286 | 0.0218 | 0.0058 | 0.0187 | 经营现金流成长，训练期偏弱但样本外更强。 |

方向不稳定或暂不建议直接使用：

| 因子 | Train RankIC | Valid RankIC | Test RankIC | 问题 |
|---|---:|---:|---:|---|
| `rd_neglect_20` | 0.0094 | -0.0456 | 0.0091 | 验证期反向明显，可能强依赖 2022 年成长风格切换。 |
| `capex_efficiency` | -0.0082 | 0.0199 | 0.0213 | 训练期反向，需分行业或滞后处理。 |
| `asset_turnover_improve` | -0.0263 | -0.0085 | 0.0042 | 方向不稳，且训练期/验证期为负。 |
| `low_leverage_quality` | 0.0166 | -0.0123 | -0.0019 | 杠杆暴露可能和行业/金融地产混在一起。 |
| `debt_cash_safety` | 0.0115 | -0.0008 | -0.0067 | 偿债安全单独使用不稳定，建议行业中性后复查。 |
| `gross_margin_quality` | -0.0294 | 0.0004 | -0.0012 | 覆盖和方向都不理想，不入库。 |

### 推荐入库因子

第一批建议保留 6 个基本面因子：

| 入库名 | 来源候选 | 使用方向 | 经济含义 | 建议用法 |
|---|---|---|---|---|
| `alpha_fund_roe_value_pb` | `roe_value_pb` | 越大越好 | 高 ROE 且 PB 不高，代表质量合理估值。 | 月频或财报披露后周频调仓；和低波/流动性因子组合。 |
| `alpha_fund_cash_profit_cover` | `cash_profit_cover` | 越大越好 | 经营现金流覆盖净利润，利润含金量高。 | 作为财报质量过滤器或质量子因子主腿。 |
| `alpha_fund_earnings_yield_quality` | `earnings_yield_quality` | 越大越好 | 低 PE/高盈利收益率且现金流质量好。 | 价值质量主因子，注意行业暴露。 |
| `alpha_fund_shareholder_yield_quality` | `shareholder_yield_quality` | 越大越好 | 股息回报和盈利能力兼具。 | 防御型或红利质量策略中使用。 |
| `alpha_fund_value_attention_gap_20` | `value_attention_gap_20` | 越大越好 | 便宜质量公司但短期成交关注不拥挤。 | 作为反拥挤价值质量因子，配合量价确认。 |
| `alpha_fund_quality_growth_hmean` | `quality_growth_hmean` | 越大越好 | 收入和利润同步成长，并由现金流确认。 | 成长质量子因子，适合与估值质量混合。 |

第二批观察因子：

| 因子 | 使用建议 |
|---|---|
| `quality_roe_ocf` | 与 `quality_roa_ocf` 和 `roe_value_pb` 有较强质量暴露重合，可在去相关后保留。 |
| `quality_liquidity_confirm_20` | 稳定但强度一般，适合作为基本面信号的市场确认腿。 |
| `low_accrual_to_assets` | 测试期变弱，但经济含义清晰，适合作财报质量过滤器。 |
| `ocf_growth_quality` | 训练期弱、样本外较强，建议后续单独按财报季复查。 |

### 注意事项

1. 本轮财报数据按历史 HS300 成分股下载，适配本次 HS300 指增研究；如果切到全 A，需要把财报下载 universe 改成全 A 股票列表并控制接口频率。
2. 因子当前未做行业中性化。`roe_value_pb`、`earnings_yield_quality`、`shareholder_yield_quality` 可能带有金融、地产、公用事业等行业暴露，进入组合前应做行业/市值中性复查。
3. 挖掘日志中的 `ConstantInputWarning` 来自某些财报因子在非公告日保持不变，导致部分单日截面相关无法定义；评估器会跳过无效 IC，不影响最终均值。
4. 当前标签仍是未来 10 日收益，基本面因子更自然的持有期可能是 20-60 日。下一轮应增加月频标签或多 horizon 评估。
