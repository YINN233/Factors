# 中证500基本面 Alpha 挖掘记录

日期：2026-07-06

## 目标与计划

本轮把基本面 Alpha 挖掘股票池从 HS300 切到中证500历史成分股（`000905.SH`），并在原有财务指标连接方式上扩展更多算子。

| 步骤 | 状态 | 说明 |
|---|---|---|
| 复核四份中信建投 Alpha 框架 | 已完成 | 继续遵守 PIT 对齐、算子约束、IC/RankIC 评估、去冗余筛选的流程。 |
| 切换股票池 | 已完成 | `fetcher`、`fundamental_builder`、`run_alpha_mining` 支持 `000905.SH` 和 `--universe csi500`。 |
| 下载中证500财报数据 | 已完成 | 使用历史中证500成分股下载四张财报/指标表，不在代码和文档保存 token。 |
| 构建 PIT 日频面板 | 已完成 | 使用 `--index-universe-only` 只输出中证500成分行，降低内存并避免全 A 混入。 |
| 扩展算子和候选因子 | 已完成 | 新增稳健标准化、行业内排序、行业/市值中性化、线性衰减、滚动稳定性等算子与候选。 |
| train/valid/test 挖掘 | 已完成 | 纯基本面候选 44 个，输出跨样本汇总。 |
| 测试 | 已完成 | `test_alpha_mining.py` 覆盖新增算子、PIT 对齐和候选冒烟。 |

## 报告框架约束

四份报告对本轮实现的落地约束：

| 报告 | 本轮采用方式 |
|---|---|
| 专题二十三：基本面因子挖掘统一框架 | 财报按公告日 `ann_date/f_ann_date` 做 PIT 对齐；候选公式用可解释算子连接质量、成长、估值、现金流、偿债和效率字段。 |
| 专题二十八：分钟因子模型 | 高频部分暂不引入；保留日频成交额/换手率作为市场确认腿。 |
| 专题二十九：隔夜-日内异象 | 对混合因子保留成交额、换手率关注度差异，后续可接入隔夜/日内分解。 |
| 专题三十：量价 X 基本面统一框架 | 数据、表达、评估、筛选、文档都走统一流水线；基础字段先合成日频 PIT 面板再挖掘。 |

## 新增与可用算子

本轮在原有 `rank/zscore/ts_mean/ts_std/ts_corr/slope/rsquare` 基础上，补充了以下算子：

| 算子 | 作用 | 典型用途 |
|---|---|---|
| `signed_log1p(x)` | 保留符号的 `log1p(abs(x))` | 压缩极端财务比率。 |
| `signed_power(x, power)` | 保留符号的幂变换 | 调整非线性暴露。 |
| `cs_robust_zscore(x)` | 横截面 median/MAD 稳健标准化 | 降低极端值影响。 |
| `cs_scale(x)` | 横截面绝对值和归一化 | 组合权重或稀疏信号缩放。 |
| `cs_winsorize(x)` | 横截面分位截尾 | 处理财务异常值。 |
| `group_rank(x, industry)` | 行业内横截面 rank | 降低行业结构差异。 |
| `industry_neutralize(x)` | 每日行业内去均值 | 控制行业暴露。 |
| `size_neutralize(x, log_mv)` | 每日对市值做一元回归取残差 | 控制市值暴露。 |
| `ts_zscore(x, window)` | 个股滚动时间序列标准化 | 检测自身趋势变化。 |
| `ts_coef_var(x, window)` | 滚动变异系数 | 衡量现金流/利润稳定性。 |
| `ts_skew/ts_kurt` | 滚动偏度/峰度 | 识别非对称或尾部波动。 |
| `ts_decay_linear(x, window)` | 线性衰减均值，越近权重越高 | 对公告后信号做近端强化。 |
| `ts_argmax/ts_argmin` | 滚动窗口极值出现位置 | 判断改善或恶化是否偏近期。 |

## 数据覆盖

### 原始数据

历史中证500成分股共 1,179 只。

| 原始表 | 行数 | 股票数 | 报告期范围 | 公告日范围 |
|---|---:|---:|---|---|
| `income_000905_SH` | 38,813 | 1,179 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `balancesheet_000905_SH` | 38,946 | 1,179 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `cashflow_000905_SH` | 48,570 | 1,179 | 2014-12-31 至 2026-03-31 | 2018-01-05 至 2026-05-15 |
| `fina_indicator_000905_SH` | 38,347 | 1,179 | 2018-03-31 至 2026-03-31 | 2018-04-04 至 2026-06-19 |

### PIT 日频面板

| 文件 | 行数 | 股票数 | 交易日期间 | 交易日数 |
|---|---:|---:|---|---:|
| `fundamental_daily_000905_SH.parquet` | 1,677,172 | 1,179 | 2018-01-31 至 2026-07-03 | 2,040 |
| `train_fundamental_000905_SH.parquet` | 623,668 | 865 | 2018-01-31 至 2021-12-31 | 952 |
| `valid_fundamental_000905_SH.parquet` | 208,887 | 920 | 2022-01-04 至 2022-12-30 | 242 |
| `test_fundamental_000905_SH.parquet` | 844,617 | 1,159 | 2023-01-03 至 2026-07-03 | 846 |

核心字段覆盖率：

| 字段 | Train | Valid | Test |
|---|---:|---:|---:|
| `roe_ttm` | 75.52% | 99.98% | 99.95% |
| `cashflow_to_profit` | 75.52% | 99.98% | 99.96% |
| `revenue_yoy` | 74.31% | 100.00% | 99.96% |
| `net_profit_yoy` | 74.31% | 100.00% | 99.96% |
| `free_cashflow_ttm` | 72.95% | 93.57% | 81.44% |
| `cash_to_liab` | 95.35% | 97.91% | 98.06% |
| `working_capital_pressure` | 75.28% | 94.31% | 94.62% |
| `pb` | 99.63% | 99.43% | 99.55% |
| `pe_ttm` | 87.55% | 83.77% | 80.18% |
| `ps_ttm` | 99.88% | 99.86% | 99.92% |
| `dv_ttm` | 76.36% | 73.66% | 75.54% |

## 候选因子扩展

在 HS300 已有质量、成长、现金流、偿债、效率、估值、市场确认候选基础上，本轮新增的重点候选如下：

| 因子 | 表达式 | 经济含义 | 使用方式 |
|---|---|---|---|
| `robust_margin_value_ps` | `robust_zscore(net_margin_ttm) - robust_zscore(ps_ttm)` | 净利率高且 PS 不贵，用稳健标准化降低异常值影响。 | 可作为利润率估值因子，适合和 PB/PE 价值质量因子并列。 |
| `fcf_cash_conversion` | `rank(free_cashflow_ttm / n_cashflow_act_ttm) + rank(cashflow_to_profit)` | 自由现金流能从经营现金流中沉淀，且利润有现金流支持。 | 作为现金流质量过滤器，训练期偏弱但验证/测试较强。 |
| `liquidity_solvency_value` | `rank(cash_to_liab) + rank(current_ratio) - rank(pb)` | 现金与流动资产偿债能力强，同时账面估值不贵。 | 防御型基本面腿，适合弱市或信用风险上升阶段。 |
| `dupont_value_quality` | `rank(net_margin_ttm) + rank(asset_turnover_ttm) - rank(pb)` | 净利率和周转率共同解释 ROE，扣除 PB 估值。 | 本轮中证500最稳定的候选之一，建议优先入库。 |
| `inventory_receivable_light` | `-rank(inventories/assets) - rank(receivables/assets) + rank(ar_turn)` | 存货和应收占用低，应收周转更快。 | 营运效率辅助因子，单独使用稳定性一般。 |
| `industry_neutral_roe_value_pb` | `industry_neutralize(rank(roe_ttm) - rank(pb))` | 在行业内比较高 ROE 低 PB，降低行业估值结构影响。 | 建议替代或补充未中性的 `roe_value_pb`。 |
| `industry_rank_cash_profit_cover` | `group_rank(n_cashflow_act_ttm / net_profit_ttm, industry)` | 行业内比较利润现金含量。 | 适合做行业中性质量腿。 |
| `size_neutral_earnings_yield_quality` | `size_neutralize(rank(1/pe_ttm) + rank(cashflow_to_profit), log_mv)` | 便宜且利润现金含量高，同时剔除市值暴露。 | 训练期方向反转，暂不推荐直接入库。 |
| `decayed_quality_growth_20` | `ts_decay_linear(rank(revenue_yoy)+rank(net_profit_yoy)+rank(cashflow_to_profit),20)` | 对质量成长信号做近端线性衰减强化。 | 三段为正，但验证期较弱；可作为成长质量观察项。 |
| `stable_cash_conversion_20` | `zscore(cashflow_to_profit) - zscore(ts_coef_var(cashflow_to_profit,20))` | 偏好高且稳定的现金利润覆盖。 | 样本方向不稳，暂不入库。 |
| `margin_trend_value_20` | `zscore(ts_zscore(net_margin_ttm,20)) - zscore(ps_ttm)` | 净利率相对自身改善且 PS 不贵。 | test 强、valid 反向，需按行业/财报季复查。 |
| `cash_value_attention_gap_20` | `robust_zscore(rank(cash_to_liab)-rank(pb)) - robust_zscore(turnover/mean(turnover,20))` | 现金缓冲价值好且短期交易关注不拥挤。 | 稳定正向但强度中等，适合作反拥挤辅助腿。 |

## 挖掘命令

构建中证500 PIT 面板：

```bash
venv/bin/python -m factors.data.fundamental_builder \
  --start 20180101 --end 20260706 \
  --raw-suffix 000905_SH \
  --split-suffix 000905_SH \
  --index-code 000905.SH \
  --index-universe-only \
  --output-name fundamental_daily_000905_SH.parquet
```

挖掘纯基本面候选：

```bash
venv/bin/python run_alpha_mining.py --split train --fundamental --factor-set fundamental --universe csi500 --output outputs/fundamental_alpha_csi500_train --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
venv/bin/python run_alpha_mining.py --split valid --fundamental --factor-set fundamental --universe csi500 --output outputs/fundamental_alpha_csi500_valid --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
venv/bin/python run_alpha_mining.py --split test --fundamental --factor-set fundamental --universe csi500 --output outputs/fundamental_alpha_csi500_test --min-abs-ic 0.003 --min-coverage 0.45 --max-pair-corr 0.85
```

## 挖掘结果

三段样本：

| Split | 时间 | 行数 | 股票数 | 交易日数 | 候选数 |
|---|---|---:|---:|---:|---:|
| Train | 2018-01-31 至 2021-12-31 | 623,668 | 865 | 952 | 44 |
| Valid | 2022-01-04 至 2022-12-30 | 208,887 | 920 | 242 | 44 |
| Test | 2023-01-03 至 2026-07-03 | 844,617 | 1,159 | 846 | 44 |

跨样本方向一致的头部因子如下。`Min Abs RankIC` 是三段样本中最弱的绝对 RankIC。

| 因子 | 方向 | Train RankIC | Valid RankIC | Test RankIC | Min Abs RankIC | 平均 Abs RankIC | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `dupont_value_quality` | 正向 | 0.0330 | 0.0309 | 0.0482 | 0.0309 | 0.0374 | 本轮最稳定；净利率+周转率-PB。 |
| `cash_buffer_value` | 正向 | 0.0282 | 0.0376 | 0.0361 | 0.0282 | 0.0340 | 现金缓冲相对负债强且 PB 不贵。 |
| `robust_margin_value_ps` | 正向 | 0.0270 | 0.0347 | 0.0421 | 0.0270 | 0.0346 | 稳健利润率估值，test 更强。 |
| `growth_value_balance` | 正向 | 0.0280 | 0.0264 | 0.0310 | 0.0264 | 0.0285 | 收入/利润成长与 PE 估值平衡。 |
| `eps_bps_value_quality` | 正向 | 0.0292 | 0.0233 | 0.0294 | 0.0233 | 0.0273 | 每股收益、每股净资产与 PB 的价值质量。 |
| `quality_growth_hmean` | 正向 | 0.0263 | 0.0197 | 0.0257 | 0.0197 | 0.0239 | 收入利润同步成长并由现金流确认。 |
| `industry_neutral_roe_value_pb` | 正向 | 0.0332 | 0.0188 | 0.0382 | 0.0188 | 0.0301 | 行业内高 ROE 低 PB，优于控制行业暴露。 |
| `margin_value_ps` | 正向 | 0.0179 | 0.0290 | 0.0407 | 0.0179 | 0.0292 | 净利率估值，和稳健版含义接近。 |
| `liquidity_solvency_value` | 正向 | 0.0191 | 0.0174 | 0.0183 | 0.0174 | 0.0183 | 偿债安全+低 PB，稳定但强度中等。 |
| `roe_value_pb` | 正向 | 0.0155 | 0.0363 | 0.0457 | 0.0155 | 0.0325 | HS300 也有效；中证500训练期相对弱。 |
| `shareholder_yield_quality` | 正向 | 0.0152 | 0.0265 | 0.0197 | 0.0152 | 0.0205 | 红利质量稳定正向。 |
| `industry_rank_cash_profit_cover` | 正向 | 0.0126 | 0.0176 | 0.0113 | 0.0113 | 0.0138 | 行业内利润现金含量稳定有效。 |

方向不稳定或暂不建议直接使用：

| 因子 | Train RankIC | Valid RankIC | Test RankIC | 问题 |
|---|---:|---:|---:|---|
| `size_neutral_earnings_yield_quality` | -0.0150 | 0.0375 | 0.0383 | 训练期反向，市值中性后可能改变了早期价值暴露。 |
| `earnings_yield_quality` | -0.0070 | 0.0425 | 0.0369 | 样本外强但训练期反向，不能只凭 test 入库。 |
| `margin_trend_value_20` | 0.0139 | -0.0297 | 0.0425 | 验证期明显反向，可能受 2022 年风格切换影响。 |
| `growth_turnover_confirm_20` | 0.0381 | -0.0004 | 0.0176 | 验证期接近 0，量价确认腿不稳定。 |
| `cash_profit_cover` | -0.0059 | 0.0222 | 0.0209 | 训练期反向；行业内版本更稳。 |
| `quality_attention_gap_robust_20` | -0.0332 | 0.0011 | 0.0119 | 训练期反向明显。 |

## 推荐入库

第一批建议保留 8 个中证500基本面因子：

| 入库名 | 来源候选 | 使用方向 | 经济含义 | 建议用法 |
|---|---|---|---|---|
| `alpha_csi500_fund_dupont_value_quality` | `dupont_value_quality` | 越大越好 | 净利率和资产周转率共同好，且 PB 不贵。 | 中证500质量价值主因子，月频或财报后周频调仓。 |
| `alpha_csi500_fund_cash_buffer_value` | `cash_buffer_value` | 越大越好 | 现金相对负债缓冲强，同时账面估值合理。 | 防御型基本面腿，可和波动率/流动性约束组合。 |
| `alpha_csi500_fund_robust_margin_value_ps` | `robust_margin_value_ps` | 越大越好 | 高净利率低 PS，用 MAD 降低异常值影响。 | 利润率估值核心因子，适合行业中性后复查。 |
| `alpha_csi500_fund_growth_value_balance` | `growth_value_balance` | 越大越好 | 收入利润同步成长，且 PE 不高。 | 成长价值平衡因子，避免纯成长过贵。 |
| `alpha_csi500_fund_eps_bps_value_quality` | `eps_bps_value_quality` | 越大越好 | EPS/BPS 高且 PB 不贵。 | 简洁稳定的每股价值质量因子。 |
| `alpha_csi500_fund_quality_growth_hmean` | `quality_growth_hmean` | 越大越好 | 收入和利润同步成长，并有现金流质量确认。 | 成长质量子因子。 |
| `alpha_csi500_fund_industry_neutral_roe_value_pb` | `industry_neutral_roe_value_pb` | 越大越好 | 行业内高 ROE 低 PB。 | 优先用于行业中性组合，替代裸 `roe_value_pb`。 |
| `alpha_csi500_fund_liquidity_solvency_value` | `liquidity_solvency_value` | 越大越好 | 短期偿债能力和现金缓冲好，估值不贵。 | 风险控制或低信用风险环境下的辅助腿。 |

第二批观察因子：

| 因子 | 使用建议 |
|---|---|
| `margin_value_ps` | 与 `robust_margin_value_ps` 高度接近，优先保留稳健版。 |
| `roe_value_pb` | HS300 和中证500都有效，但中证500上行业中性版本更干净。 |
| `shareholder_yield_quality` | 稳定正向，适合红利质量子策略，不一定适合作为主因子。 |
| `industry_rank_cash_profit_cover` | 比裸 `cash_profit_cover` 更稳定，适合作为财报质量过滤器。 |
| `decayed_quality_growth_20` | 三段为正但验证期较弱，适合作为成长质量观察项。 |
| `cash_value_attention_gap_20` | 稳定正向但强度中等，可作为反拥挤辅助腿。 |

## 注意事项

1. 本轮结果是“当前候选库、当前标签、当前中证500历史成分池”下的最好一批基本面因子，不等于已经穷尽所有基本面 Alpha。
2. 标签仍是项目默认未来收益标签；基本面因子更自然的持有期可能是 20-60 日，下一步应做多 horizon 复查。
3. 部分 PIT 财报字段在非公告日保持不变，日志里的 `ConstantInputWarning` 来自无效截面相关日，评估器会跳过。
4. `--index-universe-only` 输出的是中证500成分行，适合本轮挖掘；如果要做全 A 对比，需要关闭该参数并保证内存充足或进一步分块落盘。
5. 推荐入库前仍应做行业/市值中性后的组合回测、换手和容量约束检查。

