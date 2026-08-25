# 65 个公开因子缺字段补齐设计

日期：2026-07-10

## 背景

当前公开因子库只包含两批融量公开因子文本中的 65 个条目。上一轮已经完成可直接复现或可近似复现因子的计算、验证和中证500约束指增测试，但仍有若干条目因为 `MAIN_IN_FLOW_*`、`SLARGE_IN_FLOW_*`、`LARGE_OUT_FLOW_V2`、`NET_MF_AMOUNT_V2`、`REINSTATEMENT_CHG_60D`、`FACTOR_CNE5_*`、`AF_CLOSE` 等字段缺失而被标记为 `skipped` 或 `partial`。

本轮目标是补齐这些字段后重新验证这 65 个公开因子。范围明确限定为现有 65 个公开因子，不新增自研候选因子，也不扩大公开因子池。

## 范围

本轮只做三件事：

1. 为 65 个公开因子中的缺字段建立可审计字段映射。
2. 将原本 `skipped` 或 `partial` 的公开因子尽量转为 `direct` 或 `proxy` 计算。
3. 重跑同一套中证500验证门和约束指增，不新增第 66 个以后因子。

不做：

1. 不继续挖掘新的基本面因子或价量因子。
2. 不把非 Tushare 原生字段伪装成原始生产字段。
3. 不跳过验证门直接让补字段因子入模。

## 字段映射

| 公开字段 | 本地补齐口径 | 状态 | 说明 |
| --- | --- | --- | --- |
| `AF_CLOSE` | `close_adj` | direct | 已有 `daily + adj_factor` 复权价，作为公开表达式里的调整收盘价 |
| `AF_HIGH` | `high_adj` | direct | 已有复权最高价 |
| `AF_LOW` | `low_adj` | direct | 已有复权最低价 |
| `AF_OPEN` | `open_adj` | direct | 已有复权开盘价 |
| `AF_VWAP` | `amount / volume` | proxy | 本地成交额/成交量近似 VWAP |
| `REINSTATEMENT_CHG_60D` | `close_adj / delay(close_adj, 60) - 1` | proxy | 用 60 日复权收益近似公开字段中的复权变动 |
| `MAIN_IN_FLOW_V2` | 大单净额 + 特大单净额 | proxy | 来自 Tushare `moneyflow`，不是融量原始主力资金字段 |
| `MAIN_IN_FLOW_20D_V2` | `ts_sum(MAIN_IN_FLOW_V2, 20)` | proxy | 20 日主力资金净流入累计 |
| `MAIN_IN_FLOW_DAYS_10D_V2` | `ts_sum(MAIN_IN_FLOW_V2 > 0, 10)` | proxy | 10 日主力净流入天数 |
| `MAIN_IN_FLOW_DAYS_20D_V2` | `ts_sum(MAIN_IN_FLOW_V2 > 0, 20)` | proxy | 20 日主力净流入天数 |
| `SLARGE_IN_FLOW_V2` | 特大单净额 | proxy | 来自 Tushare `moneyflow` 的特大单买卖额差 |
| `LARGE_OUT_FLOW_V2` | 大单卖出额 + 特大单卖出额 | proxy | 用卖出额表示大额资金流出压力 |
| `NET_MF_AMOUNT_V2` | Tushare `net_mf_amount`，缺失时用主力净额 | proxy | 总资金净流入字段，优先使用 Tushare 原字段 |
| `CON_FUND_DAY_IN_10D_V2` | `ts_sum(MAIN_IN_FLOW_V2 > 0, 10)` | proxy | 没有机构专属字段，仅作为资金流入天数 proxy |
| `FACTOR_CNE5_BETA` | 本地 `style_volatility` | proxy | 不是 CNE5 原始 Beta，仅用于恢复公开表达式结构 |
| `FACTOR_CNE5_SIZE` | 本地 `style_size` | proxy | 不是 CNE5 原始 Size，仅用于恢复公开表达式结构 |
| `FACTOR_VROC12D` | `volume / delay(volume, 12) - 1` | proxy | 已有本地 proxy |
| `FACTOR_TVSD20D` | `ts_std(cs_regression_resid(close_adj ~ volume), 20)` | proxy | 已有本地 proxy |
| `FACTOR_VOL60D` | `ts_std(log1p(volume), 60)` | proxy | 已有本地 proxy |

## 数据流

```text
Tushare moneyflow
        |
        v
data/raw/moneyflow_20180101_20260706.parquet
        |
        v
moneyflow feature builder
        |
        v
validation panel with MAIN_IN_FLOW / SLARGE_IN_FLOW / NET_MF fields
        |
        v
public_factors.py only for factors 01-65
        |
        v
validation -> selected_public_features -> XGB -> constrained backtest -> report
```

`moneyflow` 缺失或不可下载时，流水线应失败并说明原因，不能静默退回到全 NaN 后误报因子无效。

## 因子状态调整

本轮将优先恢复以下原本缺字段条目：

| 来源 | 编号 | 当前问题 | 新状态目标 |
| --- | --- | --- | --- |
| Pack1 | 02 | 部分缺主力资金 | `proxy`，加入主力资金腿 |
| Pack1 | 11 | 缺大额流出 | `proxy` |
| Pack1 | 12 | 缺行业主力资金排序 | `proxy` |
| Pack1 | 13 | 缺主力/机构资金流入天数 | `proxy` |
| Pack1 | 14 | 11 的重复条目 | `proxy_duplicate` 或保留 duplicate 标注 |
| Pack1 | 15 | 部分缺主力资金 | `proxy`，加入主力资金腿 |
| Pack1 | 26 | 缺净资金流 | `proxy` |
| Pack2 | 33 | 缺复权变动 | `proxy` |
| Pack2 | 34-39 | 缺主力/超大单资金流 | `proxy` |
| Pack2 | 40 | 缺 CNE5 风格 | `proxy`，明确非原始 CNE5 |
| Pack2 | 41-43 | 缺主力/超大单资金流 | `proxy` |
| Pack2 | 44 | 缺 `AF_CLOSE` | `direct` |
| Pack2 | 45-47 | 缺复权变动/主力资金流 | `proxy` |
| Pack2 | 49 | 缺主力资金流 | `proxy` |
| Pack2 | 52-53 | 缺主力资金流 | `proxy` |

## 验证口径

所有补齐后的因子仍使用既有验证门：

1. 训练期决定方向。
2. 验证期 RankIC 必须为正。
3. 测试期和 2026YTD 不能明显反向。
4. 相关性去重后才能进入模型候选。
5. 最终仍以约束指增样本外表现和风险暴露作为组合层验证。

## 报告口径

报告必须明确区分：

1. 原始公开表达式。
2. 本地字段映射。
3. direct/proxy/duplicate 状态。
4. 补字段后是否通过本地验证。
5. 是否进入最终模型。

本轮结论不能写成“公开因子全部有效”，只能写成“在本地字段映射和验证口径下，哪些公开因子仍有样本外增量，哪些失效或需隔离”。

