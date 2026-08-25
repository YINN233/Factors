# 融量公开因子 Pack2 扩展设计

日期：2026-07-09

## 背景

根目录新增 `融量公开因子2.txt`，包含第 `27-65` 号公开因子，共 39 个条目。上一轮已经完成：

1. `融量量化公开因子.txt` 第 `1-26` 号因子的 A 阶段本地复现。
2. 中证500公开因子验证门、特征筛选、XGBoost 预测层、行业/风格约束指数增强回测。
3. 外源 alpha 的标准接入口、验证状态管理和报告闭环。

本轮目标不是盲目“多接几个因子”，而是把 `Pack2` 放进同一套严谨流程里，先判断哪些能在 A 阶段本地复现，哪些因为字段缺失只能挂起，哪些虽然可算但最终不值得进入模型。

## 目标

本轮要回答四个问题：

1. `27-65` 中哪些因子能在当前本地数据上直接或近似复现。
2. 这些新增因子经过中证500验证门后，哪些仍然有效。
3. 它们是否能改善当前“公开价量 + 基本面 + 约束优化”的指数增强方案。
4. 哪些因子应明确留到 B 阶段，避免用不可靠代理污染结论。

## 非目标

本轮不做以下事项：

1. 不直接下载并引入新的资金流字段。
2. 不为 `MAIN_IN_FLOW_*`、`SLARGE_IN_FLOW_*`、`CNE5_*`、`REINSTATEMENT_CHG_60D` 生造不可信代理。
3. 不新开一套独立的 Pack2 回测框架。
4. 不把公开文本中给出的 IC/Sharpe 直接当成项目结论。

## 方案选择

### 方案 1：只扩展前半段少量因子

只接 `27-33` 中最容易复现的价量条目。

优点：

- 改动最小。
- 最快出结果。

缺点：

- 会漏掉 `54-65` 中一批其实能由现有价量/换手率字段复现的因子。
- 后续还得再拆第二次扩展。

### 方案 2：把 `27-65` 全部纳入统一元数据框架，只实现 A 阶段可复现子集

对全部 39 个条目统一登记元数据；A 阶段能复现的落地计算函数，缺字段的显式标记 `skipped`，并在 `skip_reason` 中写明 `pending_b_stage`。

优点：

- 口径完整，未来补 B 阶段时可以直接续上。
- 报告里能清楚区分“不能用”与“还没到数据阶段”。
- 不需要新开第二套接入/验证逻辑。

缺点：

- 需要补一批表达式映射和元数据。

### 方案 3：连 B 阶段占位代理一起强行接入

优点：

- 一次性接入最多。

缺点：

- 容易引入失真代理。
- 会污染验证门和模型结论。

### 推荐

采用方案 2。

理由：这条路线最符合当前项目的批判性研究框架。它既不遗漏可复现因子，也不为了“多几个结果”去伪造资金流代理。

## 总体设计

继续复用现有四层结构：

1. `public factor registry`
2. `validation gate`
3. `model feature selection`
4. `constrained index enhancement`

新增内容只扩在公开因子层，不改外层回测逻辑。

数据流：

```text
融量公开因子2.txt
    |
    v
Pack2 元数据登记
    |
    |-- A 阶段可复现 -> 本地计算
    |-- 缺字段       -> skipped + skip_reason=pending_b_stage
    v
统一 public factor values
    |
    v
中证500验证门
    |
    v
与现有公开因子 + 基本面因子共同筛选
    |
    v
XGBoost / 约束指增 / 报告更新
```

## 因子分层

### A 阶段优先实现

以下条目原则上用当前字段直接或近似复现：

- `27` 反向量价排序乘积
- `28` 波动率差异因子
- `29` 成交量稳定收盘价
- `30` 短期波动调整收益
- `31` 价差动量
- `32` 逆向波动率协方差
- `48` 量价衰减协同
- `50` 量价协同波动负向选股
- `51` 多维度反转
- `54` 换手率波动
- `55` 波动换手耦合
- `56` 偏离度成交量加权
- `57` 波动调整反转
- `58` 流动性稳定度
- `59` 价量偏离波动率
- `60` 均线过滤反转
- `61` 量价背离协动
- `62` 反转换手增强
- `63` 量价效率
- `64` 残差波动率
- `65` 换手率波动动量

其中 `28`、`32`、`48` 依赖的专有中间量会统一映射为本地 proxy：

- `FACTOR_VROC12D` -> `volume` 的 12 日变化率 proxy
- `FACTOR_TVSD20D` -> `TS_STDDEV(CS_REGRESSION(CLOSE, VOLUME, OUT_TYPE=0), 20)` proxy
- `FACTOR_VOL60D` -> 统一采用本地定义的 60 日波动特征 proxy，并在元数据中标记

### B 阶段挂起

以下条目依赖当前本地缺失字段，先不进入 A 阶段主模型：

- `33` 复权价波动率比
- `34-46` 中绝大多数主力/超大单资金流条目
- `47`
- `49`
- `52`
- `53`

挂起原因主要是：

- `MAIN_IN_FLOW_20D_V2`
- `MAIN_IN_FLOW_V2`
- `SLARGE_IN_FLOW_V2`
- `REINSTATEMENT_CHG_60D`
- `FACTOR_CNE5_BETA`
- `FACTOR_CNE5_SIZE`

这些条目仍要进入元数据表，但状态不应伪装成失败；应明确区分“当前不可复现”和“已验证失败”。

## 命名和元数据

Pack2 因子统一命名为：

```text
rl2_<编号>_<slug>
```

示例：

- `rl2_27_reverse_price_volume_rank`
- `rl2_54_turnover_volatility_inverse`

每个条目保留以下元数据：

| 字段 | 含义 |
|---|---|
| `source` | 固定为 `rongliang_public_pack2` |
| `source_factor_id` | 原始编号，如 `27` |
| `factor_name` | 本地唯一名称 |
| `expression` | 原始表达式摘要 |
| `local_expression` | 本地落地表达式 |
| `availability` | `direct`, `proxy`, `partial`, `skipped` |
| `validation_status` | `pending`, `passed`, `failed`, `quarantined`, `skipped` |
| `required_columns` | 依赖字段 |
| `missing_columns` | 缺失字段 |
| `proxy_reason` | 近似原因 |
| `skip_reason` | 跳过原因 |

## 算子口径

为了避免文本描述与表达式口径不一致，本轮明确以下本地定义：

1. `RANK(X)`：按交易日做横截面百分位排序。
2. `TS_PERCENTAGE(X, N)`：按股票做最近 `N` 日时序百分位排名，不解释成简单百分比变化。
3. `TS_RANK(X, N)`：按股票做最近 `N` 日时序末值排名。
4. `SCALE(RANK(X))`：本地落为对横截面 rank 做 `cs_zscore`。
5. `AF_CLOSE`：映射到 `close_adj`。
6. `TURN_RATE`：映射到 `turnover_rate`。
7. `VWAP`：映射到 `amount / volume`。
8. `CHANGE_PCT`：映射到 `close_adj` 的 1 日收益率。

若原文解释与表达式冲突，优先按表达式落地，并在元数据或报告中注明。

## 代码改动边界

本轮只做以下结构性增量：

1. 扩展 `factors/alpha/operators.py`
   - 补充少量缺失辅助算子或别名，如 `ts_percentage`、必要的 `scale` 语义支持。
2. 扩展 `factors/alpha/public_factors.py`
   - 新增 Pack2 因子 spec 和计算函数。
3. 复用 `factors/alpha/validation.py`
   - 不开新验证门，只复用现有验证逻辑。
4. 复用 `factors/reports/public_factor_validation.py`
   - 增加 Pack2 统计输出。
5. 复用 `factors/reports/constrained_index_enhancement.py`
   - 用新增公开因子重新做筛选与回测。
6. 更新报告脚本与 Markdown
   - 增加 Pack2 可复现性、验证结果和增量影响说明。

本轮不拆分新模块，避免在研究迭代期引入额外重构风险。

## 输出

需要新增或更新以下产物：

| 文件 | 内容 |
|---|---|
| `public_factor_metadata.csv` | 包含 Pack1 + Pack2 全部元数据 |
| `public_factor_validation_summary.csv` | 包含 Pack2 新增因子验证结果 |
| `selected_public_features.csv` | 新筛选后的公开因子特征 |
| `xgb_feature_importance.csv` | 新增 Pack2 后的模型特征贡献 |
| `scenario_summary.csv` | 新增 Pack2 后的约束指增表现 |
| `docs/csi500_xgb_constrained_index_enhancement_2026-07-09.md` | 报告增补或重写 |

## 测试

本轮测试分三层：

1. 单元/烟雾测试
   - 新增算子和 Pack2 因子在合成数据上能跑通。
2. 验证测试
   - `public_factor_validation` 能输出 Pack2 因子的 `availability` 和 `validation_status`。
3. 端到端测试
   - `constrained_index_enhancement` 能在加入 Pack2 后完整输出回测结果。

至少要补以下断言：

- Pack2 新因子列能被计算出来。
- 缺字段的条目被正确标记为 `skipped`。
- 现有 `test_alpha_mining.py` 仍然通过。

## 风险与约束

1. `Pack2` 文本里有不少专有中间量定义不清，proxy 口径必须统一，不能每个因子各自猜一套。
2. 文本中 `TS_PERCENTAGE` 的解释有时混用“时序百分位”和“变化率”，必须用一套本地规则固定下来。
3. 新增公开因子数量较多，若全部进入验证门可能增加运行时间，需要延续上一轮的批量 rank 和窄表优化。
4. 即使单因子验证通过，也不等于组合层有增量；最终仍以受约束指增表现为准。

## 实施后判定标准

本轮扩展完成后，应能明确回答：

1. `Pack2` 里哪些因子在 A 阶段真正可复现。
2. 哪些新增因子通过了中证500验证门。
3. 它们是否进入最终公开特征集合。
4. 加入后是否改善受约束指数增强的测试期和 2026YTD 表现。
5. 哪些条目必须等待 B 阶段字段后再复检。
