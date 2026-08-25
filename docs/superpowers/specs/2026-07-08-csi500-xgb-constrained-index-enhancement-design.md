# 中证500 XGBoost 约束指数增强设计

日期：2026-07-08

## 背景

导师对当前中证500指数增强方案的核心反馈是：现有方案更像“多因子选股后按指数权重倾斜”，缺少真正指数增强应有的行业和风格约束。当前最佳旧方案为 `ytd_core3_monthly_s0.25`，使用三个基本面因子以 `benchmark_weight * exp(strength * zscore(score))` 方式调整权重。该方案已经有正超额，但组合层没有显式行业、风格、主动权重、换手约束。

根目录新增 `融量量化公开因子.txt`，其中包含 2026 年以来公开发布的 26 个公开因子。公开文本里的 IC 和 Sharpe 只能作为研究线索，不能直接作为本项目结论。所有因子必须在本地中证500样本、当前收益口径和当前组合约束体系下重新验证。

本设计按“先 A 后 B”执行：

- A 阶段：只使用当前本地已有数据，复现可直接或可近似实现的公开因子，叠加此前挖掘出的基本面因子，训练 XGBoost，进入行业/风格约束组合优化。
- B 阶段：在 A 阶段闭环后，新增 Tushare 资金流/大单资金数据，补齐依赖 `MAIN_IN_FLOW_*`、`LARGE_OUT_FLOW_*`、`NET_MF_AMOUNT_*` 等字段的公开因子。

## 目标

A 阶段要回答以下问题：

1. 公开因子里哪些在本地中证500样本仍然有效。
2. 哪些公开因子已经失效、方向反转、覆盖率不足，或只能作为 proxy 使用。
3. 公开价量因子加入后是否优于只使用基本面因子。
4. XGBoost 是否相对线性合成有样本外增量。
5. 加上行业和风格约束后，超额收益还剩多少。
6. 组合是否具备真正指数增强属性，而不是简单多因子选股。
7. 2026 年以来表现、行业暴露、风格暴露、个股暴露是否合理。

## 非目标

A 阶段不做以下事项：

1. 不把公开因子文本中的 IC/Sharpe 当作本项目结论。
2. 不强行复现当前本地数据缺失的资金流因子。
3. 不使用未来数据做方向校准、调参或筛选。
4. 不让 XGBoost 直接决定组合权重。
5. 不以裸多空或简单分组表现替代约束后指增表现。
6. 不把高度相关的一组重复动量/反转因子全部塞进模型。

## 总体架构

A 阶段拆成五个模块：

| 模块 | 职责 | 主要输出 |
|---|---|---|
| 公开因子解析与实现 | 从公开因子文本中挑出当前数据可直接或近似复现的表达式，统一命名、字段依赖和计算函数 | `public_factor_values.parquet` |
| 因子验证层 | 对公开因子和基本面因子做覆盖率、IC、RankIC、分组、稳定性、相关性和失效检查 | `public_factor_availability.csv`, `factor_validation_summary.csv` |
| XGBoost 预测层 | 使用通过验证的公开价量因子、基本面因子和必要风格特征预测未来截面收益/排名 | `xgb_predictions.parquet`, `xgb_feature_importance.csv` |
| 风格暴露层 | 计算行业、规模、估值、动量、波动、流动性、质量、杠杆等暴露 | `style_exposures.parquet` |
| 约束优化层 | 最大化预测 alpha，同时约束行业、风格、个股权重、主动权重和换手 | `constrained_weights.csv`, `constrained_daily_returns.csv` |

数据流：

```text
本地中证500 PIT 面板
    |
    |-- 公开价量因子计算
    |-- 已挖基本面因子读取
    |-- 风格暴露计算
    v
因子验证与筛选
    |
    v
XGBoost 横截面预测
    |
    v
约束组合优化
    |
    v
指增回测 + 归因 + 稳定性报告
```

关键原则：

1. 公开因子必须先验证再入模。
2. XGBoost 只输出 alpha，不直接生成组合。
3. 风格约束必须进入优化器，而不是只做事后展示。
4. 组合以中证500基准权重为锚。
5. 每一步都落中间产物和 Markdown 报告，便于向导师解释取舍。

## 公开因子处理

A 阶段使用当前本地字段：复权 `open/high/low/close`、`volume/amount`、`turnover_rate`、`total_mv/log_mv`、行业、估值和基本面字段。资金流字段暂不进入主模型。

| 编号 | 原公开因子主题 | A 阶段处理 | 原因 |
|---:|---|---|---|
| 1 | 价量偏度峰度复合 | 直接或近似复现 | `CLOSE/VOLUME/CHANGE_PCT` 可得 |
| 2 | 量秩投影主力下影 | 部分近似，资金腿跳过 | `MAIN_IN_FLOW_20D_V2` 缺失 |
| 3 | PVT 协方差 | 近似复现 | `PVT1D` 可由价量构造 |
| 4 | 估值-量价修正 | 部分近似 | 行业 BM/EVEBITDA 专有字段缺失 |
| 5 | EMA/中间价趋势偏离 | 近似复现 | EMA 和中间价可构造 |
| 6 | 负向波动压缩动量 | 直接复现 | OHLCV 足够 |
| 7 | 10 日量价联动趋势 | 直接复现 | 收盘价和成交量足够 |
| 8 | 方差比率背离 | 近似复现 | 用 120 日盈亏方差比与 60 日收益方差构造波动结构差 |
| 9 | 过滤式综合动量 | 部分近似 | `CETOPTTM/PE_LFY` 口径缺失，用本地盈利/估值字段做过滤或标记 proxy |
| 10 | 现金流价量趋势 | 近似复现 | 现金流基本面字段有，原始口径需标记 |
| 11 | 大单流出动量反转 | A 阶段跳过 | `LARGE_OUT_FLOW_V2` 缺失 |
| 12 | 行业主力资金盈利质量 | A 阶段跳过或仅保留盈利质量腿 | 主力资金字段缺失 |
| 13 | 量价资金流截面分位数 | A 阶段跳过 | 主力/机构资金字段缺失 |
| 14 | 大单流出动量反转 | A 阶段跳过 | 与 11 重复且资金字段缺失 |
| 15 | 量价资金非线性 | 部分近似 | 资金流腿缺失，价量腿可实现 |
| 16 | 衰减量价复合 | 直接复现 | `CHANGE_PCT/VOLUME` 可得 |
| 17 | 波动趋势复合 | 直接复现 | EMA/MA/VOL/Sharpe 可构造 |
| 18 | 换手率调整价格异常 | 直接复现 | `TURN_RATE/AF_CLOSE` 可得 |
| 19 | 非线性量价极端反转 | 近似复现 | 多项式回归可实现，方向要验证 |
| 20 | 价格动能衰减反转 | 直接或近似复现 | OHLC 足够 |
| 21 | 对数动量逆向排序 | 部分近似 | `FACTOR_ROCTTM` 缺失，可用 ROC proxy 或跳过 |
| 22 | 成交量背离复合动量 | 直接复现 | `CLOSE/VOLUME/TURN_RATE` 可得 |
| 23 | 换手率相对强度反转 | 直接复现 | `TURN_RATE` 可得 |
| 24 | 30 日价格反转 | 直接复现 | `AF_CLOSE` 可得 |
| 25 | 高开动量衰减 | 直接复现 | `HIGH/OPEN` 可得 |
| 26 | 资金流最大回撤 | A 阶段跳过 | `NET_MF_AMOUNT_V2` 缺失 |

每个因子带以下元数据：

| 字段 | 含义 |
|---|---|
| `factor_id` | 对应公开文本编号 |
| `factor_name` | 本地实现名称 |
| `source_expression` | 原表达式摘要 |
| `local_expression` | 本地实现表达式 |
| `availability` | `direct`, `proxy`, `partial`, `skipped` |
| `required_columns` | 本地字段依赖 |
| `missing_columns` | 缺失字段 |
| `proxy_reason` | 近似实现原因 |
| `used_in_model` | 是否进入 A 阶段主模型 |

## 外源 Alpha 接入口

未来可能继续引入新的外源 alpha 因子库。这些因子默认都视为未经二次验证的研究线索，不能绕过本项目验证流程直接进入 XGBoost 或组合优化。

设计上保留统一接入口，后续任何外源 alpha 都必须先转换为标准 long-form 面板：

```text
trade_date, ts_code, factor_name, factor_value, source, version, release_date
```

推荐接入方式：

| 层级 | 职责 |
|---|---|
| 外源适配器 | 把 CSV、Parquet、API 或表达式库转换为标准 alpha 面板 |
| 元数据登记 | 记录来源、版本、字段依赖、发布时间、是否 proxy、是否可复现 |
| 验证闸门 | 自动跑覆盖率、IC、RankIC、方向校准、稳定性、相关性和组合增量 |
| 特征入模清单 | 只有通过验证闸门的因子才写入 XGBoost 特征列表 |
| 审计输出 | 每个外源因子都有保留/剔除原因，方便复盘和向导师解释 |

标准元数据字段：

| 字段 | 含义 |
|---|---|
| `source` | 因子来源，例如 `rongliang_public`, `vendor_x`, `internal_research` |
| `source_factor_id` | 原始因子编号或名称 |
| `factor_name` | 本地唯一因子名 |
| `version` | 外源库版本或接入日期 |
| `release_date` | 因子发布日期，用于检查是否存在未来信息 |
| `expression` | 原始表达式或摘要 |
| `required_columns` | 字段依赖 |
| `availability` | `direct`, `proxy`, `precomputed`, `skipped` |
| `validation_status` | `pending`, `passed`, `failed`, `quarantined` |
| `used_in_model` | 是否进入当前主模型 |

外源因子的默认状态是 `pending`。如果字段缺失、发布时间晚于使用日期、样本外显著反向、与已有因子高度重复且无增量，状态应改为 `failed` 或 `quarantined`。`quarantined` 表示暂不进主模型，但保留观察和后续复检。

该接入口也服务 B 阶段资金流因子：资金流下载完成后，相关因子仍然先作为外源 alpha 进入验证闸门，而不是直接进入模型。

## 因子验证与淘汰

公开因子进入 XGBoost 前必须通过本地验证。验证对象包括公开价量因子、proxy 因子和此前挖掘出的基本面因子。

验证层级：

1. 覆盖率检查：中证500样本内日均覆盖率低于阈值的剔除。
2. 单因子 IC/RankIC：按全样本、训练期、验证期、测试期、2026 YTD 分开统计。
3. 年度/月度稳定性：检查正 IC 月份比例、年度失效窗口和 2026 年以来是否继续有效。
4. 分组单调性：检查 Top-Bottom 是否稳定，避免单一极端月份拉动。
5. 方向校准：仅使用训练期决定方向，验证期和测试期只验证，不重新调方向。
6. 相关性去重：同类高相关因子只保留样本外更稳的。
7. 增量检验：比较公开因子、基本面因子、公开+基本面、XGBoost 的增量。
8. 组合层检验：最终以带行业/风格约束后的指增表现为准。

保留规则：

```text
keep = coverage_ok
    and train_direction_not_zero
    and validation_rankic_after_direction > 0
    and test_rankic_after_direction >= 0
    and ytd_2026_rankic_after_direction not strongly negative
    and not redundant_with_better_factor
```

`not strongly negative` 定义为：2026 YTD RankIC 均值不低于 -0.01，且正 IC 日比例不低于 45%。这个门槛用于避免半年度样本太短时误删弱但有互补性的因子；最终是否进入模型还要看相关性去重和组合层增量。

关键输出：

| 输出 | 内容 |
|---|---|
| `public_factor_availability.csv` | 逐个公开因子说明能否使用、字段依赖和跳过原因 |
| `public_factor_validation_summary.csv` | IC、RankIC、稳定性、分组收益、保留/剔除结论 |
| `public_factor_correlation.csv` | 因子相关性和去重依据 |
| `public_factor_validation.md` | 批判性文字总结 |

## XGBoost 预测层

XGBoost 的定位是截面 alpha 预测器。它每天对中证500成分股输出预期收益或预期排名分数，权重由后续优化器决定。

训练标签：

| 标签 | 定义 | 用途 |
|---|---|---|
| `fwd_1d_rank` | T 到 T+1 收益的日内截面 rank | 更贴近日频调仓和短周期价量因子 |
| `fwd_5d_rank` | T 到 T+5 复合收益的截面 rank | 噪声更低，更适合周频/月频指增 |

主线先使用 `fwd_5d_rank`，并把 `fwd_1d_rank` 作为敏感性测试。

时间切分：

| 区间 | 用途 |
|---|---|
| 2018-2022 | 训练 |
| 2023-2024 | 验证、调参、early stopping、方向校准检查 |
| 2025-2026YTD | 最终测试 |
| 2026YTD | 单独重点报告 |

特征组：

| 特征组 | 示例 |
|---|---|
| 已挖基本面因子 | `eps_bps_value_quality`, `quality_growth_hmean`, `industry_neutral_roe_value_pb` |
| 公开价量因子 | 通过验证的 direct/proxy 公开因子 |
| 风格特征 | `log_mv`, `pb`, `pe_ttm`, 20/60 日动量, 20/60 日波动, 换手率 |
| 行业信息 | 不作为 alpha 主来源，主要用于约束和归因 |

防过拟合规则：

1. 先筛因子，再训练模型。
2. 使用浅树和强正则。
3. 只能用验证期做 early stopping，不能看测试期调参。
4. 必须与线性合成和旧方案对照。
5. 特征重要性只作解释，不能替代样本外验证。
6. 月度、年度和 2026 YTD 稳定性必须单独报告。

建议参数范围：

```text
max_depth = 2 or 3
learning_rate = 0.03-0.08
n_estimators <= 300
subsample = 0.7-0.9
colsample_bytree = 0.6-0.9
reg_lambda > 1
reg_alpha >= 0
```

当前环境里 `xgboost` 尚未安装。实现阶段先尝试安装 `xgboost`；如果安装失败，临时使用 `sklearn.ensemble.HistGradientBoostingRegressor` 作为降级模型，并在报告中明确它不是 XGBoost。

模型对照：

| 模型 | 目的 |
|---|---|
| 基本面线性合成 | 当前基准 |
| 公开价量线性合成 | 检查公开因子自身是否有用 |
| 基本面 + 公开价量线性合成 | 检查非线性模型是否必要 |
| XGBoost | 检查非线性增量 |
| XGBoost + 约束优化 | 最终指增结果 |

模型输出：

| 输出 | 内容 |
|---|---|
| `xgb_predictions.parquet` | `trade_date`, `ts_code`, `pred`, `pred_rank`, `label` |
| `xgb_feature_importance.csv` | gain 和 permutation importance |
| `xgb_model_summary.csv` | 训练、验证、测试 RankIC 和分年度表现 |
| `xgb_vs_baselines.csv` | 与线性基准对照 |

## 风格暴露层

每天对风格因子做截面 winsorize 和 zscore，并计算组合主动暴露。

风格清单：

| 风格 | 字段或构造 |
|---|---|
| 规模 | `log_mv` |
| 估值 | `pb`, `pe_ttm`, `ps_ttm` 或其组合 zscore |
| 动量 | 20/60 日复权收益 |
| 波动 | 20/60 日收益波动 |
| 流动性 | `turnover_rate`, `amount` |
| 质量 | `roe_ttm`, `cashflow_to_profit` |
| 杠杆 | `debt_to_assets` |

基准暴露：

```text
benchmark_exposure(style, date) = sum_i b_i * style_i
```

组合主动暴露：

```text
active_exposure(style, date) =
    sum_i w_i * style_i - sum_i b_i * style_i
```

暴露层输出：

| 输出 | 内容 |
|---|---|
| `style_exposures.parquet` | 每日个股风格暴露 |
| `style_active_exposure.csv` | 每日组合风格主动暴露 |
| `industry_active_exposure.csv` | 每日行业主动偏离 |

## 约束组合优化

优化变量：

```text
w_i = 组合中股票 i 的权重
b_i = 中证500基准权重
a_i = XGBoost 预测 alpha
```

目标函数：

```text
maximize:
    sum_i w_i * a_i
    - lambda_active * sum_i (w_i - b_i)^2
    - lambda_turnover * sum_i |w_i - w_prev_i|
```

如果 L1 换手项求解不稳定，A 阶段先改为换手硬约束或先只报告换手。

硬约束：

| 约束 | A 阶段设置 |
|---|---|
| 满仓 | `sum(w)=1` |
| long-only | `w_i >= 0` |
| 个股上限 | `w_i <= min(2%, b_i + 0.5%)` 或固定 2% |
| 行业偏离 | `abs(industry_weight_port - industry_weight_bench) <= 1%-2%` |
| 单股票主动权重 | `abs(w_i - b_i) <= 0.5%-1%` |
| 总主动权重 | `0.5 * sum(abs(w_i-b_i)) <= 10%-20%` |
| 风格暴露 | 每个风格主动暴露限制在 0.10 到 0.25 个截面标准差 |
| 换手 | 月频目标控制，先做网格测试 |

组合版本网格：

| 组合 | 说明 |
|---|---|
| `current_exp_score` | 旧的 `benchmark * exp(score)` 基准 |
| `xgb_no_style_constraint` | XGBoost，只做个股上限和行业约束 |
| `xgb_industry_style_tight` | 行业和风格严格约束 |
| `xgb_industry_style_mid` | 行业和风格中等约束，推荐主线 |
| `xgb_industry_style_loose` | 行业和风格宽松约束 |

组合层输出：

| 输出 | 内容 |
|---|---|
| `constrained_daily_returns.csv` | 日收益、基准收益、超额、换手、主动权重 |
| `constrained_weights.csv` | 每次调仓权重 |
| `scenario_summary.csv` | 各约束版本表现 |
| `attribution_summary.csv` | 行业、风格、选股贡献拆解 |

## 回测和报告

A 阶段主报告：

```text
docs/csi500_xgb_constrained_index_enhancement_2026-07-08.md
```

输出目录：

```text
outputs/csi500_xgb_constrained_index_enhancement/
```

报告结构：

| 章节 | 内容 |
|---|---|
| 摘要结论 | 是否可做增强，最优约束组合是哪一个 |
| 公开因子可用性 | 26 个公开因子逐个列出 direct/proxy/skipped 和原因 |
| 因子有效性验证 | IC、RankIC、年度/月度稳定性、失效因子表 |
| XGBoost 模型结果 | 训练/验证/测试表现、特征重要性、与线性基准对比 |
| 约束组合回测 | 年度、月度、季度、YTD、胜率、IR、回撤、换手 |
| 行业暴露 | 最新和历史行业主动偏离图表 |
| 风格暴露 | 规模、估值、动量、波动、流动性等主动暴露 |
| 归因分析 | 超额来自 alpha、行业、风格还是个股 |
| 批判性结论 | 哪些因子不可信，哪些结果可能过拟合 |
| B 阶段计划 | 资金流字段下载和公开因子补全 |

所有图片使用 Markdown `![](...)` 嵌入报告。

## 测试计划

新增或扩展以下测试：

| 测试 | 检查点 |
|---|---|
| 公开因子计算 smoke test | 可复现因子在合成数据上能跑通，不产生全空 |
| 因子验证测试 | 覆盖率、方向校准、相关性去重逻辑正确 |
| 风格暴露测试 | 截面 zscore、基准暴露、主动暴露计算正确 |
| 组合优化测试 | 权重和为 1、非负、个股上限、行业约束、风格约束满足 |
| 回测口径测试 | 不使用标准化后的 `close_adj` 当真实收益 |
| 报告生成测试 | 关键 CSV/PNG/MD 输出存在 |

继续保留现有测试：

```bash
venv/bin/python test_alpha_mining.py
```

必要时新增：

```bash
venv/bin/python test_xgb_constrained_enhancement.py
```

## B 阶段扩展

B 阶段在 A 阶段闭环后启动，只补齐资金流公开因子，不改变 A 阶段主架构。

待补字段：

| 字段类型 | 可能来源 |
|---|---|
| 主力净流入 | Tushare `moneyflow` 或相关接口 |
| 大单流入/流出 | Tushare moneyflow 大单字段 |
| 机构资金 | 如果 Tushare 不提供，标记不可复现 |
| 净资金流最大回撤 | 基于 moneyflow 构造 |

B 阶段新增步骤：

1. 扩展 `factors/data/fetcher.py` 下载资金流字段。
2. 将资金流字段 point-in-time 对齐到中证500面板。
3. 实现依赖资金流的公开因子。
4. 重新跑因子验证、XGBoost、约束组合优化和报告。

## 成功标准

A 阶段完成标准：

| 目标 | 标准 |
|---|---|
| 样本外有效性 | 2025-2026 测试期正超额，RankIC 不显著反向 |
| 指增属性 | 行业和风格主动暴露受控，有每日记录 |
| 相对旧方案 | 至少一个约束版本在 IR、回撤或暴露质量上优于旧方案 |
| 2026YTD | 单独报告，不能被长期样本平均掩盖 |
| 可解释性 | 保留和淘汰因子都有明确原因 |
| 可复现性 | 脚本、CSV、PNG、Markdown 全部落盘 |

若公开因子或 XGBoost 未能带来样本外增量，也应如实报告，并保留“旧基本面因子 + 约束优化”作为备选路线。
