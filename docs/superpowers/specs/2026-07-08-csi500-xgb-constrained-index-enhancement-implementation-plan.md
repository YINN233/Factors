# 中证500 XGBoost 约束指数增强实现计划

日期：2026-07-08

对应设计文档：`docs/superpowers/specs/2026-07-08-csi500-xgb-constrained-index-enhancement-design.md`

## 原则

1. A 阶段只使用当前本地已有数据；B 阶段再补资金流。
2. 未经本地二次验证的外源 alpha 默认 `pending`，不能直接入模。
3. XGBoost 只负责输出 alpha，组合权重由约束优化器决定。
4. 行业和风格暴露必须进入优化器约束。
5. 所有结论必须落盘到 CSV/PNG/Markdown，并重点报告 2025-2026 与 2026 YTD。

## 阶段 1：公开因子与外源 Alpha 接口

新增：

| 文件 | 内容 |
|---|---|
| `factors/alpha/external_alpha.py` | 外源 alpha 元数据、标准面板校验、状态枚举 |
| `factors/alpha/public_factors.py` | 融量公开因子 A 阶段 direct/proxy 实现 |

修改：

| 文件 | 内容 |
|---|---|
| `factors/alpha/operators.py` | 补充 WMA、rolling quantile、max drawdown、多项式回归等缺失算子 |
| `test_alpha_mining.py` | 增加公开因子和外源 alpha smoke tests |

输出：

| 文件 | 内容 |
|---|---|
| `public_factor_availability.csv` | 26 个公开因子的 direct/proxy/skipped 状态和原因 |
| `public_factor_values.parquet` | 可计算公开因子值 |

验收：

```bash
venv/bin/python test_alpha_mining.py
```

## 阶段 2：因子验证闸门

新增：

| 文件 | 内容 |
|---|---|
| `factors/alpha/validation.py` | 覆盖率、方向校准、相关性去重、稳定性验证函数 |
| `factors/reports/public_factor_validation.py` | 公开因子验证脚本 |

验证项目：

| 项目 | 要求 |
|---|---|
| 覆盖率 | 日均覆盖率和有效交易日数达标 |
| IC/RankIC | 全样本、训练、验证、测试、2026 YTD 分开统计 |
| 方向 | 训练期定方向，验证/测试不重调 |
| 稳定性 | 年度、季度、月度、2026 YTD |
| 分组 | Top-Bottom 和单调性 |
| 去重 | 高相关同类因子只保留样本外更稳的 |
| 增量 | 检查相对基本面因子和公开因子线性合成的增量 |

输出：

| 文件 | 内容 |
|---|---|
| `public_factor_validation_summary.csv` | 逐因子验证结果和状态 |
| `public_factor_correlation.csv` | 相关性和去重依据 |
| `selected_model_features.csv` | XGBoost 特征白名单 |

## 阶段 3：XGBoost 预测层

新增：

| 文件 | 内容 |
|---|---|
| `factors/models/xgb_alpha.py` | XGBoost 训练、预测；不可用时降级 sklearn GBDT |
| `factors/reports/xgb_alpha_report.py` | 模型诊断和基准对比 |

训练口径：

| 项目 | 设置 |
|---|---|
| 主标签 | `fwd_5d_rank` |
| 敏感性标签 | `fwd_1d_rank` |
| 训练期 | 2018-2022 |
| 验证期 | 2023-2024 |
| 测试期 | 2025-2026YTD |
| 调参 | 只看验证期 |

输出：

| 文件 | 内容 |
|---|---|
| `xgb_predictions.parquet` | 每日每股预测分数 |
| `xgb_model_summary.csv` | 训练、验证、测试和 2026 YTD RankIC |
| `xgb_feature_importance.csv` | 特征重要性 |
| `xgb_vs_baselines.csv` | 与线性合成和旧方案对比 |

## 阶段 4：风格暴露与约束优化

新增：

| 文件 | 内容 |
|---|---|
| `factors/portfolio/style_exposures.py` | 规模、估值、动量、波动、流动性、质量、杠杆暴露 |
| `factors/portfolio/constrained_optimizer.py` | cvxpy 约束优化器 |

修改：

| 文件 | 内容 |
|---|---|
| `factors/portfolio/constraints.py` | 扩展行业、风格、主动权重、换手约束 |

场景：

| 场景 | 说明 |
|---|---|
| `current_exp_score` | 旧方案基准 |
| `xgb_no_style_constraint` | XGBoost + 个股上限 + 行业约束 |
| `xgb_industry_style_tight` | 严格风格约束 |
| `xgb_industry_style_mid` | 中等风格约束，主线 |
| `xgb_industry_style_loose` | 宽松风格约束 |

输出：

| 文件 | 内容 |
|---|---|
| `style_exposures.parquet` | 个股风格暴露 |
| `style_active_exposure.csv` | 组合主动风格暴露 |
| `industry_active_exposure.csv` | 行业主动偏离 |
| `constrained_weights.csv` | 调仓权重 |
| `constrained_daily_returns.csv` | 日度收益和超额 |

## 阶段 5：回测、归因和报告

新增：

| 文件 | 内容 |
|---|---|
| `factors/reports/constrained_index_enhancement.py` | 主回测脚本 |
| `factors/reports/constrained_enhancement_report.py` | 图表和 Markdown 报告生成 |
| `docs/csi500_xgb_constrained_index_enhancement_2026-07-08.md` | 最终研究报告 |

报告必须包含：

| 章节 | 内容 |
|---|---|
| 摘要结论 | 最优方案和是否优于旧方案 |
| 公开因子可用性 | 26 个公开因子的 direct/proxy/skipped 表 |
| 失效因子 | 失效、反向、冗余和隔离原因 |
| XGBoost 表现 | 样本外、2026 YTD、特征重要性 |
| 组合回测 | 年度、季度、月度、日度稳定性 |
| 行业/风格暴露 | 最新和历史主动偏离 |
| 归因 | alpha、行业、风格、个股贡献 |
| 风险 | 过拟合、proxy 误差、资金流缺失、交易约束不足 |

图片统一使用：

```text
![](../outputs/csi500_xgb_constrained_index_enhancement/xxx.png)
```

## 阶段 6：最终校验

测试：

```bash
venv/bin/python test_alpha_mining.py
```

如新增独立测试：

```bash
venv/bin/python test_xgb_constrained_enhancement.py
```

最终检查：

| 检查 | 命令或方式 |
|---|---|
| 图片引用 | `rg -n "!\\[.*\\]\\(" docs/csi500_xgb_constrained_index_enhancement_2026-07-08.md` |
| token 泄露 | `rg "<known-token-patterns-redacted>" -n . -g '!venv/**' -g '!data/raw/**' -g '!outputs/**'` |
| 约束满足 | 检查权重、行业、风格约束诊断表 |
| 样本外结论 | 检查 2025-2026 和 2026 YTD 单独表 |

## 实现顺序

1. 外源 alpha 元数据与公开因子实现。
2. 因子验证脚本和可用性输出。
3. XGBoost 或降级模型训练预测。
4. 风格暴露计算。
5. 约束优化器和指增回测。
6. 归因、图表和 Markdown 报告。
7. 测试、图片链接和 token 泄露检查。
