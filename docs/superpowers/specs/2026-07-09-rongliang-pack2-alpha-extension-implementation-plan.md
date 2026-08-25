# 融量公开因子 Pack2 扩展实现计划

日期：2026-07-09

对应设计文档：`docs/superpowers/specs/2026-07-09-rongliang-pack2-alpha-extension-design.md`

## 原则

1. 只把 A 阶段可复现的 Pack2 因子真正落地计算。
2. 缺字段条目统一保留元数据，但状态落为 `skipped`，并在 `skip_reason` 中写明 `pending_b_stage`。
3. 不新开第二套公开因子框架，Pack2 直接复用现有 `public_factors -> validation -> model -> constrained backtest -> report` 流程。
4. 若原文解释与表达式冲突，优先按表达式实现，并在元数据/报告中注明。
5. 所有新增结果都要进入现有 Markdown 报告和 CSV 输出。

## 阶段 1：算子与基础 proxy

修改：

| 文件 | 内容 |
|---|---|
| `factors/alpha/operators.py` | 增加 `ts_percentage`，必要时补充 `scale`/别名语义支持 |
| `test_alpha_mining.py` | 为新增算子补 smoke test |

需要固定的 proxy：

| 专有字段 | A 阶段 proxy |
|---|---|
| `AF_CLOSE` | `close_adj` |
| `TURN_RATE` | `turnover_rate` |
| `VWAP` | `amount / volume` |
| `CHANGE_PCT` | `close_adj` 1 日收益率 |
| `FACTOR_VROC12D` | `volume` 的 12 日变化率 |
| `FACTOR_TVSD20D` | `TS_STDDEV(CS_REGRESSION(CLOSE, VOLUME, OUT_TYPE=0), 20)` |
| `FACTOR_VOL60D` | 本地统一的 60 日波动 proxy |

验收：

```bash
venv/bin/python test_alpha_mining.py
```

## 阶段 2：Pack2 公开因子 registry

修改：

| 文件 | 内容 |
|---|---|
| `factors/alpha/public_factors.py` | 新增 Pack2 计算函数、spec 和 metadata |

实现方式：

1. 将 Pack1 与 Pack2 分成两个内部 spec 列表，再由统一 `public_alpha_specs()` 汇总。
2. Pack2 A 阶段可复现条目落地为 `rl2_<id>_<slug>`。
3. B 阶段缺字段条目仍进入 metadata，但不进 `factor_values`。

本轮优先实现的 Pack2 因子：

```text
27, 28, 29, 30, 31, 32, 48, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65
```

本轮挂起的 Pack2 因子：

```text
33, 34-47, 49, 52, 53
```

验收：

- `calculate_public_factors()` 结果中出现 Pack2 新列。
- `public_factor_metadata.csv` 中出现 `rongliang_public_pack2` 来源。
- 缺字段条目被标记为 `skipped` 且 `skip_reason` 包含 `pending_b_stage`。

## 阶段 3：验证门与特征筛选

复用：

| 文件 | 内容 |
|---|---|
| `factors/alpha/validation.py` | 仅复用，不另开新逻辑 |
| `factors/reports/public_factor_validation.py` | 用 Pack1+Pack2 的统一公开因子集重跑验证 |

输出：

| 文件 | 内容 |
|---|---|
| `public_factor_validation_summary.csv` | Pack1+Pack2 联合验证结果 |
| `selected_public_features.csv` | 新筛选后的公开因子特征 |

重点检查：

1. Pack2 因子里哪些 `passed`。
2. 哪些只在训练/验证有效、但在 `2026YTD` 变弱，被 `quarantined`。
3. 是否存在 Pack2 因子替代掉旧 Pack1 因子的情况。

## 阶段 4：模型与受约束指增

复用：

| 文件 | 内容 |
|---|---|
| `factors/reports/constrained_index_enhancement.py` | 直接用扩展后的公开因子集重跑 |
| `factors/models/xgb_alpha.py` | 复用现有模型及 fallback 重要性逻辑 |

目标：

1. 观察 Pack2 加入后公开特征总数与最终入模特征数。
2. 比较旧版 `scenario_summary.csv` 与新版结果。
3. 重点看：
   - `xgb_no_style_constraint`
   - `xgb_industry_style_tight`
   - `xgb_industry_style_mid`
   - `current_exp_score`

判定标准：

- 若 Pack2 只提升训练/全样本，不提升测试或 `2026YTD`，报告中要明确写出“可算但无组合增量”。
- 若 Pack2 带来更高的样本外超额但也显著提高主动风险，报告中要写清是约束前收益还是约束后收益。

## 阶段 5：报告更新

修改：

| 文件 | 内容 |
|---|---|
| `factors/reports/constrained_enhancement_report.py` | 增加 Pack2 统计与结论 |
| `docs/csi500_xgb_constrained_index_enhancement_2026-07-09.md` | 重生成包含 Pack2 的阶段报告 |

报告新增内容：

1. Pack2 条目总数、可复现数、挂起数。
2. Pack2 中通过验证的因子清单。
3. Pack2 中失效/隔离的代表性因子及原因。
4. Pack2 是否改善最终受约束指数增强。
5. 明确哪些条目必须等 B 阶段资金流字段。

## 阶段 6：最终校验

测试：

```bash
venv/bin/python test_alpha_mining.py
venv/bin/python -m factors.reports.constrained_index_enhancement --start 20180101 --end 20260706
venv/bin/python -m factors.reports.constrained_enhancement_report
```

静态检查：

```bash
rg -n "!\[.*\]\(" docs/csi500_xgb_constrained_index_enhancement_2026-07-09.md
rg "<known-token-patterns-redacted>" -n . -g '!venv/**' -g '!data/raw/**' -g '!outputs/**'
```

## 实现顺序

1. 补 `ts_percentage` 和必要算子语义。
2. 在 `public_factors.py` 中加入 Pack2 metadata 和 A 阶段可复现因子。
3. 补 smoke tests。
4. 跑 `test_alpha_mining.py`。
5. 跑公开因子验证与受约束指增流水线。
6. 更新报告和图片。
