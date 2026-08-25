# 65 个公开因子缺字段补齐实现计划

日期：2026-07-10

## 阶段 1：数据层补齐

目标：为现有 65 个公开因子补齐缺失字段，不新增因子。

修改文件：

| 文件 | 修改 |
| --- | --- |
| `factors/data/fetcher.py` | 增加 `fetch_moneyflow()`，并在 `run_fetch_all()` 中可选拉取 |
| `factors/reports/public_factor_validation.py` | validation panel 合并 moneyflow 衍生字段 |
| `factors/reports/constrained_index_enhancement.py` | 指增 panel 复用同一字段补齐逻辑 |

实现要点：

1. 从 Tushare `moneyflow` 拉取字段：
   - `ts_code`
   - `trade_date`
   - `buy_lg_amount`
   - `sell_lg_amount`
   - `buy_elg_amount`
   - `sell_elg_amount`
   - `net_mf_amount`
2. 保存到 `data/raw/moneyflow_<start>_<end>.parquet`。
3. 构造字段：
   - `MAIN_IN_FLOW_V2`
   - `MAIN_IN_FLOW_20D_V2`
   - `MAIN_IN_FLOW_DAYS_10D_V2`
   - `MAIN_IN_FLOW_DAYS_20D_V2`
   - `SLARGE_IN_FLOW_V2`
   - `LARGE_OUT_FLOW_V2`
   - `NET_MF_AMOUNT_V2`
   - `CON_FUND_DAY_IN_10D_V2`
4. 若 moneyflow 文件不存在，公开因子验证脚本应继续运行但保留 `skipped`，并在报告中注明资金流字段未补齐；正式补字段回测前必须先下载。

## 阶段 2：字段派生层

目标：在一个统一函数里完成公开因子字段映射，避免 validation 和 index enhancement 两套口径。

新增或修改函数：

| 函数 | 作用 |
| --- | --- |
| `augment_public_factor_fields(panel, raw_dir, start, end)` | 合并资金流、复权字段和本地风格 proxy |
| `_load_moneyflow_features(raw_dir, start, end, ts_codes)` | 加载并构造资金流字段 |

派生字段：

1. `AF_CLOSE = close_adj`
2. `AF_HIGH = high_adj`
3. `AF_LOW = low_adj`
4. `AF_OPEN = open_adj`
5. `AF_VWAP = amount / volume`
6. `REINSTATEMENT_CHG_60D = close_adj / delay(close_adj, 60) - 1`
7. `FACTOR_CNE5_BETA = style_volatility`
8. `FACTOR_CNE5_SIZE = style_size`

## 阶段 3：公开因子实现调整

目标：只在 65 个公开因子内恢复 skipped/partial 条目。

修改文件：

| 文件 | 修改 |
| --- | --- |
| `factors/alpha/public_factors.py` | 将可恢复 skipped specs 改为 proxy/direct specs，并补计算函数 |

优先恢复：

1. Pack1：`02, 11, 12, 13, 14, 15, 26`
2. Pack2：`33-47, 49, 52, 53`

注意：

1. `14` 是 `11` 的重复条目，应保留 duplicate 信息，验证时仍可计算但报告要标注重复。
2. `40` 是 CNE5 proxy，不能报告为原始 CNE5。
3. 所有资金流相关条目都应标为 `proxy`，因为 Tushare 资金流口径不等同于融量原字段。

## 阶段 4：重跑流水线

命令：

```bash
venv/bin/python -m factors.data.fetcher --start 20180101 --end 20260706 --index-code 000905.SH
venv/bin/python -m factors.reports.constrained_index_enhancement --start 20180101 --end 20260706
venv/bin/python -m factors.reports.constrained_enhancement_report
```

输出：

| 文件 | 目的 |
| --- | --- |
| `public_factor_metadata.csv` | 65 个因子的最新 direct/proxy/skipped 状态 |
| `public_factor_validation_summary.csv` | 补字段后逐因子验证结果 |
| `selected_public_features.csv` | 相关性去重后入模公开因子 |
| `scenario_summary.csv` | 补字段后约束指增表现 |
| `docs/csi500_xgb_constrained_index_enhancement_2026-07-10.md` | 新报告 |

## 阶段 5：测试和审计

测试：

1. `venv/bin/python test_alpha_mining.py`
2. 检查 `calculate_public_factors()` 仍只返回 65 个公开因子以内的列。
3. 检查 token 不出现在代码、文档、输出报告中。
4. 检查原本 skipped 数量下降，且没有新增非公开因子列。

审计重点：

1. 补字段后若因子仍失效，报告要保留失败结论。
2. 若资金流因子显著有效，需要区分单因子有效和组合增量有效。
3. 若补字段导致模型表现下降，应如实写入报告，不做选择性呈现。

