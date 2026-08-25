# 国信联合动量复现实现计划

## 1. 实现范围

本轮实现先覆盖可复现主线：

1. 下载并缓存 2010 年以来中信一级行业指数日线、中证全指、股票日线、复权因子、每日指标和指数权重；
2. 构建股票日频面板，包含复权收益、成交量、成交额、Tushare 行业和中信一级行业 proxy；
3. 计算 `IMAX20`、`ICM`、`VICM`、`ICR`、`VICR`、`CMC` 和市场联合动量复合因子；
4. 输出 RankIC、ICIR、十分组收益、多空净值、参数敏感性和股票池对照；
5. 生成 Markdown 复现报告，报告中对比 PDF 原始数值和本地复现数值。

暂不把该因子接入现有 XGBoost 指数增强模型；接入模型属于复现通过后的下一阶段。

## 2. 文件与模块

| 文件 | 动作 | 说明 |
| --- | --- | --- |
| `factors/reports/guoxin_co_momentum.py` | 新增 | 数据下载、行业映射、因子计算、回测和报告入口 |
| `test_alpha_mining.py` | 修改 | 增加联合动量权重和玩具面板因子计算 smoke test |
| `docs/guoxin_co_momentum_reproduction_2026-07-13.md` | 生成 | 最终复现报告 |
| `outputs/guoxin_co_momentum_reproduction/` | 生成 | 数据缓存、因子值、汇总表和图片 |

## 3. 数据下载策略

- 中信一级行业指数：从 `index_basic(market="CI")` 取 `CI005001.CI` 至 `CI005030.CI`，再逐代码调用 `index_daily`。
- 市场指数：使用 `000985.CSI` 中证全指。
- 股票数据：复用现有 Tushare 日线、每日指标和复权因子接口。
- 指数股票池：复用 `index_weight` 下载中证500、沪深300；中证1000、国证2000若接口无数据则在报告中披露。
- 下载分两档：
  - `--smoke`：2019-2020 小样本，用于开发和测试；
  - 完整复现：20100101-20231231，后续可追加 20260706。

## 4. 因子计算策略

- 每只股票独立滚动 20 个交易日窗口。
- `ICM/VICM` 选择排序变量最大的前 `n` 日。
- `ICR/VICR` 选择排序变量最小的前 `n` 日。
- 半衰期权重为 `2 ** (-(rank - 1) / n)`，第 1 名权重为 1，第 `n` 名权重约为 0.5。
- 若窗口内有效排序值或行业收益不足，则因子为缺失，不做静默填补。
- 默认标签为下一月持有收益，用于月频 RankIC 和分组收益；另输出未来 5 日 RankIC 供现有 alpha 框架对照。

## 5. 验证与输出

实现后依次运行：

1. `python -m py_compile factors/reports/guoxin_co_momentum.py`
2. `PYTHONDONTWRITEBYTECODE=1 ... venv/bin/python test_alpha_mining.py`
3. 小样本 smoke：
   `venv/bin/python -m factors.reports.guoxin_co_momentum --start 20190101 --end 20201231 --smoke`
4. 完整复现：
   `venv/bin/python -m factors.reports.guoxin_co_momentum --start 20100101 --end 20231231`

最终报告至少包含：

- PDF 摘要表；
- 本地复现摘要表；
- 因子 RankIC 和 ICIR；
- 十分组月度收益；
- 参数敏感性；
- 股票池对照；
- 与原文差异和可能原因。

## 6. 主要风险

1. 中信行业归属只能用 proxy，可能降低与原文数值的一致性；
2. 长周期全 A 日线下载耗时较长，可能遇到接口频率限制；
3. ST 摘帽 3 个月过滤不一定能完整实现；
4. 2010-2023 全量因子计算可能需要分批优化，必要时先按中证500复现，再扩展到全市场。

