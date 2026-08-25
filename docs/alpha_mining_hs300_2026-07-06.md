# 沪深300日频 Alpha 初筛记录

日期：2026-07-06

## 1. 数据与样本

本轮使用 tushare pro 拉取并缓存了以下原始数据：

| 数据表 | 文件 | 覆盖范围 | 行数 |
|---|---|---:|---:|
| 日行情 | `data/raw/daily_20180101_20260706.parquet` | 2018-01-02 至 2026-07-03 | 9,466,720 |
| 日指标 | `data/raw/daily_basic_20180101_20260706.parquet` | 2018-01-02 至 2026-07-03 | 9,402,922 |
| 复权因子 | `data/raw/adj_factor_20180101_20260706.parquet` | 2018-01-02 至 2026-07-06 | 9,727,346 |
| 沪深300权重 | `data/raw/index_weight_000300_SH_20180101_20260706.parquet` | 2018-01-31 至 2026-07-01 | 53,100 |
| 公司行业 | `data/raw/stock_company.parquet` | 当前上市公司 | 5,527 |

`index_weight` 为月度/调仓日记录，已在 `builder.py` 中按股票使用最近一期可见权重向后对齐到日频，避免只有月末日期有非零权重。

processed 数据按现有标签定义构建：

```text
label = close_adj(T + forward_days) / close_adj(T + 1) - 1
forward_days = 10
```

本轮 Alpha 挖掘只使用 `index_weight > 0` 的沪深300成分股样本：

| Split | 日期范围 | 样本行数 | 股票数 | 交易日数 |
|---|---|---:|---:|---:|
| Train | 2018-01-02 至 2021-12-31 | 355,196 | 477 | 952 |
| Valid | 2022-01-04 至 2022-12-30 | 116,986 | 506 | 242 |
| Test | 2023-01-03 至 2026-07-03 | 441,863 | 559 | 846 |

## 2. 筛选规则

候选因子来自 `factors/alpha/candidates.py`，窗口为 5、10、20 日。每个候选在 train、valid、test 三段分别计算 RankIC、RankIC IR、覆盖率和 turnover。

本轮稳定候选要求：

1. train、valid、test 三段 RankIC 方向一致；
2. 三段最小 `abs(RankIC_mean) >= 0.005`；
3. 三段最小覆盖率 `>= 0.90`；
4. 经济含义能够解释，且不明显依赖未来信息。

完整汇总文件：

- `outputs/alpha_mining_hs300_summary/all_split_summary.csv`
- `outputs/alpha_mining_hs300_summary/stable_factors.csv`

## 3. 稳定候选

| 因子 | 表达式 | 方向 | Train RankIC | Valid RankIC | Test RankIC | 经济含义 | 使用建议 |
|---|---|---:|---:|---:|---:|---|---|
| `turnover_stability_10` | `-ts_std(turnover_rate,10)` | 正向 | 0.0455 | 0.0871 | 0.0561 | 换手率越稳定，说明交易拥挤和资金扰动越低；在沪深300内更像低噪声流动性质量信号。 | active 核心候选，优先使用 10 日窗口。 |
| `turnover_stability_20` | `-ts_std(turnover_rate,20)` | 正向 | 0.0474 | 0.0802 | 0.0585 | 较长窗口的换手稳定性，更偏慢变量，换手更低。 | 可作为 `turnover_stability_10` 的低频替代。 |
| `turnover_stability_5` | `-ts_std(turnover_rate,5)` | 正向 | 0.0377 | 0.0827 | 0.0539 | 短期换手稳定性，对近期资金扰动更敏感。 | 放入 history/variant，避免与 10 日窗口重复过高。 |
| `liquidity_preference_5` | `-ts_mean(abs(return_1d) / amount,5)` | 正向 | 0.0183 | 0.0501 | 0.0121 | Amihud 式低冲击流动性。单位成交额引发的价格波动越小，交易容量越好。 | active 候选，适合和组合优化的交易成本目标一起使用。 |
| `liquidity_preference_20` | `-ts_mean(abs(return_1d) / amount,20)` | 正向 | 0.0135 | 0.0419 | 0.0064 | 较慢的低冲击流动性，降低短期噪声。 | 低频调仓时使用；日频模型中可保留为平滑版本。 |
| `volatility_5` | `ts_std(close / delay(close,1) - 1,5)` | 反向 | -0.0205 | -0.0395 | -0.0128 | 近期波动越高，未来 10 日收益越弱；在沪深300内体现短期风险惩罚。 | 使用时取负号，即 `inv_volatility_5 = -volatility_5`。 |
| `volatility_20` | `ts_std(close / delay(close,1) - 1,20)` | 反向 | -0.0185 | -0.0241 | -0.0172 | 中期波动风险惩罚，样本外方向稳定。 | 使用时取负号；可作为风控型 Alpha 或风险暴露控制变量。 |
| `low_close_support_20` | `ts_mean((close - low) / close,20)` | 正向 | 0.0189 | 0.0130 | 0.0175 | 收盘价相对日内低点有支撑，说明盘中承接较好，市场确认更充分。 | active 候选，建议与成交确认或低波动信号组合。 |
| `low_close_support_5` | `ts_mean((close - low) / close,5)` | 正向 | 0.0194 | 0.0094 | 0.0138 | 短期收盘支撑，反映最近几天尾盘承接。 | variant，适合更高频调仓。 |

## 4. 暂缓入 active 的候选

`range_pressure_5/20 = -ts_mean(high / low - 1, n)` 在三段中 RankIC 均为负，统计方向稳定，但这意味着应使用其反向，即更大的日内振幅对应更高后续收益。该结论可能来自沪深300内短期活跃度或波动风险补偿，而不是原始命名中的“低振幅稳定偏好”。

处理方式：

1. 暂不进入 active 因子库；
2. 后续改名并重构为 `range_activity_n = ts_mean(high / low - 1, n)`；
3. 单独做分组收益和行业/市值暴露检查后再决定是否保留。

## 5. 推荐入库版本

为避免多窗口同质化，本轮建议先沉淀以下 4 个 active 因子：

| 入库名 | 原始候选 | 方向处理 | 定位 |
|---|---|---|---|
| `alpha_turnover_stability_10` | `turnover_stability_10` | 原值 | 低拥挤、低资金扰动 |
| `alpha_liquidity_preference_5` | `liquidity_preference_5` | 原值 | 低冲击、可交易性 |
| `alpha_inv_volatility_5` | `volatility_5` | 取负 | 短期风险惩罚 |
| `alpha_low_close_support_20` | `low_close_support_20` | 原值 | 日内承接/市场确认 |

其余稳定窗口放入 history/variant，用于后续模型融合或滚动窗口敏感性分析。

## 6. 使用方式

### 6.1 重新下载原始数据

不要把 token 写进代码或文档。推荐使用环境变量：

```bash
export TUSHARE_TOKEN="你的token"
python -m factors.data.fetcher --start 20180101 --end 20260706
```

### 6.2 重建 processed 数据

```bash
python -m factors.data.builder --start 20180101 --end 20260706 --window 20 --forward 10
```

输出：

- `data/processed/train.parquet`
- `data/processed/valid.parquet`
- `data/processed/test.parquet`
- `data/processed/meta.json`

### 6.3 运行沪深300 Alpha 挖掘

```bash
python run_alpha_mining.py --split train --universe hs300 --output outputs/alpha_mining_hs300_train --windows 5,10,20 --min-abs-ic 0.005 --min-coverage 0.5
python run_alpha_mining.py --split valid --universe hs300 --output outputs/alpha_mining_hs300_valid --windows 5,10,20 --min-abs-ic 0.005 --min-coverage 0.5
python run_alpha_mining.py --split test  --universe hs300 --output outputs/alpha_mining_hs300_test  --windows 5,10,20 --min-abs-ic 0.005 --min-coverage 0.5
```

每个 split 输出：

- `factor_values.parquet`：逐日逐股票候选因子值；
- `candidate_summary.csv`：候选因子 RankIC、覆盖率、turnover、复杂度分；
- `selected_factors.csv`：按当前阈值和相关性去冗余后的候选。

### 6.4 接入训练或回测

将 active 因子从 `factor_values.parquet` 按 `trade_date, ts_code` 合并到 processed 数据：

```python
import pandas as pd

base = pd.read_parquet("data/processed/train.parquet")
fac = pd.read_parquet("outputs/alpha_mining_hs300_train/factor_values.parquet")

active = [
    "turnover_stability_10",
    "liquidity_preference_5",
    "volatility_5",
    "low_close_support_20",
]
fac = fac[["trade_date", "ts_code"] + active].copy()
fac["alpha_inv_volatility_5"] = -fac["volatility_5"]
fac = fac.rename(columns={
    "turnover_stability_10": "alpha_turnover_stability_10",
    "liquidity_preference_5": "alpha_liquidity_preference_5",
    "low_close_support_20": "alpha_low_close_support_20",
})
fac = fac.drop(columns=["volatility_5"])

merged = base.merge(fac, on=["trade_date", "ts_code"], how="left")
```

合并后建议做每日截面 z-score，再加入模型特征列或与 GRU/PortfolioNet 的预测信号做等权/回归融合。

## 7. 风险与后续复核

1. 本轮只使用日频量价和日指标，没有接入基本面 PIT 字段；后续可扩展到 `quality_liquidity_confirm`、研发忽视等混合模板。
2. 当前检验主要看 RankIC 和覆盖率，仍需补充分组收益、多空净值、行业/市值暴露和交易成本回测。
3. 多窗口候选相关性可能较高，入 active 库时应保留代表窗口，避免因子库同质化。
4. `range_pressure` 统计方向与原始经济假设不一致，必须单独复核后再使用。
