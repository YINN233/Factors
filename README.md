# Alpha Signal 因子库 & PortfolioNet 端到端指增组合优化

> 暑期实习项目：面向沪深300指数增强，复现 Two-Stage 基线与 PortfolioNet 端到端优化网络。

---

## 项目结构

```
factors/
├── data/
│   ├── fetcher.py      # tushare 原始数据下载
│   ├── builder.py      # X/M/y 构造 + 截面归一化
│   └── dataset.py      # PyTorch Dataset（按日截面 batch）
├── alpha/
│   ├── features.py     # 特征工程工具函数
│   ├── schema.py       # 字段元数据：频率/量纲/语义/PIT 要求
│   ├── operators.py    # 受约束因子算子：截面/时序/量价/中性化
│   ├── candidates.py   # 默认候选因子库：量价、隔夜-日内、混合因子模板
│   ├── miner.py        # 因子计算、评估、去冗余筛选流水线
│   ├── gru_model.py    # GRU 预测网络（共享）
│   └── evaluator.py    # 因子评估体系（IC/RankIC/分组/多空/turnover）
├── portfolio/
│   ├── constraints.py  # 指增约束生成器
│   ├── two_stage.py    # Two-Stage 外部 cvxpy 优化器
│   └── opt_layer.py    # PortfolioNet 可微优化层（cvxpylayers）
├── backtest/
│   └── engine.py       # 轻量回测引擎（日频截面调仓）
└── reports/
    ├── factor_report.py      # 因子评估报告生成
    └── portfolio_report.py   # 组合表现报告生成

train_two_stage.py      # Two-Stage 基线训练
train_portfolionet.py   # PortfolioNet 端到端训练
run_alpha_mining.py     # 因子候选挖掘与筛选入口
```

---

## 快速开始

### 1. 环境安装

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch cvxpy cvxpylayers tushare pandas numpy pyarrow matplotlib seaborn scikit-learn tqdm streamlit
```

### 2. 配置 tushare token

```bash
export TUSHARE_TOKEN="你的token"
```

也可以在命令行临时传入 `--token`。不建议把 token 写进代码文件或提交到仓库。

### 3. 下载原始数据

```bash
source venv/bin/activate
python -m factors.data.fetcher --start 20180101 --end 20241231
```

> tushare 免费版有调用频次限制，建议分几天拉取或使用积分版。

### 4. 构造数据集（X/M/y）

```bash
python -m factors.data.builder --window 20 --forward 10
```

输出：`data/processed/train.parquet`, `valid.parquet`, `test.parquet`

### 4.1 挖掘候选 Alpha 因子

```bash
python run_alpha_mining.py --split train --output outputs/alpha_mining_train --windows 5,20
```

输出：
  - `factor_values.parquet`：候选因子逐日逐股票取值
  - `candidate_summary.csv`：IC、RankIC、覆盖率、换手、复杂度、综合分
  - `selected_factors.csv`：经过覆盖率、IC 门槛和相关性去冗余后的候选

当前候选库参考四份中信建投“逐鹿”Alpha 报告总结，先覆盖三类可执行模板：

| 类别 | 代表候选 | 研究含义 |
|------|----------|----------|
| 日频量价 | `mom_close_5`, `amount_expansion_20`, `price_volume_corr_20` | 趋势、成交确认、量价关系 |
| 隔夜-日内 | `intraday_strength_20`, `overnight_reversal_20`, `oi_spread_20` | A 股 T+1 和隔夜/日内收益结构 |
| 量价 X 基本面 | `quality_liquidity_confirm_20`, `rd_neglect_20` | 在 PIT 基本面字段存在时启用基本面主线 + 量价确认模板 |

### 5. 训练 Two-Stage 基线

```bash
python train_two_stage.py --epochs 10 --hidden 64 --ic_lambda 0.3
```

输出：`outputs/two_stage/`
  - `best_gru.pt`：最优模型权重
  - `predictions_valid.parquet` / `predictions_test.parquet`：因子预测值
  - `eval_*/factor_evaluation.png`：因子评估图表
  - `backtest_*.csv` / `weights_*.parquet`：组合回测结果

### 6. 训练 PortfolioNet

```bash
python train_portfolionet.py --epochs 10 --hidden 64 --gamma 1.0
```

输出：`outputs/portfolionet/`
  - `best_portfolionet.pt`：最优模型权重
  - `predictions_*.parquet`：Smart Factor + 组合权重
  - `eval_*/factor_evaluation.png`：Smart Factor 评估图表
  - `backtest_*.csv` / `weights_*.parquet`：端到端组合回测结果

### 7. 股指期货监测驾驶舱（MVP）

监测模块优先使用 Tushare Pro，原始数据和派生结果分别缓存到
`data/raw/monitoring/` 与 `data/processed/monitoring/`。刷新数据和打开页面是两个独立步骤，页面不会在渲染时调用 Tushare。

```bash
# 小窗口验证（先确认 token 和权限）
export TUSHARE_TOKEN="你的token"
python -m factors.monitoring.cli refresh \
  --start 20260801 --end 20260805 --mode futures --force

# 2017 年至今的正式回填：按年度缓存、可断点续传，并自动合并全区间数据
python -m factors.monitoring.cli backfill \
  --start 20170101 --end 20260806 --mode all \
  --classification-path factors/monitoring/etf_classification.csv

# 可选：补抓最新指数成分股的已披露分红和最新价格快照（逐股查询，较慢）
python -m factors.monitoring.cli backfill \
  --start 20170101 --end 20260806 --mode all \
  --classification-path factors/monitoring/etf_classification.csv \
  --with-dividends

# ETF 需要先维护 factors/monitoring/etf_classification.csv，
# 再按 ETF 代码分批拉取行情和份额
python -m factors.monitoring.cli prepare-etf-classification \
  --as-of-date 20260805 \
  --etf-codes 510300.SH,510500.SH,512480.SH \
  --output factors/monitoring/etf_classification_review.csv

# 人工核对 review 队列后，将确认的行写入 etf_classification.csv，
# category=scale 时 risk_bucket=low_risk；industry/theme/strategy/style 时为 risk。
python -m factors.monitoring.cli refresh \
  --start 20170101 --end 20260805 --mode etf \
  --etf-codes 510300.SH,510500.SH

# 生成派生表（无网络）
python -m factors.monitoring.cli build \
  --start 20170101 --end 20260806 --mode all

# 启动本地驾驶舱
streamlit run factors/dashboard/app.py
```

驾驶舱包含四个页面：

- **今日摘要**：汇总实际交易日、整体风险偏好、四品种主力基差状态、证据质量和冲突；
- **配对比较**：选择多头、空头和期限，对比同期限原始年化基差差值及其 PIT 历史分位；
- **基差与期限结构**：查看单品种近月、次月、季月和次季月合约以及历史原始基差；
- **ETF 风险偏好**：查看成交、换手、份额、行情修正和分阶段成本后验证。

`build --mode all` 和 `backfill --mode all` 会额外生成：

```text
data/processed/monitoring/decision_summary_<start>_<end>.parquet
data/processed/monitoring/pair_basis_<start>_<end>.parquet
```

配对差值定义为“多头原始年化基差减空头原始年化基差”。较低的 PIT 分位表示多头腿相对便宜、基差结构更有利，但不代表多头指数未来一定跑赢空头指数。ETF 信号只表示整体风险环境，不用于推断两个指数的行业、主题或 AI 相对暴露。

`index_weight` 的 Tushare 历史口径是月末/调仓日权重，首版会用价格重估的 proxy 并在页面标记；ETF 风险型/低风险型分类只有在分类表中 `reviewed=True` 后才进入正式信号。

当前首版只在请求的 as-of 日期计算成分股分红点数；未运行 `--with-dividends` 或成分股分红/价格输入不完整时，页面会显示原始基差并标记 `unavailable / D`，不会把缺失分红当成已知的零分红。ETF 页面使用的默认窗口是活动 5 日、份额 20 日、行情 20 日和验证 60 日；报告没有完整披露这些窗口，因此信号会标记为参数未披露，且人工分类使其最高为 C 级证据。

截至 2026-08-06，正式缓存覆盖 2017-01-03 至 2026-08-06；IM 从上市日 2022-07-22 开始。当前数据包含 31,868 条真实期货合约日行情、9,316 条指数日行情、23,865 条 ETF 日行情和 23,994 条 ETF 份额记录。最新成分股分红缓存覆盖 1,800 只股票，但由于缺少卖方历史预测快照，含分红基差保持 C 级部分口径；历史图明确使用原始年化基差。公司数据库待补字段见 `data/processed/monitoring/company_database_gaps.csv`。

2026-08-07 的期货和 ETF 行情已经发布，但 ETF 份额接口尚无当日记录，因此驾驶舱共同完整截止日保持 2026-08-06。修改历史、口径变化、回填行数和验证结果统一记录在 `CHANGELOG_MONITORING.md`。

需要在 Tailnet 内共享时，可在运行 Streamlit 的 Windows 主机执行 `tailscale serve --bg 8501`。导师必须加入同一 Tailnet；不要使用无认证的公网端口或 Tailscale Funnel 暴露驾驶舱。

---

## 核心设计说明

### Alpha 挖掘 Pipeline

- **字段元数据**：`schema.py` 记录字段频率、量纲、语义和是否需要 PIT 对齐，避免把不可比变量随意组合。
- **算子注册**：`operators.py` 提供 `safe_div`、截面 `rank/zscore`、时序均值/波动/相关等基础算子。
- **候选库**：`candidates.py` 固化一批可解释候选，后续可加入 GP/LLM 生成的新表达式。
- **多目标评估**：`miner.py` 同时看 IC、RankIC、覆盖率、换手、复杂度和候选间相关性，避免只按样本内 IC 排序。
- **因子入库接口**：`run_alpha_mining.py` 生成可归档的候选摘要和 selected factors，后续可以接入 active/history/archive 因子库。

### 数据 Pipeline

- **X**：20 天量价面板（OHLCV + 技术指标），每日截面 z-score 归一化
- **M**：对数市值、行业 one-hot、成分股权重（指数增强约束来源）
- **y**：未来 10 个交易日（T+1~T+11）累计收益

### Two-Stage 基线

1. **Stage 1**：GRU 预测个股收益，损失函数 `MSE + λ·(-IC)`
2. **Stage 2**：每天截面 cvxpy 优化，约束包括满仓、个股上限 2%、行业偏离 3%
3. **回测**：日频调仓，扣除 0.15% 双边手续费

### PortfolioNet 端到端

1. **GRU** 输出预测收益 `r_hat`
2. **OptLayer**（cvxpylayers）将 `r_hat` 和约束参数映射为组合权重 `w`
3. **组合级损失**：`-portfolio_return + γ·tracking_error`
4. **梯度回传**：通过 KKT 条件自动优化 GRU 参数，使信号适配组合目标
5. **Smart Factor**：训练完成后提取 GRU 的 `r_hat` 作为端到端因子，独立评估

---

## 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--window` | 历史回看天数 | 20 |
| `--forward` | 预测未来天数（标签） | 10 |
| `--hidden` | GRU 隐层维度 | 64 |
| `--layers` | GRU 层数 | 2 |
| `--dropout` | Dropout 率 | 0.2 |
| `--lr` | 学习率 | 1e-3 |
| `--epochs` | 训练轮数 | 10 |
| `--ic_lambda` | IC 损失权重 | 0.3 |
| `--gamma` | 跟踪误差惩罚 | 1.0 |
| `--fee` | 双边手续费率 | 0.0015 |

---

## 注意事项

1. **tushare 免费版限制**：`daily` 接口每秒最多调用几次，大量数据建议分次拉取。入职后切公司数据源只需替换 `fetcher.py`。
2. **固定股票池**：PortfolioNet 的 OptLayer 内部固定维度（默认 300），每天实际股票数不足时会自动 padding。沪深300成分股每天最多 300 只，停牌日会略少，不影响求解。
3. **训练速度**：PortfolioNet 每 batch 需要求解凸优化问题，比 Two-Stage 慢 10~50 倍。建议先用小 epoch 验证逻辑，再跑完整训练。
4. **约束扩展**：当前已实现满仓、个股上限、行业偏离。成分股权重偏离和换手率约束可在 `constraints.py` / `two_stage.py` 中扩展。

---

## 里程碑

| 阶段 | 状态 | 说明 |
|------|------|------|
| P0 数据基建 | ✅ | 数据下载、构造、Dataset |
| P1 基线因子 | ✅ | GRU 训练 + 因子评估 |
| P2 Two-Stage 组合 | ✅ | cvxpy 优化 + 回测 |
| P3 PortfolioNet | ✅ | cvxpylayers + 端到端训练 |
| P4 沉淀交付 | 🔄 | 代码文档 + 研究报告 |

---

*项目从 0 搭建，后续可根据实习期间的数据源和算力资源逐步调参优化。*
