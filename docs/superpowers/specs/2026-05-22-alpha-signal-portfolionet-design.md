# Alpha Signal 因子库扩展 & PortfolioNet 端到端指增组合优化复现

> 设计日期：2026-05-22  
> 目标指数：沪深300指数增强  
> 数据源：tushare（过渡）/ 公司数据（正式）

---

## 1. 项目目标

在真实指增约束下，用机器学习挖掘可交易 Alpha，并复现端到端组合优化网络 PortfolioNet（含 Two-Stage 对照）。

### 核心交付物
1. **可复用因子模块**：GRU 量价因子 + 因子评估体系（IC、RankIC、分组、turnover）
2. **组合优化模块**：Two-Stage 外部优化器 + PortfolioNet 可微优化层
3. **Smart Factor**：PortfolioNet 端到端训练得到的隐含因子，经评估后沉淀入库
4. **研究报告**：因子库报告 + 组合表现报告（超额收益、跟踪误差、信息比率）

---

## 2. 整体架构

项目按 **数据 → 信号 → 优化 → 回测 → 风控/归因** 链路组织，代码拆分为 5 个独立模块：

```
factors/
├── data/              # 数据 pipeline
│   ├── fetcher.py     # tushare 原始数据下载
│   └── builder.py     # X, M, y 构造 + 截面归一化
├── alpha/             # Alpha 因子与预测模型
│   ├── features.py    # 量价特征工程（OHLCV + 技术指标）
│   ├── gru_model.py   # GRU 预测网络（Two-Stage & PortfolioNet 共享）
│   └── evaluator.py   # 因子评估体系（IC, RankIC, 分组收益, turnover）
├── portfolio/         # 组合优化
│   ├── constraints.py # 指增约束生成（市值/行业/成分股权重偏离）
│   ├── two_stage.py   # Two-Stage：外部 cvxpy 优化器（日频截面）
│   └── opt_layer.py   # PortfolioNet：cvxpylayers 可微优化层
├── backtest/          # 回测引擎
│   └── engine.py      # 截面回测：逐日调仓、收益计算、费后净值
└── reports/           # 输出与沉淀
    ├── factor_report.py       # 单因子评估报告模板
    └── portfolio_report.py    # 组合表现报告（超额收益、跟踪误差、IR）
```

### 关键设计原则
- **数据层与模型层解耦**：`data/builder.py` 输出标准化 `parquet`，模型层只读不写
- **GRU 模型共享**：Two-Stage 和 PortfolioNet 使用同一个 `gru_model.py`，对照实验更公平
- **回测引擎只做轻量评估**：日频截面调仓、收益归因、费后统计，不做复杂事件驱动
- **模块边界清晰**：每个模块可独立测试，接口通过 numpy/pandas/Tensor 传递

---

## 3. 数据 Pipeline（X, M, y）

### 3.1 数据来源

| 数据 | tushare 接口 | 用途 |
|------|-------------|------|
| 沪深300成分股权重 | `index_weight` | 约束特征 M + 权重偏离计算 |
| 日行情 | `daily` + `daily_basic` | OHLCV、换手率、市值 |
| 行业分类 | `stock_company` | 一级行业 one-hot |

### 3.2 X（模型输入 — 量价面板）

- **时间窗口**：20 个交易日（可调，GRU 通常 20-60 天）
- **特征维度**（每只股票每天）：
  - 原始：`open, high, low, close, volume, amount`
  - 衍生：`returns(1d), volatility(5d), SMA(5/10/20), EMA, RSI(14), MACD`
  - **截面归一化**：每日对所有股票做 z-score 或 rank 归一化（预计算在 `builder.py` 中完成）
- **输出形状**：`(stocks, time_steps=20, features)`，按日截面 batch

### 3.3 M（约束特征）

每只股票的静态/准静态约束输入：
- `log_market_cap`：对数市值（过去1个交易日）
- `industry_onehot`：一级行业 one-hot 编码
- `index_weight`：该股票在沪深300中的权重（用于控制 `|w_i - benchmark_i| <= delta` 的偏离约束）

### 3.4 y（标签）

- `y = (close_{T+11} / close_{T+1}) - 1`
- 未来 10 个交易日（T+1 ~ T+11）的累计收益率
- 额外计算 `y_rank` 作为稳健性参考标签

### 3.5 数据流

```
tushare API
   ↓
fetcher.py 下载原始数据 → 本地 parquet（按表存储）
   ↓
builder.py 截面对齐 → 停牌处理（mask 掉） → 特征计算 → 标签打标
   ↓
输出 train.parquet / valid.parquet / test.parquet（时间切分）
   ↓
PyTorch Dataset（按日截面 batch：(stocks, 20, features)）
```

### 3.6 防前视偏差

- 特征只用 T 日及之前数据
- 标签从 T+1 日开始
- 价格使用复权价（`adj_factor`），保证收益率连续
- 停牌股票当日不参与训练和预测

---

## 4. Alpha 因子模块

### 4.1 特征工程（features.py）

- 原始 OHLCV 计算技术指标
- 所有特征每日截面 z-score / rank 归一化
- 输出和 X 保持一致，供 GRU 直接使用

### 4.2 GRU 预测网络（gru_model.py）

| 设计点 | 选择 |
|--------|------|
| 输入 | 每只股票独立：`(time_steps=20, features)` |
| 结构 | 2层 GRU，`hidden_size=64`，`dropout=0.2` |
| 输出 | 每只股票一个标量 `r_hat`（预期未来10日收益） |
| 损失函数 | `MSE(r_hat, y) + λ * (-IC(r_hat, y))`，λ 取 0.1~0.5 |

> IC 损失直接优化截面排序的线性相关性，对量化因子比对原始收益更实用。

### 4.3 因子评估（evaluator.py）

对任意因子序列（`r_hat`）进行全面评估：
- **IC / RankIC**：日频截面计算，输出均值、标准差、IR（IC / std）
- **分组测试**：按因子值将股票分为 5/10 组，计算各组累计收益曲线
- **多空对冲**：Top 组 vs Bottom 组，计算对冲收益、最大回撤、夏普比率
- **Turnover**：相邻两期权重变化的均值，衡量因子稳定性
- **衰减分析**：IC 在不同预测周期（1d, 5d, 10d, 20d）的表现

---

## 5. 组合优化模块

### 5.1 约束生成（constraints.py）

每天根据 M 生成约束矩阵和边界：
- 行业归属矩阵 `A_sector`（`sectors x stocks`）
- 基准权重向量 `b`（来自 `index_weight`）
- 个股上下界 `l`, `u`
- 行业偏离边界 `δ_sector`
- 权重偏离边界 `δ_weight`

### 5.2 Two-Stage 基线（two_stage.py）

**Stage 1**：GRU 输出 `r_hat` 作为 alpha 信号（独立训练完成）。

**Stage 2**：每天截面独立运行 cvxpy 优化器。

**优化问题**（指数增强，含核心 3 条约束）：

```
maximize:   w^T @ r_hat
subject to:
    sum(w) = 1              ← 满仓
    w >= 0                  ← 不做空
    w_i <= 0.02             ← 个股上限 2%
    |A_sector @ w - b_sector| <= 0.03   ← 行业偏离 ≤ 3%
```

> 注：成分股权重偏离、换手率约束作为 **扩展项**，核心功能完成后逐步添加。

### 5.3 PortfolioNet 端到端（opt_layer.py）

在 GRU 后串联两个新模块：

#### 约束感知生成器（简化版）
直接将 M（市值、行业、基准权重）与 GRU 输出 `r_hat` 拼接，作为优化层的上下文输入。

#### OptLayer（cvxpylayers）
- **输入**：`r_hat`（来自GRU）、约束参数（`A_sector`, `b`, `l`, `u`）
- **内部**：把上述 cvxpy 优化问题封装成 `CvxpyLayer`
- **输出**：组合权重 `w`
- **反向传播**：自动求解 KKT 条件，梯度流回 GRU，实现 **"预测信号自动适配组合目标"**

```
输入 X ──→ GRU ──→ r_hat ──┐
                             ├──→ [CvxpyLayer] ──→ w（组合权重）
输入 M ──→ 约束参数 ─────────┘         ↑
                                       └── 梯度回传，优化 GRU 参数
```

#### Smart Factor 提取
训练完成后，冻结 OptLayer，单独把 GRU 的 `r_hat` 输出作为一个 **端到端学习出的因子**，用 `alpha/evaluator.py` 做完整评估，和 Two-Stage GRU 因子做对比。

---

## 6. 训练策略

### 6.1 Two-Stage

- **Stage 1 训练**：监督学习，`MSE + IC` 损失，按日截面 batch
- **Stage 2 推断**：冻结 GRU，每天用 cvxpy 求解优化问题得到权重
- **评估**：回测引擎跑组合净值，计算超额收益、跟踪误差、IR

### 6.2 PortfolioNet

- **组合级损失函数**：`Loss = -mean(w^T @ r_future) + γ * TrackingError(w, benchmark)`
  - `r_future`：真实未来 10 日收益（训练时已知）
  - `γ`：跟踪误差惩罚强度，控制与 benchmark 的偏离
- **联合训练**：GRU 参数和优化层一起优化，梯度通过 cvxpylayers 回传
- **冻结评估**：训练完成后提取 Smart Factor，独立评估

### 6.3 时间切分（防泄漏）

| 区间 | 用途 |
|------|------|
| 2018-01 ~ 2021-12 | 训练集 |
| 2022-01 ~ 2022-12 | 验证集（调参） |
| 2023-01 ~ 2024-12 | 测试集（最终报告） |

---

## 7. 回测引擎（backtest/engine.py）

### 7.1 核心逻辑

- **日频截面调仓**：每天开盘前根据前一日收盘后的预测信号生成权重
- **收益计算**：按收盘价计算持仓收益，`portfolio_return = sum(w_i * r_i)`
- **费后净值**：扣除双边手续费（假设 0.15%）和冲击成本（简化按 turnover * 0.1%）
- **Benchmark**：沪深300指数同期收益

### 7.2 统计指标

| 指标 | 说明 |
|------|------|
| 累计超额收益 | 组合净值 / 基准净值 - 1 |
| 年化超额收益 | 几何年化 |
| 跟踪误差（TE）| 超额收益序列的年化标准差 |
| 信息比率（IR）| 年化超额收益 / 跟踪误差 |
| 最大回撤 | 超额收益曲线的最大回撤 |
| 胜率 | 月度/日度超额收益为正的比例 |
| 换手率 | 日均权重变化 L1 范数 |

---

## 8. 里程碑与执行顺序

按 **"先因子，后组合"** 的顺序推进，确保每一步有可验证产出：

| 阶段 | 内容 | 产出 |
|------|------|------|
| **P0：数据基建** | tushare 拉取、builder 构造 X/M/y、Dataset | 可用数据 pipeline |
| **P1：基线因子** | GRU 训练、evaluator 评估、特征工程 | 可交易 GRU 因子 + 评估报告 |
| **P2：Two-Stage 组合** | constraints + cvxpy 优化 + 回测引擎 | 指数增强组合 + 表现报告 |
| **P3：PortfolioNet** | cvxpylayers 集成、联合训练、Smart Factor 提取 | 端到端组合 + Smart Factor 评估 |
| **P4：沉淀交付** | 代码文档、因子库报告、组合报告 | 完整交付物 |

---

## 9. 技术栈

| 用途 | 工具 |
|------|------|
| 深度学习框架 | PyTorch |
| 凸优化 | cvxpy + cvxpylayers |
| 数据处理 | pandas, numpy, pyarrow (parquet) |
| 数据获取 | tushare |
| 可视化 | matplotlib, seaborn |
| 开发环境 | Python 3.10+, 虚拟环境（venv/conda） |

---

## 10. 风险与简化项

| 风险/简化 | 应对策略 |
|------------|----------|
| tushare 免费版数据量/频率受限 | 先完成核心逻辑，正式入职后切公司数据源 |
| cvxpylayers 求解稳定性 | 监控 daily batch 求解时间，异常 fallback 到 Two-Stage |
| 训练时间长 | 先用 2018-2022 数据快速迭代，测试集最后跑 |
| 约束复杂度 | 先做核心 3 条（满仓、个股上限、行业偏离），逐步扩展 |
| 冲击成本模型简化 | 报告中注明简化假设，后续可用更精细模型替换 |

---

*Spec written. Ready for review.*
