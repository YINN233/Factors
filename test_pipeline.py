"""
端到端 smoke test：用合成数据跑通 Two-Stage 和 PortfolioNet 的全流程。
用于快速验证代码逻辑和维度匹配，无需真实 tushare 数据。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from factors.alpha.gru_model import GRUAlphaModel, mse_ic_loss
from factors.alpha.evaluator import FactorEvaluator
from factors.data.dataset import AlphaDataset
from factors.portfolio.constraints import IndexEnhancementConstraints
from factors.portfolio.two_stage import optimize_daily
from factors.portfolio.opt_layer import PortfolioNet
from factors.backtest.engine import BacktestEngine


def generate_synthetic_data(n_stocks: int = 50, n_days: int = 100, n_features: int = 20):
    """生成合成 processed 数据，直接放到 data/processed/。"""
    PROCESSED_DIR = Path("data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    industries = ["银行", "房地产", "科技", "消费", "医药"]

    np.random.seed(42)
    records = []
    for date in dates:
        for i in range(n_stocks):
            records.append({
                "ts_code": f"{i:06d}.SZ",
                "trade_date": date,
                "open_adj": 10.0 + np.random.randn(),
                "high_adj": 11.0 + np.random.randn(),
                "low_adj": 9.0 + np.random.randn(),
                "close_adj": 10.0 + np.random.randn() * 0.02,
                "volume": np.random.randint(1e6, 1e8),
                "amount": np.random.randint(1e7, 1e9),
                "total_mv": np.random.uniform(1e10, 1e12),
                "turnover_rate": np.random.uniform(0, 5),
                "industry": industries[i % len(industries)],
                "index_weight": np.random.uniform(0, 0.01),
                "log_mv": np.log(np.random.uniform(1e10, 1e12)),
                # 预计算特征（builder 里会计算，这里直接放归一化后的占位值）
                "returns_1d": np.random.randn() * 0.02,
                "volatility_5d": np.random.randn() * 0.01,
                "sma_5": 10.0,
                "sma_10": 10.0,
                "sma_20": 10.0,
                "ema_12": 10.0,
                "ema_26": 10.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "rsi_14": 50.0,
                "close_sma5_ratio": 1.0,
                "close_sma20_ratio": 1.0,
                "volume_change": 0.0,
                "label": np.random.randn() * 0.02,  # 合成标签
                "label_rank": np.random.rand(),
            })

    df = pd.DataFrame(records)
    # 让 factor 和 label 有点相关性
    df["label"] = df["returns_1d"] * 0.5 + np.random.randn(len(df)) * 0.01

    # 时间切分
    train = df[df["trade_date"] <= dates[60]].copy()
    valid = df[(df["trade_date"] > dates[60]) & (df["trade_date"] <= dates[80])].copy()
    test = df[df["trade_date"] > dates[80]].copy()

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    valid.to_parquet(PROCESSED_DIR / "valid.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    feature_cols = [
        "returns_1d", "volatility_5d", "sma_5", "sma_10", "sma_20",
        "ema_12", "ema_26", "macd", "macd_signal", "rsi_14",
        "close_sma5_ratio", "close_sma20_ratio", "volume_change",
    ]
    meta = {
        "feature_cols": feature_cols,
        "constraint_cols": ["log_mv", "industry", "index_weight"],
        "time_window": 20,
        "industry_list": industries,
    }
    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 合成数据生成完成：{len(train)} train / {len(valid)} valid / {len(test)} test")
    return PROCESSED_DIR


def smoke_test_two_stage():
    print("\n🧪 Smoke Test: Two-Stage")
    device = torch.device("cpu")

    ds = AlphaDataset("train", time_window=20)
    model = GRUAlphaModel(input_size=len(ds.feature_cols), hidden_size=32, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dates = list(ds.samples_by_date.keys())[:5]  # 只跑5天
    for date in dates:
        batch = ds.get_batch_by_date(date)
        if batch is None:
            continue
        X, M, y, mask, codes = batch
        X, y = X.to(device), y.to(device)
        r_hat = model(X)
        valid = ~torch.isnan(y)
        if valid.sum() < 3:
            continue
        loss, mse, ic = mse_ic_loss(r_hat[valid], y[valid], ic_lambda=0.3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 推断
    model.eval()
    records = []
    with torch.no_grad():
        for date in dates:
            batch = ds.get_batch_by_date(date)
            if batch is None:
                continue
            X, M, y, mask, codes = batch
            r_hat = model(X.to(device))
            for i, code in enumerate(codes):
                records.append({
                    "trade_date": date,
                    "ts_code": code,
                    "factor": float(r_hat[i].cpu()),
                    "label": float(y[i]),
                })
    pred_df = pd.DataFrame(records)

    # 评估
    ev = FactorEvaluator(pred_df)
    result = ev.evaluate(n_groups=5)
    print("因子评估通过")

    # 组合优化
    cons_gen = IndexEnhancementConstraints(max_stock_weight=0.05, sector_deviation=0.1)
    for date in dates[:3]:
        sub = pred_df[pred_df["trade_date"] == date]
        if len(sub) < 5:
            continue
        # 需要行业信息，临时补上
        sub = sub.merge(
            ds.df[["trade_date", "ts_code", "industry", "index_weight"]].drop_duplicates(),
            on=["trade_date", "ts_code"],
            how="left",
        )
        sub["industry"] = sub["industry"].fillna("其他")
        sub["index_weight"] = sub["index_weight"].fillna(0.0)
        constraints = cons_gen.build(sub)
        w = optimize_daily(sub["factor"].values.astype(np.float32), constraints)
        if w is not None:
            print(f"  {date.date()}: 权重和={w.sum():.4f}, 范围=[{w.min():.4f}, {w.max():.4f}]")

    print("✅ Two-Stage smoke test 通过")


def smoke_test_portfolionet():
    print("\n🧪 Smoke Test: PortfolioNet")
    device = torch.device("cpu")

    ds = AlphaDataset("train", time_window=20)
    n_industry = ds.n_industry

    model = PortfolioNet(
        input_size=len(ds.feature_cols),
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        max_n_stocks=50,
        n_sectors=n_industry,
        gamma=1.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dates = list(ds.samples_by_date.keys())[:3]
    for date in dates:
        batch = ds.get_batch_by_date(date)
        if batch is None:
            continue
        X, M, y, mask, codes = batch
        X, M, y = X.to(device), M.to(device), y.to(device)
        valid = ~torch.isnan(y)
        if valid.sum() < 5:
            continue
        X, M, y = X[valid], M[valid], y[valid]

        # 构造约束
        index_weight = M[:, 1].cpu().numpy()
        industry_onehot = M[:, 2:].cpu().numpy()
        A_sector = industry_onehot.T.astype(np.float32)
        b_sector = A_sector @ index_weight
        b_sector = b_sector / (b_sector.sum() + 1e-9)
        constraints = {
            "n_stocks": len(M),
            "n_sectors": A_sector.shape[0],
            "A_sector": A_sector,
            "b_sector": b_sector,
            "l": np.zeros(len(M), dtype=np.float32),
            "u": np.full(len(M), 0.05, dtype=np.float32),
            "sector_deviation": 0.1,
        }
        benchmark = M[:, 1]
        benchmark = benchmark / (benchmark.sum() + 1e-9)

        result = model(X, constraints, y, benchmark)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  {date.date()}: loss={loss.item():.4f}, port_ret={result['portfolio_return'].item():.4f}")

    print("✅ PortfolioNet smoke test 通过")


if __name__ == "__main__":
    generate_synthetic_data(n_stocks=50, n_days=100)
    smoke_test_two_stage()
    smoke_test_portfolionet()
    print("\n🎉 全部 smoke test 通过！代码逻辑和维度匹配正确。")
