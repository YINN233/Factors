"""
Two-Stage 基线训练脚本：
1. 训练 GRU 预测模型（MSE + IC 损失）
2. 保存预测结果
3. 因子评估
4. Two-Stage 组合优化 + 回测
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from factors.alpha.gru_model import GRUAlphaModel, mse_ic_loss
from factors.alpha.evaluator import FactorEvaluator
from factors.data.dataset import AlphaDataset
from factors.portfolio.constraints import IndexEnhancementConstraints
from factors.portfolio.two_stage import optimize_daily
from factors.backtest.engine import BacktestEngine


def train_epoch(model, dataset, optimizer, device, ic_lambda=0.3):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ic = 0.0
    n_batches = 0

    dates = list(dataset.samples_by_date.keys())
    np.random.shuffle(dates)

    for date in dates:
        batch = dataset.get_batch_by_date(date)
        if batch is None:
            continue
        X, M, y, mask, codes = batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        r_hat = model(X)  # (n_stocks,)

        # 过滤停牌/缺失标签的股票
        valid = ~torch.isnan(y)
        if valid.sum() < 3:
            continue
        r_hat_valid = r_hat[valid]
        y_valid = y[valid]

        loss, mse, ic = mse_ic_loss(r_hat_valid, y_valid, ic_lambda=ic_lambda)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_ic += ic.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mse / max(n_batches, 1), total_ic / max(n_batches, 1)


def evaluate_epoch(model, dataset, device):
    model.eval()
    records = []
    dates = sorted(dataset.samples_by_date.keys())

    with torch.no_grad():
        for date in dates:
            batch = dataset.get_batch_by_date(date)
            if batch is None:
                continue
            X, M, y, mask, codes = batch
            X = X.to(device)
            y = y.to(device)
            r_hat = model(X)

            valid = ~torch.isnan(y)
            for i in range(len(codes)):
                records.append({
                    "trade_date": date,
                    "ts_code": codes[i],
                    "factor": float(r_hat[i].cpu()),
                    "label": float(y[i].cpu()),
                })

    df = pd.DataFrame(records)
    return df


def run_two_stage_backtest(pred_df, split, fee_rate=0.0015):
    """
    对预测结果做 Two-Stage 优化 + 回测。
    pred_df 需要包含 trade_date, ts_code, factor, label, 以及约束信息。
    """
    from factors.data.builder import PROCESSED_DIR

    # 加载原始 processed 数据获取约束信息
    proc_df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
    proc_df = proc_df[["trade_date", "ts_code", "industry", "index_weight", "log_mv"]].copy()
    proc_df["trade_date"] = pd.to_datetime(proc_df["trade_date"])

    pred_df = pred_df.merge(proc_df, on=["trade_date", "ts_code"], how="left")

    cons_gen = IndexEnhancementConstraints()
    weights_records = []

    dates = sorted(pred_df["trade_date"].unique())
    for date in tqdm(dates, desc=f"Two-Stage optimize ({split})"):
        sub = pred_df[pred_df["trade_date"] == date]
        sub = sub.dropna(subset=["factor"])
        if len(sub) < 5:
            continue

        constraints = cons_gen.build(sub, industry_col="industry", index_weight_col="index_weight")
        r_hat = sub["factor"].values.astype(np.float32)
        w = optimize_daily(r_hat, constraints)
        if w is None:
            continue

        for i, row in sub.iterrows():
            weights_records.append({
                "trade_date": date,
                "ts_code": row["ts_code"],
                "weight": float(w[i]),
            })

    weights_df = pd.DataFrame(weights_records)
    if weights_df.empty:
        print("[Two-Stage] 无有效权重，跳过回测。")
        return None

    # 回测
    prices_df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")[["trade_date", "ts_code", "close_adj"]]
    prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"])
    bm_df = prices_df[["trade_date", "ts_code"]].copy()
    bm_df = bm_df.merge(
        pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")[["trade_date", "ts_code", "index_weight"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    bm_df["index_weight"] = bm_df["index_weight"].fillna(0.0)

    engine = BacktestEngine(weights_df, prices_df, bm_df, fee_rate=fee_rate)
    daily_df = engine.run()
    stats = engine.summary()
    return daily_df, stats, weights_df


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_ds = AlphaDataset("train", time_window=args.window)
    valid_ds = AlphaDataset("valid", time_window=args.window)

    # 模型
    model = GRUAlphaModel(
        input_size=len(train_ds.feature_cols),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_valid_loss = float("inf")
    output_dir = Path("outputs/two_stage")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mse, train_ic = train_epoch(
            model, train_ds, optimizer, device, ic_lambda=args.ic_lambda
        )
        valid_df = evaluate_epoch(model, valid_ds, device)
        # 验证集上计算 IC
        valid_df = valid_df.dropna()
        valid_ic = valid_df.groupby("trade_date").apply(
            lambda x: x["factor"].corr(x["label"], method="pearson")
        ).mean()
        valid_mse = ((valid_df["factor"] - valid_df["label"]) ** 2).mean()

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} MSE: {train_mse:.4f} IC: {train_ic:.4f} | "
              f"Valid MSE: {valid_mse:.4f} IC: {valid_ic:.4f}")

        scheduler.step()

        # 保存最优模型
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), output_dir / "best_gru.pt")

    # 最终推断
    print("\n🔄 最终推断（valid + test）...")
    test_ds = AlphaDataset("test", time_window=args.window)
    model.load_state_dict(torch.load(output_dir / "best_gru.pt", map_location=device))

    for split, ds in [("valid", valid_ds), ("test", test_ds)]:
        pred_df = evaluate_epoch(model, ds, device)
        pred_df.to_parquet(output_dir / f"predictions_{split}.parquet", index=False)

        # 因子评估
        print(f"\n📊 {split.upper()} 因子评估")
        ev = FactorEvaluator(pred_df, factor_col="factor", label_col="label")
        ev.evaluate(n_groups=5, save_dir=output_dir / f"eval_{split}")

        # Two-Stage 组合优化 + 回测
        print(f"\n📈 {split.upper()} Two-Stage 组合回测")
        result = run_two_stage_backtest(pred_df, split, fee_rate=args.fee)
        if result:
            daily_df, stats, weights_df = result
            print(f"组合统计: {stats}")
            daily_df.to_csv(output_dir / f"backtest_{split}.csv", index=False)
            weights_df.to_parquet(output_dir / f"weights_{split}.parquet", index=False)

    print("\n✅ Two-Stage 训练与回测完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ic_lambda", type=float, default=0.3)
    parser.add_argument("--fee", type=float, default=0.0015)
    args = parser.parse_args()
    main(args)
