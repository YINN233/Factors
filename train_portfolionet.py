"""
PortfolioNet 端到端训练脚本：
1. 联合训练 GRU + OptLayer（组合级损失）
2. 提取 Smart Factor（GRU 输出 r_hat）
3. 因子评估 + 回测
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from factors.portfolio.opt_layer import PortfolioNet
from factors.alpha.evaluator import FactorEvaluator
from factors.data.dataset import AlphaDataset
from factors.portfolio.constraints import IndexEnhancementConstraints
from factors.backtest.engine import BacktestEngine


def build_constraints_from_batch(M: torch.Tensor, max_stock_weight: float = 0.02, sector_deviation: float = 0.03):
    """
    从 M 解析约束参数。
    M: (n_stocks, 2 + n_industry)
      M[:, 0] = log_mv
      M[:, 1] = index_weight
      M[:, 2:] = industry_onehot
    """
    n_stocks = M.shape[0]
    index_weight = M[:, 1].cpu().numpy()
    industry_onehot = M[:, 2:].cpu().numpy()  # (n_stocks, n_industry)

    # A_sector: (n_industry, n_stocks)
    A_sector = industry_onehot.T.astype(np.float32)
    b_sector = A_sector @ index_weight
    b_sector = b_sector / (b_sector.sum() + 1e-9)

    l = np.zeros(n_stocks, dtype=np.float32)
    u = np.full(n_stocks, max_stock_weight, dtype=np.float32)

    return {
        "n_stocks": n_stocks,
        "n_sectors": A_sector.shape[0],
        "A_sector": A_sector,
        "b_sector": b_sector,
        "l": l,
        "u": u,
        "sector_deviation": sector_deviation,
    }


def train_epoch(model, dataset, optimizer, device, gamma=1.0):
    model.train()
    total_loss = 0.0
    total_port_ret = 0.0
    total_te = 0.0
    n_batches = 0

    dates = list(dataset.samples_by_date.keys())
    np.random.shuffle(dates)

    for date in dates:
        batch = dataset.get_batch_by_date(date)
        if batch is None:
            continue
        X, M, y, mask, codes = batch
        X = X.to(device)
        M = M.to(device)
        y = y.to(device)

        # 过滤停牌/缺失
        valid = ~torch.isnan(y)
        if valid.sum() < 5:
            continue
        X = X[valid]
        M = M[valid]
        y = y[valid]

        # 构建约束
        constraints = build_constraints_from_batch(M)
        benchmark = M[:, 1]  # index_weight 作为 benchmark 权重，归一化
        benchmark = benchmark / (benchmark.sum() + 1e-9)

        # 组合优化目标：最大化组合收益（用真实 y 代替未来收益）
        # PortfolioNet 的 forward 需要 r_future，这里用 y 充当
        result = model(X, constraints, y, benchmark)

        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_port_ret += result["portfolio_return"].item()
        total_te += result["tracking_error"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_port_ret / max(n_batches, 1), total_te / max(n_batches, 1)


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
            M = M.to(device)
            y = y.to(device)

            valid = ~torch.isnan(y)
            X_valid = X[valid]
            M_valid = M[valid]
            y_valid = y[valid]
            codes_valid = [codes[i] for i in range(len(codes)) if valid[i]]

            if len(codes_valid) < 5:
                continue

            constraints = build_constraints_from_batch(M_valid)
            benchmark = M_valid[:, 1]
            benchmark = benchmark / (benchmark.sum() + 1e-9)

            result = model(X_valid, constraints, y_valid, benchmark)
            r_hat = result["r_hat"]
            w = result["w"]

            for i, code in enumerate(codes_valid):
                records.append({
                    "trade_date": date,
                    "ts_code": code,
                    "smart_factor": float(r_hat[i].cpu()),
                    "weight": float(w[i].cpu()),
                    "label": float(y_valid[i].cpu()),
                })

    df = pd.DataFrame(records)
    return df


def run_portfolio_backtest(pred_df, split, fee_rate=0.0015):
    """
    直接用 PortfolioNet 输出的 weight 做回测（无需二次优化）。
    """
    from factors.data.builder import PROCESSED_DIR

    prices_df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")[["trade_date", "ts_code", "close_adj"]]
    prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"])

    bm_df = prices_df[["trade_date", "ts_code"]].copy()
    bm_df = bm_df.merge(
        pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")[["trade_date", "ts_code", "index_weight"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    bm_df["index_weight"] = bm_df["index_weight"].fillna(0.0)

    weights_df = pred_df[["trade_date", "ts_code", "weight"]].copy()
    # 每天归一化（PortfolioNet 理论上已归一化，但数值精度可能导致微小偏差）
    weights_df = weights_df.groupby("trade_date").apply(lambda x: x.assign(weight=x["weight"] / x["weight"].sum()))
    weights_df = weights_df.reset_index(drop=True)

    engine = BacktestEngine(weights_df, prices_df, bm_df, fee_rate=fee_rate)
    daily_df = engine.run()
    stats = engine.summary()
    return daily_df, stats, weights_df


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = AlphaDataset("train", time_window=args.window)
    valid_ds = AlphaDataset("valid", time_window=args.window)
    test_ds = AlphaDataset("test", time_window=args.window)

    # 获取行业数量
    n_industry = train_ds.n_industry
    n_stocks_approx = 300  # 沪深300成分股数，OptLayer 初始化用（实际求解时按当天实际数量）

    model = PortfolioNet(
        input_size=len(train_ds.feature_cols),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        max_n_stocks=n_stocks_approx,
        n_sectors=n_industry,
        gamma=args.gamma,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")
    output_dir = Path("outputs/portfolionet")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_port_ret, train_te = train_epoch(
            model, train_ds, optimizer, device, gamma=args.gamma
        )
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} PortRet: {train_port_ret:.4f} TE: {train_te:.4f}")
        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), output_dir / "best_portfolionet.pt")

    # 最终推断
    print("\n🔄 最终推断（valid + test）...")
    model.load_state_dict(torch.load(output_dir / "best_portfolionet.pt", map_location=device))

    for split, ds in [("valid", valid_ds), ("test", test_ds)]:
        pred_df = evaluate_epoch(model, ds, device)
        pred_df.to_parquet(output_dir / f"predictions_{split}.parquet", index=False)

        # Smart Factor 评估
        print(f"\n📊 {split.upper()} Smart Factor 评估")
        ev = FactorEvaluator(pred_df, factor_col="smart_factor", label_col="label")
        ev.evaluate(n_groups=5, save_dir=output_dir / f"eval_{split}")

        # PortfolioNet 回测
        print(f"\n📈 {split.upper()} PortfolioNet 回测")
        daily_df, stats, weights_df = run_portfolio_backtest(pred_df, split, fee_rate=args.fee)
        print(f"组合统计: {stats}")
        daily_df.to_csv(output_dir / f"backtest_{split}.csv", index=False)
        weights_df.to_parquet(output_dir / f"weights_{split}.parquet", index=False)

    print("\n✅ PortfolioNet 训练与回测完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fee", type=float, default=0.0015)
    args = parser.parse_args()
    main(args)
