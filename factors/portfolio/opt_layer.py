"""
PortfolioNet 可微组合优化层（OptLayer）。
使用 cvxpylayers 将 cvxpy 优化问题封装为 PyTorch 可微层，
实现 GRU 预测信号 → 优化权重 → 梯度回传的端到端链路。
"""

import warnings
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

try:
    from cvxpylayers.torch import CvxpyLayer
except ImportError:
    CvxpyLayer = None
    print("[WARN] cvxpylayers 未安装，PortfolioNet 无法使用。请先安装 cvxpylayers。")


class OptLayer(nn.Module):
    """
    可微组合优化层。
    对每一天的截面，输入 r_hat 和约束参数，输出组合权重 w。
    反向传播时梯度自动流回 GRU。
    支持可变股票数量（<= max_n_stocks），内部固定维度做 padding。
    """

    def __init__(
        self,
        max_n_stocks: int,
        n_sectors: int,
    ):
        super().__init__()
        self.max_n = max_n_stocks
        self.n_sectors = n_sectors

        if CvxpyLayer is None:
            raise ImportError("cvxpylayers 未安装")

        # 定义 cvxpy 变量和参数（按最大维度）
        w = cp.Variable(max_n_stocks)
        r_hat_param = cp.Parameter(max_n_stocks)
        A_param = cp.Parameter((n_sectors, max_n_stocks))
        b_param = cp.Parameter(n_sectors)
        l_param = cp.Parameter(max_n_stocks)
        u_param = cp.Parameter(max_n_stocks)
        delta_param = cp.Parameter(1)

        # 优化目标：最大化 r_hat^T w（与 Two-Stage 一致）
        objective = cp.Maximize(r_hat_param @ w)

        constraints = [
            cp.sum(w) == 1.0,
            w >= l_param,
            w <= u_param,
            A_param @ w >= b_param - delta_param,
            A_param @ w <= b_param + delta_param,
        ]

        problem = cp.Problem(objective, constraints)

        self.layer = CvxpyLayer(
            problem,
            parameters=[r_hat_param, A_param, b_param, l_param, u_param, delta_param],
            variables=[w],
        )

    def forward(
        self,
        r_hat: torch.Tensor,
        A_sector: torch.Tensor,
        b_sector: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            r_hat: (n_stocks,) 预测收益，n_stocks <= max_n
            A_sector: (n_sectors, n_stocks)
            b_sector: (n_sectors,)
            l, u: (n_stocks,)
            delta: scalar tensor 或 (1,)

        返回：
            w: (n_stocks,) 组合权重
        """
        n = r_hat.shape[0]
        device = r_hat.device
        if n < self.max_n:
            pad = self.max_n - n
            r_hat = torch.cat([r_hat, torch.zeros(pad, device=device)])
            A_sector = torch.cat([A_sector, torch.zeros(self.n_sectors, pad, device=device)], dim=1)
            l = torch.cat([l, torch.zeros(pad, device=device)])
            u = torch.cat([u, torch.zeros(pad, device=device)])
        elif n > self.max_n:
            r_hat = r_hat[:self.max_n]
            A_sector = A_sector[:, :self.max_n]
            l = l[:self.max_n]
            u = u[:self.max_n]

        w_full, = self.layer(r_hat, A_sector, b_sector, l, u, delta)
        return w_full[:n]


class PortfolioNet(nn.Module):
    """
    端到端 PortfolioNet：GRU + OptLayer。
    训练目标：组合级损失。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_n_stocks: int = 300,
        n_sectors: int = 30,
        gamma: float = 1.0,
    ):
        from factors.alpha.gru_model import GRUAlphaModel
        super().__init__()
        self.gru = GRUAlphaModel(input_size, hidden_size, num_layers, dropout)
        self.opt_layer = OptLayer(max_n_stocks, n_sectors)
        self.gamma = gamma

    def forward(
        self,
        X: torch.Tensor,          # (n_stocks, time_window, input_size)
        constraints: Dict,
        r_future: torch.Tensor,   # (n_stocks,) 真实未来收益（训练时）
        benchmark: torch.Tensor,  # (n_stocks,) 基准权重
    ) -> Dict[str, torch.Tensor]:
        """
        端到端 forward。
        返回字典包含 w, portfolio_return, tracking_error, loss。
        """
        # 1. GRU 预测
        r_hat = self.gru(X)  # (n_stocks,)

        # 2. OptLayer 生成权重
        A = torch.from_numpy(constraints["A_sector"]).float().to(X.device)
        b = torch.from_numpy(constraints["b_sector"]).float().to(X.device)
        l = torch.from_numpy(constraints["l"]).float().to(X.device)
        u = torch.from_numpy(constraints["u"]).float().to(X.device)
        delta = torch.tensor([constraints["sector_deviation"]], dtype=torch.float32, device=X.device)

        w = self.opt_layer(r_hat, A, b, l, u, delta)  # (n_stocks,)

        # 3. 组合表现
        w = w.to(r_future.dtype)
        portfolio_return = w @ r_future  # scalar
        tracking_error = torch.mean((w - benchmark) ** 2)

        # 4. 组合级损失
        loss = -portfolio_return + self.gamma * tracking_error

        return {
            "w": w,
            "r_hat": r_hat,
            "portfolio_return": portfolio_return,
            "tracking_error": tracking_error,
            "loss": loss,
        }


if __name__ == "__main__":
    if CvxpyLayer is None:
        print("cvxpylayers 未安装，跳过测试。")
    else:
        # 简单测试
        n = 10
        n_sec = 3
        opt = OptLayer(n, n_sec)
        r_hat = torch.randn(n)
        A = torch.randn(n_sec, n)
        b = torch.ones(n_sec) / n_sec
        l = torch.zeros(n)
        u = torch.full((n,), 0.5)
        delta = torch.tensor([0.1])
        w = opt(r_hat, A, b, l, u, delta)
        print("w sum:", w.sum().item())
        print("w shape:", w.shape)
