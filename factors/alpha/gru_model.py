"""
GRU 预测网络：输入量价面板 (time_window, features)，输出单只股票未来收益预测 r_hat。
Two-Stage 和 PortfolioNet 共享此模型。
"""

from typing import Optional

import torch
import torch.nn as nn


class GRUAlphaModel(nn.Module):
    """
    参数：
        input_size: 特征维度（由 meta.json 决定，约 20+）
        hidden_size: GRU 隐层维度，默认 64
        num_layers: GRU 层数，默认 2
        dropout: dropout 率，默认 0.2
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time_window, input_size) 或 (time_window, input_size)
        返回: (batch, 1) 或 (1,)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, T, F)

        # gru_out: (batch, T, hidden), hn: (num_layers, batch, hidden)
        gru_out, hn = self.gru(x)
        # 取最后一个时间步的输出
        last_out = gru_out[:, -1, :]  # (batch, hidden)
        r_hat = self.fc(last_out).squeeze(-1)  # (batch,)
        return r_hat


def mse_ic_loss(
    r_hat: torch.Tensor,
    y: torch.Tensor,
    ic_lambda: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    组合损失：MSE + λ * (-IC)。
    IC 计算需要 r_hat 和 y 在同一截面上（同一天的所有股票）。
    因此调用此函数时，传入的 r_hat 和 y 应为 **同一天** 的所有股票。

    参数：
        r_hat: (n_stocks,) 当天所有股票的预测值
        y: (n_stocks,) 当天所有股票的真实标签
        ic_lambda: IC 损失的权重
    """
    mse = torch.mean((r_hat - y) ** 2)

    # 计算截面 Pearson IC
    r_hat_centered = r_hat - r_hat.mean()
    y_centered = y - y.mean()
    numerator = torch.sum(r_hat_centered * y_centered)
    denominator = torch.sqrt(torch.sum(r_hat_centered ** 2) * torch.sum(y_centered ** 2)) + eps
    ic = numerator / denominator

    loss = mse - ic_lambda * ic
    return loss, mse, ic


def build_model_from_meta(hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
    """从 meta.json 读取特征维度，自动构造模型。"""
    import json
    from pathlib import Path

    meta_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    input_size = len(meta["feature_cols"])
    return GRUAlphaModel(input_size, hidden_size, num_layers, dropout)


if __name__ == "__main__":
    model = GRUAlphaModel(input_size=20, hidden_size=64, num_layers=2)
    x = torch.randn(4, 20, 20)  # (batch=4, T=20, F=20)
    out = model(x)
    print("output shape:", out.shape)  # (4,)
