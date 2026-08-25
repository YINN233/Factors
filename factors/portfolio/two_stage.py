"""
Two-Stage 组合优化：外部 cvxpy 优化器。
每天截面独立求解指数增强组合权重。
"""

from typing import Dict, Optional

import cvxpy as cp
import numpy as np


def optimize_daily(
    r_hat: np.ndarray,
    constraints: Dict,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    求解单日指数增强组合优化问题。

    参数：
        r_hat: (n_stocks,) 预测收益
        constraints: 来自 IndexEnhancementConstraints.build() 的字典
        verbose: 是否打印求解器日志

    返回：
        w: (n_stocks,) 组合权重，若求解失败返回 None
    """
    n = constraints["n_stocks"]
    A = constraints["A_sector"]      # (n_sectors, n_stocks)
    b = constraints["b_sector"]      # (n_sectors,)
    l = constraints["l"]             # (n_stocks,)
    u = constraints["u"]             # (n_stocks,)
    delta = constraints["sector_deviation"]

    w = cp.Variable(n)

    # 目标：最大化预期收益
    objective = cp.Maximize(r_hat @ w)

    # 约束
    cons = [
        cp.sum(w) == 1.0,          # 满仓
        w >= l,                    # 下限（含非负）
        w <= u,                    # 上限
        A @ w >= b - delta,      # 行业下限偏离
        A @ w <= b + delta,      # 行业上限偏离
    ]

    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception as e:
        # fallback 到默认求解器
        try:
            prob.solve(verbose=verbose)
        except Exception:
            print(f"[Two-Stage] 求解失败: {e}")
            return None

    if w.value is None:
        print("[Two-Stage] 求解无最优解")
        return None

    return np.array(w.value).flatten()


if __name__ == "__main__":
    # 简单测试
    np.random.seed(0)
    n = 300
    r_hat = np.random.randn(n)
    n_sectors = 10
    A = np.random.randint(0, 2, size=(n_sectors, n)).astype(np.float32)
    # 确保每只股票至少属于一个行业
    for j in range(n):
        if A[:, j].sum() == 0:
            A[0, j] = 1.0
    b = np.ones(n_sectors) / n_sectors
    cons = {
        "n_stocks": n,
        "A_sector": A,
        "b_sector": b,
        "l": np.zeros(n),
        "u": np.full(n, 0.02),
        "sector_deviation": 0.03,
    }
    w = optimize_daily(r_hat, cons)
    print("权重和:", w.sum())
    print("权重范围:", w.min(), "~", w.max())
    print("非零个数:", (w > 1e-6).sum())
