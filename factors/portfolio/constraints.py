"""
约束生成模块：根据每天的约束特征 M，生成优化问题所需的矩阵和边界。
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class IndexEnhancementConstraints:
    """
    指数增强约束生成器。
    输入：当天所有股票的 DataFrame（含 industry, log_mv, index_weight）
    输出：字典，包含优化问题所需的全部参数。
    """

    def __init__(
        self,
        max_stock_weight: float = 0.02,
        sector_deviation: float = 0.03,
        weight_deviation: float = 0.01,  # 预留，暂未启用
        turnover_limit: float = 0.20,    # 预留，暂未启用
    ):
        self.max_stock_weight = max_stock_weight
        self.sector_deviation = sector_deviation
        self.weight_deviation = weight_deviation
        self.turnover_limit = turnover_limit

    def build(
        self,
        df: pd.DataFrame,
        industry_col: str = "industry",
        index_weight_col: str = "index_weight",
    ) -> Dict:
        """
        生成当日约束参数。
        返回字典：
            n_stocks: 股票数量
            A_sector: (n_sectors, n_stocks) 行业归属矩阵
            b_sector: (n_sectors,) 基准行业权重（按 index_weight 加权）
            l: (n_stocks,) 个股下限（默认 0）
            u: (n_stocks,) 个股上限
            sector_deviation: 行业偏离边界
        """
        n_stocks = len(df)
        industries = sorted(df[industry_col].unique())
        n_sectors = len(industries)
        ind_to_idx = {ind: i for i, ind in enumerate(industries)}

        # 行业归属矩阵
        A_sector = np.zeros((n_sectors, n_stocks), dtype=np.float32)
        for j, ind in enumerate(df[industry_col].values):
            A_sector[ind_to_idx[ind], j] = 1.0

        # 基准行业权重 = 每只股票的 index_weight 按行业汇总后归一化
        index_weights = df[index_weight_col].values.astype(np.float32)
        b_sector = A_sector @ index_weights  # (n_sectors,)
        # 归一化（理论上 index_weights 之和应为 1，但可能有微小偏差）
        b_sector = b_sector / (b_sector.sum() + 1e-9)

        # 个股上下界
        l = np.zeros(n_stocks, dtype=np.float32)
        u = np.full(n_stocks, self.max_stock_weight, dtype=np.float32)

        return {
            "n_stocks": n_stocks,
            "n_sectors": n_sectors,
            "A_sector": A_sector,
            "b_sector": b_sector,
            "l": l,
            "u": u,
            "sector_deviation": self.sector_deviation,
        }


if __name__ == "__main__":
    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"],
        "industry": ["银行", "房地产", "银行", "科技"],
        "index_weight": [0.01, 0.005, 0.008, 0.0],
    })
    cons = IndexEnhancementConstraints()
    params = cons.build(df)
    print("n_stocks:", params["n_stocks"])
    print("n_sectors:", params["n_sectors"])
    print("A_sector shape:", params["A_sector"].shape)
    print("b_sector:", params["b_sector"])
    print("u:", params["u"])
