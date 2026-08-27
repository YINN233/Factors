"""
PyTorch Dataset：将构造好的 parquet 数据转为模型输入。
每个 sample = 一只股票在某一天的 (X_panel, M, y)。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class AlphaDataset(Dataset):
    """
    参数：
        split: 'train' | 'valid' | 'test'
        time_window: 历史回看天数（默认 20）
    输出：
        X: (time_window, n_features) 面板特征
        M: (n_constraint_features,) 约束特征（log_mv, industry_onehot, index_weight）
        y: scalar 标签
        valid_mask: (time_window,) bool，True 表示该天该股票正常交易
    """

    def __init__(
        self,
        split: str = "train",
        time_window: int = 20,
        use_rank_label: bool = False,
    ):
        assert split in ("train", "valid", "test")
        self.split = split
        self.time_window = time_window
        self.use_rank_label = use_rank_label

        # 加载 parquet
        self.df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
        self.df = self.df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 加载元数据
        with open(PROCESSED_DIR / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_cols: List[str] = meta["feature_cols"]
        self.constraint_cols: List[str] = meta["constraint_cols"]

        # 行业编码（使用完整行业列表，保证 train/valid/test 一致）
        self.industry_codes = meta.get("industry_list", sorted(self.df["industry"].unique().tolist()))
        self.industry_to_idx = {ind: i for i, ind in enumerate(self.industry_codes)}
        self.n_industry = len(self.industry_codes)

        # 建立快速索引：ts_code -> 该股票所有日期的 DataFrame（按 trade_date 排序）
        self.stock_frames: Dict[str, pd.DataFrame] = {}
        for code, sub in self.df.groupby("ts_code"):
            sub = sub.sort_values("trade_date").reset_index(drop=True)
            self.stock_frames[code] = sub

        # 建立样本列表：(ts_code, trade_date_index_in_stock_frame)
        self.samples: List[Tuple[str, int]] = []
        self.samples_by_date: Dict[pd.Timestamp, List[int]] = {}
        for code, sub in self.stock_frames.items():
            n = len(sub)
            # 至少需要 time_window 个历史数据才能构造面板
            for i in range(time_window, n):
                # 同时要求当天标签不为空
                if not pd.isna(sub.iloc[i]["label"]):
                    idx = len(self.samples)
                    self.samples.append((code, i))
                    date = sub.iloc[i]["trade_date"]
                    self.samples_by_date.setdefault(date, []).append(idx)

        print(f"[{split}] 构建完成：{len(self.samples)} 个样本，{len(self.samples_by_date)} 个交易日，{len(self.feature_cols)} 个特征，{self.n_industry} 个行业")

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch_by_date(self, date: pd.Timestamp) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]:
        """
        获取某一天的全部样本，拼接为 batch。
        返回: X_batch, M_batch, y_batch, mask_batch, ts_codes_list
        """
        indices = self.samples_by_date.get(date)
        if not indices:
            return None
        batch = [self[i] for i in indices]
        X = torch.stack([b[0] for b in batch])   # (n_stocks, T, F)
        M = torch.stack([b[1] for b in batch])     # (n_stocks, M_dim)
        y = torch.stack([b[2] for b in batch])     # (n_stocks,)
        mask = torch.stack([b[3] for b in batch])  # (n_stocks, T)
        codes = [self.samples[i][0] for i in indices]
        return X, M, y, mask, codes

    def _get_industry_onehot(self, industry: str) -> np.ndarray:
        vec = np.zeros(self.n_industry, dtype=np.float32)
        vec[self.industry_to_idx.get(industry, self.industry_to_idx.get("未知", 0))] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        code, pos = self.samples[idx]
        sub = self.stock_frames[code]

        # 取过去 time_window 天的数据 [pos - time_window, pos)（不含当天）
        # 或者 [pos - time_window + 1, pos]（含当天）— 这里选择含当天
        start = pos - self.time_window + 1
        panel = sub.iloc[start : pos + 1]

        # X 面板
        X = panel[self.feature_cols].values.astype(np.float32)  # (time_window, n_features)
        # 缺失值填 0（截面归一化后已处理大部分，但停牌日可能仍有 nan）
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # valid_mask：当天是否有效（close_adj 不为 nan 且 volume > 0）
        valid_mask = (~panel["close_adj"].isna() & (panel["volume"] > 0)).values.astype(np.float32)

        # M：约束特征（取当天的值）
        row = sub.iloc[pos]
        log_mv = float(row["log_mv"]) if not pd.isna(row["log_mv"]) else 0.0
        index_weight = float(row["index_weight"])
        industry_onehot = self._get_industry_onehot(row["industry"])
        M = np.concatenate([
            np.array([log_mv, index_weight], dtype=np.float32),
            industry_onehot,
        ])  # (2 + n_industry,)

        # y
        y_val = float(row["label_rank"]) if self.use_rank_label else float(row["label"])
        if np.isnan(y_val):
            y_val = 0.0

        return (
            torch.from_numpy(X),           # (T, F)
            torch.from_numpy(M),           # (M_dim,)
            torch.tensor(y_val, dtype=torch.float32),
            torch.from_numpy(valid_mask),  # (T,)
        )


def collate_fn(batch):
    """
    将 list of (X, M, y, mask) 拼接为 batch。
    由于每天股票数量不同，这里先返回 list，后续在训练/推断时按日截面重组。
    """
    # 直接返回 list，上层按日截面聚合
    return batch


if __name__ == "__main__":
    ds = AlphaDataset("train", time_window=20)
    X, M, y, mask = ds[0]
    print("X shape:", X.shape)
    print("M shape:", M.shape)
    print("y:", y.item())
    print("mask shape:", mask.shape)
