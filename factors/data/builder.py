"""
数据构造模块：将原始数据合并为 X（特征面板）、M（约束特征）、y（标签）。
输出截面归一化后的 train/valid/test parquet。
"""

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_raw(
    start_date: str = "20180101",
    end_date: str = "20241231",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载全部原始 parquet 数据。"""
    idx_code_str = "000300_SH"
    index_weight = pd.read_parquet(
        RAW_DIR / f"index_weight_{idx_code_str}_{start_date}_{end_date}.parquet"
    )
    daily = pd.read_parquet(RAW_DIR / f"daily_{start_date}_{end_date}.parquet")
    daily_basic = pd.read_parquet(RAW_DIR / f"daily_basic_{start_date}_{end_date}.parquet")
    company = pd.read_parquet(RAW_DIR / "stock_company.parquet")
    adj_factor = pd.read_parquet(RAW_DIR / f"adj_factor_{start_date}_{end_date}.parquet")
    return index_weight, daily, daily_basic, company, adj_factor


def build_base_table(
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    adj_factor: pd.DataFrame,
    company: pd.DataFrame,
    index_weight: pd.DataFrame,
) -> pd.DataFrame:
    """
    合并 daily + daily_basic + adj_factor + company + index_weight，生成基础大表。
    关键步骤：
    1. daily 与 adj_factor merge 计算复权价
    2. 与 daily_basic merge 补充换手率、市值
    3. 与 company merge 补充行业
    4. 与 index_weight merge 补充成分股权重
    """
    daily = daily.rename(columns={"vol": "volume"}) if "vol" in daily.columns else daily
    if "volume" not in daily.columns:
        raise KeyError("daily data must contain 'volume' or tushare's 'vol' column")

    # 复权价格
    df = daily.merge(adj_factor[["ts_code", "trade_date", "adj_factor"]],
                     on=["ts_code", "trade_date"], how="left")
    for col in ["open", "high", "low", "close"]:
        df[f"{col}_adj"] = df[col] * df["adj_factor"]

    # 与 daily_basic 合并（保留估值与流动性字段，基本面挖掘会用到 pe/pb/dividend）
    basic_cols = [
        "ts_code", "trade_date", "total_mv", "circ_mv", "turnover_rate",
        "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm",
    ]
    basic_cols = [col for col in basic_cols if col in daily_basic.columns]
    df = df.merge(
        daily_basic[basic_cols],
        on=["ts_code", "trade_date"],
        how="left",
    )

    # 行业分类
    df = df.merge(company[["ts_code", "industry"]], on="ts_code", how="left")

    # 成分股权重。tushare 的 index_weight 通常是月度/调仓日记录，
    # 这里按股票使用最近一期可见权重向后对齐到日频。
    index_weight = index_weight.rename(columns={"con_code": "ts_code", "weight": "index_weight"})
    index_weight = index_weight[["ts_code", "trade_date", "index_weight"]].copy()
    index_weight["trade_date"] = pd.to_datetime(index_weight["trade_date"])
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    index_weight = index_weight.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    df = pd.merge_asof(
        df,
        index_weight,
        on="trade_date",
        by="ts_code",
        direction="backward",
    )
    # 非成分股 index_weight 填 0
    df["index_weight"] = df["index_weight"].fillna(0.0)

    # 只保留需要的列
    keep_cols = [
        "ts_code", "trade_date",
        "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount",
        "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
        "industry", "index_weight",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在基础表上计算技术指标和衍生特征。
    注意：所有计算只使用当日及之前数据，无未来信息。
    """
    df = df.copy()
    df["returns_1d"] = df.groupby("ts_code")["close_adj"].pct_change(1)

    # 波动率（5日）
    df["volatility_5d"] = df.groupby("ts_code")["returns_1d"].transform(
        lambda x: x.rolling(5).std()
    )

    # SMA
    df["sma_5"] = df.groupby("ts_code")["close_adj"].transform(lambda x: x.rolling(5).mean())
    df["sma_10"] = df.groupby("ts_code")["close_adj"].transform(lambda x: x.rolling(10).mean())
    df["sma_20"] = df.groupby("ts_code")["close_adj"].transform(lambda x: x.rolling(20).mean())

    # EMA (12, 26) —— 用于 MACD
    df["ema_12"] = df.groupby("ts_code")["close_adj"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df["ema_26"] = df.groupby("ts_code")["close_adj"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df.groupby("ts_code")["macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )

    # RSI(14)
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi_14"] = df.groupby("ts_code")["close_adj"].transform(_rsi)

    # 对数市值
    df["log_mv"] = np.log(df["total_mv"].replace(0, np.nan))

    # 价格/均线比值
    df["close_sma5_ratio"] = df["close_adj"] / df["sma_5"]
    df["close_sma20_ratio"] = df["close_adj"] / df["sma_20"]

    # volume 变化率
    df["volume_change"] = df.groupby("ts_code")["volume"].pct_change(1)

    return df


def compute_labels(df: pd.DataFrame, forward_days: int = 10) -> pd.DataFrame:
    """
    构造标签 y：未来 forward_days 个交易日（T+1 ~ T+forward_days）的累计收益。
    即 y = close_{T+forward_days} / close_{T+1} - 1
    """
    df = df.copy()
    # future close at T+forward_days
    df["future_close"] = df.groupby("ts_code")["close_adj"].shift(-forward_days)
    # tomorrow close at T+1
    df["next_close"] = df.groupby("ts_code")["close_adj"].shift(-1)
    df["label"] = df["future_close"] / df["next_close"] - 1.0
    df["label_rank"] = df.groupby("trade_date")["label"].transform(
        lambda x: x.rank(pct=True)
    )
    df = df.drop(columns=["future_close", "next_close"])
    return df


def cross_sectional_normalize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    每日截面 z-score 归一化（对数值特征）。
    对每一天，对每个特征，计算该日所有股票的 mean/std，做 z-score。
    极端值用 median±5*mad 截断（稳健处理）。
    """
    df = df.copy()
    for col in feature_cols:
        # 先稳健截断
        median = df.groupby("trade_date")[col].transform("median")
        mad = df.groupby("trade_date")[col].transform(lambda x: (x - x.median()).abs().median())
        lower = median - 5 * mad
        upper = median + 5 * mad
        df[col] = df[col].clip(lower=lower, upper=upper)

        # z-score
        mean = df.groupby("trade_date")[col].transform("mean")
        std = df.groupby("trade_date")[col].transform("std")
        df[col] = (df[col] - mean) / std.replace(0, np.nan)
        df[col] = df[col].fillna(0.0)
    return df


def build_dataset(
    start_date: str = "20180101",
    end_date: str = "20241231",
    time_window: int = 20,
    forward_days: int = 10,
    train_end: str = "2021-12-31",
    valid_end: str = "2022-12-31",
    cache: bool = True,
) -> None:
    """
    主入口：构造完整的 train/valid/test 数据集。
    每个 sample 是一条 (ts_code, trade_date) 记录，包含：
      - feature_cols：归一化后的特征列
      - log_mv, industry, index_weight：约束特征 M
      - label, label_rank：标签 y
      - time_window：过去 20 天的特征面板（在 Dataset 中再展开）
    """
    cache_path = PROCESSED_DIR / f"dataset_{start_date}_{end_date}_T{time_window}_F{forward_days}.parquet"
    if cache and cache_path.exists():
        print(f"✅ 数据集已存在：{cache_path}")
        return

    print("🔄 加载原始数据...")
    index_weight, daily, daily_basic, company, adj_factor = load_raw(start_date, end_date)

    print("🔄 构建基础大表...")
    df = build_base_table(daily, daily_basic, adj_factor, company, index_weight)

    print("🔄 计算特征...")
    df = compute_features(df)

    print("🔄 计算标签...")
    df = compute_labels(df, forward_days=forward_days)

    # 定义特征列
    feature_cols = [
        "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount",
        "turnover_rate",
        "returns_1d", "volatility_5d",
        "sma_5", "sma_10", "sma_20",
        "ema_12", "ema_26", "macd", "macd_signal",
        "rsi_14",
        "close_sma5_ratio", "close_sma20_ratio",
        "volume_change",
    ]

    # 约束特征 M
    constraint_cols = ["log_mv", "industry", "index_weight"]

    # 剔除缺失值过多的行（至少要有 time_window 历史数据才能构造面板）
    # 这里先不剔除，在 Dataset 里用 mask 处理停牌

    print("🔄 截面归一化...")
    df = cross_sectional_normalize(df, feature_cols)

    # 行业编码为 one-hot（先保留字符串，在 Dataset 里处理）
    df["industry"] = df["industry"].fillna("未知")

    # 时间切分
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df_train = df[df["trade_date"] <= pd.Timestamp(train_end)].copy()
    df_valid = df[(df["trade_date"] > pd.Timestamp(train_end)) & (df["trade_date"] <= pd.Timestamp(valid_end))].copy()
    df_test = df[df["trade_date"] > pd.Timestamp(valid_end)].copy()

    # 保存
    df_train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    df_valid.to_parquet(PROCESSED_DIR / "valid.parquet", index=False)
    df_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    # 获取完整行业列表（从全量数据中，确保 train/valid/test 编码一致）
    full_industry_list = sorted(df["industry"].dropna().unique().tolist())

    # 保存元数据
    meta = {
        "feature_cols": feature_cols,
        "constraint_cols": constraint_cols,
        "time_window": time_window,
        "industry_list": full_industry_list,
    }
    import json
    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据集构造完成：")
    print(f"   训练集：{len(df_train)} 条，{df_train['trade_date'].min().date()} ~ {df_train['trade_date'].max().date()}")
    print(f"   验证集：{len(df_valid)} 条，{df_valid['trade_date'].min().date()} ~ {df_valid['trade_date'].max().date()}")
    print(f"   测试集：{len(df_test)} 条，{df_test['trade_date'].min().date()} ~ {df_test['trade_date'].max().date()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--forward", type=int, default=10)
    args = parser.parse_args()
    build_dataset(
        start_date=args.start,
        end_date=args.end,
        time_window=args.window,
        forward_days=args.forward,
    )
