"""
特征工程工具模块：支持扩展更多量价/技术指标。
builder.py 中已包含核心特征计算，这里提供可复用的单函数接口。
"""

import numpy as np
import pandas as pd


def compute_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """计算收益率。"""
    return series.pct_change(periods)


def compute_volatility(series: pd.Series, window: int = 5) -> pd.Series:
    """滚动波动率。"""
    return series.pct_change(1).rolling(window).std()


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 指标。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标，返回 macd, macd_signal, macd_hist。"""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    macd_hist = macd - macd_signal
    return pd.DataFrame({
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    })


def zscore(series: pd.Series) -> pd.Series:
    """序列 z-score 标准化。"""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def rank(series: pd.Series) -> pd.Series:
    """序列 rank 标准化到 [0, 1]。"""
    return series.rank(pct=True)
