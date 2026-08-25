"""
Reusable alpha operators on long-form panels.

Expected panel keys are ``trade_date`` and ``ts_code`` unless overridden.  The
functions preserve the input index so they can be assigned back to the source
DataFrame safely.
"""

from typing import Optional

import numpy as np
import pandas as pd


def safe_div(x: pd.Series, y: pd.Series, eps: float = 1e-12) -> pd.Series:
    out = x.astype(float) / y.astype(float).where(y.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def spread(x: pd.Series, y: pd.Series) -> pd.Series:
    return safe_div(x - y, x.abs() + y.abs())


def geom_mean(x: pd.Series, y: pd.Series) -> pd.Series:
    valid = (x >= 0) & (y >= 0)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[valid] = np.sqrt(x.loc[valid] * y.loc[valid])
    return out


def harm_mean(x: pd.Series, y: pd.Series) -> pd.Series:
    return safe_div(2 * x * y, x + y)


def signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x.astype(float)) * np.log1p(x.astype(float).abs())


def signed_power(x: pd.Series, power: float) -> pd.Series:
    return np.sign(x.astype(float)) * np.power(x.astype(float).abs(), power)


def cs_rank(
    df: pd.DataFrame,
    col: str,
    date_col: str = "trade_date",
) -> pd.Series:
    return df.groupby(date_col, sort=False)[col].rank(pct=True)


def cs_zscore(
    df: pd.DataFrame,
    col: str,
    date_col: str = "trade_date",
) -> pd.Series:
    values = df[col].astype(float)
    grouped = values.groupby(df[date_col], sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    out = safe_div(values - mean, std)
    out = out.mask(std.isna() | (std == 0), 0.0)
    return out.mask(values.isna())


def cs_robust_zscore(
    df: pd.DataFrame,
    col: str,
    date_col: str = "trade_date",
) -> pd.Series:
    values = df[col].astype(float)
    grouped = values.groupby(df[date_col], sort=False)
    median = grouped.transform("median")
    mad = (values - median).abs().groupby(df[date_col], sort=False).transform("median")
    out = safe_div(values - median, 1.4826 * mad)
    out = out.mask(mad.isna() | (mad == 0), 0.0)
    return out.mask(values.isna())


def cs_scale(
    df: pd.DataFrame,
    col: str,
    date_col: str = "trade_date",
) -> pd.Series:
    values = df[col].astype(float)
    denom = values.abs().groupby(df[date_col], sort=False).transform("sum")
    out = safe_div(values, denom)
    out = out.mask(denom.isna() | (denom == 0), 0.0)
    return out.mask(values.isna())


def ts_delay(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
    code_col: str = "ts_code",
) -> pd.Series:
    return df.groupby(code_col, sort=False)[col].shift(periods)


def ts_mean(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )


def ts_sum(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).sum()
    )


def ts_std(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )


def ts_var(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).var()
    )


def ts_median(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).median()
    )


def ts_quantile(
    df: pd.DataFrame,
    col: str,
    window: int,
    q: float,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).quantile(q)
    )


def ts_wma(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    weights = np.arange(1, window + 1, dtype=float)

    def _wma(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        w = weights[-len(valid):]
        return float(np.dot(valid, w) / w.sum())

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_wma, raw=True)
    )


def ts_zscore(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    mean = ts_mean(df, col, window, min_periods=min_periods, code_col=code_col)
    std = ts_std(df, col, window, min_periods=min_periods, code_col=code_col)
    return safe_div(df[col] - mean, std)


def ts_coef_var(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    mean = ts_mean(df, col, window, min_periods=min_periods, code_col=code_col)
    std = ts_std(df, col, window, min_periods=min_periods, code_col=code_col)
    return safe_div(std, mean.abs())


def ts_skew(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).skew()
    )


def ts_kurt(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).kurt()
    )


def ts_decay_linear(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _decay(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        weights = np.arange(1, len(valid) + 1, dtype=float)
        return float(np.dot(valid, weights) / weights.sum())

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_decay, raw=True)
    )


def ts_argmax(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _argmax_pos(s: pd.Series) -> float:
        values = s.to_numpy(dtype=float)
        if np.isnan(values).all():
            return np.nan
        return (float(np.nanargmax(values)) + 1.0) / len(values)

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_argmax_pos, raw=False)
    )


def ts_argmin(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _argmin_pos(s: pd.Series) -> float:
        values = s.to_numpy(dtype=float)
        if np.isnan(values).all():
            return np.nan
        return (float(np.nanargmin(values)) + 1.0) / len(values)

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_argmin_pos, raw=False)
    )


def ts_min(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).min()
    )


def ts_max(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).max()
    )


def ts_max_drawdown(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _max_drawdown(values: np.ndarray) -> float:
        values = values.astype(float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        peak = np.maximum.accumulate(values)
        denom = np.where(np.abs(peak) > 1e-12, np.abs(peak), np.nan)
        drawdown = (values - peak) / denom
        if np.isnan(drawdown).all():
            return 0.0
        return float(np.nanmin(drawdown))

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_max_drawdown, raw=True)
    )


def ts_pct(
    df: pd.DataFrame,
    col: str,
    periods: int,
    code_col: str = "ts_code",
) -> pd.Series:
    lag = ts_delay(df, col, periods=periods, code_col=code_col)
    return safe_div(df[col], lag) - 1.0


def ts_delta(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
    code_col: str = "ts_code",
) -> pd.Series:
    return df[col] - ts_delay(df, col, periods=periods, code_col=code_col)


def ts_rank(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _last_rank(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        last = values[-1]
        if np.isnan(last):
            return np.nan
        valid = values[~np.isnan(values)]
        return float((valid <= last).sum() / len(valid))

    return df.groupby(code_col, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(_last_rank, raw=True)
    )


def ts_percentage(
    df: pd.DataFrame,
    col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    return ts_rank(df, col, window, min_periods=min_periods, code_col=code_col)


def ts_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    pieces = []
    for _, sub in df.groupby(code_col, sort=False):
        corr = sub[x_col].rolling(window, min_periods=min_periods).corr(sub[y_col])
        pieces.append(corr)
    if not pieces:
        return pd.Series(dtype=float, index=df.index)
    return pd.concat(pieces).reindex(df.index)


def ts_covariance(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    pieces = []
    for _, sub in df.groupby(code_col, sort=False):
        cov = sub[x_col].rolling(window, min_periods=min_periods).cov(sub[y_col])
        pieces.append(cov)
    if not pieces:
        return pd.Series(dtype=float, index=df.index)
    return pd.concat(pieces).reindex(df.index)


def ts_slope(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window
    pieces = []
    for _, sub in df.groupby(code_col, sort=False):
        x = sub[x_col].astype(float)
        y = sub[y_col].astype(float)
        cov = x.rolling(window, min_periods=min_periods).cov(y)
        var = x.rolling(window, min_periods=min_periods).var()
        pieces.append(safe_div(cov, var))
    if not pieces:
        return pd.Series(dtype=float, index=df.index)
    return pd.concat(pieces).reindex(df.index)


def ts_rsquare(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    corr = ts_corr(df, y_col, x_col, window, min_periods=min_periods, code_col=code_col)
    return corr.pow(2)


def cs_regression_resid(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    date_col: str = "trade_date",
    add_intercept: bool = True,
) -> pd.Series:
    def _resid(sub: pd.DataFrame) -> pd.Series:
        cols = [y_col] + x_cols
        valid = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
        out = pd.Series(np.nan, index=sub.index, dtype=float)
        if len(valid) <= len(x_cols) + int(add_intercept):
            return out
        y = valid[y_col].astype(float).to_numpy()
        x = valid[x_cols].astype(float).to_numpy()
        if add_intercept:
            x = np.column_stack([np.ones(len(valid)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ beta
        except np.linalg.LinAlgError:
            return out
        out.loc[valid.index] = y - pred
        return out

    return df.groupby(date_col, sort=False, group_keys=False).apply(_resid).reindex(df.index)


def ts_poly_resid(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int,
    degree: int = 2,
    min_periods: Optional[int] = None,
    code_col: str = "ts_code",
) -> pd.Series:
    min_periods = min_periods or window

    def _last_resid(sub: pd.DataFrame) -> pd.Series:
        y = sub[y_col].astype(float)
        x = sub[x_col].astype(float)
        records = []
        for idx in range(len(sub)):
            start = max(0, idx - window + 1)
            yy = y.iloc[start:idx + 1]
            xx = x.iloc[start:idx + 1]
            valid = pd.concat([yy, xx], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < max(min_periods, degree + 2):
                records.append(np.nan)
                continue
            xv = valid.iloc[:, 1].to_numpy()
            yv = valid.iloc[:, 0].to_numpy()
            try:
                coef = np.polyfit(xv, yv, degree)
                pred = np.polyval(coef, x.iloc[idx])
            except (np.linalg.LinAlgError, ValueError):
                records.append(np.nan)
                continue
            records.append(float(y.iloc[idx] - pred))
        return pd.Series(records, index=sub.index, dtype=float)

    return df.groupby(code_col, sort=False, group_keys=False).apply(_last_resid).reindex(df.index)


def cs_winsorize(
    df: pd.DataFrame,
    col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    date_col: str = "trade_date",
) -> pd.Series:
    grouped = df.groupby(date_col, sort=False)[col]
    lower = grouped.transform(lambda s: s.quantile(lower_q))
    upper = grouped.transform(lambda s: s.quantile(upper_q))
    return df[col].clip(lower=lower, upper=upper)


def size_neutralize(
    df: pd.DataFrame,
    col: str,
    size_col: str = "log_mv",
    date_col: str = "trade_date",
) -> pd.Series:
    def _neutralize(sub: pd.DataFrame) -> pd.Series:
        valid = sub[[col, size_col]].replace([np.inf, -np.inf], np.nan).dropna()
        out = pd.Series(np.nan, index=sub.index, dtype=float)
        if len(valid) < 3 or valid[size_col].std() == 0:
            return out
        x = valid[size_col].astype(float)
        y = valid[col].astype(float)
        beta = y.cov(x) / x.var()
        alpha = y.mean() - beta * x.mean()
        out.loc[valid.index] = y - (alpha + beta * x)
        return out

    return df.groupby(date_col, sort=False, group_keys=False).apply(_neutralize).reindex(df.index)


def industry_neutralize(
    df: pd.DataFrame,
    col: str,
    date_col: str = "trade_date",
    industry_col: str = "industry",
) -> pd.Series:
    keys = [date_col, industry_col]
    industry_mean = df.groupby(keys, sort=False)[col].transform("mean")
    return df[col] - industry_mean


def group_rank(
    df: pd.DataFrame,
    col: str,
    group_col: str = "industry",
    date_col: str = "trade_date",
) -> pd.Series:
    return df.groupby([date_col, group_col], sort=False)[col].rank(pct=True)
