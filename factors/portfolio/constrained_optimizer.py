"""Constrained long-only index-enhancement optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizerConfig:
    benchmark_col: str = "csi500_index_weight"
    method: str = "fast"
    max_stock_weight: float = 0.02
    max_stock_active: float = 0.008
    max_active_share: float = 0.20
    max_industry_active: float = 0.015
    max_style_active: float = 0.20
    turnover_limit: float | None = None
    lambda_active: float = 1.0
    lambda_turnover: float = 0.10


def _normalize_benchmark(sub: pd.DataFrame, benchmark_col: str) -> pd.Series:
    bench = sub[benchmark_col].astype(float).clip(lower=0).fillna(0.0)
    if bench.sum() <= 0:
        bench = pd.Series(1.0 / len(sub), index=sub.index)
    else:
        bench = bench / bench.sum()
    return bench


def _fallback_weights(sub: pd.DataFrame, config: OptimizerConfig) -> pd.Series:
    bench = _normalize_benchmark(sub, config.benchmark_col)
    weights = bench.clip(upper=config.max_stock_weight)
    if weights.sum() <= 0:
        return pd.Series(1.0 / len(sub), index=sub["ts_code"].to_numpy())
    weights = weights / weights.sum()
    return pd.Series(weights.to_numpy(), index=sub["ts_code"].to_numpy())


def _active_bounds(bench: np.ndarray, config: OptimizerConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(-config.max_stock_active, -bench)
    upper = np.minimum(config.max_stock_active, config.max_stock_weight - bench)
    upper = np.maximum(upper, lower)
    return lower, upper


def _project_active(active: np.ndarray, lower: np.ndarray, upper: np.ndarray, max_iter: int = 30) -> np.ndarray:
    out = np.clip(np.asarray(active, dtype=float), lower, upper)
    for _ in range(max_iter):
        diff = -float(out.sum())
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = upper - out
        else:
            room = out - lower
        total_room = float(room[room > 0].sum())
        if total_room <= 1e-12:
            break
        step = min(abs(diff), total_room) * np.sign(diff)
        out = out + step * np.where(room > 0, room / total_room, 0.0)
        out = np.clip(out, lower, upper)
    return out


def _standardize(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(out), out, np.nan)
    med = np.nanmedian(out)
    out = np.where(np.isfinite(out), out, med if np.isfinite(med) else 0.0)
    std = np.nanstd(out)
    if std > 1e-12:
        out = (out - np.nanmean(out)) / std
    else:
        out = out - np.nanmean(out)
    return out


def _residualize_alpha(sub: pd.DataFrame, alpha: np.ndarray, style_cols: Sequence[str]) -> np.ndarray:
    cols = [np.ones(len(sub))]
    if "industry" in sub.columns:
        dummies = pd.get_dummies(sub["industry"].fillna("unknown"), dtype=float)
        if dummies.shape[1] > 1:
            cols.extend(dummies.iloc[:, 1:].to_numpy().T)
    for style in style_cols:
        if style in sub.columns:
            cols.append(_standardize(sub[style].to_numpy()))
    x = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(x, alpha, rcond=None)
        alpha = alpha - x @ beta
    except np.linalg.LinAlgError:
        alpha = alpha - np.nanmean(alpha)
    return _standardize(alpha)


def _constraint_shrink(
    active: np.ndarray,
    sub: pd.DataFrame,
    bench: np.ndarray,
    style_cols: Sequence[str],
    config: OptimizerConfig,
) -> np.ndarray:
    shrink = 1.0
    active_share = 0.5 * float(np.abs(active).sum())
    if active_share > config.max_active_share > 0:
        shrink = min(shrink, config.max_active_share / active_share)

    if "industry" in sub.columns and config.max_industry_active is not None and config.max_industry_active > 0:
        tmp = pd.DataFrame({"industry": sub["industry"].to_numpy(), "active": active})
        max_ind = float(tmp.groupby("industry", sort=False)["active"].sum().abs().max())
        if max_ind > config.max_industry_active:
            shrink = min(shrink, config.max_industry_active / max_ind)

    for style in style_cols:
        if style not in sub.columns or config.max_style_active is None or config.max_style_active <= 0:
            continue
        values = sub[style].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        exposure = abs(float(active @ values))
        if exposure > config.max_style_active:
            shrink = min(shrink, config.max_style_active / exposure)

    if shrink < 1.0:
        active = active * max(shrink * 0.995, 0.0)
    return active


def _optimize_fast(
    sub: pd.DataFrame,
    alpha_col: str,
    style_cols: Sequence[str],
    config: OptimizerConfig,
) -> pd.Series:
    bench = _normalize_benchmark(sub, config.benchmark_col).to_numpy()
    alpha = sub[alpha_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    alpha = _standardize(alpha)
    if style_cols or "industry" in sub.columns:
        alpha = _residualize_alpha(sub, alpha, style_cols)

    lower, upper = _active_bounds(bench, config)
    if np.nanstd(alpha) <= 1e-12:
        active = np.zeros_like(bench)
    else:
        active = alpha / (np.abs(alpha).sum() + 1e-12) * (2.0 * config.max_active_share)
    active = _project_active(active, lower, upper)
    active = _constraint_shrink(active, sub, bench, style_cols, config)
    active = _project_active(active, lower, upper)
    active = _constraint_shrink(active, sub, bench, style_cols, config)
    weights = bench + active
    weights = np.clip(weights, 0.0, config.max_stock_weight)
    if weights.sum() <= 0:
        return _fallback_weights(sub, config)
    weights = weights / weights.sum()
    return pd.Series(weights, index=sub["ts_code"].to_numpy())


def _optimize_cvxpy(
    sub: pd.DataFrame,
    alpha_col: str,
    style_cols: Sequence[str] = (),
    prev_weights: pd.Series | None = None,
    config: OptimizerConfig = OptimizerConfig(),
) -> pd.Series:
    sub = sub.dropna(subset=["ts_code"]).copy().reset_index(drop=True)
    if sub.empty or alpha_col not in sub.columns:
        return pd.Series(dtype=float)

    n = len(sub)
    bench = _normalize_benchmark(sub, config.benchmark_col).to_numpy()
    alpha = sub[alpha_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    alpha = alpha - np.nanmean(alpha)
    std = np.nanstd(alpha)
    if std > 0:
        alpha = alpha / std

    w = cp.Variable(n)
    objective = alpha @ w - config.lambda_active * cp.sum_squares(w - bench)

    if prev_weights is not None and not prev_weights.empty:
        prev = prev_weights.reindex(sub["ts_code"], fill_value=0.0).to_numpy()
        objective -= config.lambda_turnover * cp.sum_squares(w - prev)
    else:
        prev = None

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= np.minimum(config.max_stock_weight, bench + config.max_stock_active),
        cp.abs(w - bench) <= config.max_stock_active,
        0.5 * cp.norm1(w - bench) <= config.max_active_share,
    ]

    if prev is not None and config.turnover_limit is not None:
        constraints.append(0.5 * cp.norm1(w - prev) <= config.turnover_limit)

    if "industry" in sub.columns and config.max_industry_active is not None:
        for _, idx in sub.groupby("industry").groups.items():
            idx = np.array(list(idx), dtype=int)
            constraints.append(cp.abs(cp.sum(w[idx]) - float(bench[idx].sum())) <= config.max_industry_active)

    for style in style_cols:
        if style not in sub.columns:
            continue
        values = sub[style].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        bench_exp = float(bench @ values)
        constraints.append(cp.abs(w @ values - bench_exp) <= config.max_style_active)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            return _fallback_weights(sub, config)

    if w.value is None:
        return _fallback_weights(sub, config)
    values = np.asarray(w.value, dtype=float)
    values = np.clip(values, 0.0, None)
    if values.sum() <= 0:
        return _fallback_weights(sub, config)
    values = values / values.sum()
    return pd.Series(values, index=sub["ts_code"].to_numpy())


def optimize_constrained_weights(
    sub: pd.DataFrame,
    alpha_col: str,
    style_cols: Sequence[str] = (),
    prev_weights: pd.Series | None = None,
    config: OptimizerConfig = OptimizerConfig(),
) -> pd.Series:
    sub = sub.dropna(subset=["ts_code"]).copy().reset_index(drop=True)
    if sub.empty or alpha_col not in sub.columns:
        return pd.Series(dtype=float)
    if config.method == "cvxpy":
        return _optimize_cvxpy(sub, alpha_col, style_cols=style_cols, prev_weights=prev_weights, config=config)
    return _optimize_fast(sub, alpha_col, style_cols=style_cols, config=config)


def weight_diagnostics(
    weights: pd.Series,
    sub: pd.DataFrame,
    style_cols: Sequence[str] = (),
    config: OptimizerConfig = OptimizerConfig(),
) -> dict:
    codes = sub["ts_code"].to_numpy()
    bench = _normalize_benchmark(sub, config.benchmark_col)
    bench = pd.Series(bench.to_numpy(), index=codes)
    idx = weights.index.union(bench.index)
    active = weights.reindex(idx, fill_value=0.0) - bench.reindex(idx, fill_value=0.0)
    out = {
        "active_share": float(0.5 * active.abs().sum()),
        "max_stock_weight": float(weights.max()) if not weights.empty else 0.0,
        "max_abs_stock_active": float(active.abs().max()) if not active.empty else 0.0,
    }
    if "industry" in sub.columns:
        industry = pd.Series(sub["industry"].to_numpy(), index=codes)
        bench_ind = bench.groupby(industry).sum()
        port_ind = weights.reindex(codes, fill_value=0.0).groupby(industry).sum()
        ind_active = port_ind.sub(bench_ind, fill_value=0.0)
        max_industry_active = float(ind_active.abs().max()) if not ind_active.empty else 0.0
        out["max_abs_industry_active"] = max_industry_active
        out["max_industry_active"] = max_industry_active
    for style in style_cols:
        if style in sub.columns:
            values = pd.Series(
                sub[style].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(),
                index=codes,
            )
            out[f"{style}_active"] = float(
                weights.reindex(values.index, fill_value=0.0).dot(values)
                - bench.reindex(values.index, fill_value=0.0).dot(values)
            )
    return out
