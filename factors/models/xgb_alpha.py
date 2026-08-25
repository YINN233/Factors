"""XGBoost-style cross-sectional alpha model with sklearn fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from factors.alpha.validation import daily_ic, summarize_ic


@dataclass(frozen=True)
class AlphaModelConfig:
    label_col: str = "fwd_5d_rank"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    ytd_start: str = "2026-01-01"
    random_state: int = 7


@dataclass
class AlphaModelResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    feature_importance: pd.DataFrame
    backend: str


def _split_masks(dates: pd.Series, config: AlphaModelConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = dates <= pd.Timestamp(config.train_end)
    valid = (dates >= pd.Timestamp(config.valid_start)) & (dates <= pd.Timestamp(config.valid_end))
    test = dates >= pd.Timestamp(config.test_start)
    return train, valid, test


def _make_model(config: AlphaModelConfig):
    try:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_lambda=5.0,
                reg_alpha=0.1,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=config.random_state,
                n_jobs=4,
            ),
            "xgboost",
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=config.random_state,
            ),
            "sklearn_hist_gradient_boosting",
        )


def _fill_features(train_x: pd.DataFrame, all_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    med = train_x.replace([np.inf, -np.inf], np.nan).median()
    return train_x.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0), all_x.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0), med


def _prediction_corr_importance(feature_frame: pd.DataFrame, pred: pd.Series, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    y = pred.astype(float).to_numpy()
    y = y - np.nanmean(y)
    y_std = np.nanstd(y)
    for feature in feature_cols:
        x = feature_frame[feature].astype(float).to_numpy()
        x = x - np.nanmean(x)
        x_std = np.nanstd(x)
        if x_std <= 1e-12 or y_std <= 1e-12:
            importance = 0.0
        else:
            importance = abs(float(np.nanmean(x * y) / (x_std * y_std)))
        rows.append({"feature": feature, "importance": importance, "importance_type": "prediction_corr"})
    return pd.DataFrame(rows)


def _feature_importance(
    model,
    feature_cols: Sequence[str],
    backend: str,
    feature_frame: pd.DataFrame | None = None,
    pred: pd.Series | None = None,
) -> pd.DataFrame:
    if backend == "xgboost":
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        rows = [
            {"feature": feature, "importance": float(scores.get(feature, 0.0)), "importance_type": "xgboost_gain"}
            for feature in feature_cols
        ]
    else:
        values = getattr(model, "feature_importances_", None)
        if values is None and feature_frame is not None and pred is not None:
            return _prediction_corr_importance(feature_frame, pred, feature_cols).sort_values("importance", ascending=False).reset_index(drop=True)
        if values is None:
            values = np.zeros(len(feature_cols))
        rows = [
            {"feature": feature, "importance": float(value), "importance_type": "model_native"}
            for feature, value in zip(feature_cols, values)
        ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def summarize_predictions(pred: pd.DataFrame, config: AlphaModelConfig) -> pd.DataFrame:
    ic_df = daily_ic(pred, "pred", config.label_col)
    rows = []
    for period, start, end in [
        ("train", None, config.train_end),
        ("valid", config.valid_start, config.valid_end),
        ("test", config.test_start, None),
        ("ytd_2026", config.ytd_start, None),
    ]:
        row = {"period": period}
        row.update(summarize_ic(ic_df, period, start=start, end=end))
        rows.append(row)
    return pd.DataFrame(rows)


def train_predict_alpha_model(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    config: AlphaModelConfig = AlphaModelConfig(),
) -> AlphaModelResult:
    if not feature_cols:
        raise ValueError("feature_cols is empty")
    required = {"trade_date", "ts_code", config.label_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"model panel missing required columns: {missing}")

    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.dropna(subset=[config.label_col]).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = work["trade_date"]
    train_mask, valid_mask, _ = _split_masks(dates, config)
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("train or validation split is empty")

    model, backend = _make_model(config)
    train_x_raw = work.loc[train_mask, list(feature_cols)]
    all_x_raw = work[list(feature_cols)]
    train_x, all_x, _ = _fill_features(train_x_raw, all_x_raw)
    train_y = work.loc[train_mask, config.label_col].astype(float)

    if backend == "xgboost":
        valid_x = all_x.loc[valid_mask]
        valid_y = work.loc[valid_mask, config.label_col].astype(float)
        try:
            model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=False)
        except TypeError:
            model.fit(train_x, train_y)
    else:
        model.fit(train_x, train_y)

    out = work[["trade_date", "ts_code", config.label_col]].copy()
    out["pred"] = model.predict(all_x)
    out["pred_rank"] = out.groupby("trade_date", sort=False)["pred"].rank(pct=True) - 0.5
    summary = summarize_predictions(out, config)
    importance = _feature_importance(model, feature_cols, backend, all_x, out["pred"])
    return AlphaModelResult(predictions=out, summary=summary, feature_importance=importance, backend=backend)
