"""Directional, point-in-time basis comparisons for index-futures pairs."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .contracts import FUTURE_PRODUCTS, ensure_columns
from .dividend_basis import add_historical_percentile


QUALITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def _worse_quality(left: object, right: object) -> str:
    values = [str(left).upper(), str(right).upper()]
    return max(values, key=lambda value: QUALITY_ORDER.get(value, QUALITY_ORDER["D"])) if values else "D"


def classify_pair_structure(percentile: object, quality: object) -> str:
    """Classify relative carry without implying a relative-return forecast."""

    value = pd.to_numeric(pd.Series([percentile]), errors="coerce").iloc[0]
    if str(quality).upper() == "D" or pd.isna(value):
        return "insufficient"
    if float(value) <= 20.0:
        return "favorable"
    if float(value) >= 80.0:
        return "unfavorable"
    return "neutral"


def match_pair_legs(
    basis: pd.DataFrame,
    *,
    max_expiry_gap_days: int = 7,
) -> pd.DataFrame:
    """Match all ordered product pairs by date and tenor rank."""

    required = {
        "trade_date",
        "product",
        "ts_code",
        "expiry_date",
        "tenor_rank",
        "raw_annualized_basis",
    }
    if basis.empty:
        return pd.DataFrame()
    ensure_columns(basis, required)
    data = basis.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["expiry_date"] = pd.to_datetime(data["expiry_date"], errors="coerce")
    data["tenor_rank"] = pd.to_numeric(data["tenor_rank"], errors="coerce")
    data["raw_annualized_basis"] = pd.to_numeric(data["raw_annualized_basis"], errors="coerce")
    data = data[data["product"].isin(FUTURE_PRODUCTS)].dropna(subset=["trade_date", "product", "tenor_rank"])
    keys = ["trade_date", "product", "tenor_rank"]
    if data.duplicated(keys).any():
        raise ValueError("basis contains duplicate product/date/tenor rows")
    if "raw_basis_quality" not in data.columns:
        data["raw_basis_quality"] = "D"
    products = [product for product in FUTURE_PRODUCTS if product in set(data["product"])]
    frames: list[pd.DataFrame] = []
    for long_product, short_product in permutations(products, 2):
        long_leg = data[data["product"].eq(long_product)][
            ["trade_date", "tenor_rank", "ts_code", "expiry_date", "raw_annualized_basis", "raw_basis_quality"]
        ].rename(
            columns={
                "ts_code": "long_contract",
                "expiry_date": "long_expiry_date",
                "raw_annualized_basis": "long_raw_annualized_basis",
                "raw_basis_quality": "long_raw_basis_quality",
            }
        )
        short_leg = data[data["product"].eq(short_product)][
            ["trade_date", "tenor_rank", "ts_code", "expiry_date", "raw_annualized_basis", "raw_basis_quality"]
        ].rename(
            columns={
                "ts_code": "short_contract",
                "expiry_date": "short_expiry_date",
                "raw_annualized_basis": "short_raw_annualized_basis",
                "raw_basis_quality": "short_raw_basis_quality",
            }
        )
        matched = long_leg.merge(short_leg, on=["trade_date", "tenor_rank"], how="inner", validate="one_to_one")
        if matched.empty:
            continue
        matched.insert(1, "long_product", long_product)
        matched.insert(2, "short_product", short_product)
        matched["expiry_gap_days"] = (
            matched["long_expiry_date"] - matched["short_expiry_date"]
        ).dt.days.abs()
        matched["pair_quality"] = [
            _worse_quality(left, right)
            for left, right in zip(matched["long_raw_basis_quality"], matched["short_raw_basis_quality"])
        ]
        comparable = (
            matched["expiry_gap_days"].le(max(0, int(max_expiry_gap_days)))
            & matched["long_raw_annualized_basis"].notna()
            & matched["short_raw_annualized_basis"].notna()
            & matched["pair_quality"].ne("D")
        )
        matched["pair_basis_spread"] = np.where(
            comparable,
            matched["long_raw_annualized_basis"] - matched["short_raw_annualized_basis"],
            np.nan,
        )
        frames.append(matched)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_pair_point_in_time_percentile(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add an expanding percentile within each directional pair and tenor."""

    if pairs.empty:
        return pairs.copy()
    out = add_historical_percentile(
        pairs,
        value_column="pair_basis_spread",
        group_columns=("long_product", "short_product", "tenor_rank"),
        point_in_time=True,
    )
    return out.rename(columns={"historical_percentile": "pair_historical_percentile"})


def build_pair_basis_history(
    basis: pd.DataFrame,
    *,
    max_expiry_gap_days: int = 7,
) -> pd.DataFrame:
    """Build all auditable ordered-pair comparisons from a basis table."""

    matched = match_pair_legs(basis, max_expiry_gap_days=max_expiry_gap_days)
    if matched.empty:
        return matched
    out = add_pair_point_in_time_percentile(matched)
    out["pair_structure_status"] = [
        classify_pair_structure(percentile, quality)
        for percentile, quality in zip(out["pair_historical_percentile"], out["pair_quality"])
    ]
    out["point_in_time"] = True
    keys = ["trade_date", "long_product", "short_product", "tenor_rank"]
    if out.duplicated(keys).any():
        raise ValueError("pair basis output contains duplicate directional keys")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)

