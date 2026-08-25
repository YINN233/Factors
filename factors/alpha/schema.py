"""
Field metadata for constrained alpha research.

The report summaries emphasize that alpha formulas should be constrained by
frequency, dimension, semantic role, and time availability.  This module keeps
that metadata close to the executable factor code.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable


class Frequency(str, Enum):
    DAILY = "daily"
    MINUTE = "minute"
    QUARTERLY = "quarterly"


class Dimension(str, Enum):
    PRICE = "price"
    VOLUME = "volume"
    MONEY = "money"
    RATIO = "ratio"
    RETURN = "return"
    SCORE = "score"
    CATEGORY = "category"


class Semantics(str, Enum):
    PRICE = "price"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    INDUSTRY = "industry"
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dimension: Dimension
    frequency: Frequency
    semantics: Semantics
    description: str = ""
    pit_required: bool = False


DEFAULT_DAILY_FIELDS = {
    "open_adj": FieldSpec("open_adj", Dimension.PRICE, Frequency.DAILY, Semantics.PRICE),
    "high_adj": FieldSpec("high_adj", Dimension.PRICE, Frequency.DAILY, Semantics.PRICE),
    "low_adj": FieldSpec("low_adj", Dimension.PRICE, Frequency.DAILY, Semantics.PRICE),
    "close_adj": FieldSpec("close_adj", Dimension.PRICE, Frequency.DAILY, Semantics.PRICE),
    "volume": FieldSpec("volume", Dimension.VOLUME, Frequency.DAILY, Semantics.LIQUIDITY),
    "amount": FieldSpec("amount", Dimension.MONEY, Frequency.DAILY, Semantics.LIQUIDITY),
    "turnover_rate": FieldSpec("turnover_rate", Dimension.RATIO, Frequency.DAILY, Semantics.LIQUIDITY),
    "total_mv": FieldSpec("total_mv", Dimension.MONEY, Frequency.DAILY, Semantics.VALUE),
    "log_mv": FieldSpec("log_mv", Dimension.SCORE, Frequency.DAILY, Semantics.VALUE),
    "industry": FieldSpec("industry", Dimension.CATEGORY, Frequency.DAILY, Semantics.INDUSTRY),
    "index_weight": FieldSpec("index_weight", Dimension.RATIO, Frequency.DAILY, Semantics.BENCHMARK),
}


OPTIONAL_FUNDAMENTAL_FIELDS = {
    "operating_cf_margin_ttm": FieldSpec(
        "operating_cf_margin_ttm",
        Dimension.RATIO,
        Frequency.QUARTERLY,
        Semantics.QUALITY,
        pit_required=True,
    ),
    "cashflow_to_profit": FieldSpec(
        "cashflow_to_profit",
        Dimension.RATIO,
        Frequency.QUARTERLY,
        Semantics.QUALITY,
        pit_required=True,
    ),
    "rd_expense_intensity": FieldSpec(
        "rd_expense_intensity",
        Dimension.RATIO,
        Frequency.QUARTERLY,
        Semantics.GROWTH,
        pit_required=True,
    ),
    "capex_to_assets": FieldSpec(
        "capex_to_assets",
        Dimension.RATIO,
        Frequency.QUARTERLY,
        Semantics.GROWTH,
        pit_required=True,
    ),
}


def build_field_registry(extra_fields: Iterable[FieldSpec] = ()) -> Dict[str, FieldSpec]:
    registry = dict(DEFAULT_DAILY_FIELDS)
    registry.update(OPTIONAL_FUNDAMENTAL_FIELDS)
    for field in extra_fields:
        registry[field.name] = field
    return registry
