"""ETF universe construction and auditable risk-category mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import ensure_columns


CLASSIFICATION_COLUMNS = [
    "ts_code",
    "category",
    "risk_bucket",
    "effective_start",
    "effective_end",
    "classification_basis",
    "reviewed",
]
RISK_CATEGORIES = {"industry", "theme", "strategy", "style"}
ALL_CATEGORIES = {"scale", *RISK_CATEGORIES}


def stock_exchange_fund_universe(fund_basic: pd.DataFrame, *, as_of_date: str | None = None) -> pd.DataFrame:
    """Return exchange-listed equity funds eligible for manual ETF review.

    Tushare's fund master does not expose the report's five ETF categories.
    This function therefore only creates an eligible universe; it does not
    claim that all returned funds are correctly classified ETFs.
    """

    ensure_columns(fund_basic, ["ts_code", "fund_type", "type", "list_date", "delist_date"])
    out = fund_basic.copy()
    for col in ("list_date", "delist_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    stock_like = out["fund_type"].eq("股票型") | out["type"].eq("股票型")
    out = out[stock_like].copy()
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        out = out[(out["list_date"].isna() | (out["list_date"] <= as_of)) & (out["delist_date"].isna() | (out["delist_date"] > as_of))]
    out["classification_hint"] = out.apply(_classification_hint, axis=1)
    out["eligible_for_review"] = True
    return out.sort_values("ts_code").reset_index(drop=True)


def classification_review_template(
    fund_basic: pd.DataFrame,
    *,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """Create a conservative manual-review queue from ``fund_basic``.

    Tushare does not expose the report's risk taxonomy.  The returned rows
    intentionally leave ``category``/``risk_bucket`` unset and
    ``reviewed=False``.  ``classification_hint`` and the source master fields
    help a researcher review the queue without turning name matching into a
    production signal.
    """

    universe = stock_exchange_fund_universe(fund_basic, as_of_date=as_of_date)
    if universe.empty:
        return pd.DataFrame(
            columns=CLASSIFICATION_COLUMNS
            + ["name", "benchmark", "classification_hint", "list_date", "delist_date"]
        )
    name = universe["name"].astype("string") if "name" in universe.columns else pd.Series("", index=universe.index, dtype="string")
    benchmark = universe["benchmark"].astype("string") if "benchmark" in universe.columns else pd.Series("", index=universe.index, dtype="string")
    out = pd.DataFrame(
        {
            "ts_code": universe["ts_code"].astype(str),
            "category": pd.Series(pd.NA, index=universe.index, dtype="string"),
            "risk_bucket": pd.Series(pd.NA, index=universe.index, dtype="string"),
            "effective_start": universe["list_date"],
            "effective_end": universe["delist_date"],
            "classification_basis": "manual_review_required",
            "reviewed": False,
            "name": name,
            "benchmark": benchmark,
            "classification_hint": universe["classification_hint"],
            "list_date": universe["list_date"],
            "delist_date": universe["delist_date"],
        }
    )
    return out.sort_values("ts_code").reset_index(drop=True)


def validate_classification(frame: pd.DataFrame) -> pd.DataFrame:
    """Return invalid reviewed rows with a reason, without mutating labels."""

    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "reason"])
    ensure_columns(frame, CLASSIFICATION_COLUMNS)
    rows: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        reviewed = bool(row.reviewed) if pd.notna(row.reviewed) else False
        if reviewed and row.category not in ALL_CATEGORIES:
            rows.append({"ts_code": str(row.ts_code), "reason": "invalid_category"})
        if reviewed and row.risk_bucket not in {"low_risk", "risk"}:
            rows.append({"ts_code": str(row.ts_code), "reason": "invalid_risk_bucket"})
        if reviewed and row.category in ALL_CATEGORIES:
            expected_bucket = "low_risk" if row.category == "scale" else "risk"
            if row.risk_bucket != expected_bucket:
                rows.append({"ts_code": str(row.ts_code), "reason": "category_bucket_mismatch"})
        if pd.notna(row.effective_start) and pd.notna(row.effective_end) and row.effective_end < row.effective_start:
            rows.append({"ts_code": str(row.ts_code), "reason": "effective_end_before_start"})
    return pd.DataFrame(rows, columns=["ts_code", "reason"])


def _classification_hint(row: pd.Series) -> str:
    """Offer a non-binding hint for manual review, never a production label."""

    text = " ".join(str(row.get(col, "")) for col in ("name", "benchmark"))
    if any(word in text for word in ("行业", "证券", "银行", "医药", "消费", "能源", "半导体", "科技")):
        return "industry_or_theme_review"
    if any(word in text for word in ("红利", "价值", "成长", "质量", "低波", "量化")):
        return "strategy_or_style_review"
    if any(word in text for word in ("沪深300", "中证500", "中证1000", "上证50", "中证A500", "宽基")):
        return "scale_review"
    return "unclassified_review"


def load_classification(path: str | Path) -> pd.DataFrame:
    """Load the time-aware manual classification file."""

    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    frame = pd.read_csv(source, dtype={"ts_code": str})
    ensure_columns(frame, CLASSIFICATION_COLUMNS)
    out = frame.copy()
    for column in ("effective_start", "effective_end"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
    out["reviewed"] = out["reviewed"].astype("boolean")
    invalid = validate_classification(out)
    if not invalid.empty:
        details = ", ".join(f"{row.ts_code}:{row.reason}" for row in invalid.itertuples(index=False))
        raise ValueError(f"invalid ETF classification rows: {details}")
    # A code may change category over time, but duplicate effective intervals
    # are ambiguous and would multiply daily rows in the merge below.
    duplicate_keys = ["ts_code", "effective_start", "effective_end"]
    if out.duplicated(duplicate_keys).any():
        duplicates = out[out.duplicated(duplicate_keys, keep=False)]
        codes = ", ".join(duplicates["ts_code"].astype(str).unique())
        raise ValueError(f"duplicate ETF classification intervals: {codes}")
    return out


def apply_classification(
    fund_daily: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    include_unreviewed: bool = False,
) -> pd.DataFrame:
    """Attach the classification valid on each daily record."""

    ensure_columns(fund_daily, ["ts_code", "trade_date"])
    if classification.empty:
        result = fund_daily.copy()
        result["category"] = pd.NA
        result["risk_bucket"] = "unclassified"
        result["reviewed"] = False
        return result.iloc[0:0].copy() if not include_unreviewed else result
    data = fund_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    cls = classification.copy()
    for column in ("effective_start", "effective_end"):
        cls[column] = pd.to_datetime(cls[column], errors="coerce")
    merged = data.merge(cls, on="ts_code", how="left", suffixes=("", "_classification"))
    valid = (
        merged["effective_start"].isna() | (merged["trade_date"] >= merged["effective_start"])
    ) & (
        merged["effective_end"].isna() | (merged["trade_date"] <= merged["effective_end"])
    )
    merged = merged[valid].copy()
    merged["risk_bucket"] = merged["risk_bucket"].fillna("unclassified")
    merged["reviewed"] = merged["reviewed"].fillna(False).astype(bool)
    if not include_unreviewed:
        merged = merged[merged["reviewed"] & merged["risk_bucket"].isin(["low_risk", "risk"])]
    return merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
