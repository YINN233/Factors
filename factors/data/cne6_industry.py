"""Point-in-time SW2021 level-one industry data for CNE6 V2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "cne6_enhanced_v2"

SW_CLASSIFICATION_FILE = "cne6_sw2021_l1_classify.parquet"
SW_MEMBERS_FILE = "cne6_sw2021_l1_members.parquet"
CITIC_MEMBERS_FILE = "cne6_citic_l1_members_audit.parquet"
CITIC_L1_CODES = tuple(f"CI005{number:03d}.CI" for number in range(1, 31))


class IndustryPro(Protocol):
    def index_classify(self, **kwargs) -> pd.DataFrame: ...

    def index_member_all(self, **kwargs) -> pd.DataFrame: ...

    def ci_index_member(self, **kwargs) -> pd.DataFrame: ...


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def normalize_sw_classification(
    classification: pd.DataFrame,
    expected_count: int | None = 31,
) -> pd.DataFrame:
    """Validate the SW2021 L1 classification contract."""

    _require_columns(classification, {"index_code", "industry_name"}, "SW2021 classification")
    out = classification.copy()
    out["index_code"] = out["index_code"].astype("string").str.strip()
    out["industry_name"] = out["industry_name"].astype("string").str.strip()
    if "level" in out.columns:
        out = out[out["level"].astype("string").str.upper() == "L1"].copy()
    if "src" in out.columns:
        out = out[out["src"].astype("string").str.upper() == "SW2021"].copy()
    if out["index_code"].isna().any() or out["industry_name"].isna().any():
        raise ValueError("SW2021 L1 classification contains blank codes or names")
    if out["index_code"].duplicated(keep=False).any():
        duplicates = sorted(out.loc[out["index_code"].duplicated(keep=False), "index_code"].unique())
        raise ValueError(f"duplicate SW2021 L1 codes: {duplicates}")
    if expected_count is not None and len(out) != expected_count:
        raise ValueError(f"expected {expected_count} SW2021 L1 industries, got {len(out)}")
    return out.sort_values("index_code").reset_index(drop=True)


def _validate_member_intervals(members: pd.DataFrame) -> None:
    invalid = members["in_date"].isna() | (
        members["out_date"].notna() & (members["out_date"] <= members["in_date"])
    )
    if invalid.any():
        sample = members.loc[invalid, ["ts_code", "l1_code", "in_date", "out_date"]].head(5)
        raise ValueError(f"invalid SW2021 industry intervals: {sample.to_dict('records')}")

    conflicts: list[dict] = []
    for ts_code, sub in members.groupby("ts_code", sort=False):
        ordered = sub.sort_values(["in_date", "out_date"], na_position="last")
        previous_out: pd.Timestamp | None = None
        has_previous = False
        for row in ordered.itertuples(index=False):
            if has_previous and (previous_out is None or row.in_date < previous_out):
                conflicts.append(
                    {
                        "ts_code": ts_code,
                        "l1_code": row.l1_code,
                        "in_date": row.in_date,
                        "previous_out_date": previous_out,
                    }
                )
                if len(conflicts) >= 5:
                    break
            previous_out = None if pd.isna(row.out_date) else pd.Timestamp(row.out_date)
            has_previous = True
        if len(conflicts) >= 5:
            break
    if conflicts:
        raise ValueError(f"overlapping SW2021 industry intervals: {conflicts}")


def normalize_sw_members(
    members: pd.DataFrame,
    classification: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize historical membership and reject ambiguous intervals."""

    _require_columns(members, {"l1_code", "ts_code", "in_date", "out_date"}, "SW2021 members")
    classes = normalize_sw_classification(classification, expected_count=None)
    name_by_code = classes.set_index("index_code")["industry_name"]

    out = members.copy()
    out["l1_code"] = out["l1_code"].astype("string").str.strip()
    out["ts_code"] = out["ts_code"].astype("string").str.strip()
    out["in_date"] = _parse_dates(out["in_date"])
    out["out_date"] = _parse_dates(out["out_date"])
    if "l1_name" not in out.columns:
        out["l1_name"] = out["l1_code"].map(name_by_code)
    else:
        out["l1_name"] = out["l1_name"].astype("string").str.strip()
        out["l1_name"] = out["l1_name"].fillna(out["l1_code"].map(name_by_code))

    unknown_codes = sorted(set(out["l1_code"].dropna()).difference(name_by_code.index))
    if unknown_codes:
        raise ValueError(f"SW2021 members contain unknown L1 codes: {unknown_codes[:10]}")
    if out[["l1_code", "l1_name", "ts_code"]].isna().any().any():
        raise ValueError("SW2021 members contain blank codes or names")

    columns = ["l1_code", "l1_name", "ts_code", "in_date", "out_date"]
    optional = [col for col in ["name", "is_new"] if col in out.columns]
    out = out[columns + optional].drop_duplicates(columns).reset_index(drop=True)
    _validate_member_intervals(out)
    return out.sort_values(["ts_code", "in_date", "l1_code"]).reset_index(drop=True)


def attach_pit_industry(
    panel: pd.DataFrame,
    members: pd.DataFrame,
    coverage_threshold: float | None = None,
) -> pd.DataFrame:
    """Attach SW2021 L1 industry using half-open historical intervals."""

    _require_columns(panel, {"trade_date", "ts_code"}, "CNE6 panel")
    _require_columns(members, {"l1_code", "l1_name", "ts_code", "in_date", "out_date"}, "SW2021 members")
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("CNE6 panel contains duplicate trade_date/ts_code rows")
    _validate_member_intervals(members)

    left = panel.copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"], errors="coerce")
    left["ts_code"] = left["ts_code"].astype("string")
    left["_industry_row_order"] = range(len(left))
    left = left.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    right = members[["ts_code", "l1_code", "l1_name", "in_date", "out_date"]].copy()
    right["ts_code"] = right["ts_code"].astype("string")
    right["in_date"] = pd.to_datetime(right["in_date"], errors="coerce")
    right["out_date"] = pd.to_datetime(right["out_date"], errors="coerce")
    right = right.sort_values(["in_date", "ts_code"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        by="ts_code",
        left_on="trade_date",
        right_on="in_date",
        direction="backward",
        allow_exact_matches=True,
    )
    active = merged["l1_code"].notna() & (
        merged["out_date"].isna() | (merged["trade_date"] < merged["out_date"])
    )
    merged["industry_sw_l1_code"] = merged["l1_code"].where(active).astype("string")
    merged["industry_sw_l1_name"] = merged["l1_name"].where(active).astype("string")
    merged["industry_sw_l1_in_date"] = merged["in_date"].where(active)
    merged["industry_sw_l1_out_date"] = merged["out_date"].where(active)
    merged["industry_source"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[active, "industry_source"] = "SW2021"

    coverage = float(active.mean()) if len(merged) else 0.0
    if coverage_threshold is not None and coverage < coverage_threshold:
        raise ValueError(
            f"SW2021 industry coverage {coverage:.4%} is below required {coverage_threshold:.4%}"
        )

    drop_cols = ["l1_code", "l1_name", "in_date", "out_date"]
    return (
        merged.sort_values("_industry_row_order")
        .drop(columns=drop_cols + ["_industry_row_order"])
        .reset_index(drop=True)
    )


def industry_coverage_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize yearly PIT industry coverage for the model universe."""

    _require_columns(panel, {"trade_date", "ts_code", "industry_sw_l1_code"}, "V2 panel")
    work = panel[["trade_date", "ts_code", "industry_sw_l1_code"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["year"] = work["trade_date"].dt.year
    work["matched"] = work["industry_sw_l1_code"].notna()
    rows = []
    for year, sub in work.dropna(subset=["year"]).groupby("year", sort=True):
        matched = int(sub["matched"].sum())
        rows.append(
            {
                "year": int(year),
                "rows": int(len(sub)),
                "matched_rows": matched,
                "unmatched_rows": int(len(sub) - matched),
                "coverage": matched / len(sub) if len(sub) else 0.0,
                "stocks": int(sub["ts_code"].nunique()),
                "industries": int(sub.loc[sub["matched"], "industry_sw_l1_code"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fetch_sw2021_industry(
    pro: IndustryPro,
    raw_dir: Path = RAW_DIR,
    cache: bool = True,
    expected_count: int = 31,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch SW2021 L1 classification and full historical memberships."""

    classify_path = raw_dir / SW_CLASSIFICATION_FILE
    members_path = raw_dir / SW_MEMBERS_FILE
    if cache and classify_path.exists() and members_path.exists():
        classification = normalize_sw_classification(pd.read_parquet(classify_path), expected_count)
        members = normalize_sw_members(pd.read_parquet(members_path), classification)
        history_flags = set(members.get("is_new", pd.Series(dtype="string")).dropna().astype(str))
        if "N" in history_flags:
            return classification, members

    classification = normalize_sw_classification(
        pro.index_classify(level="L1", src="SW2021"),
        expected_count=expected_count,
    )
    frames = []
    for code in classification["index_code"]:
        for is_new in ("Y", "N"):
            frame = pro.index_member_all(l1_code=code, is_new=is_new)
            if frame is not None and not frame.empty:
                if "is_new" not in frame.columns:
                    frame = frame.assign(is_new=is_new)
                frames.append(frame)
    if not frames:
        raise RuntimeError("Tushare returned no SW2021 L1 historical memberships")
    members = normalize_sw_members(pd.concat(frames, ignore_index=True), classification)
    if cache:
        _atomic_parquet(classification, classify_path)
        _atomic_parquet(members, members_path)
    return classification, members


def fetch_citic_members_audit(
    pro: IndustryPro,
    raw_dir: Path = RAW_DIR,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch CITIC L1 memberships for cross-classification audit only."""

    path = raw_dir / CITIC_MEMBERS_FILE
    if cache and path.exists():
        out = pd.read_parquet(path)
        for col in ["in_date", "out_date"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out

    frames = []
    for code in CITIC_L1_CODES:
        frame = pro.ci_index_member(l1_code=code)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Tushare returned no CITIC L1 memberships")
    out = pd.concat(frames, ignore_index=True).drop_duplicates().copy()
    _require_columns(out, {"l1_code", "l1_name", "ts_code", "in_date", "out_date"}, "CITIC members")
    out["in_date"] = _parse_dates(out["in_date"])
    out["out_date"] = _parse_dates(out["out_date"])
    out = out.sort_values(["ts_code", "in_date", "l1_code"]).reset_index(drop=True)
    if cache:
        _atomic_parquet(out, path)
    return out


def industry_crosswalk_audit(
    sw_members: pd.DataFrame,
    citic_members: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build a stock-count crosswalk without using CITIC as a fallback."""

    date = pd.Timestamp(as_of_date)

    def _active(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        work = frame.copy()
        work["in_date"] = pd.to_datetime(work["in_date"], errors="coerce")
        work["out_date"] = pd.to_datetime(work["out_date"], errors="coerce")
        mask = (work["in_date"] <= date) & (work["out_date"].isna() | (date < work["out_date"]))
        selected = work.loc[mask, ["ts_code", "l1_code", "l1_name"]].copy()
        if selected["ts_code"].duplicated().any():
            raise ValueError(f"multiple active {prefix} industries on {date.date()}")
        return selected.rename(columns={"l1_code": f"{prefix}_l1_code", "l1_name": f"{prefix}_l1_name"})

    sw = _active(sw_members, "sw")
    citic = _active(citic_members, "citic")
    joined = sw.merge(citic, on="ts_code", how="outer")
    rows = (
        joined.groupby(["sw_l1_code", "sw_l1_name", "citic_l1_code", "citic_l1_name"], dropna=False)
        .size()
        .rename("stocks")
        .reset_index()
    )
    rows.insert(0, "as_of_date", date)
    return rows.sort_values(["sw_l1_code", "citic_l1_code"], na_position="last").reset_index(drop=True)


def run(token_env: str = "TUSHARE_TOKEN", cache: bool = True, include_citic: bool = True) -> None:
    from factors.data.cne6_fetcher import get_tushare_pro

    pro = get_tushare_pro(token_env=token_env)
    classification, members = fetch_sw2021_industry(pro, cache=cache)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset": "sw2021_l1_classification",
                "rows": len(classification),
                "stocks": pd.NA,
                "start": pd.NaT,
                "end": pd.NaT,
            },
            {
                "dataset": "sw2021_l1_members",
                "rows": len(members),
                "stocks": members["ts_code"].nunique(),
                "start": members["in_date"].min(),
                "end": members["out_date"].max(),
            },
        ]
    ).to_csv(OUTPUT_DIR / "industry_fetch_audit.csv", index=False)
    if include_citic:
        citic = fetch_citic_members_audit(pro, cache=cache)
        industry_crosswalk_audit(members, citic, pd.Timestamp.today().normalize()).to_csv(
            OUTPUT_DIR / "industry_crosswalk_latest.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-citic", action="store_true")
    args = parser.parse_args()
    run(token_env=args.token_env, cache=not args.no_cache, include_citic=not args.skip_citic)


if __name__ == "__main__":
    main()
