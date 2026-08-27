"""Versioned orchestration for the CNE6 enhanced V2 risk model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factors.data.cne6_industry import attach_pit_industry, industry_coverage_audit
from factors.risk.cne6_regression import run_constrained_factor_return_regression
from factors.risk.cne6_v2_exposures import compute_v2_exposures
from factors.risk.cne6_v2_spec import descriptor_metadata_v2, descriptor_specs_v2
from factors.risk.eigenfactor_covariance import (
    covariance_to_long,
    eigenfactor_effective_dates,
    estimate_covariance_with_fallback,
    restore_display_covariance,
)
from factors.risk.multifrequency_specific_risk import (
    SpecificRiskConfig,
    estimate_multifrequency_specific_risk,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "cne6_enhanced_v2"
DEFAULT_HISTORY = ROOT_DIR / "data" / "processed" / "cne6_csi500_daily_history.parquet"
DEFAULT_MEMBERS = ROOT_DIR / "data" / "raw" / "cne6_sw2021_l1_members.parquet"


@dataclass(frozen=True)
class ModelPaths:
    root: Path = DEFAULT_OUTPUT

    @property
    def panel(self) -> Path:
        return self.root / "cne6_csi500_daily_panel_v2.parquet"

    @property
    def history(self) -> Path:
        return self.root / "cne6_csi500_daily_history_v2.parquet"

    @property
    def industry_audit(self) -> Path:
        return self.root / "industry_mapping_audit.csv"

    @property
    def descriptors(self) -> Path:
        return self.root / "descriptor_exposures.parquet"

    @property
    def styles(self) -> Path:
        return self.root / "style_exposures.parquet"

    @property
    def descriptor_metadata(self) -> Path:
        return self.root / "descriptor_metadata.csv"

    @property
    def descriptor_admission(self) -> Path:
        return self.root / "descriptor_admission.parquet"

    @property
    def factor_returns(self) -> Path:
        return self.root / "factor_returns.csv"

    @property
    def specific_returns(self) -> Path:
        return self.root / "specific_returns.parquet"

    @property
    def regression_diagnostics(self) -> Path:
        return self.root / "regression_diagnostics.csv"

    @property
    def covariance_base(self) -> Path:
        return self.root / "factor_covariance_base.parquet"

    @property
    def covariance_eigenfactor(self) -> Path:
        return self.root / "factor_covariance_eigenfactor.parquet"

    @property
    def covariance_diagnostics(self) -> Path:
        return self.root / "factor_covariance_diagnostics.csv"

    @property
    def specific_risk(self) -> Path:
        return self.root / "specific_risk_multifrequency.parquet"

    @property
    def manifest(self) -> Path:
        return self.root / "model_manifest.json"


def _atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        raise ValueError(f"unsupported output format: {path}")
    temporary.replace(path)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _configuration_hash(parameters: dict) -> str:
    descriptor_config = [asdict(spec) for spec in descriptor_specs_v2()]
    payload = json.dumps(
        {"descriptors": descriptor_config, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_model_manifest(
    paths: ModelPaths,
    stage: str,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    parameters: dict,
) -> dict:
    paths.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": "enhanced_v2",
        "stage": stage,
        "configuration_hash": _configuration_hash(parameters),
        "git_commit": _git_commit(),
        "parameters": parameters,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    temporary = paths.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(paths.manifest)
    return manifest


def build_versioned_panel(
    history: pd.DataFrame,
    industry_members: pd.DataFrame,
    formal_start: str | pd.Timestamp = "2021-01-01",
    threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = attach_pit_industry(history, industry_members)
    member_mask = (
        enriched["csi500_member"].fillna(False).astype(bool)
        if "csi500_member" in enriched.columns
        else pd.Series(True, index=enriched.index)
    )
    audit = industry_coverage_audit(enriched.loc[member_mask])
    formal_year = pd.Timestamp(formal_start).year
    audit["validation_scope"] = np.where(audit["year"] >= formal_year, "formal", "research_backfill")
    failed = audit[(audit["year"] >= formal_year) & (audit["coverage"] < threshold)]
    if not failed.empty:
        raise ValueError(
            "formal SW2021 industry coverage failed: "
            + failed[["year", "coverage"]].to_dict("records").__str__()
        )
    return enriched, audit


def run_panel_stage(
    paths: ModelPaths,
    history_path: Path = DEFAULT_HISTORY,
    members_path: Path = DEFAULT_MEMBERS,
    formal_start: str = "2021-01-01",
    threshold: float = 0.99,
) -> None:
    history = pd.read_parquet(history_path)
    members = pd.read_parquet(members_path)
    enriched, audit = build_versioned_panel(history, members, formal_start=formal_start, threshold=threshold)
    member = enriched.loc[enriched["csi500_member"].fillna(False).astype(bool)].copy()
    _atomic_write_frame(enriched, paths.history)
    _atomic_write_frame(member, paths.panel)
    _atomic_write_frame(audit, paths.industry_audit)
    write_model_manifest(
        paths,
        "panel",
        {"history": history_path, "industry_members": members_path},
        {"history_v2": paths.history, "panel_v2": paths.panel, "industry_audit": paths.industry_audit},
        {"formal_start": formal_start, "industry_coverage_threshold": threshold},
    )


def _exposure_input_columns(path: Path) -> list[str]:
    available = set(pq.ParquetFile(path).schema.names)
    requested = {"trade_date", "ts_code", "csi500_member", "total_mv"}
    for spec in descriptor_specs_v2():
        requested.update(spec.required_columns)
    requested.update(
        {
            "returns_1d",
            "csi500_return",
            "close_adj",
            "close",
            "amount",
            "turnover_rate",
            "total_assets",
            "total_liab",
            "net_profit_ttm",
            "n_cashflow_act_ttm",
        }
    )
    return sorted(requested.intersection(available))


def run_exposure_stage(
    paths: ModelPaths,
    calculation_start: str = "2020-01-01",
    formal_start: str = "2021-01-01",
) -> None:
    formal_codes = pd.read_parquet(
        paths.panel,
        columns=["ts_code"],
        filters=[("trade_date", ">=", pd.Timestamp(formal_start))],
    )["ts_code"].dropna().astype(str).unique().tolist()
    history = pd.read_parquet(
        paths.history,
        columns=_exposure_input_columns(paths.history),
        filters=[
            ("trade_date", ">=", pd.Timestamp(calculation_start)),
            ("ts_code", "in", formal_codes),
        ],
    )
    descriptors, styles, admission = compute_v2_exposures(history)
    formal_date = pd.Timestamp(formal_start)
    descriptors = descriptors[descriptors["trade_date"] >= formal_date].reset_index(drop=True)
    styles = styles[styles["trade_date"] >= formal_date].reset_index(drop=True)
    admission = admission[admission["trade_date"] >= formal_date].reset_index(drop=True)
    _atomic_write_frame(descriptors, paths.descriptors)
    _atomic_write_frame(styles, paths.styles)
    _atomic_write_frame(admission, paths.descriptor_admission)
    _atomic_write_frame(descriptor_metadata_v2(history.columns), paths.descriptor_metadata)
    write_model_manifest(
        paths,
        "exposures",
        {"history_v2": paths.history},
        {
            "descriptors": paths.descriptors,
            "styles": paths.styles,
            "admission": paths.descriptor_admission,
            "metadata": paths.descriptor_metadata,
        },
        {
            "calculation_start": calculation_start,
            "formal_start": formal_start,
            "minimum_descriptor_coverage": 0.70,
            "minimum_effective_weight": 0.60,
        },
    )


def run_regression_stage(paths: ModelPaths, formal_start: str = "2021-01-01") -> None:
    panel = pd.read_parquet(paths.panel)
    panel = panel[pd.to_datetime(panel["trade_date"]) >= pd.Timestamp(formal_start)].copy()
    styles = pd.read_parquet(paths.styles)
    factors, residuals, diagnostics = run_constrained_factor_return_regression(panel, styles)
    _atomic_write_frame(factors, paths.factor_returns)
    _atomic_write_frame(residuals, paths.specific_returns)
    _atomic_write_frame(diagnostics, paths.regression_diagnostics)
    write_model_manifest(
        paths,
        "regression",
        {"panel_v2": paths.panel, "styles": paths.styles},
        {
            "factor_returns": paths.factor_returns,
            "specific_returns": paths.specific_returns,
            "diagnostics": paths.regression_diagnostics,
        },
        {"formal_start": formal_start, "return_alignment": "t_minus_1_exposure_to_t_return", "weight": "sqrt_total_mv"},
    )


def _industry_weights_for_date(panel: pd.DataFrame, date: pd.Timestamp, factor_columns: list[str]) -> dict[str, float]:
    sub = panel[panel["trade_date"] == date].copy()
    caps = sub.groupby("industry_sw_l1_code", sort=False)["total_mv"].sum()
    weights = {}
    for column in factor_columns:
        if column.startswith("industry_"):
            weights[column] = float(caps.get(column.removeprefix("industry_"), 0.0))
    if sum(weights.values()) <= 0:
        return {column: 1.0 for column in weights}
    return weights


def run_covariance_stage(
    paths: ModelPaths,
    simulations: int = 500,
    formal_start: str = "2021-01-01",
) -> None:
    factors = pd.read_csv(paths.factor_returns, parse_dates=["trade_date"]).sort_values("trade_date")
    panel = pd.read_parquet(paths.panel, columns=["trade_date", "industry_sw_l1_code", "total_mv"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    mappings = eigenfactor_effective_dates(factors["trade_date"])
    mappings = mappings[mappings["effective_date"] >= pd.Timestamp(formal_start)]
    factor_columns = [column for column in factors.columns if column != "trade_date"]
    base_frames = []
    final_frames = []
    diagnostics = []
    for row in mappings.itertuples(index=False):
        history = factors[factors["trade_date"] <= row.calibration_date].tail(504)
        if len(history) < 252:
            continue
        recent_required = history.tail(252)
        available = [column for column in factor_columns if recent_required[column].notna().all()]
        if "country" not in available or sum(column.startswith("industry_") for column in available) < 2:
            continue
        selected = history[available]
        industry_weights = _industry_weights_for_date(panel, row.calibration_date, available)
        published = estimate_covariance_with_fallback(
            selected,
            industry_weights=industry_weights,
            simulations=simulations,
            random_seed=729 + int(pd.Timestamp(row.calibration_date).strftime("%Y%m%d")),
        )
        final = covariance_to_long(
            published.matrix,
            published.columns,
            row.effective_date,
            published.method,
        )
        final["calibration_date"] = row.calibration_date
        final_frames.append(final)
        if published.base_estimate is not None:
            base_display = restore_display_covariance(published.base_estimate.matrix, published.basis)
            base = covariance_to_long(base_display, published.columns, row.effective_date, "ewma_newey_west")
            base["calibration_date"] = row.calibration_date
            base_frames.append(base)
        diagnostics.append(
            {
                "calibration_date": row.calibration_date,
                "effective_date": row.effective_date,
                "method": published.method,
                "fallback_reason": published.fallback_reason,
                "factors": len(published.columns),
                "min_eigenvalue": float(np.linalg.eigvalsh(published.matrix).min()),
                "simulations": simulations if published.adjustment is not None else 0,
                "clipped_multipliers": published.adjustment.clipped_multipliers if published.adjustment is not None else np.nan,
            }
        )
    if not final_frames:
        raise ValueError("no covariance estimates were produced")
    _atomic_write_frame(pd.concat(final_frames, ignore_index=True), paths.covariance_eigenfactor)
    _atomic_write_frame(pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame(), paths.covariance_base)
    _atomic_write_frame(pd.DataFrame(diagnostics), paths.covariance_diagnostics)
    write_model_manifest(
        paths,
        "covariance",
        {"factor_returns": paths.factor_returns, "panel_v2": paths.panel},
        {
            "base": paths.covariance_base,
            "eigenfactor": paths.covariance_eigenfactor,
            "diagnostics": paths.covariance_diagnostics,
        },
        {"window": 504, "minimum_history": 252, "half_life": 90, "newey_west_lags": 2, "simulations": simulations},
    )


def run_specific_risk_stage(paths: ModelPaths, config: SpecificRiskConfig | None = None) -> None:
    residuals = pd.read_parquet(paths.specific_returns)
    residuals["trade_date"] = pd.to_datetime(residuals["trade_date"])
    minimum_date = residuals["trade_date"].min()
    maximum_date = residuals["trade_date"].max()
    panel = pd.read_parquet(
        paths.panel,
        columns=["trade_date", "ts_code", "industry_sw_l1_code", "total_mv"],
        filters=[
            ("trade_date", ">=", minimum_date),
            ("trade_date", "<=", maximum_date),
        ],
    )
    config = SpecificRiskConfig() if config is None else config
    risk = estimate_multifrequency_specific_risk(residuals, panel, config=config)
    _atomic_write_frame(risk, paths.specific_risk)
    write_model_manifest(
        paths,
        "specific_risk",
        {"specific_returns": paths.specific_returns, "panel_v2": paths.panel},
        {"specific_risk": paths.specific_risk},
        asdict(config),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["panel", "exposures", "regression", "covariance", "specific-risk", "report", "all"],
        required=True,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--members", default=str(DEFAULT_MEMBERS))
    parser.add_argument("--formal-start", default="2021-01-01")
    parser.add_argument("--calculation-start", default="2020-01-01")
    parser.add_argument("--industry-coverage-threshold", type=float, default=0.99)
    parser.add_argument("--simulations", type=int, default=500)
    args = parser.parse_args()
    paths = ModelPaths(Path(args.output))
    stages = ["panel", "exposures", "regression", "covariance", "specific-risk", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "panel":
            run_panel_stage(
                paths,
                Path(args.history),
                Path(args.members),
                formal_start=args.formal_start,
                threshold=args.industry_coverage_threshold,
            )
        elif stage == "exposures":
            run_exposure_stage(paths, calculation_start=args.calculation_start, formal_start=args.formal_start)
        elif stage == "regression":
            run_regression_stage(paths, formal_start=args.formal_start)
        elif stage == "covariance":
            run_covariance_stage(paths, simulations=args.simulations, formal_start=args.formal_start)
        elif stage == "specific-risk":
            run_specific_risk_stage(paths)
        elif stage == "report":
            from factors.reports.cne6_v2_report import generate_report

            generate_report(paths)


if __name__ == "__main__":
    main()
