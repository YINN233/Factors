"""Incremental CNE6 R2 optimization experiments inside the CSI500 universe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factors.risk.cne6_regression import run_factor_return_regression


def _summary(diagnostics: pd.DataFrame, variant: str) -> dict:
    ok = diagnostics.loc[diagnostics["regression_status"] == "ok"].copy()
    if ok.empty:
        return {
            "variant": variant,
            "dates": 0,
            "mean_r2": float("nan"),
            "median_r2": float("nan"),
            "mean_adj_r2": float("nan"),
            "share_r2_gt_05": float("nan"),
            "mean_n_obs": float("nan"),
            "mean_n_factors": float("nan"),
            "mean_n_industries": float("nan"),
            "mean_condition_number": float("nan"),
            "mean_resid_std": float("nan"),
        }
    return {
        "variant": variant,
        "dates": int(len(ok)),
        "mean_r2": float(ok["r2"].mean()),
        "median_r2": float(ok["r2"].median()),
        "mean_adj_r2": float(ok["adj_r2"].mean()),
        "share_r2_gt_05": float((ok["r2"] > 0.5).mean()),
        "mean_n_obs": float(ok["n_obs"].mean()),
        "mean_n_factors": float(ok["n_factors"].mean()),
        "mean_n_industries": float(ok["n_industries"].mean()),
        "mean_condition_number": float(ok["condition_number"].mean()),
        "mean_resid_std": float(ok["resid_std"].mean()),
    }


def _write_variant(
    panel: pd.DataFrame,
    style: pd.DataFrame,
    output_dir: Path,
    variant: str,
    *,
    include_industry: bool = True,
    style_columns: list[str] | None = None,
    industry_min_obs: int = 0,
    fit_method: str = "wls",
    write_artifacts: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    available_style_cols = [c for c in style.columns if c.startswith("style_") and not c.endswith("_n")]
    # None means "use all available styles"; [] is a deliberate no-style ablation.
    selected_style_cols = available_style_cols if style_columns is None else style_columns
    missing = sorted(set(selected_style_cols) - set(available_style_cols))
    if missing:
        raise ValueError(f"unknown style columns: {missing}")
    style_input = style[["trade_date", "ts_code"] + selected_style_cols].copy()
    factor_returns, residuals, diagnostics = run_factor_return_regression(
        panel,
        style_input,
        return_mode="same_day",
        industry_min_obs=industry_min_obs,
        fit_method=fit_method,
        include_industry=include_industry,
    )
    if write_artifacts:
        factor_returns.to_csv(output_dir / f"{variant}_factor_returns.csv", index=False)
        residuals.to_parquet(output_dir / f"{variant}_specific_returns.parquet", index=False)
    diagnostics.to_csv(output_dir / f"{variant}_diagnostics.csv", index=False)
    return _summary(diagnostics, variant)


def run(
    panel_path: Path,
    style_path: Path,
    output_dir: Path,
    reuse_existing: bool = True,
    with_style_ablations: bool = False,
    recompute_standard: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    style = pd.read_parquet(style_path)
    rows: list[dict] = []

    existing = {
        "baseline_original": Path("outputs/cne6_r2_return_definition_comparison/r2_return_definition_diagnostics.csv"),
        "p0_continuous_returns_nonlinear_size": Path("outputs/cne6_r2_optimization_p0/regression_diagnostics.csv"),
        "industry_min_obs_5": Path("outputs/cne6_r2_optimization_industry5/regression_diagnostics.csv"),
        "industry_min_obs_10": Path("outputs/cne6_r2_optimization_industry10/regression_diagnostics.csv"),
        "huber": Path("outputs/cne6_r2_optimization_huber/regression_diagnostics.csv"),
    }
    for variant, path in existing.items():
        if not reuse_existing or not path.exists():
            continue
        if recompute_standard and variant != "baseline_original":
            continue
        diagnostics = pd.read_csv(path, parse_dates=["trade_date"])
        if variant == "baseline_original":
            diagnostics = diagnostics.loc[diagnostics["return_mode"] == "same_day"]
        rows.append(_summary(diagnostics, variant))

    style_cols = [c for c in style.columns if c.startswith("style_") and not c.endswith("_n")]

    if recompute_standard:
        rows.append(
            _write_variant(
                panel,
                style,
                Path("outputs/cne6_r2_optimization_p0"),
                "p0_continuous_returns_nonlinear_size",
            )
        )
        for min_obs, path in [
            (5, Path("outputs/cne6_r2_optimization_industry5")),
            (10, Path("outputs/cne6_r2_optimization_industry10")),
        ]:
            rows.append(
                _write_variant(
                    panel,
                    style,
                    path,
                    f"industry_min_obs_{min_obs}",
                    industry_min_obs=min_obs,
                )
            )
        rows.append(
            _write_variant(
                panel,
                style,
                Path("outputs/cne6_r2_optimization_huber"),
                "huber",
                fit_method="huber",
            )
        )
    no_style_dir = output_dir / "no_style"
    rows.append(
        _write_variant(
            panel,
            style,
            no_style_dir,
            "no_style",
            style_columns=[],
            include_industry=True,
        )
    )

    if with_style_ablations:
        # These are diagnostics only.  Avoid writing tens of GB of repeated
        # residual/factor files while keeping every daily R2 observable.
        for removed in style_cols:
            kept = [col for col in style_cols if col != removed]
            variant = f"leave_one_out_{removed.removeprefix('style_')}"
            rows.append(
                _write_variant(
                    panel,
                    style,
                    output_dir / "style_leave_one_out",
                    variant,
                    style_columns=kept,
                    include_industry=True,
                    write_artifacts=False,
                )
            )
    no_industry_dir = output_dir / "no_industry"
    rows.append(
        _write_variant(
            panel,
            style,
            no_industry_dir,
            "no_industry",
            style_columns=style_cols,
            include_industry=False,
        )
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "optimization_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/cne6_csi500_daily_panel.parquet")
    parser.add_argument("--style", default="outputs/cne6_reproduction/style_exposures.parquet")
    parser.add_argument("--output", default="outputs/cne6_r2_optimization")
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--with-style-ablations", action="store_true")
    parser.add_argument("--recompute-standard", action="store_true")
    args = parser.parse_args()
    summary = run(
        Path(args.panel),
        Path(args.style),
        Path(args.output),
        reuse_existing=not args.no_reuse_existing,
        with_style_ablations=args.with_style_ablations,
        recompute_standard=args.recompute_standard,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
