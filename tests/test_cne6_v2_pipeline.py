import json

import pandas as pd
import pytest

from factors.risk.cne6_v2_pipeline import (
    ModelPaths,
    build_versioned_panel,
    write_model_manifest,
)
from factors.reports.cne6_portfolio_attribution import run_attribution


def _members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "l1_code": ["I1", "I2"],
            "l1_name": ["行业一", "行业二"],
            "ts_code": ["A", "B"],
            "in_date": pd.to_datetime(["2010-01-01", "2021-01-01"]),
            "out_date": [pd.NaT, pd.NaT],
        }
    )


def test_versioned_panel_enforces_coverage_only_from_formal_start():
    history = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2019-01-02", "2019-01-02", "2021-01-04", "2021-01-04"]),
            "ts_code": ["A", "B", "A", "B"],
            "csi500_member": [True] * 4,
        }
    )

    panel, audit = build_versioned_panel(history, _members(), formal_start="2021-01-01", threshold=0.99)

    assert len(panel) == 4
    by_year = audit.set_index("year")
    assert by_year.loc[2019, "coverage"] == 0.5
    assert by_year.loc[2019, "validation_scope"] == "research_backfill"
    assert by_year.loc[2021, "coverage"] == 1.0
    assert by_year.loc[2021, "validation_scope"] == "formal"


def test_versioned_panel_rejects_low_formal_period_coverage():
    history = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2021-01-04", "2021-01-04"]),
            "ts_code": ["A", "B"],
            "csi500_member": [True, True],
        }
    )
    only_one_member = _members().iloc[[0]].copy()

    with pytest.raises(ValueError, match="formal SW2021 industry coverage failed"):
        build_versioned_panel(history, only_one_member, formal_start="2021-01-01", threshold=0.99)


def test_manifest_is_atomic_and_contains_configuration_hash(tmp_path):
    paths = ModelPaths(root=tmp_path)

    manifest = write_model_manifest(
        paths,
        stage="exposures",
        inputs={"panel": tmp_path / "panel.parquet"},
        outputs={"styles": tmp_path / "styles.parquet"},
        parameters={"formal_start": "2021-01-01"},
    )

    saved = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert saved == manifest
    assert len(saved["configuration_hash"]) == 64
    assert saved["model_version"] == "enhanced_v2"
    assert saved["stage"] == "exposures"
    assert not paths.manifest.with_suffix(".json.tmp").exists()


def test_attribution_accepts_v2_covariance_and_specific_variance():
    date = pd.Timestamp("2024-01-31")
    weights = pd.DataFrame(
        {"trade_date": [date, date], "ts_code": ["A", "B"], "weight": [0.6, 0.4]}
    )
    panel = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["A", "B"],
            "industry_sw_l1_code": ["I1", "I2"],
            "csi500_index_weight": [0.5, 0.5],
        }
    )
    styles = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["A", "B"],
            "style_size": [1.0, -1.0],
            "style_size_effective_weight": [1.0, 1.0],
        }
    )
    covariance = pd.DataFrame(
        {
            "trade_date": [date, date, date],
            "factor_i": ["style_size", "industry_I1", "industry_I2"],
            "factor_j": ["style_size", "industry_I1", "industry_I2"],
            "covariance": [0.0001, 0.0002, 0.0002],
        }
    )
    specific = pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["A", "B"],
            "specific_variance_daily": [0.0003, 0.0004],
        }
    )

    exposures, risk, _ = run_attribution(
        weights,
        panel,
        styles,
        covariance,
        specific,
        window=None,
        industry_col="industry_sw_l1_code",
        specific_variance_col="specific_variance_daily",
    )

    assert "industry_I1" in set(exposures["factor"])
    assert "style_size_effective_weight" not in set(exposures["factor"])
    assert risk.loc[0, "specific_var_daily"] > 0
