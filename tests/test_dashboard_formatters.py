import numpy as np
import pandas as pd

from factors.dashboard.components.formatters import (
    display_frame,
    format_percent,
    quality_text,
    status_text,
)


def test_formatters_keep_missing_values_explicit():
    assert format_percent(-0.01234) == "-1.23%"
    assert format_percent(np.nan) == "不可用"
    assert quality_text("C") == "C · 代理/人工口径"
    assert status_text("limited_support") == "支持有限"


def test_display_frame_uses_chinese_labels_without_mutating_source():
    source = pd.DataFrame(
        {
            "product": ["IC"],
            "raw_annualized_basis": [-0.08],
            "basis_percentile": [20.0],
        }
    )
    original = source.copy(deep=True)
    shown = display_frame(source)
    pd.testing.assert_frame_equal(source, original)
    assert list(shown.columns) == ["品种", "原始年化基差", "历史分位"]
    assert shown.loc[0, "原始年化基差"] == "-8.00%"
    assert shown.loc[0, "历史分位"] == "20.0"


def test_evidence_groups_and_validation_splits_are_localized():
    shown = display_frame(pd.DataFrame({"evidence_group": ["pair_basis"], "split": ["holdout"]}))
    assert shown.loc[0, "证据组"] == "配对基差结构"
    assert shown.loc[0, "样本阶段"] == "留出期"


def test_tenor_rank_uses_research_labels():
    shown = display_frame(pd.DataFrame({"tenor_rank": [1, 2, 3, 4]}))
    assert shown["期限序号"].tolist() == ["近月", "次月", "季月", "次季月"]
