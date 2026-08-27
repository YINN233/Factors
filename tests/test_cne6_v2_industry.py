import pandas as pd
import pytest

from factors.data.cne6_industry import (
    attach_pit_industry,
    fetch_sw2021_industry,
    industry_coverage_audit,
    normalize_sw_classification,
    normalize_sw_members,
)


def _classification() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI"],
            "industry_name": ["农林牧渔", "采掘"],
            "level": ["L1", "L1"],
            "src": ["SW2021", "SW2021"],
        }
    )


def test_pit_industry_uses_half_open_membership_intervals():
    panel = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ts_code": ["000001.SZ"] * 5,
            "industry": ["静态细行业"] * 5,
        }
    )
    members = pd.DataFrame(
        {
            "l1_code": ["801010.SI"],
            "l1_name": ["农林牧渔"],
            "ts_code": ["000001.SZ"],
            "in_date": ["20240102"],
            "out_date": ["20240104"],
        }
    )

    out = attach_pit_industry(panel, normalize_sw_members(members, _classification()))

    by_date = out.set_index("trade_date")
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-01"), "industry_sw_l1_code"])
    assert by_date.loc[pd.Timestamp("2024-01-02"), "industry_sw_l1_code"] == "801010.SI"
    assert by_date.loc[pd.Timestamp("2024-01-03"), "industry_sw_l1_name"] == "农林牧渔"
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-04"), "industry_sw_l1_code"])
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-05"), "industry_sw_l1_code"])
    assert out.loc[out["industry_sw_l1_code"].isna(), "industry_source"].isna().all()


def test_overlapping_membership_intervals_are_rejected():
    members = pd.DataFrame(
        {
            "l1_code": ["801010.SI", "801020.SI"],
            "l1_name": ["农林牧渔", "采掘"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "in_date": ["20240102", "20240103"],
            "out_date": ["20240105", None],
        }
    )

    with pytest.raises(ValueError, match="overlapping SW2021 industry intervals"):
        normalize_sw_members(members, _classification())


def test_unmatched_stock_is_not_filled_from_static_industry():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "ts_code": ["999999.SZ"],
            "industry": ["银行"],
        }
    )
    members = normalize_sw_members(
        pd.DataFrame(
            {
                "l1_code": ["801010.SI"],
                "l1_name": ["农林牧渔"],
                "ts_code": ["000001.SZ"],
                "in_date": ["20240101"],
                "out_date": [None],
            }
        ),
        _classification(),
    )

    out = attach_pit_industry(panel, members)

    assert pd.isna(out.loc[0, "industry_sw_l1_code"])
    assert pd.isna(out.loc[0, "industry_sw_l1_name"])
    assert out.loc[0, "industry"] == "银行"


def test_classification_contract_rejects_duplicate_codes():
    duplicated = pd.concat([_classification(), _classification().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate SW2021 L1 codes"):
        normalize_sw_classification(duplicated, expected_count=None)


def test_industry_coverage_audit_reports_yearly_coverage():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2025-01-02"]),
            "ts_code": ["A", "B", "A"],
            "industry_sw_l1_code": ["801010.SI", None, "801010.SI"],
        }
    )

    audit = industry_coverage_audit(panel).set_index("year")

    assert audit.loc[2024, "rows"] == 2
    assert audit.loc[2024, "matched_rows"] == 1
    assert audit.loc[2024, "coverage"] == 0.5
    assert audit.loc[2025, "coverage"] == 1.0


def test_multiple_stocks_are_matched_without_cross_talk():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "ts_code": ["A", "B", "A", "B"],
        }
    )
    members = normalize_sw_members(
        pd.DataFrame(
            {
                "l1_code": ["801010.SI", "801020.SI"],
                "l1_name": ["农林牧渔", "采掘"],
                "ts_code": ["A", "B"],
                "in_date": ["20240101", "20240103"],
                "out_date": [None, None],
            }
        ),
        _classification(),
    )

    out = attach_pit_industry(panel, members)

    assert out["industry_sw_l1_code"].tolist() == ["801010.SI", pd.NA, "801010.SI", "801020.SI"]


def test_coverage_threshold_fails_loudly():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ts_code": ["A", "B"],
        }
    )
    members = normalize_sw_members(
        pd.DataFrame(
            {
                "l1_code": ["801010.SI"],
                "l1_name": ["农林牧渔"],
                "ts_code": ["A"],
                "in_date": ["20240101"],
                "out_date": [None],
            }
        ),
        _classification(),
    )

    with pytest.raises(ValueError, match="coverage 50.0000%"):
        attach_pit_industry(panel, members, coverage_threshold=0.99)


class _FakeIndustryPro:
    def __init__(self):
        self.member_calls = []

    def index_classify(self, **kwargs):
        assert kwargs == {"level": "L1", "src": "SW2021"}
        return _classification()

    def index_member_all(self, **kwargs):
        code = kwargs["l1_code"]
        is_new = kwargs["is_new"]
        self.member_calls.append((code, is_new))
        current = is_new == "Y"
        return pd.DataFrame(
            {
                "l1_code": [code],
                "l1_name": [_classification().set_index("index_code").loc[code, "industry_name"]],
                "ts_code": [("A" if code == "801010.SI" else "B") + ("_NEW" if current else "_OLD")],
                "in_date": ["20240101" if current else "20100101"],
                "out_date": [None if current else "20200101"],
                "is_new": [is_new],
            }
        )


def test_fetch_sw2021_industry_caches_normalized_contract(tmp_path):
    pro = _FakeIndustryPro()

    classification, members = fetch_sw2021_industry(pro, raw_dir=tmp_path, expected_count=2)

    assert pro.member_calls == [
        ("801010.SI", "Y"),
        ("801010.SI", "N"),
        ("801020.SI", "Y"),
        ("801020.SI", "N"),
    ]
    assert len(classification) == 2
    assert set(members["is_new"]) == {"Y", "N"}
    assert len(members) == 4
    assert (tmp_path / "cne6_sw2021_l1_classify.parquet").exists()
    assert (tmp_path / "cne6_sw2021_l1_members.parquet").exists()

    cached_pro = _FakeIndustryPro()
    cached_pro.index_classify = lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache not used"))
    cached = fetch_sw2021_industry(cached_pro, raw_dir=tmp_path, expected_count=2)
    assert len(cached[0]) == 2
    assert len(cached[1]) == 4
