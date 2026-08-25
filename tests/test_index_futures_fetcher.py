import pandas as pd

from factors.data.index_futures_fetcher import IndexFuturesFetcher


class FakePro:
    def fut_basic(self, exchange):
        return pd.DataFrame(
            {
                "ts_code": ["IF2608.CFX"],
                "fut_code": ["IF"],
                "list_date": ["2026-01-01"],
                "delist_date": ["2026-08-21"],
            }
        )

    def fut_daily(self, ts_code, start_date, end_date):
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": ["2026-08-03"], "close": [100.0]})

    def fut_mapping(self, ts_code, start_date, end_date):
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": ["2026-08-03"], "mapping_ts_code": ["IF2608.CFX"]})

    def index_daily(self, ts_code, start_date, end_date):
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": ["2026-08-03"], "close": [101.0]})


class FakeETFPro:
    def __init__(self):
        self.daily_ranges = []
        self.share_ranges = []

    def fund_daily(self, ts_code, start_date, end_date):
        self.daily_ranges.append((start_date, end_date))
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": [end_date], "close": [1.0], "vol": [10.0], "amount": [20.0]})

    def fund_share(self, ts_code, start_date, end_date):
        self.share_ranges.append((start_date, end_date))
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": [end_date], "fd_share": [100.0]})


def test_fetcher_caches_and_audits_with_fake_pro(tmp_path):
    fetcher = IndexFuturesFetcher(pro=FakePro(), raw_dir=tmp_path)
    basic = fetcher.fetch_futures_basic()
    daily = fetcher.fetch_futures_daily("2026-08-01", "2026-08-05")
    mapping = fetcher.fetch_futures_mapping("2026-08-01", "2026-08-05", products=["IF"])
    index = fetcher.fetch_index_daily("2026-08-01", "2026-08-05", products=["IF"])
    audit = fetcher.flush_audit()
    assert basic.iloc[0]["fut_code"] == "IF"
    assert daily.iloc[0]["product"] == "IF"
    assert mapping.iloc[0]["mapping_ts_code"] == "IF2608.CFX"
    assert index.iloc[0]["product"] == "IF"
    assert not audit.empty
    assert (tmp_path / "data_availability.csv").exists()


def test_etf_history_is_requested_in_bounded_date_chunks(tmp_path):
    pro = FakeETFPro()
    fetcher = IndexFuturesFetcher(pro=pro, raw_dir=tmp_path)
    daily = fetcher.fetch_etf_daily(["510300.SH"], "2020-01-01", "2021-01-01")
    shares = fetcher.fetch_etf_shares(["510300.SH"], "2020-01-01", "2021-01-01")
    assert len(daily) == len(pro.daily_ranges)
    assert len(shares) == len(pro.share_ranges)
    assert len(pro.daily_ranges) > 1
    assert all(start <= end for start, end in pro.daily_ranges)
    assert all(start <= end for start, end in pro.share_ranges)


def test_partial_etf_cache_is_not_reused_for_a_larger_request(tmp_path):
    pro = FakeETFPro()
    fetcher = IndexFuturesFetcher(pro=pro, raw_dir=tmp_path)
    fetcher.fetch_etf_daily(["510300.SH"], "2026-08-01", "2026-08-05")
    result = fetcher.fetch_etf_daily(["510300.SH", "510500.SH"], "2026-08-01", "2026-08-05")
    assert set(result["ts_code"]) == {"510300.SH", "510500.SH"}
    assert len(pro.daily_ranges) == 3
