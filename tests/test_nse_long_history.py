"""Tests for the real long-history cache boundary."""

import sys
from types import SimpleNamespace

import pandas as pd

from app.data.nse_long_history import canonical_yfinance_symbol, fetch_long_history


def test_alias_is_canonicalized_before_provider_fetch():
    assert canonical_yfinance_symbol("BAJAJ_AUTO") == "BAJAJ-AUTO"
    assert canonical_yfinance_symbol("reliance") == "RELIANCE"


def test_fetch_preserves_exchange_local_date_and_writes_provenance(tmp_path, monkeypatch):
    index = pd.DatetimeIndex(
        ["2020-01-02 00:00:00+05:30", "2020-01-03 00:00:00+05:30"],
    )
    frame = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Adj Close": [100.0, 101.0],
        "Volume": [1000, 1100],
        "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0],
    }, index=index)

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            return frame

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    manifest = fetch_long_history(
        ["RELIANCE"], output_dir=tmp_path, include_index=False,
    )
    stored = pd.read_parquet(tmp_path / "RELIANCE_v5.parquet")
    assert stored.index[0] == pd.Timestamp("2020-01-02")
    assert manifest["entries"][0]["price_semantics"] == "raw_ohlc_and_adjusted_close"
    assert manifest["entries"][0]["status"] == "ok"
    assert manifest["manifest_sha256"]
