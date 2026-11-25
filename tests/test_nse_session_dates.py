"""
Offline regression tests for NSE session-date handling.

The fixtures are the exact shapes the real providers return (verified live on
2026-11-23): jugaad-data delivers the 2026-11-13 session as the naive
timestamp 2026-11-12 18:30:00; yfinance delivers 2026-11-13 00:00+05:30.
"""

from datetime import date, datetime

import pandas as pd

import app.data.nse as nse
from app.data.nse import normalize_session_dates


def test_jugaad_utc_instants_map_to_ist_session_dates():
    raw = pd.Series(pd.to_datetime(["2026-11-12 18:30:00", "2026-11-15 18:30:00"]))
    assert list(normalize_session_dates(raw)) == [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-16")]


def test_yfinance_tz_aware_dates_map_to_session_dates():
    raw = pd.Series(pd.DatetimeIndex(["2026-11-13", "2026-11-16"]).tz_localize("Asia/Kolkata"))
    assert list(normalize_session_dates(raw)) == [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-16")]


def test_midnight_dates_are_unchanged():
    raw = pd.Series(pd.to_datetime(["2026-11-13", "2026-11-08"]))  # 11-08: Diwali Muhurat (Sunday)
    assert list(normalize_session_dates(raw)) == list(raw)


def _bars(dates, source, close=100.0):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000, "source": source,
    })


def test_yfinance_tops_up_sessions_jugaad_has_not_published(monkeypatch):
    """jugaad-data lags; a non-empty but stale result must not hide newer sessions."""
    monkeypatch.setattr(nse, "ist_today", lambda: date(2026, 11, 23))
    monkeypatch.setattr(nse, "_session_closed", lambda now=None: True)
    monkeypatch.setattr(nse, "_fetch_jugaad", lambda s, a, b: _bars(["2026-11-19", "2026-11-20"], "jugaad-data:NSE_EQ"))
    asked = {}

    def fake_yf(symbol, start, end):
        asked["range"] = (start, end)
        return _bars(["2026-11-23"], "yfinance:TCS.NS", close=101.0)

    monkeypatch.setattr(nse, "_fetch_yfinance", fake_yf)
    df = nse._fetch_range("TCS", date(2026, 11, 19), date(2026, 11, 23))
    assert asked["range"] == (date(2026, 11, 21), date(2026, 11, 23))
    assert list(df["date"].dt.date) == [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)]
    assert list(df["source"].str.split(":").str[0]) == ["jugaad-data", "jugaad-data", "yfinance"]


def test_todays_bar_is_provisional_until_close(monkeypatch):
    monkeypatch.setattr(nse, "ist_today", lambda: date(2026, 11, 23))
    monkeypatch.setattr(nse, "_session_closed", lambda now=None: False)
    monkeypatch.setattr(nse, "_fetch_jugaad", lambda s, a, b: None)
    monkeypatch.setattr(nse, "_fetch_yfinance", lambda s, a, b: _bars(["2026-11-20", "2026-11-23"], "yfinance:X.NS"))
    df = nse._fetch_range("X", date(2026, 11, 20), date(2026, 11, 23))
    assert list(df["provisional"]) == [False, True]


def test_cache_refetches_tail_when_request_reaches_today(monkeypatch, tmp_path):
    """A bar cached mid-session must be replaced by later fetches."""
    monkeypatch.setattr(nse, "ist_today", lambda: date(2026, 11, 23))
    monkeypatch.setattr(nse, "_session_closed", lambda now=None: False)
    calls = []

    def fake_range(symbol, start, end):
        calls.append((start, end))
        df = _bars(["2026-11-20", "2026-11-23"], "yfinance:X.NS", close=100.0 + len(calls))
        df["provisional"] = [False, True]
        return df

    monkeypatch.setattr(nse, "_fetch_range", fake_range)
    monkeypatch.setattr(nse, "_flag_ingestion_outliers", lambda df, s: None)
    first = nse.fetch_nse_ohlcv("X", date(2026, 11, 20), cache_dir=tmp_path)
    second = nse.fetch_nse_ohlcv("X", date(2026, 11, 20), cache_dir=tmp_path)
    assert len(calls) == 2 and calls[1][0] == date(2026, 11, 23)
    assert second["close"].iloc[-1] == 102.0 and first["close"].iloc[-1] == 101.0


def test_legacy_cache_without_provenance_is_discarded(monkeypatch, tmp_path):
    monkeypatch.setattr(nse, "ist_today", lambda: date(2026, 11, 23))
    legacy = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-12 18:30:00", "2026-11-13 00:00:00"]),  # same session twice
        "open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0], "volume": 1, "symbol": "X",
    })
    legacy.to_parquet(tmp_path / "X.parquet", index=False)
    monkeypatch.setattr(nse, "_fetch_range", lambda s, a, b: _bars(["2026-11-13"], "yfinance:X.NS"))
    monkeypatch.setattr(nse, "_flag_ingestion_outliers", lambda df, s: None)
    df = nse.fetch_nse_ohlcv("X", date(2026, 11, 13), date(2026, 11, 13), cache_dir=tmp_path)
    assert list(df["close"]) == [100.0] and df["source"].iloc[0] == "yfinance:X.NS"


def test_future_end_date_is_clamped_to_today(monkeypatch, tmp_path):
    monkeypatch.setattr(nse, "ist_today", lambda: date(2026, 11, 23))
    seen = {}
    monkeypatch.setattr(nse, "_fetch_range", lambda s, a, b: seen.setdefault("end", b) and pd.DataFrame())
    nse.fetch_nse_ohlcv("X", date(2026, 11, 1), date(2027, 1, 1), cache_dir=tmp_path)
    assert seen["end"] == date(2026, 11, 23)
