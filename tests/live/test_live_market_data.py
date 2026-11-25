"""
Live data-layer tests: are the bars real, current, correctly dated, and
consistent across providers?  Every assertion here runs against today's
market data.  See conftest.py for how to enable.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd
import pytest

from app.data.nse import _fetch_jugaad, _fetch_yfinance, _session_closed, fetch_nse_ohlcv, ist_today

pytestmark = pytest.mark.live

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]


@pytest.fixture(scope="module")
def live_bars(tmp_path_factory):
    """Fresh fetch into an empty cache — nothing reused from data_cache/."""
    cache = tmp_path_factory.mktemp("nse_live_cache")
    end = ist_today()
    start = end - timedelta(days=45)
    return {s: fetch_nse_ohlcv(s, start, end, cache_dir=cache) for s in SYMBOLS}


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_last_bar_is_the_latest_nse_session(live_bars, latest_session, symbol):
    df = live_bars[symbol]
    assert not df.empty, f"no live bars for {symbol}"
    assert df["date"].max().date() == latest_session, (
        f"{symbol} last bar {df['date'].max().date()} != latest NSE session {latest_session}"
    )


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_bar_falls_on_a_real_nse_session(live_bars, nifty_sessions, symbol):
    """Catches the jugaad-data off-by-one (IST date delivered as UTC 18:30)."""
    df = live_bars[symbol]
    window = nifty_sessions[nifty_sessions >= df["date"].min()]
    # Yahoo's index history omits special sessions that NSE's own archive
    # records (e.g. Diwali Muhurat trading on 2026-11-08, a Sunday).  An
    # extra date is accepted only when NSE itself supplied the bar.  A
    # one-day shift is still caught by the strict `missing` check below,
    # because shifting also removes bars from genuine sessions.
    extra = df[~df["date"].isin(window)]
    not_from_nse = extra[~extra["source"].str.startswith("jugaad-data")]
    missing = sorted(set(window) - set(df["date"]))
    assert not_from_nse.empty, (
        f"{symbol} non-session bars not backed by NSE archive: "
        f"{not_from_nse[['date', 'source']].to_dict('records')}"
    )
    assert not missing, f"{symbol} missing sessions: {missing}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_ohlcv_is_internally_consistent(live_bars, symbol):
    df = live_bars[symbol]
    assert df["date"].is_monotonic_increasing and not df["date"].duplicated().any()
    assert (df["date"] == df["date"].dt.normalize()).all(), "dates must be session midnights"
    for col in ("open", "high", "low", "close"):
        assert df[col].notna().all() and (df[col] > 0).all(), col
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (df["volume"] > 0).all()


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_bar_has_provenance(live_bars, symbol):
    df = live_bars[symbol]
    assert df["source"].str.len().gt(0).all()
    assert set(df["source"].str.split(":").str[0]) <= {"jugaad-data", "yfinance"}


def test_provisional_flag_matches_session_state(live_bars):
    today = pd.Timestamp(ist_today())
    for symbol, df in live_bars.items():
        today_rows = df[df["date"] == today]
        assert (df.loc[df["date"] != today, "provisional"] == False).all()  # noqa: E712
        if not today_rows.empty:
            assert bool(today_rows["provisional"].iloc[0]) is (not _session_closed()), symbol


@pytest.mark.parametrize("symbol", ["RELIANCE", "TCS"])
def test_jugaad_and_yahoo_agree_bar_for_bar(symbol):
    """Two independent providers must report the same as-traded closes."""
    end = ist_today()
    start = end - timedelta(days=30)
    jug = _fetch_jugaad(symbol, start, end)
    yfd = _fetch_yfinance(symbol, start, end)
    if jug is None or jug.empty:
        pytest.skip("jugaad-data (NSE archive) unavailable right now")
    assert yfd is not None and not yfd.empty, "Yahoo returned nothing"
    both = jug.merge(yfd, on="date", suffixes=("_nse", "_yf"))
    assert len(both) >= 5, f"too little overlap: {len(both)} sessions"
    rel = (both["close_nse"] - both["close_yf"]).abs() / both["close_nse"]
    assert rel.max() < 0.005, both.loc[rel.idxmax(), ["date", "close_nse", "close_yf"]].to_dict()


def test_cache_round_trip_preserves_bars(tmp_path):
    end = ist_today()
    start = end - timedelta(days=20)
    first = fetch_nse_ohlcv("ITC", start, end, cache_dir=tmp_path)
    second = fetch_nse_ohlcv("ITC", start, end, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        first[["date", "close", "source"]], second[["date", "close", "source"]]
    )


def test_india_vix_includes_latest_session(latest_session):
    from app.data.vix import fetch_vix_yfinance

    vix = fetch_vix_yfinance(latest_session - timedelta(days=20), ist_today())
    assert vix is not None and not vix.empty
    assert vix.index.max().date() == latest_session
    last = float(vix["Close"].iloc[-1])
    assert math.isfinite(last) and 5 < last < 100


def test_news_feed_is_recent():
    from app.data.news import fetch_news

    items = fetch_news("RELIANCE", max_items=15)
    assert items, "no headlines from any RSS source"
    assert all(item.get("title") and item.get("source") for item in items)
    newest = max(pd.to_datetime(i["date"], errors="coerce") for i in items if i.get("date"))
    assert pd.Timestamp(ist_today()) - newest.normalize() <= pd.Timedelta(days=7), newest


def test_live_fundamentals_have_sources_and_finite_values():
    from app.data.fundamentals import get_full_fundamentals

    rows = get_full_fundamentals("TCS", start_fy=ist_today().year - 2)
    assert len(rows) >= 10
    assert all(r.source for r in rows)
    assert all(r.value is None or math.isfinite(r.value) for r in rows)
    market = [r for r in rows if r.metric == "price_close_fy_end"]
    assert market, "no market-derived metrics"


def test_mcx_bhavcopy_is_real_or_explicitly_unavailable():
    """MCX blocks automated access at times; it must fail loudly, never fake rows."""
    from app.data.mcx import fetch_mcx_history

    try:
        df = fetch_mcx_history(ist_today() - timedelta(days=7), commodity="GOLD")
    except Exception as exc:  # provider refusal is an honest outcome
        pytest.skip(f"MCX bhavcopy unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    if df.empty:
        pytest.skip("MCX returned no rows for the last week")
    assert (df["close"] > 0).all()


def test_statement_currency_is_reported_not_assumed():
    """Infosys files in USD on Yahoo; its amounts must not be labelled INR."""
    import yfinance as yf

    from app.data.fundamentals import compute_statement_fundamentals

    ccy = (yf.Ticker("INFY.NS").info or {}).get("financialCurrency")
    rows = [r for r in compute_statement_fundamentals("INFY") if r.metric == "revenue"]
    assert rows, "no INFY revenue rows"
    expected_unit = "INR_crore" if ccy == "INR" else f"{ccy}_million"
    assert {r.unit for r in rows} == {expected_unit}
