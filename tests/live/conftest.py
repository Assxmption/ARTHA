"""
Live-market integration tests — real NSE/Yahoo/RSS feeds, no fixtures.

These hit the network and depend on market hours and provider uptime, so
they are opt-in and never run in the default `pytest` invocation:

    ARTHA_LIVE=1 python -m pytest tests/live -v
    ARTHA_LIVE=1 ARTHA_LIVE_LLM=1 python -m pytest tests/live -v   # + LLM writer

The independent reference for "what is the latest NSE session" is the NIFTY
50 index history from Yahoo, so freshness checks never hard-code a calendar
(weekends, exchange holidays and special sessions such as Diwali Muhurat
trading are all taken from the market itself).
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

LIVE = os.getenv("ARTHA_LIVE", "").lower() in ("1", "true", "yes")
LIVE_LLM = os.getenv("ARTHA_LIVE_LLM", "").lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config, items):
    skip_live = pytest.mark.skip(reason="live-market test; set ARTHA_LIVE=1 to run")
    skip_llm = pytest.mark.skip(reason="live LLM test; set ARTHA_LIVE_LLM=1 to run")
    for item in items:
        if "live" in item.keywords and not LIVE:
            item.add_marker(skip_live)
        if "live_llm" in item.keywords and not (LIVE and LIVE_LLM):
            item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def nifty_sessions() -> pd.DatetimeIndex:
    """NSE session dates over the last ~60 days, from the NIFTY 50 index."""
    import yfinance as yf

    from app.data.nse import normalize_session_dates

    hist = yf.Ticker("^NSEI").history(period="3mo", auto_adjust=False)
    if hist.empty:
        pytest.fail("Yahoo returned no NIFTY 50 history — cannot establish the NSE calendar")
    return pd.DatetimeIndex(normalize_session_dates(pd.Series(hist.index)))


@pytest.fixture(scope="session")
def latest_session(nifty_sessions) -> date:
    return nifty_sessions.max().date()


@pytest.fixture
def live_window() -> tuple[date, date]:
    from app.data.nse import ist_today

    today = ist_today()
    return today - timedelta(days=45), today
