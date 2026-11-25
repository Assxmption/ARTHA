"""
Live Kite Connect tests — real broker data. Skipped unless ARTHA_LIVE=1 AND a
Kite session exists for today (log in via /api/kite/login first).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.data import kite
from app.data.nse import ist_today

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", autouse=True)
def require_kite():
    if not kite.is_connected():
        pytest.skip("No Kite session today — log in at /api/kite/login")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_status_reports_connected_without_secrets(client):
    body = client.get("/api/kite/status").json()
    assert body["connected"] is True
    assert "access_token" not in str(body) and "api_secret" not in str(body).lower()


def test_daily_bars_come_from_kite_and_end_on_latest_session(client, latest_session):
    body = client.get("/api/quant/ohlcv/RELIANCE", params={"days": 30, "interval": "1d"}).json()
    assert body["source"].startswith("kite:") and body["live"] is True
    last = datetime.utcfromtimestamp(body["data"][-1]["time"]).date()
    assert last == latest_session


def test_kite_and_yahoo_daily_closes_agree(latest_session):
    import yfinance as yf

    c = kite.get_client()
    bars = c.historical(c.instrument_token("TCS"), "day",
                        datetime.combine(latest_session - timedelta(days=20), datetime.min.time()),
                        datetime.combine(latest_session, datetime.max.time().replace(microsecond=0)))
    k = bars.assign(d=bars["date"].dt.normalize()).set_index("d")["close"]
    y = yf.Ticker("TCS.NS").history(start=str(latest_session - timedelta(days=20)), auto_adjust=False)["Close"]
    y.index = y.index.tz_localize(None).normalize()
    both = pd.concat([k, y], axis=1, keys=["kite", "yahoo"]).dropna()
    assert len(both) >= 5
    # Yahoo sometimes carries the last trade instead of NSE's official close.
    assert ((both.kite - both.yahoo).abs() / both.kite).max() < 0.01


def test_intraday_minute_bars_cover_the_session(client, latest_session):
    body = client.get("/api/quant/ohlcv/NIFTY", params={"days": 1, "interval": "1m"}).json()
    assert body["source"].startswith("kite:")
    times = [datetime.utcfromtimestamp(b["time"]) for b in body["data"]]
    assert times[0].date() == latest_session and times[0].strftime("%H:%M") == "09:15"


def test_quote_is_live_with_circuit_limits(client):
    q = client.get("/api/quant/quote/RELIANCE").json()
    assert q["live"] is True and q["source"].startswith("kite:")
    assert q["lower_circuit"] < q["ltp"] < q["upper_circuit"]
    assert set(q["depth"]) == {"buy", "sell"}


def test_option_chain_is_live_and_iv_is_consistent_with_vix(client):
    oc = client.get("/api/quant/options/NIFTY").json()
    assert oc["source"] == "kite"
    assert datetime.fromisoformat(oc["expiry"]).date() >= ist_today()
    atm = min(range(len(oc["calls"])), key=lambda i: abs(oc["calls"][i]["strike"] - oc["spotPrice"]))
    ivs = [x for x in (oc["calls"][atm]["impliedVolatility"], oc["puts"][atm]["impliedVolatility"]) if x]
    assert ivs, "ATM IV could not be solved"
    from app.data.vix import fetch_vix_kite

    vix = float(fetch_vix_kite(ist_today() - timedelta(days=10))["Close"].iloc[-1])
    # Nearest-expiry ATM IV and the 30-day VIX differ, but not by a factor of 2.
    assert 0.5 * vix < sum(ivs) / len(ivs) < 2.0 * vix


def test_session_replay_on_kite_data_reconciles_to_official_closes(latest_session):
    from app.quant.session_replay import fetch_daily_closes, replay_session

    r = replay_session(latest_session, save=False)
    assert r["sources"]["minute"].startswith("kite") and r["sources"]["daily"].startswith("kite")
    held = r["book"]["weights"]
    closes, _ = fetch_daily_closes(list(held), latest_session - timedelta(days=10), latest_session)
    prev, cur = closes.loc[pd.Timestamp(r["previous_session"])], closes.loc[pd.Timestamp(r["session"])]
    expected = sum(w * (cur[s] / prev[s] - 1) for s, w in held.items())
    assert r["result"]["book_return_eod"] == pytest.approx(expected, abs=1e-6)
