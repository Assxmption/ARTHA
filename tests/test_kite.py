"""Kite Connect client: auth flow, session handling, chunking — no network (httpx.MockTransport)."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timedelta

import httpx
import pandas as pd
import pytest

import app.data.kite as kite


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("KITE_API_KEY", "testkey")
    monkeypatch.setenv("KITE_API_SECRET", "testsecret")
    monkeypatch.setattr(kite, "SESSION_FILE", tmp_path / "session.json")
    monkeypatch.setattr(kite, "INSTRUMENTS_DIR", tmp_path / "instruments")
    for lim in (kite._quote_limiter, kite._historical_limiter, kite._default_limiter):
        monkeypatch.setattr(lim, "min_interval", 0.0)
    return tmp_path


def test_module_is_read_only():
    src = inspect.getsource(kite).lower()
    for forbidden in ("/orders", "place_order", "/gtt", "/portfolio/positions", "modify_order"):
        assert forbidden not in src, forbidden


def test_checksum_is_sha256_of_key_token_secret():
    assert kite.checksum("k", "t", "s") == hashlib.sha256(b"kts").hexdigest()


def test_login_state_is_single_use(configured):
    url, state = kite.login_url()
    assert "api_key=testkey" in url and "redirect_params=state%3D" in url
    assert kite.consume_state(state) is True
    assert kite.consume_state(state) is False
    assert kite.consume_state("forged") is False


def test_exchange_request_token_stores_private_session(configured):
    def handler(req: httpx.Request):
        form = dict(x.split("=") for x in req.content.decode().split("&"))
        assert form["checksum"] == kite.checksum("testkey", "rt123", "testsecret")
        return httpx.Response(200, json={"status": "success", "data": {"access_token": "AT", "user_id": "AB1234"}})

    session = kite.exchange_request_token("rt123", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert session.access_token == "AT"
    stored = configured / "session.json"
    assert json.loads(stored.read_text())["user_id"] == "AB1234"
    assert oct(stored.stat().st_mode & 0o777) == "0o600"


def test_session_expires_at_the_0600_rollover():
    created = datetime(2026, 11, 23, 9, 0, tzinfo=kite.IST).isoformat()
    s = kite.KiteSession("AT", "U", created)
    assert s.is_current(datetime(2026, 11, 24, 5, 59, tzinfo=kite.IST))
    assert not s.is_current(datetime(2026, 11, 24, 6, 1, tzinfo=kite.IST))


def test_token_exception_clears_session_and_raises_not_connected(configured):
    kite.save_session(kite.KiteSession("AT", "U", datetime.now(kite.IST).isoformat()))
    client = kite.KiteClient(
        kite.load_session(),
        http=httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(403, json={"status": "error", "error_type": "TokenException", "message": "expired"}))),
    )
    with pytest.raises(kite.KiteNotConnected):
        client.profile()
    assert not (configured / "session.json").exists()


def test_historical_is_chunked_to_kite_limits(configured):
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.url.params["from"], req.url.params["to"]))
        day = req.url.params["from"][:10]
        return httpx.Response(200, json={"status": "success", "data": {"candles": [
            [f"{day}T09:15:00+0530", 1, 2, 0.5, 1.5, 100]]}})

    client = kite.KiteClient(kite.KiteSession("AT", "U", datetime.now(kite.IST).isoformat()),
                             http=httpx.Client(transport=httpx.MockTransport(handler)))
    start = datetime(2026, 1, 1)
    frame = client.historical(123, "minute", start, start + timedelta(days=150))
    assert len(calls) == 3  # 60-day limit for minute data
    assert frame["date"].iloc[0] == pd.Timestamp("2026-01-01 09:15")  # IST wall clock, tz-naive


def test_not_connected_without_session(configured):
    with pytest.raises(kite.KiteNotConnected):
        kite.get_client()


@pytest.mark.parametrize("sym, expected", [("NIFTY", ("NSE", "NIFTY 50")), ("^NSEI", ("NSE", "NIFTY 50")),
                                          ("RELIANCE.NS", ("NSE", "RELIANCE")), ("BANKNIFTY", ("NSE", "NIFTY BANK"))])
def test_symbol_mapping(sym, expected):
    assert kite.kite_symbol(sym) == expected
