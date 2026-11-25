"""Top-bar index ticker: one batched Kite quote, labelled Yahoo fallback — no network."""

from __future__ import annotations

import pytest

import app.data.intraday as intraday
from app.data import kite


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(intraday, "_ticker_cache", {"at": 0.0, "body": None})


class FakeKite:
    calls = 0

    def quote(self, keys):
        FakeKite.calls += 1
        data = {
            "NSE:NIFTY 50": (25349.65, 25800.6),
            "NSE:NIFTY BANK": (62585.35, 64635.2),
            "BSE:SENSEX": (82435.99, 83919.71),
            "NSE:INDIA VIX": (17.7, 17.32),
        }
        return {k: {"last_price": v[0], "ohlc": {"close": v[1]}} for k, v in data.items() if k in keys}


def test_kite_path_uses_a_single_batched_quote(monkeypatch):
    FakeKite.calls = 0
    monkeypatch.setattr(kite, "get_client", lambda: FakeKite())
    body = intraday.get_index_ticker()
    assert FakeKite.calls == 1
    assert body["live"] is True and body["source"] == "kite"
    nifty = next(i for i in body["items"] if i["label"] == "NIFTY 50")
    assert nifty["change"] == pytest.approx(-450.95)
    assert nifty["change_pct"] == pytest.approx(-1.75)
    assert [i["label"] for i in body["items"]] == ["NIFTY 50", "BANKNIFTY", "SENSEX", "INDIA VIX"]


def test_result_is_cached_briefly(monkeypatch):
    FakeKite.calls = 0
    monkeypatch.setattr(kite, "get_client", lambda: FakeKite())
    intraday.get_index_ticker()
    intraday.get_index_ticker()
    assert FakeKite.calls == 1


def test_falls_back_to_yahoo_and_says_so(monkeypatch):
    def not_connected():
        raise kite.KiteNotConnected("no session")

    class FI:
        last_price, previous_close = 100.0, 98.0

    class T:
        def __init__(self, sym):
            self.fast_info = FI()

    import yfinance

    monkeypatch.setattr(kite, "get_client", not_connected)
    monkeypatch.setattr(yfinance, "Ticker", T)
    body = intraday.get_index_ticker()
    assert body["live"] is False and "delayed" in body["source"]
    assert all(i["change_pct"] == pytest.approx(2.04) for i in body["items"])


def test_missing_previous_close_gives_none_not_a_made_up_change(monkeypatch):
    class K:
        def quote(self, keys):
            return {k: {"last_price": 10.0, "ohlc": {"close": 0}} for k in keys}

    monkeypatch.setattr(kite, "get_client", lambda: K())
    body = intraday.get_index_ticker()
    assert all(i["change"] is None and i["change_pct"] is None for i in body["items"])
