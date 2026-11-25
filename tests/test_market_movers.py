"""Market movers: rankings, breadth, honest gaps, and source selection."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import kite, market_movers as mm


def _r(sym, pct, tv=None, vol=None):
    return {"symbol": sym, "change_pct": pct, "traded_value": tv, "volume": vol}


def test_rankings_and_breadth():
    rows = [_r("A", 2.0, 10), _r("B", -1.0, 50), _r("C", 0.0, 5), _r("D", 5.0, 1), _r("E", -3.0, 20)]
    s = mm.summarize_movers(rows, top_n=2)
    assert [r["symbol"] for r in s["gainers"]] == ["D", "A"]
    assert [r["symbol"] for r in s["losers"]] == ["E", "B"]
    assert [r["symbol"] for r in s["most_active"]] == ["B", "E"]
    assert s["active_metric"] == "traded_value"
    assert s["breadth"] == {"advances": 2, "declines": 2, "unchanged": 1, "ad_ratio": 1.0}


def test_unchanged_never_listed_as_gainer_or_loser():
    s = mm.summarize_movers([_r("A", 0.0), _r("B", 0.0)])
    assert s["gainers"] == [] and s["losers"] == []
    assert s["breadth"]["ad_ratio"] is None  # no declines → ratio undefined, not infinite


def test_missing_quotes_are_reported_not_ranked():
    s = mm.summarize_movers([_r("A", 1.0), _r("GONE", None)])
    assert s["unquoted"] == ["GONE"]
    assert all(r["symbol"] != "GONE" for r in s["rows"])
    assert s["breadth"]["advances"] == 1


def test_ties_break_on_symbol():
    s = mm.summarize_movers([_r("Z", 1.0), _r("A", 1.0)])
    assert [r["symbol"] for r in s["gainers"]] == ["A", "Z"]


def test_empty_universe():
    s = mm.summarize_movers([])
    assert s["gainers"] == s["losers"] == s["most_active"] == []
    assert s["breadth"] == {"advances": 0, "declines": 0, "unchanged": 0, "ad_ratio": None}


def test_volume_used_when_no_traded_value():
    s = mm.summarize_movers([_r("A", 1.0, None, 100), _r("B", -1.0, None, 300)])
    assert [r["symbol"] for r in s["most_active"]] == ["B", "A"]
    assert s["active_metric"] == "volume"


def test_row_rejects_zero_or_nan_previous_close():
    assert mm._row("X", 100, 0)["change_pct"] is None
    assert mm._row("X", 100, float("nan"))["change_pct"] is None
    r = mm._row("X", 110, 100, volume=10, avg_price=105)
    assert r["change_pct"] == 10.0 and r["traded_value"] == 1050.0


class _FakeClient:
    def __init__(self, quotes):
        self.quotes = quotes
        self.calls = 0

    def quote(self, keys):
        self.calls += 1
        return {k: self.quotes[k] for k in keys if k in self.quotes}


def test_kite_single_batched_call_and_cache(monkeypatch):
    client = _FakeClient({
        "NSE:AAA": {"last_price": 105, "ohlc": {"close": 100}, "volume": 10, "average_price": 104},
        "NSE:BBB": {"last_price": 95, "ohlc": {"close": 100}, "volume": 20, "average_price": 96},
    })
    monkeypatch.setattr(kite, "get_client", lambda: client)
    monkeypatch.setattr(kite, "kite_symbol", lambda s: ("NSE", s))
    monkeypatch.setattr(mm, "_yahoo_rows", lambda s: pytest.fail("Yahoo must not be used when Kite works"))
    cache = mm.MoversCache(symbols=lambda: ["AAA", "BBB", "CCC"])
    out = cache.get(top_n=5)
    assert out["source"] == "kite" and out["live"] is True
    assert [r["symbol"] for r in out["gainers"]] == ["AAA"]
    assert [r["symbol"] for r in out["losers"]] == ["BBB"]
    assert out["unquoted"] == ["CCC"]
    cache.get(top_n=5)
    assert client.calls == 1


def test_yahoo_fallback_is_labelled_and_drops_stale(monkeypatch):
    def not_connected():
        raise kite.KiteNotConnected("no session")

    monkeypatch.setattr(kite, "get_client", not_connected)
    idx = pd.to_datetime(["2026-11-20", "2026-11-23"])
    fresh = pd.DataFrame({"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [100.0, 102.0], "Volume": [5, 7]}, index=idx)
    old = pd.DataFrame({"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [50.0, 40.0], "Volume": [5, 7]},
                       index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
    raw = pd.concat({"AAA.NS": fresh, "OLD.NS": old.reindex(idx.union(old.index))}, axis=1, sort=True)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(mm, "_yfinance_symbol", lambda s: f"{s}.NS")
    out = mm.MoversCache(symbols=lambda: ["AAA", "OLD"]).get()
    assert out["live"] is False and "delayed" in out["source"]
    assert [r["symbol"] for r in out["gainers"]] == ["AAA"]
    assert out["gainers"][0]["change_pct"] == 2.0
    assert out["stale"] == ["OLD"] and "OLD" in out["unquoted"]
