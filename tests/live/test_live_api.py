"""Live API tests: the endpoints the frontend calls, against real data."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.data.nse import ist_today

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def test_ohlcv_daily_ends_on_latest_session(client, latest_session):
    r = client.get("/api/quant/ohlcv/RELIANCE", params={"days": 60})
    assert r.status_code == 200, r.text
    body = r.json()
    last = datetime.fromtimestamp(body["data"][-1]["time"], tz=timezone.utc).date()
    assert last == latest_session
    assert body["last_bar"]["source"]


def test_ohlcv_intraday_is_from_latest_session(client, latest_session):
    r = client.get("/api/quant/ohlcv/RELIANCE", params={"days": 2, "interval": "5m"})
    assert r.status_code == 200, r.text
    bars = r.json()["data"]
    assert bars, "no intraday bars"
    last = datetime.fromtimestamp(bars[-1]["time"], tz=timezone.utc).date()
    assert last == latest_session


def test_unknown_symbol_is_404_not_500(client):
    r = client.get("/api/quant/ohlcv/NOTAREALSYMBOLXYZ", params={"days": 30})
    assert r.status_code == 404, r.text


def test_regime_is_computed_on_data_through_latest_session(client, latest_session):
    from app.api import quant_routes

    if hasattr(quant_routes.get_regime, "_regime_cache"):
        delattr(quant_routes.get_regime, "_regime_cache")
    r = client.get("/api/quant/regime")
    body = r.json()
    assert body["current_regime"] in {"BULL", "BEAR", "SIDEWAYS"}, body
    assert body["date_range"].endswith(str(latest_session))


def test_quick_quote_matches_latest_close(client, latest_session):
    import yfinance as yf

    r = client.get("/api/company/TCS/quick")
    assert r.status_code == 200, r.text
    ref = yf.Ticker("TCS.NS").history(period="5d", auto_adjust=False)["Close"].iloc[-1]
    assert abs(r.json()["price"] - float(ref)) / float(ref) < 0.01


def test_option_chain_is_real_or_503_never_synthetic(client):
    r = client.get("/api/quant/options/NIFTY")
    if r.status_code == 503:
        assert r.json()["detail"]["code"] == "OPTIONS_DATA_UNAVAILABLE"
        pytest.skip("NSE option chain refused automated access (honest 503)")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("source") not in (None, "none", "synthetic", "mock")
    assert body["calls"] and body["puts"]


@pytest.fixture(scope="module")
def fresh_live_cache(tmp_path_factory):
    """Recompute the dashboard from live data instead of reusing data_cache/live."""
    import app.quant.live_dashboard as ld

    original = ld.live_dashboard_cache
    ld.live_dashboard_cache = ld.LiveDashboardCache(path=tmp_path_factory.mktemp("live") / "d.json")
    yield ld.live_dashboard_cache
    ld.live_dashboard_cache = original


def test_dashboard_portfolio_is_computed_through_latest_session(client, fresh_live_cache, latest_session):
    r = client.get("/api/quant/portfolio")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["as_of"] == str(latest_session)
    assert body["stale"] is False
    assert body["windows"]["test_end"] == str(latest_session)
    assert body["n_days"] == body["windows"]["oos_sessions"]
    assert body["universe_size"] >= 50
    # gate flag must agree with the reported numbers (rule 6)
    expected = (body["vt_sharpe"] >= 1.0 and body["vt_max_dd"] <= 0.25 and body["n_days"] >= 504)
    assert body["vt_validated"] is expected
    assert "report_version" not in body  # no longer served from a static report


def test_dashboard_pairs_are_live_and_gate_consistent(client, fresh_live_cache, latest_session):
    body = client.get("/api/quant/pairs").json()
    assert body["as_of"] == str(latest_session)
    assert body["total_scanned"] > 1000
    assert body["validated"] == sum(p["validated"] for p in body["pairs"])
    for p in body["pairs"]:
        assert p["signal_date"] == str(latest_session)
        assert p["validated"] is (
            p["oos"]["sharpe_ratio"] >= 1.0 and p["oos"]["max_drawdown"] <= 0.25
            and p["oos"]["num_trading_days"] >= 504
        )


def test_regime_reports_real_time_in_state(client, latest_session):
    body = client.get("/api/quant/regime").json()
    assert body["as_of"] == str(latest_session)
    assert body["sessions_in_state"] >= 1
    assert 0.0 <= body["prob_leave_next_session"] <= 1.0
