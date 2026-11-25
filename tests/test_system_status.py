"""System status: reports gate thresholds from code and never leaks secrets."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app
from app.quant import backtest


def test_status_reports_code_gate_and_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value_should_never_appear")
    body = TestClient(app).get("/api/system/status").json()
    assert body["validation_gate"] == {
        "min_oos_sharpe": backtest.MIN_OOS_SHARPE,
        "max_oos_drawdown": backtest.MAX_OOS_DRAWDOWN,
        "min_oos_sessions": backtest.MIN_OOS_TRADING_DAYS,
    }
    assert body["order_placement"] is False
    assert set(body["llm"]) >= {"groq", "gemini"}
    assert "gsk_secret_value_should_never_appear" not in json.dumps(body)


def test_kite_session_window_rolls_at_six_ist(monkeypatch):
    from app.api import kite_routes
    from app.data import kite

    monkeypatch.setattr(kite, "load_session", lambda: kite.KiteSession("tok", "AB1234", "2026-11-23T05:30:00+05:30"))
    w = kite_routes._session_window()
    # Issued before 06:00 → belongs to the previous token day, so it dies at 06:00 the same morning.
    assert w["session_expires_at"] == "2026-11-23T06:00:00+05:30"
    monkeypatch.setattr(kite, "load_session", lambda: kite.KiteSession("tok", "AB1234", "2026-11-23T18:17:00+05:30"))
    assert kite_routes._session_window()["session_expires_at"] == "2026-11-24T06:00:00+05:30"
    monkeypatch.setattr(kite, "load_session", lambda: None)
    assert kite_routes._session_window() == {}
