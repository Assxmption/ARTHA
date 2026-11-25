"""
Shared test configuration.

Offline tests must be deterministic and must never reach a broker: Kite is
disabled for every test unless the live suite is explicitly enabled
(ARTHA_LIVE=1, see tests/live/conftest.py).
"""

from __future__ import annotations

import os

import pytest

_LIVE = os.getenv("ARTHA_LIVE", "").lower() in ("1", "true", "yes")
# The paper-trading scheduler must never start (and write the real ledger) under tests.
os.environ["PAPER_TRADING_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def _no_kite_in_offline_tests(request, monkeypatch, tmp_path):
    if _LIVE and "live" in request.keywords:
        return
    for var in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    try:
        import app.data.kite as kite
    except ImportError:
        return
    monkeypatch.setattr(kite, "SESSION_FILE", tmp_path / "kite_session.json")
    monkeypatch.setattr(kite, "_client", None)
