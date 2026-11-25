"""Regression tests for the pre-deployment security review (2026-11-25)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DIST = Path("frontend/dist/index.html")


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


@pytest.mark.skipif(not DIST.exists(), reason="frontend build not present")
@pytest.mark.parametrize("path", [
    "/..%2F..%2Fpyproject.toml",
    "/%2e%2e/%2e%2e/pyproject.toml",
    "/..%2f..%2f.env.example",
    "/assets/..%2F..%2F..%2Fpyproject.toml",
    "/%2e%2e%2f%2e%2e%2fdata_cache%2fkite%2fsession.json",
])
def test_spa_catch_all_never_serves_files_outside_dist(client, path):
    r = client.get(path)
    index = DIST.read_text()
    # Either the SPA shell or a 404 — never the file the path points at.
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.text == index
    assert "[project]" not in r.text and "access_token" not in r.text


@pytest.mark.skipif(not DIST.exists(), reason="frontend build not present")
def test_spa_serves_real_dist_files_and_index(client):
    assert client.get("/paper-trading").text == DIST.read_text()
    assert client.get("/api/does-not-exist").status_code == 404


# ── Authentication and operator checks ────────────────────────────────────────

import time as _time

import jwt as _jwt

SECRET = "test-secret-for-hs256-at-least-32-bytes-long!!"


def _token(**claims):
    base = {"sub": "user-1", "aud": "authenticated", "role": "authenticated", "exp": int(_time.time()) + 600}
    base.update(claims)
    return _jwt.encode({k: v for k, v in base.items() if v is not None}, SECRET, algorithm="HS256")


@pytest.fixture
def prod_auth(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    monkeypatch.setenv("ADMIN_EMAILS", "owner@example.com")
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)


def test_supabase_anon_key_is_not_a_login(prod_auth):
    from fastapi import HTTPException

    from app.security.auth import verify_token

    anon = _jwt.encode({"role": "anon", "iss": "supabase", "exp": int(_time.time()) + 600}, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException):
        verify_token(anon)
    with pytest.raises(HTTPException):          # right audience but service/anon role
        verify_token(_token(role="anon"))
    with pytest.raises(HTTPException):          # expired
        verify_token(_token(exp=int(_time.time()) - 10))
    with pytest.raises(HTTPException) as e:     # forged with another secret
        verify_token(_jwt.encode({"sub": "x", "aud": "authenticated", "exp": int(_time.time()) + 60}, "x" * 40, algorithm="HS256"))
    assert "Signature" not in str(e.value.detail)   # decoder messages are not echoed
    assert verify_token(_token())["sub"] == "user-1"


def test_operator_only_routes(prod_auth, client):
    owner = {"Authorization": f"Bearer {_token(email='owner@example.com')}"}
    member = {"Authorization": f"Bearer {_token(email='someone@example.com')}"}
    for method, path in [("post", "/api/kite/logout"), ("post", "/api/kite/login-url"),
                         ("post", "/api/paper/run"),
                         ("post", "/api/screener/build"), ("post", "/api/screener/refresh"),
                         ("post", "/api/quant/live/refresh")]:
        assert getattr(client, method)(path).status_code == 401, path
        assert getattr(client, method)(path, headers=member).status_code == 403, path
    # The operator gets past the gate (the handler itself may then fail for other reasons).
    assert client.post("/api/kite/login-url", headers=owner).status_code not in (401, 403)


def test_kite_status_hides_account_from_visitors(prod_auth, client, monkeypatch):
    import app.api.kite_routes as kr

    body = {"configured": True, "connected": True, "user_id": "AB1234", "user_name": "Owner",
            "exchanges": ["NSE"], "session_expires_at": "x"}

    async def fake():
        return dict(body)

    monkeypatch.setattr(kr, "_kite_status_body", fake)
    public = client.get("/api/kite/status").json()
    assert public == {"configured": True, "connected": True}
    owner = {"Authorization": f"Bearer {_token(email='owner@example.com')}"}
    assert client.get("/api/kite/status", headers=owner).json()["user_id"] == "AB1234"


def test_production_config_check(monkeypatch):
    from app.security.auth import check_production_config

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    assert len(check_production_config()) == 2
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    assert check_production_config() == []


def test_dev_identity_is_never_admin_in_production(monkeypatch):
    from app.security.auth import is_admin

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    assert not is_admin({"sub": "anonymous", "role": "development"})
    monkeypatch.setenv("APP_ENV", "development")
    assert is_admin({"sub": "anonymous", "role": "development"})


# ── Client IP and rate limiting ───────────────────────────────────────────────

class _Req:
    def __init__(self, peer, xff=None):
        self.client = type("C", (), {"host": peer})()
        self.headers = {"x-forwarded-for": xff} if xff else {}


def test_client_ip_ignores_spoofed_forwarded_for(monkeypatch):
    from app.security.net import client_ip

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "0")
    assert client_ip(_Req("10.0.0.5", "1.2.3.4")) == "10.0.0.5"
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    # Attacker sends "1.2.3.4"; our proxy appends the real client 203.0.113.9.
    assert client_ip(_Req("10.0.0.5", "1.2.3.4, 203.0.113.9")) == "203.0.113.9"
    assert client_ip(_Req("10.0.0.5", "not-an-ip")) == "10.0.0.5"
    assert client_ip(_Req("10.0.0.5")) == "10.0.0.5"


def test_rate_limiter_memory_is_bounded():
    import asyncio

    from app.security import RateLimiterMiddleware

    rl = RateLimiterMiddleware(app=None, rpm=5, burst=5)
    rl.MAX_TRACKED_CLIENTS = 100
    now = _time.time()
    for i in range(500):
        rl._windows[f"ip{i}"] = __import__("collections").deque([now])
    rl._sweep(now)
    assert len(rl._windows) <= 100
    rl._windows = {"old": __import__("collections").deque([now - 120])}
    rl._sweep(now)
    assert rl._windows == {}
