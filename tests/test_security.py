"""
Test — Security Layer
======================
Unit tests for input sanitization and security utilities.

Covers:
  - Query sanitization (XSS, SQL injection detection)
  - Edge cases (empty, too long, legitimate financial queries)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.security import sanitize_query
from app.security.auth import get_current_user


class TestQuerySanitization:
    """Test input sanitization against injection attacks."""

    # ── Valid queries that must pass ────────────────────────────────────

    def test_normal_query(self):
        result = sanitize_query("Analyze HDFC Bank fundamentals")
        assert result == "Analyze HDFC Bank fundamentals"

    def test_query_with_special_chars(self):
        """Financial queries may contain &, %, parentheses — allow them."""
        result = sanitize_query("Compare P&G vs HUL (revenue growth 2023-2025)")
        assert "P&G" in result

    def test_query_with_numbers(self):
        result = sanitize_query("What is TCS stock performance in FY2025?")
        assert "FY2025" in result

    def test_query_strips_whitespace(self):
        result = sanitize_query("  HDFCBANK analysis  ")
        assert result == "HDFCBANK analysis"

    # ── XSS attacks that must be rejected ──────────────────────────────

    def test_xss_script_tag(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("Hello <script>alert('xss')</script>")

    def test_xss_javascript_uri(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("javascript:alert(1)")

    def test_xss_event_handler(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query('Test onerror=alert(1)')

    def test_xss_data_uri(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("data:text/html,<h1>test</h1>")

    # ── SQL injection attacks that must be rejected ────────────────────

    def test_sqli_drop_table(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("'; DROP TABLE users; --")

    def test_sqli_or_bypass(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("admin' OR 1=1 --")

    def test_sqli_union_select(self):
        with pytest.raises(ValueError, match="dangerous"):
            sanitize_query("x UNION SELECT * FROM secrets")

    # ── Length / size attacks ──────────────────────────────────────────

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            sanitize_query("ab")

    def test_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            sanitize_query("a" * 2001)

    def test_max_length_ok(self):
        """Exactly at the limit should pass."""
        result = sanitize_query("a" * 2000)
        assert len(result) == 2000

    # ── Edge cases ────────────────────────────────────────────────────

    def test_empty_string(self):
        with pytest.raises(ValueError):
            sanitize_query("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError):
            sanitize_query("   ")


class TestRouteAuthentication:
    """Route dependencies must match the middleware's dev/prod policy."""

    def test_development_without_credentials_is_usable(self, monkeypatch):
        monkeypatch.setenv("AUTH_REQUIRED", "false")
        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        assert get_current_user(None)["role"] == "development"

    def test_production_without_credentials_fails_closed(self, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setenv("AUTH_REQUIRED", "true")
        with pytest.raises(HTTPException) as exc:
            get_current_user(None)
        assert exc.value.status_code == 401


def test_csp_allows_the_configured_supabase_origin_only(monkeypatch):
    """The SPA signs in against Supabase from the browser; CSP must permit exactly that origin."""
    from app.security import _supabase_connect_sources

    monkeypatch.setenv("SUPABASE_URL", "https://abcdefghijklmnop.supabase.co")
    sources = _supabase_connect_sources()
    assert sources == "https://abcdefghijklmnop.supabase.co wss://abcdefghijklmnop.supabase.co"


def test_csp_rejects_non_https_supabase_url(monkeypatch, tmp_path):
    import app.security as sec

    monkeypatch.setenv("SUPABASE_URL", "http://evil.example.com")
    monkeypatch.delenv("VITE_SUPABASE_URL", raising=False)
    # Falls back to Supabase-hosted projects only — never an arbitrary host.
    assert sec._supabase_connect_sources() == "https://*.supabase.co wss://*.supabase.co"


def test_served_csp_header_includes_supabase_in_connect_src(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("SUPABASE_URL", "https://abcdefghijklmnop.supabase.co")
    csp = TestClient(app).get("/api/health").headers["content-security-policy"]
    connect = next(d for d in csp.split(";") if d.strip().startswith("connect-src"))
    assert "https://abcdefghijklmnop.supabase.co" in connect and "'self'" in connect
    assert " * " not in f" {connect} "


@pytest.mark.parametrize("env, expected", [("development", (600, 40)), ("production", (60, 10))])
def test_rate_limit_defaults_depend_on_environment(monkeypatch, env, expected):
    from app.security import RateLimiterMiddleware

    monkeypatch.setenv("APP_ENV", env)
    monkeypatch.delenv("RATE_LIMIT_RPM", raising=False)
    monkeypatch.delenv("RATE_LIMIT_BURST", raising=False)
    mw = RateLimiterMiddleware(app=lambda *a: None)
    assert (mw.rpm, mw.burst) == expected


def test_rate_limit_env_override_wins(monkeypatch):
    from app.security import RateLimiterMiddleware

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("RATE_LIMIT_RPM", "30")
    assert RateLimiterMiddleware(app=lambda *a: None).rpm == 30
