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
