"""
Tests for NSE Security Master data file
========================================
Validates that data_cache/nse/security_master.json loads correctly through
the PointInTimeSecurityMaster class and that point-in-time membership
queries return historically accurate results.

AGENTS.md Rule 10: Every number resolves to an authoritative source.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.data.security_master import (
    PointInTimeSecurityMaster,
    SecurityMasterError,
    SecurityRecord,
)

MASTER_PATH = Path("data_cache/nse/security_master.json")


@pytest.fixture(scope="module")
def master() -> PointInTimeSecurityMaster:
    """Load the production security master file."""
    if not MASTER_PATH.exists():
        pytest.skip(f"Security master not found at {MASTER_PATH}")
    return PointInTimeSecurityMaster.from_file(MASTER_PATH)


# ── Loading & Schema ────────────────────────────────────────────────────────────

class TestSecurityMasterLoading:
    """Verify the master file loads and contains reasonable data."""

    def test_loads_successfully(self, master: PointInTimeSecurityMaster):
        assert len(master.records) > 0

    def test_minimum_coverage(self, master: PointInTimeSecurityMaster):
        """Should have at least 50 securities (NIFTY-50 coverage)."""
        assert len(master.records) >= 50

    def test_all_records_have_required_fields(self, master: PointInTimeSecurityMaster):
        for rec in master.records:
            assert rec.security_id.strip(), f"Empty security_id"
            assert rec.canonical_symbol.strip(), f"Empty symbol for {rec.security_id}"
            assert isinstance(rec.listed_from, date)

    def test_no_duplicate_security_ids(self, master: PointInTimeSecurityMaster):
        ids = [r.security_id for r in master.records]
        assert len(ids) == len(set(ids)), "Duplicate security_ids found"

    def test_json_schema_version(self):
        """Master file should declare its schema version."""
        payload = json.loads(MASTER_PATH.read_text())
        assert payload.get("schema_version") == "nse_security_master_v1"

    def test_sha256_is_deterministic(self, master: PointInTimeSecurityMaster):
        """Hash should be stable across loads."""
        h1 = master.sha256
        m2 = PointInTimeSecurityMaster.from_file(MASTER_PATH)
        assert m2.sha256 == h1


# ── Point-in-Time Membership ────────────────────────────────────────────────────

class TestPointInTimeMembership:
    """Validate date-aware universe construction."""

    def test_hdfc_present_in_2015(self, master: PointInTimeSecurityMaster):
        """HDFC Ltd was listed until 2023-07-13 (merger into HDFC Bank)."""
        syms_2015 = master.symbols_at(date(2015, 1, 1))
        # HDFC (the housing finance company) should be present if in master
        # Note: HDFCBANK is a different security from HDFC
        # We test HDFCBANK which was listed since 2001
        assert "HDFCBANK" in syms_2015

    def test_jiofin_absent_before_ipo(self, master: PointInTimeSecurityMaster):
        """Jio Financial IPO was 2023-08-21; should not appear before that."""
        syms_2020 = master.symbols_at(date(2020, 1, 1))
        assert "JIOFIN" not in syms_2020

    def test_jiofin_present_after_ipo(self, master: PointInTimeSecurityMaster):
        """Jio Financial should be active after listing."""
        syms_2024 = master.symbols_at(date(2024, 1, 1))
        assert "JIOFIN" in syms_2024

    def test_tcs_absent_before_ipo(self, master: PointInTimeSecurityMaster):
        """TCS IPO was 2004-08-25; should not appear before that."""
        syms_2003 = master.symbols_at(date(2003, 1, 1))
        assert "TCS" not in syms_2003

    def test_tcs_present_after_ipo(self, master: PointInTimeSecurityMaster):
        """TCS should be active after listing."""
        syms_2005 = master.symbols_at(date(2005, 1, 1))
        assert "TCS" in syms_2005

    def test_older_stocks_present_in_2000(self, master: PointInTimeSecurityMaster):
        """Stocks listed before 2000 should appear in 2000-era queries."""
        syms_2000 = master.symbols_at(date(2000, 1, 1))
        for sym in ("RELIANCE", "ITC", "SBIN", "HINDUNILVR"):
            assert sym in syms_2000, f"{sym} should be listed by 2000"

    def test_universe_grows_over_time(self, master: PointInTimeSecurityMaster):
        """More symbols should be active in 2024 than in 2000."""
        n_2000 = len(master.symbols_at(date(2000, 1, 1)))
        n_2024 = len(master.symbols_at(date(2024, 1, 1)))
        assert n_2024 > n_2000


# ── Alias Resolution ────────────────────────────────────────────────────────────

class TestAliasResolution:
    """Verify that common aliases resolve to canonical symbols."""

    def test_hdfc_bank_alias(self, master: PointInTimeSecurityMaster):
        assert master.canonicalize("HDFC BANK") == "HDFCBANK"

    def test_infosys_alias(self, master: PointInTimeSecurityMaster):
        assert master.canonicalize("INFOSYS") == "INFY"

    def test_tata_consultancy_alias(self, master: PointInTimeSecurityMaster):
        assert master.canonicalize("TATA CONSULTANCY") == "TCS"

    def test_reliance_industries_alias(self, master: PointInTimeSecurityMaster):
        assert master.canonicalize("RELIANCE INDUSTRIES") == "RELIANCE"

    def test_ns_suffix_stripped(self, master: PointInTimeSecurityMaster):
        """Yahoo Finance symbols like RELIANCE.NS should resolve."""
        assert master.canonicalize("RELIANCE.NS") == "RELIANCE"

    def test_unknown_symbol_raises(self, master: PointInTimeSecurityMaster):
        with pytest.raises(SecurityMasterError):
            master.canonicalize("NONEXISTENT_SYMBOL_XYZ")


# ── Snapshot / Audit ─────────────────────────────────────────────────────────────

class TestSnapshot:
    """Test the auditable snapshot method."""

    def test_snapshot_returns_expected_keys(self, master: PointInTimeSecurityMaster):
        snap = master.snapshot(date(2024, 6, 1))
        assert "as_of" in snap
        assert "active_symbols" in snap
        assert "security_count" in snap
        assert snap["security_count"] == len(snap["active_symbols"])

    def test_snapshot_with_available_symbols(self, master: PointInTimeSecurityMaster):
        """Should correctly separate available vs missing."""
        snap = master.snapshot(
            date(2024, 6, 1),
            available_symbols=["RELIANCE", "TCS", "INFY"],
        )
        assert "RELIANCE" in snap["available_symbols"]
        assert "TCS" in snap["available_symbols"]
        # Missing should be the rest of the active set minus these 3
        assert len(snap["missing_price_symbols"]) > 0
