"""
Test — Fact Store Citation Traceability Gate
=============================================
HARD GATE on report generation (AGENTS.md rule 10).

This test runs the Writer's citation checker against a fixture Fact Store
and confirms that every numeric claim in the output resolves to a Fact
Store entry with a non-empty `source` field.

A report that fails this test does not ship.

Reference: AGENTS.md rule 10, docs/ARTHA_ARCHITECTURE.md §4.8
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

from app.agents.writer_agent import check_citations
from app.factstore.schemas import (
    FactType,
    FundamentalRow,
    NarrativeFact,
    QuantSignal,
    RiskFlag,
    Severity,
    SignalType,
)
from app.factstore.store import FactStore


@pytest.fixture
def store_with_fixture_data(tmp_path):
    """
    Create a Fact Store populated with fixture data that simulates
    a completed analysis pipeline.
    """
    db_path = str(tmp_path / "citation_test.db")
    store = FactStore(db_path=db_path)
    job_id = "citation-test-job"

    # Write known facts with known fact_ids
    facts = [
        FundamentalRow(
            fact_id="aaa11111bbbb2222cccc3333dddd4444",
            job_id=job_id,
            symbol="HDFCBANK",
            fiscal_year=2025,
            metric="revenue",
            value=250000.0,
            unit="INR_crore",
            source="NSE_ANNUAL_REPORT_2025",
        ),
        FundamentalRow(
            fact_id="eee55555ffff6666aaaa7777bbbb8888",
            job_id=job_id,
            symbol="HDFCBANK",
            fiscal_year=2025,
            metric="operating_margin",
            value=32.5,
            unit="percent",
            source="NSE_ANNUAL_REPORT_2025",
        ),
        FundamentalRow(
            fact_id="cccc9999ddddaaaaeeeebbbbiiii0000",
            job_id=job_id,
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="revenue",
            value=230000.0,
            unit="INR_crore",
            source="NSE_ANNUAL_REPORT_2024",
        ),
    ]

    store.put_facts(facts)
    return store, job_id


class TestCitationResolution:
    """
    Test the post-generation citation check.
    """

    def test_all_citations_resolve(self, store_with_fixture_data):
        """A report where all fact_id references exist → valid."""
        store, job_id = store_with_fixture_data

        report = (
            "HDFC Bank reported revenue of ₹2,50,000 crore "
            "[fact_id: aaa11111bbbb2222cccc3333dddd4444] in FY2025, "
            "with an operating margin of 32.5% "
            "[fact_id: eee55555ffff6666aaaa7777bbbb8888]. "
            "This represents growth from the FY2024 revenue of ₹2,30,000 crore "
            "[fact_id: cccc9999ddddaaaaeeeebbbbiiii0000]."
        )

        result = check_citations(report, job_id, store)
        assert result["valid"] is True
        assert result["total_citations"] == 3
        assert result["resolved"] == 3
        assert result["unresolved"] == []

    def test_truncated_ids_resolve(self, store_with_fixture_data):
        """Truncated 8-char fact_id prefixes should still resolve."""
        store, job_id = store_with_fixture_data

        report = (
            "Revenue was ₹2,50,000 crore [fact_id: aaa11111] "
            "with margin at 32.5% [fact_id: eee55555]."
        )

        result = check_citations(report, job_id, store)
        assert result["valid"] is True
        assert result["resolved"] == 2

    def test_unresolved_citation_fails(self, store_with_fixture_data):
        """A report citing a nonexistent fact_id → invalid."""
        store, job_id = store_with_fixture_data

        report = (
            "Revenue was ₹2,50,000 crore [fact_id: aaa11111] "
            "and the P/E ratio was 22.5 [fact_id: ffffffff99999999]."
        )

        result = check_citations(report, job_id, store)
        assert result["valid"] is False
        assert result["resolved"] == 1
        assert len(result["unresolved"]) == 1

    def test_no_citations_is_invalid(self, store_with_fixture_data):
        """A report with no fact_id references at all → invalid."""
        store, job_id = store_with_fixture_data

        report = (
            "HDFC Bank is a great company with strong fundamentals. "
            "Revenue grew impressively this year."
        )

        result = check_citations(report, job_id, store)
        assert result["valid"] is False
        assert result["total_citations"] == 0

    def test_duplicate_citations_counted_once(self, store_with_fixture_data):
        """Duplicate references to the same fact_id are de-duplicated."""
        store, job_id = store_with_fixture_data

        report = (
            "Revenue [fact_id: aaa11111] was strong. "
            "As noted, revenue [fact_id: aaa11111] grew year-over-year."
        )

        result = check_citations(report, job_id, store)
        assert result["valid"] is True
        assert result["total_citations"] == 1  # de-duplicated
        assert result["resolved"] == 1


class TestCitationEdgeCases:
    """Edge cases for the citation checker."""

    def test_empty_report(self, store_with_fixture_data):
        """An empty report has no citations → invalid."""
        store, job_id = store_with_fixture_data

        result = check_citations("", job_id, store)
        assert result["valid"] is False
        assert result["total_citations"] == 0

    def test_empty_store(self, tmp_path):
        """Citations against an empty store → all unresolved."""
        store = FactStore(db_path=str(tmp_path / "empty.db"))

        report = "Some claim [fact_id: abcdef1234567890]."
        result = check_citations(report, "empty-job", store)
        assert result["valid"] is False

    def test_source_must_be_nonempty(self, tmp_path):
        """
        A fact with an empty source should fail the source check.

        Note: The Pydantic schema enforces min_length=1 on source,
        so this test verifies the schema-level guard rather than
        the citation checker.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FundamentalRow(
                job_id="test",
                symbol="TEST",
                fiscal_year=2025,
                metric="revenue",
                value=100.0,
                source="",  # This should fail validation
            )
