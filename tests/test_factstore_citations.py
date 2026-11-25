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


class TestNumericClaimsMustMatchCitedFacts:
    """
    A resolving citation is not enough: every number on a cited line must be
    a value held by a fact cited on that line (AGENTS.md rules 1 and 10).
    """

    def test_invented_number_next_to_real_citation_fails(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        report = "Revenue jumped 9999% to ₹42 lakh crore [fact_id: aaa11111]."
        result = check_citations(report, job_id, store)
        assert result["valid"] is False
        assert result["resolved"] == 1
        assert result["unsupported_numeric_claims"][0]["numbers"] == ["9999", "42"]

    def test_llm_computed_growth_rate_fails(self, store_with_fixture_data):
        """250000 vs 230000 is +8.7%, but no fact holds 8.7 → rejected (rule 1)."""
        store, job_id = store_with_fixture_data
        report = (
            "Revenue grew 8.7% to ₹2,50,000 crore "
            "[fact_id: aaa11111] [fact_id: cccc9999]."
        )
        result = check_citations(report, job_id, store)
        assert result["valid"] is False
        assert result["unsupported_numeric_claims"][0]["numbers"] == ["8.7"]

    def test_number_only_matches_facts_cited_on_same_line(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        # 32.5 is a real value, but of eee55555 — not of the cited aaa11111.
        report = "Operating margin was 32.5% [fact_id: aaa11111]."
        result = check_citations(report, job_id, store)
        assert result["valid"] is False

    def test_rounded_value_matches_to_displayed_precision(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        assert check_citations("Margin of 33% [fact_id: eee55555].", job_id, store)["valid"]
        assert check_citations("Margin of 32.50% [fact_id: eee55555].", job_id, store)["valid"]
        assert not check_citations("Margin of 34% [fact_id: eee55555].", job_id, store)["valid"]

    def test_western_and_indian_digit_grouping(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        for figure in ("2,50,000", "250,000", "250000.00"):
            report = f"Revenue ₹{figure} crore [fact_id: aaa11111]."
            assert check_citations(report, job_id, store)["valid"], figure

    def test_fiscal_year_of_cited_fact_is_allowed(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        report = "In 2025 revenue was ₹2,50,000 crore [fact_id: aaa11111]."
        assert check_citations(report, job_id, store)["valid"] is True

    def test_list_markers_and_citation_source_text_are_not_claims(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        report = (
            "### 2. Revenue\n"
            "1. Revenue ₹2,50,000 crore [fact_id: aaa11111…, source: NSE_FILING_2019_PG7]"
        )
        result = check_citations(report, job_id, store)
        assert result["valid"] is True, result

    def test_narrative_fact_cannot_launder_numbers(self, store_with_fixture_data):
        """Citing an LLM-written NarrativeFact must not support its numbers."""
        store, job_id = store_with_fixture_data
        store.put_fact(NarrativeFact(
            fact_id="ddd00000aaaa1111bbbb2222cccc3333",
            job_id=job_id,
            source="FUNDAMENTALS_AGENT_NARRATION",
            agent_name="fundamentals_agent",
            section="fundamental_analysis",
            content="Revenue grew 8.7% year over year.",
        ))
        report = "Revenue grew 8.7% [fact_id: ddd00000]."
        result = check_citations(report, job_id, store)
        assert result["valid"] is False
        assert result["unresolved"] == []  # the id resolves; the number doesn't

    def test_numbers_inside_cited_risk_detail_are_supported(self, store_with_fixture_data):
        store, job_id = store_with_fixture_data
        store.put_fact(RiskFlag(
            fact_id="fff00000aaaa1111bbbb2222cccc3333",
            job_id=job_id,
            source="risk_agent.leverage",
            symbol_or_pair="HDFCBANK",
            flag_type="leverage",
            detail="Debt/equity 2.3 exceeds the 2.0 threshold",
            severity=Severity.WARNING,
        ))
        ok = "Leverage: D/E of 2.3 is above 2.0 [fact_id: fff00000]."
        bad = "Leverage: D/E of 2.9 is above 2.0 [fact_id: fff00000]."
        assert check_citations(ok, job_id, store)["valid"] is True
        assert check_citations(bad, job_id, store)["valid"] is False


class TestCitationSpellings:
    """LLMs copy the context table's "Fact ID" header; the gate must still parse it."""

    @pytest.mark.parametrize(
        "citation",
        ["[fact_id: aaa11111]", "[Fact ID aaa11111…]", "(fact id: AAA11111)", "[fact-id #aaa11111]"],
    )
    def test_spelling_variants_resolve_and_numbers_are_still_checked(
        self, store_with_fixture_data, citation
    ):
        store, job_id = store_with_fixture_data
        good = f"Revenue ₹2,50,000 crore in FY 2025 {citation}."
        bad = f"Revenue ₹3,10,000 crore in FY 2025 {citation}."
        assert check_citations(good, job_id, store)["valid"] is True
        result = check_citations(bad, job_id, store)
        assert result["valid"] is False
        assert result["unsupported_numeric_claims"][0]["numbers"] == ["3,10,000"]


def test_lookback_window_labels_are_not_claims(store_with_fixture_data):
    store, job_id = store_with_fixture_data
    ok = "The 52-week high context: revenue ₹2,50,000 crore [fact_id: aaa11111]."
    assert check_citations(ok, job_id, store)["valid"] is True
    # A bare number is still a claim even near such a phrase.
    bad = "Revenue rose 52 percent over the 52-week window [fact_id: aaa11111]."
    assert check_citations(bad, job_id, store)["valid"] is False


def test_typographic_hyphens_and_spaces_are_normalised(store_with_fixture_data):
    """Real gpt-oss output: U+2011 hyphen, U+202F narrow no-break spaces."""
    store, job_id = store_with_fixture_data
    line = "52‑week context – revenue ₹2,50,000 crore [Fact ID aaa11111…]"
    assert check_citations(line, job_id, store)["valid"] is True
    wrong = "revenue ₹4,077.00 crore [Fact ID aaa11111…]"
    assert check_citations(wrong, job_id, store)["valid"] is False


@pytest.mark.parametrize(
    "citation",
    ["[Fact ID: aaa11111…; eee55555…]", "[fact_ids: aaa11111, eee55555]", "(Fact IDs aaa11111 and eee55555)"],
)
def test_multiple_ids_in_one_citation(store_with_fixture_data, citation):
    """Real gpt-oss output cites several facts inside one bracket."""
    store, job_id = store_with_fixture_data
    line = f"Revenue ₹2,50,000 crore and margin 32.5% {citation}."
    result = check_citations(line, job_id, store)
    assert result["valid"] is True, result
    assert result["resolved"] == 2


@pytest.mark.parametrize("label", ["FY 2023-2026", "FY2023–FY2026", "FY24/25", "FY 2025"])
def test_fiscal_year_labels_are_not_claims(store_with_fixture_data, label):
    store, job_id = store_with_fixture_data
    assert check_citations(f"Margins stayed stable across {label}.", job_id, store)["uncited_numeric_claims"] == []
    # A bare year in prose is still a claim that needs a citation.
    assert check_citations("Margins peaked in 2019.", job_id, store)["valid"] is False
