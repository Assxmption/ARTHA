"""
Tests for Fundamentals RAG Narrator
=====================================
Tests that the narrate_fundamentals() function:
  - Cites only fact_ids that exist in the fixture Fact Store
  - Produces NarrativeFact with non-empty referenced_fact_ids
  - Handles empty fact lists gracefully
  - Validates citations and flags invalid ones
  - format_facts_for_narration handles edge cases

LLM is mocked for deterministic testing.
"""

from __future__ import annotations

import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.agents.fundamentals_agent import (
    format_facts_for_narration,
    narrate_fundamentals,
    _categorize_facts,
    _format_categorized_facts,
    _extract_fact_ids,
    _validate_citations,
)
from app.factstore.schemas import (
    FundamentalRow,
    FactType,
    NarrativeFact,
    RiskFlag,
    Severity,
)
from app.factstore.store import FactStore


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_store():
    """Create a temporary FactStore backed by SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = FactStore(db_path=f.name)
        yield store
        store.close()


@pytest.fixture
def sample_facts() -> list[FundamentalRow]:
    """Create a set of sample FundamentalRow facts with known fact_ids."""
    job_id = "test_job_001"
    facts = [
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_INCOME_STMT",
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="revenue",
            value=245678.50,
            unit="INR_crore",
        ),
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_INCOME_STMT",
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="net_income",
            value=56789.00,
            unit="INR_crore",
        ),
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_BALANCE_SHEET",
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="total_debt",
            value=123456.00,
            unit="INR_crore",
        ),
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_INCOME_STMT",
            symbol="HDFCBANK",
            fiscal_year=2023,
            metric="revenue",
            value=198765.00,
            unit="INR_crore",
        ),
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_INCOME_STMT",
            symbol="HDFCBANK",
            fiscal_year=2023,
            metric="net_income",
            value=45678.00,
            unit="INR_crore",
        ),
        FundamentalRow(
            job_id=job_id,
            source="OHLCV_DERIVED",
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="annual_return",
            value=15.5,
            unit="percent",
        ),
        FundamentalRow(
            job_id=job_id,
            source="YFINANCE_INCOME_STMT",
            symbol="HDFCBANK",
            fiscal_year=2024,
            metric="eps",
            value=85.20,
            unit="INR",
        ),
    ]
    return facts


@pytest.fixture
def store_with_facts(temp_store, sample_facts):
    """FactStore pre-loaded with sample facts."""
    temp_store.put_facts(sample_facts)
    return temp_store, sample_facts


# ── format_facts_for_narration tests ────────────────────────────────────────────


class TestFormatFacts:
    def test_empty_list(self):
        """Should return a meaningful message for empty facts."""
        result = format_facts_for_narration([])
        assert "No fundamental data available" in result

    def test_contains_fact_ids(self, sample_facts):
        """Formatted output should contain fact_id prefixes."""
        result = format_facts_for_narration(sample_facts)
        for fact in sample_facts:
            assert fact.fact_id[:8] in result

    def test_contains_critical_instruction(self, sample_facts):
        """Output should contain the citation instruction."""
        result = format_facts_for_narration(sample_facts)
        assert "CRITICAL" in result
        assert "Fact ID" in result

    def test_contains_all_metrics(self, sample_facts):
        """All metric names should appear in the output."""
        result = format_facts_for_narration(sample_facts)
        for fact in sample_facts:
            assert fact.metric in result

    def test_none_value_displays_na(self):
        """Facts with None value should display as N/A."""
        fact = FundamentalRow(
            job_id="test",
            source="TEST",
            symbol="TEST",
            fiscal_year=2024,
            metric="missing_metric",
            value=None,
        )
        result = format_facts_for_narration([fact])
        assert "N/A" in result


# ── Categorization tests ───────────────────────────────────────────────────────


class TestCategorization:
    def test_revenue_in_profitability(self, sample_facts):
        """Revenue should be categorized under Profitability."""
        cats = _categorize_facts(sample_facts)
        profitability_metrics = [f.metric for f in cats.get("Profitability", [])]
        assert "revenue" in profitability_metrics

    def test_debt_in_leverage(self, sample_facts):
        """Total debt should be categorized under Leverage & Liquidity."""
        cats = _categorize_facts(sample_facts)
        leverage_metrics = [f.metric for f in cats.get("Leverage & Liquidity", [])]
        assert "total_debt" in leverage_metrics

    def test_unknown_metric_in_other(self):
        """Unknown metrics should go to 'Other'."""
        fact = FundamentalRow(
            job_id="test", source="TEST", symbol="TEST",
            fiscal_year=2024, metric="custom_unusual_metric", value=42.0,
        )
        cats = _categorize_facts([fact])
        assert "Other" in cats
        assert len(cats["Other"]) == 1

    def test_formatted_output_has_sections(self, sample_facts):
        """Formatted categorized output should have section headers."""
        cats = _categorize_facts(sample_facts)
        result = _format_categorized_facts(cats)
        assert "### Profitability" in result


# ── Citation extraction tests ──────────────────────────────────────────────────


class TestCitationExtraction:
    def test_extract_8char_prefix(self):
        """Should extract 8-char hex prefixes followed by …"""
        text = "Revenue was ₹245,678 crore [abcd1234…] and net income was ₹56,789 [ef567890…]."
        ids = _extract_fact_ids(text)
        assert "abcd1234" in ids
        assert "ef567890" in ids

    def test_extract_full_32char_id(self):
        """Should extract full 32-char hex IDs."""
        full_id = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
        text = f"The metric [fact:{full_id}] shows growth."
        ids = _extract_fact_ids(text)
        assert full_id in ids

    def test_extract_with_dots(self):
        """Should extract IDs with ... instead of …"""
        text = "Revenue [abcd1234...] grew strongly."
        ids = _extract_fact_ids(text)
        assert "abcd1234" in ids

    def test_no_false_positives(self):
        """Should not extract random hex-like substrings."""
        text = "The company grew 20% year-over-year."
        ids = _extract_fact_ids(text)
        assert len(ids) == 0

    def test_empty_text(self):
        """Should return empty list for empty text."""
        assert _extract_fact_ids("") == []


# ── Citation validation tests ──────────────────────────────────────────────────


class TestCitationValidation:
    def test_valid_citations(self, store_with_facts):
        """Citations matching known facts should be valid."""
        store, facts = store_with_facts
        # Build a narrative that cites known fact IDs
        narrative = (
            f"Revenue was strong [{facts[0].fact_id[:8]}…] "
            f"and net income [{facts[1].fact_id[:8]}…] grew."
        )
        valid, invalid = _validate_citations(
            narrative, facts, facts[0].job_id, store,
        )
        assert len(valid) >= 2
        assert len(invalid) == 0

    def test_invalid_citations(self, store_with_facts):
        """Citations not matching any fact should be invalid."""
        store, facts = store_with_facts
        narrative = "Revenue was ₹100 [deadbeef…] which is made up."
        valid, invalid = _validate_citations(
            narrative, facts, facts[0].job_id, store,
        )
        assert "deadbeef" in invalid

    def test_mixed_citations(self, store_with_facts):
        """Mix of valid and invalid citations should be separated."""
        store, facts = store_with_facts
        narrative = (
            f"Real data [{facts[0].fact_id[:8]}…] "
            f"and fake data [00000000…]."
        )
        valid, invalid = _validate_citations(
            narrative, facts, facts[0].job_id, store,
        )
        assert len(valid) >= 1
        assert len(invalid) >= 1

    def test_no_citations(self, store_with_facts):
        """Narrative with no citations should return empty lists."""
        store, facts = store_with_facts
        valid, invalid = _validate_citations(
            "No numbers cited here.", facts, facts[0].job_id, store,
        )
        assert valid == []
        assert invalid == []


# ── narrate_fundamentals tests ─────────────────────────────────────────────────


class TestNarrateFundamentals:
    def test_no_facts_returns_none(self, temp_store):
        """Should return None when no facts are available."""
        result = narrate_fundamentals(
            symbol="NONEXISTENT",
            job_id="test_job_999",
            store=temp_store,
        )
        assert result is None

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_narration_produces_narrative_fact(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """Should produce a NarrativeFact with valid citations."""
        store, facts = store_with_facts

        # Mock LLM to return a narrative that cites known fact_ids
        mock_llm = MagicMock()
        narrative = (
            f"**Financial Health**: Revenue reached ₹245,678 crore "
            f"[{facts[0].fact_id[:8]}…] in FY2024, up from ₹198,765 crore "
            f"[{facts[3].fact_id[:8]}…] in FY2023. Net income grew to "
            f"₹56,789 crore [{facts[1].fact_id[:8]}…].\n\n"
            f"**Balance Sheet**: Total debt stood at ₹123,456 crore "
            f"[{facts[2].fact_id[:8]}…]."
        )
        mock_llm.call.return_value = narrative
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        assert result is not None
        assert isinstance(result, NarrativeFact)
        assert result.agent_name == "fundamentals_agent"
        assert result.section == "fundamentals_analysis"
        assert len(result.referenced_fact_ids) > 0
        assert result.content == narrative

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_narration_validates_citations(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """All referenced_fact_ids should resolve to actual facts."""
        store, facts = store_with_facts

        # Mock LLM with valid citations
        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            f"Revenue [{facts[0].fact_id[:8]}…] was strong."
        )
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        # Every referenced_fact_id should be found in the store
        for fact_id in result.referenced_fact_ids:
            retrieved = store.get_fact(fact_id)
            assert retrieved is not None, f"fact_id {fact_id} not found in store"
            assert retrieved.source, f"fact_id {fact_id} has empty source"

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_invalid_citations_create_risk_flag(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """Invalid fact_id citations should produce a RiskFlag."""
        store, facts = store_with_facts

        # Mock LLM with an invalid citation
        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            f"Revenue [{facts[0].fact_id[:8]}…] was strong, "
            f"and margins were 25% [deadbeef…]."
        )
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        # Check that a RiskFlag was written
        all_facts_in_store = store.get_facts_by_type(
            facts[0].job_id, FactType.RISK_FLAG,
        )
        risk_flags = [
            f for f in all_facts_in_store
            if isinstance(f, RiskFlag)
            and f.flag_type == "uncitable_narrative_claim"
        ]
        assert len(risk_flags) >= 1
        assert risk_flags[0].severity == Severity.WARNING

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_llm_failure_falls_back(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """LLM failure should fall back to structured summary."""
        store, facts = store_with_facts

        # Mock LLM to raise an exception
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("Rate limit exceeded")
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        # Should still produce a result (fallback)
        assert result is not None
        assert "Fundamental Summary" in result.content

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_narrative_written_to_store(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """NarrativeFact should be persisted in the store."""
        store, facts = store_with_facts

        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            f"Revenue [{facts[0].fact_id[:8]}…] grew."
        )
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        # Verify it's in the store
        retrieved = store.get_fact(result.fact_id)
        assert retrieved is not None
        assert isinstance(retrieved, NarrativeFact)
        assert retrieved.agent_name == "fundamentals_agent"

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_no_invented_numbers(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """
        The narration pipeline should not let invented numbers through.

        If the LLM invents a number with a fake fact_id, the citation
        validation should catch it and flag it.
        """
        store, facts = store_with_facts

        # LLM invents a number with a fake fact_id
        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            "The company had revenue of ₹999,999 crore [aaaabbbb…], "
            "which is completely fabricated."
        )
        mock_get_llm.return_value = mock_llm

        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=facts,
        )

        # The narrative should exist but the citation should be invalid
        assert result is not None
        # Should have created a RiskFlag
        risk_flags = store.get_facts_by_type(
            facts[0].job_id, FactType.RISK_FLAG,
        )
        assert len(risk_flags) >= 1


class TestNarrateFromStore:
    """Test narration when facts are retrieved from store (not pre-loaded)."""

    @patch("app.agents.fundamentals_agent.get_cheap_llm")
    def test_retrieves_facts_from_store(
        self, mock_get_llm, store_with_facts, sample_facts,
    ):
        """Should retrieve facts from store when not provided directly."""
        store, facts = store_with_facts

        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            f"Data for HDFCBANK [{facts[0].fact_id[:8]}…]."
        )
        mock_get_llm.return_value = mock_llm

        # Don't pass facts — let it retrieve from store
        result = narrate_fundamentals(
            symbol="HDFCBANK",
            job_id=facts[0].job_id,
            store=store,
            facts=None,  # Explicitly None
        )

        assert result is not None
        assert isinstance(result, NarrativeFact)
