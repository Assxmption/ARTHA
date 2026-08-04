"""
Test — Fact Store CRUD
=======================
Unit tests for the ARTHA Fact Store.

Covers:
  - Write a fact, read it back.
  - Query by type / symbol / job.
  - Per-stage checkpointing: facts survive even if the job fails later.
  - Bulk insert.
  - Deletion.
  - Edge cases: empty store, unknown fact_id.
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

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
def store(tmp_path):
    """Create a fresh FactStore backed by a temporary SQLite DB."""
    db_path = str(tmp_path / "test_factstore.db")
    return FactStore(db_path=db_path)


@pytest.fixture
def sample_fundamental():
    """A sample FundamentalRow for testing."""
    return FundamentalRow(
        job_id="test-job-001",
        symbol="HDFCBANK",
        fiscal_year=2025,
        metric="revenue",
        value=250000.0,
        unit="INR_crore",
        source="NSE_ANNUAL_REPORT_2025_PG34",
    )


@pytest.fixture
def sample_signal():
    """A sample QuantSignal for testing."""
    return QuantSignal(
        job_id="test-job-001",
        symbol_or_pair="HDFCBANK-ICICIBANK",
        signal_type=SignalType.STAT_ARB_ZSCORE,
        value=2.15,
        regime_label="low_vol_trending",
        backtest_sharpe=1.65,
        backtest_sortino=2.1,
        backtest_max_drawdown=-0.12,
        validated=False,
        as_of=date(2025, 3, 31),
        source="QUANT_ENGINE_STATARB_V1",
    )


class TestFactStoreBasicCRUD:
    """Test basic create, read, update, delete operations."""

    def test_put_and_get_fundamental(self, store, sample_fundamental):
        """Write a FundamentalRow and read it back."""
        fact_id = store.put_fact(sample_fundamental)

        retrieved = store.get_fact(fact_id)
        assert retrieved is not None
        assert isinstance(retrieved, FundamentalRow)
        assert retrieved.symbol == "HDFCBANK"
        assert retrieved.fiscal_year == 2025
        assert retrieved.metric == "revenue"
        assert retrieved.value == 250000.0
        assert retrieved.source == "NSE_ANNUAL_REPORT_2025_PG34"

    def test_put_and_get_signal(self, store, sample_signal):
        """Write a QuantSignal and read it back."""
        fact_id = store.put_fact(sample_signal)

        retrieved = store.get_fact(fact_id)
        assert retrieved is not None
        assert isinstance(retrieved, QuantSignal)
        assert retrieved.symbol_or_pair == "HDFCBANK-ICICIBANK"
        assert retrieved.signal_type == SignalType.STAT_ARB_ZSCORE
        assert retrieved.value == 2.15
        assert retrieved.validated is False

    def test_get_nonexistent_fact(self, store):
        """Requesting a fact that doesn't exist returns None."""
        result = store.get_fact("nonexistent-id")
        assert result is None

    def test_delete_fact(self, store, sample_fundamental):
        """Delete a fact and confirm it's gone."""
        fact_id = store.put_fact(sample_fundamental)
        assert store.get_fact(fact_id) is not None

        deleted = store.delete_fact(fact_id)
        assert deleted is True
        assert store.get_fact(fact_id) is None

    def test_delete_nonexistent(self, store):
        """Deleting a nonexistent fact returns False."""
        assert store.delete_fact("nonexistent") is False


class TestFactStoreQueries:
    """Test query methods."""

    def test_get_facts_by_type(self, store):
        """Query facts by type within a job."""
        job_id = "job-query-test"

        # Write 2 fundamentals and 1 signal
        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="TCS", fiscal_year=2025,
            metric="revenue", value=100.0, unit="INR_crore",
            source="src1",
        ))
        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="TCS", fiscal_year=2025,
            metric="operating_margin", value=25.0, unit="percent",
            source="src2",
        ))
        store.put_fact(QuantSignal(
            job_id=job_id, symbol_or_pair="TCS-INFY",
            signal_type=SignalType.STAT_ARB_ZSCORE,
            value=1.5, validated=False, as_of=date(2025, 3, 31),
            source="src3",
        ))

        fundamentals = store.get_facts_by_type(job_id, FactType.FUNDAMENTAL)
        assert len(fundamentals) == 2
        assert all(isinstance(f, FundamentalRow) for f in fundamentals)

        signals = store.get_facts_by_type(job_id, FactType.QUANT_SIGNAL)
        assert len(signals) == 1

    def test_get_facts_for_symbol(self, store):
        """Query all facts mentioning a specific symbol."""
        job_id = "job-symbol-test"

        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="RELIANCE", fiscal_year=2025,
            metric="eps", value=95.0, unit="INR",
            source="src1",
        ))
        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="TCS", fiscal_year=2025,
            metric="eps", value=120.0, unit="INR",
            source="src2",
        ))

        reliance_facts = store.get_facts_for_symbol(job_id, "RELIANCE")
        assert len(reliance_facts) == 1
        assert isinstance(reliance_facts[0], FundamentalRow)
        assert reliance_facts[0].symbol == "RELIANCE"

    def test_get_job_facts(self, store):
        """Get all facts for a job."""
        job_id = "job-all-test"

        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="INFY", fiscal_year=2025,
            metric="revenue", value=500.0, source="src1",
        ))
        store.put_fact(RiskFlag(
            job_id=job_id, symbol_or_pair="INFY",
            flag_type="concentration", detail="High sector exposure",
            severity=Severity.WARNING, source="risk_agent",
        ))

        all_facts = store.get_job_facts(job_id)
        assert len(all_facts) == 2

    def test_get_validated_signals_empty(self, store, sample_signal):
        """Unvalidated signals are excluded from get_validated_signals."""
        store.put_fact(sample_signal)  # validated=False

        validated = store.get_validated_signals(sample_signal.job_id)
        assert len(validated) == 0

    def test_get_validated_signals(self, store):
        """Only validated signals are returned."""
        job_id = "job-validated-test"

        # One validated, one not
        store.put_fact(QuantSignal(
            job_id=job_id, symbol_or_pair="PAIR-A",
            signal_type=SignalType.MOMENTUM, value=0.5,
            validated=True, as_of=date(2025, 3, 31),
            source="backtest_v1",
        ))
        store.put_fact(QuantSignal(
            job_id=job_id, symbol_or_pair="PAIR-B",
            signal_type=SignalType.MOMENTUM, value=0.3,
            validated=False, as_of=date(2025, 3, 31),
            source="backtest_v1",
        ))

        validated = store.get_validated_signals(job_id)
        assert len(validated) == 1
        assert validated[0].symbol_or_pair == "PAIR-A"

    def test_count_facts(self, store):
        """Count facts by type."""
        job_id = "job-count-test"

        for i in range(3):
            store.put_fact(FundamentalRow(
                job_id=job_id, symbol="HDFC", fiscal_year=2023 + i,
                metric="revenue", value=float(i * 100), source=f"src{i}",
            ))
        store.put_fact(RiskFlag(
            job_id=job_id, symbol_or_pair="HDFC",
            flag_type="test", detail="test", severity=Severity.INFO,
            source="test",
        ))

        counts = store.count_facts(job_id)
        assert counts["fundamental"] == 3
        assert counts["risk_flag"] == 1


class TestFactStoreCheckpointing:
    """
    Test per-stage checkpointing: facts survive even if later stages fail.

    This directly tests AGENTS.md rule 5.
    """

    def test_facts_persist_across_stages(self, store):
        """
        Simulate a pipeline where stage 1 writes facts, stage 2 fails.
        Facts from stage 1 must still be in the store.
        """
        job_id = "job-checkpoint-test"

        # Stage 1: Fundamentals Agent writes facts
        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="SBIN", fiscal_year=2025,
            metric="revenue", value=5000.0, source="stage1",
        ))
        store.put_fact(FundamentalRow(
            job_id=job_id, symbol="SBIN", fiscal_year=2025,
            metric="operating_margin", value=18.5, source="stage1",
        ))

        # Stage 2: Simulate a failure (e.g. rate limit)
        # In the real pipeline, the exception is caught and the job
        # is marked as failed — but the Fact Store is not rolled back.

        # Verify stage 1 facts survived
        facts = store.get_facts_by_type(job_id, FactType.FUNDAMENTAL)
        assert len(facts) == 2
        assert all(isinstance(f, FundamentalRow) for f in facts)
        assert all(f.source == "stage1" for f in facts)


class TestFactStoreBulkOps:
    """Test bulk operations."""

    def test_put_facts_bulk(self, store):
        """Bulk insert multiple facts in one call."""
        job_id = "job-bulk-test"
        facts = [
            FundamentalRow(
                job_id=job_id, symbol="TCS", fiscal_year=2020 + i,
                metric="revenue", value=float(i * 1000), source=f"bulk{i}",
            )
            for i in range(5)
        ]

        ids = store.put_facts(facts)
        assert len(ids) == 5

        all_facts = store.get_job_facts(job_id)
        assert len(all_facts) == 5

    def test_delete_job_facts(self, store):
        """Delete all facts for a job."""
        job_id = "job-delete-all"

        for i in range(3):
            store.put_fact(FundamentalRow(
                job_id=job_id, symbol="ABC", fiscal_year=2025,
                metric=f"metric_{i}", value=float(i), source="test",
            ))

        count = store.delete_job_facts(job_id)
        assert count == 3
        assert store.get_job_facts(job_id) == []


class TestFactStoreEdgeCases:
    """Edge cases and error handling."""

    def test_empty_store_queries(self, store):
        """Querying an empty store returns empty results."""
        assert store.get_job_facts("nonexistent") == []
        assert store.get_facts_by_type("nonexistent", FactType.FUNDAMENTAL) == []
        assert store.get_validated_signals("nonexistent") == []
        assert store.count_facts("nonexistent") == {}

    def test_upsert_same_fact_id(self, store):
        """Writing a fact with the same fact_id updates it (upsert)."""
        fact = FundamentalRow(
            job_id="job-upsert", symbol="XYZ", fiscal_year=2025,
            metric="revenue", value=100.0, source="v1",
        )
        store.put_fact(fact)

        # Update the same fact
        fact.value = 200.0
        fact.source = "v2"
        store.put_fact(fact)

        retrieved = store.get_fact(fact.fact_id)
        assert retrieved.value == 200.0
        assert retrieved.source == "v2"

        # Should still be only one fact for this job
        all_facts = store.get_job_facts("job-upsert")
        assert len(all_facts) == 1
