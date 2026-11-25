"""Tests for risk vetoes at the report-context boundary."""

from datetime import date

from app.agents.writer_agent import build_writer_context
from app.crew.analysis_crew import _build_fallback_report
from app.factstore.schemas import QuantSignal, RiskFlag, Severity, SignalType
from app.factstore.store import FactStore


def _store_with_block(tmp_path):
    store = FactStore(db_path=str(tmp_path / "writer.db"))
    job_id = "writer-risk-gate"
    signal = QuantSignal(
        job_id=job_id,
        source="fixture_backtest",
        symbol_or_pair="TCS-INFY",
        signal_type=SignalType.MOMENTUM,
        value=1.2,
        backtest_sharpe=1.2,
        backtest_max_drawdown=0.2,
        validated=True,
        validation_method="walk_forward_oos",
        validation_id="wf-fixture",
        validation_oos_days=504,
        as_of=date(2026, 11, 3),
    )
    block = RiskFlag(
        job_id=job_id,
        source="fixture_risk",
        symbol_or_pair="TCS",
        flag_type="corporate_action_pending",
        detail="Fixture block",
        severity=Severity.BLOCK,
    )
    store.put_facts([signal, block])
    return store, job_id, signal


def test_writer_context_excludes_blocked_signal(tmp_path):
    store, job_id, signal = _store_with_block(tmp_path)
    context = build_writer_context(job_id, store, "TCS")
    assert "VALIDATED QUANT SIGNALS" not in context
    assert signal.fact_id[:8] not in context
    assert "corporate_action_pending" in context


def test_fallback_report_excludes_blocked_signal(tmp_path):
    store, job_id, signal = _store_with_block(tmp_path)
    report = _build_fallback_report(job_id, store, "TCS")
    assert "Quant Signals" not in report
    assert signal.fact_id not in report
    assert "corporate_action_pending" in report


def test_fallback_report_with_numeric_headlines_passes_citation_gate(tmp_path):
    """Regression: headlines containing figures used to be uncited lines,
    so the deterministic fallback report failed its own rule-10 gate."""
    from app.agents.writer_agent import check_citations
    from app.factstore.schemas import FundamentalRow, NewsSignal

    store = FactStore(db_path=str(tmp_path / "fallback.db"))
    job_id = "fallback-headlines"
    store.put_facts([
        FundamentalRow(
            job_id=job_id, source="YFINANCE_FY2025", symbol="TCS",
            fiscal_year=2025, metric="revenue", value=255324.0, unit="INR_crore",
        ),
        NewsSignal(
            job_id=job_id, source="FINBERT_LOCAL", symbol_or_commodity="TCS",
            headline_count=3, sentiment_score=0.21, as_of=date(2026, 11, 20),
            headlines_sample=[
                "TCS Q2 profit rises 9% to Rs 12,075 crore",
                "TCS bags $1.2 billion deal\nfrom UK insurer",
            ],
        ),
    ])
    report = _build_fallback_report(job_id, store, "TCS")
    result = check_citations(report, job_id, store)
    assert result["valid"] is True, result
