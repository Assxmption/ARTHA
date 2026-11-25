"""
Tests — Multi-Agent Research Assistant
=======================================
Basic tests for the research crew pipeline.
"""

import os
import sys
from datetime import date
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


class TestResearchCrew:
    """Integration test for the Research Crew."""

    @pytest.mark.slow
    def test_crew_short_query(self):
        """
        Integration test: Run a quick crew execution.
        This calls real LLMs — only run with --run-slow flag or RUN_SLOW_TESTS=true.
        """
        if os.getenv("RUN_SLOW_TESTS", "").lower() != "true":
            pytest.skip("Skipping slow integration test. Set RUN_SLOW_TESTS=true to run.")

        from app.crew.analysis_crew import AnalysisCrew
        import uuid

        crew = AnalysisCrew()
        job_id = str(uuid.uuid4())

        progress_log = []
        def on_progress(agent, message):
            progress_log.append(f"[{agent}] {message}")

        result = crew.run(
            symbol="RELIANCE",
            job_id=job_id,
            skip_llm=True,
            on_progress=on_progress,
        )

        assert result["status"] == "completed"
        assert result["report"] is not None
        assert len(result["report"]) > 200
        print(f"\n[Crew] Report preview:\n{result['report'][:500]}")
        print(f"\n[Crew] Progress log:\n" + "\n".join(progress_log[:10]))


class TestAPISchemas:
    """Test Pydantic schemas."""

    def test_research_request_valid(self):
        from app.schemas.models import AnalysisRequest
        req = AnalysisRequest(symbol="RELIANCE")
        assert req.symbol == "RELIANCE"

    def test_research_request_too_short(self):
        from app.schemas.models import AnalysisRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            AnalysisRequest(symbol="")

    def test_job_status_model(self):
        from app.schemas.models import AnalysisJobStatus
        status = AnalysisJobStatus(
            job_id="test-123",
            symbol="RELIANCE",
            status="running",
        )
        assert status.job_id == "test-123"
        assert status.status == "running"


def test_analysis_crew_completed_job_is_idempotent(tmp_path, monkeypatch):
    """A worker restart must not duplicate completed pipeline stages."""
    import app.crew.analysis_crew as crew_module
    from app.factstore.schemas import FundamentalRow, NewsSignal

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(crew_module, "REPORTS_DIR", reports_dir)
    calls = {"fundamentals": 0, "sentiment": 0}

    def fake_fundamentals(symbol, job_id, store, start_fy):
        calls["fundamentals"] += 1
        fact = FundamentalRow(
            job_id=job_id,
            symbol=symbol,
            fiscal_year=2025,
            metric="revenue",
            value=100.0,
            unit="INR_crore",
            source="fixture_fundamentals",
        )
        store.put_fact(fact)
        return {"facts": [fact], "statement_count": 1, "market_count": 0}

    def fake_sentiment(job_id, store, symbol):
        calls["sentiment"] += 1
        fact = NewsSignal(
            job_id=job_id,
            symbol_or_commodity=symbol,
            headline_count=1,
            sentiment_score=0.2,
            as_of=date(2026, 11, 3),
            headlines_sample=["Fixture headline"],
            source="fixture_news",
        )
        store.put_fact(fact)
        return fact

    monkeypatch.setattr(crew_module, "run_fundamentals_analysis", fake_fundamentals)
    monkeypatch.setattr(crew_module, "run_sentiment_analysis", fake_sentiment)

    crew = crew_module.AnalysisCrew(db_url=f"sqlite:///{tmp_path / 'facts.db'}")
    first = crew.run("TCS", job_id="resume-test", skip_llm=True)
    second = crew.run("TCS", job_id="resume-test", skip_llm=True)

    assert first["status"] == "completed"
    assert second["resumed"] is True
    assert calls == {"fundamentals": 1, "sentiment": 1}
    assert crew.store.count_facts("resume-test")["fundamental"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
