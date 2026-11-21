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


class TestWebSearchTool:
    """Test the WebSearchTool (Serper → DuckDuckGo fallback)."""

    def test_duckduckgo_fallback(self):
        """Test DuckDuckGo search works without Serper key."""
        os.environ.pop("SERPER_API_KEY", None)

        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        result = tool._run(query="Python FastAPI tutorial", max_results=3)

        assert isinstance(result, str)
        assert len(result) > 100
        print(f"\n[DuckDuckGo] Result preview: {result[:200]}")

    def test_serper_with_key(self):
        """Test Serper Google Search if API key is present."""
        serper_key = os.getenv("SERPER_API_KEY", "").strip()
        if not serper_key:
            pytest.skip("SERPER_API_KEY not set — skipping Serper test")

        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        result = tool._run(query="CrewAI multi-agent framework", max_results=3)

        assert isinstance(result, str)
        assert "Google Search" in result
        assert len(result) > 100
        print(f"\n[Serper] Result preview: {result[:300]}")


class TestWebScraperTool:
    """Test the WebScraperTool."""

    def test_scrape_basic_url(self):
        from app.tools.web_scraper import WebScraperTool
        tool = WebScraperTool()
        result = tool._run(url="https://httpbin.org/html", max_chars=2000)

        assert isinstance(result, str)
        assert len(result) > 50
        print(f"\n[Scraper] Result preview: {result[:200]}")

    def test_invalid_url(self):
        from app.tools.web_scraper import WebScraperTool
        tool = WebScraperTool()
        result = tool._run(url="https://this-domain-does-not-exist-xyz.com")
        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower() or "timeout" in result.lower()


class TestFileReaderTool:
    """Test the FileReaderTool."""

    def test_missing_file(self):
        from app.tools.file_reader import FileReaderTool
        tool = FileReaderTool()
        result = tool._run(file_path="nonexistent_file.pdf")
        assert "not found" in result.lower()

    def test_read_text_file(self, tmp_path):
        from app.tools.file_reader import FileReaderTool
        # Create temp text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, this is test content for the file reader tool.")

        tool = FileReaderTool()
        result = tool._run(file_path=str(test_file))
        assert "Hello" in result
        assert "test content" in result


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
