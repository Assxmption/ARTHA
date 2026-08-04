"""
Tests — Multi-Agent Research Assistant
=======================================
Basic tests for the research crew pipeline.
"""

import os
import sys
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

        from app.crew.research_crew import ResearchCrew
        import uuid

        crew = ResearchCrew()
        job_id = str(uuid.uuid4())

        progress_log = []
        def on_progress(agent, message):
            progress_log.append(f"[{agent}] {message}")

        result = crew.run(
            query="What is Python programming language? Brief overview.",
            job_id=job_id,
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
        from app.schemas.models import ResearchRequest
        req = ResearchRequest(query="Compare CrewAI and LangGraph frameworks")
        assert req.query == "Compare CrewAI and LangGraph frameworks"

    def test_research_request_too_short(self):
        from app.schemas.models import ResearchRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ResearchRequest(query="short")

    def test_job_status_model(self):
        from app.schemas.models import JobStatus
        status = JobStatus(
            job_id="test-123",
            query="Test query for validation",
            status="running",
        )
        assert status.job_id == "test-123"
        assert status.status == "running"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
