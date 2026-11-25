"""End-to-end analysis on a real symbol with real fundamentals and news."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def crew(tmp_path, monkeypatch):
    import app.crew.analysis_crew as ac

    monkeypatch.setattr(ac, "REPORTS_DIR", tmp_path / "reports")
    return ac.AnalysisCrew(db_url=f"sqlite:///{tmp_path / 'facts.db'}")


def test_data_only_report_on_live_data_passes_citation_gate(crew):
    result = crew.run("TCS", skip_llm=True)
    assert result["status"] == "completed"
    assert result["stages"]["fundamentals"]["status"] == "ok"
    assert result["stages"]["fundamentals"]["facts_written"] > 0
    assert result["stages"]["writer"]["citations"]["valid"] is True


@pytest.mark.live_llm
def test_llm_report_on_live_data_ships_only_if_gate_passes(crew):
    """Real LLM, real data. Records whether the LLM text or the fallback shipped."""
    result = crew.run("INFY")
    writer = result["stages"]["writer"]
    assert result["status"] == "completed"
    assert writer["status"] in {"ok", "fallback_citation_safe"} or writer.get("fallback")
    assert writer.get("citations", {}).get("valid", True) is True
    print(f"\nwriter outcome: {writer['status']}")
    print(f"llm_rejection: {writer.get('llm_rejection')} error: {str(writer.get('error'))[:600]}")
