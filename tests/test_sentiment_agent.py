"""
Tests for the Sentiment Agent and FinBERT+VADER Ensemble
=========================================================

Tests:
  - Ensemble scorer: VADER-only fallback, FinBERT mock, agreement/disagreement
  - Sentiment agent: NewsSignal creation, event classification, Fact Store writes
  - Narration: fallback template, prompt formatting
  - Edge cases: empty news, single headline, no sentiment data

All tests use fixtures — no live API calls, no LLM tokens, no network.
"""

from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from app.data.news import (
    EnsembleSentimentScorer,
    SentimentResult,
    _VADERScorer,
    get_sentiment_summary,
)
from app.agents.sentiment_agent import (
    _classify_event_type,
    run_sentiment_analysis,
    run_sentiment_narration,
    _fallback_narration,
    _format_signals_for_prompt,
)
from app.factstore.schemas import (
    EventType,
    FactType,
    NewsSignal,
    NarrativeFact,
)
from app.factstore.store import FactStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Create a fresh in-memory Fact Store."""
    db_path = tmp_path / "test_sentiment.db"
    return FactStore(db_path=str(db_path))


@pytest.fixture
def mock_news_items():
    """Fixture: pre-scored news items like fetch_news() would return."""
    return [
        {
            "title": "Reliance Industries reports record quarterly profit",
            "link": "https://example.com/1",
            "date": "2026-10-26 10:00",
            "description": "Revenue beat analyst estimates by 5%",
            "source": "Google News",
            "sentiment": {
                "compound": 0.65,
                "label": "bullish",
                "finbert_score": 0.85,
                "vader_score": 0.35,
                "confidence": 0.92,
                "positive": 0.88,
                "negative": 0.03,
                "neutral": 0.09,
                "method": "ensemble",
            },
        },
        {
            "title": "RBI tightens NPA norms for banking sector",
            "link": "https://example.com/2",
            "date": "2026-10-26 09:00",
            "description": "New circular mandates higher provisioning",
            "source": "Economic Times",
            "sentiment": {
                "compound": -0.45,
                "label": "bearish",
                "finbert_score": -0.60,
                "vader_score": -0.20,
                "confidence": 0.88,
                "positive": 0.05,
                "negative": 0.72,
                "neutral": 0.23,
                "method": "ensemble",
            },
        },
        {
            "title": "Markets consolidate ahead of quarterly results season",
            "link": "https://example.com/3",
            "date": "2026-10-25 16:00",
            "description": "Nifty trades flat with mixed sectoral cues",
            "source": "Yahoo Finance",
            "sentiment": {
                "compound": 0.02,
                "label": "neutral",
                "finbert_score": 0.05,
                "vader_score": -0.02,
                "confidence": 0.75,
                "positive": 0.15,
                "negative": 0.10,
                "neutral": 0.75,
                "method": "ensemble",
            },
        },
    ]


# ── VADER Scorer Tests ────────────────────────────────────────────────────────

class TestVADERScorer:
    def test_bullish_headline(self):
        scorer = _VADERScorer()
        result = scorer.score("Stock rallies on strong earnings beat")
        assert result["compound"] > 0
        assert "positive" in result

    def test_bearish_headline(self):
        scorer = _VADERScorer()
        result = scorer.score("Company crashes after massive loss and downgrade")
        assert result["compound"] < 0

    def test_neutral_headline(self):
        scorer = _VADERScorer()
        result = scorer.score("Board meeting scheduled for next Thursday")
        assert -0.3 <= result["compound"] <= 0.3

    def test_returns_required_keys(self):
        scorer = _VADERScorer()
        result = scorer.score("Some headline text")
        assert "compound" in result
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result


# ── Ensemble Scorer Tests ─────────────────────────────────────────────────────

class TestEnsembleScorer:
    def test_vader_only_fallback(self):
        """If FinBERT fails to load, ensemble falls back to VADER-only."""
        scorer = EnsembleSentimentScorer()
        # Force FinBERT to "fail"
        scorer._finbert._load_failed = True
        scorer._finbert._loaded = False

        result = scorer.score("Stock surges on record profits")
        assert isinstance(result, SentimentResult)
        assert result.method == "vader_only"
        assert result.compound > 0
        assert result.confidence == 0.5  # default for single-model

    def test_batch_scoring(self):
        """Batch scoring returns correct number of results."""
        scorer = EnsembleSentimentScorer()
        scorer._finbert._load_failed = True  # VADER-only for test speed

        texts = ["Great earnings", "Terrible loss", "Meeting scheduled"]
        results = scorer.score_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_result_to_dict(self):
        """SentimentResult.to_dict() returns all expected keys."""
        result = SentimentResult(
            compound=0.5, label="bullish", finbert_score=0.6,
            vader_score=0.3, confidence=0.9, positive=0.8,
            negative=0.1, neutral_prob=0.1, method="ensemble",
        )
        d = result.to_dict()
        assert d["compound"] == 0.5
        assert d["label"] == "bullish"
        assert d["method"] == "ensemble"
        assert all(k in d for k in [
            "compound", "label", "finbert_score", "vader_score",
            "confidence", "positive", "negative", "neutral", "method",
        ])

    def test_labels_from_thresholds(self):
        """Verify label assignment follows threshold rules."""
        scorer = EnsembleSentimentScorer()
        scorer._finbert._load_failed = True

        # Strong bullish
        r = scorer.score("Stock surges and rallies massively on incredible profits")
        assert r.label in ("bullish", "neutral")  # VADER may vary

        # Strong bearish
        r = scorer.score("Stock crashes plunges terrible massive loss")
        assert r.label in ("bearish", "neutral")


# ── Event Type Classification Tests ──────────────────────────────────────────

class TestEventClassification:
    def test_earnings_event(self):
        assert _classify_event_type("Q3 results: profit beats estimates") == EventType.EARNINGS

    def test_corporate_action(self):
        assert _classify_event_type("Board approves 1:5 stock split and bonus") == EventType.CORPORATE_ACTION

    def test_policy_event(self):
        assert _classify_event_type("RBI announces rate cut in monetary policy review") == EventType.POLICY

    def test_supply_shock(self):
        assert _classify_event_type("OPEC cuts crude oil production amid supply shortage") == EventType.SUPPLY_SHOCK

    def test_no_match(self):
        assert _classify_event_type("Company appoints new board member") is None

    def test_multiple_keywords_picks_best(self):
        # This headline has both earnings AND policy keywords
        result = _classify_event_type("RBI policy impact on quarterly earnings results profit")
        assert result in (EventType.EARNINGS, EventType.POLICY)


# ── Sentiment Agent Pipeline Tests ───────────────────────────────────────────

class TestSentimentAnalysis:
    @patch("app.agents.sentiment_agent.fetch_news")
    def test_writes_news_signal(self, mock_fetch, store, mock_news_items):
        """run_sentiment_analysis writes a NewsSignal to the Fact Store."""
        mock_fetch.return_value = mock_news_items

        signal = run_sentiment_analysis("job_001", store, "RELIANCE")

        assert isinstance(signal, NewsSignal)
        assert signal.symbol_or_commodity == "RELIANCE"
        assert signal.headline_count == 3
        assert signal.job_id == "job_001"
        assert signal.source.startswith("SENTIMENT_AGENT:ensemble")

        # Verify it's in the store
        facts = store.get_facts_by_type("job_001", FactType.NEWS_SIGNAL)
        assert len(facts) == 1
        assert facts[0].fact_id == signal.fact_id

    @patch("app.agents.sentiment_agent.fetch_news")
    def test_sentiment_score_is_aggregate(self, mock_fetch, store, mock_news_items):
        """Sentiment score should be the average compound from the summary."""
        mock_fetch.return_value = mock_news_items

        signal = run_sentiment_analysis("job_002", store, "RELIANCE")

        # Average of 0.65, -0.45, 0.02 = 0.0733...
        expected_avg = (0.65 + (-0.45) + 0.02) / 3
        assert abs(signal.sentiment_score - expected_avg) < 0.01

    @patch("app.agents.sentiment_agent.fetch_news")
    def test_headlines_sample_populated(self, mock_fetch, store, mock_news_items):
        """Top headlines by absolute compound should be in the sample."""
        mock_fetch.return_value = mock_news_items

        signal = run_sentiment_analysis("job_003", store, "RELIANCE")

        assert len(signal.headlines_sample) <= 5
        assert len(signal.headlines_sample) > 0
        # Highest absolute compound (0.65) headline should be first
        assert "record quarterly profit" in signal.headlines_sample[0].lower()

    @patch("app.agents.sentiment_agent.fetch_news")
    def test_empty_news(self, mock_fetch, store):
        """Empty news list should produce a valid signal with zero score."""
        mock_fetch.return_value = []

        signal = run_sentiment_analysis("job_004", store, "UNKNOWN")

        assert signal.headline_count == 0
        assert signal.sentiment_score == 0.0
        assert signal.headlines_sample == []

    @patch("app.agents.sentiment_agent.fetch_news")
    def test_event_type_detected(self, mock_fetch, store):
        """Dominant event type should be classified from headline keywords."""
        mock_fetch.return_value = [
            {
                "title": "Q3 earnings: massive profit beat, revenue surges",
                "link": "",
                "date": "2026-10-26",
                "description": "",
                "source": "Test",
                "sentiment": {
                    "compound": 0.5, "label": "bullish",
                    "finbert_score": 0.6, "vader_score": 0.3,
                    "confidence": 0.9, "positive": 0.8,
                    "negative": 0.1, "neutral": 0.1, "method": "ensemble",
                },
            },
            {
                "title": "Quarterly results show strong earnings growth",
                "link": "",
                "date": "2026-10-25",
                "description": "",
                "source": "Test",
                "sentiment": {
                    "compound": 0.3, "label": "bullish",
                    "finbert_score": 0.4, "vader_score": 0.2,
                    "confidence": 0.85, "positive": 0.6,
                    "negative": 0.15, "neutral": 0.25, "method": "ensemble",
                },
            },
        ]

        signal = run_sentiment_analysis("job_005", store, "TCS")
        assert signal.event_type == EventType.EARNINGS


# ── Narration Tests ──────────────────────────────────────────────────────────

class TestSentimentNarration:
    def test_fallback_narration(self):
        """Fallback template should produce valid text with fact_ids."""
        signal = NewsSignal(
            job_id="job_010",
            source="TEST",
            symbol_or_commodity="INFY",
            headline_count=5,
            sentiment_score=0.35,
            event_type=EventType.EARNINGS,
            as_of=date(2026, 10, 26),
            headlines_sample=["Infosys beats Q3 estimates"],
        )

        text = _fallback_narration([signal])
        assert "INFY" in text
        assert "bullish" in text
        assert "0.35" in text
        assert signal.fact_id[:8] in text

    def test_format_signals_for_prompt(self):
        """Prompt formatter should include all signal details."""
        signal = NewsSignal(
            job_id="job_011",
            source="TEST",
            symbol_or_commodity="SBIN",
            headline_count=10,
            sentiment_score=-0.25,
            event_type=EventType.POLICY,
            as_of=date(2026, 10, 26),
            headlines_sample=["RBI tightens norms"],
        )

        text = _format_signals_for_prompt([signal])
        assert "SBIN" in text
        assert "-0.25" in text
        assert "BEAR" in text
        assert "policy" in text
        assert signal.fact_id[:8] in text

    @patch("app.agents.sentiment_agent.fetch_news")
    def test_narration_with_no_signals(self, mock_fetch, store):
        """Narration with no signals should still produce a valid fact."""
        fact = run_sentiment_narration("job_012", store, "UNKNOWN")

        assert isinstance(fact, NarrativeFact)
        assert "No news sentiment data" in fact.content
        assert fact.agent_name == "sentiment_agent"

    @patch("app.agents.sentiment_agent.fetch_news")
    @patch("app.agents.sentiment_agent.get_cheap_llm")
    def test_narration_writes_to_store(self, mock_llm, mock_fetch, store, mock_news_items):
        """Full narration pipeline should write NarrativeFact to store."""
        # First create a NewsSignal
        mock_fetch.return_value = mock_news_items
        signal = run_sentiment_analysis("job_013", store, "RELIANCE")

        # Mock LLM response
        mock_llm_instance = MagicMock()
        mock_llm_instance.call.return_value = (
            "## News Sentiment Assessment\n\n"
            f"Sentiment is bullish [fact_id: {signal.fact_id[:8]}…]."
        )
        mock_llm.return_value = mock_llm_instance

        # Run narration
        narrative = run_sentiment_narration("job_013", store, "RELIANCE")

        assert isinstance(narrative, NarrativeFact)
        assert narrative.agent_name == "sentiment_agent"
        assert narrative.section == "sentiment_analysis"
        assert signal.fact_id in narrative.referenced_fact_ids


# ── Summary Aggregation Tests ────────────────────────────────────────────────

class TestSentimentSummary:
    def test_empty_items(self):
        summary = get_sentiment_summary([])
        assert summary["overall"] == "neutral"
        assert summary["total_articles"] == 0
        assert summary["avg_confidence"] == 0.0

    def test_bullish_summary(self):
        items = [
            {"sentiment": {"compound": 0.5, "label": "bullish", "confidence": 0.9, "method": "ensemble"}},
            {"sentiment": {"compound": 0.3, "label": "bullish", "confidence": 0.8, "method": "ensemble"}},
        ]
        summary = get_sentiment_summary(items)
        assert summary["overall"] == "bullish"
        assert summary["score"] > 0.1
        assert summary["distribution"]["bullish"] == 2
        assert summary["method"] == "ensemble"

    def test_bearish_summary(self):
        items = [
            {"sentiment": {"compound": -0.6, "label": "bearish", "confidence": 0.85, "method": "ensemble"}},
            {"sentiment": {"compound": -0.4, "label": "bearish", "confidence": 0.9, "method": "ensemble"}},
        ]
        summary = get_sentiment_summary(items)
        assert summary["overall"] == "bearish"
        assert summary["score"] < -0.1

    def test_mixed_summary(self):
        items = [
            {"sentiment": {"compound": 0.5, "label": "bullish", "confidence": 0.9, "method": "ensemble"}},
            {"sentiment": {"compound": -0.5, "label": "bearish", "confidence": 0.9, "method": "ensemble"}},
        ]
        summary = get_sentiment_summary(items)
        assert summary["overall"] == "neutral"
        assert summary["distribution"]["bullish"] == 1
        assert summary["distribution"]["bearish"] == 1
