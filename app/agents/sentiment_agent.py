"""
Sentiment Agent — News-Driven Market Sentiment Analysis
=========================================================
Fetches ticker-scoped news, scores it with the FinBERT+VADER ensemble,
writes NewsSignal facts to the Fact Store, and optionally narrates
findings via the cheap LLM tier.

Pipeline:
  1. fetch_news() → raw headlines from RSS feeds (Google, Yahoo, ET)
  2. EnsembleSentimentScorer.score_batch() → per-headline scores (local,
     no LLM tokens)
  3. Aggregate → write NewsSignal to Fact Store (checkpoint)
  4. (Optional) LLM narration → NarrativeFact for the writer agent

Non-negotiable constraints (AGENTS.md):
  - Rule 1: Sentiment SCORES are computed deterministically (FinBERT +
    VADER), never by an LLM.  The LLM only narrates pre-computed facts.
  - Rule 2: No sequential prompt-stacking.  The agent reads typed
    Pydantic objects from the Fact Store.
  - Rule 10: Every claim in the narrative must reference a fact_id.

Uses cheap model for narration (AGENTS.md rule 3).

Reference: docs/ARTHA_ARCHITECTURE.md §6.1
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from app.data.news import (
    EnsembleSentimentScorer,
    fetch_news,
    get_scorer,
    get_sentiment_summary,
)
from app.factstore.schemas import (
    EventType,
    FactType,
    NarrativeFact,
    NewsSignal,
)
from app.factstore.store import FactStore
from app.llm.router import get_cheap_llm

logger = logging.getLogger(__name__)

# ── Event Type Classification ──────────────────────────────────────────────────

# Keyword patterns for coarse event-type tagging.
# These run on headline text AFTER sentiment scoring — classification,
# not sentiment computation, so they're fine as keyword rules.
EVENT_KEYWORDS: dict[EventType, set[str]] = {
    EventType.EARNINGS: {
        "earnings", "quarterly", "q1", "q2", "q3", "q4", "profit", "revenue",
        "result", "results", "net income", "ebitda", "eps", "beat", "miss",
        "guidance", "topline", "bottomline", "margin", "yoy", "qoq",
    },
    EventType.CORPORATE_ACTION: {
        "dividend", "buyback", "split", "bonus", "merger", "acquisition",
        "takeover", "demerger", "rights issue", "ipo", "ofs", "delisting",
        "amalgamation", "restructur",
    },
    EventType.POLICY: {
        "rbi", "sebi", "policy", "regulation", "rate cut", "rate hike",
        "monetary", "fiscal", "budget", "gst", "tax", "compliance",
        "circular", "norms", "framework", "ban", "approval",
    },
    EventType.SUPPLY_SHOCK: {
        "supply", "shortage", "opec", "crude", "monsoon", "drought",
        "flood", "harvest", "production cut", "inventory", "demand",
        "commodity", "inflation", "cpi", "wpi",
    },
}


def _classify_event_type(headline: str) -> Optional[EventType]:
    """
    Classify a headline into a coarse event type.
    Returns the best-matching EventType, or None if no strong match.
    """
    text_lower = headline.lower()
    best_type = None
    best_count = 0

    for event_type, keywords in EVENT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_type = event_type

    # Require at least 1 keyword match
    return best_type if best_count >= 1 else None


# ── Core Pipeline ──────────────────────────────────────────────────────────────

def run_sentiment_analysis(
    job_id: str,
    store: FactStore,
    symbol: str,
    max_headlines: int = 25,
) -> NewsSignal:
    """
    Run the full sentiment analysis pipeline for a symbol:
      1. Fetch news from RSS feeds
      2. Score with FinBERT+VADER ensemble (deterministic, no LLM)
      3. Aggregate into a NewsSignal
      4. Write to Fact Store (checkpoint — AGENTS.md rule 5)

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    symbol : str
        NSE symbol or commodity identifier.
    max_headlines : int
        Maximum headlines to fetch per source.

    Returns
    -------
    NewsSignal
        The persisted sentiment fact.
    """
    # 1. Fetch news (cached, 10-min TTL)
    news_items = fetch_news(symbol, max_items=max_headlines)
    logger.info("Fetched %d news items for %s", len(news_items), symbol)

    # 2. Aggregate sentiment (already scored by fetch_news via ensemble)
    summary = get_sentiment_summary(news_items)

    # 3. Classify dominant event type from headlines
    event_counts: dict[EventType, int] = {}
    for item in news_items:
        et = _classify_event_type(item.get("title", ""))
        if et:
            event_counts[et] = event_counts.get(et, 0) + 1

    dominant_event = max(event_counts, key=event_counts.get) if event_counts else None

    # 4. Pick representative headlines (top 5 by absolute compound)
    scored_items = [
        (item, abs(item.get("sentiment", {}).get("compound", 0)))
        for item in news_items
    ]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    top_headlines = [item["title"] for item, _ in scored_items[:5]]

    # 5. Build and persist NewsSignal
    signal = NewsSignal(
        job_id=job_id,
        source=f"SENTIMENT_AGENT:ensemble:{summary.get('method', 'unknown')}",
        symbol_or_commodity=symbol,
        headline_count=len(news_items),
        sentiment_score=summary["score"],
        event_type=dominant_event,
        as_of=date.today(),
        headlines_sample=top_headlines,
    )
    store.put_fact(signal)

    logger.info(
        "Sentiment Agent: %s → %s (score=%.4f, %d headlines, event=%s, confidence=%.2f) [%s]",
        symbol,
        summary["overall"].upper(),
        summary["score"],
        len(news_items),
        dominant_event.value if dominant_event else "none",
        summary.get("avg_confidence", 0),
        signal.fact_id[:8],
    )

    return signal


def run_sentiment_narration(
    job_id: str,
    store: FactStore,
    symbol: Optional[str] = None,
) -> NarrativeFact:
    """
    Narrate NewsSignal facts via the cheap LLM tier.

    This is the ONLY part that touches LLM tokens — the scoring itself
    is fully deterministic (FinBERT + VADER).

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    symbol : str, optional
        Filter to a specific symbol. If None, narrates all news signals.

    Returns
    -------
    NarrativeFact
        The sentiment narrative, persisted to the Fact Store.
    """
    # Read NewsSignal facts from store
    all_facts = store.get_facts_by_type(job_id, FactType.NEWS_SIGNAL)
    news_signals: list[NewsSignal] = []
    for f in all_facts:
        if isinstance(f, NewsSignal):
            if symbol is None or f.symbol_or_commodity == symbol:
                news_signals.append(f)

    if not news_signals:
        # Write a "no data" narrative
        fact = NarrativeFact(
            job_id=job_id,
            source="SENTIMENT_AGENT",
            agent_name="sentiment_agent",
            section="sentiment_analysis",
            content=f"No news sentiment data available for {symbol or 'the portfolio'}.",
            referenced_fact_ids=[],
        )
        store.put_fact(fact)
        return fact

    # Format facts for the LLM prompt
    signal_text = _format_signals_for_prompt(news_signals)

    prompt = f"""You are the ARTHA Sentiment Analyst — a specialist in Indian market
news sentiment for NSE equities and MCX commodities. You are reading
pre-computed sentiment scores from the FinBERT+VADER ensemble.

RULES:
- NEVER compute, estimate, or re-score any sentiment. All scores below
  are pre-computed by the deterministic ensemble.
- Every numeric claim MUST reference the Fact ID it comes from.
- Explain what the sentiment landscape means for the stock/portfolio.
- Flag any divergence between headline sentiment and price action.
- Note the dominant event type if present (earnings, policy, etc.).

{signal_text}

Write a 2-3 paragraph narrative titled "News Sentiment Assessment" that covers:
1. Overall sentiment direction and conviction (score + headline count)
2. Dominant themes/events driving the sentiment
3. Key headlines to watch and their potential market impact

Reference Fact IDs as [fact_id: XXXXXXXX…] for traceability."""

    try:
        llm = get_cheap_llm()
        response = llm.call([{"role": "user", "content": prompt}])
        narrative_text = str(response)
    except Exception as e:
        logger.error("LLM call failed for sentiment narration: %s", e)
        narrative_text = _fallback_narration(news_signals)

    # Persist (checkpoint)
    fact = NarrativeFact(
        job_id=job_id,
        source="SENTIMENT_AGENT",
        agent_name="sentiment_agent",
        section="sentiment_analysis",
        content=narrative_text,
        referenced_fact_ids=[s.fact_id for s in news_signals],
    )
    store.put_fact(fact)

    logger.info(
        "Sentiment Agent narration: %d chars, %d signal refs",
        len(narrative_text), len(news_signals),
    )
    return fact


# ── Prompt Formatting ──────────────────────────────────────────────────────────

def _format_signals_for_prompt(signals: list[NewsSignal]) -> str:
    """Format NewsSignal facts into structured text for the LLM prompt."""
    lines = ["## News Sentiment Signals\n"]
    lines.append("| Fact ID | Symbol | Score | Headlines | Event Type | Date |")
    lines.append("|---------|--------|-------|-----------|------------|------|")

    for s in signals:
        direction = "🟢 BULL" if s.sentiment_score > 0.1 else "🔴 BEAR" if s.sentiment_score < -0.1 else "⚪ NEUTRAL"
        lines.append(
            f"| {s.fact_id[:8]}… | {s.symbol_or_commodity} "
            f"| {s.sentiment_score:+.4f} ({direction}) "
            f"| {s.headline_count} "
            f"| {s.event_type.value if s.event_type else 'general'} "
            f"| {s.as_of} |"
        )

    lines.append("\n### Sample Headlines")
    for s in signals:
        if s.headlines_sample:
            lines.append(f"\n**{s.symbol_or_commodity}** [fact_id: {s.fact_id[:8]}…]:")
            for h in s.headlines_sample[:3]:
                lines.append(f"  - {h}")

    return "\n".join(lines)


def _fallback_narration(signals: list[NewsSignal]) -> str:
    """Template-based fallback when LLM is unavailable."""
    lines = ["## News Sentiment Assessment\n"]

    for s in signals:
        if s.sentiment_score > 0.1:
            direction = "bullish"
        elif s.sentiment_score < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"

        lines.append(
            f"**{s.symbol_or_commodity}**: Sentiment is **{direction}** "
            f"(score: {s.sentiment_score:+.4f}) based on {s.headline_count} "
            f"headlines [fact_id: {s.fact_id[:8]}…]. "
        )

        if s.event_type:
            lines.append(
                f"Dominant event type: {s.event_type.value}. "
            )

        if s.headlines_sample:
            lines.append("Key headlines:")
            for h in s.headlines_sample[:3]:
                lines.append(f"  - {h}")
            lines.append("")

    return "\n".join(lines)

