"""
ARTHA News & Sentiment Module — Ensemble (FinBERT + VADER)
============================================================
Fetches ticker-scoped financial news and performs sentiment analysis
using an ensemble of FinBERT (ProsusAI/finbert, local transformer)
and VADER (lexicon-based, finance-augmented).

Design:
  - FinBERT: fine-tuned BERT for financial sentiment.  High accuracy
    on earnings, M&A, regulatory headlines.  ~15ms/headline on CPU.
    Outputs 3-class probabilities (positive, negative, neutral).
  - VADER: rule-based, zero-latency, good for short/informal text.
    Augmented with a finance-specific lexicon.
  - Ensemble: weighted combination (configurable).  Default 65% FinBERT +
    35% VADER.  When models disagree strongly, the ensemble flags
    "low_confidence" so downstream consumers can treat it accordingly.

Architecture note (AGENTS.md rule 1):
  All sentiment *scoring* happens here (deterministic, no LLM tokens).
  Only *narration* of the scores touches the LLM quota, via the
  Sentiment Agent.

Reference: docs/ARTHA_ARCHITECTURE.md §6.1
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree

import numpy as np
import requests

logger = logging.getLogger("artha.data.news")

# ── Configuration ──────────────────────────────────────────────────────────────

# Ensemble weights — trade-off: FinBERT is more accurate on financial domain
# text; VADER is faster and handles informal/social text better.
FINBERT_WEIGHT = 0.65
VADER_WEIGHT = 0.35

# Thresholds for final label assignment
BULLISH_THRESHOLD = 0.15
BEARISH_THRESHOLD = -0.15

# Confidence: if FinBERT and VADER disagree in *direction*, flag low confidence
DISAGREEMENT_THRESHOLD = 0.30  # abs diff between the two scores

# RSS Feed URLs
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"
ET_MARKETS_RSS = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
MC_RSS = "https://www.moneycontrol.com/rss/latestnews.xml"

# Cache
_news_cache: dict = {}
_CACHE_TTL = 600  # 10 minutes


# ── Sentiment Scoring Result ──────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """Structured sentiment output from the ensemble."""
    compound: float          # [-1, +1] final ensemble score
    label: str               # "bullish" | "bearish" | "neutral"
    finbert_score: float     # [-1, +1] FinBERT contribution
    vader_score: float       # [-1, +1] VADER contribution
    confidence: float        # [0, 1] agreement-based confidence
    positive: float          # positive probability
    negative: float          # negative probability
    neutral_prob: float      # neutral probability
    method: str              # "ensemble" | "vader_only" | "finbert_only"

    def to_dict(self) -> dict:
        return {
            "compound": round(self.compound, 4),
            "label": self.label,
            "finbert_score": round(self.finbert_score, 4),
            "vader_score": round(self.vader_score, 4),
            "confidence": round(self.confidence, 4),
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral_prob, 4),
            "method": self.method,
        }


# ── FinBERT Scorer ────────────────────────────────────────────────────────────

class _FinBERTScorer:
    """
    Lazy-loaded ProsusAI/finbert.  The model is ~438MB and loads once
    on first use, then stays in memory.  Inference on CPU takes ~15ms
    per headline (batched: ~5ms/headline).
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._labels = ["positive", "negative", "neutral"]
        self._loaded = False
        self._load_failed = False

    def _ensure_loaded(self):
        if self._loaded or self._load_failed:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch  # noqa: F811

            logger.info("Loading ProsusAI/finbert (first use — may take a few seconds)...")
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self._model.eval()
            self._loaded = True
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning("FinBERT load failed — ensemble will fall back to VADER-only: %s", e)
            self._load_failed = True

    def score_batch(self, texts: list[str]) -> list[dict]:
        """
        Score a batch of texts.  Returns list of dicts:
        [{"positive": p, "negative": n, "neutral": u, "compound": c}, ...]

        compound = positive_prob - negative_prob  (range [-1, +1])
        """
        self._ensure_loaded()
        if not self._loaded:
            return [None] * len(texts)

        import torch

        # Tokenize in one batch for efficiency
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()

        results = []
        for prob in probs:
            pos, neg, neu = float(prob[0]), float(prob[1]), float(prob[2])
            results.append({
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "compound": pos - neg,  # [-1, +1]
            })
        return results

    def score(self, text: str) -> Optional[dict]:
        """Score a single text.  Returns dict or None if model unavailable."""
        results = self.score_batch([text])
        return results[0]


# ── VADER Scorer ──────────────────────────────────────────────────────────────

class _VADERScorer:
    """
    VADER with finance-augmented lexicon.  Zero-latency, rule-based.
    Falls back to keyword matching if vaderSentiment is not installed.
    """

    def __init__(self):
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()

            # Finance-specific lexicon updates
            finance_lexicon = {
                "bullish": 2.5, "bearish": -2.5,
                "upgrade": 2.0, "downgrade": -2.0,
                "outperform": 1.8, "underperform": -1.8,
                "buy": 1.5, "sell": -1.5,
                "rally": 2.0, "crash": -3.0,
                "surge": 2.0, "plunge": -2.5,
                "breakout": 1.5, "breakdown": -1.5,
                "beat": 1.5, "miss": -1.5,
                "profit": 1.5, "loss": -1.5,
                "growth": 1.5, "decline": -1.5,
                "strong": 1.0, "weak": -1.0,
                "record high": 2.0, "all-time low": -2.0,
                "dividend": 1.0, "debt": -0.5,
                "expansion": 1.5, "contraction": -1.5,
                "accumulate": 1.5, "reduce": -1.0,
                "npa": -1.5, "default": -2.0,
                "restructure": -1.0, "writeoff": -2.5,
                "acquisition": 0.8, "merger": 0.5,
                "fii": 0.5, "dii": 0.5,
                "overweight": 1.5, "underweight": -1.5,
            }
            self._vader.lexicon.update(finance_lexicon)
            logger.info("VADER sentiment analyzer loaded with finance lexicon")
        except ImportError:
            logger.warning("vaderSentiment not installed — using keyword fallback")

    def score(self, text: str) -> dict:
        """
        Returns: {"compound": float[-1,1], "positive": f, "negative": f, "neutral": f}
        """
        if self._vader:
            scores = self._vader.polarity_scores(text)
            return {
                "compound": scores["compound"],
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
            }

        return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> dict:
        """Simple keyword-based sentiment when VADER is unavailable."""
        text_lower = text.lower()

        bullish_words = {
            "rally", "surge", "gain", "up", "rise", "high", "bullish",
            "growth", "profit", "beat", "upgrade", "buy", "strong",
            "outperform", "breakout", "record", "positive", "boom",
        }
        bearish_words = {
            "crash", "fall", "drop", "down", "low", "bearish",
            "loss", "decline", "miss", "downgrade", "sell", "weak",
            "underperform", "breakdown", "negative", "slump", "plunge",
        }

        words = set(re.findall(r'\w+', text_lower))
        bull_count = len(words & bullish_words)
        bear_count = len(words & bearish_words)
        total = max(bull_count + bear_count, 1)

        compound = (bull_count - bear_count) / total
        return {
            "compound": round(compound, 4),
            "positive": round(bull_count / total, 4),
            "negative": round(bear_count / total, 4),
            "neutral": round(1 - (bull_count + bear_count) / max(len(words), 1), 4),
        }


# ── Ensemble Scorer ───────────────────────────────────────────────────────────

class EnsembleSentimentScorer:
    """
    Ensemble of FinBERT + VADER.

    Scoring pipeline:
      1. Run VADER (instant) and FinBERT (batched, ~5ms/headline) in parallel
      2. Weighted average: compound = w_finbert * finbert_compound + w_vader * vader_compound
      3. Compute confidence from model agreement
      4. Assign label from thresholds

    If FinBERT fails to load, falls back gracefully to VADER-only.
    """

    def __init__(
        self,
        finbert_weight: float = FINBERT_WEIGHT,
        vader_weight: float = VADER_WEIGHT,
    ):
        self._finbert = _FinBERTScorer()
        self._vader = _VADERScorer()
        self._w_fb = finbert_weight
        self._w_v = vader_weight
        # Normalize weights
        total = self._w_fb + self._w_v
        self._w_fb /= total
        self._w_v /= total

    def score(self, text: str) -> SentimentResult:
        """Score a single text with the ensemble."""
        results = self.score_batch([text])
        return results[0]

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        """
        Score a batch of texts with the ensemble.

        Efficient: FinBERT processes the whole batch in one forward pass.
        """
        # VADER scores (instant)
        vader_scores = [self._vader.score(t) for t in texts]

        # FinBERT scores (batched)
        finbert_scores = self._finbert.score_batch(texts)

        results = []
        for i, text in enumerate(texts):
            v = vader_scores[i]
            f = finbert_scores[i]

            vader_compound = v["compound"]

            if f is not None:
                # Full ensemble
                finbert_compound = f["compound"]
                compound = self._w_fb * finbert_compound + self._w_v * vader_compound

                # Confidence: high when both models agree in direction
                direction_agreement = (finbert_compound * vader_compound) >= 0
                abs_diff = abs(finbert_compound - vader_compound)

                if direction_agreement:
                    # Both agree → high confidence
                    confidence = 1.0 - 0.3 * min(abs_diff, 1.0)
                else:
                    # Disagree in direction → lower confidence
                    confidence = max(0.2, 0.6 - abs_diff)

                # Use FinBERT's class probabilities for the detailed breakdown
                pos = f["positive"]
                neg = f["negative"]
                neu = f["neutral"]
                method = "ensemble"
            else:
                # VADER-only fallback
                compound = vader_compound
                finbert_compound = 0.0
                confidence = 0.5  # lower default confidence for single model
                pos = v["positive"]
                neg = v["negative"]
                neu = v["neutral"]
                method = "vader_only"

            # Label from ensemble compound
            if compound >= BULLISH_THRESHOLD:
                label = "bullish"
            elif compound <= BEARISH_THRESHOLD:
                label = "bearish"
            else:
                label = "neutral"

            results.append(SentimentResult(
                compound=compound,
                label=label,
                finbert_score=finbert_compound if f else 0.0,
                vader_score=vader_compound,
                confidence=confidence,
                positive=pos,
                negative=neg,
                neutral_prob=neu,
                method=method,
            ))

        return results


# Singleton — lazy-initialized on first use
_ensemble: Optional[EnsembleSentimentScorer] = None


def get_scorer() -> EnsembleSentimentScorer:
    """Get or create the ensemble scorer singleton."""
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleSentimentScorer()
    return _ensemble


# ── RSS Feed URLs & Fetchers ──────────────────────────────────────────────────

# Company name mapping for better Google News search
COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries",
    "TCS": "TCS Tata Consultancy",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited",
    "SBIN": "State Bank India SBI",
    "BHARTIARTL": "Bharti Airtel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen Toubro",
    "HCLTECH": "HCL Technologies",
    "AXISBANK": "Axis Bank",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma",
    "TITAN": "Titan Company",
    "BAJFINANCE": "Bajaj Finance",
    "WIPRO": "Wipro",
    "NESTLEIND": "Nestle India",
    "TATAMOTORS": "Tata Motors",
    "ADANIENT": "Adani Enterprises",
    "POWERGRID": "Power Grid",
    "NTPC": "NTPC Limited",
    "ULTRACEMCO": "UltraTech Cement",
    "COALINDIA": "Coal India",
    "JSWSTEEL": "JSW Steel",
    "TATASTEEL": "Tata Steel",
    "ONGC": "ONGC",
    "BPCL": "Bharat Petroleum",
    "CIPLA": "Cipla",
    "DRREDDY": "Dr Reddys Laboratories",
    "HEROMOTOCO": "Hero MotoCorp",
    "EICHERMOT": "Eicher Motors",
    "APOLLOHOSP": "Apollo Hospitals",
    "M&M": "Mahindra and Mahindra",
    "VEDL": "Vedanta Limited",
    "DLF": "DLF Limited",
    "HINDALCO": "Hindalco Industries",
}


def _parse_rss(url: str, max_items: int = 20) -> list[dict]:
    """Parse an RSS feed and return list of news items."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ARTHA/2.0)",
        })
        resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)
        items = []

        for item in root.iter("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            description = item.findtext("description", "")
            source = item.findtext("source", "")

            # Clean HTML from description
            description = re.sub(r'<[^>]+>', '', description).strip()

            # Parse date
            parsed_date = ""
            if pub_date:
                for fmt in [
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]:
                    try:
                        dt = datetime.strptime(pub_date.strip(), fmt)
                        parsed_date = dt.strftime("%Y-%m-%d %H:%M")
                        break
                    except ValueError:
                        continue
                if not parsed_date:
                    parsed_date = pub_date[:16]

            if title:
                items.append({
                    "title": title,
                    "link": link,
                    "date": parsed_date,
                    "description": description[:300] if description else "",
                    "source": source or "",
                })

            if len(items) >= max_items:
                break

        return items

    except Exception as e:
        logger.warning("RSS feed fetch failed (%s): %s", url[:60], e)
        return []


def fetch_news(symbol: str, max_items: int = 25) -> list[dict]:
    """
    Fetch financial news for a symbol from multiple sources.
    Deduplicates by title similarity, scores with ensemble sentiment.
    """
    # Check cache
    cache_key = f"news_{symbol}"
    now = time.time()
    if cache_key in _news_cache and (now - _news_cache[cache_key].get("_ts", 0)) < _CACHE_TTL:
        return _news_cache[cache_key]["data"]

    all_items = []
    query = COMPANY_NAMES.get(symbol.upper(), symbol)

    # Source 1: Google News
    google_items = _parse_rss(GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+")))
    for item in google_items:
        item["source"] = item.get("source") or "Google News"
    all_items.extend(google_items)

    # Source 2: Yahoo Finance
    yahoo_items = _parse_rss(YAHOO_FINANCE_RSS.format(ticker=f"{symbol}.NS"))
    for item in yahoo_items:
        item["source"] = item.get("source") or "Yahoo Finance"
    all_items.extend(yahoo_items)

    # Source 3: ET Markets (general — filter to relevant items)
    et_items = _parse_rss(ET_MARKETS_RSS, max_items=10)
    sym_upper = symbol.upper()
    company_lower = query.lower()
    et_filtered = [
        item for item in et_items
        if sym_upper.lower() in (item.get("title", "") + item.get("description", "")).lower()
        or company_lower in (item.get("title", "") + item.get("description", "")).lower()
    ]
    for item in et_filtered:
        item["source"] = item.get("source") or "Economic Times"
    all_items.extend(et_filtered)

    # Deduplicate by title similarity
    seen_titles = set()
    unique_items = []
    for item in all_items:
        norm_title = re.sub(r'[^\w\s]', '', item["title"].lower()).strip()[:60]
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_items.append(item)

    # Score with ensemble (batched for FinBERT efficiency)
    if unique_items:
        texts = [f"{item['title']}. {item.get('description', '')}" for item in unique_items]
        scorer = get_scorer()
        scores = scorer.score_batch(texts)
        for item, score in zip(unique_items, scores):
            item["sentiment"] = score.to_dict()

    # Sort by date (newest first), then limit
    unique_items.sort(key=lambda x: x.get("date", ""), reverse=True)
    result = unique_items[:max_items]

    # Cache
    _news_cache[cache_key] = {"data": result, "_ts": now}
    return result


def get_sentiment_summary(news_items: list[dict]) -> dict:
    """
    Aggregate sentiment across all news items.

    Returns:
        {
            overall: "bullish"|"bearish"|"neutral",
            score: float,  # ensemble compound average
            distribution: {bullish: int, bearish: int, neutral: int},
            avg_compound: float,
            avg_confidence: float,
            method: str,  # "ensemble" or "vader_only"
            total_articles: int,
        }
    """
    if not news_items:
        return {
            "overall": "neutral",
            "score": 0.0,
            "distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "avg_compound": 0.0,
            "avg_confidence": 0.0,
            "method": "none",
            "total_articles": 0,
        }

    dist = {"bullish": 0, "bearish": 0, "neutral": 0}
    compounds = []
    confidences = []
    methods = set()

    for item in news_items:
        sentiment = item.get("sentiment", {})
        label = sentiment.get("label", "neutral")
        compound = sentiment.get("compound", 0)
        confidence = sentiment.get("confidence", 0.5)
        method = sentiment.get("method", "unknown")

        dist[label] = dist.get(label, 0) + 1
        compounds.append(compound)
        confidences.append(confidence)
        methods.add(method)

    avg_compound = sum(compounds) / len(compounds) if compounds else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    if avg_compound >= 0.1:
        overall = "bullish"
    elif avg_compound <= -0.1:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": round(avg_compound, 4),
        "distribution": dist,
        "avg_compound": round(avg_compound, 4),
        "avg_confidence": round(avg_confidence, 4),
        "method": "ensemble" if "ensemble" in methods else "vader_only",
        "total_articles": len(news_items),
    }


# ── CLI for testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"Fetching news for {symbol} (ensemble: FinBERT + VADER)...")

    items = fetch_news(symbol)
    summary = get_sentiment_summary(items)

    print(f"\n{'='*70}")
    print(f"  {symbol} — Ensemble Sentiment Summary")
    print(f"{'='*70}")
    print(f"  Overall:    {summary['overall'].upper()}")
    print(f"  Score:      {summary['score']:.4f}")
    print(f"  Confidence: {summary['avg_confidence']:.4f}")
    print(f"  Method:     {summary['method']}")
    print(f"  Articles:   {summary['total_articles']}")
    print(f"  Distribution: {json.dumps(summary['distribution'])}")
    print(f"{'='*70}")

    print(f"\nLatest articles:")
    for i, item in enumerate(items[:10], 1):
        s = item.get("sentiment", {})
        emoji = "🟢" if s.get("label") == "bullish" else "🔴" if s.get("label") == "bearish" else "⚪"
        conf_bar = "█" * int(s.get("confidence", 0) * 10)
        print(f"  {emoji} [{item['date']}] {item['title'][:75]}")
        print(f"     Ensemble: {s.get('compound', 0):.3f}  "
              f"FinBERT: {s.get('finbert_score', 0):.3f}  "
              f"VADER: {s.get('vader_score', 0):.3f}  "
              f"Conf: {conf_bar}  |  {item['source']}")
