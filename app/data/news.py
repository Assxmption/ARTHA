"""
ARTHA News & Sentiment Module
================================
Fetches ticker-scoped financial news and performs sentiment analysis.

Sources:
  1. Google News RSS (free, no key needed)
  2. Yahoo Finance RSS
  3. MoneyControl / Economic Times RSS

Sentiment:
  - Uses VADER for fast sentiment scoring (no GPU needed)
  - Aggregates into bullish/bearish/neutral distribution

Reference: docs/ARTHA_ARCHITECTURE.md §4.4
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger("artha.data.news")

# ── RSS Feed URLs ──────────────────────────────────────────────────────────────

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"
ET_MARKETS_RSS = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
MC_RSS = "https://www.moneycontrol.com/rss/latestnews.xml"

# Cache
_news_cache: dict = {}
_CACHE_TTL = 600  # 10 minutes


# ── Sentiment Analyzer ────────────────────────────────────────────────────────

class _SentimentAnalyzer:
    """
    VADER-based financial sentiment. Falls back to keyword matching if
    VADER is not installed.
    """
    
    def __init__(self):
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            
            # Add finance-specific lexicon updates
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
            }
            self._vader.lexicon.update(finance_lexicon)
            logger.info("VADER sentiment analyzer loaded with finance lexicon")
        except ImportError:
            logger.warning("vaderSentiment not installed — using keyword fallback")
    
    def score(self, text: str) -> dict:
        """
        Analyze sentiment of text.
        
        Returns:
            {compound: float[-1,1], pos: float, neg: float, neu: float, label: str}
        """
        if self._vader:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.15:
                label = "bullish"
            elif compound <= -0.15:
                label = "bearish"
            else:
                label = "neutral"
            return {
                "compound": round(compound, 4),
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "label": label,
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
        
        if compound > 0.1:
            label = "bullish"
        elif compound < -0.1:
            label = "bearish"
        else:
            label = "neutral"
        
        return {
            "compound": round(compound, 4),
            "positive": round(bull_count / total, 4),
            "negative": round(bear_count / total, 4),
            "neutral": round(1 - (bull_count + bear_count) / max(len(words), 1), 4),
            "label": label,
        }


_analyzer = _SentimentAnalyzer()


# ── News Fetchers ──────────────────────────────────────────────────────────────

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
    Deduplicates by title similarity.
    """
    # Check cache
    cache_key = f"news_{symbol}"
    now = time.time()
    if cache_key in _news_cache and (now - _news_cache[cache_key].get("_ts", 0)) < _CACHE_TTL:
        return _news_cache[cache_key]["data"]
    
    all_items = []
    
    # Company name mapping for better search
    company_names = {
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
    }
    
    query = company_names.get(symbol.upper(), symbol)
    
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
    
    # Source 3: ET Markets (general)
    et_items = _parse_rss(ET_MARKETS_RSS, max_items=10)
    # Filter to relevant items only
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
        # Normalize title for dedup
        norm_title = re.sub(r'[^\w\s]', '', item["title"].lower()).strip()[:60]
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_items.append(item)
    
    # Add sentiment to each item
    for item in unique_items:
        text = f"{item['title']}. {item.get('description', '')}"
        item["sentiment"] = _analyzer.score(text)
    
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
            score: float,
            distribution: {bullish: int, bearish: int, neutral: int},
            avg_compound: float,
        }
    """
    if not news_items:
        return {
            "overall": "neutral",
            "score": 0.0,
            "distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "avg_compound": 0.0,
            "total_articles": 0,
        }
    
    dist = {"bullish": 0, "bearish": 0, "neutral": 0}
    compounds = []
    
    for item in news_items:
        sentiment = item.get("sentiment", {})
        label = sentiment.get("label", "neutral")
        compound = sentiment.get("compound", 0)
        
        dist[label] = dist.get(label, 0) + 1
        compounds.append(compound)
    
    avg_compound = sum(compounds) / len(compounds) if compounds else 0
    
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
        "total_articles": len(news_items),
    }


# ── CLI for testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"Fetching news for {symbol}...")
    
    items = fetch_news(symbol)
    summary = get_sentiment_summary(items)
    
    print(f"\n{'='*60}")
    print(f"  {symbol} — News Sentiment Summary")
    print(f"{'='*60}")
    print(f"  Overall: {summary['overall'].upper()}")
    print(f"  Score:   {summary['score']:.4f}")
    print(f"  Articles: {summary['total_articles']}")
    print(f"  Distribution: {json.dumps(summary['distribution'])}")
    print(f"{'='*60}")
    
    print(f"\nLatest articles:")
    for i, item in enumerate(items[:10], 1):
        emoji = "🟢" if item["sentiment"]["label"] == "bullish" else "🔴" if item["sentiment"]["label"] == "bearish" else "⚪"
        print(f"  {emoji} [{item['date']}] {item['title'][:80]}")
        print(f"     Score: {item['sentiment']['compound']:.3f} | Source: {item['source']}")
