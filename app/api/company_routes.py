"""
ARTHA Company View & News API Routes
=======================================
Single-call endpoints for instant company analysis and news.

Endpoints:
  GET  /api/company/{symbol}       → Full company view (fundamentals + chart + analysis)
  GET  /api/company/{symbol}/quick → Lightweight price + key metrics only
  GET  /api/news/{symbol}          → News with sentiment analysis
  GET  /api/news/{symbol}/summary  → Aggregated sentiment summary

These are the "zero-wait" endpoints — they return instantly
using cached data or fast live fetches (no LLM calls).

Reference: docs/ARTHA_ARCHITECTURE.md §4.7
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("artha.api.company")
router = APIRouter(tags=["Company"])

# Cache for company data (avoid re-fetching within 5 minutes)
_company_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


# ── Company View ───────────────────────────────────────────────────────────────

@router.get("/api/company/{symbol}")
async def get_company_view(symbol: str, days: int = 180):
    """
    Instant company view — single call returns everything the UI needs.
    
    Returns:
      - Company info (name, sector, market cap)
      - Current price + change
      - Key fundamental metrics (PE, EPS, ROE, debt-to-equity, etc.)
      - Price history (OHLCV) for charting
      - SMA crossover signals
      - News with sentiment
      
    All data is fetched live but cached for 5 minutes.
    No LLM calls — this is purely data-driven.
    """
    symbol = symbol.upper().strip()
    
    # Check cache
    cache_key = f"company_{symbol}_{days}"
    now = time.time()
    if cache_key in _company_cache and (now - _company_cache[cache_key].get("_ts", 0)) < _CACHE_TTL:
        return _company_cache[cache_key]["data"]
    
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        
        # Basic info
        company_info = {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": (info.get("longBusinessSummary", "") or "")[:500],
            "website": info.get("website", ""),
            "exchange": "NSE",
            "currency": "INR",
        }
        
        # Price data
        hist = ticker.history(period=f"{days}d")
        if hist.empty:
            raise HTTPException(404, f"No data found for {symbol}")
        
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        
        price_data = {
            "current": round(current_price, 2),
            "previousClose": round(prev_close, 2),
            "change": round(current_price - prev_close, 2),
            "changePct": round((current_price / prev_close - 1) * 100, 2),
            "dayHigh": round(float(hist['High'].iloc[-1]), 2),
            "dayLow": round(float(hist['Low'].iloc[-1]), 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "fiftyTwoWeekHigh": round(float(hist['High'].max()), 2),
            "fiftyTwoWeekLow": round(float(hist['Low'].min()), 2),
        }
        
        # Key fundamentals from yfinance info
        fundamentals = {
            "marketCap": _format_large_number(info.get("marketCap", 0)),
            "marketCapRaw": info.get("marketCap", 0),
            "pe": _safe_round(info.get("trailingPE")),
            "forwardPE": _safe_round(info.get("forwardPE")),
            "eps": _safe_round(info.get("trailingEps")),
            "bookValue": _safe_round(info.get("bookValue")),
            "priceToBook": _safe_round(info.get("priceToBook")),
            "dividendYield": _safe_round(info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0),
            "roe": _safe_round(info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0),
            "roa": _safe_round(info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else 0),
            "debtToEquity": _safe_round(info.get("debtToEquity")),
            "revenue": _format_large_number(info.get("totalRevenue", 0)),
            "revenueGrowth": _safe_round(info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0),
            "profitMargin": _safe_round(info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0),
            "operatingMargin": _safe_round(info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else 0),
            "freeCashFlow": _format_large_number(info.get("freeCashflow", 0)),
            "beta": _safe_round(info.get("beta")),
        }
        
        # OHLCV chart data
        close = hist['Close']
        sma10 = close.rolling(10).mean()
        sma21 = close.rolling(21).mean()
        sma50 = close.rolling(50).mean()
        
        chart_data = []
        for i, (dt, row) in enumerate(hist.iterrows()):
            entry = {
                "time": int(dt.timestamp()),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume']),
            }
            
            # Add SMAs if available
            if i >= 9 and not np.isnan(sma10.iloc[i]):
                entry["sma10"] = round(float(sma10.iloc[i]), 2)
            if i >= 20 and not np.isnan(sma21.iloc[i]):
                entry["sma21"] = round(float(sma21.iloc[i]), 2)
            if i >= 49 and not np.isnan(sma50.iloc[i]):
                entry["sma50"] = round(float(sma50.iloc[i]), 2)
            
            # Signal detection
            signal = 0
            if i >= 21:
                if sma10.iloc[i] > sma21.iloc[i] and sma10.iloc[i-1] <= sma21.iloc[i-1]:
                    signal = 1  # BUY
                elif sma10.iloc[i] < sma21.iloc[i] and sma10.iloc[i-1] >= sma21.iloc[i-1]:
                    signal = -1  # SELL
            entry["signal"] = signal
            
            chart_data.append(entry)
        
        # Technical snapshot
        rsi_14 = _compute_rsi(close, 14)
        technicals = {
            "sma10": round(float(sma10.iloc[-1]), 2) if not np.isnan(sma10.iloc[-1]) else None,
            "sma21": round(float(sma21.iloc[-1]), 2) if not np.isnan(sma21.iloc[-1]) else None,
            "sma50": round(float(sma50.iloc[-1]), 2) if not np.isnan(sma50.iloc[-1]) else None,
            "rsi14": round(float(rsi_14), 2) if rsi_14 is not None else None,
            "trend": "bullish" if sma10.iloc[-1] > sma21.iloc[-1] else "bearish",
            "above50SMA": bool(current_price > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        }
        
        # News + sentiment
        from app.data.news import fetch_news, get_sentiment_summary
        news_items = fetch_news(symbol, max_items=10)
        sentiment = get_sentiment_summary(news_items)
        
        result = {
            "company": company_info,
            "price": price_data,
            "fundamentals": fundamentals,
            "technicals": technicals,
            "sentiment": sentiment,
            "news": news_items[:5],  # Top 5 for the card view
            "chart": chart_data,
            "lastUpdated": datetime.now().isoformat(),
        }
        
        # Cache
        _company_cache[cache_key] = {"data": result, "_ts": now}
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Company view failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(500, f"Failed to fetch data for {symbol}: {e}")


@router.get("/api/company/{symbol}/quick")
async def get_company_quick(symbol: str):
    """
    Lightweight company snapshot — just price and key metrics.
    Optimized for list views and watchlists.
    """
    symbol = symbol.upper().strip()
    
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="2d")
        
        if hist.empty:
            raise HTTPException(404, f"No data for {symbol}")
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        
        return {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", symbol)),
            "price": round(current, 2),
            "change": round(current - prev, 2),
            "changePct": round((current / prev - 1) * 100, 2),
            "marketCap": _format_large_number(info.get("marketCap", 0)),
            "pe": _safe_round(info.get("trailingPE")),
            "volume": int(hist['Volume'].iloc[-1]),
            "sector": info.get("sector", "N/A"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── News Endpoints ─────────────────────────────────────────────────────────────

@router.get("/api/news/{symbol}")
async def get_news(symbol: str, limit: int = 20):
    """
    Get news articles for a symbol with sentiment analysis.
    
    Each article includes:
      - Title, link, date, source
      - Sentiment score (compound, label)
    """
    from app.data.news import fetch_news, get_sentiment_summary
    
    symbol = symbol.upper().strip()
    items = fetch_news(symbol, max_items=limit)
    summary = get_sentiment_summary(items)
    
    return {
        "symbol": symbol,
        "sentiment": summary,
        "articles": items,
        "total": len(items),
    }


@router.get("/api/news/{symbol}/summary")
async def get_news_sentiment_summary(symbol: str):
    """Get just the aggregated sentiment summary for a symbol."""
    from app.data.news import fetch_news, get_sentiment_summary
    
    symbol = symbol.upper().strip()
    items = fetch_news(symbol, max_items=25)
    summary = get_sentiment_summary(items)
    
    return {
        "symbol": symbol,
        **summary,
    }


# ── Multi-company batch endpoint ───────────────────────────────────────────────

@router.get("/api/watchlist")
async def get_watchlist(
    symbols: str = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL",
):
    """
    Batch quick view for multiple symbols (comma-separated).
    Returns lightweight snapshots for a watchlist.
    """
    import yfinance as yf
    
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
    
    results = []
    for sym in sym_list:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            info = ticker.info or {}
            hist = ticker.history(period="2d")
            
            if hist.empty:
                continue
            
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
            
            results.append({
                "symbol": sym,
                "name": info.get("longName", info.get("shortName", sym)),
                "price": round(current, 2),
                "change": round(current - prev, 2),
                "changePct": round((current / prev - 1) * 100, 2),
                "marketCap": _format_large_number(info.get("marketCap", 0)),
                "pe": _safe_round(info.get("trailingPE")),
                "sector": info.get("sector", "N/A"),
            })
        except Exception as e:
            logger.warning("Watchlist fetch failed for %s: %s", sym, e)
            results.append({"symbol": sym, "error": str(e)})
    
    return {"stocks": results, "total": len(results)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_round(val, digits: int = 2):
    """Safely round a value that might be None."""
    if val is None or val != val:  # NaN check
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _format_large_number(val) -> str:
    """Format large numbers as ₹X.XX Cr or ₹X.XX L."""
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    
    if val >= 1e12:
        return f"₹{val / 1e12:.2f}T"
    elif val >= 1e7:
        return f"₹{val / 1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹{val / 1e5:.2f} L"
    elif val >= 1000:
        return f"₹{val / 1000:.1f}K"
    else:
        return f"₹{val:.0f}"


def _compute_rsi(prices, period: int = 14):
    """Compute RSI for a price series."""
    import numpy as np
    
    if len(prices) < period + 1:
        return None
    
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    
    if last_loss == 0 or np.isnan(last_loss):
        return 100.0
    
    rs = last_gain / last_loss
    return 100 - (100 / (1 + rs))
