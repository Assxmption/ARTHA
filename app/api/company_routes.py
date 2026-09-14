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
            "current": _safe_round(current_price),
            "previousClose": _safe_round(prev_close),
            "change": _safe_round(current_price - prev_close),
            "changePct": _safe_round((current_price / prev_close - 1) * 100) if prev_close else 0,
            "dayHigh": _safe_round(float(hist['High'].iloc[-1])),
            "dayLow": _safe_round(float(hist['Low'].iloc[-1])),
            "volume": int(hist['Volume'].iloc[-1]) if not np.isnan(hist['Volume'].iloc[-1]) else 0,
            "fiftyTwoWeekHigh": _safe_round(float(hist['High'].max())),
            "fiftyTwoWeekLow": _safe_round(float(hist['Low'].min())),
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
            "price": _safe_round(current),
            "change": _safe_round(current - prev),
            "changePct": _safe_round((current / prev - 1) * 100) if prev else 0,
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

@router.get("/api/search")
async def search_symbols(q: str):
    """
    Search for stock symbols using Yahoo Finance search API.
    Filters for NSE and BSE stocks.
    """
    if not q or len(q) < 1:
        return {"results": []}
        
    import httpx
    
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=10&newsCount=0"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            quotes = data.get("quotes", [])
            # Filter for Indian exchanges (NSI for NSE, BSE for Bombay Stock Exchange)
            indian_stocks = [
                {
                    "symbol": q.get("symbol", "").replace(".NS", "").replace(".BO", ""),
                    "name": q.get("shortname") or q.get("longname", ""),
                    "exchange": "NSE" if q.get("exchange") == "NSI" else "BSE",
                    "type": q.get("quoteType", "EQUITY"),
                }
                for q in quotes
                if q.get("exchange") in ["NSI", "BSE"] and q.get("quoteType") in ["EQUITY", "ETF", "MUTUALFUND"]
            ]
            
            # Deduplicate by symbol (prefer NSE over BSE)
            seen = set()
            deduped = []
            
            # Local fallback for 1-2 letter searches (Yahoo global search favors US stocks)
            LOCAL_TOP_STOCKS = [
                {"symbol": "RELIANCE", "name": "Reliance Industries", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "HDFCBANK", "name": "HDFC Bank", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "INFY", "name": "Infosys", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "ICICIBANK", "name": "ICICI Bank", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "ITC", "name": "ITC Limited", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "MASTEK", "name": "Mastek Limited", "exchange": "NSE", "type": "EQUITY"},
                {"symbol": "TATAMOTORS", "name": "Tata Motors", "exchange": "NSE", "type": "EQUITY"}
            ]
            
            q_lower = q.lower()
            for stock in LOCAL_TOP_STOCKS:
                if q_lower in stock["symbol"].lower() or q_lower in stock["name"].lower():
                    seen.add(stock["symbol"])
                    deduped.append(stock)
            
            for stock in indian_stocks:
                if stock["symbol"] not in seen:
                    seen.add(stock["symbol"])
                    deduped.append(stock)
            
            # Simple heuristic scoring for better ranking
            def score_stock(s):
                sym = s["symbol"].lower()
                name = s["name"].lower()
                
                if sym == q_lower: return 100
                if sym.startswith(q_lower): return 80
                if name.startswith(q_lower): return 60
                
                # Check if any word in the name starts with the query
                words = name.split()
                if any(w.startswith(q_lower) for w in words): return 40
                
                if q_lower in sym: return 10
                if q_lower in name: return 5
                return 0
                
            deduped.sort(key=score_stock, reverse=True)
            
            return {"results": deduped[:8]}
    except Exception as e:
        logger.error("Search failed for %s: %s", q, e)
        return {"results": []}


# ── Multi-company batch endpoint ───────────────────────────────────────────────

@router.get("/api/watchlist")
def get_watchlist(
    symbols: str = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL",
):
    """
    Batch quick view for multiple symbols (comma-separated).
    Returns lightweight snapshots for a watchlist.
    """
    import yfinance as yf
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
    
    def fetch_symbol(sym):
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            hist = ticker.history(period="5d").dropna(subset=["Close"])
            
            if len(hist) >= 2:
                c1, c2 = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
                if pd.isna(c1) or pd.isna(c2):
                    price = change = change_pct = None
                else:
                    price = round(float(c1), 2)
                    prev = round(float(c2), 2)
                    change = round(price - prev, 2)
                    change_pct = round((change / prev) * 100, 2) if prev != 0 else 0.0
            elif len(hist) == 1:
                c1 = hist["Close"].iloc[0]
                if pd.isna(c1):
                    price = change = change_pct = None
                else:
                    price = round(float(c1), 2)
                    change = 0.0
                    change_pct = 0.0
            else:
                price = change = change_pct = None

            try:
                info = ticker.info
            except Exception:
                info = {}

            volume = None
            if len(hist) > 0 and "Volume" in hist.columns:
                try:
                    volume_val = hist["Volume"].iloc[-1]
                    if pd.notna(volume_val):
                        volume = int(volume_val)
                except Exception:
                    pass

            return {
                "symbol": sym,
                "name": info.get("longName") or info.get("shortName") or sym,
                "price": price,
                "change": change,
                "changePct": change_pct,
                "volume": volume,
                "volumeFormatted": _format_large_number(volume, prefix="") if volume else "—",
                "marketCap": _format_large_number(info.get("marketCap")),
                "pe": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
                "sector": info.get("sector") or "Unknown"
            }
        except Exception as e:
            logger.warning("Watchlist fetch failed for %s: %s", sym, e)
            return {"symbol": sym, "error": str(e)}

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sym = {executor.submit(fetch_symbol, sym): sym for sym in sym_list}
        # Keep original order
        res_map = {}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            res_map[sym] = future.result()
            
        for sym in sym_list:
            results.append(res_map[sym])
    
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


def _format_large_number(val, prefix="₹") -> str:
    """Format large numbers as prefix + X.XX Cr or L."""
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    
    if val >= 1e12:
        return f"{prefix}{val / 1e12:.2f}T"
    elif val >= 1e7:
        return f"{prefix}{val / 1e7:.2f} Cr"
    elif val >= 1e5:
        return f"{prefix}{val / 1e5:.2f} L"
    elif val >= 1000:
        return f"{prefix}{val / 1000:.1f}K"
    else:
        return f"{prefix}{val:.0f}"


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
