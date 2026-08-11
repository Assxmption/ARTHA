"""
NSE F&O Data — Real Options Chain + Futures Data
==================================================
Fetches real NSE India options chain and futures data.

Strategy:
  1. Primary: nselib — purpose-built NSE scraper (MIT license)
  2. Fallback: Direct NSE API with proper headers
  3. Last resort: nsetools for basic data

Reference: docs/ARTHA_ARCHITECTURE.md §4.2 (data layer)
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
import json

logger = logging.getLogger("artha.data.nse_fno")

# NSE API endpoints
NSE_BASE = "https://www.nseindia.com"
NSE_OPTION_CHAIN = f"{NSE_BASE}/api/option-chain-equities"
NSE_INDEX_OPTION_CHAIN = f"{NSE_BASE}/api/option-chain-indices"

# Headers to mimic browser — NSE blocks raw API calls
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
}

# Cache for session cookies and data
_session_cache: dict = {}
_data_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


def _get_nse_session() -> requests.Session:
    """
    Get an authenticated NSE session with cookies.
    NSE requires visiting the main page first to get cookies.
    """
    now = time.time()
    if "_session" in _session_cache and (now - _session_cache.get("_ts", 0)) < 600:
        return _session_cache["_session"]
    
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    
    # Visit main page to get cookies
    try:
        resp = session.get(NSE_BASE, timeout=10)
        resp.raise_for_status()
        logger.info("NSE session established (cookies: %d)", len(session.cookies))
    except Exception as e:
        logger.warning("Failed to establish NSE session: %s", e)
    
    _session_cache["_session"] = session
    _session_cache["_ts"] = now
    return session


def fetch_option_chain_nse_direct(symbol: str) -> Optional[dict]:
    """
    Fetch option chain directly from NSE India API.
    
    Returns the raw NSE option chain response with real OI, volumes, IV, etc.
    """
    cache_key = f"oc_{symbol}"
    now = time.time()
    if cache_key in _data_cache and (now - _data_cache[cache_key].get("_ts", 0)) < _CACHE_TTL:
        return _data_cache[cache_key]["data"]
    
    session = _get_nse_session()
    
    # Indices use a different endpoint
    index_symbols = {"NIFTY", "BANKNIFTY", "NIFTY BANK", "FINNIFTY", "MIDCPNIFTY"}
    if symbol.upper() in index_symbols:
        url = NSE_INDEX_OPTION_CHAIN
    else:
        url = NSE_OPTION_CHAIN
    
    try:
        resp = session.get(url, params={"symbol": symbol.upper()}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if "records" not in data:
            logger.warning("NSE option chain response missing 'records' for %s", symbol)
            return None
        
        _data_cache[cache_key] = {"data": data, "_ts": now}
        return data
    
    except requests.exceptions.HTTPError as e:
        logger.warning("NSE API HTTP error for %s: %s", symbol, e)
        return None
    except Exception as e:
        logger.error("NSE option chain fetch failed for %s: %s", symbol, e)
        return None


def fetch_option_chain_nselib(symbol: str) -> Optional[dict]:
    """
    Fetch option chain using the nselib library.
    
    Returns data in a normalized format matching NSE's API structure.
    """
    try:
        from nselib import derivatives
        
        # Try the new 2.5.1 method
        try:
            data = derivatives.nse_live_option_chain(symbol.upper())
        except AttributeError:
            # Fallbacks for other versions
            try:
                from nselib import capital_market
                data = capital_market.option_chain(symbol.upper())
            except AttributeError:
                data = derivatives.option_chain(symbol.upper())
        
        if data is not None and not data.empty:
            return _normalize_nselib_data(data, symbol)
        
        return None
    except ImportError:
        logger.info("nselib not installed, skipping")
        return None
    except Exception as e:
        logger.warning("nselib option chain failed for %s: %s", symbol, e)
        return None


def _normalize_nselib_data(df, symbol: str) -> dict:
    """Convert nselib DataFrame to our standard format."""
    import pandas as pd
    
    # nselib returns a DataFrame with columns like:
    # CE OI, CE Chng in OI, CE Volume, CE IV, CE LTP, CE Net Chng, Strike Price,
    # PE LTP, PE Net Chng, PE Volume, PE OI, PE Chng in OI, PE IV
    
    records = []
    for _, row in df.iterrows():
        record = {
            "strikePrice": row.get("Strike Price", row.get("strikePrice", 0)),
            "CE": {},
            "PE": {},
        }
        
        # Call data
        if "CE LTP" in row or "CE_LTP" in row:
            record["CE"] = {
                "lastPrice": row.get("CE LTP", row.get("CE_LTP", 0)),
                "openInterest": row.get("CE OI", row.get("CE_OI", 0)),
                "changeinOpenInterest": row.get("CE Chng in OI", row.get("CE_changeinOI", 0)),
                "totalTradedVolume": row.get("CE Volume", row.get("CE_Volume", 0)),
                "impliedVolatility": row.get("CE IV", row.get("CE_IV", 0)),
                "change": row.get("CE Net Chng", row.get("CE_change", 0)),
            }
        
        # Put data
        if "PE LTP" in row or "PE_LTP" in row:
            record["PE"] = {
                "lastPrice": row.get("PE LTP", row.get("PE_LTP", 0)),
                "openInterest": row.get("PE OI", row.get("PE_OI", 0)),
                "changeinOpenInterest": row.get("PE Chng in OI", row.get("PE_changeinOI", 0)),
                "totalTradedVolume": row.get("PE Volume", row.get("PE_Volume", 0)),
                "impliedVolatility": row.get("PE IV", row.get("PE_IV", 0)),
                "change": row.get("PE Net Chng", row.get("PE_change", 0)),
            }
        
        records.append(record)
    
    return {
        "records": {
            "data": records,
            "strikePrices": sorted(set(r["strikePrice"] for r in records)),
        },
        "filtered": {
            "data": records,
        },
    }


def get_option_chain(symbol: str) -> dict:
    """
    Get option chain with fallback strategy:
    1. nselib (primary)
    2. Direct NSE API (fallback)
    3. Structured empty response (last resort)
    
    Returns a normalized dict with calls, puts, spot price, expiry info.
    """
    raw = None
    source = "none"
    
    # Try nselib first
    raw = fetch_option_chain_nselib(symbol)
    if raw:
        source = "nselib"
    
    # Fallback to direct NSE
    if not raw:
        raw = fetch_option_chain_nse_direct(symbol)
        if raw:
            source = "nse_direct"
    
    if not raw:
        logger.warning(f"All real option chain fetches failed for {symbol}. Using mock fallback.")
        return _generate_mock_option_chain(symbol)
    
    return _format_option_chain(raw, symbol, source)


def _format_option_chain(raw: dict, symbol: str, source: str) -> dict:
    """
    Format NSE option chain data into a clean frontend-friendly structure.
    """
    records = raw.get("records", raw.get("filtered", {}))
    data = records.get("data", [])
    
    if not data:
        return {
            "symbol": symbol.upper(),
            "spotPrice": 0,
            "expiry": "",
            "calls": [],
            "puts": [],
            "source": source,
        }
    
    # Get spot price from records metadata
    spot = 0
    underlying_value = records.get("underlyingValue", 0)
    if underlying_value:
        spot = float(underlying_value)
    
    # Get expiry dates
    expiry_dates = records.get("expiryDates", [])
    nearest_expiry = expiry_dates[0] if expiry_dates else ""
    
    calls = []
    puts = []
    
    for record in data:
        strike = record.get("strikePrice", 0)
        
        # Filter to nearest expiry only (or all if no expiry info)
        if nearest_expiry and record.get("expiryDate", nearest_expiry) != nearest_expiry:
            continue
        
        ce = record.get("CE", {})
        pe = record.get("PE", {})
        
        if ce:
            calls.append({
                "strike": float(strike),
                "lastPrice": float(ce.get("lastPrice", 0)),
                "volume": int(ce.get("totalTradedVolume", 0)),
                "openInterest": int(ce.get("openInterest", 0)),
                "changeinOI": int(ce.get("changeinOpenInterest", 0)),
                "impliedVolatility": float(ce.get("impliedVolatility", 0)),
                "change": float(ce.get("change", 0)),
                "bidPrice": float(ce.get("bidprice", 0)),
                "askPrice": float(ce.get("askPrice", 0)),
            })
        
        if pe:
            puts.append({
                "strike": float(strike),
                "lastPrice": float(pe.get("lastPrice", 0)),
                "volume": int(pe.get("totalTradedVolume", 0)),
                "openInterest": int(pe.get("openInterest", 0)),
                "changeinOI": int(pe.get("changeinOpenInterest", 0)),
                "impliedVolatility": float(pe.get("impliedVolatility", 0)),
                "change": float(pe.get("change", 0)),
                "bidPrice": float(pe.get("bidprice", 0)),
                "askPrice": float(pe.get("askPrice", 0)),
            })
    
    # Sort by strike
    calls.sort(key=lambda x: x["strike"])
    puts.sort(key=lambda x: x["strike"])
    
    # If no spot from metadata, estimate from ATM strike
    if spot == 0 and calls:
        mid_idx = len(calls) // 2
        spot = calls[mid_idx]["strike"]
    
    return {
        "symbol": symbol.upper(),
        "spotPrice": round(spot, 2),
        "expiry": nearest_expiry,
        "expiryDates": expiry_dates,
        "calls": calls,
        "puts": puts,
        "totalCallOI": sum(c["openInterest"] for c in calls),
        "totalPutOI": sum(p["openInterest"] for p in puts),
        "pcr": round(
            sum(p["openInterest"] for p in puts) / max(sum(c["openInterest"] for c in calls), 1),
            3
        ),
        "source": source,
    }


# ── CLI for testing ─────────────────────────────────────────────────────────────

def _generate_mock_option_chain(symbol: str) -> dict:
    import yfinance as yf
    import numpy as np
    from datetime import timedelta
    
    try:
        ticker = yf.Ticker(f"{symbol}.NS" if symbol not in ["NIFTY", "BANKNIFTY"] else ("^NSEI" if symbol == "NIFTY" else "^NSEBANK"))
        hist = ticker.history(period="1d")
        spot = float(hist['Close'].iloc[-1]) if not hist.empty else 1000.0
    except:
        spot = 1000.0
        
    strike_step = 50 if spot < 5000 else 100
    base_strike = round(spot / strike_step) * strike_step
    
    calls = []
    puts = []
    
    # Generate 10 strikes above and below ATM
    for i in range(-10, 11):
        strike = base_strike + (i * strike_step)
        
        # Simple pricing model for mock data
        distance = abs(strike - spot)
        moneyness = distance / spot
        
        # IV smile mock
        iv = 15.0 + (moneyness * 100)
        
        # Mock OI and volume (higher near ATM)
        base_oi = int(100000 * (1 - min(moneyness * 5, 0.9)))
        base_vol = int(base_oi * 0.5)
        
        # Call pricing: ITM when strike < spot
        is_call_itm = strike < spot
        call_intrinsic = max(0, spot - strike)
        call_time_value = spot * 0.02 * (1 - min(moneyness * 5, 0.9))
        call_price = call_intrinsic + call_time_value
        
        calls.append({
            "strike": float(strike),
            "lastPrice": round(call_price, 2),
            "volume": base_vol,
            "openInterest": base_oi,
            "changeinOI": int(base_oi * (np.random.random() - 0.3)), # Random bias
            "impliedVolatility": round(iv, 2),
            "change": round(call_price * (np.random.random() - 0.5) * 0.1, 2),
            "bidPrice": round(call_price * 0.99, 2),
            "askPrice": round(call_price * 1.01, 2),
        })
        
        # Put pricing: ITM when strike > spot
        put_intrinsic = max(0, strike - spot)
        put_time_value = spot * 0.02 * (1 - min(moneyness * 5, 0.9))
        put_price = put_intrinsic + put_time_value
        
        puts.append({
            "strike": float(strike),
            "lastPrice": round(put_price, 2),
            "volume": int(base_vol * 1.1), # Slight put bias
            "openInterest": int(base_oi * 1.2),
            "changeinOI": int(base_oi * (np.random.random() - 0.3)),
            "impliedVolatility": round(iv * 1.1, 2),
            "change": round(put_price * (np.random.random() - 0.5) * 0.1, 2),
            "bidPrice": round(put_price * 0.99, 2),
            "askPrice": round(put_price * 1.01, 2),
        })
        
    next_thursday = date.today()
    while next_thursday.weekday() != 3: # Thursday
        next_thursday += timedelta(days=1)
        
    return {
        "symbol": symbol.upper(),
        "spotPrice": round(spot, 2),
        "expiry": next_thursday.strftime("%d-%b-%Y"),
        "expiryDates": [next_thursday.strftime("%d-%b-%Y")],
        "calls": calls,
        "puts": puts,
        "totalCallOI": sum(c["openInterest"] for c in calls),
        "totalPutOI": sum(p["openInterest"] for p in puts),
        "pcr": round(sum(p["openInterest"] for p in puts) / max(sum(c["openInterest"] for c in calls), 1), 3),
        "source": "mock_fallback",
        "error": "Real APIs failed, using synthetic data",
    }

# ── CLI for testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"Fetching option chain for {symbol}...")
    
    result = get_option_chain(symbol)
    
    print(f"\nSource: {result.get('source', 'unknown')}")
    print(f"Spot: ₹{result.get('spotPrice', 0):,.2f}")
    print(f"Expiry: {result.get('expiry', 'N/A')}")
    print(f"Calls: {len(result.get('calls', []))}")
    print(f"Puts: {len(result.get('puts', []))}")
    print(f"Total Call OI: {result.get('totalCallOI', 0):,}")
    print(f"Total Put OI: {result.get('totalPutOI', 0):,}")
    print(f"PCR: {result.get('pcr', 0):.3f}")
    
    if result.get("calls"):
        print(f"\nTop 5 Calls by OI:")
        top_calls = sorted(result["calls"], key=lambda x: x["openInterest"], reverse=True)[:5]
        for c in top_calls:
            print(f"  Strike ₹{c['strike']:>8,.0f} | LTP ₹{c['lastPrice']:>8.2f} | OI {c['openInterest']:>10,} | IV {c['impliedVolatility']:.2f}%")
    
    if result.get("error"):
        print(f"\nError: {result['error']}")
