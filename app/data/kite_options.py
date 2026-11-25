"""
Option chain from Kite Connect (NFO instrument master + live quotes).

Replaces the NSE website scrape, which NSE blocks for automated clients.
Kite supplies prices, volume, OI and order-book depth; it does not supply
implied volatility or previous-day OI, so:
  * IV is solved here from the option's mid (or last) price with the repo's
    Black-Scholes solver — labelled `iv_method` so its origin is explicit;
  * change in OI is reported as None rather than invented.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from app.data import kite

logger = logging.getLogger(__name__)

STRIKES_EACH_SIDE = 15
_UNDERLYING_QUOTE = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK", "FINNIFTY": "NSE:NIFTY FIN SERVICE"}


def _years_to_expiry(expiry: date, now: datetime) -> float:
    expiry_close = datetime.combine(expiry, dtime(15, 30))
    seconds = (expiry_close - now).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24 * 3600)


def _mid(q: dict) -> Optional[float]:
    depth = q.get("depth") or {}
    bid = (depth.get("buy") or [{}])[0].get("price") or 0
    ask = (depth.get("sell") or [{}])[0].get("price") or 0
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    ltp = q.get("last_price")
    return float(ltp) if ltp else None


def get_option_chain_kite(symbol: str, expiry: Optional[str] = None) -> Optional[dict]:
    """Nearest-expiry (or chosen) chain around the money; None if Kite is unavailable."""
    from app.quant.options_pricing import DEFAULT_RISK_FREE_RATE, implied_volatility

    try:
        client = kite.get_client()
    except kite.KiteNotConnected:
        return None
    underlying = symbol.upper()
    try:
        nfo = client.instruments("NFO")
        chain = nfo[(nfo["name"] == underlying) & (nfo["instrument_type"].isin(["CE", "PE"]))].copy()
        if chain.empty:
            return None
        chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.date
        now_ist = kite.datetime.now(kite.IST).replace(tzinfo=None)
        # A contract expiring today is dead after 15:30 IST; roll to the next one.
        expiries = sorted(
            e for e in chain["expiry"].unique()
            if datetime.combine(e, dtime(15, 30)) > now_ist
        )
        if not expiries:
            return None
        chosen = date.fromisoformat(expiry) if expiry else expiries[0]
        chain = chain[chain["expiry"] == chosen]

        spot_key = _UNDERLYING_QUOTE.get(underlying, f"NSE:{underlying}")
        spot = client.ltp([spot_key]).get(spot_key)
        if not spot:
            return None
        strikes = np.sort(chain["strike"].unique())
        atm = int(np.abs(strikes - spot).argmin())
        window = strikes[max(0, atm - STRIKES_EACH_SIDE): atm + STRIKES_EACH_SIDE + 1]
        chain = chain[chain["strike"].isin(window)]
        keys = [f"NFO:{ts}" for ts in chain["tradingsymbol"]]
        quotes = client.quote(keys)
    except kite.KiteNotConnected:
        return None
    except Exception as e:
        logger.warning("Kite option chain failed for %s: %s", symbol, e)
        return None

    now = kite.datetime.now(kite.IST).replace(tzinfo=None)
    t = _years_to_expiry(chosen, now)
    sides: dict[str, dict[float, dict]] = {"CE": {}, "PE": {}}
    for _, row in chain.iterrows():
        q = quotes.get(f"NFO:{row['tradingsymbol']}")
        if not q:
            continue
        price = _mid(q)
        iv = (
            implied_volatility(price, float(spot), float(row["strike"]), t, DEFAULT_RISK_FREE_RATE,
                               "call" if row["instrument_type"] == "CE" else "put")
            if price and t > 0 else np.nan
        )
        prev = (q.get("ohlc") or {}).get("close")
        ltp = float(q.get("last_price") or 0)
        depth = q.get("depth") or {}
        sides[row["instrument_type"]][float(row["strike"])] = {
            "strike": float(row["strike"]),
            "tradingsymbol": row["tradingsymbol"],
            "lastPrice": ltp,
            "volume": int(q.get("volume") or 0),
            "openInterest": int(q.get("oi") or 0),
            "changeinOI": None,  # Kite quotes carry no previous-day OI
            "impliedVolatility": None if not np.isfinite(iv) else round(float(iv) * 100, 2),
            "change": round(ltp - prev, 2) if prev else None,
            "bidPrice": float((depth.get("buy") or [{}])[0].get("price") or 0),
            "askPrice": float((depth.get("sell") or [{}])[0].get("price") or 0),
            "lotSize": int(row["lot_size"]),
        }
    common = sorted(set(sides["CE"]) & set(sides["PE"]))
    calls = [sides["CE"][k] for k in common]
    puts = [sides["PE"][k] for k in common]
    if not calls:
        return None
    total_call_oi = sum(c["openInterest"] for c in calls)
    total_put_oi = sum(p["openInterest"] for p in puts)
    return {
        "symbol": underlying,
        "spotPrice": round(float(spot), 2),
        "expiry": chosen.isoformat(),
        "expiryDates": [e.isoformat() for e in expiries[:8]],
        "calls": calls,
        "puts": puts,
        "totalCallOI": total_call_oi,
        "totalPutOI": total_put_oi,
        "pcr": round(total_put_oi / max(total_call_oi, 1), 3),
        "iv_method": f"Black-Scholes on mid price, r={DEFAULT_RISK_FREE_RATE}, calendar-time to 15:30 IST expiry",
        "source": "kite",
        "as_of": now.isoformat(timespec="seconds"),
    }
