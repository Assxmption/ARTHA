"""
Intraday bars and live quotes for the chart.

Source priority: Kite Connect (exchange-grade, live) when a session is
active; otherwise Yahoo (labelled delayed).  Every response says which
source it came from so the UI never presents delayed data as live.
Timestamps are naive IST wall-clock (the chart renders them as-is).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from app.data import kite
from app.data.nse import _now_ist, _session_closed, _yfinance_symbol, ist_today
from app.data.nse_calendar import is_market_open, market_status

logger = logging.getLogger(__name__)

# chart interval → (kite interval, yahoo interval, max lookback days on Yahoo)
INTRADAY_INTERVALS = {
    "1m": ("minute", "1m", 7),
    "3m": ("3minute", None, 7),      # Yahoo has no 3m; resampled from 1m
    "5m": ("5minute", "5m", 59),
    "15m": ("15minute", "15m", 59),
    "30m": ("30minute", "30m", 59),
    "60m": ("60minute", "60m", 365),
}
INTRADAY_INTERVALS["1h"] = INTRADAY_INTERVALS["60m"]


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = frame.set_index("date").resample(rule, label="left", closed="left", origin="start_day", offset="15min")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).dropna(subset=["open"])
    return out.reset_index()


def fetch_intraday(symbol: str, interval: str, days: int) -> tuple[pd.DataFrame, str]:
    """Intraday OHLCV for `days` calendar days back; returns (frame, source)."""
    if interval not in INTRADAY_INTERVALS:
        raise ValueError(f"Unsupported intraday interval {interval!r}")
    kite_iv, yahoo_iv, yahoo_max = INTRADAY_INTERVALS[interval]
    now = _now_ist().replace(tzinfo=None)
    start = datetime.combine(ist_today() - timedelta(days=max(days, 1) - 1), datetime.min.time())

    try:
        client = kite.get_client()
        exchange, tradingsymbol = kite.kite_symbol(symbol)
        token = client.instrument_token(tradingsymbol, exchange)
        frame = client.historical(token, kite_iv, start, now)
        if not frame.empty:
            return frame[["date", "open", "high", "low", "close", "volume"]], f"kite:{exchange}:{tradingsymbol}"
    except kite.KiteNotConnected:
        pass
    except Exception as e:
        logger.warning("Kite intraday fetch failed for %s %s: %s", symbol, interval, e)

    import yfinance as yf

    span = min(max(days, 1), yahoo_max)
    hist = yf.Ticker(_yfinance_symbol(symbol)).history(period=f"{span}d", interval=yahoo_iv or "1m")
    if hist.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), "none"
    hist = hist.reset_index().rename(columns={
        "Datetime": "date", "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    frame = hist[["date", "open", "high", "low", "close", "volume"]]
    if yahoo_iv is None:
        frame = _resample(frame, "3min")
    return frame.reset_index(drop=True), f"yfinance:{_yfinance_symbol(symbol)} (delayed)"


# ── Quotes ──────────────────────────────────────────────────────────────────────

_quote_cache: dict[str, tuple[float, dict]] = {}
_quote_lock = threading.Lock()
_QUOTE_TTL_S = 1.0  # Kite allows ~1 quote request/second; share it across viewers


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def get_quote(symbol: str) -> dict:
    """Live quote (Kite) or delayed snapshot (Yahoo), normalised."""
    key = symbol.upper()
    with _quote_lock:
        hit = _quote_cache.get(key)
        if hit and time.monotonic() - hit[0] < _QUOTE_TTL_S:
            return hit[1]

    result: Optional[dict] = None
    try:
        client = kite.get_client()
        exchange, tradingsymbol = kite.kite_symbol(symbol)
        q = client.quote([f"{exchange}:{tradingsymbol}"]).get(f"{exchange}:{tradingsymbol}")
        if q:
            prev = _f((q.get("ohlc") or {}).get("close"))
            ltp = _f(q.get("last_price"))
            depth = q.get("depth") or {}
            result = {
                "symbol": key,
                "source": f"kite:{exchange}:{tradingsymbol}",
                "live": True,
                "ltp": ltp,
                "prev_close": prev,
                "change": None if ltp is None or not prev else round(ltp - prev, 2),
                "change_pct": None if ltp is None or not prev else round((ltp / prev - 1) * 100, 2),
                "open": _f((q.get("ohlc") or {}).get("open")),
                "high": _f((q.get("ohlc") or {}).get("high")),
                "low": _f((q.get("ohlc") or {}).get("low")),
                "volume": q.get("volume"),
                "avg_price": _f(q.get("average_price")),
                "upper_circuit": _f(q.get("upper_circuit_limit")),
                "lower_circuit": _f(q.get("lower_circuit_limit")),
                "buy_quantity": q.get("buy_quantity"),
                "sell_quantity": q.get("sell_quantity"),
                "last_trade_time": str(q.get("last_trade_time") or ""),
                "depth": {
                    side: [
                        {"price": _f(l.get("price")), "quantity": l.get("quantity"), "orders": l.get("orders")}
                        for l in depth.get(side, [])[:5]
                    ]
                    for side in ("buy", "sell")
                },
            }
    except kite.KiteNotConnected:
        pass
    except Exception as e:
        logger.warning("Kite quote failed for %s: %s", symbol, e)

    if result is None:
        import yfinance as yf

        fi = yf.Ticker(_yfinance_symbol(symbol)).fast_info
        ltp, prev = _f(fi.last_price), _f(fi.previous_close)
        result = {
            "symbol": key,
            "source": f"yfinance:{_yfinance_symbol(symbol)} (delayed)",
            "live": False,
            "ltp": ltp,
            "prev_close": prev,
            "change": None if ltp is None or not prev else round(ltp - prev, 2),
            "change_pct": None if ltp is None or not prev else round((ltp / prev - 1) * 100, 2),
            "open": _f(fi.open), "high": _f(fi.day_high), "low": _f(fi.day_low),
            "volume": _f(fi.last_volume),
            "depth": None,
        }
    result["market_open"] = is_market_open(_now_ist())
    with _quote_lock:
        _quote_cache[key] = (time.monotonic(), result)
    return result


# ── Index ticker (one batched quote) ───────────────────────────────────────────

TICKER_INDICES = (
    ("NIFTY 50", "NSE:NIFTY 50", "^NSEI"),
    ("BANKNIFTY", "NSE:NIFTY BANK", "^NSEBANK"),
    ("SENSEX", "BSE:SENSEX", "^BSESN"),
    ("INDIA VIX", "NSE:INDIA VIX", "^INDIAVIX"),
)
_ticker_cache: dict = {"at": 0.0, "body": None}
_TICKER_TTL_S = 5.0


def get_index_ticker() -> dict:
    """Live index levels for the top bar: one Kite quote call, else Yahoo (delayed)."""
    with _quote_lock:
        if _ticker_cache["body"] and time.monotonic() - _ticker_cache["at"] < _TICKER_TTL_S:
            return _ticker_cache["body"]

    items: list[dict] = []
    source = None
    try:
        client = kite.get_client()
        quotes = client.quote([k for _, k, _ in TICKER_INDICES])
        for label, key, _ in TICKER_INDICES:
            q = quotes.get(key)
            if not q:
                continue
            ltp, prev = _f(q.get("last_price")), _f((q.get("ohlc") or {}).get("close"))
            items.append({
                "label": label, "ltp": ltp,
                "change": None if ltp is None or not prev else round(ltp - prev, 2),
                "change_pct": None if ltp is None or not prev else round((ltp / prev - 1) * 100, 2),
            })
        source = "kite"
    except kite.KiteNotConnected:
        pass
    except Exception as e:
        logger.warning("Kite index ticker failed: %s", e)

    if not items:
        import yfinance as yf

        for label, _, ysym in TICKER_INDICES:
            try:
                fi = yf.Ticker(ysym).fast_info
                ltp, prev = _f(fi.last_price), _f(fi.previous_close)
            except Exception:
                continue
            items.append({
                "label": label, "ltp": ltp,
                "change": None if ltp is None or not prev else round(ltp - prev, 2),
                "change_pct": None if ltp is None or not prev else round((ltp / prev - 1) * 100, 2),
            })
        source = "yfinance (delayed)"

    now = _now_ist()
    body = {
        "items": items,
        "source": source,
        "live": source == "kite",
        "market_open": is_market_open(now),
        "market_status": market_status(now)["reason"],
        "as_of": now.isoformat(timespec="seconds"),
    }
    with _quote_lock:
        _ticker_cache.update(at=time.monotonic(), body=body)
    return body
