"""
Intraday (5-minute) price history from Kite Connect, cached locally.

One parquet file per symbol under ``data_cache/intraday/``; re-running only
fetches the missing tail.  Bars are as-traded (Kite candles are not adjusted
for splits/bonuses/demergers) — intraday returns are unaffected, but anything
that compares across days must mask corporate-action days (see
``app/quant/intraday_research.py``).

Read-only Kite usage (historical endpoint only).  Fetched once, reused.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache/intraday")
INTERVAL = "5minute"
SESSION_START, SESSION_END = time(9, 15), time(15, 30)
INDEX_TOKENS = {"NIFTY": "NIFTY 50"}


def _path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('&', '_and_')}_5m.parquet"


def load(symbol: str) -> pd.DataFrame:
    p = _path(symbol)
    if not p.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return pd.read_parquet(p)


def _token(client, symbol: str) -> int:
    from app.data import kite

    if symbol in INDEX_TOKENS:
        frame = client.instruments("NSE")
        row = frame[(frame["tradingsymbol"] == INDEX_TOKENS[symbol]) & (frame["segment"] == "INDICES")]
        return int(row["instrument_token"].iloc[0])
    exchange, ts = kite.kite_symbol(symbol)
    return client.instrument_token(ts, exchange)


def update(symbol: str, start: date, end: date, client=None) -> pd.DataFrame:
    """Ensure the cache covers [start, end]; returns the cached frame."""
    from app.data import kite

    client = client or kite.get_client()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    have = load(symbol)
    fetch_from = datetime.combine(start, SESSION_START)
    if not have.empty:
        first, last = pd.Timestamp(have["date"].min()), pd.Timestamp(have["date"].max())
        if first.date() <= start + timedelta(days=7):            # head already covered (allow holidays)
            fetch_from = (last + pd.Timedelta(minutes=5)).to_pydatetime()
    fetch_to = datetime.combine(end, SESSION_END)
    if fetch_from > fetch_to:
        return have
    token = _token(client, symbol)
    new = client.historical(token, INTERVAL, fetch_from, fetch_to)
    if not new.empty:
        new = new[["date", "open", "high", "low", "close", "volume"]]
        for c in ("open", "high", "low", "close"):
            new[c] = new[c].astype("float32")
        new["volume"] = new["volume"].astype("float64")
    frame = pd.concat([have, new], ignore_index=True) if not have.empty else new
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    tmp = _path(symbol).with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(_path(symbol))
    return frame


def update_many(symbols: Iterable[str], start: date, end: date, log_every: int = 5) -> dict:
    """Update several symbols; returns {symbol: rows or error string}."""
    from app.data import kite

    client = kite.get_client()
    out: dict[str, object] = {}
    symbols = list(symbols)
    for i, s in enumerate(symbols, 1):
        try:
            out[s] = len(update(s, start, end, client))
        except Exception as e:  # keep going; report at the end
            out[s] = f"{type(e).__name__}: {e}"
            logger.warning("intraday history %s failed: %s", s, e)
        if i % log_every == 0 or i == len(symbols):
            logger.info("intraday history: %d/%d symbols", i, len(symbols))
    return out


def panel(symbols: Iterable[str], field: str = "close", start: Optional[date] = None,
          end: Optional[date] = None) -> pd.DataFrame:
    """Wide frame (timestamp × symbol) of one field, regular-session bars only."""
    cols = {}
    for s in symbols:
        f = load(s)
        if f.empty:
            continue
        f = f.set_index(pd.DatetimeIndex(f["date"]))[field]
        cols[s] = f
    if not cols:
        return pd.DataFrame()
    wide = pd.DataFrame(cols).sort_index()
    t = wide.index.time
    wide = wide[(t >= SESSION_START) & (t < SESSION_END)]
    if start:
        wide = wide[wide.index.date >= start]
    if end:
        wide = wide[wide.index.date <= end]
    return wide


def split_dates(symbols: Iterable[str], cache: Path = CACHE_DIR / "corporate_actions.json") -> dict[str, list[str]]:
    """
    Split/bonus ex-dates per symbol (Yahoo "splits", which include bonus issues),
    cached for a day.  Used to mask days where as-traded prices jump for
    non-economic reasons.  Demergers are not listed by Yahoo; the research
    harness also masks implausible (>40 %) overnight jumps as a backstop.
    """
    import json
    import time as _time

    symbols = list(symbols)
    if cache.exists() and _time.time() - cache.stat().st_mtime < 86400:
        data = json.loads(cache.read_text())
        if all(s in data for s in symbols):
            return {s: data[s] for s in symbols}
    import yfinance as yf

    from app.data.nse import _yfinance_symbol

    data: dict[str, list[str]] = {}
    for s in symbols:
        if s in INDEX_TOKENS:
            data[s] = []
            continue
        try:
            splits = yf.Ticker(_yfinance_symbol(s)).splits
            data[s] = sorted({str(pd.Timestamp(d).date()) for d in splits.index})
        except Exception as e:
            logger.warning("split dates for %s unavailable: %s", s, e)
            data[s] = []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data
