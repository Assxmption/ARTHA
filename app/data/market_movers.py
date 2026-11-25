"""
Market movers and breadth for the ARTHA universe.

Universe: every name the point-in-time security master lists as active
today. That is ARTHA's large-cap universe (the one the live engine trades),
*not* the official NIFTY 50 constituent list, which NSE's blocked endpoints
would be needed for. The response says so in `universe`.

Source priority matches the rest of the live stack:
  * Kite Connect: one batched `quote` call for the whole universe
    (Kite allows up to 500 instruments per call), so the data is live and consistent;
  * otherwise Yahoo daily bars, labelled delayed. A symbol whose latest Yahoo
    bar is older than the universe's latest session is reported as stale
    instead of being ranked against fresher prices.

Nothing is invented. A symbol with no quote or no previous close stays in
`unquoted` and is left out of the rankings and breadth counts.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from app.data import kite
from app.data.nse import _now_ist, _session_closed, _yfinance_symbol, ist_today
from app.data.nse_calendar import is_market_open, market_status

logger = logging.getLogger(__name__)

MASTER_PATH = Path("data_cache/nse/security_master.json")
KITE_TTL_S = 5.0        # Kite quote rate limit is ~1 req/s; one call serves every viewer
YAHOO_TTL_S = 60.0      # delayed data; don't hammer Yahoo
KITE_MAX_INSTRUMENTS = 500


def _f(v) -> Optional[float]:
    try:
        out = None if v is None else float(v)
    except (TypeError, ValueError):
        return None
    return out if out is not None and out == out else None  # NaN → None


def _row(symbol: str, ltp, prev, volume=None, avg_price=None, high=None, low=None, open_=None) -> dict:
    ltp, prev = _f(ltp), _f(prev)
    ok = ltp is not None and prev not in (None, 0.0)
    vol = _f(volume)
    avg = _f(avg_price)
    # Traded value needs the session VWAP; ltp × volume would misstate it.
    traded_value = vol * avg if vol is not None and avg is not None else None
    return {
        "symbol": symbol,
        "ltp": ltp,
        "prev_close": prev,
        "change": round(ltp - prev, 2) if ok else None,
        "change_pct": round((ltp / prev - 1) * 100, 2) if ok else None,
        "open": _f(open_),
        "high": _f(high),
        "low": _f(low),
        "volume": int(vol) if vol is not None else None,
        "traded_value": round(traded_value, 2) if traded_value is not None else None,
    }


def summarize_movers(rows: Iterable[dict], top_n: int = 10) -> dict:
    """Rankings and breadth from normalised rows (pure; unit-tested).

    Rows without a change are excluded from every ranking and breadth count.
    Ties break on symbol so the ordering is deterministic.
    """
    rows = list(rows)
    priced = [r for r in rows if r.get("change_pct") is not None]
    by_change = sorted(priced, key=lambda r: (-r["change_pct"], r["symbol"]))
    gainers = [r for r in by_change if r["change_pct"] > 0][:top_n]
    losers = [r for r in sorted(priced, key=lambda r: (r["change_pct"], r["symbol"])) if r["change_pct"] < 0][:top_n]

    def active_key(r):
        # Traded value when known (Kite), else volume (Yahoo has no VWAP).
        v = r.get("traded_value")
        return v if v is not None else (r.get("volume") or 0)

    active = sorted(
        (r for r in priced if r.get("traded_value") is not None or r.get("volume")),
        key=lambda r: (-active_key(r), r["symbol"]),
    )[:top_n]
    advances = sum(1 for r in priced if r["change_pct"] > 0)
    declines = sum(1 for r in priced if r["change_pct"] < 0)
    unchanged = len(priced) - advances - declines
    return {
        "gainers": gainers,
        "losers": losers,
        "most_active": active,
        "active_metric": "traded_value" if any(r.get("traded_value") is not None for r in active) else "volume",
        "breadth": {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "ad_ratio": round(advances / declines, 2) if declines else None,
        },
        "rows": by_change,
        "unquoted": sorted(r["symbol"] for r in rows if r.get("change_pct") is None),
    }


def _kite_rows(symbols: list[str]) -> Optional[list[dict]]:
    try:
        client = kite.get_client()
    except kite.KiteNotConnected:
        return None
    keys: dict[str, str] = {}
    for sym in symbols:
        exchange, ts = kite.kite_symbol(sym)
        keys[f"{exchange}:{ts}"] = sym
    rows: list[dict] = []
    try:
        key_list = list(keys)
        quotes: dict = {}
        for i in range(0, len(key_list), KITE_MAX_INSTRUMENTS):
            quotes.update(client.quote(key_list[i:i + KITE_MAX_INSTRUMENTS]))
    except kite.KiteNotConnected:
        return None
    except Exception as e:
        logger.warning("Kite movers quote failed: %s", e)
        return None
    for key, sym in keys.items():
        q = quotes.get(key)
        if not q:
            rows.append(_row(sym, None, None))
            continue
        ohlc = q.get("ohlc") or {}
        rows.append(_row(
            sym, q.get("last_price"), ohlc.get("close"), q.get("volume"), q.get("average_price"),
            ohlc.get("high"), ohlc.get("low"), ohlc.get("open"),
        ))
    return rows


def _yahoo_rows(symbols: list[str]) -> tuple[list[dict], list[str]]:
    """Latest vs previous daily close per symbol; also returns stale symbols."""
    import yfinance as yf

    tickers = {_yfinance_symbol(s): s for s in symbols}
    raw = yf.download(
        list(tickers), period="7d", interval="1d", group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for ysym, sym in tickers.items():
        try:
            f = raw[ysym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        f = f.dropna(subset=["Close"])
        if not f.empty:
            frames[sym] = f
    latest = max((f.index[-1] for f in frames.values()), default=None)
    rows, stale = [], []
    for sym in symbols:
        f = frames.get(sym)
        if f is None or len(f) < 2:
            rows.append(_row(sym, None, None))
            continue
        if f.index[-1] != latest:
            stale.append(sym)
            rows.append(_row(sym, None, None))
            continue
        last, prev = f.iloc[-1], f.iloc[-2]
        rows.append(_row(sym, last["Close"], prev["Close"], last.get("Volume"), None,
                         last.get("High"), last.get("Low"), last.get("Open")))
    return rows, stale


def universe_symbols(master_path: Path = MASTER_PATH) -> list[str]:
    from app.data.security_master import PointInTimeSecurityMaster

    master = PointInTimeSecurityMaster.from_file(master_path)
    return sorted(master.symbols_at(ist_today()))


class MoversCache:
    def __init__(self, symbols: Callable[[], list[str]] = universe_symbols):
        self._symbols = symbols
        self._lock = threading.Lock()
        self._at = 0.0
        self._ttl = 0.0
        self._body: Optional[dict] = None

    def get(self, top_n: int = 10) -> dict:
        with self._lock:
            if self._body and time.monotonic() - self._at < self._ttl:
                return self._shape(self._body, top_n)
            symbols = self._symbols()
            stale: list[str] = []
            rows = _kite_rows(symbols)
            if rows is not None and any(r["change_pct"] is not None for r in rows):
                source, live, ttl = "kite", True, KITE_TTL_S
            else:
                rows, stale = _yahoo_rows(symbols)
                source, live, ttl = "yfinance (delayed, daily bars)", False, YAHOO_TTL_S
            now = _now_ist()
            body = {
                "rows_all": rows,
                "stale": sorted(stale),
                "source": source,
                "live": live,
                "market_open": is_market_open(now),
                "market_status": market_status(now)["reason"],
                "as_of": now.isoformat(timespec="seconds"),
                "universe": {
                    "name": "ARTHA large-cap universe (security master, active today)",
                    "size": len(symbols),
                    "note": "Not the official NIFTY 50 constituent list.",
                },
            }
            self._body, self._at, self._ttl = body, time.monotonic(), ttl
            return self._shape(body, top_n)

    @staticmethod
    def _shape(body: dict, top_n: int) -> dict:
        summary = summarize_movers(body["rows_all"], top_n=top_n)
        out = {k: v for k, v in body.items() if k != "rows_all"}
        out.update(summary)
        return out


movers_cache = MoversCache()
