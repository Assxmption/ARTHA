"""
Symbol resolution and instrument search.

`resolve_symbol` maps a former ticker to the current one using the security
master's aliases (ZOMATO → ETERNAL, TATAMOTORS → TMPV), so old symbols in
watchlists, bookmarks and hard-coded lists keep working everywhere prices
are fetched.

`search_instruments` is a local, ranked search over:
  * the security master: ARTHA's universe, plus former tickers and names;
  * Kite's NSE instrument master (about 10k equities, ETFs and indices,
    cached daily by app.data.kite) when a Kite session exists.
It needs no network call per keystroke, unlike the old Yahoo search, which
was slow and favoured US listings. Yahoo stays as a last resort when nothing
local matches.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MASTER_PATH = Path("data_cache/nse/security_master.json")
_TICKER_RE = re.compile(r"^[A-Z0-9&_-]{1,20}$")
# Kite index names → the symbols ARTHA charts them under (Yahoo-mappable too).
_INDEX_ALIASES = {"NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY", "INDIA VIX": "INDIAVIX"}
_CORE_INDICES = {"NIFTY", "BANKNIFTY", "INDIAVIX", "NIFTY FIN SERVICE", "NIFTY MIDCAP 100"}
_SERIES_RE = re.compile(r"-[A-Z0-9]{2}$")


# ── Former-ticker resolution ──────────────────────────────────────────────────

_alias_lock = threading.Lock()
_alias_cache: dict = {"mtime": None, "map": {}}


def _alias_map(path: Path = MASTER_PATH) -> dict[str, str]:
    """Ticker-shaped aliases → canonical symbol; reloaded when the file changes."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _alias_lock:
        if _alias_cache["mtime"] == mtime:
            return _alias_cache["map"]
        mapping: dict[str, str] = {}
        try:
            from app.data.security_master import PointInTimeSecurityMaster

            master = PointInTimeSecurityMaster.from_file(path)
            for rec in master.records:
                for alias in rec.aliases:
                    if _TICKER_RE.match(alias) and alias != rec.canonical_symbol:
                        mapping[alias] = rec.canonical_symbol
        except Exception as e:  # a broken master must not break every price fetch
            logger.warning("Security master aliases unavailable: %s", e)
        _alias_cache.update(mtime=mtime, map=mapping)
        return mapping


def resolve_symbol(symbol: str) -> str:
    """Current ticker for `symbol` (unchanged if it isn't a known former ticker)."""
    s = str(symbol).strip().upper().removesuffix(".NS")
    return _alias_map(MASTER_PATH).get(s, s)  # module global read at call time


# ── Search ────────────────────────────────────────────────────────────────────

@dataclass
class _Entry:
    symbol: str
    name: str
    exchange: str
    kind: str                       # "EQ" | "INDEX"
    in_universe: bool = False
    former: tuple[str, ...] = ()    # former tickers that now point here
    names: tuple[str, ...] = ()     # extra searchable names
    _sym: str = field(init=False, repr=False)
    _texts: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sym = self.symbol.lower()
        self._texts = tuple(t.lower() for t in (self.name, *self.names) if t)


def _score(e: _Entry, q: str) -> tuple[int, Optional[str]]:
    """Higher is better; also returns the former ticker that matched, if any."""
    best, via = 0, None
    if e._sym == q:
        best = 1000
    elif e._sym.startswith(q):
        best = 800 - min(len(e._sym) - len(q), 50)
    for f in e.former:
        fl = f.lower()
        s = 950 if fl == q else (700 - min(len(fl) - len(q), 50) if fl.startswith(q) else 0)
        if s > best:
            best, via = s, f
    if e.kind == "INDEX" and e._texts and e._texts[0].startswith(q):
        # An index's name is effectively its ticker ("Nifty Bank" for BANKNIFTY).
        best = max(best, 800 - min(len(e._texts[0]) - len(q), 50))
    for t in e._texts:
        if t.startswith(q):
            best = max(best, 600)
        elif any(w.startswith(q) for w in re.split(r"[\s\-&.()]+", t) if w):
            best = max(best, 500)
        elif len(q) >= 3 and q in t:
            best = max(best, 200)
    if best < 300 and len(q) >= 2 and q in e._sym:
        best = max(best, 300)
    if best and e.in_universe:
        best += 200         # ARTHA's own universe first (a name match beats an obscure ticker prefix)
    if best and e.kind == "INDEX":
        best += 150 if e.symbol in _CORE_INDICES else 20
    return best, via


class InstrumentIndex:
    def __init__(self, master_path: Path = MASTER_PATH):
        self.master_path = master_path
        self._lock = threading.Lock()
        self._key: Optional[tuple] = None
        self._entries: list[_Entry] = []
        self.sources: list[str] = []

    def _kite_frame(self):
        try:
            from app.data import kite

            client = kite.get_client()
            return client.instruments("NSE"), str(kite._session_trading_day())
        except Exception:
            return None, None

    def _build(self) -> None:
        try:
            mtime = self.master_path.stat().st_mtime
        except OSError:
            mtime = None
        frame, kite_day = self._kite_frame()
        key = (mtime, kite_day)
        if key == self._key and self._entries:
            return

        from datetime import date

        from app.data.security_master import PointInTimeSecurityMaster

        by_symbol: dict[str, _Entry] = {}
        sources = []
        if frame is not None:
            rows = frame[(frame["segment"].isin(["NSE", "INDICES"]))]
            rows = rows[(rows["segment"] == "INDICES") | (rows["instrument_type"] == "EQ")]
            for sym, name, seg in zip(rows["tradingsymbol"], rows["name"].fillna(""), rows["segment"]):
                sym = str(sym)
                # Skip debt/SGB/rights series ("765TMCV27-N0", "SGBJUN31I-GB", "XYZ-RE"):
                # a two-character series suffix or no company name. Keeps BAJAJ-AUTO, M&M.
                if seg != "INDICES" and (
                    not _TICKER_RE.match(sym) or _SERIES_RE.search(sym) or not name or str(name).upper() == sym
                ):
                    continue
                if seg == "INDICES":
                    sym = _INDEX_ALIASES.get(sym, sym)
                by_symbol[sym] = _Entry(sym, str(name).title() if name else sym, "NSE",
                                        "INDEX" if seg == "INDICES" else "EQ")
            sources.append("kite-instruments")

        try:
            master = PointInTimeSecurityMaster.from_file(self.master_path)
            today = date.today()
            for rec in master.records:
                if not rec.is_active(today):
                    continue
                former = tuple(a for a in rec.aliases if _TICKER_RE.match(a) and a != rec.canonical_symbol)
                names = tuple(a for a in rec.aliases if not _TICKER_RE.match(a))
                e = by_symbol.get(rec.canonical_symbol)
                if e is None:
                    e = _Entry(rec.canonical_symbol, (names[0].title() if names else rec.canonical_symbol), "NSE", "EQ")
                    by_symbol[rec.canonical_symbol] = e
                e.in_universe = True
                e.former = former
                e.names = tuple(dict.fromkeys((*e.names, *(n.title() for n in names))))
                e.__post_init__()
            sources.append("security-master")
        except Exception as ex:
            logger.warning("Security master unavailable for search: %s", ex)

        self._entries = list(by_symbol.values())
        self._key = key
        self.sources = sources

    def search(self, query: str, limit: int = 8) -> list[dict]:
        q = query.strip().lower().removesuffix(".ns")
        if not q:
            return []
        with self._lock:
            self._build()
            entries = self._entries
        scored = []
        for e in entries:
            s, via = _score(e, q)
            if s:
                scored.append((s, e, via))
        scored.sort(key=lambda t: (-t[0], len(t[1].symbol), t[1].symbol))
        return [
            {
                "symbol": e.symbol,
                "name": e.name,
                "exchange": e.exchange,
                "type": e.kind,
                "in_universe": e.in_universe,
                **({"matched_former": via} if via else {}),
                **({"former": list(e.former)} if e.former else {}),
            }
            for _, e, via in scored[:limit]
        ]


instrument_index = InstrumentIndex()
