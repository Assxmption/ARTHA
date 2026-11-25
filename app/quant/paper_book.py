"""
Paper books — a virtual order ledger driven by ARTHA's deterministic signals
===========================================================================

Every order here is SIMULATED.  No broker order API is imported or called;
Kite is used only to *read* prices (AGENTS.md rule 9).  Nothing in this module
is investment advice (rule 11).

Two books, each with its own virtual capital:

* ``validated`` — only signals that pass the rule-6 gate (walk-forward OOS
  Sharpe ≥ 1, max drawdown ≤ 25 %, ≥ 504 OOS sessions).  Today that is the
  stat-arb pairs the live engine marks ``validated``.  The book holds exactly
  the position each pair's own backtest holds after the latest close, so the
  paper book and the validated backtest are the same strategy.
* ``shadow`` — the predeclared momentum/reversal sleeve portfolio, which
  currently FAILS the gate.  It is paper-traded as a forward test only, its
  P&L is always reported apart from the validated book and labelled
  UNVALIDATED, and it is never handed to an LLM for narration.

Mechanics (all deterministic):

* Signals come from the live engine (app/quant/live_dashboard.py), which is
  recomputed after each session close from adjusted daily closes.  Signals
  from close *t* are filled during session *t+1* — never before the close
  that produced them.
* Fills happen only during NSE hours and only against a fresh price
  (observed today, at most ``MAX_PRICE_AGE`` old): Kite LTP when a session
  exists, otherwise Yahoo 1-minute bars (delayed).  Every order stores the
  price source and observation time it was filled against.
* Fill price = reference price ± ``COST_BPS`` (15 bp: 5 bp brokerage + 10 bp
  impact), the same per-unit-turnover cost both backtests charge.
* Target quantity = weight × book NAV / price, truncated toward zero.  A
  position is traded when it opens, closes or flips, or when it has drifted
  by more than ``DRIFT_FRACTION`` of its target *and* the trade is at least
  ``MIN_TRADE_FRACTION`` of NAV — so the books do not churn on noise.
* Shorts are allowed (both strategies are long/short).  In Indian cash
  equities an overnight short is not possible; the paper book treats a short
  as the equivalent single-stock-futures exposure and says so in the report.
* Positions, cash and realized P&L are never stored as mutable state: they
  are recomputed by replaying the order log (average-cost method), so the
  ledger is auditable end to end.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
LEDGER_PATH = Path("data_cache/paper/ledger.db")
LOCK_PATH = Path("data_cache/paper/scheduler.lock")

INITIAL_CAPITAL = 5_00_00_000.0      # ₹5 crore per book
COST_BPS = 15.0                       # = statarb_backtest (5 + 10) per leg; = sleeve cost_bps
MAX_PRICE_AGE = timedelta(minutes=20)
MAX_SIGNAL_AGE_DAYS = 5               # calendar days from signal close to fill (covers long weekends)
MIN_TRADE_FRACTION = 0.002            # ₹1 lakh on ₹5 Cr
DRIFT_FRACTION = 0.10

# Pair exit rules (statarb.generate_pair_signals / statarb_backtest.backtest_pair defaults).
PAIR_Z_EXIT, PAIR_Z_STOP, PAIR_MAX_HOLDING_DAYS = 0.5, 4.0, 60


@dataclass(frozen=True)
class BookSpec:
    key: str
    label: str
    validated: bool
    description: str


BOOKS: dict[str, BookSpec] = {
    "validated": BookSpec(
        "validated", "Validated signals", True,
        "Only signals that pass the walk-forward OOS gate (Sharpe ≥ 1, max DD ≤ 25 %, ≥ 504 sessions). "
        "Currently the validated stat-arb pairs.",
    ),
    "shadow": BookSpec(
        "shadow", "Shadow forward test (UNVALIDATED)", False,
        "The predeclared momentum/reversal sleeve portfolio. It currently fails the validation gate and is "
        "paper-traded only as a forward test. Not a validated signal.",
    ),
}


@dataclass(frozen=True)
class Mark:
    symbol: str
    price: float
    observed_at: datetime     # timezone-aware
    source: str


@dataclass(frozen=True)
class Target:
    symbol: str
    weight: float             # signed fraction of book NAV
    reason: str
    signal_ref: str
    signal_as_of: Optional[str]
    facts: dict = field(default_factory=dict)


# ── Accounting (pure) ─────────────────────────────────────────────────────────

def apply_fill(qty: int, avg: float, signed_qty: int, price: float) -> tuple[int, float, float]:
    """
    Average-cost update of a signed position.

    Returns (new_qty, new_avg_price, realized_pnl).  Adding to a position
    re-averages the cost; reducing it realizes P&L on the closed part at the
    old average; crossing zero closes the old side fully and opens the rest
    at ``price``.
    """
    if signed_qty == 0:
        return qty, avg, 0.0
    if qty == 0 or (qty > 0) == (signed_qty > 0):
        new_qty = qty + signed_qty
        new_avg = (abs(qty) * avg + abs(signed_qty) * price) / abs(new_qty)
        return new_qty, new_avg, 0.0
    closing = min(abs(signed_qty), abs(qty))
    realized = closing * (price - avg) * (1 if qty > 0 else -1)
    new_qty = qty + signed_qty
    if new_qty == 0:
        return 0, 0.0, realized
    if (new_qty > 0) == (qty > 0):
        return new_qty, avg, realized
    return new_qty, price, realized


@dataclass
class PositionState:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[str] = None
    last_order_at: Optional[str] = None


def replay(orders: Iterable[dict]) -> tuple[dict[str, PositionState], float, float, float]:
    """
    Rebuild positions from an order log (oldest first).

    Returns (open_positions, cash_flow, realized_pnl, cost_paid) where
    cash_flow is the net cash effect of all fills (sells add, buys subtract)
    and cost_paid is the slippage/charges embedded in fill prices.
    """
    book: dict[str, PositionState] = {}
    cash_flow = realized_total = cost_total = 0.0
    for o in orders:
        signed = o["quantity"] if o["side"] == "BUY" else -o["quantity"]
        st = book.setdefault(o["symbol"], PositionState(o["symbol"]))
        prev = st.quantity
        st.quantity, st.avg_price, realized = apply_fill(prev, st.avg_price, signed, o["fill_price"])
        st.realized_pnl += realized
        realized_total += realized
        cash_flow -= signed * o["fill_price"]
        cost_total += abs(o["fill_price"] - o["ref_price"]) * o["quantity"]
        if st.quantity == 0:
            st.opened_at = None
        elif prev == 0 or (prev > 0) != (st.quantity > 0):   # opened, or flipped side
            st.opened_at = o["placed_at"]
        st.last_order_at = o["placed_at"]
    open_positions = {s: p for s, p in book.items() if p.quantity != 0}
    return open_positions, cash_flow, realized_total, cost_total


# ── Ledger (SQLite) ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    book TEXT PRIMARY KEY,
    initial_capital REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    ref_price REAL NOT NULL,
    fill_price REAL NOT NULL,
    cost_bps REAL NOT NULL,
    notional REAL NOT NULL,
    position_before INTEGER NOT NULL,
    position_after INTEGER NOT NULL,
    realized_pnl REAL NOT NULL,
    reason TEXT NOT NULL,
    signal_ref TEXT NOT NULL,
    signal_as_of TEXT,
    price_source TEXT NOT NULL,
    price_observed_at TEXT NOT NULL,
    run_id INTEGER
);
CREATE INDEX IF NOT EXISTS orders_book ON orders (book, id);
CREATE TABLE IF NOT EXISTS marks (
    symbol TEXT PRIMARY KEY,
    price REAL NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nav (
    book TEXT NOT NULL,
    ts TEXT NOT NULL,
    session_date TEXT NOT NULL,
    cash REAL NOT NULL,
    long_value REAL NOT NULL,
    short_value REAL NOT NULL,
    nav REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    PRIMARY KEY (book, ts)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    session_open INTEGER NOT NULL,
    traded INTEGER NOT NULL,
    signal_as_of TEXT,
    n_orders INTEGER NOT NULL DEFAULT 0,
    note TEXT
);
CREATE TABLE IF NOT EXISTS signal_snapshots (
    as_of TEXT PRIMARY KEY,
    computed_at TEXT,
    saved_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    weight REAL NOT NULL,
    reason TEXT NOT NULL,
    signal_ref TEXT NOT NULL,
    signal_as_of TEXT,
    facts TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (book, symbol)
);
"""


def _rows(cur: sqlite3.Cursor) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class PaperLedger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    # books
    def ensure_book(self, book: str, capital: float, now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO books (book, initial_capital, created_at) VALUES (?, ?, ?)",
                (book, capital, now.isoformat()),
            )

    def books(self) -> dict[str, dict]:
        with self._lock:
            return {r["book"]: r for r in _rows(self._conn.execute("SELECT * FROM books"))}

    # orders
    def orders(self, book: Optional[str] = None) -> list[dict]:
        with self._lock:
            if book:
                cur = self._conn.execute("SELECT * FROM orders WHERE book = ? ORDER BY id", (book,))
            else:
                cur = self._conn.execute("SELECT * FROM orders ORDER BY id")
            return _rows(cur)

    def state(self, book: str) -> tuple[dict[str, PositionState], float, float, float]:
        """(open positions, cash, realized P&L, cost paid) for a book."""
        capital = self.books().get(book, {}).get("initial_capital", 0.0)
        positions, flow, realized, cost = replay(self.orders(book))
        return positions, capital + flow, realized, cost

    def record_order(
        self, *, book: str, placed_at: datetime, symbol: str, side: str, quantity: int,
        ref_price: float, cost_bps: float, reason: str, signal_ref: str, signal_as_of: Optional[str],
        price_source: str, price_observed_at: datetime, run_id: Optional[int] = None,
    ) -> dict:
        if side not in ("BUY", "SELL") or quantity <= 0 or not (ref_price > 0):
            raise ValueError(f"invalid order {side} {quantity} {symbol} @ {ref_price}")
        mult = 1 + cost_bps / 10_000 if side == "BUY" else 1 - cost_bps / 10_000
        fill = round(ref_price * mult, 4)
        with self._lock, self._conn:
            positions, _, _, _ = self.state(book)
            before = positions.get(symbol, PositionState(symbol))
            signed = quantity if side == "BUY" else -quantity
            after_qty, _, realized = apply_fill(before.quantity, before.avg_price, signed, fill)
            row = dict(
                book=book, placed_at=placed_at.isoformat(), symbol=symbol, side=side, quantity=quantity,
                ref_price=round(ref_price, 4), fill_price=fill, cost_bps=cost_bps,
                notional=round(fill * quantity, 2), position_before=before.quantity,
                position_after=after_qty, realized_pnl=round(realized, 2), reason=reason,
                signal_ref=signal_ref, signal_as_of=signal_as_of, price_source=price_source,
                price_observed_at=price_observed_at.isoformat(), run_id=run_id,
            )
            cur = self._conn.execute(
                f"INSERT INTO orders ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
            row["id"] = cur.lastrowid
        return row

    # marks
    def upsert_marks(self, marks: Iterable[Mark]) -> None:
        with self._lock, self._conn:
            for m in marks:
                self._conn.execute(
                    "INSERT INTO marks (symbol, price, observed_at, source) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(symbol) DO UPDATE SET price = excluded.price, "
                    "observed_at = excluded.observed_at, source = excluded.source "
                    "WHERE excluded.observed_at >= marks.observed_at",
                    (m.symbol, m.price, m.observed_at.isoformat(), m.source),
                )

    def marks(self) -> dict[str, Mark]:
        with self._lock:
            return {
                r["symbol"]: Mark(r["symbol"], r["price"], datetime.fromisoformat(r["observed_at"]), r["source"])
                for r in _rows(self._conn.execute("SELECT * FROM marks"))
            }

    # signal snapshots: the last *completed-close* engine output, so a
    # provisional intraday recompute never drives (or blocks) fills.
    def save_snapshot(self, dashboard: dict, now: datetime) -> None:
        keep = {k: dashboard.get(k) for k in ("as_of", "computed_at", "provisional", "engine_version", "gate")}
        keep["pairs"] = dashboard.get("pairs", {})
        keep["portfolio"] = {k: dashboard.get("portfolio", {}).get(k) for k in
                             ("book_weights", "vt_scale_next_session", "current_composition", "weights")}
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO signal_snapshots VALUES (?, ?, ?, ?)",
                (keep["as_of"], keep.get("computed_at"), now.isoformat(), json.dumps(keep, default=str)),
            )

    def latest_snapshot(self) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM signal_snapshots ORDER BY as_of DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    # targets (the signal view behind each order; kept for the report)
    def replace_targets(self, book: str, targets: dict[str, Target], now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM targets WHERE book = ?", (book,))
            for t in targets.values():
                self._conn.execute(
                    "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (book, t.symbol, t.weight, t.reason, t.signal_ref, t.signal_as_of,
                     json.dumps(t.facts, default=str), now.isoformat()),
                )

    def targets(self, book: str) -> dict[str, dict]:
        with self._lock:
            rows = _rows(self._conn.execute("SELECT * FROM targets WHERE book = ?", (book,)))
        for r in rows:
            r["facts"] = json.loads(r["facts"])
        return {r["symbol"]: r for r in rows}

    # nav + runs
    def record_nav(self, book: str, now: datetime, **values: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO nav (book, ts, session_date, cash, long_value, short_value, nav, "
                "realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (book, now.isoformat(), now.date().isoformat(), values["cash"], values["long_value"],
                 values["short_value"], values["nav"], values["realized_pnl"], values["unrealized_pnl"]),
            )

    def nav_history(self, book: str, daily: bool = True) -> list[dict]:
        """NAV snapshots; with ``daily`` only the last snapshot of each session date."""
        with self._lock:
            if daily:
                cur = self._conn.execute(
                    "SELECT n.* FROM nav n JOIN (SELECT session_date, MAX(ts) ts FROM nav WHERE book = ? "
                    "GROUP BY session_date) d ON n.ts = d.ts WHERE n.book = ? ORDER BY n.ts", (book, book),
                )
            else:
                cur = self._conn.execute("SELECT * FROM nav WHERE book = ? ORDER BY ts", (book,))
            return _rows(cur)

    def start_run(self, now: datetime, session_open: bool, traded: bool, signal_as_of: Optional[str]) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (started_at, session_open, traded, signal_as_of) VALUES (?, ?, ?, ?)",
                (now.isoformat(), int(session_open), int(traded), signal_as_of),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, now: datetime, n_orders: int, note: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, n_orders = ?, note = ? WHERE id = ?",
                (now.isoformat(), n_orders, note, run_id),
            )

    def runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)))


# ── Signals → targets ─────────────────────────────────────────────────────────

def _merge(targets: dict[str, Target], t: Target) -> None:
    prev = targets.get(t.symbol)
    if prev is None:
        targets[t.symbol] = t
        return
    targets[t.symbol] = Target(
        t.symbol, prev.weight + t.weight, f"{prev.reason} | {t.reason}",
        f"{prev.signal_ref}+{t.signal_ref}", t.signal_as_of,
        {"components": [*prev.facts.get("components", [prev.facts]), t.facts]},
    )


def validated_targets(dashboard: dict) -> dict[str, Target]:
    """
    Per-symbol weights of the validated pair book.

    Capital is split equally across every validated pair (flat ones keep
    their share in cash).  Within a pair the legs follow the backtest's
    return definition ``position · (r_A − h · r_B) / (1 + |h|)``: A gets
    ``position / (1 + |h|)`` and B gets ``−position · h / (1 + |h|)`` of the
    pair's capital, with h the hedge ratio fixed at entry.
    """
    pairs = [p for p in dashboard.get("pairs", {}).get("pairs", []) if p.get("validated") is True]
    out: dict[str, Target] = {}
    n = len(pairs)
    for p in pairs:
        pos = int(p.get("position_after_close") or 0)
        if pos == 0:
            continue
        h = p.get("entry_hedge_ratio")
        if h is None:
            h = p.get("hedge_ratio")
        if h is None or not (abs(h) < 1e6):
            continue
        denom = 1 + abs(h)
        oos = p.get("oos", {})
        direction = "long" if pos > 0 else "short"
        facts = {
            "pair": p["pair"], "position": pos, "entry_hedge_ratio": h,
            "z_score": p.get("z_score"), "signal_state": p.get("signal_state"),
            "half_life_sessions": p.get("half_life"), "train_p_value": p.get("train_p_value"),
            "oos_sharpe": oos.get("sharpe_ratio"), "oos_max_drawdown": oos.get("max_drawdown"),
            "oos_sessions": oos.get("num_trading_days"),
            "exit_rule": (
                f"Exit when |z| ≤ {PAIR_Z_EXIT}, stop when |z| ≥ {PAIR_Z_STOP}, "
                f"or after {PAIR_MAX_HOLDING_DAYS} sessions in the trade"
            ),
        }
        base = (
            f"{p['pair']} {direction} spread (backtest position after {p.get('signal_date')} close = {pos:+d}, "
            f"entry hedge {h:.4f}, z = {p.get('z_score')}); validated: OOS Sharpe {oos.get('sharpe_ratio')}, "
            f"max DD {oos.get('max_drawdown')}"
        )
        for sym, w in ((p["x"], pos / denom / n), (p["y"], -pos * h / denom / n)):
            _merge(out, Target(sym, w, base, f"pair:{p['pair']}", p.get("signal_date"), facts))
    return out


def shadow_targets(dashboard: dict) -> dict[str, Target]:
    """Net sleeve-book weights × the vol-target scale the engine set for the next session."""
    port = dashboard.get("portfolio", {})
    scale = port.get("vt_scale_next_session")
    if scale is None or not (0 < scale < 10):
        return {}
    comp = port.get("current_composition", {})
    as_of = dashboard.get("as_of")
    out: dict[str, Target] = {}
    for sym, w in port.get("book_weights", {}).items():
        longs = sorted(n for n, b in comp.items() if sym in b.get("long", []))
        shorts = sorted(n for n, b in comp.items() if sym in b.get("short", []))
        next_dec = {n: comp[n].get("next_decision_date") for n in (*longs, *shorts)}
        legs = ", ".join([*(f"long in {n}" for n in longs), *(f"short in {n}" for n in shorts)])
        out[sym] = Target(
            sym, w * scale,
            f"Sleeve book net {w:+.4%} × vol-target scale {scale:.4f} ({legs}); UNVALIDATED forward test",
            "sleeves", as_of,
            {"sleeves_long": longs, "sleeves_short": shorts, "book_weight": w, "vt_scale": scale,
             "next_rebalance": next_dec,
             "exit_rule": "Held until a sleeve that holds it re-ranks it out on its next rebalance date"},
        )
    return out


TARGET_BUILDERS: dict[str, Callable[[dict], dict[str, Target]]] = {
    "validated": validated_targets,
    "shadow": shadow_targets,
}


# ── Prices ────────────────────────────────────────────────────────────────────

def _kite_marks(symbols: list[str]) -> dict[str, Mark]:
    try:
        from app.data import kite

        client = kite.get_client()
    except Exception:
        return {}
    keys = {}
    for s in symbols:
        ex, ts = kite.kite_symbol(s)
        keys[f"{ex}:{ts}"] = s
    try:
        quotes = client.quote(list(keys))
    except Exception as e:
        logger.warning("Kite quote failed for paper marks: %s", e)
        return {}
    out = {}
    for key, sym in keys.items():
        q = quotes.get(key) or {}
        price, ltt = q.get("last_price"), q.get("last_trade_time") or q.get("timestamp")
        if not price or not ltt:
            continue
        try:
            observed = datetime.fromisoformat(str(ltt).replace("T", " ")).replace(tzinfo=IST)
        except ValueError:
            continue
        out[sym] = Mark(sym, float(price), observed, "kite-ltp")
    return out


def _yahoo_marks(symbols: list[str]) -> dict[str, Mark]:
    if not symbols:
        return {}
    try:
        import pandas as pd
        import yfinance as yf

        from app.data.nse import _yfinance_symbol
    except ImportError:
        return {}
    tickers = {_yfinance_symbol(s): s for s in symbols}
    try:
        raw = yf.download(list(tickers), period="5d", interval="1m", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
    except Exception as e:
        logger.warning("Yahoo intraday failed for paper marks: %s", e)
        return {}
    out = {}
    for ysym, sym in tickers.items():
        try:
            frame = raw[ysym] if isinstance(raw.columns, pd.MultiIndex) else raw
            close = frame["Close"].dropna()
        except KeyError:
            continue
        if close.empty:
            continue
        ts = pd.Timestamp(close.index[-1])
        ts = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(IST)
        out[sym] = Mark(sym, float(close.iloc[-1]), ts.to_pydatetime(), "yahoo-1m (delayed)")
    return out


def fetch_marks(symbols: list[str]) -> dict[str, Mark]:
    """Latest trade price per symbol: Kite when connected, Yahoo 1-minute bars for the rest."""
    marks = _kite_marks(symbols)
    missing = [s for s in symbols if s not in marks]
    marks.update(_yahoo_marks(missing))
    return marks


def _default_signals() -> dict:
    from app.quant.live_dashboard import live_dashboard_cache

    return live_dashboard_cache.get()


# ── Engine ────────────────────────────────────────────────────────────────────

def in_session(now: datetime) -> bool:
    """NSE session in progress (holiday calendar in app/data/nse_calendar.py)."""
    from app.data.nse_calendar import is_market_open

    return is_market_open(now)


def is_fresh(mark: Optional[Mark], now: datetime) -> bool:
    if mark is None or not (mark.price > 0):
        return False
    obs = mark.observed_at.astimezone(IST)
    if obs.date() != now.astimezone(IST).date():
        return False   # yesterday's close (or a holiday) is never a fill price
    age = now - obs
    return -timedelta(minutes=2) <= age <= MAX_PRICE_AGE   # small allowance for clock skew


def signal_status(dashboard: dict, now: datetime) -> tuple[bool, str]:
    as_of = dashboard.get("as_of")
    if not as_of:
        return False, "signals unavailable"
    if dashboard.get("provisional"):
        return False, f"signals for {as_of} are provisional (session still forming)"
    age = (now.astimezone(IST).date() - date.fromisoformat(as_of)).days
    if age < 0:
        return False, f"signal date {as_of} is in the future"
    if age > MAX_SIGNAL_AGE_DAYS:
        return False, f"signals from {as_of} are {age} days old"
    return True, f"signals from the {as_of} close"


def valuation(positions: dict[str, PositionState], marks: dict[str, Mark]) -> dict:
    long_v = short_v = unrealized = 0.0
    unpriced = []
    for sym, p in positions.items():
        m = marks.get(sym)
        price = m.price if m else p.avg_price
        if m is None:
            unpriced.append(sym)
        v = p.quantity * price
        if p.quantity > 0:
            long_v += v
        else:
            short_v += v
        unrealized += p.quantity * (price - p.avg_price)
    return {"long_value": long_v, "short_value": short_v, "unrealized_pnl": unrealized, "unpriced": unpriced}


class PaperEngine:
    def __init__(
        self,
        ledger: PaperLedger,
        signals: Callable[[], dict] = _default_signals,
        prices: Callable[[list[str]], dict[str, Mark]] = fetch_marks,
        now: Callable[[], datetime] = lambda: datetime.now(IST),
        capital: float = INITIAL_CAPITAL,
        books: Iterable[str] = tuple(BOOKS),
    ):
        self.ledger, self.signals, self.prices, self.now = ledger, signals, prices, now
        self.capital, self.books = capital, tuple(books)

    def run_cycle(self) -> dict:
        now = self.now()
        for b in self.books:
            self.ledger.ensure_book(b, self.capital, now)
        try:
            dashboard = self.signals() or {}
        except Exception as e:  # no signals: still mark positions to market
            logger.warning("Paper engine: signals unavailable: %s", e)
            dashboard = {}
        if dashboard.get("as_of") and not dashboard.get("provisional"):
            self.ledger.save_snapshot(dashboard, now)
        else:
            # Intraday the live engine recomputes on a still-forming bar; fills
            # use the last completed close instead (signal t → fill t+1).
            dashboard = self.ledger.latest_snapshot() or dashboard
        signal_ok, signal_note = signal_status(dashboard, now)
        targets = {b: TARGET_BUILDERS[b](dashboard) if dashboard else {} for b in self.books}
        if signal_ok:
            for b in self.books:
                self.ledger.replace_targets(b, targets[b], now)

        held = {b: self.ledger.state(b)[0] for b in self.books}
        symbols = sorted({s for b in self.books for s in (*targets[b], *held[b])})
        fresh_marks = self.prices(symbols) if symbols else {}
        self.ledger.upsert_marks(fresh_marks.values())

        session = in_session(now)
        trade = session and signal_ok
        run_id = self.ledger.start_run(now, session, trade, dashboard.get("as_of"))
        orders, skipped = [], []
        if trade:
            for b in self.books:
                orders += self._rebalance(b, targets[b], fresh_marks, now, run_id, skipped)

        all_marks = self.ledger.marks()
        books = {}
        for b in self.books:
            positions, cash, realized, cost = self.ledger.state(b)
            val = valuation(positions, all_marks)
            nav = cash + val["long_value"] + val["short_value"]
            self.ledger.record_nav(b, now, cash=cash, long_value=val["long_value"],
                                   short_value=val["short_value"], nav=nav, realized_pnl=realized,
                                   unrealized_pnl=val["unrealized_pnl"])
            books[b] = {"nav": round(nav, 2), "positions": len(positions), "unpriced": val["unpriced"]}

        if not session:
            from app.data.nse_calendar import market_status

            note = f"{market_status(now)['reason']} — marked to market only; {signal_note}"
        elif not signal_ok:
            note = f"no orders: {signal_note}"
        else:
            note = f"{len(orders)} orders; {signal_note}" + (f"; skipped {len(skipped)} (no fresh price)" if skipped else "")
        self.ledger.finish_run(run_id, self.now(), len(orders), note)
        return {"run_id": run_id, "at": now.isoformat(), "session_open": session, "traded": trade,
                "signal_as_of": dashboard.get("as_of"), "orders": orders, "skipped": skipped,
                "books": books, "note": note}

    def _rebalance(self, book: str, targets: dict[str, Target], marks: dict[str, Mark],
                   now: datetime, run_id: int, skipped: list) -> list[dict]:
        positions, cash, _, _ = self.ledger.state(book)
        val = valuation(positions, {**self.ledger.marks(), **marks})
        nav = cash + val["long_value"] + val["short_value"]
        if nav <= 0:
            return []
        plan = []
        for sym in sorted(set(targets) | set(positions)):
            mark = marks.get(sym)
            if not is_fresh(mark, now):
                skipped.append({"book": book, "symbol": sym, "reason": "no fresh price"})
                continue
            t = targets.get(sym)
            weight = t.weight if t else 0.0
            target_qty = int(weight * nav / mark.price)   # truncates toward zero
            have = positions[sym].quantity if sym in positions else 0
            delta = target_qty - have
            if delta == 0:
                continue
            structural = have == 0 or target_qty == 0 or (have > 0) != (target_qty > 0)
            if not structural:
                threshold = max(MIN_TRADE_FRACTION * nav, DRIFT_FRACTION * abs(target_qty) * mark.price)
                if abs(delta) * mark.price < threshold:
                    continue
            if t:
                reason, ref, as_of = t.reason, t.signal_ref, t.signal_as_of
            else:
                reason = f"Exit: {sym} is no longer in the {book} target book"
                ref, as_of = "exit", None
            plan.append((delta, sym, mark, reason, ref, as_of))
        # Sells first so a long/short rebalance never transiently borrows cash it is about to receive.
        plan.sort(key=lambda p: (p[0] > 0, p[1]))
        out = []
        for delta, sym, mark, reason, ref, as_of in plan:
            out.append(self.ledger.record_order(
                book=book, placed_at=now, symbol=sym, side="BUY" if delta > 0 else "SELL",
                quantity=abs(delta), ref_price=mark.price, cost_bps=COST_BPS, reason=reason,
                signal_ref=ref, signal_as_of=as_of, price_source=mark.source,
                price_observed_at=mark.observed_at, run_id=run_id,
            ))
        return out


# ── Scheduler ─────────────────────────────────────────────────────────────────

class PaperScheduler:
    """
    Keeps the paper engine running: every ``interval_open`` seconds during NSE
    hours, every ``interval_closed`` seconds otherwise (marks only).

    A file lock makes sure only one process trades the ledger (e.g. several
    uvicorn workers, or a reload overlapping the old process).
    """

    def __init__(self, engine_factory: Callable[[], PaperEngine], interval_open: int = 300,
                 interval_closed: int = 1800, lock_path: Path = LOCK_PATH):
        self._factory = engine_factory
        self.interval_open, self.interval_closed = interval_open, interval_closed
        self.lock_path = Path(lock_path)
        self._lock_fh = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._run_lock = threading.Lock()
        self.state = "stopped"
        self.last_result: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.next_run_at: Optional[str] = None
        self._engine: Optional[PaperEngine] = None

    def _acquire(self) -> bool:
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._lock_fh = fh
        return True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self._acquire():
            self.state = "standby"   # another process owns the ledger
            logger.info("Paper scheduler on standby: another process holds %s", self.lock_path)
            return
        self._stop.clear()
        self.state = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True, name="paper-scheduler")
        self._thread.start()
        logger.info("Paper scheduler started (open every %ss, closed every %ss)",
                    self.interval_open, self.interval_closed)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._lock_fh:
            self._lock_fh.close()
            self._lock_fh = None
        self.state = "stopped"

    def engine(self) -> PaperEngine:
        if self._engine is None:
            self._engine = self._factory()
        return self._engine

    def run_now(self) -> dict:
        with self._run_lock:
            try:
                self.last_result = self.engine().run_cycle()
                self.last_error = None
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.exception("Paper cycle failed")
                raise
            return self.last_result

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_now()
            except Exception:
                pass
            wait = self.interval_open if in_session(datetime.now(IST)) else self.interval_closed
            self.next_run_at = (datetime.now(IST) + timedelta(seconds=wait)).isoformat()
            self._wake.wait(timeout=wait)
            self._wake.clear()

    def status(self) -> dict:
        return {"state": self.state, "last_error": self.last_error, "next_run_at": self.next_run_at,
                "interval_open_seconds": self.interval_open, "interval_closed_seconds": self.interval_closed,
                "last_run": {k: v for k, v in (self.last_result or {}).items() if k != "orders"} or None}


paper_scheduler = PaperScheduler(lambda: PaperEngine(PaperLedger()))
