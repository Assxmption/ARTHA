"""
Live intraday paper trading — 5-minute strategies on Kite data, simulated fills.

Every order is SIMULATED (AGENTS.md rule 9): Kite is used only to read candles
and quotes.  Not investment advice (rule 11).

Each cycle runs just after a 5-minute bar closes:
  1. builds a cube from cached history (previous sessions) + today's completed
     5-minute bars from Kite (the still-forming bar is dropped);
  2. evaluates the SAME causal strategy functions the research harness
     validated (app/quant/intraday_research.py) and takes their target weights
     at the last completed slot;
  3. routes strategies that passed the latest walk-forward OOS gate to the
     ``intraday`` book and the rest to ``intraday_shadow`` (UNVALIDATED forward
     test); each book splits its capital equally across its strategies;
  4. trades the difference at the live Kite price with side-aware intraday costs.

All positions are flat by 15:20 (``EXIT_SLOT``); if a square-off was missed (the
server was down), the first cycle afterwards closes whatever is still open and
says so in the order reason.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.quant import intraday_research as ir
from app.quant.paper_book import (
    IST, INITIAL_CAPITAL, INTRADAY_BOOKS, Mark, PaperLedger, fetch_marks, in_session, is_fresh, valuation,
)

logger = logging.getLogger(__name__)

REPORT_DIR = Path("docs/backtest_reports")
LOCK_PATH = Path("data_cache/paper/intraday_scheduler.lock")
HISTORY_SESSIONS = 5            # previous sessions loaded for gaps / pair lookbacks
MIN_TRADE_NOTIONAL = 20_000.0   # ignore rebalances smaller than this (₹)


def latest_validation(report_dir: Path = REPORT_DIR) -> Optional[dict]:
    files = sorted(report_dir.glob("intraday_oos_*.json"))
    if not files:
        return None
    rep = json.loads(files[-1].read_text())
    rep["_file"] = str(files[-1])
    return rep


ADMISSION_RULE = (
    "Validated (walk-forward OOS gate) → 'intraday'. Otherwise a strategy is forward-tested in "
    "'intraday_shadow' only if its TRAIN-window gross mean return before costs is > 0 (no OOS data used); "
    "else it is paused. Rule adopted 2026-11-27, after the v1 OOS results were known — disclosed."
)


def admission(report: Optional[dict]) -> dict[str, dict]:
    """Per strategy: {'book': 'intraday'|'intraday_shadow'|None, 'reason': str}."""
    out: dict[str, dict] = {}
    for name, r in (report or {}).get("strategies", {}).items():
        if r.get("validated") is True:
            out[name] = {"book": "intraday", "reason": "Passed the walk-forward OOS gate"}
            continue
        mean = r.get("train_gross_mean_bp")
        if mean is None:
            out[name] = {"book": None, "reason": "Paused: report has no train-window gross evidence"}
        elif mean > 0:
            out[name] = {"book": "intraday_shadow",
                         "reason": f"Failed the gate; train gross {mean:+.2f} bp/day > 0 → forward test (UNVALIDATED)"}
        else:
            out[name] = {"book": None,
                         "reason": f"Paused: failed the gate and train gross {mean:+.2f} bp/day ≤ 0 (no edge before costs)"}
    return out


def book_assignment(report: Optional[dict]) -> dict[str, list[str]]:
    """{'intraday': [...], 'intraday_shadow': [...], 'paused': [...]} per ``admission``."""
    adm = admission(report)
    return {
        "intraday": sorted(n for n, a in adm.items() if a["book"] == "intraday"),
        "intraday_shadow": sorted(n for n, a in adm.items() if a["book"] == "intraday_shadow"),
        "paused": sorted(n for n, a in adm.items() if a["book"] is None),
    }


def completed_slot(now: datetime) -> int:
    """Index of the last 5-minute bar whose close has passed (−1 before 09:20)."""
    t = now.astimezone(IST)
    minutes = (t.hour * 60 + t.minute) - (9 * 60 + 15)
    return min(ir.SLOTS - 1, minutes // 5 - 1)


def _today_bars(symbols: list[str], now: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Today's completed 5-minute bars from Kite (open, close wide frames)."""
    from app.data import intraday_history as ih
    from app.data import kite

    client = kite.get_client()
    start = datetime.combine(now.astimezone(IST).date(), ih.SESSION_START)
    cutoff = pd.Timestamp(now.astimezone(IST).replace(tzinfo=None)) - pd.Timedelta(minutes=5)
    opens, closes = {}, {}
    for s in symbols:
        try:
            f = client.historical(ih._token(client, s), "5minute", start, now.astimezone(IST).replace(tzinfo=None))
        except Exception as e:
            logger.warning("intraday bars for %s unavailable: %s", s, e)
            continue
        if f.empty:
            continue
        f = f[pd.to_datetime(f["date"]) <= cutoff]            # drop the still-forming bar
        idx = pd.DatetimeIndex(f["date"])
        opens[s] = pd.Series(f["open"].values, index=idx)
        closes[s] = pd.Series(f["close"].values, index=idx)
    return pd.DataFrame(opens), pd.DataFrame(closes)


class IntradayEngine:
    def __init__(
        self,
        ledger: PaperLedger,
        symbols: Callable[[], list[str]],
        now: Callable[[], datetime] = lambda: datetime.now(IST),
        today_bars: Callable[[list[str], datetime], tuple[pd.DataFrame, pd.DataFrame]] = _today_bars,
        prices: Callable[[list[str]], dict[str, Mark]] = fetch_marks,
        validation: Callable[[], Optional[dict]] = latest_validation,
        capital: float = INITIAL_CAPITAL,
    ):
        self.ledger, self.symbols, self.now = ledger, symbols, now
        self.today_bars, self.prices, self.validation = today_bars, prices, validation
        self.capital = capital
        self._pairs: dict[date, list[ir.PairSpec]] = {}
        self._splits: Optional[dict] = None

    # history ------------------------------------------------------------------
    def _history(self, symbols: list[str], today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        from app.data import intraday_history as ih

        start = today - timedelta(days=HISTORY_SESSIONS * 2 + 7)
        o = ih.panel(symbols, "open", start=start, end=today - timedelta(days=1))
        c = ih.panel(symbols, "close", start=start, end=today - timedelta(days=1))
        return o, c

    def _pairs_for(self, today: date, symbols: list[str]) -> list[ir.PairSpec]:
        if today not in self._pairs:
            from app.data import intraday_history as ih

            # Walk-forward: re-select on the latest ~500 sessions before today.
            cube = ir.build_cube(ih.panel(symbols, "open", end=today - timedelta(days=1)),
                                 ih.panel(symbols, "close", end=today - timedelta(days=1)))
            train = cube.shape[0]
            self._pairs = {today: ir.select_pairs(cube, train) if train > 120 else []}
        return self._pairs[today]

    # cycle --------------------------------------------------------------------
    def run_cycle(self) -> dict:
        now = self.now()
        today = now.astimezone(IST).date()
        for b in INTRADAY_BOOKS:
            self.ledger.ensure_book(b, self.capital, now)
        held = {b: self.ledger.state(b)[0] for b in INTRADAY_BOOKS}
        report = self.validation()
        assign = book_assignment(report)
        session = in_session(now)
        k = completed_slot(now)
        run_id = self.ledger.start_run(now, session, session, f"intraday:{report.get('_file','') if report else 'none'}")

        # Outside the session: make sure nothing is carried overnight.
        if not session or k >= ir.EXIT_SLOT - 1:
            orders = self._square_off(held, now, run_id,
                                      "Square-off: intraday positions are closed by 15:20"
                                      if session else "Forced square-off: session close was missed")
            self._snapshot(now)
            note = (f"[intraday] square-off, {len(orders)} orders" if orders
                    else "[intraday] outside trading window — flat")
            self.ledger.finish_run(run_id, self.now(), len(orders), note)
            return {"at": now.isoformat(), "orders": orders, "note": note, "assignment": assign}

        if k < 0:
            self.ledger.finish_run(run_id, self.now(), 0, "[intraday] waiting for the first bar")
            return {"at": now.isoformat(), "orders": [], "note": "waiting for the first bar", "assignment": assign}

        symbols = self.symbols()
        ho, hc = self._history(symbols, today)
        to, tc = self.today_bars(symbols, now)
        if tc.empty:
            self.ledger.finish_run(run_id, self.now(), 0, "[intraday] no bars from Kite yet")
            return {"at": now.isoformat(), "orders": [], "note": "no bars", "assignment": assign}
        o = pd.concat([ho, to]).sort_index()
        c = pd.concat([hc, tc]).sort_index()
        o, c = o[~o.index.duplicated(keep="last")], c[~c.index.duplicated(keep="last")]
        cube = ir.build_cube(o, c, partial_last_day=True)
        if not cube.days or cube.days[-1] != today:
            self.ledger.finish_run(run_id, self.now(), 0, "[intraday] today's bars incomplete")
            return {"at": now.isoformat(), "orders": [], "note": "incomplete", "assignment": assign}
        if self._splits is None:
            from app.data import intraday_history as ih

            try:
                self._splits = ih.split_dates(cube.symbols)
            except Exception:
                self._splits = {}
        ca = ir.corporate_action_mask(cube, self._splits)
        last_k = int(np.max(np.where(np.isfinite(cube.close[-1]).any(axis=1))[0]))
        pairs = self._pairs_for(today, cube.symbols)

        targets: dict[str, dict[str, tuple[float, str]]] = {}
        for book, names in assign.items():
            if book not in INTRADAY_BOOKS:
                continue
            tw: dict[str, tuple[float, str]] = {}
            for name in names:
                W = ir.strategy_weights(name, cube, ca, pairs)
                w = W[-1, last_k] / max(1, len(names))
                for j in np.nonzero(np.abs(w) > 1e-12)[0]:
                    sym = cube.symbols[j]
                    prev, why = tw.get(sym, (0.0, ""))
                    tw[sym] = (prev + float(w[j]), (why + " + " if why else "") + name)
            targets[book] = tw

        want = sorted({s for b in targets for s in targets[b]} | {s for b in held for s in held[b]})
        marks = self.prices(want) if want else {}
        self.ledger.upsert_marks(marks.values())
        orders = []
        for book in INTRADAY_BOOKS:
            orders += self._rebalance(book, targets.get(book, {}), marks, now, run_id, last_k)
        self._snapshot(now)
        slot_t = ir.SLOT_TIMES[last_k]
        note = (f"[intraday] {len(orders)} orders at the {slot_t:%H:%M} bar; validated: "
                f"{', '.join(assign['intraday']) or 'none'}; shadow: {', '.join(assign['intraday_shadow']) or 'none'}")
        self.ledger.finish_run(run_id, self.now(), len(orders), note)
        return {"at": now.isoformat(), "orders": orders, "note": note, "assignment": assign, "slot": str(slot_t)}

    def _rebalance(self, book, targets, marks, now, run_id, slot) -> list[dict]:
        positions, cash, _, _ = self.ledger.state(book)
        val = valuation(positions, {**self.ledger.marks(), **marks})
        nav = cash + val["long_value"] + val["short_value"]
        cm = ir.CostModel()
        plan = []
        for sym in sorted(set(targets) | set(positions)):
            mark = marks.get(sym)
            if not is_fresh(mark, now):
                continue
            w, why = targets.get(sym, (0.0, ""))
            target_qty = int(w * nav / mark.price)
            have = positions[sym].quantity if sym in positions else 0
            delta = target_qty - have
            if delta == 0 or (target_qty != 0 and abs(delta) * mark.price < MIN_TRADE_NOTIONAL):
                continue
            reason = (f"{why} target {w:+.4%} of book at the {ir.SLOT_TIMES[slot]:%H:%M} bar"
                      if why else f"Exit: no {book} strategy holds {sym} after the {ir.SLOT_TIMES[slot]:%H:%M} bar")
            plan.append((delta, sym, mark, reason, why or "exit"))
        plan.sort(key=lambda p: (p[0] > 0, p[1]))
        out = []
        for delta, sym, mark, reason, ref in plan:
            side = "BUY" if delta > 0 else "SELL"
            out.append(self.ledger.record_order(
                book=book, placed_at=now, symbol=sym, side=side, quantity=abs(delta), ref_price=mark.price,
                cost_bps=(cm.buy if side == "BUY" else cm.sell) * 1e4, reason=reason,
                signal_ref=f"intraday:{ref}", signal_as_of=now.astimezone(IST).date().isoformat(),
                price_source=mark.source, price_observed_at=mark.observed_at, run_id=run_id,
            ))
        return out

    def _square_off(self, held, now, run_id, reason) -> list[dict]:
        syms = sorted({s for b in held for s in held[b]})
        if not syms:
            return []
        marks = self.prices(syms)
        self.ledger.upsert_marks(marks.values())
        known = {**self.ledger.marks(), **marks}
        cm = ir.CostModel()
        out = []
        for book, positions in held.items():
            for sym, p in sorted(positions.items()):
                m = known.get(sym)
                if m is None:
                    continue
                side = "SELL" if p.quantity > 0 else "BUY"
                out.append(self.ledger.record_order(
                    book=book, placed_at=now, symbol=sym, side=side, quantity=abs(p.quantity), ref_price=m.price,
                    cost_bps=(cm.buy if side == "BUY" else cm.sell) * 1e4, reason=reason,
                    signal_ref="intraday:square-off", signal_as_of=now.astimezone(IST).date().isoformat(),
                    price_source=m.source, price_observed_at=m.observed_at, run_id=run_id,
                ))
        return out

    def _snapshot(self, now) -> None:
        marks = self.ledger.marks()
        for b in INTRADAY_BOOKS:
            positions, cash, realized, _ = self.ledger.state(b)
            val = valuation(positions, marks)
            self.ledger.record_nav(b, now, cash=cash, long_value=val["long_value"], short_value=val["short_value"],
                                   nav=cash + val["long_value"] + val["short_value"], realized_pnl=realized,
                                   unrealized_pnl=val["unrealized_pnl"])


def _universe() -> list[str]:
    from app.data.market_movers import universe_symbols

    return universe_symbols()


class IntradayScheduler:
    """Runs a cycle 20 s after every 5-minute bar close during the session (single owner via file lock)."""

    def __init__(self, engine_factory: Callable[[], IntradayEngine], lock_path: Path = LOCK_PATH):
        self._factory, self.lock_path = engine_factory, Path(lock_path)
        self._engine: Optional[IntradayEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock_fh = None
        self._run_lock = threading.Lock()
        self.state, self.last_result, self.last_error, self.next_run_at = "stopped", None, None, None

    def engine(self) -> IntradayEngine:
        if self._engine is None:
            self._engine = self._factory()
        return self._engine

    def start(self) -> None:
        import fcntl

        if self._thread and self._thread.is_alive():
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            self.state = "standby"
            return
        self._lock_fh = fh
        self._stop.clear()
        self.state = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True, name="intraday-paper")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._lock_fh:
            self._lock_fh.close()
            self._lock_fh = None
        self.state = "stopped"

    def run_now(self) -> dict:
        with self._run_lock:
            try:
                self.last_result = self.engine().run_cycle()
                self.last_error = None
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.exception("Intraday cycle failed")
                raise
            return self.last_result

    @staticmethod
    def _next_wake(now: datetime) -> datetime:
        if not in_session(now):
            return now + timedelta(minutes=15)
        base = now.replace(second=20, microsecond=0)
        mins = (5 - base.minute % 5) % 5 or 5
        nxt = base + timedelta(minutes=mins)
        return nxt if nxt > now else nxt + timedelta(minutes=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_now()
            except Exception:
                pass
            now = datetime.now(IST)
            wake = self._next_wake(now)
            self.next_run_at = wake.isoformat()
            self._stop.wait(timeout=max(5.0, (wake - now).total_seconds()))

    def status(self) -> dict:
        return {"state": self.state, "last_error": self.last_error, "next_run_at": self.next_run_at,
                "last_run": {k: v for k, v in (self.last_result or {}).items() if k != "orders"} or None}


intraday_scheduler = IntradayScheduler(lambda: IntradayEngine(PaperLedger(), _universe))
