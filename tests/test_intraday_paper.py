"""Live intraday paper engine: routing, fills, square-off, never overnight."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.quant import intraday_paper as ip
from app.quant import intraday_research as ir
from app.quant import paper_book as pb

IST = pb.IST
SYMS = [f"S{i}" for i in range(15)]


def _frames(day: date, upto_slot: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([datetime.combine(day, ir.SLOT_TIMES[k]) for k in range(upto_slot + 1)])
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.003, (len(idx), len(SYMS))), axis=0)),
                     index=idx, columns=SYMS)
    return c.shift(1).fillna(c.iloc[0]), c


def _hist():
    o, c = [], []
    for i, d in enumerate([date(2026, 11, 23), date(2026, 11, 25), date(2026, 11, 26)]):
        a, b = _frames(d, ir.SLOTS - 1, i)
        o.append(a)
        c.append(b)
    return pd.concat(o), pd.concat(c)


REPORT = {"_file": "x.json", "strategies": {
    "gap_fade": {"validated": False, "train_gross_mean_bp": 0.2},
    "intraday_momentum": {"validated": False, "train_gross_mean_bp": 0.5},
    "xs_reversal_30m": {"validated": False, "train_gross_mean_bp": 0.1},
    "intraday_pairs": {"validated": False, "train_gross_mean_bp": 0.3}}}


def _engine(tmp_path, now, report=REPORT):
    clock = {"now": now}

    def today_bars(symbols, at):
        k = ip.completed_slot(at)
        return _frames(at.date(), k, 9)

    def prices(symbols):
        return {s: pb.Mark(s, 100.0, clock["now"], "test") for s in symbols}

    eng = ip.IntradayEngine(pb.PaperLedger(tmp_path / "l.db"), symbols=lambda: SYMS, now=lambda: clock["now"],
                            today_bars=today_bars, prices=prices, validation=lambda: report, capital=1e7)
    ho, hc = _hist()
    eng._history = lambda symbols, today: (ho, hc)
    eng._pairs_for = lambda today, symbols: [ir.PairSpec(0, 1, 1.0, "S0/S1")]
    eng._splits = {}
    return eng, clock


def test_completed_slot():
    assert ip.completed_slot(datetime(2026, 11, 27, 9, 19, tzinfo=IST)) == -1
    assert ip.completed_slot(datetime(2026, 11, 27, 9, 20, 20, tzinfo=IST)) == 0
    assert ip.completed_slot(datetime(2026, 11, 27, 10, 15, 20, tzinfo=IST)) == 11
    assert ip.completed_slot(datetime(2026, 11, 27, 15, 45, tzinfo=IST)) == ir.SLOTS - 1


def test_admission_rule_uses_train_evidence_only():
    rep = {"strategies": {
        "a": {"validated": True},
        "b": {"validated": False, "train_gross_mean_bp": 0.22, "oos_gross_no_costs": {"sharpe_ratio": -5}},
        "c": {"validated": False, "train_gross_mean_bp": -0.47, "oos_gross_no_costs": {"sharpe_ratio": 9}},
        "d": {"validated": False},
    }}
    assert ip.book_assignment(rep) == {"intraday": ["a"], "intraday_shadow": ["b"], "paused": ["c", "d"]}
    adm = ip.admission(rep)
    assert "train gross +0.22" in adm["b"]["reason"] and adm["c"]["book"] is None   # OOS never consulted
    assert ip.book_assignment(None) == {"intraday": [], "intraday_shadow": [], "paused": []}


def test_paused_strategies_never_trade(tmp_path):
    rep = {"_file": "z.json", "strategies": {"xs_reversal_30m": {"validated": False, "train_gross_mean_bp": -0.7}}}
    eng, _ = _engine(tmp_path, datetime(2026, 11, 27, 10, 15, 20, tzinfo=IST), report=rep)
    assert eng.run_cycle()["orders"] == []


def test_cycle_trades_only_shadow_when_nothing_validated(tmp_path):
    eng, clock = _engine(tmp_path, datetime(2026, 11, 27, 10, 15, 20, tzinfo=IST))
    r = eng.run_cycle()
    assert r["orders"], "expected the shadow strategies to trade at the 10:10 bar"
    assert {o["book"] for o in r["orders"]} == {"intraday_shadow"}
    cm = ir.CostModel()
    for o in r["orders"]:
        expected = 100 * (1 + cm.buy) if o["side"] == "BUY" else 100 * (1 - cm.sell)
        assert o["fill_price"] == pytest.approx(expected, rel=1e-6)
        assert o["signal_ref"].startswith("intraday:")


def test_square_off_by_1520_and_never_overnight(tmp_path):
    eng, clock = _engine(tmp_path, datetime(2026, 11, 27, 10, 15, 20, tzinfo=IST))
    eng.run_cycle()
    assert eng.ledger.state("intraday_shadow")[0], "positions should be open mid-session"
    clock["now"] = datetime(2026, 11, 27, 15, 20, 20, tzinfo=IST)
    r = eng.run_cycle()
    assert r["orders"] and all("Square-off" in o["reason"] for o in r["orders"])
    assert eng.ledger.state("intraday_shadow")[0] == {}


def test_missed_close_is_force_squared_next_cycle(tmp_path):
    eng, clock = _engine(tmp_path, datetime(2026, 11, 27, 10, 15, 20, tzinfo=IST))
    eng.run_cycle()
    clock["now"] = datetime(2026, 11, 27, 19, 0, tzinfo=IST)       # server was down at 15:20
    r = eng.run_cycle()
    assert r["orders"] and all("Forced square-off" in o["reason"] for o in r["orders"])
    assert eng.ledger.state("intraday_shadow")[0] == {}


def test_validated_book_trades_when_a_strategy_passes(tmp_path):
    rep = {"_file": "y.json", "strategies": {"gap_fade": {"validated": True}, "xs_reversal_30m": {"validated": False}}}
    eng, _ = _engine(tmp_path, datetime(2026, 11, 27, 9, 25, 20, tzinfo=IST), report=rep)
    r = eng.run_cycle()
    books = {o["book"] for o in r["orders"]}
    assert "intraday" in books                                       # gap_fade trades at 09:20
    assert all(o["reason"].startswith("gap_fade") for o in r["orders"] if o["book"] == "intraday")
