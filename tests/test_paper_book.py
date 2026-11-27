"""Paper books: accounting, signal → order translation, session gating, report and exports."""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from app.data import nse_calendar as cal
from app.quant import paper_book as pb
from app.quant import paper_report as pr

IST = pb.IST
OPEN_DAY = datetime(2026, 11, 25, 10, 0, tzinfo=IST)   # Wednesday, regular session
HOLIDAY = datetime(2026, 11, 24, 11, 0, tzinfo=IST)    # Guru Nanak Jayanti


# ── Accounting ────────────────────────────────────────────────────────────────

def test_apply_fill_long_add_reduce_close():
    q, a, r = pb.apply_fill(0, 0.0, 10, 100.0)
    assert (q, a, r) == (10, 100.0, 0.0)
    q, a, r = pb.apply_fill(q, a, 10, 110.0)
    assert (q, a, r) == (20, 105.0, 0.0)
    q, a, r = pb.apply_fill(q, a, -5, 120.0)
    assert (q, a) == (15, 105.0) and r == pytest.approx(75.0)
    q, a, r = pb.apply_fill(q, a, -15, 100.0)
    assert (q, a) == (0, 0.0) and r == pytest.approx(-75.0)


def test_apply_fill_short_and_flip():
    q, a, r = pb.apply_fill(0, 0.0, -10, 200.0)
    assert (q, a, r) == (-10, 200.0, 0.0)
    q, a, r = pb.apply_fill(q, a, 4, 190.0)          # cover 4 at a profit
    assert (q, a) == (-6, 200.0) and r == pytest.approx(40.0)
    q, a, r = pb.apply_fill(q, a, 10, 210.0)         # cover 6 at a loss, open 4 long at 210
    assert (q, a) == (4, 210.0) and r == pytest.approx(-60.0)


def _order(side, qty, fill, ref=None, sym="AAA", at="2026-11-25T10:00:00+05:30"):
    return {"symbol": sym, "side": side, "quantity": qty, "fill_price": fill,
            "ref_price": ref if ref is not None else fill, "placed_at": at}


def test_replay_cash_realized_cost_and_open_dates():
    orders = [
        _order("BUY", 10, 101.0, 100.0, at="t1"),
        _order("SELL", 10, 109.0, 110.0, at="t2"),   # closes: +80 realized
        _order("SELL", 5, 99.0, 100.0, at="t3"),     # opens a short
    ]
    pos, flow, realized, cost = pb.replay(orders)
    assert realized == pytest.approx(80.0)
    assert flow == pytest.approx(-1010 + 1090 + 495)
    assert cost == pytest.approx(10 + 10 + 5)
    assert pos["AAA"].quantity == -5 and pos["AAA"].opened_at == "t3"


# ── Signals → targets ─────────────────────────────────────────────────────────

def _pair(x, y, pos, hedge, validated=True, sharpe=1.3):
    return {"pair": f"{x}/{y}", "x": x, "y": y, "position_after_close": pos, "entry_hedge_ratio": hedge,
            "hedge_ratio": hedge, "z_score": 0.9, "signal_state": "NEUTRAL", "signal_date": "2026-11-23",
            "half_life": 10.7, "train_p_value": 0.0002, "validated": validated,
            "oos": {"sharpe_ratio": sharpe, "max_drawdown": 0.07, "num_trading_days": 504}}


def _dashboard(pairs=None, book=None, scale=0.9, as_of="2026-11-23", provisional=False):
    return {
        "as_of": as_of, "provisional": provisional,
        "pairs": {"pairs": pairs if pairs is not None else [_pair("KOTAKBANK", "PNB", -1, 1.5)]},
        "portfolio": {
            "book_weights": book if book is not None else {"TCS": 0.04, "INFY": -0.02},
            "vt_scale_next_session": scale,
            "current_composition": {
                "momentum_6m": {"long": ["TCS"], "short": [], "next_decision_date": "2026-12-01"},
                "momentum_long_short": {"long": ["TCS"], "short": ["INFY"], "next_decision_date": "2026-12-08"},
            },
        },
    }


def test_validated_targets_follow_backtest_leg_definition():
    t = pb.validated_targets(_dashboard())
    # position −1, h = 1.5: A = −1/2.5, B = +1.5/2.5 of the pair's capital
    assert t["KOTAKBANK"].weight == pytest.approx(-0.4)
    assert t["PNB"].weight == pytest.approx(0.6)
    assert "validated" in t["KOTAKBANK"].reason and t["PNB"].signal_ref == "pair:KOTAKBANK/PNB"
    assert "Exit when |z| ≤ 0.5" in t["PNB"].facts["exit_rule"]


def test_validated_targets_split_capital_and_skip_flat_or_unvalidated():
    d = _dashboard(pairs=[
        _pair("A", "B", 1, 1.0), _pair("C", "D", 0, 1.0), _pair("E", "F", 1, 1.0, validated=False),
    ])
    t = pb.validated_targets(d)
    assert set(t) == {"A", "B"}                       # flat pair holds cash, unvalidated never traded
    assert t["A"].weight == pytest.approx(0.25)       # 1/2 capital per validated pair, 1/(1+1) per leg
    assert t["B"].weight == pytest.approx(-0.25)


def test_shadow_targets_scale_and_sleeve_attribution():
    t = pb.shadow_targets(_dashboard())
    assert t["TCS"].weight == pytest.approx(0.036)
    assert t["INFY"].weight == pytest.approx(-0.018)
    assert t["TCS"].facts["sleeves_long"] == ["momentum_6m", "momentum_long_short"]
    assert "UNVALIDATED" in t["TCS"].reason
    assert pb.shadow_targets(_dashboard(scale=None)) == {}


# ── Engine ────────────────────────────────────────────────────────────────────

class Prices:
    def __init__(self, prices, at):
        self.prices, self.at = dict(prices), at

    def __call__(self, symbols):
        return {s: pb.Mark(s, self.prices[s], self.at, "test") for s in symbols if s in self.prices}


PRICES = {"KOTAKBANK": 500.0, "PNB": 150.0, "TCS": 2500.0, "INFY": 1500.0}


def _engine(tmp_path, dashboard=None, at=OPEN_DAY, prices=None, capital=1_00_00_000.0, signals_as_of=None):
    clock = {"now": at}
    feed = Prices(prices or PRICES, at)
    eng = pb.PaperEngine(pb.PaperLedger(tmp_path / "ledger.db"), signals=lambda: dashboard or _dashboard(),
                         prices=feed, now=lambda: clock["now"], capital=capital, signals_as_of=signals_as_of)
    return eng, clock, feed


def test_cycle_places_orders_matching_targets(tmp_path):
    eng, _, _ = _engine(tmp_path)
    r = eng.run_cycle()
    assert r["traded"] and r["session_open"]
    by = {(o["book"], o["symbol"]): o for o in r["orders"]}
    k = by[("validated", "KOTAKBANK")]
    assert k["side"] == "SELL" and k["quantity"] == int(0.4 * 1e7 / 500)
    assert k["fill_price"] == pytest.approx(500 * (1 - 15 / 10_000))
    p = by[("validated", "PNB")]
    assert p["side"] == "BUY" and p["quantity"] == int(0.6 * 1e7 / 150)
    assert p["fill_price"] == pytest.approx(150 * 1.0015)
    assert p["price_source"] == "test" and p["signal_as_of"] == "2026-11-23"
    assert by[("shadow", "TCS")]["quantity"] == int(0.036 * 1e7 / 2500)
    assert by[("shadow", "INFY")]["side"] == "SELL"
    # Sells are recorded before buys within a book.
    sides = [o["side"] for o in r["orders"] if o["book"] == "validated"]
    assert sides == sorted(sides, key=lambda s: s != "SELL")


def test_cycle_is_idempotent(tmp_path):
    eng, _, _ = _engine(tmp_path)
    assert eng.run_cycle()["orders"]
    assert eng.run_cycle()["orders"] == []


def test_no_orders_on_holiday_weekend_or_after_close(tmp_path):
    for at in (HOLIDAY, datetime(2026, 11, 28, 11, 0, tzinfo=IST), datetime(2026, 11, 25, 16, 0, tzinfo=IST)):
        eng, _, _ = _engine(tmp_path / at.strftime("%d%H"), at=at)
        r = eng.run_cycle()
        assert r["orders"] == [] and not r["session_open"]
        assert eng.ledger.nav_history("validated")    # still marked to market
    eng, _, _ = _engine(tmp_path / "h2", at=HOLIDAY)
    assert "Guru Nanak" in eng.run_cycle()["note"]


def test_stale_price_is_never_a_fill(tmp_path):
    eng, _, feed = _engine(tmp_path)
    feed.at = OPEN_DAY - timedelta(minutes=25)
    r = eng.run_cycle()
    assert r["orders"] == [] and len(r["skipped"]) == 4
    feed.at = datetime(2026, 11, 23, 15, 29, tzinfo=IST)   # yesterday's close
    assert eng.run_cycle()["orders"] == []


def test_provisional_or_old_signals_do_not_trade(tmp_path):
    eng, _, _ = _engine(tmp_path, dashboard=_dashboard(provisional=True))
    assert eng.run_cycle()["orders"] == []      # no completed-close snapshot to fall back on
    eng, _, _ = _engine(tmp_path / "old", dashboard=_dashboard(as_of="2026-11-13"))
    r = eng.run_cycle()
    assert r["orders"] == [] and "2026-11-23 close is required" in r["note"]


def test_intraday_provisional_recompute_uses_last_close_snapshot(tmp_path):
    d = _dashboard()
    eng, clock, feed = _engine(tmp_path, dashboard=d, at=datetime(2026, 11, 23, 18, 0, tzinfo=IST))
    eng.run_cycle()                               # after Monday's close: snapshot saved, no orders
    assert eng.ledger.latest_snapshot()["as_of"] == "2026-11-23"
    # Wednesday session: the live engine now returns a provisional intraday result
    # with different weights. Fills must follow the completed 23-Nov close instead.
    d.update(as_of="2026-11-25", provisional=True)
    d["pairs"]["pairs"][0]["position_after_close"] = 1
    clock["now"] = feed.at = OPEN_DAY
    r = eng.run_cycle()
    k = next(o for o in r["orders"] if o["symbol"] == "KOTAKBANK")
    assert k["side"] == "SELL" and k["signal_as_of"] == "2026-11-23"
    assert r["signal_as_of"] == "2026-11-23"


def test_stale_signals_trigger_last_close_recompute(tmp_path):
    # Server was down on Wednesday: Thursday's session finds only Monday's signals.
    thursday = datetime(2026, 11, 26, 10, 0, tzinfo=IST)
    calls = []

    def recompute(day):
        calls.append(day)
        return _dashboard(as_of=day.isoformat())

    eng, _, _ = _engine(tmp_path, dashboard=_dashboard(as_of="2026-11-23"), at=thursday, signals_as_of=recompute)
    r = eng.run_cycle()
    assert calls == [date(2026, 11, 25)]                       # the last completed session
    assert r["signal_as_of"] == "2026-11-25" and r["orders"]
    assert all(o["signal_as_of"] == "2026-11-25" for o in r["orders"] if o["book"] == "shadow")
    eng.run_cycle()
    assert calls == [date(2026, 11, 25)]                       # cached as a snapshot, not recomputed


def test_stale_signals_never_trade_when_recompute_fails(tmp_path):
    thursday = datetime(2026, 11, 26, 10, 0, tzinfo=IST)

    def broken(day):
        raise RuntimeError("yahoo down")

    eng, _, _ = _engine(tmp_path, dashboard=_dashboard(as_of="2026-11-23"), at=thursday, signals_as_of=broken)
    r = eng.run_cycle()
    assert r["orders"] == [] and "2026-11-25 close is required" in r["note"]


def test_signal_status_requires_last_completed_session():
    wed = datetime(2026, 11, 25, 11, 0, tzinfo=IST)   # Tuesday 24th was a holiday
    assert pb.signal_status({"as_of": "2026-11-23"}, wed)[0] is True
    assert pb.signal_status({"as_of": "2026-11-20"}, wed)[0] is False
    mon = datetime(2026, 11, 30, 10, 0, tzinfo=IST)
    assert pb.signal_status({"as_of": "2026-11-27"}, mon)[0] is True   # Friday's close on Monday
    assert pb.last_completed_session(datetime(2026, 11, 25, 16, 0, tzinfo=IST)) == date(2026, 11, 25)


def test_signals_unavailable_still_marks(tmp_path):
    def boom():
        raise RuntimeError("no data")
    eng = pb.PaperEngine(pb.PaperLedger(tmp_path / "l.db"), signals=boom, prices=Prices(PRICES, OPEN_DAY),
                         now=lambda: OPEN_DAY, signals_as_of=None)
    r = eng.run_cycle()
    assert r["orders"] == [] and "unavailable" in r["note"]


def test_drift_threshold_and_exit(tmp_path):
    d = _dashboard()
    eng, clock, feed = _engine(tmp_path, dashboard=d)
    eng.run_cycle()
    # Small price move: positions drift a little → no churn.
    feed.prices = {**PRICES, "PNB": 151.0}
    clock["now"] = feed.at = OPEN_DAY + timedelta(minutes=5)
    assert [o for o in eng.run_cycle()["orders"] if o["book"] == "validated"] == []
    # Pair exits in the backtest → the book closes both legs, with realized P&L.
    d["pairs"]["pairs"][0]["position_after_close"] = 0
    feed.prices = {**PRICES, "KOTAKBANK": 490.0, "PNB": 155.0}
    clock["now"] = feed.at = OPEN_DAY + timedelta(minutes=10)
    exits = [o for o in eng.run_cycle()["orders"] if o["book"] == "validated"]
    assert {(o["symbol"], o["side"], o["position_after"]) for o in exits} == {("KOTAKBANK", "BUY", 0), ("PNB", "SELL", 0)}
    assert all(o["realized_pnl"] > 0 for o in exits)
    assert all(o["reason"].startswith("Exit:") for o in exits)
    positions, cash, realized, _ = eng.ledger.state("validated")
    assert positions == {} and cash == pytest.approx(1e7 + realized)


def test_nav_identity_total_pnl_is_realized_plus_unrealized(tmp_path):
    eng, clock, feed = _engine(tmp_path)
    eng.run_cycle()
    feed.prices = {k: v * 1.03 for k, v in PRICES.items()}
    clock["now"] = feed.at = OPEN_DAY + timedelta(minutes=5)
    eng.run_cycle()
    rep = pr.build_report(eng.ledger, now=clock["now"])
    for b in rep["books"]:
        assert b["total_pnl"] == pytest.approx(b["realized_pnl"] + b["unrealized_pnl"], abs=0.05)
        assert b["nav"] == pytest.approx(b["cash"] + b["long_value"] + b["short_value"], abs=0.05)
    c = rep["combined"]
    assert c["total_pnl"] == pytest.approx(sum(b["total_pnl"] for b in rep["books"]), abs=0.05)


# ── Report & exports ──────────────────────────────────────────────────────────

@pytest.fixture
def traded_ledger(tmp_path):
    eng, clock, feed = _engine(tmp_path)
    eng.run_cycle()
    feed.prices = {**PRICES, "KOTAKBANK": 495.0, "TCS": 2550.0}
    clock["now"] = feed.at = OPEN_DAY + timedelta(minutes=5)
    eng.run_cycle()
    return eng.ledger, clock["now"]


def test_report_positions_and_potential(traded_ledger):
    ledger, now = traded_ledger
    rep = pr.build_report(ledger, now=now)
    v = next(b for b in rep["books"] if b["book"] == "validated")
    s = next(b for b in rep["books"] if b["book"] == "shadow")
    assert v["validated"] is True and s["validated"] is False
    k = next(p for p in v["positions"] if p["symbol"] == "KOTAKBANK")
    assert k["side"] == "SHORT" and k["unrealized_pnl"] > 0          # short gained as price fell
    assert k["if_closed_now"] == pytest.approx(k["unrealized_pnl"] - k["exit_cost_estimate"], abs=0.02)
    assert k["z_score"] == 0.9 and "Exit when" in k["exit_rule"]
    assert v["metrics"]["status"] == "insufficient_history"
    assert "not investment advice" in rep["disclaimer"].lower()


def test_csv_tables_and_zip(traded_ledger):
    rep = pr.build_report(traded_ledger[0], now=traded_ledger[1])
    rows = list(csv.reader(io.StringIO(pr.to_csv(rep, "orders"))))
    assert rows[0][:5] == ["Order #", "Book", "Placed at (IST)", "Symbol", "Side"]
    assert len(rows) - 1 == rep["combined"]["orders_count"]
    summary = list(csv.reader(io.StringIO(pr.to_csv(rep, "summary"))))
    assert summary[-1][0].startswith("All books")
    with zipfile.ZipFile(io.BytesIO(pr.to_csv_zip(rep))) as z:
        assert set(z.namelist()) == {f"paper_{t}.csv" for t in pr.TABLES} | {"README.txt"}
    with pytest.raises(KeyError):
        pr.table(rep, "nope")


def test_xlsx_has_every_sheet(traded_ledger):
    from openpyxl import load_workbook

    rep = pr.build_report(traded_ledger[0], now=traded_ledger[1])
    wb = load_workbook(io.BytesIO(pr.to_xlsx(rep)))
    assert wb.sheetnames == ["Summary", "Orders", "Open positions", "Realized", "NAV history"]
    assert wb["Orders"].max_row - 1 == rep["combined"]["orders_count"]


def test_pdf_renders_with_and_without_orders(traded_ledger, tmp_path):
    rep = pr.build_report(traded_ledger[0], now=traded_ledger[1])
    assert pr.to_pdf(rep)[:5] == b"%PDF-"
    empty = pb.PaperLedger(tmp_path / "empty.db")
    for b in pb.BOOKS:
        empty.ensure_book(b, 1e7, OPEN_DAY)
    assert pr.to_pdf(pr.build_report(empty, now=OPEN_DAY))[:5] == b"%PDF-"


def test_api_report_and_exports(traded_ledger, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    ledger, now = traded_ledger
    monkeypatch.setattr(pb.paper_scheduler, "_engine", pb.PaperEngine(ledger, signals=dict, prices=lambda s: {}))
    client = TestClient(app)
    rep = client.get("/api/paper/report").json()
    assert rep["combined"]["orders_count"] > 0
    st = client.get("/api/paper/status").json()
    assert st["order_placement"] is False
    for fmt, magic in (("xlsx", b"PK"), ("pdf", b"%PDF"), ("zip", b"PK")):
        r = client.get(f"/api/paper/export/{fmt}")
        assert r.status_code == 200 and r.content.startswith(magic)
        assert "attachment" in r.headers["content-disposition"]
    r = client.get("/api/paper/export/csv?table=positions")
    assert r.status_code == 200 and "Symbol" in r.text
    assert client.get("/api/paper/export/csv?table=bogus").status_code == 400
    assert client.get("/api/paper/export/docx").status_code == 400


def test_scheduler_single_owner(tmp_path):
    lock = tmp_path / "s.lock"
    a = pb.PaperScheduler(lambda: None, lock_path=lock)
    b = pb.PaperScheduler(lambda: None, lock_path=lock)
    assert a._acquire() and not b._acquire()
    a._lock_fh.close()
    assert b._acquire()
    b._lock_fh.close()


def test_paper_modules_have_no_order_placement_path():
    for path in ("app/quant/paper_book.py", "app/quant/paper_report.py", "app/api/paper_routes.py"):
        src = Path(path).read_text()
        for needle in ("place_order", "/orders/", "orders/regular", "modify_order"):
            assert needle not in src, f"{path} references {needle}"


# ── Calendar ──────────────────────────────────────────────────────────────────

def test_nse_calendar():
    assert cal.holiday_name(date(2026, 11, 24)) == "Prakash Gurpurb Sri Guru Nanak Dev"
    assert not cal.is_market_open(HOLIDAY)
    assert cal.is_market_open(OPEN_DAY)
    assert not cal.is_market_open(datetime(2026, 11, 25, 9, 10, tzinfo=IST))
    assert cal.is_market_open(datetime(2026, 2, 1, 14, 0, tzinfo=IST))         # Budget Sunday session
    assert not cal.is_market_open(datetime(2026, 2, 1, 15, 10, tzinfo=IST))    # it closes at 15:00
    assert cal.previous_trading_day(date(2026, 11, 25)) == date(2026, 11, 23)
    assert cal.last_close_boundary(HOLIDAY) == datetime(2026, 11, 23, 15, 30, tzinfo=IST)
    assert cal.market_status(HOLIDAY)["reason"].startswith("NSE closed")
    assert len(cal.NSE_HOLIDAYS) == 15 and all(d.weekday() < 5 for d in cal.NSE_HOLIDAYS)


def test_inr_grouping():
    assert pr.inr_group(12345678.9) == "1,23,45,678.90"
    assert pr.inr_group(-999.5) == "-999.50"
    assert pr.inr_group(100000, 0) == "1,00,000"
    assert pr.inr_group(0) == "0.00"
