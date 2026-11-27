"""Epoch restatement: archive (never delete), manifest, restart, disclosure."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta

from app.quant import paper_book as pb
from app.quant import paper_report as pr
from tests.test_paper_book import OPEN_DAY, _engine


def test_archive_and_restart(tmp_path):
    eng, clock, feed = _engine(tmp_path)
    eng.run_cycle()
    before = pr.build_report(eng.ledger, now=OPEN_DAY)
    n_orders = before["combined"]["orders_count"]
    assert n_orders > 0
    later = OPEN_DAY + timedelta(hours=6)
    man = eng.ledger.archive_and_restart("Stale-signal bug fixed; books restated for a clean epoch.", later,
                                         archive_dir=tmp_path / "archive", capital=1e7)
    arch = tmp_path / "archive"
    db = next(arch.glob("ledger_epoch1_*.db"))
    assert man["sha256"] == hashlib.sha256(db.read_bytes()).hexdigest()
    assert json.loads(next(arch.glob("*.manifest.json")).read_text())["epoch"] == 1
    # Archive is a complete, readable copy of epoch 1.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    assert con.execute("select count(*) from orders").fetchone()[0] == n_orders
    con.close()
    # The manifest numbers come from the ledger, and match the pre-restart report.
    v = next(b for b in before["books"] if b["book"] == "validated")
    assert man["books"]["validated"]["total_pnl"] == v["total_pnl"]
    # New epoch: every book back at capital, no orders, restatement disclosed.
    after = pr.build_report(eng.ledger, now=later)
    assert after["epoch"]["epoch"] == 2
    assert after["combined"]["orders_count"] == 0
    assert {b["book"] for b in after["books"]} == set(pb.BOOKS)
    assert all(b["nav"] == 1e7 for b in after["books"])
    assert any("Restatement (epoch 2" in n for n in after["notes"])
    assert after["events"][-1]["kind"] == "restatement"
    assert pr.to_pdf(after)[:5] == b"%PDF-"


def test_before_and_after_costs(tmp_path):
    eng, clock, feed = _engine(tmp_path)
    eng.run_cycle()
    rep = pr.build_report(eng.ledger, now=OPEN_DAY)
    for b in rep["books"]:
        assert abs(b["gross_pnl"] - (b["total_pnl"] + b["costs_paid"])) < 0.05
    v = next(b for b in rep["books"] if b["book"] == "validated")
    assert v["costs_paid"] > 0 and v["gross_pnl"] > v["total_pnl"]     # flat prices: before-cost ≈ 0, after-cost < 0
    cols, rows = pr.table(rep, "orders")
    assert ("cost_inr", "Cost ₹") in cols and all(r["cost_inr"] >= 0 for r in rows)


def test_restate_endpoint_requires_operator_and_confirm(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    ledger = pb.PaperLedger(tmp_path / "l.db")
    monkeypatch.setattr(pb.paper_scheduler, "_engine", pb.PaperEngine(ledger, signals=dict, prices=lambda s: {}))
    c = TestClient(app)
    assert c.post("/api/paper/restate", json={"reason": "x" * 30}).status_code == 400   # confirm missing
    assert c.post("/api/paper/restate", json={"reason": "short", "confirm": True}).status_code == 422
