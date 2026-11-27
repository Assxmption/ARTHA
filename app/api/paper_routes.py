"""
Paper books API — simulated orders driven by ARTHA's deterministic signals.

No route here (or anywhere it calls) places a real order; Kite is read-only
(AGENTS.md rule 9).  Reports are simulated performance, not advice (rule 11).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.quant import paper_report
from app.quant.paper_book import BOOKS, IST, paper_scheduler
from app.security.auth import optional_user, require_admin

router = APIRouter(prefix="/api/paper", tags=["paper-books"])


def _report() -> dict:
    from app.quant.intraday_paper import intraday_scheduler

    status = paper_scheduler.status()
    status["intraday"] = intraday_scheduler.status()
    return paper_report.build_report(paper_scheduler.engine().ledger, scheduler_status=status)


@router.get("/status")
async def status():
    report = await run_in_threadpool(_report)
    return {
        "scheduler": report["scheduler"],
        "order_placement": False,
        "combined": report["combined"],
        "books": [{k: b[k] for k in ("book", "label", "validated", "nav", "total_pnl", "total_return_pct",
                                     "orders_count", "open_positions")} for b in report["books"]],
        "last_runs": report["last_runs"][:5],
    }


@router.get("/report")
async def report():
    return await run_in_threadpool(_report)


@router.post("/run")
async def run_now(user=Depends(require_admin)):
    """Run one engine cycle now (marks always; orders only in session with fresh prices)."""
    try:
        result = await run_in_threadpool(paper_scheduler.run_now)
    except Exception:
        raise HTTPException(500, "Paper cycle failed; see server logs")
    return result


_MEDIA = {
    "csv": "text/csv; charset=utf-8",
    "zip": "application/zip",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf": "application/pdf",
}


@router.get("/export/{fmt}")
async def export(fmt: str, table: str = Query("orders", max_length=20), user=Depends(optional_user)):
    if fmt not in _MEDIA:
        raise HTTPException(400, f"Unknown format {fmt!r}; use csv, zip, xlsx or pdf")
    if fmt == "csv" and table not in paper_report.TABLES:
        raise HTTPException(400, f"Unknown table {table!r}; use one of {', '.join(paper_report.TABLES)}")
    rep = await run_in_threadpool(_report)
    stamp = datetime.now(IST).strftime("%Y%m%d_%H%M")
    if fmt == "csv":
        body, name = ("﻿" + paper_report.to_csv(rep, table)).encode("utf-8"), f"artha_paper_{table}_{stamp}.csv"
    elif fmt == "zip":
        body, name = await run_in_threadpool(paper_report.to_csv_zip, rep), f"artha_paper_csv_{stamp}.zip"
    elif fmt == "xlsx":
        body, name = await run_in_threadpool(paper_report.to_xlsx, rep), f"artha_paper_report_{stamp}.xlsx"
    else:
        body, name = await run_in_threadpool(paper_report.to_pdf, rep), f"artha_paper_report_{stamp}.pdf"
    return Response(body, media_type=_MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


@router.get("/books")
async def books():
    return [{"book": s.key, "label": s.label, "validated": s.validated, "description": s.description}
            for s in BOOKS.values()]


@router.get("/intraday")
async def intraday_status():
    """Intraday engine state and which strategies are validated vs shadow (latest OOS report)."""
    from app.quant.intraday_paper import ADMISSION_RULE, admission, book_assignment, intraday_scheduler, latest_validation

    rep = await run_in_threadpool(latest_validation)
    summary = {}
    for name, r in (rep or {}).get("strategies", {}).items():
        summary[name] = {"validated": r.get("validated"), "description": r.get("description"),
                         "oos": {k: r["oos"].get(k) for k in ("sharpe_ratio", "max_drawdown", "annualized_return",
                                                              "num_trading_days", "profit_factor")}}
    adm = admission(rep)
    for name in summary:
        summary[name]["admission"] = adm.get(name)
    return {"scheduler": intraday_scheduler.status(), "assignment": book_assignment(rep),
            "admission_rule": ADMISSION_RULE,
            "report_file": (rep or {}).get("_file"), "strategies": summary, "order_placement": False}


@router.post("/intraday/run")
async def intraday_run(user=Depends(require_admin)):
    from app.quant.intraday_paper import intraday_scheduler

    try:
        return await run_in_threadpool(intraday_scheduler.run_now)
    except Exception:
        raise HTTPException(500, "Intraday cycle failed; see server logs")


class RestateRequest(BaseModel):
    reason: str = Field(..., min_length=20, max_length=1000)
    confirm: bool = False


@router.post("/restate")
async def restate(req: RestateRequest, user=Depends(require_admin)):
    """
    Archive the current epoch (never deleted; manifest with sha256) and restart
    every book at ₹5 Cr. Operator only; requires confirm=true. Both schedulers
    are paused for the duration so no cycle writes mid-restatement.
    """
    if not req.confirm:
        raise HTTPException(400, "Set confirm=true to archive the ledger and restart all books.")
    from datetime import datetime

    from app.quant.intraday_paper import intraday_scheduler
    from app.quant.paper_book import IST

    def _do():
        with paper_scheduler._run_lock, intraday_scheduler._run_lock:
            return paper_scheduler.engine().ledger.archive_and_restart(req.reason, datetime.now(IST))

    return await run_in_threadpool(_do)


@router.get("/archive")
async def archives():
    """Archived epochs with their manifests (read-only)."""
    import json
    from pathlib import Path

    out = []
    for m in sorted(Path("data_cache/paper/archive").glob("*.manifest.json")):
        try:
            out.append(json.loads(m.read_text()))
        except (OSError, ValueError):
            continue
    return {"archives": out}
