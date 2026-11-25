"""
Paper books API — simulated orders driven by ARTHA's deterministic signals.

No route here (or anywhere it calls) places a real order; Kite is read-only
(AGENTS.md rule 9).  Reports are simulated performance, not advice (rule 11).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.quant import paper_report
from app.quant.paper_book import BOOKS, IST, paper_scheduler
from app.security.auth import optional_user, require_admin

router = APIRouter(prefix="/api/paper", tags=["paper-books"])


def _report() -> dict:
    return paper_report.build_report(paper_scheduler.engine().ledger, scheduler_status=paper_scheduler.status())


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
