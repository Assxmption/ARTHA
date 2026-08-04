"""
ARTHA Analysis Routes — LLM Explainer Pipeline API
=====================================================
REST endpoints for the Fact Store → Agent Crew → Report pipeline.

Endpoints:
  POST /api/analyze              → Start analysis job for an NSE symbol
  GET  /api/analyze/{job_id}     → Get job status + report
  GET  /api/analyze/{job_id}/stream → SSE live progress
  GET  /api/analysis-reports     → List all analysis reports

These endpoints use the AnalysisCrew (direct Python orchestration)
instead of the old CrewAI-based ResearchCrew.

Reference: docs/ARTHA_ARCHITECTURE.md §4.5
"""

from __future__ import annotations

import asyncio
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.crew.analysis_crew import AnalysisCrew
from app.schemas.models import (
    AnalysisJobStatus,
    AnalysisRequest,
    AnalysisResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory job store ────────────────────────────────────────────────────────
_analysis_jobs: dict[str, dict] = {}
_analysis_progress: dict[str, list[str]] = {}

# Thread pool for analysis (blocking I/O + LLM calls)
_executor = ThreadPoolExecutor(max_workers=2)
_crew = AnalysisCrew()


# ── Start Analysis ─────────────────────────────────────────────────────────────

@router.post("/api/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start an ARTHA analysis job for an NSE symbol.

    The pipeline runs:
      Stage 1: Fundamental data collection + quant signal injection
      Stage 2: Quant narration + risk checks (via Fact Store)
      Stage 3: Final report synthesis (deep LLM model)

    Returns immediately with a job_id. Poll /api/analyze/{job_id} for results.
    """
    symbol = request.symbol.upper().strip()
    job_id = f"analysis_{symbol}_{uuid.uuid4().hex[:8]}"

    _analysis_jobs[job_id] = {
        "job_id": job_id,
        "symbol": symbol,
        "status": "pending",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "report": None,
        "report_file": None,
        "stages": None,
        "error": None,
    }
    _analysis_progress[job_id] = [
        f"[{datetime.utcnow().strftime('%H:%M:%S')}] Analysis queued for {symbol}"
    ]

    background_tasks.add_task(
        _run_analysis_job, job_id, symbol, request.skip_llm
    )

    return AnalysisResponse(
        job_id=job_id,
        symbol=symbol,
        status="pending",
        message=f"Analysis job started for {symbol}. Poll /api/analyze/{job_id} for results.",
    )


# ── Get Analysis Status ───────────────────────────────────────────────────────

@router.get("/api/analyze/{job_id}", response_model=AnalysisJobStatus, tags=["Analysis"])
async def get_analysis_status(job_id: str):
    """Get the current status and result of an analysis job."""
    if job_id not in _analysis_jobs:
        # Try loading from saved reports
        report_text = _crew.get_report(job_id)
        if report_text:
            return AnalysisJobStatus(
                job_id=job_id,
                symbol=job_id.split("_")[1] if "_" in job_id else "UNKNOWN",
                status="completed",
                report=report_text,
            )
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found.")

    job = _analysis_jobs[job_id]
    return AnalysisJobStatus(
        job_id=job["job_id"],
        symbol=job["symbol"],
        status=job["status"],
        duration_seconds=job.get("duration_seconds"),
        report=job.get("report"),
        report_file=job.get("report_file"),
        stages=job.get("stages"),
        error=job.get("error"),
        progress_messages=_analysis_progress.get(job_id, []),
    )


# ── SSE Progress Stream ──────────────────────────────────────────────────────

@router.get("/api/analyze/{job_id}/stream", tags=["Analysis"])
async def stream_analysis_progress(job_id: str):
    """
    Server-Sent Events stream for real-time analysis progress.
    Connect to receive live stage updates from the agent crew.
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        last_index = 0

        while True:
            job = _analysis_jobs.get(job_id, {})
            messages = _analysis_progress.get(job_id, [])

            for msg in messages[last_index:]:
                yield {"event": "progress", "data": msg}
                last_index += 1

            yield {"event": "status", "data": job.get("status", "unknown")}

            if job.get("status") in ("completed", "failed"):
                if job.get("report"):
                    yield {"event": "complete", "data": "Analysis completed."}
                elif job.get("error"):
                    yield {"event": "error", "data": job.get("error", "Unknown error")}
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


# ── List Analysis Reports ─────────────────────────────────────────────────────

@router.get("/api/analysis-reports", tags=["Analysis"])
async def list_analysis_reports():
    """List all saved analysis reports."""
    reports = _crew.list_reports()
    return {"reports": reports, "total": len(reports)}


# ── Background Job Runner ─────────────────────────────────────────────────────

def _on_analysis_progress(job_id: str, stage: str, message: str):
    """Callback for live progress updates."""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{stage.upper()}] {message}"
    _analysis_progress.setdefault(job_id, []).append(log_entry)

    if job_id in _analysis_jobs:
        _analysis_jobs[job_id]["current_stage"] = stage


async def _run_analysis_job(job_id: str, symbol: str, skip_llm: bool):
    """Run the analysis crew in a background thread."""
    _analysis_jobs[job_id]["status"] = "running"
    _analysis_progress[job_id].append(
        f"[{datetime.utcnow().strftime('%H:%M:%S')}] [SYSTEM] Pipeline starting..."
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _crew.run(
                symbol=symbol,
                job_id=job_id,
                skip_llm=skip_llm,
                on_progress=lambda stage, msg: _on_analysis_progress(job_id, stage, msg),
            ),
        )

        _analysis_jobs[job_id].update({
            "status": result.get("status", "completed"),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": result.get("duration_seconds"),
            "report": result.get("report"),
            "report_file": result.get("report_file"),
            "stages": result.get("stages"),
        })

    except Exception as e:
        logger.exception("Analysis job %s failed", job_id)
        _analysis_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
        })
        _analysis_progress[job_id].append(
            f"[{datetime.utcnow().strftime('%H:%M:%S')}] [SYSTEM] Fatal error: {e}"
        )
