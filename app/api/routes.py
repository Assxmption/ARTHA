"""
FastAPI Routes
==============
REST API endpoints for the Multi-Agent Research Assistant.

Endpoints:
  GET  /                        → Serve frontend SPA
  GET  /health                  → Health check
  POST /api/research            → Start research job
  GET  /api/research/{id}       → Get job status + result
  GET  /api/research/{id}/stream→ SSE live progress stream
  GET  /api/reports             → List all saved reports
  GET  /api/reports/{id}        → Get specific report
  DELETE /api/reports/{id}      → Delete report
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas.models import (
    ResearchRequest,
    ResearchResponse,
    JobStatus,
    ReportSummary,
    ReportListResponse,
    HealthResponse,
)
from app.crew.research_crew import ResearchCrew

router = APIRouter()

# ── In-memory Job Store ────────────────────────────────────────────────────────
# Maps job_id → JobStatus dict
# In Phase 2, replace with Redis
_jobs: dict[str, dict] = {}
_progress_log: dict[str, list[str]] = {}

# Thread pool for crew.kickoff (blocking I/O)
_executor = ThreadPoolExecutor(max_workers=4)
_crew = ResearchCrew()


# ── Health Check ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and configuration summary."""
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        search_provider="Serper (Google)" if serper_key else "DuckDuckGo (fallback)",
        llm_provider=os.getenv("MODEL_PROVIDER", "groq"),
        llm_model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    )


# ── Research Jobs ──────────────────────────────────────────────────────────────

@router.post("/api/research", response_model=ResearchResponse, tags=["Research"])
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new multi-agent research job.
    Returns immediately with a job_id. Poll /api/research/{job_id} for results.
    """
    job_id = str(uuid.uuid4())
    query = request.query.strip()

    # Initialise job state
    _jobs[job_id] = {
        "job_id": job_id,
        "query": query,
        "status": "pending",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "report": None,
        "error": None,
    }
    _progress_log[job_id] = [
        f"[{datetime.utcnow().strftime('%H:%M:%S')}] Job created. Query: {query}"
    ]

    # Run crew in background thread
    background_tasks.add_task(_run_research_job, job_id, query)

    return ResearchResponse(
        job_id=job_id,
        status="pending",
        message="Research job started. Use job_id to track progress.",
        query=query,
    )


@router.get("/api/research/{job_id}", response_model=JobStatus, tags=["Research"])
async def get_research_status(job_id: str):
    """Get the current status and result of a research job."""
    if job_id not in _jobs:
        # Try to load from disk
        saved = _crew.get_report(job_id)
        if saved:
            return JobStatus(
                job_id=job_id,
                query=saved.get("query", ""),
                status=saved.get("status", "completed"),
                started_at=saved.get("started_at"),
                completed_at=saved.get("completed_at"),
                duration_seconds=saved.get("duration_seconds"),
                report=saved.get("report"),
            )
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        query=job["query"],
        status=job["status"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration_seconds=job.get("duration_seconds"),
        report=job.get("report"),
        error=job.get("error"),
        progress_messages=_progress_log.get(job_id, []),
    )


@router.get("/api/research/{job_id}/stream", tags=["Research"])
async def stream_research_progress(job_id: str):
    """
    Server-Sent Events stream for real-time research progress.
    Connect to this endpoint to receive live agent updates.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        last_index = 0

        while True:
            job = _jobs.get(job_id, {})
            messages = _progress_log.get(job_id, [])

            # Send any new messages
            for msg in messages[last_index:]:
                yield {
                    "event": "progress",
                    "data": msg,
                }
                last_index += 1

            # Send status update
            yield {
                "event": "status",
                "data": job.get("status", "unknown"),
            }

            # Stop streaming when done
            if job.get("status") in ("completed", "failed"):
                if job.get("report"):
                    yield {
                        "event": "complete",
                        "data": "Research completed successfully.",
                    }
                elif job.get("error"):
                    yield {
                        "event": "error",
                        "data": job.get("error", "Unknown error"),
                    }
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


# ── Reports Management ─────────────────────────────────────────────────────────

@router.get("/api/reports", response_model=ReportListResponse, tags=["Reports"])
async def list_reports():
    """List all saved research reports."""
    reports_data = _crew.list_reports()
    return ReportListResponse(
        reports=[ReportSummary(**r) for r in reports_data],
        total=len(reports_data),
    )


@router.get("/api/reports/{job_id}", tags=["Reports"])
async def get_report(job_id: str):
    """Retrieve a specific saved report by job ID."""
    report = _crew.get_report(job_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{job_id}' not found.")
    return report


@router.delete("/api/reports/{job_id}", tags=["Reports"])
async def delete_report(job_id: str):
    """Delete a research report and its associated files."""
    deleted = _crew.delete_report(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Report '{job_id}' not found.")
    # Also clean up in-memory store
    _jobs.pop(job_id, None)
    _progress_log.pop(job_id, None)
    return {"message": f"Report {job_id} deleted successfully."}


# ── Background Job Runner ──────────────────────────────────────────────────────

def _on_progress(job_id: str, agent: str, message: str):
    """Callback invoked by ResearchCrew to log progress."""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{agent.upper()}] {message}"
    _progress_log.setdefault(job_id, []).append(log_entry)

    # Update running status
    if job_id in _jobs:
        if "planner" in agent.lower():
            _jobs[job_id]["current_agent"] = "planner"
        elif "researcher" in agent.lower() or "research" in message.lower():
            _jobs[job_id]["current_agent"] = "researcher"
        elif "verif" in agent.lower() or "verif" in message.lower():
            _jobs[job_id]["current_agent"] = "verifier"
        elif "writ" in agent.lower() or "report" in message.lower():
            _jobs[job_id]["current_agent"] = "writer"


async def _run_research_job(job_id: str, query: str):
    """Run the research crew in a background thread."""
    # Mark as running
    _jobs[job_id]["status"] = "running"
    _progress_log[job_id].append(
        f"[{datetime.utcnow().strftime('%H:%M:%S')}] [SYSTEM] Research pipeline starting..."
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _crew.run(
                query=query,
                job_id=job_id,
                on_progress=lambda agent, msg: _on_progress(job_id, agent, msg),
            ),
        )

        # Update job with result
        _jobs[job_id].update({
            "status": result.get("status", "completed"),
            "completed_at": result.get("completed_at"),
            "failed_at": result.get("failed_at"),
            "duration_seconds": result.get("duration_seconds"),
            "report": result.get("report"),
            "error": result.get("error"),
        })

    except Exception as e:
        _jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat(),
        })
        _progress_log[job_id].append(
            f"[{datetime.utcnow().strftime('%H:%M:%S')}] [SYSTEM] Fatal error: {e}"
        )
