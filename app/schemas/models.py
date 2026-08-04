"""
Pydantic Schemas / Request-Response Models
==========================================
All API request and response data structures.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Request body for starting a new research job."""
    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research question or topic to investigate",
        examples=[
            "Compare CrewAI and LangGraph for production AI agent systems",
            "Analyze the current state of multimodal AI models in 2025",
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Compare CrewAI and LangGraph for production AI agent systems"
            }
        }


# ── Response Models ────────────────────────────────────────────────────────────

JobStatusType = Literal["pending", "running", "completed", "failed"]


class JobStatus(BaseModel):
    """Status and result of a research job."""
    job_id: str
    query: str
    status: JobStatusType
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    report: Optional[str] = None
    error: Optional[str] = None
    progress_messages: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Response when a research job is submitted."""
    job_id: str
    status: JobStatusType
    message: str
    query: str


class ReportSummary(BaseModel):
    """Summary of a saved report for listing."""
    job_id: str
    query: str
    status: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None


class ReportListResponse(BaseModel):
    """Response for listing all reports."""
    reports: list[ReportSummary]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    search_provider: str
    llm_provider: str
    llm_model: str


class SSEMessage(BaseModel):
    """Server-Sent Events message structure."""
    event: str
    data: str
    job_id: str


# ── ARTHA Analysis (Explainer) Models ──────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request body for starting an ARTHA stock analysis (explainer pipeline)."""
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NSE symbol to analyse (e.g. RELIANCE, HDFCBANK, TCS)",
        examples=["RELIANCE", "HDFCBANK", "TCS", "INFY"],
    )
    start_fy: int = Field(
        default=2022,
        ge=2015,
        le=2030,
        description="First fiscal year to analyse",
    )
    skip_llm: bool = Field(
        default=False,
        description="If True, skip LLM narration and produce data-only report",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "RELIANCE",
                "start_fy": 2022,
                "skip_llm": False,
            }
        }


class AnalysisJobStatus(BaseModel):
    """Status and result of an ARTHA analysis (explainer) job."""
    job_id: str
    symbol: str
    status: JobStatusType
    duration_seconds: Optional[float] = None
    report: Optional[str] = None
    report_file: Optional[str] = None
    stages: Optional[dict] = None
    error: Optional[str] = None
    progress_messages: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Response when an analysis job is submitted."""
    job_id: str
    symbol: str
    status: JobStatusType
    message: str
