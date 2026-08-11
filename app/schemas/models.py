"""
Pydantic Schemas / Request-Response Models
==========================================
All API request and response data structures.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Response Models ────────────────────────────────────────────────────────────

JobStatusType = Literal["pending", "running", "completed", "failed"]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    search_provider: str
    llm_provider: str
    llm_model: str


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
