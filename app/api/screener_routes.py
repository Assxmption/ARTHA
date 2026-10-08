"""
Screener API Routes
====================
REST endpoints for Screener.in-style stock filtering.

Endpoints:
  POST /api/screener/query     → Apply filters, return matching stocks
  GET  /api/screener/metrics   → Available metrics catalog (for dropdowns)
  GET  /api/screener/sectors   → Unique sectors in cache
  POST /api/screener/refresh   → Force cache refresh
  GET  /api/screener/info      → Cache metadata (age, count)

All data is from deterministic yfinance sources — no LLM computation.
AGENTS.md Rule 1: No LLM computes or asserts a number.
AGENTS.md Rule 11: This is informational, not investment advice.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.data.screener_cache import (
    ScreenerFilter,
    query_screener,
    get_available_metrics,
    get_sectors,
    get_cache_info,
    build_screener_cache,
    refresh_cache,
    VALID_OPERATORS,
    METRIC_BY_KEY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/screener", tags=["screener"])


# ── Request/Response Models ────────────────────────────────────────────────────

class FilterItem(BaseModel):
    """A single filter condition."""
    metric: str = Field(..., description="Metric key, e.g. 'pe', 'roe'")
    operator: str = Field(
        ..., description="Comparison operator: gt, lt, gte, lte, eq, between"
    )
    value: float = Field(..., description="Threshold value")
    value2: Optional[float] = Field(
        None, description="Second value for 'between' operator"
    )


class ScreenerRequest(BaseModel):
    """Request body for stock screening."""
    filters: list[FilterItem] = Field(
        default_factory=list, description="AND-combined filter conditions"
    )
    sort_by: str = Field("market_cap", description="Metric key to sort by")
    sort_desc: bool = Field(True, description="Sort descending")
    limit: int = Field(50, ge=1, le=200, description="Max results")
    sector: Optional[str] = Field(None, description="Sector filter (exact match)")


# ── Preset Screens ──────────────────────────────────────────────────────────────

PRESET_SCREENS = {
    "value_picks": {
        "name": "Value Picks",
        "description": "Low PE, high ROE, low debt",
        "filters": [
            {"metric": "pe", "operator": "lt", "value": 15},
            {"metric": "roe", "operator": "gt", "value": 15},
            {"metric": "debt_to_equity", "operator": "lt", "value": 1.0},
        ],
        "sort_by": "roe",
    },
    "growth_stocks": {
        "name": "Growth Stocks",
        "description": "High revenue growth, profitable",
        "filters": [
            {"metric": "revenue_growth", "operator": "gt", "value": 20},
            {"metric": "operating_margin", "operator": "gt", "value": 15},
        ],
        "sort_by": "revenue_growth",
    },
    "dividend_champions": {
        "name": "Dividend Champions",
        "description": "High yield, low payout",
        "filters": [
            {"metric": "dividend_yield", "operator": "gt", "value": 2},
            {"metric": "pe", "operator": "lt", "value": 25},
        ],
        "sort_by": "dividend_yield",
    },
    "quality_largecap": {
        "name": "Quality Large-Cap",
        "description": "Big market cap, high ROE, low debt",
        "filters": [
            {"metric": "market_cap", "operator": "gt", "value": 50000},
            {"metric": "roe", "operator": "gt", "value": 20},
            {"metric": "debt_to_equity", "operator": "lt", "value": 0.5},
        ],
        "sort_by": "market_cap",
    },
    "low_pe_profitable": {
        "name": "Low PE & Profitable",
        "description": "PE under 12, positive margins",
        "filters": [
            {"metric": "pe", "operator": "lt", "value": 12},
            {"metric": "profit_margin", "operator": "gt", "value": 5},
            {"metric": "pe", "operator": "gt", "value": 0},
        ],
        "sort_by": "pe",
        "sort_desc": False,
    },
    "high_roe_moat": {
        "name": "High ROE Moat",
        "description": "Consistently high returns on equity",
        "filters": [
            {"metric": "roe", "operator": "gt", "value": 25},
            {"metric": "operating_margin", "operator": "gt", "value": 20},
        ],
        "sort_by": "roe",
    },
}


# ── Routes ──────────────────────────────────────────────────────────────────────

@router.post("/query")
async def screen_stocks(request: ScreenerRequest):
    """
    Apply filters to the screener cache and return matching stocks.

    Filters are AND-combined. Example:
      filters: [
        {"metric": "pe", "operator": "lt", "value": 20},
        {"metric": "roe", "operator": "gt", "value": 15}
      ]
    """
    # Validate filters
    for f in request.filters:
        if f.metric not in METRIC_BY_KEY:
            raise HTTPException(
                400,
                f"Unknown metric: '{f.metric}'. "
                f"Available: {list(METRIC_BY_KEY.keys())}",
            )
        if f.operator not in VALID_OPERATORS:
            raise HTTPException(
                400,
                f"Invalid operator: '{f.operator}'. "
                f"Available: {list(VALID_OPERATORS)}",
            )
        if f.operator == "between" and f.value2 is None:
            raise HTTPException(
                400,
                f"'between' operator requires 'value2' field",
            )

    # Convert to internal filter objects
    filters = [
        ScreenerFilter(
            metric=f.metric,
            operator=f.operator,
            value=f.value,
            value2=f.value2,
        )
        for f in request.filters
    ]

    results, total = query_screener(
        filters=filters,
        sort_by=request.sort_by,
        sort_desc=request.sort_desc,
        limit=request.limit,
        sector=request.sector,
    )

    cache_info = get_cache_info()

    return {
        "results": results,
        "total_matches": total,
        "returned": len(results),
        "cache_age_hours": cache_info.get("cache_age_hours", 0),
        "cache_total_symbols": cache_info.get("total_symbols", 0),
        "applied_filters": [
            {"metric": f.metric, "operator": f.operator, "value": f.value}
            for f in request.filters
        ],
    }


@router.get("/metrics")
async def list_metrics():
    """Return available metrics for the filter dropdown."""
    return {"metrics": get_available_metrics()}


@router.get("/sectors")
async def list_sectors():
    """Return unique sectors in the screener cache."""
    sectors = get_sectors()
    return {"sectors": sectors}


@router.get("/presets")
async def list_presets():
    """Return preset screen configurations."""
    return {"presets": PRESET_SCREENS}


@router.get("/info")
async def cache_status():
    """Return cache metadata: total symbols, age, timestamps."""
    return get_cache_info()


@router.post("/refresh")
async def force_refresh(background_tasks: BackgroundTasks):
    """
    Trigger a cache refresh in the background.

    Only re-fetches stale entries (older than 24 hours).
    """
    background_tasks.add_task(refresh_cache)
    return {
        "status": "refresh_started",
        "message": "Cache refresh is running in the background. "
                   "Check /api/screener/info for progress.",
    }


@router.post("/build")
async def build_cache(background_tasks: BackgroundTasks):
    """
    Build the screener cache from scratch (first time or full rebuild).

    This fetches data for all ~200 NIFTY symbols and may take 2-3 minutes.
    """
    background_tasks.add_task(build_screener_cache)
    return {
        "status": "build_started",
        "message": "Building screener cache for ~200 symbols. "
                   "This may take 2-3 minutes. Check /api/screener/info.",
    }
