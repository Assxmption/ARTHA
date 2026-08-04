"""
Fact Store Package
==================
Typed, shared state for the ARTHA agent crew.
Every agent reads and writes Pydantic-typed facts here instead of
passing growing prompt transcripts between stages.
"""

from app.factstore.schemas import (
    FundamentalRow,
    QuantSignal,
    RiskFlag,
    NewsSignal,
    NarrativeFact,
    ReportCitation,
)
from app.factstore.store import FactStore

__all__ = [
    "FundamentalRow",
    "QuantSignal",
    "RiskFlag",
    "NewsSignal",
    "NarrativeFact",
    "ReportCitation",
    "FactStore",
]
