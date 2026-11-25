"""
Orchestrator Agent — Thin redirect to AnalysisCrew
====================================================
The original orchestrator stub used a single CrewAI Agent.  The real
pipeline now lives in ``app.crew.analysis_crew.AnalysisCrew``, which
orchestrates Fact Store–centric stages without sequential context-stacking.

This module is kept for backward-compatibility only.  New code should import
``AnalysisCrew`` directly from ``app.crew.analysis_crew``.

Reference: docs/ARTHA_ARCHITECTURE.md §4.5, §4.6
"""

from __future__ import annotations

import logging
import warnings

# Re-export the real orchestrator so callers that do
# ``from app.agents.orchestrator import AnalysisCrew`` still work.
from app.crew.analysis_crew import AnalysisCrew  # noqa: F401

logger = logging.getLogger(__name__)


def create_orchestrator_agent():
    """
    **Deprecated** — use ``AnalysisCrew()`` directly.

    Returns an ``AnalysisCrew`` instance for backward-compatibility.
    The old CrewAI ``Agent`` wrapper has been removed because the
    system now uses direct Python orchestration through the Fact Store
    (AGENTS.md rule 2 — no sequential prompt-stacking).
    """
    warnings.warn(
        "create_orchestrator_agent() is deprecated. "
        "Use AnalysisCrew() directly — see docs/ARTHA_ARCHITECTURE.md §4.5.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info(
        "create_orchestrator_agent() called — returning AnalysisCrew instance. "
        "Please migrate to importing AnalysisCrew directly."
    )
    return AnalysisCrew()
