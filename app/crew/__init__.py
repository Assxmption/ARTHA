"""
Crew Package (imported lazily: crewai is heavy and only needed for AI reports).
"""

__all__ = ["AnalysisCrew"]


def __getattr__(name):
    if name == "AnalysisCrew":
        from app.crew.analysis_crew import AnalysisCrew

        return AnalysisCrew
    raise AttributeError(name)
