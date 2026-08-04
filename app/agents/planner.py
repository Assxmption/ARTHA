"""
Planner Agent
=============
Role: Research Strategist
Responsibility: Understand user intent and decompose queries into
structured, executable research sub-tasks.
"""

from crewai import Agent
from app.config import get_llm


def create_planner_agent() -> Agent:
    """
    Create and return the Planner Agent.

    The Planner is the entry point of the research pipeline.
    It analyses the user query, identifies key research dimensions,
    and produces a structured research plan for downstream agents.
    """
    llm = get_llm()

    return Agent(
        role="Chief Research Strategist",
        goal=(
            "Analyse the user's research query deeply and decompose it into "
            "3 to 7 specific, actionable research sub-tasks. Each sub-task must "
            "be concrete, targeted, and independently researchable. Produce a "
            "clear research execution plan that guides the Research Agent."
        ),
        backstory=(
            "You are an elite research strategist with 20 years of experience at "
            "top-tier consulting firms like McKinsey, BCG, and Gartner. You have "
            "an exceptional ability to break down complex, multi-faceted research "
            "questions into structured, manageable components. You understand how "
            "to prioritise information gathering, identify knowledge gaps, and "
            "design research plans that lead to comprehensive, unbiased analysis. "
            "Your research plans are renowned for their thoroughness, logical flow, "
            "and practical execution guidance."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
