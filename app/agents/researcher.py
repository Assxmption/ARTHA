"""
Researcher Agent
================
Role: Senior Research Analyst
Responsibility: Execute research tasks using web search and web scraping.
Gathers raw intelligence from multiple sources with source attribution.
"""

from crewai import Agent
from app.config import get_llm
from app.tools.web_search import WebSearchTool
from app.tools.web_scraper import WebScraperTool
from app.tools.file_reader import FileReaderTool


def create_researcher_agent() -> Agent:
    """
    Create and return the Researcher Agent.

    The Researcher executes the plan created by the Planner by
    searching the web, scraping relevant pages, and reading
    local documents to gather comprehensive information.
    """
    llm = get_llm()

    search_tool = WebSearchTool()
    scraper_tool = WebScraperTool()
    file_tool = FileReaderTool()

    return Agent(
        role="Senior Research Analyst",
        goal=(
            "Execute the research plan thoroughly by searching the web for "
            "up-to-date information, reading relevant documentation, and "
            "gathering evidence from authoritative sources. Collect comprehensive "
            "data with clear source attribution (URLs, titles, dates) for every "
            "key finding. Aim for depth, breadth, and accuracy."
        ),
        backstory=(
            "You are a world-class research analyst with deep expertise in "
            "technology, business intelligence, and competitive analysis. "
            "You have a systematic, evidence-based approach: you never state "
            "a claim without backing it with a verifiable source. You are "
            "skilled at identifying the most authoritative sources — official "
            "documentation, peer-reviewed research, expert blogs, and GitHub "
            "repositories. You search deeply, read carefully, and extract "
            "only the most relevant, accurate information. You always cite "
            "your sources with URLs and context so findings can be verified."
        ),
        llm=llm,
        tools=[search_tool, scraper_tool, file_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        max_rpm=10,
    )
