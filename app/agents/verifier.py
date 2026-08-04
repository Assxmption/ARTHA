"""
Verifier Agent
==============
Role: Fact-Checking & Quality Assurance Specialist
Responsibility: Validate research findings, detect contradictions,
assess source credibility, and flag potential hallucinations.
"""

from crewai import Agent
from app.config import get_llm
from app.tools.web_search import WebSearchTool


def create_verifier_agent() -> Agent:
    """
    Create and return the Verifier Agent.

    The Verifier critically analyses the research output for accuracy,
    consistency, and credibility. It cross-validates claims using
    additional searches and provides a quality assessment.
    """
    llm = get_llm()

    search_tool = WebSearchTool()

    return Agent(
        role="Chief Verification & Fact-Checking Specialist",
        goal=(
            "Critically review all research findings for accuracy, consistency, "
            "and credibility. Cross-validate key claims using independent searches. "
            "Identify any contradictions, outdated information, or unverified "
            "assertions. Flag low-confidence claims and provide an overall "
            "quality assessment with confidence ratings for major findings."
        ),
        backstory=(
            "You are a rigorous fact-checker and quality assurance specialist "
            "with extensive experience at leading fact-checking organisations "
            "and research institutions. You have a forensic eye for detail and "
            "an unwavering commitment to accuracy. You understand how AI models "
            "can hallucinate, how sources can be biased, and how information "
            "can become outdated. You systematically cross-reference claims "
            "against multiple independent sources, challenge assumptions, and "
            "produce clear quality assessments that include confidence levels. "
            "Your verification reports help readers understand exactly how "
            "reliable each piece of information is."
        ),
        llm=llm,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        max_rpm=8,
    )
