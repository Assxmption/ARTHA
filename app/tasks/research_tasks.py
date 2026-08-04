"""
Research Tasks
==============
Defines the four CrewAI Task objects for the research pipeline.
Each task is bound to a specific agent and passes context forward.

Pipeline: Planning → Research → Verification → Writing
"""

from crewai import Task, Agent


def create_planning_task(planner_agent: Agent, query: str) -> Task:
    """
    Task 1: Research Planning
    Planner Agent analyses the query and produces a structured research plan.
    """
    return Task(
        description=f"""
Analyse the following research query and create a comprehensive research plan:

RESEARCH QUERY: {query}

Your research plan must include:

1. **Query Analysis**: What is the user really asking for? What are the core
   dimensions of this research question?

2. **Research Sub-Tasks**: Break the query into 3-7 specific, targeted sub-tasks.
   Each sub-task should be independently researchable and contribute to a complete answer.
   Format each as:
   - Sub-task N: [Specific research question]
   - Priority: [High/Medium/Low]
   - Search Keywords: [Suggested search terms]
   - Expected Sources: [Documentation/GitHub/News/Blogs/Official sites]

3. **Key Metrics to Investigate**: What specific data points, features, statistics,
   or comparisons are most important?

4. **Potential Information Sources**: List the most authoritative sources to target
   (e.g., official documentation sites, GitHub repos, key blogs/publications).

5. **Research Scope**: Define what is in-scope and out-of-scope to keep the
   research focused and actionable.

Be specific, thorough, and practical. This plan will directly guide the
Research Agent's information gathering.
""",
        expected_output="""
A structured research plan in clear Markdown format containing:
- Query Analysis section (2-3 paragraphs)
- List of 3-7 specific research sub-tasks with priorities and search keywords
- Key metrics and data points to collect
- Recommended sources with URLs where known
- Defined scope boundaries

The plan must be detailed enough for a research agent to execute independently.
""",
        agent=planner_agent,
    )


def create_research_task(
    researcher_agent: Agent,
    query: str,
    planning_task: Task,
) -> Task:
    """
    Task 2: Web Research
    Researcher Agent executes the plan, gathering information with source attribution.
    """
    return Task(
        description=f"""
Execute the research plan from the Planner and gather comprehensive information
about the following topic:

ORIGINAL QUERY: {query}

Follow the research plan precisely. For each sub-task:

1. **Search Thoroughly**: Use web search to find relevant, authoritative sources.
   Perform multiple searches with different keywords if needed.

2. **Read Source Content**: For the most important URLs found, use the web scraper
   to extract full content — don't rely only on search snippets.

3. **Collect with Attribution**: For every key fact or finding, record:
   - The specific claim/data point
   - Source URL
   - Source name/title
   - Approximate date (if available)

4. **Cover All Sub-Tasks**: Ensure every sub-task from the research plan has
   corresponding findings. Don't skip any.

5. **Gather Comparative Data**: If the query involves comparison, collect specific,
   parallel data points for each item being compared.

6. **Include Concrete Examples**: Find real-world use cases, code examples,
   benchmarks, or case studies where relevant.

Present all findings in a structured format with clear source attribution.
""",
        expected_output="""
A comprehensive research report containing:
- Findings organised by research sub-task
- Each finding accompanied by source URL and source name
- Specific data points, statistics, and quotes where found
- Real-world examples and use cases
- At least 8-15 distinct, authoritative sources cited
- A summary list of all sources (title + URL) at the end

Format should be clear Markdown with proper headings and source citations.
""",
        agent=researcher_agent,
        context=[planning_task],
    )


def create_verification_task(
    verifier_agent: Agent,
    query: str,
    research_task: Task,
) -> Task:
    """
    Task 3: Fact Verification
    Verifier Agent validates findings, checks for contradictions and hallucinations.
    """
    return Task(
        description=f"""
Critically review and verify the research findings for the following query:

ORIGINAL QUERY: {query}

Your verification process must:

1. **Validate Key Claims**: For the 5-8 most important claims in the research,
   independently search for corroboration or contradicting evidence.

2. **Detect Contradictions**: Identify any statements in the research that
   contradict each other or that appear to conflict with widely known facts.

3. **Assess Source Credibility**:
   - Rate each major source (Official docs/GitHub = High, Blogs = Medium, etc.)
   - Flag any sources that appear unreliable, outdated (>2 years old), or biased

4. **Check Currency**: Verify that information is up-to-date. Flag anything
   that may have changed significantly since the source was published.

5. **Identify Gaps**: Note important aspects of the query that were not adequately
   covered by the research.

6. **Confidence Assessment**: For each major research finding, provide a
   confidence rating:
   - ✅ HIGH CONFIDENCE: Verified by multiple authoritative sources
   - ⚠️ MEDIUM CONFIDENCE: From one reliable source, plausible but unverified
   - ❌ LOW CONFIDENCE: Unverified, potentially outdated, or conflicting info

7. **Flag Potential Issues**: Note any claims that appear to be hallucinated,
   exaggerated, or unsupported by evidence.
""",
        expected_output="""
A verification report containing:
- Verification status for each key claim (with confidence ratings ✅⚠️❌)
- List of any contradictions or inconsistencies found
- Source credibility assessment for all major sources
- List of information gaps that need acknowledgement in the final report
- Overall quality assessment of the research (1-10 score with rationale)
- Specific recommendations for the Writer Agent on how to handle uncertain claims

Format in clear Markdown with verification status symbols.
""",
        agent=verifier_agent,
        context=[research_task],
    )


def create_writing_task(
    writer_agent: Agent,
    query: str,
    planning_task: Task,
    research_task: Task,
    verification_task: Task,
) -> Task:
    """
    Task 4: Report Writing
    Writer Agent synthesises all findings into the final professional report.
    """
    return Task(
        description=f"""
Synthesise all research and verification findings into a comprehensive,
professional analytical report for the following query:

ORIGINAL QUERY: {query}

Create a polished report in Markdown format with the following structure:

---

# [Report Title — descriptive, specific to the query]

## Executive Summary
A concise 3-5 paragraph overview of the most important findings and
recommendations. Write for a senior executive audience — focus on
significance and actionability, not methodology.

## Key Findings
A bulleted list of the 5-10 most important, specific findings from the research.
Each finding should be concrete, evidence-based, and cited.

## Detailed Analysis

### [Section per major research dimension]
In-depth analysis of each key aspect of the query. Include:
- Specific data, statistics, and examples
- Comparisons where relevant
- Expert perspectives from sources
- Context and implications

## Comparative Analysis (if query involves comparison)
A clear, structured comparison using a table or structured format
covering: features, strengths, weaknesses, use cases, pricing (if relevant),
community/adoption metrics.

## Recommendations
5-8 actionable, prioritised recommendations based on the analysis.
Each recommendation should include:
- The specific recommendation
- Rationale / supporting evidence
- Implementation considerations

## Limitations & Caveats
Honestly acknowledge:
- Information gaps identified in verification
- Areas of uncertainty (use ⚠️ for medium confidence, ❌ for low confidence)
- Scope boundaries of this research

## References
Complete numbered list of all sources cited, formatted as:
[N] Source Title — URL (Accessed: [month year if known])

---

Use the verification report to appropriately qualify uncertain claims.
Only present HIGH and MEDIUM confidence information as fact.
""",
        expected_output="""
A complete, professionally written research report in Markdown format with:
- All required sections (Executive Summary through References)
- Minimum 1500 words of substantive content
- All key claims cited with source references
- A comparative table (if the query involves comparison)
- Numbered references section with all URLs
- Appropriate confidence qualifiers for uncertain claims
- Clear, professional language suitable for executive and technical audiences
""",
        agent=writer_agent,
        context=[planning_task, research_task, verification_task],
    )
