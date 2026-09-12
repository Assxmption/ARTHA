"""
Fundamentals Agent — Screener++ Row-Level Analysis
===================================================
Retrieves structured fundamentals from the data layer, writes
FundamentalRow facts to the Fact Store, then produces a Harvey-style
descriptive narrative explaining what the numbers mean.

Non-negotiable constraints (AGENTS.md):
  - Rule 1: NEVER computes or asserts a number.  All numeric values
    come from the data layer and are written as FundamentalRow facts
    with a non-empty `source` field.
  - Rule 10: Every narrative claim must cite a specific fact_id.

Uses cheap model (AGENTS.md rule 3).

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Fundamentals Agent row)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from crewai import Agent

from app.data.fundamentals import get_full_fundamentals, fundamentals_to_dataframe
from app.factstore.schemas import FundamentalRow, NarrativeFact, FactType
from app.factstore.store import FactStore
from app.llm.router import get_cheap_llm

logger = logging.getLogger(__name__)


def create_fundamentals_agent() -> Agent:
    """
    Create the ARTHA Fundamentals Agent.

    This agent retrieves row-level structured fundamentals and writes
    them to the Fact Store, then narrates what the numbers mean.
    """
    llm = get_cheap_llm()

    return Agent(
        role="ARTHA Fundamentals Analyst",
        goal=(
            "Retrieve row-level structured fundamentals for the target "
            "NSE company and fiscal years. Write each metric as a "
            "FundamentalRow fact to the Fact Store. Then produce a "
            "narrative that explains year-over-year trends and what "
            "the numbers mean for the company's financial health. "
            "CRITICAL: Never invent or estimate a number — only narrate "
            "values that already exist as FundamentalRow facts. Every "
            "numeric claim in your narrative must reference a specific "
            "fact_id."
        ),
        backstory=(
            "You are a senior equity research analyst specialising in "
            "Indian markets (NSE). You have deep expertise in reading "
            "annual reports and financial statements for Indian companies. "
            "Your analysis style follows the Screener.in tradition of "
            "row-level rigor — every number traceable to a specific "
            "reported figure — combined with a Harvey AI–style narrative "
            "layer that explains what the numbers actually mean for the "
            "company, not just displaying them. You never fabricate "
            "financial data. If a metric is unavailable, you say so "
            "explicitly rather than estimating."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


# ── Fact Store integration ──────────────────────────────────────────────────────

def run_fundamentals_analysis(
    symbol: str,
    job_id: str,
    store: FactStore,
    start_fy: int = 2022,
    end_fy: Optional[int] = None,
) -> dict:
    """
    Execute the FULL fundamentals analysis pipeline:
      1. Retrieve financial statement data (income, balance sheet, cash flow)
         from yfinance — revenue, EBITDA, net income, EPS, debt, equity,
         operating/net margins, ROE, current ratio, operating cash flow, etc.
      2. Compute market-derived metrics (returns, volatility, volume, prices)
         from OHLCV data — always available as supplement.
      3. Write ALL FundamentalRow facts to the Fact Store (immediate checkpoint).
      4. Build a Screener-style summary table.
      5. Return the facts and table for narration by the agent.

    This function is the deterministic data-gathering step.
    The LLM narration happens separately, downstream, using only the
    facts written here.

    Parameters
    ----------
    symbol : str
        NSE symbol.
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    start_fy : int
        First fiscal year to analyse.
    end_fy : int, optional
        Last fiscal year.  Defaults to current FY.

    Returns
    -------
    dict
        Keys: 'facts' (list[FundamentalRow]), 'table' (pd.DataFrame),
        'symbol', 'fiscal_years', 'statement_count', 'market_count'.
    """
    if end_fy is None:
        # Current Indian fiscal year
        today = date.today()
        end_fy = today.year if today.month >= 4 else today.year

    # ── Use get_full_fundamentals for BOTH statement + market data ──────
    try:
        all_facts = get_full_fundamentals(symbol, start_fy, end_fy)
    except Exception as e:
        logger.error("Full fundamentals pipeline failed for %s: %s", symbol, e)
        all_facts = []

    # Bind all facts to this job
    for fact in all_facts:
        fact.job_id = job_id

    # Count what we got
    stmt_count = sum(1 for f in all_facts if "YFINANCE" in f.source)
    market_count = sum(1 for f in all_facts if "OHLCV" in f.source)
    logger.info(
        "Fundamentals for %s: %d statement metrics + %d market metrics = %d total",
        symbol, stmt_count, market_count, len(all_facts),
    )

    # ── Per-stage checkpoint: write facts immediately ───────────────────
    if all_facts:
        store.put_facts(all_facts)
        logger.info(
            "Wrote %d FundamentalRow facts for %s (FY%d–FY%d) to Fact Store",
            len(all_facts), symbol, start_fy, end_fy,
        )

    # ── Build summary table ────────────────────────────────────────────
    table = fundamentals_to_dataframe(all_facts)

    return {
        "facts": all_facts,
        "table": table,
        "symbol": symbol,
        "fiscal_years": list(range(start_fy, end_fy + 1)),
        "statement_count": stmt_count,
        "market_count": market_count,
    }


def format_facts_for_narration(facts: list[FundamentalRow]) -> str:
    """
    Format FundamentalRow facts into a structured text block suitable
    for inclusion in an LLM prompt.

    Each fact includes its fact_id so the LLM can reference it in
    its narrative (traceability requirement).
    """
    if not facts:
        return "No fundamental data available."

    lines = ["## Available Fundamental Data\n"]
    lines.append("| Fact ID | Symbol | FY | Metric | Value | Unit | Source |")
    lines.append("|---------|--------|----|--------|-------|------|--------|")

    for f in sorted(facts, key=lambda x: (x.symbol, x.fiscal_year, x.metric)):
        val_str = f"{f.value:.2f}" if f.value is not None else "N/A"
        lines.append(
            f"| {f.fact_id[:8]}… | {f.symbol} | {f.fiscal_year} "
            f"| {f.metric} | {val_str} | {f.unit or '—'} | {f.source} |"
        )

    lines.append(
        "\n**CRITICAL**: When narrating these numbers, you MUST reference "
        "the Fact ID of each value you mention. Do not invent or estimate "
        "any numbers not present in this table."
    )
    return "\n".join(lines)


# ── Metric Categories ───────────────────────────────────────────────────────────

# Grouping metrics by category for structured narration.
# The LLM receives facts organized by these categories, which produces
# a more coherent narrative than a flat alphabetical list.
_METRIC_CATEGORIES: dict[str, list[str]] = {
    "Profitability": [
        "revenue", "gross_profit", "operating_income", "net_income",
        "ebitda", "operating_margin", "net_margin", "gross_margin",
    ],
    "Returns": [
        "roe", "roce", "roa", "roic",
    ],
    "Leverage & Liquidity": [
        "total_debt", "net_debt", "debt_to_equity", "current_ratio",
        "cash", "interest_coverage", "total_equity",
    ],
    "Valuation & Per-Share": [
        "eps", "eps_diluted", "book_value_per_share", "pe_ratio",
        "market_cap",
    ],
    "Growth & Cash Flow": [
        "revenue_growth_yoy", "net_income_growth_yoy",
        "operating_cash_flow", "free_cash_flow", "capex",
    ],
    "Market Data": [
        "annual_return", "annual_volatility", "avg_daily_volume",
        "close_price_latest", "52w_high", "52w_low",
    ],
}


def _categorize_facts(facts: list[FundamentalRow]) -> dict[str, list[FundamentalRow]]:
    """
    Group FundamentalRow facts by metric category.

    Facts whose metric doesn't match any category go into "Other".
    """
    # Build a lookup: metric_name → category_name
    metric_to_cat: dict[str, str] = {}
    for cat, metrics in _METRIC_CATEGORIES.items():
        for m in metrics:
            metric_to_cat[m] = cat

    categorized: dict[str, list[FundamentalRow]] = {}
    for f in facts:
        cat = metric_to_cat.get(f.metric, "Other")
        categorized.setdefault(cat, []).append(f)

    return categorized


def _format_categorized_facts(
    categorized: dict[str, list[FundamentalRow]],
) -> str:
    """
    Format categorized facts into a structured prompt section.

    This produces a more organized input for the LLM than a flat table,
    leading to better-structured narratives.
    """
    sections = []

    for cat_name in list(_METRIC_CATEGORIES.keys()) + ["Other"]:
        facts = categorized.get(cat_name, [])
        if not facts:
            continue

        lines = [f"\n### {cat_name}\n"]
        lines.append("| Fact ID | FY | Metric | Value | Unit |")
        lines.append("|---------|-----|--------|-------|------|")

        for f in sorted(facts, key=lambda x: (x.fiscal_year, x.metric)):
            val_str = f"{f.value:.2f}" if f.value is not None else "N/A"
            lines.append(
                f"| {f.fact_id[:8]}… | {f.fiscal_year} "
                f"| {f.metric} | {val_str} | {f.unit or '—'} |"
            )

        sections.append("\n".join(lines))

    return "\n".join(sections)


def _extract_fact_ids(text: str) -> list[str]:
    """
    Extract fact_id references from a narrative.

    Fact IDs appear as 8-character hex prefixes followed by "…" in
    the narrative (matching the format we give the LLM). We also
    accept the full 32-char hex ID if the LLM copies the whole thing.

    Returns a list of fact_id prefixes found in the text.
    """
    import re
    # Match 8-hex-char followed by optional "…" or "..."
    pattern = r'\b([0-9a-f]{8})(?:…|\.\.\.)'
    prefixes = re.findall(pattern, text, re.IGNORECASE)

    # Also match full 32-char hex IDs
    full_pattern = r'\b([0-9a-f]{32})\b'
    full_ids = re.findall(full_pattern, text, re.IGNORECASE)

    return prefixes + full_ids


def _validate_citations(
    narrative: str,
    known_facts: list[FundamentalRow],
    job_id: str,
    store: FactStore,
) -> tuple[list[str], list[str]]:
    """
    Validate that all fact_id citations in a narrative resolve to
    actual facts in the Fact Store with non-empty source fields.

    Returns
    -------
    (valid_ids, invalid_ids) : tuple[list[str], list[str]]
        valid_ids: fact_ids that were found and have source
        invalid_ids: fact_ids that were cited but not found
    """
    cited_prefixes = _extract_fact_ids(narrative)
    if not cited_prefixes:
        return [], []

    # Build a lookup from fact_id prefix → full fact_id
    prefix_to_id: dict[str, str] = {}
    for f in known_facts:
        prefix_to_id[f.fact_id[:8]] = f.fact_id
        prefix_to_id[f.fact_id] = f.fact_id  # Full ID too

    valid_ids: list[str] = []
    invalid_ids: list[str] = []

    for prefix in cited_prefixes:
        full_id = prefix_to_id.get(prefix.lower())
        if full_id is None:
            # Try the store directly
            fact = store.get_fact(prefix)
            if fact is not None and fact.source:
                valid_ids.append(prefix)
            else:
                invalid_ids.append(prefix)
        else:
            valid_ids.append(full_id)

    return list(set(valid_ids)), list(set(invalid_ids))


# ── Narration ───────────────────────────────────────────────────────────────────

# The narration prompt template. This is carefully structured to:
# 1. Forbid number invention (AGENTS.md rule 1)
# 2. Require fact_id citations (AGENTS.md rule 10)
# 3. Guide Harvey-style synthesis (trends, not just display)
_NARRATION_PROMPT = """You are a senior equity research analyst specializing in Indian markets (NSE).

Given the structured fundamental data below for {symbol}, write a concise narrative analysis.

ABSOLUTE RULES:
1. NEVER invent, estimate, or compute any number. Only narrate values present in the data below.
2. Every numeric value you mention MUST cite its Fact ID in brackets, e.g. [abc12345…].
3. If a metric is missing, say "not available" — do not guess.
4. Focus on TRENDS: year-over-year changes, inflection points, and what the numbers mean.
5. Keep the narrative to 3-5 paragraphs maximum.

STRUCTURE your analysis as:
- **Financial Health**: Profitability trends, margin trajectory, earnings quality
- **Balance Sheet**: Leverage trends, liquidity position, capital efficiency
- **Growth & Cash Flow**: Revenue/earnings growth rates, cash generation ability
- **Key Risks**: Any deteriorating metrics, overleveraging, margin compression

{facts_section}

Write your analysis now. Remember: cite fact_ids for every number."""


def narrate_fundamentals(
    symbol: str,
    job_id: str,
    store: FactStore,
    facts: Optional[list[FundamentalRow]] = None,
) -> Optional[NarrativeFact]:
    """
    Produce a Harvey-style narrative for a symbol's fundamentals.

    This is the narration step that was previously unwired. It:
      1. Retrieves FundamentalRow facts from the Fact Store by job_id + symbol.
      2. Groups facts by metric category (profitability, leverage, etc.).
      3. Makes 1 cheap LLM call to narrate trends with fact_id citations.
      4. Validates all cited fact_ids exist in the store.
      5. Writes a NarrativeFact to the store with referenced_fact_ids.
      6. If any citation is invalid, writes a RiskFlag with severity=WARNING.

    Constraints enforced:
      - 1 cheap LLM call per symbol (AGENTS.md rule 3)
      - Zero new external API calls
      - All claims traced to fact_ids (AGENTS.md rule 10)
      - No number computed by the LLM (AGENTS.md rule 1)

    Parameters
    ----------
    symbol : str
        NSE symbol.
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    facts : list[FundamentalRow], optional
        Pre-loaded facts. If None, retrieved from the store.

    Returns
    -------
    NarrativeFact or None
        The narrative fact, or None if no facts are available.
    """
    from app.factstore.schemas import RiskFlag, Severity

    # Step 1: Retrieve facts if not provided
    if facts is None:
        all_facts = store.get_facts_for_symbol(job_id, symbol)
        facts = [
            f for f in all_facts
            if isinstance(f, FundamentalRow)
        ]

    if not facts:
        logger.warning(
            "No FundamentalRow facts found for %s in job %s. "
            "Cannot narrate without data.",
            symbol, job_id,
        )
        return None

    logger.info(
        "Narrating %d fundamental facts for %s",
        len(facts), symbol,
    )

    # Step 2: Categorize and format
    categorized = _categorize_facts(facts)
    facts_section = _format_categorized_facts(categorized)

    # Step 3: Build prompt and call cheap LLM (1 call per symbol)
    prompt = _NARRATION_PROMPT.format(
        symbol=symbol,
        facts_section=facts_section,
    )

    try:
        llm = get_cheap_llm()
        # Direct LLM invocation — not through CrewAI Agent overhead.
        # We only need a single structured narration, not a multi-step
        # agent loop. This keeps it at exactly 1 cheap LLM call.
        response = llm.call(messages=[{"role": "user", "content": prompt}])

        if hasattr(response, "content"):
            narrative_text = response.content
        elif hasattr(response, "text"):
            narrative_text = response.text
        elif isinstance(response, str):
            narrative_text = response
        else:
            narrative_text = str(response)

    except Exception as e:
        logger.error(
            "LLM narration failed for %s: %s. "
            "Falling back to structured summary.",
            symbol, e,
        )
        # Fallback: use the formatted facts directly (no LLM call)
        narrative_text = (
            f"## {symbol} — Fundamental Summary\n\n"
            f"Data available for {len(facts)} metrics across "
            f"FY{min(f.fiscal_year for f in facts)}-"
            f"FY{max(f.fiscal_year for f in facts)}.\n\n"
            + format_facts_for_narration(facts)
        )

    # Step 4: Validate citations
    valid_ids, invalid_ids = _validate_citations(
        narrative_text, facts, job_id, store,
    )

    if invalid_ids:
        logger.warning(
            "Narrative for %s contains %d invalid fact_id citations: %s",
            symbol, len(invalid_ids), invalid_ids[:5],
        )
        # Write a RiskFlag for invalid citations (AGENTS.md rule 10)
        risk_flag = RiskFlag(
            job_id=job_id,
            source=f"fundamentals_agent_narration_{symbol}",
            symbol_or_pair=symbol,
            flag_type="uncitable_narrative_claim",
            detail=(
                f"Narrative for {symbol} contains {len(invalid_ids)} "
                f"fact_id references that could not be resolved to "
                f"actual Fact Store entries: {invalid_ids[:5]}. "
                f"These claims may be fabricated."
            ),
            severity=Severity.WARNING,
        )
        store.put_fact(risk_flag)

    logger.info(
        "Narrative for %s: %d valid citations, %d invalid",
        symbol, len(valid_ids), len(invalid_ids),
    )

    # Step 5: Write NarrativeFact
    narrative_fact = NarrativeFact(
        job_id=job_id,
        source=f"fundamentals_agent_narration_{symbol}",
        agent_name="fundamentals_agent",
        section="fundamentals_analysis",
        content=narrative_text,
        referenced_fact_ids=valid_ids,
    )

    store.put_fact(narrative_fact)
    logger.info(
        "Wrote NarrativeFact for %s (fact_id=%s, %d citations)",
        symbol, narrative_fact.fact_id[:8], len(valid_ids),
    )

    return narrative_fact

