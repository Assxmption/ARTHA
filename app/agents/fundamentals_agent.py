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
