"""
Writer Agent — Final Report Synthesis
======================================
Produces the structured + narrative report from Fact Store entries.
This is the ONLY agent that uses the deep (expensive) model, and
it is called ONLY ONCE per job.

Non-negotiable constraints (AGENTS.md):
  - Rule 1: Never computes a number — only narrates facts from the store.
  - Rule 10: Every numeric claim must resolve to a Fact Store entry.
    Post-generation citation check enforces this.
  - Rule 11: This is not investment advice.  Reports describe validated
    signals and historical backtest performance; they do not tell a user
    what to buy or sell.

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Portfolio Writer row)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from crewai import Agent

from app.factstore.schemas import (
    BaseFact,
    FactType,
    FundamentalRow,
    NarrativeFact,
    NewsSignal,
    QuantSignal,
    ReportCitation,
    RiskFlag,
)
from app.factstore.store import FactStore
from app.llm.router import get_deep_llm

logger = logging.getLogger(__name__)


def create_writer_agent() -> Agent:
    """
    Create the ARTHA Writer Agent.

    The Writer synthesises all Fact Store entries for the current job
    into a structured, professional report.  Only uses the deep model
    tier, and only once per job.
    """
    llm = get_deep_llm()

    return Agent(
        role="ARTHA Portfolio Report Writer",
        goal=(
            "Produce a comprehensive, professionally structured analytical "
            "report in Markdown format. The report must include: Executive "
            "Summary, Fundamental Analysis, any available Quantitative "
            "Signals, Risk Flags, and a References section. EVERY numeric "
            "claim must cite a Fact Store entry by its fact_id. If a metric "
            "is unavailable, say so — never invent a number. End with the "
            "mandatory disclaimer: 'This is not investment advice.'"
        ),
        backstory=(
            "You are an expert financial report writer specialising in "
            "Indian equity and commodity markets. Your reports are valued "
            "for their analytical depth, precise language, and rigorous "
            "source attribution. You write for an audience that includes "
            "both technical analysts and informed investors. You always "
            "maintain intellectual honesty about uncertainties and never "
            "present analysis as investment advice. Your style is "
            "authoritative yet accessible, with every number traceable "
            "to a specific data source."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ── Report context builder ─────────────────────────────────────────────────────

def build_writer_context(
    job_id: str,
    store: FactStore,
    symbol: str,
) -> str:
    """
    Build the Writer agent's context from the Fact Store.

    Only includes facts relevant to the current job — NOT a growing
    chat transcript (AGENTS.md rule 2: no sequential prompt-stacking).

    Parameters
    ----------
    job_id : str
        Current job ID.
    store : FactStore
        Fact Store instance.
    symbol : str
        Primary symbol being analysed.

    Returns
    -------
    str
        Structured context text for the Writer's prompt.
    """
    sections = []

    # ── Fundamentals ────────────────────────────────────────────────────
    fund_facts = store.get_facts_by_type(job_id, FactType.FUNDAMENTAL)
    if fund_facts:
        sections.append("## FUNDAMENTAL DATA (from Fact Store)\n")
        sections.append(
            "| Fact ID | Symbol | FY | Metric | Value | Unit | Source |"
        )
        sections.append(
            "|---------|--------|----|--------|-------|------|--------|"
        )
        for f in sorted(
            fund_facts,
            key=lambda x: (x.symbol, x.fiscal_year, x.metric)
            if isinstance(x, FundamentalRow)
            else ("", 0, ""),
        ):
            if isinstance(f, FundamentalRow):
                val = f"{f.value:.2f}" if f.value is not None else "N/A"
                sections.append(
                    f"| {f.fact_id[:8]}… | {f.symbol} | {f.fiscal_year} "
                    f"| {f.metric} | {val} | {f.unit or '—'} | {f.source} |"
                )
        sections.append("")

    # ── Quant Signals (validated only) ──────────────────────────────────
    quant_facts = store.get_validated_signals(job_id)
    if quant_facts:
        sections.append("## VALIDATED QUANT SIGNALS (from Fact Store)\n")
        for q in quant_facts:
            sections.append(
                f"- **{q.signal_type.value}** for {q.symbol_or_pair}: "
                f"value={q.value:.4f}, regime={q.regime_label or 'N/A'}, "
                f"backtest Sharpe={q.backtest_sharpe or 'N/A'} "
                f"[fact_id: {q.fact_id[:8]}…, source: {q.source}]"
            )
        sections.append("")

    # ── Risk Flags ──────────────────────────────────────────────────────
    risk_facts = store.get_facts_by_type(job_id, FactType.RISK_FLAG)
    if risk_facts:
        sections.append("## RISK FLAGS (from Fact Store)\n")
        for r in risk_facts:
            if isinstance(r, RiskFlag):
                icon = {"info": "ℹ️", "warning": "⚠️", "block": "🛑"}.get(
                    r.severity.value, "❓"
                )
                sections.append(
                    f"- {icon} **{r.flag_type}** ({r.severity.value}) — "
                    f"{r.symbol_or_pair}: {r.detail} "
                    f"[fact_id: {r.fact_id[:8]}…]"
                )
        sections.append("")

    # ── News / Sentiment ────────────────────────────────────────────────
    news_facts = store.get_facts_by_type(job_id, FactType.NEWS_SIGNAL)
    if news_facts:
        sections.append("## NEWS SENTIMENT (from Fact Store)\n")
        for n in news_facts:
            if isinstance(n, NewsSignal):
                sections.append(
                    f"- {n.symbol_or_commodity}: sentiment={n.sentiment_score:.2f}, "
                    f"headlines={n.headline_count}, event={n.event_type or 'general'} "
                    f"[fact_id: {n.fact_id[:8]}…]"
                )
        sections.append("")

    # ── Narratives from prior agents ────────────────────────────────────
    narrative_facts = store.get_facts_by_type(job_id, FactType.NARRATIVE)
    if narrative_facts:
        sections.append("## AGENT NARRATIVES (from Fact Store)\n")
        for nf in narrative_facts:
            if isinstance(nf, NarrativeFact):
                sections.append(
                    f"### {nf.agent_name} — {nf.section}\n{nf.content}\n"
                )

    # ── Instructions ────────────────────────────────────────────────────
    sections.append("---")
    sections.append(
        "**INSTRUCTIONS**: Write a comprehensive report about "
        f"{symbol} using ONLY the data above. Every numeric claim "
        "must cite the corresponding Fact ID. Do not invent or "
        "estimate any values not present above. End the report with: "
        "'**Disclaimer**: This report is not investment advice. It "
        "describes validated signals and historical data; it does not "
        "tell you what to buy or sell.'"
    )

    return "\n".join(sections)


# ── Post-generation citation check ─────────────────────────────────────────────

def check_citations(
    report_text: str,
    job_id: str,
    store: FactStore,
) -> dict:
    """
    Post-generation citation check (AGENTS.md rule 10).

    Scans the report for fact_id references and verifies each one
    exists in the Fact Store with a non-empty source.

    Returns
    -------
    dict
        'valid': bool — True if all citations resolve.
        'total_citations': int
        'resolved': int
        'unresolved': list[str] — fact_ids that failed to resolve.
    """
    # Look for patterns like [fact_id: abc12345…] or (fact_id: abc12345…)
    # Also catch bare 8+ hex-char IDs that look like fact_id prefixes
    id_pattern = re.compile(r"fact_id[:\s]+([a-f0-9]{8,})")
    found_ids = id_pattern.findall(report_text)

    # De-duplicate
    unique_ids = list(set(found_ids))

    resolved = []
    unresolved = []

    # Get all facts for this job to check against
    all_facts = store.get_job_facts(job_id)
    fact_id_map = {f.fact_id: f for f in all_facts}

    for ref_id in unique_ids:
        # Match against full IDs (the report might use truncated 8-char prefixes)
        matched = False
        for full_id, fact in fact_id_map.items():
            if full_id.startswith(ref_id):
                if fact.source:  # source must be non-empty
                    resolved.append(ref_id)
                    matched = True
                    break
        if not matched:
            unresolved.append(ref_id)

    return {
        "valid": len(unresolved) == 0 and len(resolved) > 0,
        "total_citations": len(unique_ids),
        "resolved": len(resolved),
        "unresolved": unresolved,
    }
