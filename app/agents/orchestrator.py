"""
Orchestrator Agent
==================
Decomposes a user query into which data slices, fact types, and
risk checks are needed.  Writes a task plan to the Fact Store.

Uses cheap model (AGENTS.md rule 3 — retrieval/classification
doesn't need the expensive model).

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Orchestrator/Planner row)
"""

from __future__ import annotations

from crewai import Agent

from app.llm.router import get_cheap_llm


def create_orchestrator_agent() -> Agent:
    """
    Create the ARTHA Orchestrator agent.

    The Orchestrator analyses the user's query and produces a structured
    task plan specifying:
      - Which symbols / sectors to analyse
      - Which fiscal years to cover
      - Which fact types to collect (fundamentals, quant signals, news)
      - What risk checks to run

    It does NOT do any analysis itself — it only plans.
    """
    llm = get_cheap_llm()

    return Agent(
        role="ARTHA Orchestrator",
        goal=(
            "Analyse the user's Indian markets / commodities query and produce "
            "a structured task plan. Identify: (1) which NSE symbols or MCX "
            "commodities to analyse, (2) which fiscal years to cover, "
            "(3) which fact types to collect (fundamentals, quant signals, "
            "sentiment), and (4) what risk checks are needed. "
            "Output a clear, machine-parseable plan — not a narrative essay."
        ),
        backstory=(
            "You are a financial research operations coordinator specialising "
            "in Indian equities (NSE) and commodities (MCX). You understand "
            "the ARTHA system's capabilities: row-level fundamentals analysis, "
            "quantitative signal generation (regime detection, statistical "
            "arbitrage), and FinBERT-based news sentiment. Your job is to "
            "decompose any query into the specific data retrieval and analysis "
            "steps the downstream agents need to execute. You never analyse "
            "data yourself — you plan what others should do."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
