# Fundamentals Agent

**Status**: Implemented
**Path**: `app/agents/fundamentals_agent.py`

The Fundamentals Agent acts as the bridge between the structured Screener-style row data (from `yfinance`) and the human-readable report.

## Core Responsibilities
1. **Data Gathering (Deterministic)**: Retrieves financial statement data (Income, Balance Sheet, Cash Flow) and market data via the Data Layer.
2. **Persistence (Checkpointing)**: Writes every retrieved metric into the [[Fact_Store]] as a `FundamentalRow` immediately.
3. **Narration**: Calls the cheap LLM to produce a narrative explaining year-over-year trends and the meaning of the numbers (e.g., what a dropping operating margin implies).

## AGENTS.md Compliance
- **Rule 1 (No Math)**: The agent NEVER calculates metrics. It only narrates pre-calculated values stored in the Fact Store. If a metric is missing, it explicitly states so.
- **Rule 10 (Traceability)**: Every numeric claim in the LLM's narrative must cite the exact `fact_id` of the `FundamentalRow`.
