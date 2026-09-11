# Agent Crew Orchestration

**Status**: Implemented
**Path**: `app/agents/orchestrator.py`

The Agent Crew is the reasoning and reporting layer of ARTHA. Unlike typical CrewAI setups where agents pass huge context windows to one another, ARTHA strictly uses asynchronous, Fact Store-mediated communication (AGENTS.md Rule 2).

## Tiered Model Routing (AGENTS.md Rule 3)
ARTHA uses tiered model routing to protect API budgets:
- **Cheap Models (8B Class)**: Orchestrator, Fundamentals, Quant Narrator, Sentiment.
- **Mid-Tier Models**: Risk Agent (requires nuanced rule interpretation).
- **Deep Models (e.g., Llama 3 70B / GPT-4)**: Writer Agent (only called once per job for the final synthesis).

## Orchestrator Agent
- **Role**: Financial research operations coordinator.
- **Goal**: Decomposes the user query into a structured task plan. Identifies symbols, fiscal years, and which fact types to collect.
- **Rules**: Never analyzes data itself — purely plans downstream execution.

## The Crew
The crew consists of specialized agents:
- [[Fundamentals_Agent]]: Translates row-level data into Harvey-style text.
- [[Quant_Narrator_Agent]]: Translates math signals into actionable English.
- [[Risk_Agent]]: Enforces hard stops on the portfolio via severity flags.
- [[Writer_Agent]]: Synthesizes the final intelligence report.
