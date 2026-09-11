# Writer Agent

**Status**: Implemented
**Path**: `app/agents/writer_agent.py`

The Portfolio Writer Agent is responsible for synthesizing all facts, signals, and prior agent narratives into the final, polished Markdown report.

## Resource Allocation (AGENTS.md Rule 3)
This is the **ONLY** agent in the ARTHA system that utilizes the Deep LLM tier (e.g., Llama 3 70B, GPT-4). It is invoked exactly once at the very end of the pipeline to conserve API budget.

## Context Building (No Prompt Stacking)
Instead of feeding the Writer the raw transcript of previous agents (which causes token explosion), `build_writer_context` constructs a tight, structured context purely from the [[Fact_Store]].
- Fundamentals Table
- Validated Quant Signals Table
- Risk Flags
- Previous agent narrative paragraphs.

## Post-Generation Citation Check (AGENTS.md Rule 10)
ARTHA enforces an absolute ban on LLM hallucination for numbers.
The function `check_citations` scans the final markdown report produced by the Writer for `fact_id` references (`[fact_id: XXXXXXXX]`). It cross-references every single ID against the actual Fact Store database. 
If the Writer hallucinates a number and invents a fake citation, or fails to cite a source for a numeric claim, the report fails the validation gate.

## Mandatory Disclaimer
Enforces AGENTS.md Rule 11 by ensuring every report ends with: *"This report is not investment advice. It describes validated signals and historical data; it does not tell you what to buy or sell."*
