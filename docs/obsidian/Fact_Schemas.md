# Fact Schemas

**Status**: Implemented
**Path**: `app/factstore/schemas.py`

ARTHA utilizes strongly-typed Pydantic models for all facts passing through the [[Fact_Store]]. This enforces rigorous data contracts between the deterministic Quant Engine and the stochastic LLM Crew.

## Base Architecture
Every fact inherits from `BaseFact`:
- `fact_id`: Auto-generated UUID.
- `job_id`: The analysis job ID.
- `fact_type`: Discriminator enum (`FactType`).
- `source`: **MANDATORY** provenance string (e.g. `'NSE_BHAVCOPY_2025-03-31'`). This is the traceability anchor preventing LLM hallucination (AGENTS.md rule 10).
- `created_at`: Timestamp.

## Concrete Fact Types

### 1. `FundamentalRow`
- Represents a single row-level metric (e.g., revenue, eps, roe) for a company and fiscal year.
- Written by the Fundamentals Agent.

### 2. `QuantSignal`
- Produced by the deterministic Quant Engine (e.g., StatArb Z-score, Factor composites).
- **CRITICAL RULE**: `validated` defaults to `False`. It can ONLY be set to `True` by the [[Quant_Backtester]]. An LLM finding a signal "plausible" is never sufficient to flip this flag (AGENTS.md rule 6).

### 3. `OptionsSignal`
- Analogous to `QuantSignal` but tracks complex options strategy attributes: legs, net premium, max profit, max loss, margin required, portfolio greeks, and the regime context.
- `validated` follows the same strict rule via the Options Backtester.

### 4. `NewsSignal`
- Represents local sentiment scoring.
- Incorporates `sentiment_score` computed locally by FinBERT. No LLM tokens are spent to generate this score, only to narrate its impact.

### 5. `RiskFlag`
- Emitted by the Risk Agent.
- Can possess a `Severity` of `INFO`, `WARNING`, or `BLOCK`.
- A `BLOCK` flag forces the signal to be omitted from the final report (hard veto).

### 6. `NarrativeFact`
- Narrative text produced by an LLM agent.
- Must contain `referenced_fact_ids` pointing to the exact facts narrated, satisfying the traceability gate.

### 7. `ReportCitation`
- The final verification link pairing a claim text in the final report back to the `fact_id` in the Fact Store.
