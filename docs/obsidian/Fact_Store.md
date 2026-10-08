# Fact Store

**Status**: Implemented
**Path**: `app/factstore/store.py`

The Fact Store is the central persistence layer and the *only* medium through which agents communicate sequentially. It strictly enforces the architectural rule banning sequential prompt stacking.

## Dual-Backend Architecture
- **PostgreSQL**: Used for production deployments (e.g., Supabase) with connection pooling (`psycopg2`).
- **SQLite**: Used for local development and testing (`data_cache/factstore.db`) with per-thread connections.
- Automatically selects the backend based on the `DATABASE_URL` environment variable.

## Key Design Principles
1. **Per-Stage Checkpointing (AGENTS.md Rule 5)**:
   - Data is persisted immediately after each stage (or even intra-stage), never buffered until the end of the job. 
   - A failure at stage N never discards work completed in stages 1..N-1.
2. **Asynchronous Orchestration**: 
   - Agents read typed Pydantic objects from the Fact Store. They do not pass raw text transcripts to one another.
3. **Traceability (AGENTS.md Rule 10)**:
   - Every fact injected into the Fact Store has a mandatory `source` field (the provenance anchor).
   - This anchor allows the final report citations to mathematically trace every single number back to its origin.

## Core Operations
- `put_fact` / `put_facts`: Writes one or multiple facts to the DB (using `INSERT OR REPLACE` for SQLite and `ON CONFLICT DO UPDATE` for Postgres).
- `get_fact` / `get_facts_by_type` / `get_job_facts`: Retrieves facts natively deserialized back into their specific Pydantic `BaseFact` subclass.
- `get_validated_signals`: Retrieves only `QuantSignal`s or `OptionsSignal`s that have `validated=True` (checked by the Backtester).

## Linkages
- Stores [[Fact_Schemas]] schemas.
- Written to by [[Data_Layer]] ingestion scripts, [[Alpha_Loop]], and every Agent in the Crew.
- Read from by the LLM Agents (Narrator, Risk, Writer) to compose the final intelligence report.
