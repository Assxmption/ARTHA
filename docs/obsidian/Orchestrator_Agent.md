# Orchestrator Agent

Located in `app/agents/orchestrator.py`.

Responsible for high-level coordination of the [[Agent_Crew]] and delegating tasks. The Orchestrator ensures that the pipeline executes correctly and that the [[Fact_Store]] is checkpointed at each stage.

- Uses `get_cheap_llm()` to route tasks without burning through high-tier API quotas.
- Implements **Per-stage checkpointing** (AGENTS.md rule 5): If stage N fails, it can resume from stage N on retry because state is safely stored in the Fact Store.
