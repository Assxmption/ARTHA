# ARTHA Architecture Index

Welcome to the ARTHA Knowledge Graph. ARTHA is an Indian Markets (NSE) and Commodities (MCX) Intelligence System.

This graph is divided into four main architectural pillars:

1. **[[Data_Layer]]**: Bounded, structured, and cached ingestion pipeline (bhavcopy, jugaad-data, MCX).
2. **[[Quant_Engine]]**: The deterministic math backbone. LLMs *never* compute numbers here.
3. **[[Fact_Store]]**: The typed, shared state memory that eliminates sequential prompt bloat.
4. **[[Agent_Crew]]**: Specialized LLMs tiered by model size for reasoning, planning, and narration.

## Core Philosophy
- **Deterministic Math for Numbers**: Everything numeric is handled by the `app/quant/` module.
- **LLMs for Reasoning/Narrative**: Text generation relies entirely on pre-calculated facts.
- **Traceability**: Every generated claim must cite a Fact ID (Rule 10).

## Repository Structure Map
- **`app/`**
  - **`data/`**
    - `nse.py`, `mcx.py`, `fundamentals.py`, `news.py` ➡️ See [[Data_Layer]]
  - **`quant/`** ➡️ See [[Quant_Engine]]
    - `regime.py` ➡️ See [[Regime_Detector]]
    - `statarb.py` ➡️ See [[Stat_Arb_Screener]]
    - `factors.py` ➡️ See [[Factor_Library]]
    - `alpha_loop.py` ➡️ See [[Alpha_Loop]]
    - `backtest.py` ➡️ See [[Walk_Forward_Backtester]]
    - `options_pricing.py` ➡️ See [[Options_Engine]]
  - **`factstore/`** ➡️ See [[Fact_Store]]
    - `schemas.py` ➡️ See [[Fact_Schemas]]
    - `store.py`
  - **`agents/`** ➡️ See [[Agent_Crew]]
    - `orchestrator.py` ➡️ See [[Orchestrator_Agent]]
    - `fundamentals_agent.py` ➡️ See [[Fundamentals_Agent]]
    - `quant_narrator_agent.py` ➡️ See [[Quant_Narrator_Agent]]
    - `risk_agent.py` ➡️ See [[Risk_Agent]]
    - `writer_agent.py` ➡️ See [[Writer_Agent]]
    - `sentiment_agent.py` 
  - **`llm/`**
    - `router.py` (Tiered LiteLLM config, caching, failover chain)
  - **`api/`**
    - `routes.py`
  - `main.py`
- **`tests/`**
  - `test_statarb.py`, `test_regime.py`
  - `test_factstore_citations.py` (Enforces traceability gate)
- **`docs/`**
  - `ARTHA_ARCHITECTURE.md` (System spec)
- **`data_cache/`**
- **`frontend/`**
