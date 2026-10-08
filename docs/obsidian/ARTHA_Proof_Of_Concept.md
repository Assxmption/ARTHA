# ARTHA: Production-Grade Market Intelligence Proof of Concept

**Status**: Architecture & PoC Complete
**Target**: Indian Equities (NSE) & Commodities (MCX)

ARTHA is a highly structured, production-grade intelligence system. It explicitly rejects the naive "Prompt an LLM to trade" approach. Instead, it pairs a **deterministic, quantitative backend** (inspired by statistical arbitrage and HMM regime detection) with a **multi-agent LLM crew** constrained strictly to a reasoning and reporting role.

## The Core Problem Being Solved
Standard LLM trading agents fail due to:
1. **Context Window Collapse**: Sequential prompt stacking blows up the context window and hits provider rate limits.
2. **Numeric Hallucination**: LLMs are terrible at math and cannot be trusted to calculate indicators, backtest PnL, or price options.
3. **Absence of Risk Controls**: LLMs do not respect portfolio volatility limits.

## The ARTHA Architecture (The Solution)

### 1. Data Layer
The foundation. Handles raw data ingestion, basic cleaning, and FinBERT-based local sentiment scoring (saving LLM API costs).
- [[Data_Layer]]
- [[Data_Ingestion_NSE]]
- [[Data_Ingestion_MCX]]
- [[Data_Fundamentals]]
- [[Data_News]]

### 2. The Quant Engine (Deterministic Math)
All numbers, scores, and probabilities are computed here. LLMs NEVER touch this code.
- [[Regime_Detector]]: Hidden Markov Model to classify BULL/BEAR/SIDEWAYS markets.
- [[Stat_Arb_Screener]]: Cointegration tests for pair trading.
- [[Alpha_Factors]]: Z-score composites of momentum and mean-reversion.
- [[Signal_Factory]]: Signal construction logic.
- [[Quant_Simulator]]: Evaluates Alpha in shadow/historical modes.
- [[Portfolio_Optimizer]]: Allocates target weights using Ledoit-Wolf shrinkage.
- [[Risk_Manager]]: Enforces a 6-layer defense system (position limits, drawdown breakers).
- [[Alpha_Loop]]: The orchestrator connecting signal generation to the Fact Store.

### 3. The Options Engine (Derivatives)
Pricing and strategy selection.
- [[Options_Engine]]: Overview.
- [[Options_Pricing]]: BSM, Heston, and Merton Jump-Diffusion models.
- [[Options_Strategies]]: Iron Condors, Straddles, etc.

### 4. The Validation Gate
No signal is considered valid unless it passes rigorous out-of-sample (OOS) testing.
- [[Quant_Backtester]]: Anchored walk-forward engine.
- [[Options_Backtester]]: Walk-forward options backtesting.

### 5. The Fact Store (Asynchronous Persistence)
The state management layer. Agents communicate via typed DB records, NOT string concatenation. Solves context collapse.
- [[Fact_Store]]: PostgreSQL / SQLite implementation.
- [[Fact_Schemas]]: Pydantic contracts ensuring `source` traceability.

### 6. The LLM Crew (Reasoning & Narration)
The crew translates the raw math into actionable, Harvey-style reports. Tiered model routing saves costs.
- [[Agent_Crew]]: The Orchestrator and overview.
- [[Fundamentals_Agent]]: Screener++ style row-level narrative (Cheap Model).
- [[Quant_Narrator_Agent]]: Plain-language explanation of Z-scores and Regimes (Cheap Model).
- [[Risk_Agent]]: Hard veto power over SEBI compliance and drawdowns (Mid Model).
- [[Writer_Agent]]: Synthesizes the final output, runs post-gen citation checks against the Fact Store (Deep Model, run once).

## Conclusion
ARTHA is not a chatbot. It is a verifiable quantitative pipeline where an LLM is given exactly one job: to narrate what the math already proved.
