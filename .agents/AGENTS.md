# AGENTS.md — Operating Rules for Any Agent Working on This Repository

This file is read automatically by Antigravity (and any AGENTS.md-aware tool) for every task in this repo.
It is not a one-time prompt — it is the standing contract for how work gets done here. If a task prompt
ever conflicts with this file, this file wins; stop and flag the conflict instead of silently picking one.

## What this repository is

This repo currently contains a general-purpose CrewAI research assistant (FastAPI + CrewAI + Serper/DuckDuckGo
search + Playwright scraping). It is being transformed into **ARTHA**: a niche, production-grade Indian
markets (NSE) and commodities (MCX) intelligence system — row-level structured fundamentals analysis in the
style of Screener.in, narrated the way Harvey AI narrates over structured legal data, backed by a
deterministic quantitative signal engine in the spirit of the publicly-known Renaissance Technologies /
Medallion approach (statistical arbitrage, hidden Markov regime detection), with a multi-agent LLM crew on
top for reasoning and reporting, in the spirit of the published TradingAgents framework.

**Before writing any code, read `docs/ARTHA_ARCHITECTURE.md` in full.** It is the system specification —
the diagnosis of why the pre-existing CrewAI pipeline was failing (every one of its 20 saved job logs was a
Groq rate-limit failure), the full target architecture, the phased roadmap, and the papers each design
decision is grounded in. If that file is missing from the repo, **stop and ask the user for it** rather than
re-deriving the architecture from scratch or guessing at intent.

## Standard of work

Work at the level of a competitive programmer who has placed at ICPC, AtCoder, Meta Hacker Cup, and
ETHGlobal-caliber events, and who has since spent years building systems that hold up under adversarial
review — not just systems that pass a demo. Concretely, that means:

- The correct algorithm for the complexity/scale involved, not the first one that compiles. If two approaches
  are viable, name the trade-off in a comment rather than picking silently.
- Every numeric or statistical claim in code or in a generated report is either computed and tested by code
  in this repo, or explicitly marked `# TODO: not yet computed` — never asserted from training-data memory.
- Edge cases handled explicitly and with a test: empty data, single-observation series, NSE/MCX holidays,
  delisted symbols, missing bhavcopy days, NaNs/nulls, circuit-limit halts, corporate actions (splits/bonuses)
  that break a naive price series.
- No module is "done" without a unit test. This is a finance codebase — a silent correctness bug here doesn't
  just crash a demo, it can misrepresent real financial risk to a real person.

## Non-negotiable architectural rules

1. **LLMs never compute or assert a number.** All numeric computation — regime states, z-scores, cointegration
   tests, backtests, factor values, sentiment scores — happens in deterministic code in `app/quant/`
   (pandas/numpy/statsmodels/hmmlearn) or as a local FinBERT pass in `app/data/news.py`. LLM agents in
   `app/agents/` may only narrate facts that already exist in the Fact Store. If you catch an agent prompt
   asking an LLM to "estimate" or "calculate" a number, that's a bug — fix the architecture, not the prompt.

2. **No sequential prompt-stacking.** Agents read and write typed Pydantic objects in the Fact Store
   (`app/factstore/schemas.py`). Never pass a growing raw-text transcript from one agent to the next — that
   was the root cause of the original system's token-budget collapse. Each agent's prompt should contain only
   the specific Fact Store entries relevant to its current task.

3. **Tiered model routing, not one model for everything.** Cheap/high-quota models (e.g. an 8B-class model)
   handle retrieval, extraction, classification, and row-level narration. Only the final Portfolio Writer
   synthesis — and an optional Bull/Bear debate — uses the larger/deeper model, and only once per job.

4. **No fake multi-key rate-limit workarounds.** Do not rotate multiple API keys from the same provider
   account as a way to multiply one quota — most providers (confirm current behavior in provider docs, e.g.
   Groq) enforce limits at the organization level, not per key. Implement honest multi-provider failover via
   LiteLLM instead (`app/llm/router.py`): primary → secondary provider → local fallback, on real 429s, with
   jittered backoff.

5. **Per-stage checkpointing.** Every agent stage's output is persisted to the Fact Store immediately, not
   just on total job success. A failure at stage N must allow resuming from stage N on retry, never a full
   restart that discards completed work.

6. **Nothing reaches a report unvalidated.** A `QuantSignal` is only eligible for narration when
   `validated=True`, and that flag is only set after the signal clears the walk-forward out-of-sample gate in
   `app/quant/backtest.py`. An LLM finding a signal "plausible" is never sufficient.

7. **Outlier handling is two different things — do not conflate them.** (a) Bad data (fat-finger prints, feed
   glitches) gets caught and dropped at ingestion using MAD-based robust detection. (b) Genuine large market
   moves (circuit-limit hits, real news shocks) are never scrubbed — they are signal, not noise, and the
   Regime Detector needs to see them. When in doubt about which one a data point is, flag it for review rather
   than silently dropping it.

8. **Return-target discipline.** Optimize for Sharpe/Sortino ratio with a bounded max-drawdown constraint. Do
   **not** hardcode or backtest-optimize toward a fixed target return percentage — that is the direct path to
   a strategy that looks great in-sample and fails live. If a task asks you to "increase returns," the correct
   response is (a) add more genuinely uncorrelated validated signals to the portfolio, and (b) apply
   volatility-targeted position sizing on top of whatever Sharpe ratio the resulting portfolio earns — never
   to loosen the validation gate in rule 6, and never to quietly narrow the backtest window until a number
   looks good.

9. **No autonomous order placement, ever, without an explicit human-confirmation step in the loop.** Signal
   generation and historical-replay/paper simulation are always safe to automate. Live order submission is
   not, and must respect the current SEBI retail algorithmic-trading framework (Algo-ID tagging, registered
   static IP, the personal-use sub-10-orders/second threshold) — see `docs/ARTHA_ARCHITECTURE.md` Part 6.6,
   and verify current specifics directly with the relevant broker's API documentation before implementing any
   order-placement code path, since compliance details change.

10. **Full traceability.** Every number in a generated report must resolve to a specific Fact Store entry with
    a non-empty `source` field. Build `tests/test_factstore_citations.py` early and treat it as a hard gate on
    any change to report generation — a report that fails this test does not ship.

11. **This is not investment advice, and the system should never present itself as such.** Generated reports
    describe validated signals and their historical backtest performance; they do not tell a user what to buy
    or sell. Keep that framing in any user-facing copy.

## Repo structure to converge toward

```
artha/
  app/
    data/            # nse.py, mcx.py, fundamentals.py, news.py (FinBERT, local)
    quant/            # regime.py, statarb.py, factors.py, alpha_loop.py, backtest.py, rl_allocator.py
    factstore/        # schemas.py, store.py
    agents/           # orchestrator.py, fundamentals_agent.py, quant_narrator_agent.py,
                      # sentiment_agent.py, risk_agent.py, writer_agent.py
    llm/              # router.py (tiered LiteLLM config, caching, failover)
    api/              # routes.py
    main.py
  tests/
    test_statarb.py
    test_regime.py
    test_rl_allocator.py
    test_factstore_citations.py   # traceability gate — see rule 10
  docs/
    ARTHA_ARCHITECTURE.md          # the system spec — read this first
  data_cache/
  frontend/
```

## Definition of done, per module

- **Quant modules**: unit tests passing + at least one walk-forward backtest report checked in, reporting
  Sharpe/Sortino/max-drawdown honestly — including mediocre or negative results. Do not cherry-pick the
  window that looks best.
- **Agents**: a test that runs the agent against a fixture Fact Store and confirms every field it emits
  resolves to something in that fixture — no invented numbers, ever.
- **Any new external data source**: a caching layer so it's fetched once and reused, not re-fetched per query.
- **Any change touching report generation**: `tests/test_factstore_citations.py` still passes.

## When to stop and ask instead of guessing

Ask the user before proceeding if: the required reading (`docs/ARTHA_ARCHITECTURE.md`) is missing; a task
seems to ask for loosening the validation gate (rule 6) or the return-target discipline (rule 8); a task
implies live order placement (rule 9); or current broker/regulatory specifics can't be confirmed and the
implementation would depend on getting them right.
