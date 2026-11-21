# ARTHA Implementation Plan

**Status:** Active execution plan  
**Date:** 2026-11-03  
**Canonical specification:** `docs/ARTHA_ARCHITECTURE.md`

## 1. Target state

ARTHA will be a traceable Indian-markets intelligence system with four separate
trust boundaries:

1. **Data boundary** — cached, normalised NSE/MCX observations with explicit
   provenance, corporate-action handling, holiday gaps, and review flags.
2. **Quant boundary** — deterministic signals and walk-forward validation. LLMs
   cannot create, validate, or modify numeric facts.
3. **Narrative boundary** — agents read only typed, job-scoped Fact Store data;
   every numeric statement in a report resolves to a stored fact.
4. **Execution boundary** — historical replay and virtual paper trading only;
   no autonomous broker order path.

The system is considered production-ready only when every boundary has an
automated acceptance test and the end-to-end report path passes the traceability
gate.

## 1A. Execution log

### Completed in this implementation pass

- Added this phased plan as the repository operating document.
- Added structural validation evidence requirements for `QuantSignal` and
  `OptionsSignal`.
- Stopped legacy backtest JSON from automatically creating validated signals.
- Made report citation checking blocking, including uncited numeric-line checks,
  and persisted `ReportCitation` facts.
- Made fallback reports citation-safe.
- Added durable Fact Store job state and idempotent completed-job recovery.
- Added persisted job attempt counts for auditable retries.
- Added API job-state recovery after worker restart and persisted failure state.
- Added a real `/api/health` endpoint.
- Aligned route-level authentication with the development/production middleware
  policy.
- Replaced same-provider Groq key rotation with honest provider failover.
- Declared the Postgres driver and optional FinBERT installation path.
- Fixed the Heston calibration round-trip performance failure with a validated
  deterministic warm start and bounded optimizer settings.
- Enforced BLOCK risk vetoes in both writer context and deterministic fallback
  reports.
- Added provider-injected MCX bhavcopy parsing, contract identity, caching,
  holiday/weekend handling, and history aggregation in `app/data/mcx.py`.
- Added fail-closed MCX contract metadata and read-only `/api/quant/mcx/*`
  endpoints; sizing is unavailable until authoritative contract metadata is
  configured.
- Removed synthetic NSE option-chain responses from the production API path;
  unavailable real data now returns an explicit 503 and cannot feed sizing or
  trading decisions.
- Hardened paper validation with one coherent timestamped market snapshot per
  cycle, stale/partial-data rejection, no implicit equal-weight fallback, and
  backtest metrics loaded from stored artifacts rather than hardcoded values.
- Ran the first real-data 10-year train / later OOS experiment using 83 cached
  NSE EOD files. The corrected boundary now uses 82 symbols, training covered
  2014-08-09–2024-08-08, and OOS covered 2024-08-09–2026-08-21 (504 sessions).
  The frozen momentum configuration produced OOS Sharpe **-0.940** and remains
  ineligible because the cache-only universe is survivorship-biased.
- Ran the predeclared real-data sleeve matrix in
  `docs/backtest_reports/real_sleeve_matrix_10yr_20261104_034423.json`.
  Reversal-3-month was the strongest tested sleeve at OOS Sharpe **0.262**;
  equal-weight and inverse-volatility combinations were **-0.551** and
  **-0.753** respectively. No tested sleeve or portfolio is validated.
- Downloaded a separate real long-history NSE cache from yfinance with 95
  provenance-tracked series, raw OHLC, adjusted close, dividends, splits, and
  a manifest hash. The cache was checked for exchange-local date preservation.
- Ran four disjoint 10-year-train/504-session-OOS folds on 74 older symbols in
  `docs/backtest_reports/real_repeated_sleeves_10yr_4fold_20261104_043738.json`.
  The training-selected aggregate produced Sharpe **0.742**, annualised return
  **21.24%**, and maximum drawdown **38.42%**. The best fixed candidate reached
  Sharpe **0.892** and annualised return **25.28%**, but maximum drawdown was
  **35.14%** and the universe remains survivorship-biased. This is diagnostic,
  not validated performance.

### Verification after this pass

- Python suite after trust-boundary hardening: **604 passed, 2 skipped, 410 warnings**.
- MCX/risk slice: **7 focused tests passed**.
- Current full Python suite: **619 passed, 2 skipped, 410 warnings**.
- Current full Python suite after historical-validation tests: **622 passed,
  2 skipped, 410 warnings**.
- Current full Python suite after causal validation, security-master,
  fundamentals-history, factor-accounting, and paper-execution hardening:
  **646 passed, 2 skipped, 431 warnings**.
- Added a point-in-time security-master seam with canonical symbol aliases and
  listing/delisting membership intervals; the current cache-only runner remains
  explicitly survivorship-biased until an authoritative master is supplied.
- Added publication-date-aware fundamental history and rejected static current
  fundamentals from historical factor backtests.
- Changed default historical HMM usage to causal forward filtering with
  expanding pre-observation fits; hardened the generic walk-forward engine to
  require exact OOS coverage, apply costs, and fail closed on bad folds.
- Corrected factor and PCA accounting to use canonical wealth/drawdown metrics;
  paper trading now requires canonical validation evidence and liquidates
  removed target symbols.
- Added the causal real-data sleeve matrix and corrected pair-stat-arb daily
  accounting to use compounded wealth and explicit entry/exit costs.
- Frontend build: passed.
- Frontend lint: passed with existing unused-variable and hook warnings.
- Heston calibration test: passed in the full suite.

The remaining phases below are deliberately not marked complete until their
acceptance gates are implemented and tested.

## 2. Non-negotiable invariants

- A fact must have a non-empty `source`.
- A signal is narratable only after a real out-of-sample walk-forward result
  passes the configured Sharpe, drawdown, and observation-count gate.
- Stored backtest JSON is evidence, not validation authority, unless it contains
  explicit validation metadata produced by the backtest service.
- Reports with unresolved citations or uncited numeric claims are rejected.
- Risk-blocked facts cannot enter the final report context.
- Every completed stage is checkpointed and can be resumed without rerunning
  successful earlier stages.
- Data ingestion never silently drops an outlier; bad data is flagged and a
  genuine market shock is preserved.
- Paper trading uses synthetic fills. There is no broker order submission code.
- User-facing text describes research and historical validation, never advice.

## 3. Execution sequence

### Phase 0 — Trust-boundary hardening (complete)

**Deliverables**

- One authoritative validation contract for equity and options signals.
- Backtest reports carry validation evidence; the signal injector refuses
  unproven `validated=True` values.
- Report citation checking becomes a blocking gate and persists citation facts.
- Fallback reports include citations for every numeric fact they render.
- Pipeline documentation and active stage names agree.
- A real `/api/health` endpoint reports application and Fact Store status.
- Remove stale imports/tests for the deleted generic CrewAI research pipeline.

**Exit gate**

- Citation, validation, pipeline, and API integration tests pass.
- No production path assigns `validated=True` without validation evidence.

The Phase 0 exit gate is complete.  Risk veto enforcement and the blocking
citation/validation tests are now part of the regression suite.

### Phase 1 — Durable orchestration and observability

**Deliverables**

- Persist job state, stage state, attempt count, timestamps, and errors.
- Resume from the first incomplete stage after process restart.
- Store final narratives and report citations as Fact Store entries.
- Add structured health diagnostics for dependency availability, cache age, and
  active storage backend.
- Add deterministic end-to-end fixture tests for `AnalysisCrew`.

**Exit gate**

- Kill/restart simulation resumes a job without duplicate facts or duplicate
  LLM stages.
- A failed citation gate cannot produce a completed report status.

### Phase 2 — Data correctness and dependency closure (in progress)

**Deliverables**

- Make Postgres support installable and tested with a service-backed test job.
- Make FinBERT either a declared optional extra with a visible fallback status or
  a required dependency; do not leave the dependency state ambiguous.
- Make `pykalman` fallback explicit in diagnostics.
- Add data-quality metadata for adjusted prices, missing sessions, delisted
  symbols, corporate actions, and circuit-limit days.
- Add cache freshness and source provenance to every ingestion result.

MCX ingestion has started with a provider-injected, cached bhavcopy normalizer.
The remaining work is official-source integration testing, contract master/lot
metadata, and MCX-specific quality fixtures.

**Exit gate**

- Clean-environment install and import checks pass.
- Synthetic fixtures cover empty, single-row, NaN, holiday-gap, and corporate-
  action cases.

### Phase 3 — MCX commodities

**Deliverables**

- Implement `app/data/mcx.py` behind a provider interface.
- Add cached official EOD bhavcopy/history ingestion for gold, silver, crude,
  natural gas, copper, and supported agricultural contracts.
- Normalise contract code, expiry, lot size, tick size, settlement price,
  rollover, and trading-session calendars.
- Add gold/silver, crude/natural-gas, and calendar-spread data fixtures.
- Add commodity-specific Fact Store facts and supply-shock event provenance.

**Exit gate**

- No strategy can consume an MCX series without a contract/expiry identity.
- At least one honest walk-forward report is checked in for each initial spread
  family, including mediocre or negative results.

### Phase 4 — Quant research discipline

**Deliverables**

- Separate signal discovery, strategy construction, and validation APIs.
- Add purging/embargo where labels overlap.
- Add parameter/version/data-hash metadata to every validation result.
- Make portfolio allocation optimise risk-adjusted performance with drawdown
  constraints, never a fixed return target.
- Add report-level validation summaries that distinguish IS and OOS results.

**Exit gate**

- A signal with strong in-sample performance and failed OOS performance is
  demonstrably excluded from narration.
- Backtest artifacts are reproducible from checked-in scripts and data hashes.

### Phase 5 — Paper-trading validation

**Deliverables**

- Keep all fills virtual and broker-independent.
- Compare forward paper results with the exact backtest configuration.
- Add slippage, liquidity, stale-price, halt, and data-delay monitoring.
- Add explicit human confirmation and broker/regulatory review before any future
  order-placement work.

**Exit gate**

- No broker SDK or order endpoint is present in the repository until a separate
  approved implementation scope exists.

## 4. Current priority order

1. Close point-in-time universe, fundamentals, adjustment, and session-quality
   data blockers.
2. Make every historical path use the causal regime and canonical backtest
   contracts.
3. Complete repeated walk-forward validation with an untouched final holdout.
4. Measure each deterministic sleeve independently before portfolio combining.
5. Build train-only, drawdown-constrained, volatility-targeted allocation across
   accepted sleeves.
6. Re-run exact historical replay and only then continue real-data paper
   observation.
7. Keep MCX/options and broker work behind their data and validation gates.

The quantitative optimisation backlog and promotion rules are maintained in
`docs/ARTHA_OPTIMIZATION_ROADMAP.md`. The current OOS failure is retained as
research evidence; it must not be hidden by selecting a more favourable window
or by weakening the validation contract.

## 5. Change-control rules

- Each phase must leave the repository runnable.
- Every new module gets meaningful edge-case tests.
- Existing tests are never weakened to accommodate an implementation shortcut.
- Numeric claims in documentation are tied to a checked-in artifact or marked as
  not yet computed.
- Changes remain small enough to review and revert by phase.
