# ARTHA Quantitative Optimization Roadmap

**Status:** Active research backlog  
**Canonical architecture:** `docs/ARTHA_ARCHITECTURE.md`  
**Execution plan:** `docs/ARTHA_IMPLEMENTATION_PLAN.md`

## Objective

Maximise validated risk-adjusted performance across NSE and MCX while keeping
the data, signal, risk, and execution boundaries auditable. The project
aspiration is Sharpe above 1.3 with strong compounding, including approximately
25% CAGR as a stretch outcome. Those figures are evaluation objectives, not
parameters that may be used to loosen a validation gate or select a favourable
window.

The optimisation objective remains:

1. maximise out-of-sample Sharpe and Sortino;
2. constrain maximum drawdown and tail loss;
3. add only genuinely incremental, capacity-aware sleeves;
4. apply volatility-targeted sizing after signal validation;
5. let realised CAGR be an observed result.

## Evidence baseline

The latest reproducible real-data experiments are recorded in
`docs/backtest_reports/real_sleeve_matrix_10yr_20261104_034423.json` and
`docs/backtest_reports/real_data_10yr_oos_20261104_034819.json`.

- Inputs: 82 cached NSE EOD files with recorded SHA-256 hashes.
- Training window: 2014-08-09 through 2024-08-08.
- Out-of-sample window: 2024-08-09 through 2026-08-21 (504 sessions).
- Experimented sleeve: long-only cross-sectional momentum.
- Result: OOS Sharpe -0.940; NIFTY benchmark OOS Sharpe -0.357.
- Canonical rejection: survivorship-biased universe and OOS Sharpe below the
  gate.

The sleeve matrix tested six predeclared rank sleeves and two train-only
portfolio weighting methods on the same boundary. Reversal-3-month achieved
the best tested OOS Sharpe at 0.262; equal-weight and inverse-volatility
combinations were -0.551 and -0.753. These are diagnostic results, not
validated performance.

The repeated four-fold experiment is recorded in
`docs/backtest_reports/real_repeated_sleeves_10yr_4fold_20261104_043738.json`.
It used 74 older cached symbols, disjoint 504-session OOS folds, and selected
each fold's configuration from that fold's training window. The selected
aggregate was Sharpe 0.742 with 21.24% annualised return and 38.42% maximum
drawdown. The strongest fixed candidate was Sharpe 0.892 with 25.28%
annualised return and 35.14% maximum drawdown. The result is encouraging as a
research lead but fails the drawdown gate and remains survivorship-biased.

This is evidence that the tested sleeve/configuration failed its OOS test. It is
not evidence about the complete multi-sleeve ARTHA portfolio, because the
current reproducible runner has not yet evaluated every implemented sleeve
through the same data, cost, and validation protocol.

## Loose ends, ordered by impact

### P0 — Correctness blockers

1. **Point-in-time universe.** The current real-data experiment uses files that
   have complete coverage through the end date. That is survivorship-biased and
   does not model IPOs, delistings, historical index membership, or symbols that
   were investable at an earlier date but absent later.
2. **Source and corporate-action metadata.** Every price series needs source,
   retrieval time, adjusted/raw status, split/bonus/dividend treatment, and
   quality flags. A price file without those fields cannot be treated as a
   production-grade validation input.
3. **One causal backtest contract.** All sleeves must use the same explicit
   convention: information available at decision time, next executable bar,
   realistic costs, missing-price handling, liquidity limits, and no implicit
   mark-to-market or equal-weight fallback.
4. **Repeated walk-forward evaluation.** A single train/test split is not
   sufficient. The experiment protocol must run multiple chronological windows,
   preserve an untouched final period, and retain every failed result.
5. **Multiple-testing control.** Candidate configurations, universe rules,
   feature definitions, and portfolio weights must be registered before the
   final test. Parameter stability and search breadth must be recorded.

### P1 — High-value quantitative work

6. **Sleeve-by-sleeve measurement.** Run the same protocol for momentum,
   value/quality, mean reversion, PCA residual stat-arb, cointegrated pairs,
   volatility/regime overlays, and MCX calendar/inter-commodity spreads.
7. **Incremental portfolio construction.** Combine only sleeves whose OOS return
   streams add risk-adjusted value after costs. Use train-only covariance and
   risk estimates, exposure caps, correlation clustering, and drawdown-aware
   volatility targeting.
8. **Capacity and execution realism.** Add turnover, ADV participation, spread,
   price-impact, lot-size, tick-size, circuit-limit, halt, and stale-data
   constraints. Report gross and net results separately.
9. **Regime robustness.** Report results by volatility/trend regime and by
   calendar subperiod. A sleeve that works only in one regime is not a general
   validated signal.
10. **MCX and options evidence.** Do not include options or MCX sleeves in the
    portfolio until their historical observations have authoritative contract
    identity, expiry/roll treatment, and real market-data provenance.

### P2 — System and research discipline

11. **Exact replay path.** Historical replay and paper trading must consume the
    same signal, risk, sizing, and fill interfaces as the backtest, with the
    selected configuration and data hashes persisted.
12. **Validation evidence propagation.** Only deterministic backtest evidence
    can set `validated=True`; agents may narrate accepted facts but cannot create
    or alter quantitative evidence.
13. **Observability.** Persist experiment id, code/version, configuration,
    source hashes, quality exclusions, costs, fills, and rejection reasons.
14. **Dependency and data-source closure.** Clean-install checks, official MCX
    fixtures, optional FinBERT status, and provider/cache freshness diagnostics
    must be complete before live-market observation is treated as evidence.

Independent repository audits completed on 2026-11-04 confirm these are active
blockers: legacy multi-strategy artifacts use full-sample HMM labels, static
current fundamentals, inconsistent wealth-curve accounting, and evaluated-sample
variant selection. Those artifacts remain research output and are excluded from
the validated signal path.

## Experiment protocol

Every candidate experiment must produce:

- a source manifest and SHA-256 hashes;
- an as-of universe snapshot for each rebalance date;
- train, validation, and untouched OOS date ranges;
- signal/configuration/version identifiers;
- gross returns, net returns, turnover, costs, exposure, and capacity;
- annualised return, Sharpe, Sortino, maximum drawdown, tail loss, hit rate, and
  observation count;
- subperiod/regime breakdown and parameter-perturbation results;
- correlations to existing accepted sleeves;
- explicit rejection or promotion status.

The final OOS period is never used for feature selection, parameter choice,
universe selection, or portfolio weighting. No LLM participates in numeric
calculation or validation.

## Execution sequence

1. **Data contract:** add point-in-time universe and quality/provenance
   manifests; correct corporate-action and missing-session handling.
2. **Backtest contract:** make next-bar execution, costs, liquidity, halts, and
   stale data common to every sleeve.
3. **Walk-forward harness:** add repeated ten-year-train chronological windows,
   purging/embargo where labels overlap, final holdout protection, and
   multiple-testing metadata.
4. **Sleeve experiments:** measure each deterministic sleeve independently on
   real NSE/MCX data. Reject weak sleeves rather than tuning them indefinitely.
5. **Portfolio engine:** combine accepted, low-correlation sleeves with
   train-only risk estimates, exposure limits, and volatility targeting.
6. **Replay and paper validation:** replay the frozen portfolio through the
   historical data path, then observe it with real delayed/live data using only
   virtual fills.
7. **Narrative integration:** expose only accepted, cited signals to Fact Store
   agents and reports.
8. **Broker review:** consider a broker-neutral integration only after the
   preceding gates pass and current broker/regulatory requirements are verified.

### Completed in this optimization pass

- The historical runner now applies the canonical Sharpe/drawdown/observation
  gate and writes explicit rejection reasons, validation ID, and source-manifest
  hash. Its chronological holdout remains ineligible for Fact Store promotion.
- The generic walk-forward engine now requires exact OOS signal coverage,
  applies costs inside each fold, accounts for the OOS remainder, records failed
  folds and boundaries, and rejects incomplete coverage.
- Default HMM regime detection now uses causal forward filtering and expanding
  pre-observation fits instead of full-series Viterbi labels in historical use.
- Virtual paper rebalancing now liquidates removed targets, records actual
  partial-fill quantities, and uses quote observation timestamps where the
  provider supplies them.
- Added a point-in-time security-master seam with canonical IDs, aliases, and
  listing/delisting intervals. Runs without an authoritative master remain
  explicitly marked survivorship-biased.
- Added publication-date-aware fundamental history and made static fundamentals
  fail closed in historical factor backtests.
- Corrected factor-sleeve metrics to use compounded wealth, canonical Sharpe,
  and wealth-curve drawdown rather than arithmetic P&L sums.
- Applied the canonical validation thresholds to PCA stat-arb diagnostics and
  corrected its position-count lag so weights use the same next-day signal as
  returns.
- Paper trading now requires explicit canonical walk-forward evidence before an
  initialized strategy engine may generate virtual fills; legacy report metrics
  are no longer treated as paper-trading authority.

## Promotion rule

No sleeve or portfolio is promoted because it reaches Sharpe 1.3 or 25% CAGR
in-sample. Promotion requires reproducible real-data evidence, positive and
stable untouched OOS performance after costs, bounded drawdown, sufficient
observations, parameter stability, capacity evidence, and no unresolved data or
traceability defect. A failed result remains part of the research record.
