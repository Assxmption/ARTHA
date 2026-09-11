# Alpha Loop (Signal Orchestrator)

**Status**: Implemented & Validated
**Path**: `app/quant/alpha_loop.py`

The Alpha Loop acts as the master execution pipeline connecting raw data ingestion to signal generation and fact creation.

## Pipeline Workflow

1. **Regime Detection**: Calls the [[Regime_Detector]] on the index (`NIFTY50`) to ascertain the global market state (BULL, BEAR, SIDEWAYS).
2. **Quant Signal Building**:
   - **Factor Alpha**: Calls the [[Alpha_Factors]] composite generator.
   - **StatArb**: Calls the [[Stat_Arb_Screener]] for cointegrated pairs.
3. **Walk-Forward Validation**: Pipes the unvalidated signals through the [[Quant_Backtester]]. Only signals passing the OOS (Out-Of-Sample) constraints are toggled to `validated = True`.
4. **Fact Store Insertion**: Writes the validated `QuantSignal` schemas to the Fact Store for later consumption by the LLM Agents.

## AGENTS.md Compliance
- **Rule 1**: Absolute determinism. All signals are generated exclusively by math models; the LLM merely orchestrates the pipeline or reads the results.
- **Rule 5**: Checkpointing. Each stage emits structured records to prevent redundant computation in the event of pipeline failure.
- **Rule 6**: Validation Gate. Signals default to `validated = False` and are strictly excluded from the final portfolio narration if they fail walk-forward evaluation.
