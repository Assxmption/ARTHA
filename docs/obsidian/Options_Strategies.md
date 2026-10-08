# Options Strategies

**Status**: Implemented & Validated
**Path**: `app/quant/options_strategies.py`

The Options Strategies sub-module is a declarative, leg-based architecture for structuring multi-leg options strategies. It acts as the building block for the [[Options_Backtester]].

## Strategy Architecture

Strategies are composed of `StrategyLeg` objects grouped into a `StrategyMetrics` aggregate.

### `StrategyLeg`
A single option or equity position.
- **Attributes**: `strike`, `is_call`, `is_long`, `quantity`, `is_option` (to support covered calls).
- **Computes**: Individual leg margin impact (approximated for naked vs. covered).

### `StrategyMetrics`
The aggregate struct representing a complete trade position.
- **Fields**:
  - `symbol`, `strategy_type` (Enum mapping).
  - `net_premium`: Total credit received or debit paid.
  - `max_profit` / `max_loss`: Computed worst/best case scenarios at expiration.
  - `margin_required`: Estimated using SEBI-like SPAN margin heuristics (hedged strategies get reduced margin).
  - `greeks`: Aggregate Portfolio Greeks (`OptionsGreeks` object).
- **Methods**: 
  - `payoff_at_price(spot)`: Calculates the exact payoff of the strategy at expiration given a spot price.

## Supported Strategies (Factory Methods)

The module provides factory functions to instantly assemble complex setups based on a target OTM (Out-Of-The-Money) distance (derived from IV and DTE).

1. **Iron Condor** (`build_iron_condor`)
   - 4 legs: Short OTM Put, Long further OTM Put, Short OTM Call, Long further OTM Call.
   - Neutral strategy, profits from time decay and IV crush.

2. **Bull Put Spread** (`build_bull_put_spread`)
   - 2 legs: Short OTM Put, Long further OTM Put.
   - Bullish/Neutral strategy, defined risk.

3. **Bear Call Spread** (`build_bear_call_spread`)
   - 2 legs: Short OTM Call, Long further OTM Call.
   - Bearish/Neutral strategy, defined risk.

4. **Short Strangle** (`build_short_strangle`)
   - 2 legs: Short OTM Put, Short OTM Call. (Optional wings to cap tail risk).
   - High margin, undefined risk (unless wings attached).

5. **Covered Call** (`build_covered_call`)
   - 2 legs: Long 100 shares of Underlying, Short OTM Call.
   - Delta-positive, theta-positive yield enhancement.

## Linkages
- Uses `bs_price` and `bs_greeks` from [[Options_Pricing]] for building the net premium and greeks profile.
- Instances of `StrategyMetrics` are managed and tracked inside the `OptionsBacktester` in [[Options_Backtester]].
