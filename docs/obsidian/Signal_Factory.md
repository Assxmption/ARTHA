# Signal Generation Engine (Signal Factory)

Located in `app/quant/signals.py`.

This module acts as a "Medallion-class" signal factory, generating a massive matrix of features to feed into the machine learning models (`ml_alpha.py`). It computes **40 base signals** across 8 distinct families for every stock, on every trading day.

## Signal Families
1. **Momentum (9)**: Multi-timeframe price returns (1d to 252d).
2. **Mean-Reversion (6)**: Deviations from SMAs and Bollinger Bands.
3. **Volume (4)**: Z-scored volume, breakouts, VWAP deviations, and price-volume divergence.
4. **Volatility (4)**: Realized volatility, vol ratio (short/long), and vol momentum (acceleration).
5. **Fundamental (6)**: Valuation and quality metrics (EPS yield, P/B inv, ROE).
6. **Cross-Sectional (3)**: Sector-relative measures (stock mom minus sector peer mean).
7. **Seasonal (4)**: Calendar effects (Day of week, Month of year, Expiry week, Quarter-end window dressing).
8. **Technical-as-Feature (4)**: RSI, MACD, ADX, OBV — all explicitly normalized as features, not used as binary buy/sell rules.

## Cross-Sectional Normalization
Every single signal is cross-sectionally z-scored (using MAD-robust logic) across all available stocks on that day. 
- **Missing Data Handling**: If a stock is missing data for a signal (e.g., recently IPO'd stock missing 252d momentum), it defaults to `0.0`. Because all signals are z-scored, a `0.0` simply means "average," preventing the model from crashing while cleanly imputing missing data without looking forward.

## Output
The factory returns a DataFrame of shape `(n_dates × n_stocks, n_signals)`. This acts as the raw feature matrix `X` for downstream supervised learning.
