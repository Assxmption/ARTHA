# Multi-Factor Alpha Model

Located in `app/quant/factors.py`.

The Multi-Factor Alpha Model is responsible for cross-sectional stock ranking. Instead of looking at a stock in isolation, it compares all stocks on a given date to find relative outperformance.

## Factor Families
ARTHA computes four classic factors:
1. **Value**: Measured via EPS Yield (EPS / Price). Higher is cheaper.
2. **Quality**: Measured via Return on Equity (ROE).
3. **Momentum**: 12-1 month momentum. Looks at the trailing 12 months but skips the most recent 21 trading days (1 month) to avoid short-term mean reversion traps.
4. **Low Volatility**: Realized volatility. Lower volatility scores higher, exploiting the low-volatility anomaly.

## MAD-Robust Z-Scoring
Financial data is heavy-tailed and full of outliers. ARTHA uses **Median Absolute Deviation (MAD)** instead of standard deviation for cross-sectional z-scoring.
- Formula: $Scale \approx 1.4826 \times MAD$
- This ensures that a stock with a $1000\%$ earnings beat doesn't distort the scores of all other stocks in the universe.

## Composite Alpha
The individual z-scores are combined into a `composite_alpha`. By default, this is equal-weighted (25% to each of the 4 factors) to prevent data-mined curve fitting. The result is saved as a `FactorExposure` object to the [[Fact_Store]].
