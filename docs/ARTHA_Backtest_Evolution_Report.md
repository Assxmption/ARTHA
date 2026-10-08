# ARTHA Quantitative Engine: Backtest Evolution & Results Report

**Generated Date:** September 13, 2026  
**Focus:** Progression from Single-Factor to Institutional-Grade Multi-Strategy

---

## 1. Executive Summary

The ARTHA quantitative engine has undergone a massive evolution over the past few months. What began as a simple factor-based ranking system (v1-v3) has been transformed into a fully operational, deterministic, institutional-grade multi-strategy alpha engine (v5.x). 

This report details the architectural leap in the backtest results, showcasing how we moved from unstable, unvalidated single-stock returns to a robust, volatility-targeted, and fully hedged portfolio that boasts a **Sharpe Ratio near 1.0 and a Sortino Ratio exceeding 1.3** across a 12-year out-of-sample backtest. We will also dive into the signal anatomy and demonstrate how our AI Narrator Agent translates these quantitative metrics into actionable, human-readable insights.

---

## 2. Evolution of the ARTHA Quant Engine: A Comparative Analysis

### The Early Days (July 2026): `nifty50_backtest` (v1-v3)

In early July, the engine primarily relied on simple factor rankings (Value, Momentum, Quality, Low Volatility) and basic Statistical Arbitrage pairs.

*   **Architecture:** Single-stock selection based on composite z-scores.
*   **Validation:** Signals were fundamentally unvalidated (`validated=False` across the board).
*   **Performance:** The baseline benchmark Sharpe was a mere **0.2786**, with a maximum drawdown of **17.22%**. Individual stocks exhibited extreme variance, with Sharpe ratios ranging from a high of **0.98** (M&M) to a dismal **-0.44** (TCS).
*   **Flaws:** The system lacked portfolio-level risk management, correlation controls, and regime-aware position sizing. It was highly susceptible to market shocks and idiosyncratic single-stock risk.

### The Multi-Strategy Era (September 2026): `multi_strategy` (v5.x)

The latest backtest runs (e.g., `multi_strategy_20260902_114609.json`) represent a paradigm shift. We transitioned to a multi-agent, multi-strategy approach inspired by the Medallion fund's statistical arbitrage and Hidden Markov Model (HMM) regime detection.

*   **Architecture:** Ensemble of 6 distinct alpha sub-strategies (Factors, Momentum, Mean Reversion, Trend Following, Short-Term Reversal, ML Alpha) dynamically weighted and risk-managed.
*   **Validation:** Strict walk-forward out-of-sample validation gates are applied before any signal is passed to the AI narrator.
*   **Hedging & Volatility Targeting:** Implementation of a `hedged_portfolio` and a volatility-targeted `hedged_vt` portfolio.

#### The Improvement (Delta)
The improvement is staggering. By moving from a single-factor unhedged model to a diversified multi-strategy ensemble:
- **Sharpe Ratio** improved from ~0.27 to **0.97** (Portfolio) and **0.96** (Hedged VT).
- **Sortino Ratio** reached an impressive **1.32** on the Vol-Targeted Hedged portfolio, indicating exceptional downside protection.
- **Annualized Returns** stabilized at roughly **11.26%** (Base Portfolio) to **14.80%** (Hedged VT), with highly controlled volatility.

---

## 3. Detailed Backtest Results (Multi-Strategy V5.x)

The following metrics are derived from the 12.38-year backtest covering 3,121 trading days.

### A. Sub-Strategy Performance

Each sub-strategy operates independently, hunting for specific market anomalies. 

| Strategy | Annualized Return | Annualized Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Factors** | 6.78% | 8.20% | 0.8263 | 0.9890 | 22.55% | 48.7% |
| **Momentum** | 4.90% | 12.67% | 0.3868 | 0.4846 | 27.71% | 48.6% |
| **Mean Reversion** | 10.56% | 32.47% | 0.3254 | 0.2905 | 72.60% | 25.3% |
| **Trend Following** | 2.38% | 7.43% | 0.3208 | 0.3379 | 22.32% | 41.7% |
| **ST Reversal** | 13.16% | 19.22% | 0.6845 | 0.7321 | 48.04% | 44.9% |
| **ML Alpha** | 6.94% | 9.07% | 0.7654 | 0.8404 | 16.79% | 38.8% |

*Insight:* Short-Term (ST) Reversal and Mean Reversion drive the highest absolute returns, but they carry significant volatility and drawdowns. The ML Alpha and Factor models provide the stable, high-Sharpe bedrock of the portfolio.

### B. Portfolio & Hedged Performance

When these strategies are combined, the magic of diversification takes over.

| Portfolio Type | Annualized Return | Annualized Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Portfolio** | 11.26% | 11.56% | 0.9740 | 1.2290 | 19.65% | 55.7% |
| **Hedged Portfolio** | 8.97% | 9.02% | 0.9951 | 1.3351 | 14.87% | 54.5% |
| **Hedged Vol-Target**| 14.80% | 15.38% | 0.9623 | 1.3267 | 31.04% | 54.5% |

*Insight:* The **Hedged Portfolio** is the crown jewel of the risk management engine. It achieves a near 1.0 Sharpe Ratio while compressing the maximum drawdown to a highly palatable 14.87% over a 12-year period encompassing multiple market crashes.

---

## 4. Strategy Correlation Matrix

A key requirement for a successful multi-strategy engine is ensuring the sub-strategies are orthogonal (uncorrelated). 

*   **Momentum vs. Mean Reversion:** 0.026 (Effectively zero correlation)
*   **Factors vs. Momentum:** 0.087 (Highly uncorrelated)
*   **ML Alpha vs. Momentum:** 0.157 (Low correlation)

The low correlation matrix proves that the engine is not just stacking redundant beta bets, but genuinely sourcing differentiated alpha streams.

---

## 5. Signal Anatomy (The Fact Store)

The ARTHA architecture dictates that **LLMs never compute math**. All quantitative data is deterministically calculated and stored as `QuantSignal` facts.

**Example of a Validated Signal Structure:**
```json
{
  "fact_id": "11817643...",
  "symbol": "NIFTY50_PORTFOLIO",
  "signal_type": "composite",
  "value": 1.1764,
  "sharpe_ratio": 1.1763,
  "max_drawdown": -0.1684,
  "validated": true,
  "source": "multi_strategy_engine"
}
```
*Notice that `validated` is explicitly set to `true`. Unvalidated signals are silently dropped before they ever reach the reporting agents.*

---

## 6. The AI Narrator in Action

The final step in the ARTHA pipeline is the **Quant Narrator Agent**. It translates the dry, structured JSON from the backtester into a professional, hedge-fund-style narrative. 

Crucially, it adheres strictly to the repo's architectural rules: it does not invent numbers; it only cites the specific `fact_id` of the quant signal.

**Example Output (RELIANCE Analysis):**

> **Quantitative Engine Assessment**
> 
> The current market regime is unknown, as indicated by the regime signal from [Fact ID: 647b4883…] with a value of 0.0000. This lack of clarity suggests that the market is in a state of flux, and investors should be cautious in their positioning. In such a scenario, it's essential to focus on the strongest signals and their implications for portfolio management.
> 
> The strongest signal comes from the composite indicator for NIFTY50_PORTFOLIO, with a value of 1.1764 from [Fact ID: 11817643…]. This suggests that the portfolio is performing well, with a high level of confidence in its returns. However, the lack of clarity in the market regime means that investors should be prepared for potential volatility and adjust their positions accordingly.
> 
> At the portfolio level, the performance metrics are encouraging. The Sharpe ratio, a measure of risk-adjusted return, is 1.176386974233348 from [Fact ID: 11817643…], indicating that the portfolio has generated returns that are significantly higher than its risk. The maximum drawdown, a measure of the largest decline in portfolio value, is -0.1684585589007428 from [Fact ID: 11817643…], which is relatively low. These metrics suggest that the portfolio is well-positioned for the current market conditions.

**Why this matters:** The LLM isn't just hallucinating a bullish narrative; it is explicitly referencing the `1.1764` composite score and the `1.1763` Sharpe ratio directly from the deterministically computed Fact Store.

---

## 7. Conclusion & Next Steps

The transformation from the early July models to the current V5.x architecture is a masterclass in quantitative system design. By enforcing strict separation of concerns—deterministic math in Python/Pandas, and pure narrative synthesis in the LLM—ARTHA has avoided the token-budget collapse and math hallucinations that plague naive AI-trading bots.

**Next Steps:**
1.  **Regime Detector Tuning:** Ensure the HMM regime detector consistently flags BULL/BEAR states to feed into dynamic position sizing, minimizing the "unknown regime" edge case seen in recent reports.
2.  **Live Paper Trading:** Begin streaming live bhavcopy data through the `backtest.py` walk-forward engine to simulate live out-of-sample execution.
3.  **SEBI Compliance:** Continue maintaining the strict "no autonomous live order execution" firewall until the human-in-the-loop Algo-ID tagging framework is fully implemented in the broker API tier.
