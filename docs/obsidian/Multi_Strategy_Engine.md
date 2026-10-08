# Multi-Strategy Engine

## Overview
The **Multi-Strategy Portfolio Engine** (`app/quant/multi_strategy.py`) is responsible for combining return streams from multiple independent strategy sleeves into a single, unified portfolio. This encapsulates the core quantitative insight: allocating across $N$ uncorrelated strategies to increase the overall portfolio Sharpe ratio.

## Architecture & Implementation
The engine ingests daily return series (e.g., from [[Factor_Backtester]], [[Stat_Arb_Backtester]], [[Options_Backtester]]) and optimally allocates capital to maximize risk-adjusted returns.

### Key Components:
- **Vol-Targeting**: Each strategy's return stream is normalized to a target volatility (e.g., 10% annualized) using rolling realized volatility estimates. This ensures equal risk contribution before allocation.
- **Allocation Optimization**:
  - Uses correlation-aware **Maximum Sharpe (Tangency Portfolio)** optimization.
  - Employs **Ledoit-Wolf shrinkage** covariance estimation to regularize the sample covariance matrix, improving out-of-sample stability.
  - Falls back to Sharpe²-proportional (Kelly-optimal) weights if optimization fails or covariance cannot be reliably estimated.
- **Portfolio Metrics**: Computes combined statistics including Sharpe, Sortino, Calmar, Max Drawdown, and a Diversification Ratio (Portfolio Sharpe / Max Individual Sharpe).

## Role in the Pipeline
This engine acts as the final quantitative aggregation layer. Its output (the combined portfolio returns and weights) is recorded to the [[Fact_Store]] as a verified entity. The [[Writer_Agent]] then uses this deterministic output to narrate the portfolio's performance.

## Constraints
- **AGENTS.md Rule 8**: The system optimizes for Sharpe/Sortino and bounded drawdown, *never* for a fixed return percentage target.
