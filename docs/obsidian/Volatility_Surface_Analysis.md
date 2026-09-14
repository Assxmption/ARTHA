# Volatility Surface Analysis

## Overview
The **Volatility Surface Engine** (`app/quant/vol_surface.py`) constructs, interpolates, and analyzes the full implied volatility surface $\sigma(K, T)$ across strikes and expiries. This is the professional standard for options modeling, allowing trading based on relative mispricings rather than a single flat IV.

## Architecture & Implementation
The engine leverages the **SVI (Stochastic Volatility Inspired)** parameterization formulated by Jim Gatheral (2004).

### Key Components:
- **SVI Parameterization**: Fits total implied variance $w(k, T)$ using 5 parameters:
  - $a$: Level of variance.
  - $b$: Slope of wings.
  - $\rho$: Rotation (skew).
  - $m$: Translation (smile center).
  - $\sigma$: Smoothing.
- **Surface Construction**: Builds the SVI-parameterized IV matrix from raw NSE option chain data. Supports synthetic surface construction from a single ATM vol (e.g., India VIX) using an empirical skew model when chain data is unavailable.
- **Arbitrage Detection**: Enforces no-arbitrage constraints:
  - **Calendar Arbitrage**: Checks if variance decreases with time to expiry.
  - **Butterfly Arbitrage**: Checks for negative convexity in the smile.
  - **Negative Variance**: Ensures $w > 0$.
- **Skew Metrics**: Computes standard structural metrics:
  - 25-Delta Risk Reversal (skew bias).
  - 25-Delta Butterfly (smile convexity).
  - Term Structure slope.

## Role in the Pipeline
The Volatility Surface is the core object consumed by [[Options_Strategies]] and the [[Options_Backtester]]. It is a prerequisite for correctly pricing complex options structures and detecting market anomalies.

## Constraints
- **AGENTS.md Rule 1**: Surface construction and interpolation are completely deterministic.
