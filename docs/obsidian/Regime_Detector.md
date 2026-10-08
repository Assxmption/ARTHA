# Regime Detector

Located in `app/quant/regime.py`.

The Regime Detector classifies the overall market environment into distinct states. This is critical for context-gating: strategies that work in a low-volatility BULL market (like short puts) will cause catastrophic drawdowns in a high-volatility BEAR market.

## The Model: Gaussian Hidden Markov Model (HMM)
ARTHA uses a 3-state `GaussianHMM` (via `hmmlearn`).

### Feature Engineering
It trains on two specific rolling features (default 20-day window):
1. **Rolling Log-Return**: Captures the smoothed directional trend.
2. **Rolling Realized Volatility**: Captures the market's risk state.

These features are passed through a `StandardScaler` prior to fitting to ensure the larger variance of the volatility feature doesn't drown out the return feature during K-Means initialization.

### State Labeling Mechanism
Because HMM states are initialized randomly, State 0 might be "Bull" in one run and "Bear" in another. ARTHA solves this deterministically:
- It extracts the `means_` from the fitted model.
- It sorts the 3 states by their mean return feature.
- **Highest Return State** $\rightarrow$ `BULL`
- **Lowest Return State** $\rightarrow$ `BEAR`
- **Middle Return State** $\rightarrow$ `SIDEWAYS`

### Prediction
To determine the current regime, the engine uses **Viterbi decoding** (`model.predict()`), which computes the most likely sequence of hidden states, rather than just taking the marginal posterior probability of a single day. The output is saved to the [[Fact_Store]] as a `QuantSignal` for downstream agents.
