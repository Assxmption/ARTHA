# Robust Outlier Detection & Z-Scoring

**Status**: Implemented & Tested  
**Path**: `app/quant/robust_outliers.py`  
**Test Suite**: `tests/test_robust_outliers.py`  

## Overview
The **Robust Outlier Detection & Z-Scoring Module** provides the mathematical framework for handling unusual data points across ARTHA. It directly operationalizes **AGENTS.md Rule 7**, which establishes a strict dichotomy between two distinct phenomena:

1. **Bad Data (Fat-Finger Prints, Vendor Feed Glitches, Tape Errors)**:
   - Must be caught at the ingestion boundary.
   - Evaluated via multi-signal confirmation (MAD z-score + volume cross-check + price-range consistency).
   - Flagged explicitly for human review — **never silently dropped**.
2. **Genuine Large Market Moves (Circuit Breakers, Macro Shocks, Earnings Surprises)**:
   - **Never scrubbed or truncated**.
   - These points contain the highest information density and are required by the [[Regime_Detector]] to detect volatility transitions and regime shifts.

---

## Architecture & Detection Methodologies

### 1. Ingestion Bad-Data Classifier (`flag_bad_data`)
Ingested daily OHLCV bars are evaluated using three orthogonal criteria:
1. **Daily Return Robust Z-Score**:
   $$\hat{z}_{\text{ret}} = \frac{|R_t - \text{median}(R)|}{1.4826 \times \text{MAD}(R)}$$
   Flags data points where $\hat{z}_{\text{ret}} > 8.0$ (corresponding to $\approx 5.9\sigma$ under Gaussianity, statistical near-impossibility).
2. **Volume Cross-Check**:
   Computes volume log-z-score $z_{\text{vol}}$. Genuine moves are accompanied by volume surges ($|z_{\text{vol}}| \ge 1.5$), whereas data glitches typically show normal volume ($|z_{\text{vol}}| < 1.5$).
3. **Price-Range Consistency**:
   Verifies physical validity: $\text{Low}_t \le \text{Close}_t \le \text{High}_t$. Any close outside the day's high-low range is flagged immediately as corrupt data.

#### Classification Matrix:
| Price Range Valid | Return $\hat{z} > 8.0$ | Volume Surge ($|z_{\text{vol}}| \ge 1.5$) | Circuit Band Match | Final Classification |
| :--- | :--- | :--- | :--- | :--- |
| **No** | Any | Any | Any | `BAD_DATA` (Corrupt tape) |
| **Yes** | Yes | No | No | `BAD_DATA` (Suspicious price jump) |
| **Yes** | Yes | Yes / No | Yes (within 0.5% tolerance) | `CIRCUIT_HIT` (NSE Halt) |
| **Yes** | Yes | Yes | No | `GENUINE_MOVE` (Macro / News Shock) |

---

### 2. NSE Circuit-Breaker Detection (`detect_circuit_halts`)
NSE operates circuit-limit price bands:
$$\text{Bands} \in \{\pm 5\%, \pm 10\%, \pm 20\%\}$$

The detector flags an active circuit freeze when:
1. The return matches an NSE band within $\pm 0.5\%$ tolerance.
2. The intraday range exhibits **range compression**:
   - **Upper Circuit**: $\text{Close}_t \approx \text{High}_t$ and $(\text{High}_t - \text{Close}_t) < 0.05 \times (\text{High}_t - \text{Low}_t) + 0.01$
   - **Lower Circuit**: $\text{Close}_t \approx \text{Low}_t$ and $(\text{Close}_t - \text{Low}_t) < 0.05 \times (\text{High}_t - \text{Low}_t) + 0.01$

---

### 3. Stat-Arb Spread Z-Scoring (`statarb_zscore`)
Standard rolling mean/std z-scores fail in pairs trading because outliers distort the denominator, causing delayed entries. Conversely, naive MAD z-scores explode to $\pm 30$ to $\pm 50$ during structural regime transitions, causing excessive position sizing.

ARTHA resolves this using a hybrid **EWM-median/MAD estimator with Winsorization**:
1. **Initial Envelope**: Computes rolling median and rolling MAD over lookback $T$.
2. **First-Pass Winsorization**: Clips raw spread $S_t$ to $\text{median} \pm 3 \times \hat{\sigma}_{\text{MAD}}$.
3. **Exponentially Weighted Statistics**: Computes $\mu_{\text{EWM}}$ and $\sigma_{\text{EWM}}$ on the clipped spread with halflife $\tau$ (default 20 days).
4. **Final Z-Scoring & Cap**:
   $$z_t = \text{clip}\left(\frac{S_t - \mu_{\text{EWM}}}{\sigma_{\text{EWM}}}, -z_{\text{winsor}}, +z_{\text{winsor}}\right)$$
   Default $z_{\text{winsor}} = 6.0$.

---

### 4. Cross-Sectional Robust Z-Score (`robust_zscore_cross_sectional`)
Used for factor scoring in [[Alpha_Factors]] and [[Signal_Factory]]:
- Calculates $\text{median}$ and $\text{MAD}$ across all universe stocks on date $t$.
- Normalizes: $z_i = (x_i - \text{median}) / (1.4826 \times \text{MAD})$.
- Clips extreme factor exposures at $\pm 6.0$.

---

## Key Functions & Exports
- `flag_bad_data(df, symbol, mad_threshold=8.0)`: Multi-signal ingestion validator.
- `detect_circuit_halts(df, tolerance=0.005)`: NSE circuit-breaker halt detector.
- `statarb_zscore(spread, lookback=60, winsor_z=6.0, use_ewm=True)`: Stat-arb spread z-scorer.
- `robust_zscore_rolling(series, lookback=60, winsor_z=6.0)`: Generic rolling time-series z-scorer.
- `robust_zscore_cross_sectional(values, winsor_z=6.0)`: Single cross-section normalizer.

---

## Architectural Role & Dependencies
- **Data Ingestion**: Called by [[Data_NSE]] and [[Data_Layer]] to audit OHLCV series.
- **Quant Models**: Ingested by [[PCA_Stat_Arb]], [[Stat_Arb_Screener]], and [[Alpha_Factors]].
- **Compliance**: AGENTS.md Rule 7 (Outlier hygiene) and Rule 1 (Strictly deterministic math).
