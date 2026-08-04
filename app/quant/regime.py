"""
Regime Detector — Hidden Markov Model Market Regime Classification
===================================================================
Classifies market conditions into regimes (BULL, BEAR, SIDEWAYS) using
a Gaussian Hidden Markov Model on return + volatility features.

Why this matters:
  Trading signals that work in a bull regime may be disastrous in a
  crisis regime.  The Regime Detector allows downstream agents and the
  Alpha Loop to context-gate their signals.

Design:
  - 3-state Gaussian HMM (hmmlearn) on 2 features:
    (1) rolling 20-day log-return, (2) rolling 20-day realized volatility.
  - States are labeled BULL/BEAR/SIDEWAYS by sorting on mean return
    (highest mean-return state = BULL, lowest = BEAR, middle = SIDEWAYS).
  - Training on rolling windows (default 2 years = 504 trading days).
  - Output: RegimeState per date, written to Fact Store as QuantSignal.

Edge cases:
  - < 50 observations: returns UNKNOWN for all dates.
  - Constant-return series: returns SIDEWAYS (zero variance handled).
  - NaN/holiday gaps: forward-filled before HMM fitting.

Reference: docs/ARTHA_ARCHITECTURE.md §5.1
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum observations required for meaningful HMM fitting
_MIN_OBSERVATIONS = 50

# Default rolling window for features
_ROLLING_WINDOW = 20

# Number of HMM states
_N_STATES = 3

# Number of HMM fitting iterations
_N_ITER = 100

# Covariance type: 'full' allows correlated features,
# 'diag' is more robust with limited data.
# Trade-off: full is more expressive but needs more data to estimate.
_COV_TYPE = "diag"


class RegimeState(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Hidden Markov Model regime detector.

    Usage::

        detector = RegimeDetector()
        detector.fit(price_series)  # pandas Series of close prices
        regimes = detector.predict(price_series)
        current = detector.current_regime(price_series)
    """

    def __init__(
        self,
        n_states: int = _N_STATES,
        rolling_window: int = _ROLLING_WINDOW,
        n_iter: int = _N_ITER,
        covariance_type: str = _COV_TYPE,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.rolling_window = rolling_window
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model = None
        self._state_labels: dict[int, RegimeState] = {}
        self._is_fitted = False

    # ── Feature Engineering ─────────────────────────────────────────────

    def _compute_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute HMM input features from a price series.

        Features:
          1. Rolling 20-day log-return (smoothed trend signal)
          2. Rolling 20-day realized volatility (risk signal)

        Returns a DataFrame with columns ['return', 'volatility'],
        NaN rows dropped.
        """
        if prices.empty or len(prices) < self.rolling_window + 2:
            return pd.DataFrame(columns=["return", "volatility"])

        # Forward-fill any NaN/holiday gaps
        prices = prices.ffill()

        # Daily log returns
        log_ret = np.log(prices / prices.shift(1))

        # Rolling features
        rolling_return = log_ret.rolling(window=self.rolling_window).mean()
        rolling_vol = log_ret.rolling(window=self.rolling_window).std()

        features = pd.DataFrame({
            "return": rolling_return,
            "volatility": rolling_vol,
        }, index=prices.index)

        # Drop NaN rows (first rolling_window entries + any gaps)
        features = features.dropna()

        # Handle edge case: if volatility is constant zero (e.g., stock halted)
        # add a tiny epsilon to prevent singular covariance matrix
        if (features["volatility"] == 0).all():
            features["volatility"] = 1e-10

        return features

    # ── Model Fitting ───────────────────────────────────────────────────

    def fit(self, prices: pd.Series) -> bool:
        """
        Fit the HMM on a price series.

        Parameters
        ----------
        prices : pd.Series
            Close prices indexed by date.

        Returns
        -------
        bool
            True if fitting succeeded, False if insufficient data.
        """
        features = self._compute_features(prices)

        if len(features) < _MIN_OBSERVATIONS:
            logger.warning(
                "Insufficient data for HMM fitting: %d observations (need %d)",
                len(features), _MIN_OBSERVATIONS,
            )
            self._is_fitted = False
            return False

        try:
            from hmmlearn.hmm import GaussianHMM

            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )

            X = features.values
            self._model.fit(X)

            # Label states by mean return (highest = BULL, lowest = BEAR)
            self._label_states()
            self._is_fitted = True

            logger.info(
                "HMM fitted: %d observations, %d states, log-likelihood=%.2f",
                len(features), self.n_states, self._model.score(X),
            )
            return True

        except Exception as e:
            logger.error("HMM fitting failed: %s", e)
            self._is_fitted = False
            return False

    def _label_states(self) -> None:
        """
        Assign semantic labels to HMM states based on mean return.

        The state with the highest mean return → BULL.
        The state with the lowest mean return → BEAR.
        The middle state → SIDEWAYS.
        """
        if self._model is None:
            return

        # means_ shape: (n_states, n_features)
        # Feature 0 is the rolling return
        mean_returns = self._model.means_[:, 0]
        sorted_indices = np.argsort(mean_returns)

        labels = [RegimeState.BEAR, RegimeState.SIDEWAYS, RegimeState.BULL]

        self._state_labels = {}
        for rank, state_idx in enumerate(sorted_indices):
            if rank < len(labels):
                self._state_labels[state_idx] = labels[rank]
            else:
                # More than 3 states — label extras as SIDEWAYS
                self._state_labels[state_idx] = RegimeState.SIDEWAYS

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, prices: pd.Series, min_regime_days: int = 5) -> pd.Series:
        """
        Predict regime states for a price series using Viterbi decoding.

        Uses the Viterbi algorithm (most likely STATE SEQUENCE) instead
        of per-observation marginal posteriors. Then applies a minimum
        regime duration filter to prevent daily oscillation — a regime
        must persist for at least `min_regime_days` to be recorded.

        Returns a pd.Series of RegimeState values, indexed by date.
        Dates before the rolling window are labeled UNKNOWN.
        """
        if not self._is_fitted or self._model is None:
            return pd.Series(
                RegimeState.UNKNOWN,
                index=prices.index,
                dtype=object,
                name="regime",
            )

        features = self._compute_features(prices)

        if features.empty:
            return pd.Series(
                RegimeState.UNKNOWN,
                index=prices.index,
                dtype=object,
                name="regime",
            )

        try:
            hidden_states = self._model.predict(features.values)

            # Map integer states to labels
            raw_labels = [self._state_labels.get(s, RegimeState.UNKNOWN) for s in hidden_states]

            # ── Minimum duration filter ─────────────────────────────
            # Suppress regime changes that last fewer than min_regime_days.
            # This prevents the HMM from oscillating between states daily.
            filtered = list(raw_labels)
            if min_regime_days > 1 and len(filtered) > min_regime_days:
                i = 0
                while i < len(filtered):
                    # Find run of same regime
                    j = i + 1
                    while j < len(filtered) and filtered[j] == filtered[i]:
                        j += 1
                    run_len = j - i

                    # If run is too short and not at edges, merge with previous regime
                    if run_len < min_regime_days and i > 0:
                        prev_regime = filtered[i - 1]
                        for k in range(i, j):
                            filtered[k] = prev_regime

                    i = j

            labeled = pd.Series(
                filtered,
                index=features.index,
                dtype=object,
                name="regime",
            )

            # Reindex to full price series, filling missing dates with UNKNOWN
            full = labeled.reindex(prices.index, fill_value=RegimeState.UNKNOWN)
            return full

        except Exception as e:
            logger.error("HMM prediction failed: %s", e)
            return pd.Series(
                RegimeState.UNKNOWN,
                index=prices.index,
                dtype=object,
                name="regime",
            )

    def current_regime(self, prices: pd.Series) -> RegimeState:
        """Return the regime for the most recent date in the series."""
        regimes = self.predict(prices)
        if regimes.empty:
            return RegimeState.UNKNOWN
        return regimes.iloc[-1]

    # ── Diagnostics ─────────────────────────────────────────────────────

    def get_state_statistics(self) -> dict:
        """
        Return per-state statistics from the fitted model.

        Useful for understanding what each regime looks like numerically.
        """
        if not self._is_fitted or self._model is None:
            return {}

        stats = {}
        for state_idx, label in self._state_labels.items():
            mean_ret = self._model.means_[state_idx, 0]
            mean_vol = self._model.means_[state_idx, 1]

            # Annualize: daily features × √252
            ann_ret = mean_ret * 252 * 100  # approximate annualized %
            ann_vol = mean_vol * np.sqrt(252) * 100

            stats[label.value] = {
                "daily_mean_return": float(mean_ret),
                "daily_mean_volatility": float(mean_vol),
                "annualized_return_pct": round(float(ann_ret), 2),
                "annualized_volatility_pct": round(float(ann_vol), 2),
                "stationary_probability": float(
                    self._model.get_stationary_distribution()[state_idx]
                ) if hasattr(self._model, 'get_stationary_distribution') else None,
            }

        return stats

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ── Convenience function ────────────────────────────────────────────────────────

def detect_regime(
    prices: pd.Series,
    training_prices: Optional[pd.Series] = None,
) -> pd.Series:
    """
    One-shot regime detection.

    If training_prices is provided, fits on that and predicts on prices.
    Otherwise fits and predicts on the same series.
    """
    detector = RegimeDetector()
    fit_data = training_prices if training_prices is not None else prices
    detector.fit(fit_data)
    return detector.predict(prices)
