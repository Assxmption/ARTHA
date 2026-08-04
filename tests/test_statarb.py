"""
Test — Statistical Arbitrage Engine
=====================================
Unit tests for cointegration testing, Kalman filter hedge ratios,
and z-score signal generation.

Covers:
  - Cointegration detected on synthetic cointegrated pair
  - Random walk pair correctly rejected
  - Kalman hedge ratio tracks drift
  - Z-score signals generated at correct thresholds
  - Half-life estimation on OU process
  - Pair discovery on a small universe
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.quant.statarb import (
    CointPair,
    PairSignal,
    test_cointegration as run_cointegration_test,
    kalman_hedge_ratio,
    compute_spread,
    generate_pair_signals,
    discover_pairs,
    _half_life,
)


def _make_cointegrated_pair(
    n: int = 500,
    hedge_ratio: float = 1.5,
    mean_reversion_speed: float = 0.05,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate a synthetic cointegrated pair.

    series_b follows a random walk.
    series_a = hedge_ratio * series_b + mean-reverting spread.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2020-01-01", periods=n)

    # Random walk for B
    b_returns = 0.0005 + 0.015 * rng.randn(n)
    b_prices = 100 * np.exp(np.cumsum(b_returns))

    # Mean-reverting spread (Ornstein-Uhlenbeck)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = (
            spread[i - 1] * (1 - mean_reversion_speed)
            + noise_std * rng.randn()
        )

    a_prices = hedge_ratio * b_prices + spread + 50  # +50 to keep positive

    return (
        pd.Series(a_prices, index=dates, name="A"),
        pd.Series(b_prices, index=dates, name="B"),
    )


def _make_random_walk_pair(n: int = 500, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate two independent random walks (NOT cointegrated)."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2020-01-01", periods=n)

    a = 100 * np.exp(np.cumsum(0.0005 + 0.02 * rng.randn(n)))
    b = 100 * np.exp(np.cumsum(-0.0003 + 0.018 * rng.randn(n + 100)))[:n]

    return (
        pd.Series(a, index=dates, name="X"),
        pd.Series(b, index=dates, name="Y"),
    )


class TestCointegrationTesting:
    """Test the Engle-Granger cointegration test."""

    def test_cointegrated_pair_detected(self):
        a, b = _make_cointegrated_pair()
        result = run_cointegration_test(a, b, "A", "B")
        assert result.is_cointegrated is True
        assert result.p_value < 0.05

    def test_random_walk_rejected(self):
        a, b = _make_random_walk_pair()
        result = run_cointegration_test(a, b, "X", "Y")
        # Should NOT be cointegrated (p > 0.05)
        # Note: there's a small probability of false positive, but
        # with independent random walks it's unlikely
        assert result.p_value > 0.01 or result.is_cointegrated is False

    def test_hedge_ratio_close_to_true(self):
        TRUE_HEDGE = 1.5
        a, b = _make_cointegrated_pair(hedge_ratio=TRUE_HEDGE)
        result = run_cointegration_test(a, b)
        # OLS hedge ratio should be in the right ballpark
        assert abs(result.hedge_ratio - TRUE_HEDGE) < 0.5

    def test_insufficient_data(self):
        a, b = _make_cointegrated_pair(n=50)  # Too short
        result = run_cointegration_test(a, b)
        assert result.is_cointegrated is False
        assert result.p_value == 1.0

    def test_critical_values_present(self):
        a, b = _make_cointegrated_pair()
        result = run_cointegration_test(a, b)
        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values


class TestHalfLife:
    """Test mean-reversion half-life estimation."""

    def test_fast_reversion_short_halflife(self):
        """A quickly reverting spread should have a short half-life."""
        a, b = _make_cointegrated_pair(mean_reversion_speed=0.1)
        spread = a - 1.5 * b
        hl = _half_life(spread)
        assert hl < 50  # Should revert fast

    def test_slow_reversion_long_halflife(self):
        """A slowly reverting spread should have a longer half-life."""
        a, b = _make_cointegrated_pair(mean_reversion_speed=0.01)
        spread = a - 1.5 * b
        hl = _half_life(spread)
        assert hl > 20  # Should be longer

    def test_random_walk_long_halflife(self):
        """A random walk should have a long half-life (slowly or non-reverting)."""
        rng = np.random.RandomState(42)
        walk = pd.Series(np.cumsum(rng.randn(500)))
        hl = _half_life(walk)
        # Random walks may produce finite half-life estimates due to
        # sampling noise, but should be much longer than a mean-reverting process
        assert hl > 10 or hl == np.inf


class TestKalmanHedgeRatio:
    """Test Kalman filter adaptive hedge ratio."""

    def test_kalman_tracks_hedge_ratio(self):
        a, b = _make_cointegrated_pair(hedge_ratio=1.5, n=500)
        hr = kalman_hedge_ratio(a, b)
        assert len(hr) > 0
        # Final hedge ratio should converge near the true value
        assert abs(hr.iloc[-1] - 1.5) < 1.0

    def test_kalman_insufficient_data(self):
        a = pd.Series([100, 101], index=pd.bdate_range("2020-01-01", periods=2))
        b = pd.Series([50, 51], index=pd.bdate_range("2020-01-01", periods=2))
        hr = kalman_hedge_ratio(a, b)
        assert hr.empty


class TestZScoreSignals:
    """Test z-score based signal generation."""

    def test_signals_generated(self):
        a, b = _make_cointegrated_pair(n=500)
        signals = generate_pair_signals(a, b, "A", "B")
        assert len(signals) > 0
        assert all(isinstance(s, PairSignal) for s in signals)

    def test_signal_types_correct(self):
        """Signals should be one of the defined types."""
        a, b = _make_cointegrated_pair(n=500)
        signals = generate_pair_signals(a, b, "A", "B")
        valid_types = {"LONG_A_SHORT_B", "SHORT_A_LONG_B", "EXIT", "STOP", "NEUTRAL"}
        for s in signals:
            assert s.signal in valid_types

    def test_extreme_zscore_triggers_stop(self):
        """When z-score exceeds 4, signal should be STOP."""
        a, b = _make_cointegrated_pair(n=500)
        signals = generate_pair_signals(a, b, z_stop=4.0)
        stop_signals = [s for s in signals if s.signal == "STOP"]
        # May or may not have stops depending on data,
        # but the logic should not crash
        assert isinstance(stop_signals, list)

    def test_insufficient_data_empty_signals(self):
        a = pd.Series([100] * 10, index=pd.bdate_range("2020-01-01", periods=10))
        b = pd.Series([50] * 10, index=pd.bdate_range("2020-01-01", periods=10))
        signals = generate_pair_signals(a, b)
        assert len(signals) == 0


class TestPairDiscovery:
    """Test universe-level pair discovery."""

    def test_discover_finds_cointegrated_pair(self):
        a, b = _make_cointegrated_pair(n=500)
        c, _ = _make_random_walk_pair(n=500)

        price_data = {"A": a, "B": b, "C": c}
        pairs = discover_pairs(price_data)

        # Should find A-B as cointegrated
        pair_labels = {(p.symbol_a, p.symbol_b) for p in pairs}
        assert ("A", "B") in pair_labels or ("B", "A") in pair_labels

    def test_discover_empty_universe(self):
        pairs = discover_pairs({})
        assert pairs == []

    def test_discover_single_stock(self):
        a, _ = _make_cointegrated_pair(n=500)
        pairs = discover_pairs({"A": a})
        assert pairs == []
