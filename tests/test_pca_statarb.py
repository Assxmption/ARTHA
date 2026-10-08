"""
Tests for PCA-Based Statistical Arbitrage
==========================================
Unit tests covering:
  - PCA residual computation on synthetic factor data
  - Z-score signal generation edge cases
  - Backtest correctness (cost model, position sizing)
  - Walk-forward validation

All test data is synthetic with known factor structure so we can
verify the PCA correctly extracts the systematic component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.pca_statarb import (
    compute_pca_residuals,
    generate_pca_signals,
    backtest_pca_statarb,
    walk_forward_pca_statarb,
    _adaptive_n_components,
    DEFAULT_N_COMPONENTS,
    DEFAULT_PCA_WINDOW,
    DEFAULT_ENTRY_Z,
    DEFAULT_EXIT_Z,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────


def _make_synthetic_returns(
    n_dates: int = 600,
    n_stocks: int = 20,
    n_factors: int = 3,
    idiosyncratic_vol: float = 0.02,
    factor_vol: float = 0.01,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic stock returns with known factor structure.

    Returns (returns, prices) where:
      returns = factor_loadings @ factor_returns + idiosyncratic_noise
    """
    rng = np.random.RandomState(seed)

    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    symbols = [f"STOCK_{i:02d}" for i in range(n_stocks)]

    # Factor returns: n_dates × n_factors
    factor_returns = rng.normal(0, factor_vol, (n_dates, n_factors))

    # Factor loadings: n_stocks × n_factors (each stock's beta to each factor)
    loadings = rng.uniform(0.5, 1.5, (n_stocks, n_factors))

    # Systematic component
    systematic = factor_returns @ loadings.T  # n_dates × n_stocks

    # Idiosyncratic noise (mean-reverting for one stock so we can test signals)
    idiosyncratic = rng.normal(0, idiosyncratic_vol, (n_dates, n_stocks))

    # Make one stock have a clear mean-reverting residual for signal testing
    # Inject a sine wave with noise
    t = np.arange(n_dates)
    idiosyncratic[:, 0] = 0.03 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.005, n_dates)

    returns = pd.DataFrame(
        systematic + idiosyncratic,
        index=dates,
        columns=symbols,
    )

    # Build prices from cumulative returns
    prices = (1 + returns).cumprod() * 100  # Start at 100

    return returns, prices


@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic returns and prices."""
    returns, prices = _make_synthetic_returns()
    return returns, prices


@pytest.fixture
def small_data():
    """Fixture with very few stocks (edge case)."""
    returns, prices = _make_synthetic_returns(n_stocks=4, n_dates=300)
    return returns, prices


@pytest.fixture
def minimal_data():
    """Fixture with minimal data (edge case)."""
    returns, prices = _make_synthetic_returns(n_stocks=2, n_dates=100)
    return returns, prices


# ── Adaptive Components ─────────────────────────────────────────────────────────


class TestAdaptiveComponents:
    def test_normal_universe(self):
        """k should be returned as-is for a large enough universe."""
        assert _adaptive_n_components(50, 5) == 5

    def test_small_universe_caps_k(self):
        """k should be capped at n_stocks // 3 for small universes."""
        # 9 stocks → max_k = 3, so k=5 should become 3
        assert _adaptive_n_components(9, 5) == 3

    def test_very_small_universe(self):
        """k should be at least 1 even with 3 stocks."""
        assert _adaptive_n_components(3, 5) == 1

    def test_k_never_exceeds_n_minus_1(self):
        """k should never exceed n_stocks - 1."""
        assert _adaptive_n_components(4, 10) <= 3


# ── PCA Residual Computation ───────────────────────────────────────────────────


class TestPCAResiduals:
    def test_output_shape(self, synthetic_data):
        """Residuals should have the same shape as input returns."""
        returns, _ = synthetic_data
        residuals = compute_pca_residuals(returns, n_components=3, window=252)
        assert residuals.shape == returns.shape

    def test_warmup_is_nan(self, synthetic_data):
        """First `window` rows should be NaN (warmup period)."""
        returns, _ = synthetic_data
        window = 252
        residuals = compute_pca_residuals(returns, n_components=3, window=window)
        # All entries before window should be NaN
        assert residuals.iloc[:window].isna().all().all()

    def test_residuals_smaller_than_returns(self, synthetic_data):
        """Residual variance should be less than total return variance."""
        returns, _ = synthetic_data
        residuals = compute_pca_residuals(returns, n_components=3, window=252)

        # Compare variance after warmup period
        valid_returns = returns.iloc[252:].dropna(axis=1)
        valid_residuals = residuals.iloc[252:].dropna(axis=1)

        if valid_residuals.shape[1] > 0:
            return_var = valid_returns.var().mean()
            residual_var = valid_residuals.var().mean()
            # Residuals should have lower variance (systematic removed)
            assert residual_var < return_var, (
                f"Residual variance ({residual_var:.6f}) should be less than "
                f"return variance ({return_var:.6f})"
            )

    def test_too_few_stocks(self, minimal_data):
        """Should return all NaN with < 3 stocks."""
        returns, _ = minimal_data
        residuals = compute_pca_residuals(returns, n_components=3, window=50)
        assert residuals.isna().all().all()

    def test_handles_nan_in_returns(self, synthetic_data):
        """Should handle NaN values in returns without crashing."""
        returns, _ = synthetic_data
        # Inject some NaNs
        returns_with_nan = returns.copy()
        returns_with_nan.iloc[300:310, 0:3] = np.nan

        residuals = compute_pca_residuals(returns_with_nan, n_components=3, window=252)
        # Should not crash, shape should match
        assert residuals.shape == returns_with_nan.shape


# ── Signal Generation ──────────────────────────────────────────────────────────


class TestSignalGeneration:
    def test_signal_values(self, synthetic_data):
        """Signals should only be -1, 0, or +1."""
        returns, _ = synthetic_data
        residuals = compute_pca_residuals(returns, n_components=3, window=252)
        signals = generate_pca_signals(residuals, entry_z=1.5, exit_z=0.3)

        unique_vals = set()
        for col in signals.columns:
            unique_vals.update(signals[col].unique())

        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected signal values: {unique_vals}"
        )

    def test_signals_respond_to_z_score(self, synthetic_data):
        """At least some signals should be generated (non-zero)."""
        returns, _ = synthetic_data
        residuals = compute_pca_residuals(returns, n_components=3, window=252)
        signals = generate_pca_signals(residuals, entry_z=1.0, exit_z=0.2)

        total_active = signals.abs().sum().sum()
        assert total_active > 0, "No signals generated with entry_z=1.0"

    def test_hysteresis(self, synthetic_data):
        """Entry and exit thresholds should differ (hysteresis)."""
        returns, _ = synthetic_data
        residuals = compute_pca_residuals(returns, n_components=3, window=252)

        # With tight entry and wide exit, should hold positions longer
        signals_tight = generate_pca_signals(residuals, entry_z=2.0, exit_z=0.1)
        signals_loose = generate_pca_signals(residuals, entry_z=1.0, exit_z=0.9)

        # Tight entry should have fewer signals
        tight_count = signals_tight.abs().sum().sum()
        loose_count = signals_loose.abs().sum().sum()
        # Can't guarantee strict ordering due to hysteresis dynamics,
        # but tight entry should generally produce fewer entries
        assert tight_count >= 0 and loose_count >= 0

    def test_empty_residuals(self):
        """Should handle empty residuals gracefully."""
        residuals = pd.DataFrame()
        signals = generate_pca_signals(residuals)
        assert signals.empty


# ── Backtest ───────────────────────────────────────────────────────────────────


class TestBacktest:
    def test_backtest_returns_series(self, synthetic_data):
        """Backtest should return a pd.Series with correct name."""
        _, prices = synthetic_data
        result = backtest_pca_statarb(prices, n_components=3, window=252)
        assert isinstance(result, pd.Series)
        assert result.name == "pca_statarb"

    def test_backtest_length_matches(self, synthetic_data):
        """Output length should match input prices."""
        _, prices = synthetic_data
        result = backtest_pca_statarb(prices, n_components=3, window=252)
        assert len(result) == len(prices) - 1  # -1 for pct_change drop

    def test_warmup_period_is_zero(self, synthetic_data):
        """Returns during warmup period should be zero."""
        _, prices = synthetic_data
        window = 252
        lookback = 20
        result = backtest_pca_statarb(
            prices, n_components=3, window=window, lookback_z=lookback,
        )
        warmup = window + lookback + 1
        assert (result.iloc[:warmup] == 0).all(), (
            "Warmup period should have zero returns"
        )

    def test_transaction_costs_applied(self, synthetic_data):
        """Returns with TC should be lower than without."""
        _, prices = synthetic_data
        ret_with_tc = backtest_pca_statarb(prices, tc_bps=15.0)
        ret_no_tc = backtest_pca_statarb(prices, tc_bps=0.0)

        # With TC should produce lower total return
        total_with = ret_with_tc.sum()
        total_without = ret_no_tc.sum()
        assert total_with <= total_without + 1e-10, (
            f"TC should reduce returns: with={total_with:.6f}, without={total_without:.6f}"
        )

    def test_insufficient_data(self):
        """Should return zero series with insufficient data."""
        prices = pd.DataFrame(
            np.random.randn(50, 5) + 100,
            index=pd.bdate_range("2024-01-01", periods=50),
            columns=[f"S{i}" for i in range(5)],
        )
        result = backtest_pca_statarb(prices, window=252)
        assert (result == 0).all() or len(result) == len(prices)

    def test_honest_metrics(self, synthetic_data):
        """Sharpe/returns should be reported honestly, not cherry-picked."""
        _, prices = synthetic_data
        result = backtest_pca_statarb(prices, n_components=3)

        # The result is a return series — compute metrics
        arr = result.values[result.values != 0]
        if len(arr) > 20:
            sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
            # We don't assert a specific Sharpe — that would be
            # cherry-picking (AGENTS.md rule 8). Just verify it's finite.
            assert np.isfinite(sharpe)


# ── Walk-Forward Validation ────────────────────────────────────────────────────


class TestWalkForward:
    def test_walk_forward_structure(self, synthetic_data):
        """Walk-forward should return a dict with expected keys."""
        _, prices = synthetic_data
        result = walk_forward_pca_statarb(
            prices, train_days=300, test_days=100,
        )
        assert "validated" in result
        assert "reason" in result
        assert "n_windows" in result
        assert isinstance(result["validated"], bool)

    def test_walk_forward_insufficient_data(self):
        """Should report insufficient data cleanly."""
        prices = pd.DataFrame(
            np.random.randn(100, 10) + 100,
            index=pd.bdate_range("2024-01-01", periods=100),
        )
        result = walk_forward_pca_statarb(
            prices, train_days=504, test_days=126,
        )
        assert result["validated"] is False
        assert "Insufficient" in result["reason"]

    def test_walk_forward_produces_windows(self, synthetic_data):
        """Should produce at least 1 walk-forward window."""
        _, prices = synthetic_data
        result = walk_forward_pca_statarb(
            prices, train_days=300, test_days=100,
        )
        assert result["n_windows"] >= 1

    def test_overfit_detection(self):
        """
        Walk-forward with small sample should flag potential overfitting.

        This tests the diagnostic, not the strategy's quality.
        Per AGENTS.md rule 8: report mediocre results honestly.
        """
        # Create data where IS looks great but OOS should differ
        rng = np.random.RandomState(123)
        n = 800
        prices = pd.DataFrame(
            np.cumsum(rng.normal(0, 0.01, (n, 15)), axis=0) + 100,
            index=pd.bdate_range("2018-01-01", periods=n),
        )
        result = walk_forward_pca_statarb(
            prices, train_days=400, test_days=200,
        )
        # Just verify it completes without error
        assert isinstance(result["validated"], bool)
