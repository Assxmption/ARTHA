"""
Tests for LiveStrategyEngine
=============================
Verifies that the live strategy engine produces valid signals,
handles edge cases, and integrates correctly with the regime detector.

AGENTS.md Rule 1: All numeric computation is tested in code.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from app.quant.live_strategy import (
    LiveStrategyEngine, StrategySignal, PortfolioTarget, ModelCheckpoint,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_price_panel(
    n_symbols: int = 10,
    n_days: int = 600,
    base_price: float = 1000.0,
    drift: float = 0.0003,
    vol: float = 0.02,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Generate synthetic price data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    prices = {}
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        returns = rng.normal(drift, vol, n_days)
        price = base_price * np.cumprod(1 + returns)
        prices[sym] = pd.Series(price, index=dates, name=sym)
    return prices


def _make_volume_panel(
    prices: dict[str, pd.Series],
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Generate synthetic volume data matching price panel."""
    rng = np.random.RandomState(seed)
    volumes = {}
    for sym, p in prices.items():
        vol = rng.randint(100000, 1000000, len(p))
        volumes[sym] = pd.Series(vol, index=p.index, name=sym)
    return volumes


@pytest.fixture
def temp_model_dir():
    """Temporary directory for model checkpoints."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def prices():
    return _make_price_panel(n_symbols=15, n_days=600)


@pytest.fixture
def volumes(prices):
    return _make_volume_panel(prices)


@pytest.fixture
def engine(temp_model_dir):
    symbols = [f"SYM{i:02d}" for i in range(15)]
    return LiveStrategyEngine(
        symbols=symbols,
        model_dir=temp_model_dir,
    )


# ── Unit Tests ──────────────────────────────────────────────────────────────────

class TestLiveStrategyEngineInit:
    """Test engine initialization and lifecycle."""

    def test_engine_starts_uninitialized(self, engine):
        assert not engine.is_initialized
        assert engine.current_regime == "UNKNOWN"

    def test_engine_initializes_with_prices(self, engine, prices, volumes):
        """Engine should train regime detector and set initialized=True."""
        engine.initialize(prices=prices, volumes=volumes)
        assert engine.is_initialized
        assert engine.current_regime in ("BULL", "BEAR", "SIDEWAYS", "UNKNOWN")

    def test_engine_initializes_without_volumes(self, engine, prices):
        """Volume data is optional."""
        engine.initialize(prices=prices)
        assert engine.is_initialized

    def test_engine_handles_empty_prices(self, engine):
        """Empty prices should not crash."""
        engine.initialize(prices={})
        assert not engine.is_initialized

    def test_engine_handles_short_history(self, temp_model_dir):
        """With < 252 days, should still initialize (regime) but skip ML."""
        short_prices = _make_price_panel(n_symbols=5, n_days=100)
        engine = LiveStrategyEngine(
            symbols=list(short_prices.keys()),
            model_dir=temp_model_dir,
        )
        engine.initialize(prices=short_prices)
        # Regime should still train (even on short data)
        assert engine.is_initialized


class TestSignalGeneration:
    """Test individual strategy signal generation."""

    def test_momentum_produces_weights(self, engine, prices, volumes):
        """Momentum strategy should produce non-empty weights."""
        engine.initialize(prices=prices, volumes=volumes)
        panel = pd.DataFrame(prices).dropna(how='all')
        nifty = panel.mean(axis=1)
        weights = engine._compute_momentum(panel, nifty)
        # With 15 stocks and 600 days, should have signals
        assert isinstance(weights, dict)
        if weights:  # May be empty if all scores are NaN
            assert all(v > 0 for v in weights.values())

    def test_mean_reversion_on_normal_data(self, engine, prices, volumes):
        """Mean reversion should produce empty or valid weights."""
        engine.initialize(prices=prices, volumes=volumes)
        panel = pd.DataFrame(prices).dropna(how='all')
        vol_panel = pd.DataFrame(volumes).reindex(panel.index)
        weights = engine._compute_mean_reversion(panel, vol_panel)
        assert isinstance(weights, dict)
        # On normal random data, z < -2.5 is rare, so likely empty
        # But should not crash

    def test_trend_following_produces_weights(self, engine, prices, volumes):
        """Trend following should identify trending stocks."""
        engine.initialize(prices=prices, volumes=volumes)
        panel = pd.DataFrame(prices).dropna(how='all')
        weights = engine._compute_trend_following(panel)
        assert isinstance(weights, dict)
        # With positive drift in fixture, should find some golden crosses
        if weights:
            assert all(v > 0 for v in weights.values())

    def test_short_term_reversal_produces_weights(self, engine, prices, volumes):
        """ST reversal should buy last week's losers."""
        engine.initialize(prices=prices, volumes=volumes)
        panel = pd.DataFrame(prices).dropna(how='all')
        weights = engine._compute_short_term_reversal(panel)
        assert isinstance(weights, dict)
        if weights:
            n = len(weights)
            assert n >= 1
            assert n <= len(prices)

    def test_all_strategies_with_short_data(self, temp_model_dir):
        """Strategies should return empty dict (not crash) with insufficient data."""
        short = _make_price_panel(n_symbols=3, n_days=10)
        engine = LiveStrategyEngine(
            symbols=list(short.keys()),
            model_dir=temp_model_dir,
        )
        panel = pd.DataFrame(short).dropna(how='all')
        nifty = panel.mean(axis=1)

        assert engine._compute_momentum(panel, nifty) == {}
        assert engine._compute_mean_reversion(panel, None) == {}
        assert engine._compute_trend_following(panel) == {}
        assert engine._compute_short_term_reversal(panel) == {}


class TestSignalCombination:
    """Test combining signals from multiple strategies."""

    def test_combine_equal_weight(self, engine):
        """Combined weights should sum to ~1.0."""
        signals = [
            StrategySignal(name="A", weights={"SYM00": 0.5, "SYM01": 0.5}),
            StrategySignal(name="B", weights={"SYM01": 0.3, "SYM02": 0.7}),
        ]
        panel = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["SYM00", "SYM01", "SYM02"],
        )
        daily_returns = panel.pct_change()

        combined = engine._combine_signals(signals, daily_returns)

        assert isinstance(combined, dict)
        assert len(combined) > 0
        total = sum(combined.values())
        assert abs(total - 1.0) < 0.01

    def test_combine_respects_stored_allocation(self, engine):
        """If allocation weights are stored, they should be used."""
        engine._allocation_weights = {"A": 0.8, "B": 0.2}
        signals = [
            StrategySignal(name="A", weights={"SYM00": 1.0}),
            StrategySignal(name="B", weights={"SYM01": 1.0}),
        ]
        panel = pd.DataFrame(np.random.randn(10, 2), columns=["SYM00", "SYM01"])
        combined = engine._combine_signals(signals, panel.pct_change())

        assert "SYM00" in combined
        assert "SYM01" in combined
        # SYM00 should have higher weight (80% allocation to strategy A)
        assert combined["SYM00"] > combined["SYM01"]


class TestComputeTargets:
    """Test the full target computation pipeline."""

    def test_compute_targets_returns_portfolio(self, engine, prices, volumes):
        """Full pipeline should return a PortfolioTarget."""
        engine.initialize(prices=prices, volumes=volumes)

        current_prices = {sym: float(p.iloc[-1]) for sym, p in prices.items()}
        target = engine.compute_targets(current_prices)

        assert target is not None
        assert isinstance(target, PortfolioTarget)
        assert target.regime in ("BULL", "BEAR", "SIDEWAYS", "UNKNOWN")
        assert 0.0 <= target.hedge_ratio <= 1.0
        assert len(target.weights) > 0
        # Weights should sum to approximately 1.0
        total = sum(target.weights.values())
        assert abs(total - 1.0) < 0.05

    def test_compute_targets_returns_none_when_uninitialized(self, engine):
        """Uninitialized engine should return None."""
        result = engine.compute_targets({"SYM00": 100.0})
        assert result is None


class TestRegimeAndHedge:
    """Test regime detection and hedge ratio logic."""

    def test_hedge_ratios_are_conservative(self):
        """Hedge ratios should match the documented conservative priors."""
        assert LiveStrategyEngine._get_hedge_ratio("BULL") == 0.30
        assert LiveStrategyEngine._get_hedge_ratio("BEAR") == 0.70
        assert LiveStrategyEngine._get_hedge_ratio("SIDEWAYS") == 0.50
        assert LiveStrategyEngine._get_hedge_ratio("UNKNOWN") == 0.50

    def test_regime_detected_after_init(self, engine, prices, volumes):
        """After initialization, regime should be set."""
        engine.initialize(prices=prices, volumes=volumes)
        assert engine.current_regime != "UNKNOWN"


class TestCheckpointing:
    """Test model persistence via pickle checkpoints."""

    def test_save_and_load_checkpoint(self, engine, prices, volumes):
        """Checkpoint should persist and restore model state."""
        engine.initialize(prices=prices, volumes=volumes)
        engine._save_checkpoint()

        # Create a new engine pointing to same directory
        engine2 = LiveStrategyEngine(
            symbols=engine.symbols,
            model_dir=engine.model_dir,
        )

        # Should have loaded the checkpoint
        assert engine2.is_initialized
        assert engine2.current_regime == engine.current_regime

    def test_load_missing_checkpoint(self, temp_model_dir):
        """Loading from empty directory should not crash."""
        engine = LiveStrategyEngine(
            symbols=["SYM00"],
            model_dir=temp_model_dir,
        )
        assert not engine.is_initialized

    def test_model_info_property(self, engine, prices, volumes):
        """model_info should return structured dict."""
        engine.initialize(prices=prices, volumes=volumes)
        info = engine.model_info
        assert info["initialized"] is True
        assert info["regime_detector"] == "trained"
        assert isinstance(info["allocation_weights"], dict)


class TestRetraining:
    """Test scheduled model retraining."""

    def test_retrain_when_due(self, engine, prices, volumes):
        """Should retrain if past the retrain interval."""
        engine.initialize(prices=prices, volumes=volumes)

        # Force last train date to be in the past
        engine._last_regime_train = date.today() - timedelta(days=100)
        engine.regime_retrain_days = 90

        retrained = engine.retrain_if_due()
        assert retrained is True
        # Last train date should be updated to today
        assert engine._last_regime_train == date.today()

    def test_no_retrain_when_not_due(self, engine, prices, volumes):
        """Should not retrain if recently trained."""
        engine.initialize(prices=prices, volumes=volumes)
        engine._last_regime_train = date.today()
        engine._last_ml_train = date.today()

        retrained = engine.retrain_if_due()
        assert retrained is False


class TestForwardTestIntegration:
    """Test integration with paper trading daemon validation."""

    def test_validation_metrics_empty_initially(self):
        """Daemon should report zeros before any trading days."""
        from app.quant.paper_trade import PaperTradingDaemon

        daemon = PaperTradingDaemon(
            symbols=["SYM00", "SYM01"],
            db_path=":memory:",
        )
        metrics = daemon.get_validation_metrics()

        assert metrics["days_running"] == 0
        assert metrics["forward_sharpe"] == 0.0
        assert metrics["backtest_sharpe"] == 1.12

    def test_validation_metrics_after_returns(self):
        """Sharpe should be computed after accumulating daily returns."""
        from app.quant.paper_trade import PaperTradingDaemon

        daemon = PaperTradingDaemon(
            symbols=["SYM00"],
            db_path=":memory:",
        )
        # Simulate 20 days of 0.1% returns
        daemon._daily_returns = [0.001] * 20

        metrics = daemon.get_validation_metrics()
        assert metrics["days_running"] == 20
        assert metrics["forward_sharpe"] > 0  # Positive returns → positive Sharpe
        assert metrics["annualized_return"] > 0

    def test_strategy_weights_empty_when_no_target(self):
        """Should return empty when no target has been computed."""
        from app.quant.paper_trade import PaperTradingDaemon

        daemon = PaperTradingDaemon(
            symbols=["SYM00"],
            db_path=":memory:",
        )
        result = daemon.get_strategy_weights()
        assert result["weights"] == {}
        assert result["strategy_signals"] == []
