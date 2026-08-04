"""
Test — Walk-Forward Backtester
================================
Unit tests for the validation gate.

Covers:
  - Metrics computed correctly on known data
  - Sharpe ratio sanity (positive for positive returns)
  - Max drawdown computed correctly
  - Validation gate enforces thresholds
  - Transaction cost model reduces returns
  - Walk-forward produces IS and OOS metrics
  - Edge cases: empty data, single-day
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.quant.backtest import (
    BacktestMetrics,
    WalkForwardResult,
    compute_metrics,
    apply_transaction_costs,
    walk_forward_backtest,
    MIN_OOS_SHARPE,
    MAX_OOS_DRAWDOWN,
    MIN_OOS_TRADING_DAYS,
)


def _make_returns(
    n: int = 500,
    daily_mean: float = 0.0005,
    daily_std: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.RandomState(seed)
    returns = daily_mean + daily_std * rng.randn(n)
    dates = pd.bdate_range(start="2020-01-01", periods=n)
    return pd.Series(returns, index=dates, name="returns")


class TestMetricsComputation:
    """Test core metrics computation."""

    def test_positive_returns_positive_sharpe(self):
        """Positive-mean returns should produce a positive Sharpe ratio."""
        returns = _make_returns(daily_mean=0.001, daily_std=0.01)
        metrics = compute_metrics(returns)
        assert metrics.sharpe_ratio > 0

    def test_negative_returns_negative_sharpe(self):
        """Negative-mean returns should produce a negative Sharpe ratio."""
        returns = _make_returns(daily_mean=-0.001, daily_std=0.01)
        metrics = compute_metrics(returns)
        assert metrics.sharpe_ratio < 0

    def test_total_return_positive_for_up_series(self):
        returns = _make_returns(daily_mean=0.001, n=252)
        metrics = compute_metrics(returns)
        assert metrics.total_return > 0

    def test_max_drawdown_nonnegative(self):
        """Max drawdown should always be ≥ 0."""
        returns = _make_returns()
        metrics = compute_metrics(returns)
        assert metrics.max_drawdown >= 0

    def test_max_drawdown_known_value(self):
        """Construct a series with a known 10% drawdown."""
        # Up 20%, then down 10%
        returns = pd.Series(
            [0.01] * 20 + [-0.01] * 10,
            index=pd.bdate_range("2020-01-01", periods=30),
        )
        metrics = compute_metrics(returns)
        # The drawdown should be approximately 10%
        assert 0.05 <= metrics.max_drawdown <= 0.15

    def test_win_rate_correct(self):
        """Win rate should be the fraction of positive returns."""
        returns = pd.Series(
            [0.01, -0.005, 0.02, -0.01, 0.005],
            index=pd.bdate_range("2020-01-01", periods=5),
        )
        metrics = compute_metrics(returns)
        assert metrics.win_rate == 0.6  # 3 out of 5 positive

    def test_sortino_handles_no_downside(self):
        """All-positive returns should not crash Sortino computation."""
        returns = pd.Series(
            [0.01] * 50,
            index=pd.bdate_range("2020-01-01", periods=50),
        )
        metrics = compute_metrics(returns)
        # Sortino with no downside returns >= 0
        assert metrics.sortino_ratio >= 0

    def test_empty_returns(self):
        returns = pd.Series(dtype=float)
        metrics = compute_metrics(returns)
        assert metrics.total_return == 0
        assert metrics.sharpe_ratio == 0
        assert metrics.num_trading_days == 0

    def test_single_return(self):
        returns = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
        metrics = compute_metrics(returns)
        assert metrics.num_trading_days == 1


class TestValidationGate:
    """Test the hard validation gate (AGENTS.md rule 6)."""

    def test_good_strategy_validates(self):
        """A strategy with good metrics should pass validation."""
        metrics = BacktestMetrics(
            total_return=0.5,
            annualized_return=0.2,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            calmar_ratio=1.3,
            win_rate=0.55,
            num_trades=100,
            num_trading_days=600,
            volatility=0.15,
            avg_trade_return=0.001,
            profit_factor=1.5,
        )
        assert metrics.validate() is True
        assert metrics.is_valid is True

    def test_low_sharpe_fails(self):
        """Sharpe < 1.0 should fail validation."""
        metrics = BacktestMetrics(
            total_return=0.1,
            annualized_return=0.05,
            sharpe_ratio=0.8,  # Below threshold
            sortino_ratio=1.0,
            max_drawdown=0.10,
            calmar_ratio=0.5,
            win_rate=0.50,
            num_trades=100,
            num_trading_days=600,
            volatility=0.10,
            avg_trade_return=0.0005,
            profit_factor=1.2,
        )
        assert metrics.validate() is False

    def test_high_drawdown_fails(self):
        """Drawdown > 25% should fail validation."""
        metrics = BacktestMetrics(
            total_return=0.5,
            annualized_return=0.2,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.30,  # Above threshold
            calmar_ratio=0.7,
            win_rate=0.55,
            num_trades=100,
            num_trading_days=600,
            volatility=0.15,
            avg_trade_return=0.001,
            profit_factor=1.5,
        )
        assert metrics.validate() is False

    def test_insufficient_oos_days_fails(self):
        """Less than 504 OOS days should fail validation."""
        metrics = BacktestMetrics(
            total_return=0.3,
            annualized_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.10,
            calmar_ratio=1.5,
            win_rate=0.55,
            num_trades=50,
            num_trading_days=300,  # Below threshold
            volatility=0.10,
            avg_trade_return=0.001,
            profit_factor=1.5,
        )
        assert metrics.validate() is False

    def test_thresholds_match_constants(self):
        """Verify the validation thresholds are what we expect."""
        assert MIN_OOS_SHARPE == 1.0
        assert MAX_OOS_DRAWDOWN == 0.25
        assert MIN_OOS_TRADING_DAYS == 504


class TestTransactionCosts:
    """Test transaction cost model."""

    def test_costs_reduce_returns(self):
        returns = _make_returns(n=100, daily_mean=0.001)
        signals = pd.Series(1.0, index=returns.index)  # Always long
        # No position changes after initial entry → minimal cost
        net_returns = apply_transaction_costs(returns, signals)
        # First entry incurs a cost
        assert net_returns.iloc[0] < returns.iloc[0]

    def test_frequent_trading_more_costly(self):
        """More position changes should incur more costs."""
        returns = _make_returns(n=100, daily_mean=0.001)

        # Buy-and-hold: 1 trade
        signals_bah = pd.Series(1.0, index=returns.index)

        # Frequent flipping: many trades
        signals_flip = pd.Series(
            [1.0 if i % 2 == 0 else -1.0 for i in range(100)],
            index=returns.index,
        )

        net_bah = apply_transaction_costs(returns, signals_bah)
        net_flip = apply_transaction_costs(returns, signals_flip)

        # Frequent trading should have lower net return
        assert net_flip.sum() < net_bah.sum()


class TestWalkForward:
    """Test walk-forward backtesting."""

    def test_walk_forward_produces_results(self):
        returns = _make_returns(n=600)

        def dummy_signal_gen(train_data: pd.Series) -> pd.Series:
            """Always-long strategy for testing."""
            return pd.Series(1.0, index=train_data.index)

        result = walk_forward_backtest(
            returns, dummy_signal_gen, "test_strategy", n_folds=3,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.strategy_name == "test_strategy"
        assert result.out_of_sample_metrics.num_trading_days > 0

    def test_walk_forward_insufficient_data(self):
        returns = _make_returns(n=50)

        def dummy_signal_gen(train_data):
            return pd.Series(1.0, index=train_data.index)

        result = walk_forward_backtest(returns, dummy_signal_gen)
        assert result.out_of_sample_metrics.num_trading_days == 0

    def test_walk_forward_validation(self):
        """WalkForwardResult.validate() should delegate to OOS metrics."""
        result = WalkForwardResult(
            strategy_name="test",
            in_sample_metrics=BacktestMetrics(
                total_return=0, annualized_return=0, sharpe_ratio=2.0,
                sortino_ratio=2.0, max_drawdown=0.05, calmar_ratio=2.0,
                win_rate=0.6, num_trades=100, num_trading_days=600,
                volatility=0.1, avg_trade_return=0.001, profit_factor=2.0,
            ),
            out_of_sample_metrics=BacktestMetrics(
                total_return=0.3, annualized_return=0.15, sharpe_ratio=1.2,
                sortino_ratio=1.5, max_drawdown=0.10, calmar_ratio=1.5,
                win_rate=0.55, num_trades=80, num_trading_days=600,
                volatility=0.12, avg_trade_return=0.0008, profit_factor=1.6,
            ),
        )
        assert result.validate() is True
        assert result.is_validated is True
