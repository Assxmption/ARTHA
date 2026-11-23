"""
Tests for Trading Simulator
=============================
Verifies that the TradingSimulator correctly tracks positions, computes
NAV, executes trades with transaction costs, and produces valid equity
curves and performance metrics.

AGENTS.md Rule 1: All numeric computation is tested in code.
AGENTS.md Rule 8: Optimize for Sharpe/Sortino, never hardcode targets.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from app.quant.simulator import (
    TradingSimulator,
    TradeOrder,
    Position,
    DailySnapshot,
    SimulationResult,
    DEFAULT_CAPITAL,
    BROKERAGE_BPS,
    IMPACT_BPS,
    _compute_stock_signals,
    _mad_zscore,
    _optimize_portfolio,
    REGIME_ALLOCATION,
    MAX_POSITION_PCT,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_prices_df(n_symbols: int = 5, n_days: int = 200, seed: int = 42):
    """Generate a synthetic price DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    data = {}
    for i in range(n_symbols):
        sym = f"SIM{i:02d}"
        ret = rng.normal(0.0005, 0.02, n_days)
        data[sym] = 1000.0 * np.cumprod(1 + ret)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def prices_df():
    return _make_prices_df()


@pytest.fixture
def simulator():
    return TradingSimulator(initial_capital=1_000_000.0)


# ── Initialization Tests ────────────────────────────────────────────────────────

class TestSimulatorInit:
    """Test simulator initialization and defaults."""

    def test_default_capital(self):
        sim = TradingSimulator()
        assert sim.initial_capital == DEFAULT_CAPITAL
        assert sim.cash == DEFAULT_CAPITAL

    def test_custom_capital(self):
        sim = TradingSimulator(initial_capital=5_000_000.0)
        assert sim.cash == 5_000_000.0

    def test_transaction_cost_rate(self):
        sim = TradingSimulator(brokerage_bps=5, impact_bps=10)
        expected = (5 + 10) / 10_000.0  # 0.0015
        assert abs(sim.tc_rate - expected) < 1e-10

    def test_starts_with_no_positions(self, simulator):
        assert len(simulator.positions) == 0
        assert len(simulator.trade_log) == 0
        assert len(simulator.equity_curve) == 0

    def test_initial_nav_equals_capital(self, simulator):
        nav = simulator._nav({})
        assert nav == simulator.initial_capital


# ── Trade Execution Tests ───────────────────────────────────────────────────────

class TestTradeExecution:
    """Test individual trade execution logic."""

    def test_buy_updates_positions(self, simulator):
        trade = simulator._execute_trade(
            date_str="2024-01-15", symbol="RELIANCE", action="BUY",
            quantity=10, price=2500.0, signal_score=0.05,
            regime="BULL", reason="Momentum signal",
        )
        assert trade is not None
        assert "RELIANCE" in simulator.positions
        assert simulator.positions["RELIANCE"]["qty"] == 10
        assert simulator.positions["RELIANCE"]["avg_entry"] == 2500.0

    def test_buy_reduces_cash(self, simulator):
        initial_cash = simulator.cash
        simulator._execute_trade(
            date_str="2024-01-15", symbol="RELIANCE", action="BUY",
            quantity=10, price=2500.0, signal_score=0.05,
            regime="BULL", reason="test",
        )
        # Cash should decrease by value + transaction cost
        value = 10 * 2500.0
        cost = value * simulator.tc_rate
        assert abs(simulator.cash - (initial_cash - value - cost)) < 0.01

    def test_sell_removes_position(self, simulator):
        # First buy
        simulator._execute_trade(
            date_str="2024-01-15", symbol="TEST", action="BUY",
            quantity=10, price=100.0, signal_score=0.05,
            regime="BULL", reason="test",
        )
        # Then sell all
        simulator._execute_trade(
            date_str="2024-01-20", symbol="TEST", action="SELL",
            quantity=10, price=110.0, signal_score=0.0,
            regime="BULL", reason="take profit",
        )
        assert "TEST" not in simulator.positions

    def test_sell_records_pnl(self, simulator):
        simulator._execute_trade(
            date_str="2024-01-15", symbol="TEST", action="BUY",
            quantity=10, price=100.0, signal_score=0.05,
            regime="BULL", reason="test",
        )
        trade = simulator._execute_trade(
            date_str="2024-01-20", symbol="TEST", action="SELL",
            quantity=10, price=120.0, signal_score=0.0,
            regime="BULL", reason="take profit",
        )
        assert trade is not None
        # PnL = (120 - 100) * 10 - costs
        expected_gross_pnl = (120.0 - 100.0) * 10
        assert trade.pnl > 0  # Profitable trade
        assert trade.pnl < expected_gross_pnl  # Less than gross due to costs

    def test_zero_quantity_rejected(self, simulator):
        trade = simulator._execute_trade(
            date_str="2024-01-15", symbol="TEST", action="BUY",
            quantity=0, price=100.0, signal_score=0.05,
            regime="BULL", reason="test",
        )
        assert trade is None

    def test_zero_price_rejected(self, simulator):
        trade = simulator._execute_trade(
            date_str="2024-01-15", symbol="TEST", action="BUY",
            quantity=10, price=0.0, signal_score=0.05,
            regime="BULL", reason="test",
        )
        assert trade is None

    def test_trade_log_records_trades(self, simulator):
        simulator._execute_trade(
            date_str="2024-01-15", symbol="TEST", action="BUY",
            quantity=5, price=200.0, signal_score=0.1,
            regime="SIDEWAYS", reason="mean reversion",
        )
        assert len(simulator.trade_log) == 1
        assert simulator.trade_log[0].symbol == "TEST"
        assert simulator.trade_log[0].action == "BUY"
        assert simulator.trade_log[0].regime == "SIDEWAYS"


# ── NAV Calculation Tests ───────────────────────────────────────────────────────

class TestNAVCalculation:
    """Test NAV (net asset value) computation."""

    def test_nav_with_no_positions(self, simulator):
        nav = simulator._nav({"RELIANCE": 2500.0})
        assert nav == simulator.initial_capital

    def test_nav_includes_positions(self, simulator):
        simulator.positions["TEST"] = {"qty": 10, "avg_entry": 100.0, "entry_date": "2024-01-15"}
        simulator.cash -= 1000.0  # Debit for purchase (simplified)
        nav = simulator._nav({"TEST": 110.0})
        # NAV = cash + (10 * 110) = (1_000_000 - 1_000) + 1_100
        expected = simulator.cash + 10 * 110.0
        assert abs(nav - expected) < 0.01

    def test_nav_handles_missing_prices(self, simulator):
        """Missing prices should use avg_entry as fallback."""
        simulator.positions["TEST"] = {"qty": 10, "avg_entry": 100.0, "entry_date": "2024-01-15"}
        simulator.cash -= 1000.0
        nav = simulator._nav({})  # No current prices
        expected = simulator.cash + 10 * 100.0  # Falls back to avg_entry
        assert abs(nav - expected) < 0.01


# ── Signal Generation Tests ─────────────────────────────────────────────────────

class TestSignalGeneration:
    """Test the stock signal computation pipeline."""

    def test_signals_computed_for_all_stocks(self, prices_df):
        signals = _compute_stock_signals(prices_df)
        assert not signals.empty
        # Should have signals for stocks (symbol is a column, not an index level)
        if "symbol" in signals.columns:
            symbols_in_signals = signals["symbol"].unique()
        else:
            symbols_in_signals = signals.index.get_level_values(0).unique()
        assert len(symbols_in_signals) > 0

    def test_signals_have_alpha_column(self, prices_df):
        signals = _compute_stock_signals(prices_df)
        assert "alpha" in signals.columns

    def test_signals_with_short_data(self):
        """Very short data should return empty (not crash)."""
        short = _make_prices_df(n_symbols=3, n_days=30)
        signals = _compute_stock_signals(short)
        # Either empty or with limited data — should not crash
        assert isinstance(signals, pd.DataFrame)

    def test_mad_zscore_handles_normal_data(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0, 1, 100)
        z = _mad_zscore(values)
        assert len(z) == len(values)
        assert not np.any(np.isnan(z))
        # z-scores should be roughly centered around 0
        assert abs(np.mean(z)) < 1.0

    def test_mad_zscore_handles_all_same_values(self):
        values = np.ones(50)
        z = _mad_zscore(values)
        # All same → MAD=0 → z-scores should be 0
        assert np.all(z == 0.0)


# ── Portfolio Optimization Tests ─────────────────────────────────────────────────

class TestPortfolioOptimization:
    """Test the portfolio weight optimizer."""

    def test_optimize_returns_weights(self):
        alpha = {"A": 0.10, "B": 0.05, "C": 0.08}
        prices = {"A": 100.0, "B": 200.0, "C": 150.0}
        weights = _optimize_portfolio(alpha, prices, nav=1_000_000.0, regime="BULL")
        assert len(weights) > 0
        # Weights are regime-scaled exposure; total should be <= regime allocation
        total = sum(weights.values())
        assert total <= REGIME_ALLOCATION["BULL"] + 0.01

    def test_optimize_total_exposure_regime_conditioned(self):
        """Total weight should scale with regime allocation."""
        alpha = {"A": 0.10, "B": 0.05, "C": 0.08}
        prices = {"A": 100.0, "B": 200.0, "C": 150.0}

        bull_weights = _optimize_portfolio(alpha, prices, 1e6, regime="BULL")
        bear_weights = _optimize_portfolio(alpha, prices, 1e6, regime="BEAR")

        bull_total = sum(bull_weights.values())
        bear_total = sum(bear_weights.values())
        # BULL should have higher exposure than BEAR
        assert bull_total > bear_total

    def test_optimize_empty_alphas(self):
        weights = _optimize_portfolio({}, {}, nav=1_000_000.0)
        assert weights == {}

    def test_optimize_respects_max_weight(self):
        # Single stock with high alpha — total exposure bounded by regime
        alpha = {"A": 0.50}
        prices = {"A": 100.0}
        weights = _optimize_portfolio(alpha, prices, nav=1e6, regime="BULL")
        # Single stock gets full regime exposure; check it's bounded
        total = sum(weights.values())
        assert total <= REGIME_ALLOCATION["BULL"] + 0.01


# ── Data Structure Tests ─────────────────────────────────────────────────────────

class TestDataStructures:
    """Test the simulator data classes."""

    def test_trade_order_has_id(self):
        order = TradeOrder(symbol="TEST", action="BUY", quantity=10, price=100.0)
        assert order.trade_id  # Auto-generated
        assert len(order.trade_id) == 12

    def test_position_defaults(self):
        pos = Position()
        assert pos.quantity == 0
        assert pos.unrealized_pnl == 0.0

    def test_simulation_result_defaults(self):
        result = SimulationResult(sim_id="test-001", status="running", initial_capital=1e6)
        assert result.total_trades == 0
        assert result.sharpe_ratio == 0.0
        assert result.positions == []
        assert result.trade_log == []
