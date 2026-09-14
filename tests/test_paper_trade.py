"""
Tests for Paper Trading Daemon
================================
Tests the PaperTradingDaemon components:
  - VirtualBroker: trade execution, position management, NAV
  - PaperTradePersistence: SQLite state persistence
  - MarketDataFeed: (mocked) price fetching
  - PaperTradingDaemon: lifecycle, market hours, signal cycles

All tests use mocked data — no real yfinance calls.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.quant.paper_trade import (
    DaemonState,
    MarketDataFeed,
    NAVSnapshot,
    OrderSide,
    PaperTradePersistence,
    PaperTradingDaemon,
    Position,
    VirtualBroker,
    VirtualTrade,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name


@pytest.fixture
def persistence(temp_db):
    """Create a PaperTradePersistence instance."""
    p = PaperTradePersistence(temp_db)
    yield p
    p.close()


@pytest.fixture
def broker(persistence):
    """Create a VirtualBroker with 1 Cr capital."""
    return VirtualBroker(
        initial_capital=1_00_00_000.0,
        persistence=persistence,
        slippage_bps=10,
    )


# ── Persistence Tests ───────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load_positions(self, persistence):
        """Positions should survive save/load cycle."""
        positions = {
            "RELIANCE": Position(
                symbol="RELIANCE", quantity=100, avg_entry_price=2500.0,
                current_price=2600.0, strategy_source="test",
                entry_date="2024-01-01",
            ),
            "TCS": Position(
                symbol="TCS", quantity=50, avg_entry_price=3800.0,
                current_price=3900.0,
            ),
        }
        persistence.save_positions(positions)
        loaded = persistence.load_positions()

        assert len(loaded) == 2
        assert loaded["RELIANCE"].quantity == 100
        assert loaded["TCS"].avg_entry_price == 3800.0

    def test_save_and_get_trades(self, persistence):
        """Trade log should persist."""
        trade = VirtualTrade(
            timestamp="2024-01-15T10:30:00",
            symbol="HDFCBANK",
            side="BUY",
            quantity=200,
            fill_price=1650.50,
            slippage_bps=10,
            strategy_source="momentum",
        )
        persistence.save_trade(trade)
        trades = persistence.get_trades()

        assert len(trades) == 1
        assert trades[0]["symbol"] == "HDFCBANK"
        assert trades[0]["quantity"] == 200

    def test_nav_snapshot(self, persistence):
        """NAV snapshots should persist."""
        snapshot = NAVSnapshot(
            date="2024-01-15", nav=1_05_00_000.0, cash=50_00_000.0,
            holdings_value=55_00_000.0, daily_pnl=25_000.0,
            daily_return_pct=0.24, regime="BULL", hedge_ratio=0.3,
        )
        persistence.save_nav_snapshot(snapshot)
        history = persistence.get_nav_history()

        assert len(history) == 1
        assert history[0]["nav"] == 1_05_00_000.0

    def test_state_save_and_load(self, persistence):
        """Key-value state should persist."""
        persistence.save_state("cash", "10000000")
        assert persistence.load_state("cash") == "10000000"
        assert persistence.load_state("missing", "default") == "default"

    def test_today_trades_filter(self, persistence):
        """get_today_trades should only return today's trades."""
        # Yesterday's trade
        persistence.save_trade(VirtualTrade(
            timestamp="2020-01-01T10:00:00", symbol="OLD",
            side="BUY", quantity=1, fill_price=100, slippage_bps=10,
            strategy_source="",
        ))
        # Today's trade
        today = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        persistence.save_trade(VirtualTrade(
            timestamp=today, symbol="NEW",
            side="BUY", quantity=1, fill_price=200, slippage_bps=10,
            strategy_source="",
        ))
        today_trades = persistence.get_today_trades()
        assert len(today_trades) == 1
        assert today_trades[0]["symbol"] == "NEW"


# ── VirtualBroker Tests ─────────────────────────────────────────────────────────


class TestVirtualBroker:
    def test_initial_state(self, broker):
        """Broker should start with initial capital."""
        assert broker.cash == 1_00_00_000.0
        assert broker.nav == 1_00_00_000.0
        assert len(broker.positions) == 0

    def test_buy_creates_position(self, broker):
        """A BUY should create a position and deduct cash."""
        trade = broker.execute_virtual_trade(
            symbol="RELIANCE", side=OrderSide.BUY,
            quantity=100, market_price=2500.0,
            strategy_source="test",
        )

        assert trade is not None
        assert trade.side == "BUY"
        assert "RELIANCE" in broker.positions
        assert broker.positions["RELIANCE"].quantity == 100
        assert broker.cash < 1_00_00_000.0  # Cash reduced

    def test_buy_applies_slippage(self, broker):
        """BUY fill price should be above market price (adverse slippage)."""
        trade = broker.execute_virtual_trade(
            symbol="TCS", side=OrderSide.BUY,
            quantity=10, market_price=3800.0,
        )

        assert trade.fill_price > 3800.0
        expected = 3800.0 * (1 + 10 / 10000)
        assert abs(trade.fill_price - expected) < 0.01

    def test_sell_reduces_position(self, broker):
        """A SELL should reduce the position and credit cash."""
        broker.execute_virtual_trade(
            "INFY", OrderSide.BUY, 200, 1500.0,
        )
        initial_cash = broker.cash

        trade = broker.execute_virtual_trade(
            "INFY", OrderSide.SELL, 100, 1600.0,
        )

        assert trade is not None
        assert broker.positions["INFY"].quantity == 100
        assert broker.cash > initial_cash

    def test_sell_full_position_removes_it(self, broker):
        """Selling entire position should remove it from holdings."""
        broker.execute_virtual_trade("SBIN", OrderSide.BUY, 50, 600.0)
        broker.execute_virtual_trade("SBIN", OrderSide.SELL, 50, 650.0)

        assert "SBIN" not in broker.positions

    def test_sell_nonexistent_returns_none(self, broker):
        """Selling a symbol we don't hold should return None."""
        result = broker.execute_virtual_trade(
            "NONEXISTENT", OrderSide.SELL, 10, 100.0,
        )
        assert result is None

    def test_insufficient_cash_reduces_quantity(self, broker):
        """If cash is too low, should buy fewer shares."""
        # Try to buy way more than we can afford
        # 100 shares * ₹50,00,000 = ₹50 Cr — more than ₹1 Cr capital
        trade = broker.execute_virtual_trade(
            "EXPENSIVE", OrderSide.BUY, 100, 50_00_000.0,
        )

        # Should have bought fewer shares
        if trade is not None:
            assert trade.quantity < 100

    def test_nav_with_positions(self, broker):
        """NAV should reflect positions + cash."""
        broker.execute_virtual_trade("TCS", OrderSide.BUY, 100, 3800.0)

        # Update price
        broker.update_prices({"TCS": 4000.0})

        # NAV = cash + holdings
        expected_holdings = 100 * 4000.0
        assert broker.nav == pytest.approx(broker.cash + expected_holdings, rel=1e-2)

    def test_total_pnl(self, broker):
        """Total PnL should be NAV - initial capital."""
        broker.execute_virtual_trade("ITC", OrderSide.BUY, 1000, 450.0)
        broker.update_prices({"ITC": 500.0})

        assert broker.total_pnl > 0  # Price went up
        assert broker.total_pnl_pct > 0

    def test_averaging_up(self, broker):
        """Buying more of an existing position should update avg entry."""
        broker.execute_virtual_trade("HDFC", OrderSide.BUY, 100, 1500.0)
        broker.execute_virtual_trade("HDFC", OrderSide.BUY, 100, 1600.0)

        pos = broker.positions["HDFC"]
        assert pos.quantity == 200
        # Average should be between 1500 and 1600 (plus slippage)
        assert 1500 < pos.avg_entry_price < 1620

    def test_portfolio_summary(self, broker):
        """Portfolio summary should return correct structure."""
        broker.execute_virtual_trade("RELIANCE", OrderSide.BUY, 50, 2500.0)
        broker.update_prices({"RELIANCE": 2600.0})

        summary = broker.get_portfolio_summary()
        assert len(summary) == 1
        assert summary[0]["symbol"] == "RELIANCE"
        assert summary[0]["unrealized_pnl"] != 0


# ── Persistence Across Restarts ─────────────────────────────────────────────────


class TestPersistenceAcrossRestart:
    def test_broker_restores_state(self, temp_db):
        """VirtualBroker should restore positions and cash from SQLite."""
        # First broker — make some trades
        p1 = PaperTradePersistence(temp_db)
        b1 = VirtualBroker(1_00_00_000.0, p1)
        b1.execute_virtual_trade("RELIANCE", OrderSide.BUY, 100, 2500.0)
        p1.save_state("cash", str(b1.cash))
        cash_after_trade = b1.cash
        p1.close()

        # Second broker — should restore
        p2 = PaperTradePersistence(temp_db)
        b2 = VirtualBroker(1_00_00_000.0, p2)

        assert "RELIANCE" in b2.positions
        assert b2.positions["RELIANCE"].quantity == 100
        assert b2.cash == pytest.approx(cash_after_trade, rel=1e-4)
        p2.close()


# ── Daemon Tests ────────────────────────────────────────────────────────────────


class TestDaemon:
    def test_initial_state(self, temp_db):
        """Daemon should start in STOPPED state."""
        daemon = PaperTradingDaemon(
            symbols=["RELIANCE", "TCS"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
        )
        assert daemon._state == DaemonState.STOPPED

    def test_start_and_stop(self, temp_db):
        """Daemon should transition through states."""
        daemon = PaperTradingDaemon(
            symbols=["RELIANCE"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
            rebalance_interval_min=1,  # Short for testing
        )
        daemon.start()
        assert daemon._state in (DaemonState.RUNNING, DaemonState.SLEEPING)

        daemon.stop()
        assert daemon._state == DaemonState.STOPPED

    def test_status_report(self, temp_db):
        """Status should report meaningful data."""
        daemon = PaperTradingDaemon(
            symbols=["RELIANCE", "TCS", "INFY"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
        )
        status = daemon.get_status()

        assert status.state == DaemonState.STOPPED
        assert status.virtual_nav == 1_00_00_000.0
        assert status.total_pnl == 0.0
        assert status.n_positions == 0

    def test_market_status_detection(self):
        """Market status should be deterministic."""
        # Just verify the method runs without error
        status = PaperTradingDaemon.get_market_status()
        assert status in (
            "OPEN", "CLOSED", "PRE_MARKET", "CLOSED (Weekend)",
        )

    def test_reset_clears_state(self, temp_db):
        """Reset should clear all positions and trades."""
        daemon = PaperTradingDaemon(
            symbols=["RELIANCE"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
        )

        # Execute a trade
        prices = {"RELIANCE": 2500.0}
        daemon.broker.execute_virtual_trade(
            "RELIANCE", OrderSide.BUY, 50, 2500.0,
        )
        assert len(daemon.broker.positions) > 0

        # Reset
        daemon.reset(new_capital=2_00_00_000.0)

        assert len(daemon.broker.positions) == 0
        assert daemon.broker.cash == 2_00_00_000.0
        assert daemon.broker.initial_capital == 2_00_00_000.0

    @patch("app.quant.paper_trade.MarketDataFeed.fetch_current_prices")
    def test_signal_cycle_generates_trades(self, mock_fetch, temp_db):
        """A signal cycle with prices should generate virtual trades."""
        mock_fetch.return_value = {"RELIANCE": 2500.0, "TCS": 3800.0}

        daemon = PaperTradingDaemon(
            symbols=["RELIANCE", "TCS"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
        )

        # Run one signal cycle manually
        daemon._run_signal_cycle()

        # Should have created positions (equal weight rebalance)
        assert len(daemon.broker.positions) > 0
        assert daemon.broker.cash < 1_00_00_000.0

    @patch("app.quant.paper_trade.MarketDataFeed.fetch_current_prices")
    def test_signal_cycle_skips_on_no_prices(self, mock_fetch, temp_db):
        """Signal cycle should skip gracefully when no prices available."""
        mock_fetch.return_value = {}

        daemon = PaperTradingDaemon(
            symbols=["RELIANCE"],
            initial_capital=1_00_00_000.0,
            db_path=temp_db,
        )

        # Should not crash
        daemon._run_signal_cycle()
        assert len(daemon.broker.positions) == 0


# ── Position Model Tests ────────────────────────────────────────────────────────


class TestPositionModel:
    def test_market_value(self):
        pos = Position("TEST", 100, 150.0, 200.0)
        assert pos.market_value == 20000.0

    def test_unrealized_pnl(self):
        pos = Position("TEST", 100, 150.0, 200.0)
        assert pos.unrealized_pnl == 5000.0

    def test_unrealized_pnl_pct(self):
        pos = Position("TEST", 100, 100.0, 120.0)
        assert pos.unrealized_pnl_pct == pytest.approx(20.0)

    def test_zero_entry_price(self):
        pos = Position("TEST", 100, 0.0, 100.0)
        assert pos.unrealized_pnl_pct == 0.0
