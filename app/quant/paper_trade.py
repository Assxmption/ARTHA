"""
Paper Trading Daemon — Virtual Live Execution
===============================================
Runs the validated multi-strategy engine on real-time (or 15-min delayed)
market data during NSE market hours.  All executions are VIRTUAL — no
broker API calls, no live order placement.

AGENTS.md Rule 9 compliance:
  - All executions are synthetic fills against a virtual balance.
  - No broker API import or call path exists in this module.
  - No path to live execution without explicit human gate.
  - Adding a broker path requires modifying this module AND adding a
    human confirmation step — see docs/ARTHA_ARCHITECTURE.md §6.6.

AGENTS.md Rule 11 compliance:
  - This is a research/validation tool, not investment advice.
  - Reports describe simulated performance of validated signals.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time as time_mod
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────────

NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
IST_UTC_OFFSET = timedelta(hours=5, minutes=30)

# Slippage model: 10bps per trade (5bps spread + 5bps impact)
# Matches the backtest cost model for consistency.
SLIPPAGE_BPS = 10


# ── Data Models ─────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class DaemonState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"  # Outside market hours
    ERROR = "ERROR"


@dataclass
class Position:
    """A virtual holding in the paper trading portfolio."""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    strategy_source: str = ""  # Which strategy triggered this
    entry_date: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price / self.avg_entry_price - 1) * 100


@dataclass
class VirtualTrade:
    """A synthetic execution record."""
    timestamp: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    fill_price: float
    slippage_bps: float
    strategy_source: str
    pnl: float = 0.0  # Realized P&L for closing trades

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NAVSnapshot:
    """Daily NAV record for performance tracking."""
    date: str
    nav: float
    cash: float
    holdings_value: float
    daily_pnl: float
    daily_return_pct: float
    regime: str
    hedge_ratio: float


@dataclass
class DaemonStatus:
    """Current state of the paper trading daemon."""
    state: DaemonState = DaemonState.STOPPED
    started_at: Optional[str] = None
    last_signal_time: Optional[str] = None
    last_error: Optional[str] = None
    current_regime: str = "UNKNOWN"
    hedge_ratio: float = 0.0
    virtual_nav: float = 0.0
    today_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    n_positions: int = 0
    n_trades_today: int = 0
    market_status: str = "CLOSED"  # OPEN, CLOSED, PRE_MARKET


# ── Persistence Layer ───────────────────────────────────────────────────────────

class PaperTradePersistence:
    """
    SQLite-backed persistence for the paper trading daemon.

    Survives server restarts — the daemon can resume from the last
    persisted state without losing position or trade history.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL DEFAULT 0.0,
                strategy_source TEXT DEFAULT '',
                entry_date TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                slippage_bps REAL DEFAULT 0.0,
                strategy_source TEXT DEFAULT '',
                pnl REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS nav_history (
                date TEXT PRIMARY KEY,
                nav REAL NOT NULL,
                cash REAL NOT NULL,
                holdings_value REAL NOT NULL,
                daily_pnl REAL DEFAULT 0.0,
                daily_return_pct REAL DEFAULT 0.0,
                regime TEXT DEFAULT 'UNKNOWN',
                hedge_ratio REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS daemon_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def save_positions(self, positions: dict[str, Position]):
        """Persist all positions (replace-all strategy for simplicity)."""
        self._conn.execute("DELETE FROM positions")
        for pos in positions.values():
            self._conn.execute(
                "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?)",
                (pos.symbol, pos.quantity, pos.avg_entry_price,
                 pos.current_price, pos.strategy_source, pos.entry_date),
            )
        self._conn.commit()

    def load_positions(self) -> dict[str, Position]:
        """Load positions from persistence."""
        cursor = self._conn.execute("SELECT * FROM positions")
        positions = {}
        for row in cursor:
            pos = Position(
                symbol=row[0], quantity=row[1], avg_entry_price=row[2],
                current_price=row[3], strategy_source=row[4], entry_date=row[5],
            )
            positions[pos.symbol] = pos
        return positions

    def save_trade(self, trade: VirtualTrade):
        """Append a trade to the log."""
        self._conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, quantity, "
            "fill_price, slippage_bps, strategy_source, pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trade.timestamp, trade.symbol, trade.side, trade.quantity,
             trade.fill_price, trade.slippage_bps, trade.strategy_source,
             trade.pnl),
        )
        self._conn.commit()

    def get_trades(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get recent trades (most recent first)."""
        cursor = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor]

    def get_today_trades(self) -> list[dict]:
        """Get trades from today."""
        today = date.today().isoformat()
        cursor = self._conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id DESC",
            (today,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor]

    def save_nav_snapshot(self, snapshot: NAVSnapshot):
        """Upsert daily NAV snapshot."""
        self._conn.execute(
            "INSERT OR REPLACE INTO nav_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (snapshot.date, snapshot.nav, snapshot.cash,
             snapshot.holdings_value, snapshot.daily_pnl,
             snapshot.daily_return_pct, snapshot.regime, snapshot.hedge_ratio),
        )
        self._conn.commit()

    def get_nav_history(self, limit: int = 365) -> list[dict]:
        """Get NAV history (most recent first)."""
        cursor = self._conn.execute(
            "SELECT * FROM nav_history ORDER BY date DESC LIMIT ?", (limit,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor]

    def save_state(self, key: str, value: str):
        """Save a daemon state variable."""
        self._conn.execute(
            "INSERT OR REPLACE INTO daemon_state VALUES (?, ?)", (key, value),
        )
        self._conn.commit()

    def load_state(self, key: str, default: str = "") -> str:
        """Load a daemon state variable."""
        cursor = self._conn.execute(
            "SELECT value FROM daemon_state WHERE key = ?", (key,),
        )
        row = cursor.fetchone()
        return row[0] if row else default

    def close(self):
        self._conn.close()


# ── Market Data Feed ────────────────────────────────────────────────────────────

class MarketDataFeed:
    """
    Fetches live (or 15-min delayed) market data.

    Uses yfinance for initial implementation.  The interface is
    intentionally simple so it can be replaced with a Kite/broker
    feed later without changing the daemon logic.
    """

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self._cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None

    def fetch_current_prices(self) -> dict[str, float]:
        """
        Get the latest available price for each symbol.

        Returns {symbol: last_price}. Uses batch download to minimize
        API calls (one call for all symbols).
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — pip install yfinance")
            return {}

        # Batch download — single API call for all symbols
        # Append .NS suffix for NSE symbols
        yf_symbols = [f"{s}.NS" for s in self.symbols]
        try:
            data = yf.download(
                " ".join(yf_symbols),
                period="1d",
                interval="15m",
                progress=False,
                threads=True,
            )
            prices = {}
            for sym, yf_sym in zip(self.symbols, yf_symbols):
                try:
                    if len(self.symbols) == 1:
                        close = data["Close"]
                    else:
                        close = data["Close"][yf_sym]
                    last = close.dropna().iloc[-1]
                    prices[sym] = float(last)
                except (KeyError, IndexError):
                    logger.warning("No price data for %s", sym)
            self._last_fetch = datetime.now()
            return prices
        except Exception as e:
            logger.error("Failed to fetch prices: %s", e)
            return {}

    def fetch_historical(
        self, period: str = "2y", interval: str = "1d",
    ) -> dict[str, pd.Series]:
        """
        Fetch historical close prices for strategy signals.

        Returns {symbol: pd.Series of daily close prices}.
        """
        try:
            import yfinance as yf
        except ImportError:
            return {}

        yf_symbols = [f"{s}.NS" for s in self.symbols]
        try:
            data = yf.download(
                " ".join(yf_symbols),
                period=period,
                interval=interval,
                progress=False,
                threads=True,
            )
            result = {}
            for sym, yf_sym in zip(self.symbols, yf_symbols):
                try:
                    if len(self.symbols) == 1:
                        close = data["Close"].dropna()
                    else:
                        close = data["Close"][yf_sym].dropna()
                    result[sym] = close
                except (KeyError, IndexError):
                    pass
            return result
        except Exception as e:
            logger.error("Failed to fetch historical data: %s", e)
            return {}


# ── Virtual Broker ──────────────────────────────────────────────────────────────

class VirtualBroker:
    """
    Simulates trade execution with slippage.

    NO real broker API is called.  All fills are synthetic.
    """

    def __init__(
        self,
        initial_capital: float,
        persistence: PaperTradePersistence,
        slippage_bps: float = SLIPPAGE_BPS,
    ):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: dict[str, Position] = {}
        self.persistence = persistence
        self.slippage_bps = slippage_bps

        # Try to restore state from persistence
        saved_cash = persistence.load_state("cash")
        if saved_cash:
            self.cash = float(saved_cash)
            self.initial_capital = float(
                persistence.load_state("initial_capital", str(initial_capital))
            )
            self.positions = persistence.load_positions()
            logger.info(
                "Restored paper trading state: cash=%.0f, %d positions",
                self.cash, len(self.positions),
            )

    def execute_virtual_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        market_price: float,
        strategy_source: str = "",
    ) -> Optional[VirtualTrade]:
        """
        Execute a virtual trade with slippage.

        Returns the trade record, or None if the trade couldn't execute
        (e.g., insufficient cash for a buy).
        """
        if quantity <= 0 or market_price <= 0:
            return None

        # Apply slippage
        slippage_mult = 1 + (self.slippage_bps / 10000)
        if side == OrderSide.BUY:
            fill_price = market_price * slippage_mult
        else:
            fill_price = market_price / slippage_mult

        trade_value = fill_price * quantity
        realized_pnl = 0.0

        if side == OrderSide.BUY:
            if trade_value > self.cash:
                # Reduce quantity to fit available cash
                quantity = int(self.cash / fill_price)
                if quantity <= 0:
                    logger.warning(
                        "Insufficient cash for %s BUY: need %.0f, have %.0f",
                        symbol, trade_value, self.cash,
                    )
                    return None
                trade_value = fill_price * quantity

            self.cash -= trade_value

            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.quantity + quantity
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + fill_price * quantity)
                    / total_qty
                )
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=market_price,
                    strategy_source=strategy_source,
                    entry_date=date.today().isoformat(),
                )

        elif side == OrderSide.SELL:
            if symbol not in self.positions:
                logger.warning("No position to sell for %s", symbol)
                return None
            pos = self.positions[symbol]
            sell_qty = min(quantity, pos.quantity)
            realized_pnl = sell_qty * (fill_price - pos.avg_entry_price)
            self.cash += fill_price * sell_qty
            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                del self.positions[symbol]

        now = datetime.now().isoformat()
        trade = VirtualTrade(
            timestamp=now,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            fill_price=round(fill_price, 2),
            slippage_bps=self.slippage_bps,
            strategy_source=strategy_source,
            pnl=round(realized_pnl, 2),
        )

        # Persist
        self.persistence.save_trade(trade)
        self.persistence.save_positions(self.positions)
        self.persistence.save_state("cash", str(self.cash))

        logger.info(
            "Virtual %s: %s %d @ %.2f (slippage=%.0fbps, PnL=%.2f)",
            side.value, symbol, quantity, fill_price,
            self.slippage_bps, realized_pnl,
        )
        return trade

    def update_prices(self, prices: dict[str, float]):
        """Update current prices on all positions."""
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price
        self.persistence.save_positions(self.positions)

    @property
    def nav(self) -> float:
        """Current Net Asset Value."""
        holdings = sum(p.market_value for p in self.positions.values())
        return self.cash + holdings

    @property
    def total_pnl(self) -> float:
        return self.nav - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.nav / self.initial_capital - 1) * 100

    def get_portfolio_summary(self) -> list[dict]:
        """Get a summary of all positions for API responses."""
        result = []
        for pos in self.positions.values():
            result.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": round(pos.avg_entry_price, 2),
                "current_price": round(pos.current_price, 2),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pos.unrealized_pnl_pct, 2),
                "strategy_source": pos.strategy_source,
                "entry_date": pos.entry_date,
            })
        return result


# ── The Daemon ──────────────────────────────────────────────────────────────────

class PaperTradingDaemon:
    """
    Live paper trading daemon.

    Runs in a background thread during NSE market hours, executing the
    same multi-strategy engine validated in run_multi_strategy_backtest.py.

    All executions are VIRTUAL.  No broker API calls.
    See module docstring for AGENTS.md compliance.
    """

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = 5_00_00_000.0,  # ₹5 Cr
        db_path: str | Path = "data_cache/paper_trades.db",
        rebalance_interval_min: int = 15,
        weight_rebalance_days: int = 63,
    ):
        self.symbols = symbols
        self.db_path = Path(db_path)

        # Components
        self.persistence = PaperTradePersistence(self.db_path)
        self.feed = MarketDataFeed(symbols)
        self.broker = VirtualBroker(initial_capital, self.persistence)

        # Schedule
        self.rebalance_interval = rebalance_interval_min
        self.weight_rebalance_days = weight_rebalance_days

        # State
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state = DaemonState.STOPPED
        self._started_at: Optional[datetime] = None
        self._last_signal_time: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._current_regime = "UNKNOWN"
        self._hedge_ratio = 0.0
        self._today_start_nav = self.broker.nav
        self._lock = threading.Lock()

    @staticmethod
    def is_market_open() -> bool:
        """Check if NSE is currently in trading hours (9:15-15:30 IST)."""
        now_utc = datetime.now(tz=__import__('datetime').timezone.utc)
        now_ist = now_utc + IST_UTC_OFFSET
        current_time = now_ist.time()
        weekday = now_ist.weekday()

        # Mon-Fri only
        if weekday >= 5:
            return False
        return NSE_OPEN <= current_time <= NSE_CLOSE

    @staticmethod
    def get_market_status() -> str:
        """Get human-readable market status."""
        now_utc = datetime.now(tz=__import__('datetime').timezone.utc)
        now_ist = now_utc + IST_UTC_OFFSET
        current_time = now_ist.time()
        weekday = now_ist.weekday()

        if weekday >= 5:
            return "CLOSED (Weekend)"
        if current_time < time(9, 0):
            return "CLOSED"
        if current_time < NSE_OPEN:
            return "PRE_MARKET"
        if current_time <= NSE_CLOSE:
            return "OPEN"
        return "CLOSED"

    def start(self):
        """Start the daemon in a background thread."""
        if self._state == DaemonState.RUNNING:
            logger.warning("Daemon is already running")
            return

        self._stop_event.clear()
        self._state = DaemonState.RUNNING
        self._started_at = datetime.now()
        self._today_start_nav = self.broker.nav
        self.persistence.save_state("initial_capital", str(self.broker.initial_capital))

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="PaperTradingDaemon",
        )
        self._thread.start()
        logger.info("Paper trading daemon started (capital=%.0f)", self.broker.nav)

    def stop(self):
        """Stop the daemon gracefully."""
        self._stop_event.set()
        self._state = DaemonState.STOPPED
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        # Save end-of-day NAV
        self._save_nav_snapshot()
        logger.info("Paper trading daemon stopped")

    def _run_loop(self):
        """Main daemon loop."""
        logger.info("Daemon loop started — polling every %d min", self.rebalance_interval)

        while not self._stop_event.is_set():
            try:
                if self.is_market_open():
                    self._state = DaemonState.RUNNING
                    self._run_signal_cycle()
                else:
                    self._state = DaemonState.SLEEPING
                    # Save end-of-day snapshot if transitioning to closed
                    if self._last_signal_time:
                        self._save_nav_snapshot()

                # Sleep until next check
                self._stop_event.wait(timeout=self.rebalance_interval * 60)

            except Exception as e:
                self._last_error = str(e)
                self._state = DaemonState.ERROR
                logger.error("Daemon error: %s", e, exc_info=True)
                # Back off on error
                self._stop_event.wait(timeout=300)  # 5 min on error

        self._state = DaemonState.STOPPED

    def _run_signal_cycle(self):
        """
        One cycle of the signal engine:
        1. Fetch latest prices
        2. Update positions
        3. Generate target weights
        4. Execute rebalancing trades
        """
        logger.info("Running signal cycle at %s", datetime.now().isoformat())

        # 1. Fetch current prices
        prices = self.feed.fetch_current_prices()
        if not prices:
            logger.warning("No prices available — skipping cycle")
            return

        # 2. Update positions with current prices
        with self._lock:
            self.broker.update_prices(prices)

        # 3. Compute target portfolio (using historical data + current regime)
        target_weights = self._compute_target_weights()
        if target_weights is None:
            logger.warning("Could not compute target weights — skipping cycle")
            return

        # 4. Execute rebalancing trades
        self._rebalance_to_targets(target_weights, prices)

        self._last_signal_time = datetime.now()
        logger.info(
            "Signal cycle complete — NAV=%.0f, PnL=%.0f (%.2f%%)",
            self.broker.nav, self.broker.total_pnl, self.broker.total_pnl_pct,
        )

    def _compute_target_weights(self) -> Optional[dict[str, float]]:
        """
        Compute target portfolio weights using the same strategy engine
        as the backtester.

        Returns {symbol: target_weight} or None if computation fails.

        Note: This is a simplified version that uses equal-weight as the
        default.  The full strategy engine integration (running each
        sub-strategy on live data) will be phased in — see implementation plan.
        """
        # For initial deployment: equal-weight across all symbols
        # TODO: Phase 2 — integrate full sub-strategy signal generation
        n = len(self.symbols)
        if n == 0:
            return None

        # Equal weight as conservative starting point
        weights = {sym: 1.0 / n for sym in self.symbols}
        return weights

    def _rebalance_to_targets(
        self, target_weights: dict[str, float], prices: dict[str, float],
    ):
        """
        Diff target weights vs current holdings and execute virtual trades.
        """
        nav = self.broker.nav
        if nav <= 0:
            return

        # Compute target position sizes
        target_values = {
            sym: nav * weight for sym, weight in target_weights.items()
        }

        with self._lock:
            # Current position values
            current_values = {
                sym: pos.market_value
                for sym, pos in self.broker.positions.items()
            }

            # Compute trades needed
            for sym in target_weights:
                if sym not in prices:
                    continue

                price = prices[sym]
                if price <= 0:
                    continue

                target_val = target_values.get(sym, 0)
                current_val = current_values.get(sym, 0)
                diff = target_val - current_val

                # Only trade if diff > 1% of NAV (avoid churning)
                if abs(diff) < nav * 0.01:
                    continue

                qty = int(abs(diff) / price)
                if qty <= 0:
                    continue

                side = OrderSide.BUY if diff > 0 else OrderSide.SELL
                self.broker.execute_virtual_trade(
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    market_price=price,
                    strategy_source="multi_strategy_rebalance",
                )

    def _save_nav_snapshot(self):
        """Save a daily NAV snapshot."""
        nav = self.broker.nav
        today_pnl = nav - self._today_start_nav
        today_return = (
            (today_pnl / self._today_start_nav * 100)
            if self._today_start_nav > 0 else 0
        )

        snapshot = NAVSnapshot(
            date=date.today().isoformat(),
            nav=round(nav, 2),
            cash=round(self.broker.cash, 2),
            holdings_value=round(
                sum(p.market_value for p in self.broker.positions.values()), 2,
            ),
            daily_pnl=round(today_pnl, 2),
            daily_return_pct=round(today_return, 4),
            regime=self._current_regime,
            hedge_ratio=self._hedge_ratio,
        )
        self.persistence.save_nav_snapshot(snapshot)

    def get_status(self) -> DaemonStatus:
        """Get current daemon status for API responses."""
        today_trades = self.persistence.get_today_trades()
        return DaemonStatus(
            state=self._state,
            started_at=self._started_at.isoformat() if self._started_at else None,
            last_signal_time=(
                self._last_signal_time.isoformat() if self._last_signal_time else None
            ),
            last_error=self._last_error,
            current_regime=self._current_regime,
            hedge_ratio=self._hedge_ratio,
            virtual_nav=round(self.broker.nav, 2),
            today_pnl=round(self.broker.nav - self._today_start_nav, 2),
            total_pnl=round(self.broker.total_pnl, 2),
            total_pnl_pct=round(self.broker.total_pnl_pct, 2),
            n_positions=len(self.broker.positions),
            n_trades_today=len(today_trades),
            market_status=self.get_market_status(),
        )

    def get_portfolio(self) -> dict:
        """Get portfolio and recent trades for API responses."""
        return {
            "positions": self.broker.get_portfolio_summary(),
            "cash": round(self.broker.cash, 2),
            "nav": round(self.broker.nav, 2),
            "initial_capital": round(self.broker.initial_capital, 2),
        }

    def get_trade_log(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get trade history."""
        return self.persistence.get_trades(limit, offset)

    def get_nav_history(self, limit: int = 365) -> list[dict]:
        """Get NAV history."""
        return self.persistence.get_nav_history(limit)

    def reset(self, new_capital: float | None = None):
        """Reset the paper trading state (nuclear option)."""
        self.stop()
        capital = new_capital or self.broker.initial_capital
        # Clear database
        self.persistence._conn.executescript("""
            DELETE FROM positions;
            DELETE FROM trades;
            DELETE FROM nav_history;
            DELETE FROM daemon_state;
        """)
        self.broker = VirtualBroker(capital, self.persistence)
        self.persistence.save_state("initial_capital", str(capital))
        self.persistence.save_state("cash", str(capital))
        self._today_start_nav = capital
        logger.info("Paper trading state reset (capital=%.0f)", capital)
