"""
ARTHA Trading Simulator — Automated Portfolio Worker
=====================================================
The core automated trading engine that:
  1. Takes initial capital (default ₹15L)
  2. Runs the full signal pipeline daily
  3. Generates BUY/SELL orders with quantity + reason
  4. Tracks open positions with mark-to-market P&L
  5. Maintains a complete trade log
  6. Produces an equity curve for charting

Modes:
  - Historical Replay: Feed past OHLCV day-by-day (free)
  - Shadow Mode: Run daily against latest market data (no real orders)

This is NOT part of the fundamental analysis pipeline.
The quant engine and fundamental analysis are independent systems.

Reference: docs/ARTHA_ARCHITECTURE.md §4.3, §6.2
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("artha.simulator")

# ── Default Config ──────────────────────────────────────────────────────────────

DEFAULT_CAPITAL = 15_00_000.0  # ₹15 Lakh
BROKERAGE_BPS = 5              # 0.05% per trade
IMPACT_BPS = 10                # 0.10% estimated market impact
MAX_POSITION_PCT = 0.15        # 15% max per stock (higher conviction)
MIN_POSITION_PCT = 0.005       # 0.5% min (avoid dust)
REBALANCE_INTERVAL = 2         # days between rebalances (fast trader)
MAX_POSITIONS = 12             # max simultaneous positions (BULL)
MAX_POSITIONS_DEFENSIVE = 7    # fewer positions in BEAR/SIDEWAYS for concentration
MIN_HOLDING_DAYS = 3           # shorter hold for faster rotation
TRAILING_STOP_PCT = 0.10       # 10% trailing stop from peak
MIN_ALPHA_THRESHOLD = 0.03     # minimum alpha to enter a position

# ── Regime-based allocation ─────────────────────────────────────────────────────
# Key insight from architecture §6.3: thin edges compounded.
# The biggest lever is regime-conditioned exposure — don't fight the market.
REGIME_ALLOCATION = {
    "BULL": 0.95,       # Full risk-on
    "SIDEWAYS": 0.80,   # Indian markets trend up — SIDEWAYS ≠ BEAR
    "BEAR": 0.35,       # Defensive but not dead
    "UNKNOWN": 0.50,    # Moderate when uncertain
}

# Volatility targeting: scale exposure so annualized portfolio vol ≈ TARGET_VOL
# If realized vol > target, reduce exposure proportionally.
# If realized vol < target, keep exposure (don't lever up).
TARGET_VOL = 0.18  # 18% annualized target vol (Indian equities typically run 14-18%)
PROFIT_TAKE_PCT = 0.25  # take 30% profit when position gains 25%+

# ── Regime-conditioned signal weights ───────────────────────────────────────────
# Momentum dominates in trending (BULL) markets.
# Mean reversion dominates in ranging (SIDEWAYS/BEAR) markets.
SIGNAL_WEIGHTS = {
    "BULL": {
        "mom_5d": 0.15, "mom_20d": 0.30, "mom_60d": 0.20,
        "mr_sma20": -0.05, "mr_sma60": -0.05,
        "vol_ratio": -0.10, "rsi_signal": -0.15,
    },
    "SIDEWAYS": {
        "mom_5d": 0.05, "mom_20d": 0.10, "mom_60d": 0.10,
        "mr_sma20": -0.25, "mr_sma60": -0.15,
        "vol_ratio": -0.15, "rsi_signal": -0.20,
    },
    "BEAR": {
        "mom_5d": -0.05, "mom_20d": 0.05, "mom_60d": 0.05,
        "mr_sma20": -0.30, "mr_sma60": -0.20,
        "vol_ratio": -0.15, "rsi_signal": -0.25,
    },
    "UNKNOWN": {
        "mom_5d": 0.10, "mom_20d": 0.20, "mom_60d": 0.15,
        "mr_sma20": -0.15, "mr_sma60": -0.10,
        "vol_ratio": -0.10, "rsi_signal": -0.20,
    },
}


# ── Data Classes ────────────────────────────────────────────────────────────────

@dataclass
class TradeOrder:
    """A single trade executed by the simulator."""
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    date: str = ""
    symbol: str = ""
    action: str = ""           # "BUY" | "SELL"
    quantity: int = 0
    price: float = 0.0
    value: float = 0.0        # quantity × price
    cost: float = 0.0         # brokerage + impact
    signal_score: float = 0.0  # alpha score that triggered this
    regime: str = "UNKNOWN"    # market regime at time of trade
    reason: str = ""           # human-readable explanation
    pnl: float = 0.0          # realized P&L (for SELL orders)


@dataclass
class Position:
    """An open position in the portfolio."""
    symbol: str = ""
    quantity: int = 0
    avg_entry: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight: float = 0.0       # % of portfolio NAV
    entry_date: str = ""
    holding_days: int = 0


@dataclass
class DailySnapshot:
    """Portfolio state at end of day."""
    date: str
    nav: float
    cash: float
    invested: float
    n_positions: int
    day_pnl: float
    day_return_pct: float
    cumulative_return_pct: float
    benchmark_value: float = 0.0
    benchmark_return_pct: float = 0.0
    regime: str = "UNKNOWN"


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""
    sim_id: str
    status: str                 # "running" | "completed" | "error"
    initial_capital: float
    final_nav: float = 0.0
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    start_date: str = ""
    end_date: str = ""
    n_days: int = 0
    positions: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    current_signals: list = field(default_factory=list)


# ── Signal Generator ─────────────────────────────────────────────────────────────

def _compute_stock_signals(
    prices_df: pd.DataFrame,
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute alpha signals for each stock on each day.
    Returns a DataFrame with (date, symbol) index and signal columns.
    
    Signal weights are regime-conditioned:
    - BULL: heavy momentum (ride the trend)
    - BEAR/SIDEWAYS: heavy mean reversion (buy the dip, sell the rip)
    
    Also computes:
    - MACD histogram for trend confirmation
    - Bollinger %B for mean reversion confirmation  
    - Volume-weighted momentum for liquidity screening
    """
    signals = {}
    
    for symbol in prices_df.columns:
        close = prices_df[symbol].dropna()
        if len(close) < 70:
            continue
            
        s = pd.DataFrame(index=close.index)
        s['symbol'] = symbol
        
        # Momentum signals
        s['mom_5d'] = close.pct_change(5)
        s['mom_20d'] = close.pct_change(20)
        s['mom_60d'] = close.pct_change(60)
        
        # Mean reversion
        sma20 = close.rolling(20).mean()
        sma60 = close.rolling(60).mean()
        s['mr_sma20'] = (close - sma20) / sma20
        s['mr_sma60'] = (close - sma60) / sma60
        
        # Volatility ratio (prefer low vol stocks)
        vol_5 = close.pct_change().rolling(5).std()
        vol_60 = close.pct_change().rolling(60).std()
        s['vol_ratio'] = vol_5 / vol_60.replace(0, np.nan)
        
        # RSI (14-day), normalized to [-1, 1]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        s['rsi'] = 100 - (100 / (1 + rs))
        s['rsi_signal'] = (s['rsi'] - 50) / 50
        
        # MACD histogram for trend confirmation
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        s['macd_hist'] = (macd_line - signal_line) / close  # normalize
        
        # Bollinger %B for mean reversion
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        s['bb_pctb'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std.replace(0, np.nan))
        s['bb_signal'] = -(s['bb_pctb'] - 0.5) * 2  # oversold=positive, overbought=negative
        
        # 52-week high proximity — stocks near highs outperform
        high_252 = close.rolling(252, min_periods=60).max()
        s['high_proximity'] = close / high_252.replace(0, np.nan)  # 1.0 = at high, 0.8 = 20% off high
        s['high_signal'] = (s['high_proximity'] - 0.85) / 0.15  # normalize: -1 (far from high) to +1 (at high)
        s['high_signal'] = s['high_signal'].clip(-2, 2)
        
        # Trend strength — slope of 20d SMA normalized by price
        sma20_slope = sma20.diff(5) / (close * 5)  # 5-day slope of SMA20
        s['trend_strength'] = sma20_slope * 100  # scale up for z-scoring
        
        signals[symbol] = s
    
    if not signals:
        return pd.DataFrame()
    
    # Combine all stocks
    combined = pd.concat(signals.values(), axis=0)
    
    # Cross-sectional z-score each signal per date (MAD-robust)
    signal_cols = ['mom_5d', 'mom_20d', 'mom_60d', 'mr_sma20', 'mr_sma60', 
                   'vol_ratio', 'rsi_signal', 'macd_hist', 'bb_signal',
                   'high_signal', 'trend_strength']
    
    for col in signal_cols:
        combined[col] = combined.groupby(level=0)[col].transform(
            lambda x: _mad_zscore(x.values)
        )
    
    # Regime-conditioned composite alpha
    # For each day, look up the regime and use the corresponding weights
    combined['alpha'] = 0.0
    
    base_signal_cols = ['mom_5d', 'mom_20d', 'mom_60d', 'mr_sma20', 'mr_sma60',
                        'vol_ratio', 'rsi_signal']
    
    if regime_series is not None:
        for dt in combined.index.unique():
            mask = combined.index == dt
            # Get regime for this date
            if dt in regime_series.index:
                r = regime_series[dt]
                regime_str = r.value if hasattr(r, 'value') else str(r)
            else:
                regime_str = "UNKNOWN"
            
            weights = SIGNAL_WEIGHTS.get(regime_str, SIGNAL_WEIGHTS["UNKNOWN"])
            
            alpha = np.zeros(mask.sum())
            for col in base_signal_cols:
                w = weights.get(col, 0)
                alpha += w * combined.loc[mask, col].fillna(0).values
            
            # Add MACD and BB as regime-independent confirmations
            alpha += 0.10 * combined.loc[mask, 'macd_hist'].fillna(0).values
            alpha += 0.10 * combined.loc[mask, 'bb_signal'].fillna(0).values
            
            combined.loc[mask, 'alpha'] = alpha
    else:
        # No regime info — use UNKNOWN weights
        weights = SIGNAL_WEIGHTS["UNKNOWN"]
        for col in base_signal_cols:
            w = weights.get(col, 0)
            combined['alpha'] += w * combined[col].fillna(0)
        combined['alpha'] += 0.10 * combined['macd_hist'].fillna(0)
        combined['alpha'] += 0.10 * combined['bb_signal'].fillna(0)
    
    return combined


def _mad_zscore(values: np.ndarray) -> np.ndarray:
    """MAD-robust cross-sectional z-score."""
    valid = values[~np.isnan(values)]
    if len(valid) < 3:
        return np.zeros_like(values)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(valid - median))
    scale = 1.4826 * mad if mad > 1e-10 else max(np.nanstd(valid), 1e-10)
    result = (values - median) / scale
    return np.where(np.isnan(result), 0.0, result)


# ── Portfolio Optimizer (simplified) ────────────────────────────────────────────

def _optimize_portfolio(
    alpha_scores: dict[str, float],
    current_prices: dict[str, float],
    nav: float,
    regime: str = "UNKNOWN",
    vol_scalar: float = 1.0,
    max_positions: int = MAX_POSITIONS,
    max_weight: float = MAX_POSITION_PCT,
) -> dict[str, float]:
    """
    Convert alpha scores to target weights with regime-conditioned exposure.
    
    Key improvements over naive equal-weight:
    1. Only select stocks with alpha above MIN_ALPHA_THRESHOLD
    2. Conviction-based sizing: 40% equal weight + 60% alpha tilt
    3. Regime-conditioned total exposure (BULL=95%, BEAR=25%)
    4. Fewer, higher-conviction positions
    """
    if not alpha_scores:
        return {}
    
    # Get regime allocation
    target_exposure = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION["UNKNOWN"])
    
    # Dynamic max positions by regime
    regime_max_positions = max_positions
    if regime in ("BEAR", "SIDEWAYS"):
        regime_max_positions = min(max_positions, MAX_POSITIONS_DEFENSIVE)
    
    # Filter: only stocks with positive alpha above threshold
    threshold = MIN_ALPHA_THRESHOLD
    if regime == "BEAR":
        threshold = MIN_ALPHA_THRESHOLD * 2
    elif regime == "BULL":
        threshold = MIN_ALPHA_THRESHOLD * 0.5
    
    qualified = {s: a for s, a in alpha_scores.items() if a > threshold}
    
    if not qualified:
        sorted_all = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
        qualified = dict(sorted_all[:3])
    
    # Sort by alpha, take top N
    sorted_stocks = sorted(qualified.items(), key=lambda x: x[1], reverse=True)
    n_select = min(regime_max_positions, len(sorted_stocks))
    selected = dict(sorted_stocks[:n_select])
    
    if not selected:
        return {}
    
    # Conviction-based sizing: 40% equal + 60% alpha tilt
    base_weight = 1.0 / len(selected)
    alpha_values = list(selected.values())
    alpha_min = min(alpha_values)
    alpha_range = max(alpha_values) - alpha_min if len(alpha_values) > 1 else 1.0
    
    weights = {}
    for s, a in selected.items():
        if alpha_range > 1e-10:
            alpha_tilt = (a - alpha_min) / alpha_range  # 0 to 1
        else:
            alpha_tilt = 0.5
        # Higher conviction → larger position
        weights[s] = base_weight * 0.40 + base_weight * 0.60 * (alpha_tilt * 2)
    
    # Cap at max_weight
    for s in weights:
        weights[s] = min(weights[s], max_weight)
    
    # Scale to regime-conditioned exposure, with vol targeting
    total_w = sum(weights.values())
    effective_exposure = target_exposure * min(vol_scalar, 1.0)
    if total_w > 0:
        weights = {s: w / total_w * effective_exposure for s, w in weights.items()}
    
    return weights


# ── The Simulator ───────────────────────────────────────────────────────────────

class TradingSimulator:
    """
    Automated portfolio trading simulator.
    
    Runs the signal pipeline daily, generates trades, tracks positions and P&L.
    """
    
    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        brokerage_bps: float = BROKERAGE_BPS,
        impact_bps: float = IMPACT_BPS,
    ):
        self.initial_capital = initial_capital
        self.brokerage_bps = brokerage_bps
        self.impact_bps = impact_bps
        self.tc_rate = (brokerage_bps + impact_bps) / 10000.0
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}  # symbol → {qty, avg_entry, entry_date}
        self.trade_log: list[TradeOrder] = []
        self.equity_curve: list[DailySnapshot] = []
        self.daily_returns: list[float] = []
        
    def _nav(self, current_prices: dict[str, float]) -> float:
        """Calculate current NAV."""
        invested = sum(
            pos['qty'] * current_prices.get(sym, pos['avg_entry'])
            for sym, pos in self.positions.items()
        )
        return self.cash + invested
    
    def _execute_trade(
        self,
        date_str: str,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        signal_score: float,
        regime: str,
        reason: str,
    ) -> Optional[TradeOrder]:
        """Execute a single trade and update portfolio state."""
        if quantity <= 0 or price <= 0:
            return None
        
        value = quantity * price
        cost = value * self.tc_rate
        pnl = 0.0
        
        if action == "BUY":
            total_cost = value + cost
            if total_cost > self.cash:
                # Reduce quantity to fit available cash
                quantity = int((self.cash * 0.98) / (price * (1 + self.tc_rate)))
                if quantity <= 0:
                    return None
                value = quantity * price
                cost = value * self.tc_rate
                total_cost = value + cost
            
            self.cash -= total_cost
            
            if symbol in self.positions:
                # Average up
                old = self.positions[symbol]
                total_qty = old['qty'] + quantity
                old['avg_entry'] = (old['avg_entry'] * old['qty'] + price * quantity) / total_qty
                old['qty'] = total_qty
            else:
                self.positions[symbol] = {
                    'qty': quantity,
                    'avg_entry': price,
                    'entry_date': date_str,
                }
        
        elif action == "SELL":
            if symbol not in self.positions:
                return None
            
            pos = self.positions[symbol]
            sell_qty = min(quantity, pos['qty'])
            if sell_qty <= 0:
                return None
            
            value = sell_qty * price
            cost = value * self.tc_rate
            pnl = (price - pos['avg_entry']) * sell_qty - cost
            
            self.cash += value - cost
            
            pos['qty'] -= sell_qty
            if pos['qty'] <= 0:
                del self.positions[symbol]
            
            quantity = sell_qty
        
        trade = TradeOrder(
            date=date_str,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=round(price, 2),
            value=round(value, 2),
            cost=round(cost, 2),
            signal_score=round(signal_score, 4),
            regime=regime,
            reason=reason,
            pnl=round(pnl, 2),
        )
        self.trade_log.append(trade)
        return trade
    
    def run_historical_replay(
        self,
        prices_df: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
    ) -> SimulationResult:
        """
        Run a full historical replay simulation.
        
        Parameters
        ----------
        prices_df : pd.DataFrame
            Close prices with dates as index, symbols as columns.
        benchmark_prices : pd.Series, optional
            Benchmark (e.g., NIFTY 50) close prices for comparison.
        regime_series : pd.Series, optional
            Pre-computed regime labels per date.
        
        Returns
        -------
        SimulationResult with full trade log, equity curve, and metrics.
        """
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        logger.info("Starting simulation %s with ₹%.0f capital, %d stocks", 
                     sim_id, self.initial_capital, len(prices_df.columns))
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.daily_returns = []
        
        dates = prices_df.index.sort_values()
        
        # Need at least 70 days for signals
        warmup = 70
        if len(dates) < warmup + 10:
            return SimulationResult(
                sim_id=sim_id, status="error",
                initial_capital=self.initial_capital,
            )
        
        # Compute signals for all stocks (regime-conditioned)
        logger.info("Computing signals across %d dates...", len(dates))
        all_signals = _compute_stock_signals(prices_df, regime_series)
        
        if all_signals.empty:
            return SimulationResult(
                sim_id=sim_id, status="error",
                initial_capital=self.initial_capital,
            )
        
        benchmark_start = None
        prev_nav = self.initial_capital
        
        # Day-by-day simulation
        for i, dt in enumerate(dates):
            if i < warmup:
                continue
            
            date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
            
            # Get current prices
            current_prices = {}
            for sym in prices_df.columns:
                p = prices_df.loc[dt, sym]
                if not np.isnan(p):
                    current_prices[sym] = float(p)
            
            if not current_prices:
                continue
            
            # Get regime
            regime = "UNKNOWN"
            if regime_series is not None and dt in regime_series.index:
                r = regime_series[dt]
                regime = r.value if hasattr(r, 'value') else str(r)
            
            # ── Trend overlay ──────────────────────────────────────────
            # If HMM says SIDEWAYS but the benchmark SMA50 > SMA200,
            # the market is in an uptrend → treat as BULL for allocation.
            # This prevents the HMM's tendency to call everything SIDEWAYS
            # from keeping us underinvested in clear uptrends.
            if regime == "SIDEWAYS" and benchmark_prices is not None:
                try:
                    bench_up_to_now = benchmark_prices.loc[:dt]
                    if len(bench_up_to_now) >= 200:
                        sma50 = bench_up_to_now.iloc[-50:].mean()
                        sma200 = bench_up_to_now.iloc[-200:].mean()
                        sma20 = bench_up_to_now.iloc[-20:].mean()
                        if sma50 > sma200 * 1.01 and sma20 > sma50:  # uptrend confirmed
                            regime = "BULL"
                        elif sma50 < sma200 * 0.99:  # downtrend
                            regime = "BEAR"
                except Exception:
                    pass
            # Also check: if HMM says SIDEWAYS but we were in BULL and SMA is rolling over
            elif regime == "BULL" and benchmark_prices is not None:
                try:
                    bench_up_to_now = benchmark_prices.loc[:dt]
                    if len(bench_up_to_now) >= 200:
                        sma50 = bench_up_to_now.iloc[-50:].mean()
                        sma200 = bench_up_to_now.iloc[-200:].mean()
                        if sma50 < sma200 * 0.99:
                            regime = "BEAR"  # trend reversal — go defensive fast
                except Exception:
                    pass
            
            # Get today's alpha scores
            day_signals = all_signals.loc[all_signals.index == dt]
            if day_signals.empty:
                # Record snapshot without trading
                nav = self._nav(current_prices)
                self._record_snapshot(date_str, nav, prev_nav, benchmark_prices, 
                                     benchmark_start, dt, regime)
                prev_nav = nav
                continue
            
            alpha_scores = {}
            for _, row in day_signals.iterrows():
                sym = row['symbol']
                if sym in current_prices and not np.isnan(row.get('alpha', 0)):
                    alpha_scores[sym] = float(row['alpha'])
            
            # Compute target weights with regime-conditioned exposure
            nav = self._nav(current_prices)
            
            # Volatility targeting: scale regime allocation by inverse vol ratio
            vol_scalar = 1.0
            if len(self.daily_returns) >= 20:
                recent_vol = np.std(self.daily_returns[-20:]) * np.sqrt(252)
                if recent_vol > TARGET_VOL * 1.1:  # vol above target
                    vol_scalar = TARGET_VOL / recent_vol
                    vol_scalar = max(vol_scalar, 0.50)  # floor at 50%
            
            target_weights = _optimize_portfolio(
                alpha_scores, current_prices, nav, regime=regime,
                vol_scalar=vol_scalar,
            )
            
            # Check trailing stops and profit-taking every day
            self._check_trailing_stops(date_str, current_prices, regime)
            
            # Rebalance: compare target vs current
            if i % REBALANCE_INTERVAL == 0:
                self._rebalance(date_str, target_weights, current_prices, 
                               nav, alpha_scores, regime)
            
            # Record daily snapshot
            nav = self._nav(current_prices)
            
            if benchmark_prices is not None and benchmark_start is None:
                if dt in benchmark_prices.index:
                    benchmark_start = float(benchmark_prices[dt])
            
            self._record_snapshot(date_str, nav, prev_nav, benchmark_prices,
                                 benchmark_start, dt, regime)
            prev_nav = nav
        
        # Compute final metrics
        result = self._compute_result(sim_id, prices_df)
        logger.info(
            "Simulation %s complete: %d trades, Sharpe=%.3f, Return=%.1f%%, MaxDD=%.1f%%",
            sim_id, result.total_trades, result.sharpe_ratio,
            result.total_return_pct, result.max_drawdown_pct,
        )
        return result
    
    def _check_trailing_stops(
        self,
        date_str: str,
        current_prices: dict[str, float],
        regime: str,
    ):
        """
        Check trailing stop-losses and profit-taking every day.
        
        Key insight: stop-losses in BULL regimes DESTROYED returns in testing.
        Signal-based exits handle BULL rotation fine; stops are only needed
        for crash protection in BEAR.
        """
        # BULL: no trailing stops — the signal + rebalance handles rotation
        # Profit-taking still applies
        if regime == "BULL":
            # Only profit-take in BULL
            for sym in list(self.positions.keys()):
                pos = self.positions[sym]
                price = current_prices.get(sym)
                if price is None:
                    continue
                gain_pct = (price - pos['avg_entry']) / pos['avg_entry']
                if gain_pct >= PROFIT_TAKE_PCT and pos['qty'] > 1:
                    sell_qty = max(1, round(pos['qty'] * 0.30))
                    self._execute_trade(
                        date_str, sym, "SELL", sell_qty, price,
                        0.0, regime,
                        f"PROFIT-TAKE: +{gain_pct:.1%} from ₹{pos['avg_entry']:.0f}"
                    )
            return
        
        # SIDEWAYS: partial exit (50%) on 12% drawdown
        # BEAR: full exit on 8% drawdown
        if regime == "BEAR":
            stop_pct = 0.08
        else:  # SIDEWAYS or UNKNOWN
            stop_pct = 0.12
        
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            price = current_prices.get(sym)
            if price is None:
                continue
            
            # Track peak price
            if 'peak_price' not in pos:
                pos['peak_price'] = max(pos['avg_entry'], price)
            else:
                pos['peak_price'] = max(pos['peak_price'], price)
            
            # Stop-loss check
            drawdown = (pos['peak_price'] - price) / pos['peak_price']
            if drawdown >= stop_pct:
                if regime == "BEAR":
                    sell_qty = pos['qty']  # full exit in BEAR
                else:
                    sell_qty = max(1, round(pos['qty'] * 0.50))  # partial in SIDEWAYS
                
                self._execute_trade(
                    date_str, sym, "SELL", sell_qty, price,
                    0.0, regime,
                    f"STOP: {drawdown:.1%} from peak ₹{pos['peak_price']:.0f} ({regime})"
                )
                continue
            
            # Profit-taking
            gain_pct = (price - pos['avg_entry']) / pos['avg_entry']
            if gain_pct >= PROFIT_TAKE_PCT and pos['qty'] > 1:
                sell_qty = max(1, round(pos['qty'] * 0.30))
                self._execute_trade(
                    date_str, sym, "SELL", sell_qty, price,
                    0.0, regime,
                    f"PROFIT-TAKE: +{gain_pct:.1%} from ₹{pos['avg_entry']:.0f}"
                )
    
    def _rebalance(
        self,
        date_str: str,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        nav: float,
        alpha_scores: dict[str, float],
        regime: str,
    ):
        """Execute rebalance trades to match target weights."""
        # Current weights
        current_weights = {}
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos['avg_entry'])
            current_weights[sym] = (pos['qty'] * price) / nav if nav > 0 else 0
        
        # Sell positions not in target or overweight
        from datetime import datetime as _dt
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            
            # Check holding period
            try:
                entry = _dt.strptime(pos['entry_date'], '%Y-%m-%d')
                current = _dt.strptime(date_str, '%Y-%m-%d')
                holding_days = (current - entry).days
            except (ValueError, KeyError):
                holding_days = 999
            
            if holding_days < MIN_HOLDING_DAYS:
                continue
            
            if sym not in target_weights:
                price = current_prices.get(sym, pos['avg_entry'])
                self._execute_trade(
                    date_str, sym, "SELL", pos['qty'], price,
                    alpha_scores.get(sym, 0), regime,
                    f"Exit: dropped from target (held {holding_days}d)"
                )
            elif current_weights.get(sym, 0) > target_weights[sym] * (2.0 if regime == "BULL" else 1.4):
                # Trim overweight — let winners run in BULL
                pos = self.positions[sym]
                price = current_prices.get(sym, pos['avg_entry'])
                target_value = nav * target_weights[sym]
                current_value = pos['qty'] * price
                sell_value = current_value - target_value
                sell_qty = int(sell_value / price)
                if sell_qty > 0:
                    self._execute_trade(
                        date_str, sym, "SELL", sell_qty, price,
                        alpha_scores.get(sym, 0), regime,
                        f"Trim: {current_weights[sym]:.1%} → {target_weights[sym]:.1%}"
                    )
        
        # Buy new positions or add to underweight
        # Prioritize highest alpha first
        sorted_targets = sorted(target_weights.items(), key=lambda x: alpha_scores.get(x[0], 0), reverse=True)
        for sym, target_w in sorted_targets:
            if sym not in current_prices:
                continue
            price = current_prices[sym]
            current_w = current_weights.get(sym, 0)
            
            if current_w < target_w * 0.85:  # underweight by >15% — deploy faster
                target_value = nav * target_w
                current_value = self.positions.get(sym, {}).get('qty', 0) * price
                buy_value = target_value - current_value
                buy_qty = max(1, round(buy_value / price))  # round up, minimum 1 share
                
                if buy_qty > 0 and self.cash > buy_qty * price * 1.01:
                    alpha = alpha_scores.get(sym, 0)
                    self._execute_trade(
                        date_str, sym, "BUY", buy_qty, price,
                        alpha, regime,
                        f"α={alpha:.2f} w={target_w:.1%} regime={regime}"
                    )
        
        # Deploy excess cash — if we're holding too much cash relative to regime target
        regime_exposure = REGIME_ALLOCATION.get(regime, 0.50)
        nav_now = self._nav(current_prices)
        target_invested = nav_now * regime_exposure
        actual_invested = nav_now - self.cash
        excess_cash = self.cash - (nav_now * (1 - regime_exposure))
        
        if excess_cash > nav_now * 0.05:  # >5% excess cash
            # Top up highest-alpha existing positions
            for sym, target_w in sorted_targets[:5]:
                if sym not in current_prices or excess_cash < 5000:
                    break
                price = current_prices[sym]
                deploy = min(excess_cash * 0.25, nav_now * 0.03)  # deploy 25% at a time, max 3% NAV
                buy_qty = max(1, round(deploy / price))
                if self.cash > buy_qty * price * 1.01:
                    alpha = alpha_scores.get(sym, 0)
                    self._execute_trade(
                        date_str, sym, "BUY", buy_qty, price,
                        alpha, regime,
                        f"Cash deploy α={alpha:.2f} regime={regime}"
                    )
                    excess_cash -= buy_qty * price
    
    def _record_snapshot(
        self, date_str, nav, prev_nav, benchmark_prices, 
        benchmark_start, dt, regime
    ):
        """Record end-of-day portfolio snapshot."""
        day_return = (nav / prev_nav - 1) if prev_nav > 0 else 0
        cum_return = (nav / self.initial_capital - 1) * 100
        
        self.daily_returns.append(day_return)
        
        invested = sum(
            pos['qty'] * pos['avg_entry'] for pos in self.positions.values()
        )
        
        bm_value = 0.0
        bm_return = 0.0
        if benchmark_prices is not None and benchmark_start and dt in benchmark_prices.index:
            bm_value = float(benchmark_prices[dt])
            bm_return = (bm_value / benchmark_start - 1) * 100
        
        snapshot = DailySnapshot(
            date=date_str,
            nav=round(nav, 2),
            cash=round(self.cash, 2),
            invested=round(invested, 2),
            n_positions=len(self.positions),
            day_pnl=round(nav - prev_nav, 2),
            day_return_pct=round(day_return * 100, 4),
            cumulative_return_pct=round(cum_return, 2),
            benchmark_value=round(bm_value, 2),
            benchmark_return_pct=round(bm_return, 2),
            regime=regime,
        )
        self.equity_curve.append(snapshot)
    
    def _compute_result(self, sim_id: str, prices_df: pd.DataFrame) -> SimulationResult:
        """Compute final performance metrics."""
        # Get final prices
        last_date = prices_df.index[-1]
        final_prices = {}
        for sym in prices_df.columns:
            p = prices_df.loc[last_date, sym]
            if not np.isnan(p):
                final_prices[sym] = float(p)
        
        final_nav = self._nav(final_prices)
        total_return = (final_nav / self.initial_capital - 1) * 100
        total_pnl = final_nav - self.initial_capital
        
        # Trade stats
        winning = [t for t in self.trade_log if t.action == "SELL" and t.pnl > 0]
        losing = [t for t in self.trade_log if t.action == "SELL" and t.pnl <= 0]
        all_sells = [t for t in self.trade_log if t.action == "SELL"]
        
        win_rate = len(winning) / len(all_sells) * 100 if all_sells else 0
        avg_pnl = sum(t.pnl for t in all_sells) / len(all_sells) if all_sells else 0
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        returns_arr = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        ann_return = np.mean(returns_arr) * 252
        ann_vol = np.std(returns_arr) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0
        
        downside = returns_arr[returns_arr < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-10
        sortino = ann_return / downside_vol if downside_vol > 1e-10 else 0
        
        # Max drawdown
        if self.equity_curve:
            navs = np.array([s.nav for s in self.equity_curve])
            peak = np.maximum.accumulate(navs)
            drawdown = (navs - peak) / peak
            max_dd = abs(float(np.min(drawdown))) * 100
        else:
            max_dd = 0
        
        calmar = (ann_return * 100) / max_dd if max_dd > 0 else 0
        
        # Current positions
        positions_list = []
        for sym, pos in self.positions.items():
            price = final_prices.get(sym, pos['avg_entry'])
            unrealized = (price - pos['avg_entry']) * pos['qty']
            positions_list.append(Position(
                symbol=sym,
                quantity=pos['qty'],
                avg_entry=round(pos['avg_entry'], 2),
                current_price=round(price, 2),
                market_value=round(pos['qty'] * price, 2),
                unrealized_pnl=round(unrealized, 2),
                unrealized_pnl_pct=round((price / pos['avg_entry'] - 1) * 100, 2),
                weight=round(pos['qty'] * price / final_nav * 100, 2) if final_nav > 0 else 0,
                entry_date=pos['entry_date'],
            ))
        
        return SimulationResult(
            sim_id=sim_id,
            status="completed",
            initial_capital=self.initial_capital,
            final_nav=round(final_nav, 2),
            total_return_pct=round(total_return, 2),
            total_pnl=round(total_pnl, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_dd, 2),
            win_rate=round(win_rate, 1),
            total_trades=len(self.trade_log),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_trade_pnl=round(avg_pnl, 2),
            profit_factor=round(profit_factor, 3),
            calmar_ratio=round(calmar, 3),
            start_date=self.equity_curve[0].date if self.equity_curve else "",
            end_date=self.equity_curve[-1].date if self.equity_curve else "",
            n_days=len(self.equity_curve),
            positions=[vars(p) for p in positions_list],
            trade_log=[vars(t) for t in self.trade_log],
            equity_curve=[vars(s) for s in self.equity_curve],
        )


# ── CLI for quick testing ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAPITAL
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    
    print(f"Loading NIFTY 50 stocks data ({days} days)...")
    
    import yfinance as yf
    
    # NIFTY 50 top stocks
    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE", "WIPRO",
        "NESTLEIND", "M&M", "TATAMOTORS", "POWERGRID", "NTPC",
    ]
    
    # Download data
    tickers = [f"{s}.NS" for s in symbols]
    data = yf.download(tickers, period=f"{days}d", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
        prices.columns = [c.replace('.NS', '') for c in prices.columns]
    else:
        prices = data[['Close']]
        prices.columns = [symbols[0]]
    
    prices.index = prices.index.tz_localize(None) if prices.index.tz else prices.index
    
    # Download benchmark
    bench = yf.Ticker("^NSEI").history(period=f"{days}d")
    bench_prices = bench['Close']
    bench_prices.index = bench_prices.index.tz_localize(None)
    
    # Run regime detection
    from app.quant.regime import RegimeDetector
    detector = RegimeDetector()
    detector.fit(bench_prices)
    regime_series = detector.predict(bench_prices)
    current_regime = detector.current_regime(bench_prices)
    print(f"Current market regime: {current_regime}")
    
    # Run simulation
    sim = TradingSimulator(initial_capital=capital)
    result = sim.run_historical_replay(prices, bench_prices, regime_series)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  ARTHA Simulation Results")
    print(f"{'='*60}")
    print(f"  Capital:      ₹{result.initial_capital:,.0f}")
    print(f"  Final NAV:    ₹{result.final_nav:,.0f}")
    print(f"  Total P&L:    ₹{result.total_pnl:,.0f} ({result.total_return_pct:+.1f}%)")
    print(f"  Sharpe:       {result.sharpe_ratio:.3f}")
    print(f"  Sortino:      {result.sortino_ratio:.3f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"  Win Rate:     {result.win_rate:.1f}%")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Profit Factor:{result.profit_factor:.3f}")
    print(f"  Period:       {result.start_date} → {result.end_date} ({result.n_days} days)")
    print(f"{'='*60}")
    
    # Print open positions
    if result.positions:
        print(f"\nOpen Positions ({len(result.positions)}):")
        for p in result.positions:
            sign = "+" if p['unrealized_pnl'] >= 0 else ""
            print(f"  {p['symbol']:12s} {p['quantity']:>4d} × ₹{p['current_price']:>8.2f} = ₹{p['market_value']:>10,.2f}  {sign}₹{p['unrealized_pnl']:>8,.2f} ({sign}{p['unrealized_pnl_pct']:.1f}%)")
    
    # Print last 10 trades
    if result.trade_log:
        print(f"\nLast 10 Trades:")
        for t in result.trade_log[-10:]:
            emoji = "🟢" if t['action'] == "BUY" else "🔴"
            pnl_str = f" P&L: ₹{t['pnl']:+,.2f}" if t['action'] == "SELL" else ""
            print(f"  {emoji} {t['date']} {t['action']:4s} {t['symbol']:12s} {t['quantity']:>4d} × ₹{t['price']:>8.2f}{pnl_str}  [{t['reason']}]")
    
    # Save full result
    output_path = f"data/Reports/simulation_{result.sim_id}.json"
    with open(output_path, 'w') as f:
        json.dump(vars(result), f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")
