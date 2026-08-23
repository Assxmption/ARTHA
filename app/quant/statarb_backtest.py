"""
Statistical Arbitrage Backtester
==================================
Converts PairSignal sequences from statarb.py into a daily return stream
that can be fed into the multi-strategy portfolio engine.

Design:
  - Takes a cointegrated pair's price series + signal series
  - Simulates long/short spread trades based on z-score signals
  - Produces a daily return series (dollar-neutral, market-neutral)
  - Transaction costs modeled (brokerage + impact)

The output is a pd.Series of daily returns that the multi_strategy.py
engine can combine with other strategy sleeves.

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.statarb import (
    test_cointegration,
    generate_pair_signals,
    discover_pairs,
    PairSignal,
    CointPair,
)

logger = logging.getLogger(__name__)

# Transaction costs (realistic for Indian retail)
BROKERAGE_BPS = 5    # 0.05% per leg per trade
IMPACT_BPS = 10      # 0.10% estimated market impact per leg
TOTAL_COST_BPS = (BROKERAGE_BPS + IMPACT_BPS) * 2  # Both legs, round-trip


@dataclass
class PairBacktestResult:
    """Result from backtesting a single pair."""
    symbol_a: str
    symbol_b: str
    n_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float         # Cumulative return (decimal)
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_holding_days: float
    daily_returns: pd.Series


def backtest_pair(
    series_a: pd.Series,
    series_b: pd.Series,
    symbol_a: str = "A",
    symbol_b: str = "B",
    capital_per_leg: float = 10_00_000,  # ₹10L per leg
    lookback: int = 60,
    use_kalman: bool = True,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    max_holding_days: int = 60,
) -> PairBacktestResult:
    """
    Backtest a mean-reversion pairs trading strategy.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Close prices for the two instruments.
    capital_per_leg : float
        Notional capital per leg (used for return calculation).
    lookback : int
        Rolling window for z-score calculation.
    use_kalman : bool
        Use Kalman filter for hedge ratio.
    z_entry, z_exit, z_stop : float
        Signal thresholds.
    max_holding_days : int
        Force-close after this many days (prevents zombie positions).

    Returns
    -------
    PairBacktestResult
    """
    # Generate signals
    signals = generate_pair_signals(
        series_a, series_b, symbol_a, symbol_b,
        lookback=lookback, use_kalman=use_kalman,
        z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,
    )

    if len(signals) < 10:
        return PairBacktestResult(
            symbol_a=symbol_a, symbol_b=symbol_b,
            n_trades=0, winning_trades=0, losing_trades=0,
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, avg_holding_days=0,
            daily_returns=pd.Series(dtype=float),
        )

    # Build signal DataFrame for vectorized simulation
    dates = [s.date for s in signals]
    z_scores = [s.z_score for s in signals]
    signal_types = [s.signal for s in signals]
    hedge_ratios = [s.hedge_ratio for s in signals]

    # Align prices to signal dates
    price_a = series_a.reindex([s.date for s in signals]).ffill()
    price_b = series_b.reindex([s.date for s in signals]).ffill()

    # Simulate trades
    position = 0  # +1 = long A short B, -1 = short A long B, 0 = flat
    entry_spread = 0.0
    entry_date_idx = 0
    days_in_trade = 0
    trades: list[float] = []  # Per-trade returns
    holding_days: list[int] = []

    # Daily P&L tracking
    daily_pnl = pd.Series(0.0, index=dates)

    for i, sig in enumerate(signals):
        if i == 0:
            continue

        # Previous day's prices for daily P&L
        prev_a = float(price_a.iloc[i - 1])
        prev_b = float(price_b.iloc[i - 1])
        curr_a = float(price_a.iloc[i])
        curr_b = float(price_b.iloc[i])
        hr = hedge_ratios[i]

        if position != 0:
            days_in_trade += 1

            # Daily spread return
            # Long A Short B: profit from (A going up, B going down)
            ret_a = (curr_a - prev_a) / prev_a if prev_a > 0 else 0
            ret_b = (curr_b - prev_b) / prev_b if prev_b > 0 else 0
            spread_ret = position * (ret_a - hr * ret_b) / (1 + abs(hr))
            daily_pnl.iloc[i] = spread_ret

            # Check exit conditions
            should_exit = (
                signal_types[i] == "EXIT"
                or signal_types[i] == "STOP"
                or days_in_trade >= max_holding_days
                # Also exit if signal reverses
                or (position == 1 and signal_types[i] == "SHORT_A_LONG_B")
                or (position == -1 and signal_types[i] == "LONG_A_SHORT_B")
            )

            if should_exit:
                # Close position — accumulate trade return
                trade_return = daily_pnl.iloc[entry_date_idx + 1: i + 1].sum()
                # Subtract transaction costs (round trip)
                trade_return -= TOTAL_COST_BPS / 10000
                trades.append(trade_return)
                holding_days.append(days_in_trade)
                position = 0
                days_in_trade = 0

        # Check entry conditions (only if flat)
        if position == 0:
            if signal_types[i] == "LONG_A_SHORT_B":
                position = 1
                entry_date_idx = i
                days_in_trade = 0
                # Entry cost
                daily_pnl.iloc[i] -= TOTAL_COST_BPS / 10000 / 2  # Half on entry
            elif signal_types[i] == "SHORT_A_LONG_B":
                position = -1
                entry_date_idx = i
                days_in_trade = 0
                daily_pnl.iloc[i] -= TOTAL_COST_BPS / 10000 / 2

    # Compute metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t > 0)
    losses = n_trades - wins

    daily = daily_pnl.values
    total_ret = float(np.sum(daily))
    n_days = len(daily)
    n_years = max(n_days / 252, 0.01)
    ann_return = total_ret / n_years

    daily_std = float(np.std(daily)) if n_days > 1 else 1.0
    ann_vol = daily_std * np.sqrt(252)
    sharpe = (np.mean(daily) / max(daily_std, 1e-8)) * np.sqrt(252)

    downside = daily[daily < 0]
    down_std = float(np.std(downside)) if len(downside) > 1 else daily_std
    sortino = (np.mean(daily) / max(down_std, 1e-8)) * np.sqrt(252)

    # Max drawdown
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    avg_hold = float(np.mean(holding_days)) if holding_days else 0.0

    return PairBacktestResult(
        symbol_a=symbol_a, symbol_b=symbol_b,
        n_trades=n_trades,
        winning_trades=wins, losing_trades=losses,
        total_return=round(total_ret, 6),
        annualized_return=round(ann_return, 6),
        sharpe_ratio=round(float(sharpe), 4),
        sortino_ratio=round(float(sortino), 4),
        max_drawdown=round(max_dd, 6),
        avg_holding_days=round(avg_hold, 1),
        daily_returns=daily_pnl,
    )


def backtest_pair_universe(
    price_data: dict[str, pd.Series],
    pairs: Optional[list[CointPair]] = None,
    capital_per_leg: float = 10_00_000,
    lookback: int = 60,
    max_pairs: int = 10,
) -> dict[str, PairBacktestResult]:
    """
    Backtest all cointegrated pairs in a universe.

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series.
    pairs : list[CointPair], optional
        Pre-discovered pairs. If None, discovers them.
    capital_per_leg : float
        Notional per leg.
    lookback : int
        Z-score lookback.
    max_pairs : int
        Maximum pairs to backtest (ranked by p-value).

    Returns
    -------
    dict[str, PairBacktestResult]
        Pair label → backtest result.
    """
    if pairs is None:
        pairs = discover_pairs(price_data)

    results = {}
    for pair in pairs[:max_pairs]:
        sa = price_data.get(pair.symbol_a)
        sb = price_data.get(pair.symbol_b)
        if sa is None or sb is None:
            continue

        label = f"{pair.symbol_a}/{pair.symbol_b}"
        result = backtest_pair(
            sa, sb, pair.symbol_a, pair.symbol_b,
            capital_per_leg=capital_per_leg,
            lookback=lookback,
        )
        results[label] = result
        logger.info(
            "Pair %s: %d trades, Sharpe %.3f, Return %.2f%%",
            label, result.n_trades, result.sharpe_ratio,
            result.total_return * 100,
        )

    return results
