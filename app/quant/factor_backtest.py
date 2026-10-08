"""
Factor Model Backtester
========================
Converts cross-sectional factor rankings into a long/short portfolio
that produces a daily return stream for the multi-strategy engine.

Strategy:
  - Each month, rank all stocks by composite alpha (from factors.py)
  - Long top quintile (20%), short bottom quintile (20%)
  - Equal-weight within each quintile → dollar-neutral
  - Rebalance monthly (21 trading days)
  - Transaction costs modeled per AGENTS.md

The output is a pd.Series of daily returns that multi_strategy.py
combines with other strategy sleeves.

Cross-sectional momentum (Jegadeesh & Titman, 1993) is one of the
most persistent anomalies in global equities. Works particularly well
in Indian markets due to retail participation and herding behavior.

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.factors import compute_composite_alpha, FactorExposure

logger = logging.getLogger(__name__)

# Transaction costs (realistic for Indian retail)
BROKERAGE_BPS = 5    # 0.05% per leg per trade
IMPACT_BPS = 10      # 0.10% estimated market impact per leg
# Note: only charged on rebalance days, not holding days
COST_PER_REBAL_BPS = (BROKERAGE_BPS + IMPACT_BPS) * 2  # Round-trip both legs


@dataclass
class FactorBacktestResult:
    """Result from backtesting the factor model."""
    n_rebalances: int
    n_stocks_long: int
    n_stocks_short: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_monthly_turnover: float
    daily_returns: pd.Series
    long_symbols: list[list[str]]   # Per-rebalance long portfolio
    short_symbols: list[list[str]]  # Per-rebalance short portfolio


def backtest_factor_model(
    price_data: dict[str, pd.DataFrame],
    fundamentals: Optional[dict[str, dict]] = None,
    rebalance_days: int = 21,
    lookback_momentum: int = 252,
    quintile_pct: float = 0.20,
    min_stocks: int = 10,
    long_only: bool = False,
) -> FactorBacktestResult:
    """
    Run a long/short factor backtest on a universe of stocks.

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        Symbol → DataFrame with OHLCV columns ('Close', 'Volume', etc.).
    fundamentals : dict, optional
        Symbol → dict with fundamental metrics (pe, pb, roe, etc.).
    rebalance_days : int
        Days between rebalances (21 = monthly).
    lookback_momentum : int
        Days for momentum factor computation.
    quintile_pct : float
        Fraction of universe in each leg (0.20 = top/bottom 20%).
    min_stocks : int
        Minimum stocks required to run.

    Returns
    -------
    FactorBacktestResult
    """
    symbols = list(price_data.keys())
    if len(symbols) < min_stocks:
        logger.warning("Universe too small: %d stocks < %d minimum", len(symbols), min_stocks)
        return _empty_result()

    # Get all close prices aligned to a common date index
    close_dfs = {}
    for sym in symbols:
        df = price_data[sym]
        if 'Close' in df.columns and len(df) > lookback_momentum:
            close_dfs[sym] = df['Close']

    if len(close_dfs) < min_stocks:
        return _empty_result()

    # Align to common dates
    close_panel = pd.DataFrame(close_dfs)
    close_panel = close_panel.dropna(how='all')

    # Need enough history
    if len(close_panel) < lookback_momentum + rebalance_days:
        return _empty_result()

    # Determine rebalance dates
    dates = close_panel.index
    rebal_indices = list(range(lookback_momentum, len(dates), rebalance_days))

    if not rebal_indices:
        return _empty_result()

    # Daily returns of all stocks
    stock_returns = close_panel.pct_change()

    # Track portfolio returns
    daily_rets = pd.Series(0.0, index=dates[lookback_momentum:])
    n_rebalances = 0
    long_history = []
    short_history = []
    turnover_pcts = []

    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i, rebal_idx in enumerate(rebal_indices):
        # Build price_data dict as of rebalance date (pd.Series per symbol)
        hist_start = max(0, rebal_idx - lookback_momentum)
        price_slices: dict[str, pd.Series] = {}
        for sym in close_panel.columns:
            slc = close_panel[sym].iloc[hist_start:rebal_idx + 1].dropna()
            if len(slc) >= 60:
                price_slices[sym] = slc

        if len(price_slices) < min_stocks:
            continue

        # Compute factor scores using the actual factors.py API
        exposures = compute_composite_alpha(
            price_data=price_slices,
            fundamentals=fundamentals,
        )

        if len(exposures) < min_stocks:
            continue

        # Select quintiles (already sorted by composite alpha, best first)
        n_per_leg = max(1, int(len(exposures) * quintile_pct))

        long_syms = [e.symbol for e in exposures[:n_per_leg]]
        short_syms = [] if long_only else [e.symbol for e in exposures[-n_per_leg:]]

        long_history.append(long_syms)
        short_history.append(short_syms)

        # Calculate turnover
        new_long = set(long_syms)
        new_short = set(short_syms)
        turnover = len(new_long - prev_long) + len(new_short - prev_short)
        total_positions = len(new_long) + len(new_short)
        turnover_pct = turnover / max(total_positions, 1)
        turnover_pcts.append(turnover_pct)

        prev_long = new_long
        prev_short = new_short

        # Hold period: from this rebalance to next
        hold_end = rebal_indices[i + 1] if i + 1 < len(rebal_indices) else len(dates)

        for d in range(rebal_idx + 1, min(hold_end, len(dates))):
            dt = dates[d]
            if dt not in daily_rets.index:
                continue

            # Equal-weight L/S return
            long_ret = 0.0
            n_long = 0
            for sym in long_syms:
                if sym in stock_returns.columns and d < len(stock_returns):
                    r = stock_returns[sym].iloc[d]
                    if not np.isnan(r):
                        long_ret += r
                        n_long += 1

            short_ret = 0.0
            n_short = 0
            for sym in short_syms:
                if sym in stock_returns.columns and d < len(stock_returns):
                    r = stock_returns[sym].iloc[d]
                    if not np.isnan(r):
                        short_ret += r
                        n_short += 1

            # Dollar-neutral (L/S) or long-only
            portfolio_ret = 0.0
            if n_long > 0:
                portfolio_ret += long_ret / n_long
            if n_short > 0:
                portfolio_ret -= short_ret / n_short
            elif long_only and n_long > 0:
                # Long-only: scale to half exposure (comparable vol to L/S)
                portfolio_ret *= 0.5

            # Transaction costs on rebalance day only
            if d == rebal_idx + 1:
                portfolio_ret -= COST_PER_REBAL_BPS / 10000 * turnover_pct

            daily_rets.loc[dt] = portfolio_ret

        n_rebalances += 1

    # Compute metrics
    daily_arr = daily_rets.values
    daily_arr = daily_arr[~np.isnan(daily_arr)]

    if len(daily_arr) < 20:
        return _empty_result()

    total_ret = float(np.sum(daily_arr))
    n_years = max(len(daily_arr) / 252, 0.01)
    ann_ret = total_ret / n_years

    daily_std = float(np.std(daily_arr)) if len(daily_arr) > 1 else 1.0
    sharpe = (np.mean(daily_arr) / max(daily_std, 1e-8)) * np.sqrt(252)

    down = daily_arr[daily_arr < 0]
    down_std = float(np.std(down)) if len(down) > 1 else daily_std
    sortino = (np.mean(daily_arr) / max(down_std, 1e-8)) * np.sqrt(252)

    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    avg_turnover = float(np.mean(turnover_pcts)) if turnover_pcts else 0.0

    n_long = len(long_history[-1]) if long_history else 0
    n_short = len(short_history[-1]) if short_history else 0

    logger.info(
        "Factor backtest: %d rebalances, Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
        n_rebalances, sharpe, ann_ret * 100, max_dd * 100,
    )

    return FactorBacktestResult(
        n_rebalances=n_rebalances,
        n_stocks_long=n_long,
        n_stocks_short=n_short,
        total_return=round(total_ret, 6),
        annualized_return=round(ann_ret, 6),
        sharpe_ratio=round(float(sharpe), 4),
        sortino_ratio=round(float(sortino), 4),
        max_drawdown=round(max_dd, 6),
        avg_monthly_turnover=round(avg_turnover, 4),
        daily_returns=daily_rets,
        long_symbols=long_history,
        short_symbols=short_history,
    )


def _empty_result() -> FactorBacktestResult:
    """Return an empty result when data is insufficient."""
    return FactorBacktestResult(
        n_rebalances=0, n_stocks_long=0, n_stocks_short=0,
        total_return=0, annualized_return=0, sharpe_ratio=0,
        sortino_ratio=0, max_drawdown=0, avg_monthly_turnover=0,
        daily_returns=pd.Series(dtype=float),
        long_symbols=[], short_symbols=[],
    )
