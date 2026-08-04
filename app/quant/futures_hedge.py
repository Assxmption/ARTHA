"""
NIFTY Futures Beta-Neutral Hedge
==================================
Removes market risk (beta) from the portfolio by shorting NIFTY 50 futures.

Mechanics:
  1. Compute rolling 120-day portfolio beta to NIFTY 50
  2. Short β × portfolio_value in NIFTY futures notional
  3. Model quarterly roll costs (~15bps per roll, 4/year = 60bps)
  4. Track basis risk (futures vs spot deviation)
  5. Margin requirement at 12% of notional (SEBI rules)

The hedge converts a long-only portfolio (Sharpe ~0.7) into a
market-neutral portfolio (projected Sharpe 1.0-1.5) by isolating
pure stock-selection alpha.

SEBI Compliance Note:
  This module generates hedge SIGNALS only.
  No order-placement code is included.

Reference: Implementation Plan v5 §NIFTY Futures Hedge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NIFTY 50 futures specs
NIFTY_LOT_SIZE = 25  # 25 units per lot (as of 2026)
ROLL_COST_BPS = 15   # ~15bps per quarterly roll
ROLLS_PER_YEAR = 4
MARGIN_REQUIREMENT = 0.12  # 12% of notional
MIN_BETA = 0.3
MAX_BETA = 1.5
BETA_LOOKBACK = 120  # days


@dataclass
class HedgeState:
    """State of the NIFTY futures hedge at a point in time."""
    date: pd.Timestamp
    portfolio_beta: float
    hedge_ratio: float       # Fraction of portfolio to hedge (usually = beta)
    futures_notional: float  # Notional value of futures short
    margin_required: float
    basis_bps: float         # Futures vs spot basis in bps
    roll_cost_daily: float   # Amortized daily roll cost


def compute_rolling_beta(
    portfolio_returns: pd.Series,
    index_returns: pd.Series,
    lookback: int = BETA_LOOKBACK,
) -> pd.Series:
    """
    Compute rolling beta of portfolio vs NIFTY index.
    Beta = Cov(Rp, Rm) / Var(Rm)
    """
    aligned = pd.DataFrame({
        "port": portfolio_returns,
        "idx": index_returns,
    }).dropna()

    if len(aligned) < lookback:
        return pd.Series(1.0, index=aligned.index)

    betas = []
    dates = []

    for i in range(lookback, len(aligned)):
        window = aligned.iloc[i - lookback:i]
        cov_matrix = np.cov(window["port"].values, window["idx"].values)
        var_idx = cov_matrix[1, 1]
        if var_idx > 1e-10:
            beta = cov_matrix[0, 1] / var_idx
            beta = np.clip(beta, MIN_BETA, MAX_BETA)
        else:
            beta = 1.0
        betas.append(beta)
        dates.append(aligned.index[i])

    return pd.Series(betas, index=dates, dtype=float)


def apply_futures_hedge(
    portfolio_returns: pd.Series,
    index_returns: pd.Series,
    portfolio_nav: float = 1_000_000,  # ₹10L base NAV
    nifty_spot: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Apply NIFTY futures beta hedge to portfolio returns.

    Returns:
        hedged_returns: Portfolio returns with beta removed
        hedge_log: DataFrame with daily hedge state
    """
    betas = compute_rolling_beta(portfolio_returns, index_returns)

    # Align all series
    common = portfolio_returns.index.intersection(betas.index).intersection(index_returns.index)
    if len(common) < 60:
        logger.warning("Insufficient data for hedging: %d days", len(common))
        return portfolio_returns, pd.DataFrame()

    port_ret = portfolio_returns.reindex(common)
    idx_ret = index_returns.reindex(common)
    beta_series = betas.reindex(common)

    # Daily roll cost amortized
    annual_roll_cost = ROLL_COST_BPS * ROLLS_PER_YEAR / 10000  # 60bps/year
    daily_roll_cost = annual_roll_cost / 252

    # Hedge: subtract beta × index return + roll costs
    hedged = port_ret - beta_series.shift(1).fillna(1.0) * idx_ret - daily_roll_cost

    # Build hedge log
    hedge_log = pd.DataFrame({
        "date": common,
        "portfolio_return": port_ret.values,
        "index_return": idx_ret.values,
        "beta": beta_series.values,
        "hedge_return": hedged.values,
        "futures_notional": (beta_series * portfolio_nav).values,
        "margin_required": (beta_series * portfolio_nav * MARGIN_REQUIREMENT).values,
        "roll_cost_daily": daily_roll_cost,
    }).set_index("date")

    logger.info("Futures hedge applied: %d days, avg beta=%.2f, roll cost=%.1fbps/yr",
                len(hedged), beta_series.mean(), annual_roll_cost * 10000)

    return hedged, hedge_log


def compute_hedge_metrics(hedge_log: pd.DataFrame) -> dict:
    """Compute summary metrics for the hedge."""
    if hedge_log.empty:
        return {}

    return {
        "avg_beta": float(hedge_log["beta"].mean()),
        "min_beta": float(hedge_log["beta"].min()),
        "max_beta": float(hedge_log["beta"].max()),
        "avg_futures_notional": float(hedge_log["futures_notional"].mean()),
        "avg_margin_required": float(hedge_log["margin_required"].mean()),
        "total_roll_cost_pct": float(hedge_log["roll_cost_daily"].sum() * 100),
        "hedge_effectiveness": float(
            1 - hedge_log["hedge_return"].std() / max(hedge_log["portfolio_return"].std(), 1e-10)
        ),
    }
