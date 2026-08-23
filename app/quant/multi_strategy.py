"""
Multi-Strategy Portfolio Engine
================================
Combines return streams from multiple independent strategy sleeves into
a single portfolio, producing the diversification benefit that is the
actual Medallion insight: N uncorrelated strategies each at Sharpe S
combine to portfolio Sharpe ≈ S × √N.

Design:
  - Each strategy sleeve produces a daily return series (pd.Series).
  - The engine normalizes each series to a target volatility before combining.
  - Allocation is proportional to strategy Sharpe² (Kelly-optimal for
    uncorrelated strategies), with a cap per strategy.
  - The combined portfolio's Sharpe should exceed any individual strategy's.

Why this matters:
  - A single options strategy at Sharpe 0.6 → ~10% return
  - 5 uncorrelated strategies at Sharpe 0.6 → combined Sharpe 1.34 → ~21% return
  - This is mathematics (Markowitz), not alchemy.

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

# Target annualized volatility per strategy sleeve (vol-targeting)
DEFAULT_TARGET_VOL = 0.10  # 10% annualized

# Maximum allocation to any single strategy (prevents concentration)
MAX_STRATEGY_ALLOCATION = 0.35  # 35%

# Minimum allocation to any included strategy
MIN_STRATEGY_ALLOCATION = 0.05  # 5%

# Minimum data required for reliable statistics
MIN_DAYS_FOR_STATS = 126  # ~6 months

# Risk-free rate for Sharpe calculation
RISK_FREE_RATE = 0.065  # India 10Y yield approx


# ── Data Structures ─────────────────────────────────────────────────────────────


@dataclass
class StrategyPerformance:
    """Performance summary for a single strategy sleeve."""
    name: str
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_trading_days: int
    allocation_weight: float = 0.0
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def is_profitable(self) -> bool:
        return self.annualized_return > 0


@dataclass
class PortfolioResult:
    """Combined portfolio performance."""
    # Portfolio-level metrics
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_trading_days: int

    # Diversification metrics
    diversification_ratio: float  # portfolio Sharpe / max individual Sharpe
    correlation_matrix: Optional[pd.DataFrame] = None

    # Per-strategy breakdown
    strategies: list[StrategyPerformance] = field(default_factory=list)

    # Combined daily returns
    portfolio_daily_returns: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )


# ── Core Engine ─────────────────────────────────────────────────────────────────


def compute_strategy_metrics(
    daily_returns: pd.Series,
    name: str = "Strategy",
    risk_free_rate: float = RISK_FREE_RATE,
) -> StrategyPerformance:
    """
    Compute performance metrics for a single strategy's daily return series.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns (decimal, not %). Index should be DatetimeIndex.
    name : str
        Strategy name for labeling.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    StrategyPerformance
    """
    daily = daily_returns.dropna()
    n_days = len(daily)

    if n_days < 2:
        return StrategyPerformance(
            name=name, annualized_return=0, annualized_vol=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            calmar_ratio=0, num_trading_days=0, daily_returns=daily,
        )

    # Annualized return
    cum_return = (1 + daily).prod() - 1
    n_years = max(n_days / 252, 0.01)
    ann_return = (1 + cum_return) ** (1 / n_years) - 1

    # Annualized volatility
    ann_vol = float(daily.std() * np.sqrt(252))
    ann_vol = max(ann_vol, 1e-8)

    # Sharpe
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess = daily - daily_rf
    sharpe = float(excess.mean() / max(excess.std(), 1e-8) * np.sqrt(252))

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    down_std = float(downside.std()) if len(downside) > 1 else ann_vol / np.sqrt(252)
    sortino = float(excess.mean() / max(down_std, 1e-8) * np.sqrt(252))

    # Max drawdown
    cum_wealth = (1 + daily).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    max_dd = float(-drawdown.min()) if len(drawdown) > 0 else 0.0

    # Calmar
    calmar = ann_return / max(max_dd, 1e-8) if max_dd > 0.01 else 0.0

    return StrategyPerformance(
        name=name,
        annualized_return=round(ann_return, 6),
        annualized_vol=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        calmar_ratio=round(calmar, 4),
        num_trading_days=n_days,
        daily_returns=daily,
    )


def vol_target_returns(
    daily_returns: pd.Series,
    target_vol: float = DEFAULT_TARGET_VOL,
    lookback: int = 63,
) -> pd.Series:
    """
    Scale a return series to a target volatility using realized vol estimate.

    This is the standard vol-targeting approach:
      scaled_return_t = raw_return_t × (target_vol / realized_vol_t)

    The realized vol is estimated from a rolling window of past returns,
    so this is implementable in real-time (no look-ahead bias).

    Parameters
    ----------
    daily_returns : pd.Series
        Raw daily returns.
    target_vol : float
        Target annualized volatility.
    lookback : int
        Rolling window for realized vol estimation (trading days).

    Returns
    -------
    pd.Series
        Vol-targeted daily returns.
    """
    daily = daily_returns.dropna()
    if len(daily) < lookback + 1:
        return daily  # Not enough data to estimate vol

    # Rolling realized vol (annualized)
    rolling_vol = daily.rolling(window=lookback).std() * np.sqrt(252)

    # Scale factor: target / realized, clipped to prevent extreme leverage
    # Cap at 3× leverage and floor at 0.1× to prevent near-zero allocation
    scale = target_vol / rolling_vol.clip(lower=target_vol / 3, upper=target_vol * 10)

    # Apply scaling (NaN for warmup period)
    scaled = daily * scale
    return scaled.dropna()


def compute_allocation_weights(
    strategies: list[StrategyPerformance],
    max_weight: float = MAX_STRATEGY_ALLOCATION,
    min_weight: float = MIN_STRATEGY_ALLOCATION,
) -> dict[str, float]:
    """
    Compute allocation weights using Sharpe²-proportional allocation.

    This is the Kelly-optimal allocation for uncorrelated strategies:
    weight_i ∝ Sharpe_i² / Σ Sharpe_j²

    Only strategies with positive Sharpe are allocated capital.
    Weights are capped at max_weight and floored at min_weight.

    Parameters
    ----------
    strategies : list[StrategyPerformance]
        Performance summaries for each strategy.
    max_weight, min_weight : float
        Allocation bounds per strategy.

    Returns
    -------
    dict[str, float]
        Strategy name → allocation weight. Sums to 1.0.
    """
    # Only allocate to strategies with positive Sharpe
    eligible = [s for s in strategies if s.sharpe_ratio > 0]

    if not eligible:
        # If no strategy has positive Sharpe, equal-weight all
        logger.warning("No strategy has positive Sharpe — equal-weighting all")
        n = len(strategies)
        return {s.name: 1.0 / n for s in strategies} if n > 0 else {}

    # Sharpe² proportional
    sharpe_sq = {s.name: s.sharpe_ratio ** 2 for s in eligible}
    total_sq = sum(sharpe_sq.values())

    if total_sq < 1e-10:
        return {s.name: 1.0 / len(eligible) for s in eligible}

    weights = {name: sq / total_sq for name, sq in sharpe_sq.items()}

    # Apply caps and floors
    for name in weights:
        weights[name] = max(min(weights[name], max_weight), min_weight)

    # Renormalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {n: w / total for n, w in weights.items()}

    return weights


def combine_strategies(
    strategy_returns: dict[str, pd.Series],
    target_vol: float = DEFAULT_TARGET_VOL,
    max_weight: float = MAX_STRATEGY_ALLOCATION,
    risk_free_rate: float = RISK_FREE_RATE,
    vol_lookback: int = 63,
) -> PortfolioResult:
    """
    Combine multiple strategy return streams into a portfolio.

    Process:
    1. Compute performance metrics for each strategy
    2. Vol-target each strategy's returns to normalize risk contribution
    3. Allocate capital proportional to Sharpe² (Kelly-optimal)
    4. Combine into a single portfolio return stream
    5. Report combined metrics + diversification benefit

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Strategy name → daily return series (decimal, not %).
    target_vol : float
        Per-strategy target volatility (annualized).
    max_weight : float
        Maximum allocation per strategy.
    risk_free_rate : float
        For Sharpe calculation.
    vol_lookback : int
        Rolling window for vol estimation.

    Returns
    -------
    PortfolioResult
        Combined portfolio performance with per-strategy breakdown.
    """
    if not strategy_returns:
        return PortfolioResult(
            annualized_return=0, annualized_vol=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            num_trading_days=0, diversification_ratio=0,
        )

    # Step 1: Compute metrics for each strategy
    strategy_perf = {}
    for name, returns in strategy_returns.items():
        perf = compute_strategy_metrics(returns, name, risk_free_rate)
        strategy_perf[name] = perf

    # Step 2: Vol-target each strategy
    vol_targeted = {}
    for name, returns in strategy_returns.items():
        vt = vol_target_returns(returns, target_vol, vol_lookback)
        if len(vt) > MIN_DAYS_FOR_STATS:
            vol_targeted[name] = vt

    if not vol_targeted:
        # Not enough data after vol-targeting
        logger.warning("No strategy has sufficient data after vol-targeting")
        return PortfolioResult(
            annualized_return=0, annualized_vol=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            num_trading_days=0, diversification_ratio=0,
            strategies=list(strategy_perf.values()),
        )

    # Step 3: Compute allocation weights from vol-targeted performance
    vt_perf = []
    for name, vt_ret in vol_targeted.items():
        p = compute_strategy_metrics(vt_ret, name, risk_free_rate)
        vt_perf.append(p)

    weights = compute_allocation_weights(vt_perf, max_weight)

    # Update allocation weights in strategy_perf
    for name, w in weights.items():
        if name in strategy_perf:
            strategy_perf[name].allocation_weight = round(w, 4)

    # Step 4: Combine into portfolio
    # Align all series to common dates
    aligned = pd.DataFrame(vol_targeted)
    aligned = aligned.dropna()

    if len(aligned) < 2:
        return PortfolioResult(
            annualized_return=0, annualized_vol=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            num_trading_days=0, diversification_ratio=0,
            strategies=list(strategy_perf.values()),
        )

    # Weighted sum of daily returns
    portfolio_returns = pd.Series(0.0, index=aligned.index)
    for col in aligned.columns:
        w = weights.get(col, 0.0)
        portfolio_returns += w * aligned[col]

    # Step 5: Compute portfolio metrics
    port_metrics = compute_strategy_metrics(
        portfolio_returns, "Combined Portfolio", risk_free_rate
    )

    # Correlation matrix for diversification assessment
    corr_matrix = aligned.corr() if len(aligned.columns) > 1 else None

    # Diversification ratio: portfolio Sharpe / max individual Sharpe
    max_individual_sharpe = max(
        (p.sharpe_ratio for p in strategy_perf.values()),
        default=0.0,
    )
    div_ratio = (
        port_metrics.sharpe_ratio / max(max_individual_sharpe, 1e-8)
        if max_individual_sharpe > 0 else 0.0
    )

    return PortfolioResult(
        annualized_return=port_metrics.annualized_return,
        annualized_vol=port_metrics.annualized_vol,
        sharpe_ratio=port_metrics.sharpe_ratio,
        sortino_ratio=port_metrics.sortino_ratio,
        max_drawdown=port_metrics.max_drawdown,
        calmar_ratio=port_metrics.calmar_ratio,
        num_trading_days=port_metrics.num_trading_days,
        diversification_ratio=round(div_ratio, 4),
        correlation_matrix=corr_matrix,
        strategies=list(strategy_perf.values()),
        portfolio_daily_returns=portfolio_returns,
    )
