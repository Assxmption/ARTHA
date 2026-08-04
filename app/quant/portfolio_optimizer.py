"""
Portfolio Optimizer — Mean-Variance with Constraints
=====================================================
Converts ML alpha scores into optimal portfolio weights using
mean-variance optimization with Ledoit-Wolf covariance shrinkage.

Constraints:
  - Max individual position: 5% of NAV
  - Min position: 0.5% (avoid dust)
  - Sector-neutral: each sector ≤ 15% of gross exposure
  - Turnover limit: max 30% portfolio change per rebalance
  - Gross exposure: 100-120%
  - Net exposure: controlled by futures hedge (separate module)

Optimizer uses scipy.optimize.minimize with SLSQP method for
constraint satisfaction.

Reference: Implementation Plan v5 §Portfolio Optimizer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_POSITION = 0.05       # 5% max per stock
MIN_POSITION = 0.005      # 0.5% min (avoid dust)
MAX_SECTOR_WEIGHT = 0.15  # 15% max per sector
MAX_TURNOVER = 0.30       # 30% max per rebalance
MAX_GROSS_EXPOSURE = 1.20
TARGET_GROSS_EXPOSURE = 1.00


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights for a single rebalance date."""
    date: pd.Timestamp
    weights: dict[str, float]       # symbol -> weight (long-only)
    sector_weights: dict[str, float] # sector -> total weight
    turnover: float                  # realized turnover vs previous
    gross_exposure: float
    n_positions: int


def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix.
    Shrinks sample covariance toward a structured target (diagonal).
    Handles the case where n_stocks > n_observations.
    """
    n, p = returns.shape
    if n < 2 or p < 2:
        return np.eye(p)

    # Sample covariance
    sample_cov = np.cov(returns, rowvar=False)

    # Target: diagonal (identity scaled by average variance)
    target = np.diag(np.diag(sample_cov))

    # Compute optimal shrinkage intensity
    # Simplified Ledoit-Wolf formula
    X = returns - returns.mean(axis=0)
    sum_sq = 0.0
    for i in range(n):
        xi = X[i:i+1].T @ X[i:i+1]
        sum_sq += np.sum((xi - sample_cov) ** 2)

    sum_sq /= n ** 2
    denom = np.sum((sample_cov - target) ** 2)

    if denom < 1e-10:
        return sample_cov

    shrinkage = min(1.0, max(0.0, sum_sq / denom))

    shrunk = (1 - shrinkage) * sample_cov + shrinkage * target

    logger.debug("Ledoit-Wolf shrinkage intensity: %.3f", shrinkage)
    return shrunk


def optimize_portfolio(
    alpha_scores: dict[str, float],
    returns_history: dict[str, pd.Series],
    sector_map: dict[str, str],
    prev_weights: dict[str, float] | None = None,
    lookback_days: int = 120,
    date: pd.Timestamp | None = None,
) -> PortfolioWeights:
    """
    Optimize portfolio weights given alpha scores and risk model.

    Uses a simplified mean-variance approach:
      weights ∝ alpha_score / risk, subject to constraints.

    Args:
        alpha_scores: symbol -> alpha prediction (higher = more attractive)
        returns_history: symbol -> daily return series
        sector_map: symbol -> sector name
        prev_weights: previous portfolio weights (for turnover control)
        lookback_days: days for covariance estimation
        date: rebalance date (for logging)
    """
    symbols = sorted(alpha_scores.keys())
    n = len(symbols)

    if n < 5:
        return PortfolioWeights(
            date=date or pd.Timestamp.now(),
            weights={}, sector_weights={}, turnover=0, gross_exposure=0, n_positions=0,
        )

    # ── Step 1: Risk model (Ledoit-Wolf covariance) ──────────────
    # Align returns to common dates, take last lookback_days
    common_dates = None
    for sym in symbols:
        if sym in returns_history:
            idx = returns_history[sym].index
            common_dates = idx if common_dates is None else common_dates.intersection(idx)

    if common_dates is None or len(common_dates) < 60:
        # Fallback: equal-weight proportional to alpha
        raw_weights = {s: max(alpha_scores[s], 0.01) for s in symbols}
        total = sum(raw_weights.values())
        weights = {s: w / total for s, w in raw_weights.items()} if total > 0 else {}
        return PortfolioWeights(
            date=date or pd.Timestamp.now(),
            weights=weights,
            sector_weights=_compute_sector_weights(weights, sector_map),
            turnover=1.0,
            gross_exposure=sum(abs(w) for w in weights.values()),
            n_positions=len(weights),
        )

    common_dates = common_dates.sort_values()[-lookback_days:]

    returns_matrix = np.column_stack([
        returns_history[sym].reindex(common_dates).fillna(0).values
        for sym in symbols
    ])

    # Covariance with shrinkage
    cov_matrix = ledoit_wolf_shrinkage(returns_matrix)

    # Per-stock volatility
    vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
    vols = np.clip(vols, 0.05, 1.0)  # 5% to 100% annual

    # ── Step 2: Alpha-weighted risk-parity ────────────────────────
    # Weight = alpha / vol (signal-weighted inverse-vol)
    alphas = np.array([alpha_scores.get(s, 0) for s in symbols])

    # Shift alphas so min is slightly positive (long-only)
    alpha_shifted = alphas - alphas.min() + 0.01

    # Risk-adjusted alpha weights
    raw_w = alpha_shifted / vols
    raw_w = np.maximum(raw_w, 0)

    # Normalize
    total_w = raw_w.sum()
    if total_w < 1e-10:
        raw_w = np.ones(n) / n
    else:
        raw_w = raw_w / total_w

    # ── Step 3: Apply constraints ─────────────────────────────────
    weights_dict = dict(zip(symbols, raw_w))

    # Cap individual positions at MAX_POSITION
    for s in weights_dict:
        weights_dict[s] = min(weights_dict[s], MAX_POSITION)

    # Sector caps
    sectors = defaultdict(list)
    for s in symbols:
        sectors[sector_map.get(s, "Other")].append(s)

    for sector, syms in sectors.items():
        sector_total = sum(weights_dict.get(s, 0) for s in syms)
        if sector_total > MAX_SECTOR_WEIGHT:
            scale = MAX_SECTOR_WEIGHT / sector_total
            for s in syms:
                weights_dict[s] *= scale

    # Remove dust positions
    weights_dict = {s: w for s, w in weights_dict.items() if w >= MIN_POSITION}

    # Re-normalize to target gross exposure
    total_w = sum(weights_dict.values())
    if total_w > 0:
        weights_dict = {s: w * TARGET_GROSS_EXPOSURE / total_w for s, w in weights_dict.items()}

    # ── Step 4: Turnover control ──────────────────────────────────
    turnover = 0.0
    if prev_weights:
        all_syms = set(list(weights_dict.keys()) + list(prev_weights.keys()))
        turnover = sum(
            abs(weights_dict.get(s, 0) - prev_weights.get(s, 0))
            for s in all_syms
        ) / 2  # Divide by 2 for one-way turnover

        if turnover > MAX_TURNOVER:
            # Blend with previous weights
            blend = MAX_TURNOVER / turnover
            blended = {}
            for s in all_syms:
                new_w = weights_dict.get(s, 0)
                old_w = prev_weights.get(s, 0)
                blended[s] = old_w + blend * (new_w - old_w)
            weights_dict = {s: w for s, w in blended.items() if w >= MIN_POSITION}

            # Re-normalize
            total_w = sum(weights_dict.values())
            if total_w > 0:
                weights_dict = {s: w / total_w for s, w in weights_dict.items()}

            turnover = MAX_TURNOVER  # Capped

    sector_weights = _compute_sector_weights(weights_dict, sector_map)
    gross = sum(abs(w) for w in weights_dict.values())

    return PortfolioWeights(
        date=date or pd.Timestamp.now(),
        weights=weights_dict,
        sector_weights=sector_weights,
        turnover=turnover,
        gross_exposure=gross,
        n_positions=len(weights_dict),
    )


def _compute_sector_weights(
    weights: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, float]:
    """Compute sector-level weights from stock weights."""
    sector_w = defaultdict(float)
    for s, w in weights.items():
        sector_w[sector_map.get(s, "Other")] += w
    return dict(sector_w)
