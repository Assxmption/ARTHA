"""
PCA-Based Statistical Arbitrage
================================
Cross-sectional statistical arbitrage using Principal Component Analysis
to extract idiosyncratic returns and trade mean reversion in the residuals.

Algorithm:
  1. Rolling PCA on the cross-section of stock returns extracts the top-k
     principal components (systematic risk factors).
  2. Each stock's returns are regressed on these PCs. The residual is the
     stock's idiosyncratic return — the alpha source.
  3. Cumulative residuals are z-scored and traded: long when z < -entry_z,
     short when z > +entry_z, exit when z crosses exit_z toward zero.

This complements the existing Kalman-filter pair-wise stat-arb in statarb.py:
  - statarb.py: O(n²) pairs, time-varying betas, but limited to pair-wise
    cointegration (Engle-Granger). Disabled on Indian equities due to
    structural breaks.
  - pca_statarb.py: O(n) cross-sectional, captures multi-stock systematic
    risk in one pass, but assumes stationary factor loadings within each
    PCA window. Both are kept for diversification value.

Cost model mirrors statarb.py: 15bps round-trip (7.5bps per side, covering
brokerage + STT + exchange fees + slippage for liquid NIFTY-universe stocks).

All computation is deterministic — LLMs never touch this (AGENTS.md rule 1).

Reference: docs/ARTHA_ARCHITECTURE.md §4.3
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────────

DEFAULT_N_COMPONENTS = 5     # Top-k PCs (systematic factors)
DEFAULT_PCA_WINDOW = 252     # Rolling training window (1 year)
DEFAULT_ENTRY_Z = 1.5        # Enter when |z| > 1.5
DEFAULT_EXIT_Z = 0.3         # Exit when |z| < 0.3
DEFAULT_LOOKBACK_Z = 20      # Rolling z-score lookback
DEFAULT_TC_BPS = 15.0        # Round-trip transaction cost (bps)
DEFAULT_MAX_POSITION_PCT = 0.02  # Max 2% of capital per stock


# ── PCA Residual Computation ───────────────────────────────────────────────────


def _adaptive_n_components(n_stocks: int, requested_k: int) -> int:
    """
    Adapt the number of PCA components to the universe size.

    Rule: k = min(requested_k, n_stocks // 3, n_stocks - 1).
    With a very small universe, extracting too many PCs leaves no
    idiosyncratic signal in the residual.
    """
    max_k = max(n_stocks // 3, 1)
    k = min(requested_k, max_k, n_stocks - 1)
    if k != requested_k:
        logger.info(
            "Adapted n_components: %d → %d (universe has %d stocks)",
            requested_k, k, n_stocks,
        )
    return max(k, 1)


def compute_pca_residuals(
    returns_panel: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    window: int = DEFAULT_PCA_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling PCA residuals for each stock.

    For each rolling window of `window` days:
      1. Center the returns cross-section (subtract mean).
      2. Compute SVD (equivalent to PCA without eigendecomposition).
      3. Project returns onto the top-k PCs.
      4. Residual = actual return - projected return.

    Parameters
    ----------
    returns_panel : pd.DataFrame
        (n_dates × n_stocks) daily returns. NaNs are forward-filled then
        zero-filled within each window.
    n_components : int
        Number of principal components to extract.
    window : int
        Rolling window size for PCA estimation.

    Returns
    -------
    pd.DataFrame
        (n_dates × n_stocks) residual returns. First `window` rows are NaN.
    """
    n_dates, n_stocks = returns_panel.shape
    if n_stocks < 3:
        logger.warning("PCA needs ≥3 stocks; got %d", n_stocks)
        return pd.DataFrame(
            np.nan, index=returns_panel.index, columns=returns_panel.columns,
        )

    k = _adaptive_n_components(n_stocks, n_components)
    residuals = pd.DataFrame(
        np.nan, index=returns_panel.index, columns=returns_panel.columns,
    )

    for i in range(window, n_dates):
        # Extract window and handle NaNs
        window_data = returns_panel.iloc[i - window:i].copy()
        window_data = window_data.ffill().fillna(0.0)

        # Drop columns that are all zero (dead stocks in this window)
        active_cols = window_data.columns[window_data.abs().sum() > 1e-10]
        if len(active_cols) < 3:
            continue

        X = window_data[active_cols].values  # (window × n_active)

        # Center
        means = X.mean(axis=0)
        X_centered = X - means

        # SVD-based PCA (numerically stable, no covariance matrix needed)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # SVD failed (very rare) — skip this window
            logger.debug("SVD failed at index %d, skipping", i)
            continue

        # Top-k factor loadings: Vt[:k] are the PC directions
        k_eff = min(k, len(S))
        factor_loadings = Vt[:k_eff]  # (k × n_active)

        # Project the CURRENT day's return onto the factor space
        # to get the systematic component
        today_return = returns_panel.iloc[i][active_cols].fillna(0.0).values
        today_centered = today_return - means

        # Systematic component = (today_centered · V^T) · V
        projection = today_centered @ factor_loadings.T @ factor_loadings
        residual = today_centered - projection

        # Write residuals back
        for j, col in enumerate(active_cols):
            residuals.at[returns_panel.index[i], col] = residual[j]

    return residuals


# ── Z-Score Signal Generation ──────────────────────────────────────────────────


def generate_pca_signals(
    residuals: pd.DataFrame,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    lookback: int = DEFAULT_LOOKBACK_Z,
) -> pd.DataFrame:
    """
    Generate trading signals from PCA residuals using z-score mean reversion.

    Signal rules (per stock, per day):
      - Entry LONG:  z-score < -entry_z (residual is cheap → buy)
      - Entry SHORT: z-score > +entry_z (residual is rich → sell)
      - Exit: |z-score| < exit_z (residual has mean-reverted)

    The signal is +1 (long), -1 (short), or 0 (flat).

    Parameters
    ----------
    residuals : pd.DataFrame
        Residual returns from compute_pca_residuals().
    entry_z : float
        Z-score threshold for entry (absolute value).
    exit_z : float
        Z-score threshold for exit (absolute value).
    lookback : int
        Rolling window for z-score computation.

    Returns
    -------
    pd.DataFrame
        (n_dates × n_stocks) signal matrix with values in {-1, 0, +1}.
    """
    # Cumulative residuals — the "spread" that should mean-revert
    cum_residuals = residuals.cumsum()

    # Rolling z-score of cumulative residuals
    rolling_mean = cum_residuals.rolling(lookback, min_periods=max(lookback // 2, 5)).mean()
    rolling_std = cum_residuals.rolling(lookback, min_periods=max(lookback // 2, 5)).std()

    # Avoid division by zero: set std=NaN where it's too small
    rolling_std = rolling_std.where(rolling_std > 1e-10, np.nan)
    z_scores = (cum_residuals - rolling_mean) / rolling_std

    # Generate signals with hysteresis (entry/exit thresholds differ)
    signals = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)

    for col in residuals.columns:
        z = z_scores[col].values
        sig = np.zeros(len(z))
        position = 0.0

        for t in range(len(z)):
            if np.isnan(z[t]):
                sig[t] = position  # Hold current position
                continue

            if position == 0:
                # Flat — check entry
                if z[t] < -entry_z:
                    position = 1.0   # Long (residual is cheap)
                elif z[t] > entry_z:
                    position = -1.0  # Short (residual is rich)
            elif position > 0:
                # Long — check exit
                if z[t] > -exit_z:
                    position = 0.0   # Mean-reverted, exit
            elif position < 0:
                # Short — check exit
                if z[t] < exit_z:
                    position = 0.0   # Mean-reverted, exit

            sig[t] = position

        signals[col] = sig

    return signals


# ── Backtest Engine ────────────────────────────────────────────────────────────


def backtest_pca_statarb(
    prices: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    tc_bps: float = DEFAULT_TC_BPS,
    window: int = DEFAULT_PCA_WINDOW,
    lookback_z: int = DEFAULT_LOOKBACK_Z,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> pd.Series:
    """
    End-to-end PCA stat-arb backtest returning daily portfolio returns.

    Process:
      1. Compute daily returns from prices.
      2. Extract PCA residuals on rolling windows.
      3. Generate z-score trading signals.
      4. Compute daily portfolio returns with equal-weight active signals.
      5. Apply transaction costs on signal changes.

    Parameters
    ----------
    prices : pd.DataFrame
        (n_dates × n_stocks) close prices. Must have a DatetimeIndex.
    n_components : int
        Number of PCA components.
    entry_z : float
        Entry z-score threshold.
    exit_z : float
        Exit z-score threshold.
    tc_bps : float
        Round-trip transaction cost in basis points.
    window : int
        PCA estimation window.
    lookback_z : int
        Z-score computation lookback.
    max_position_pct : float
        Max portfolio weight per stock (for position sizing).

    Returns
    -------
    pd.Series
        Daily portfolio returns with DatetimeIndex.
    """
    logger.info("PCA Stat-Arb backtest: %d stocks, %d days, k=%d",
                prices.shape[1], prices.shape[0], n_components)

    # Step 1: Daily returns
    returns = prices.pct_change().iloc[1:]  # Drop first NaN row
    returns = returns.replace([np.inf, -np.inf], np.nan)

    if returns.shape[0] < window + lookback_z + 20:
        logger.warning(
            "Insufficient data for PCA stat-arb: need %d days, got %d",
            window + lookback_z + 20, returns.shape[0],
        )
        return pd.Series(0.0, index=prices.index, name="pca_statarb")

    # Step 2: PCA residuals
    residuals = compute_pca_residuals(returns, n_components, window)

    # Step 3: Trading signals
    signals = generate_pca_signals(residuals, entry_z, exit_z, lookback_z)

    # Step 4: Portfolio returns
    # Equal-weight across active signals, capped at max_position_pct
    n_stocks = signals.shape[1]
    max_weight = min(max_position_pct, 1.0 / max(n_stocks, 1))

    # Count active positions each day
    n_active = signals.abs().sum(axis=1).replace(0, np.nan)

    # Per-stock weight = min(1/n_active, max_weight)
    weight_per_stock = (1.0 / n_active).clip(upper=max_weight)

    # Daily return contribution from each stock
    # Signal × next-day return × weight (signal at close t → return t+1)
    shifted_signals = signals.shift(1)  # Lag signals by 1 day
    daily_contrib = shifted_signals.multiply(returns).multiply(
        weight_per_stock, axis=0,
    )

    portfolio_returns = daily_contrib.sum(axis=1)

    # Step 5: Transaction costs
    # TC applied on signal changes (entry + exit = round-trip)
    signal_changes = shifted_signals.diff().abs().sum(axis=1)
    tc_per_day = signal_changes * (tc_bps / 10000) * max_weight
    portfolio_returns -= tc_per_day

    portfolio_returns.name = "pca_statarb"

    # Drop warmup period
    valid_start = window + lookback_z + 1
    portfolio_returns.iloc[:valid_start] = 0.0

    # Log performance metrics
    arr = portfolio_returns.values[valid_start:]
    if len(arr) > 20 and np.std(arr) > 1e-10:
        sharpe = (np.mean(arr) / np.std(arr)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        max_dd = np.max(peak - cum)

        n_trades = int(signal_changes[valid_start:].sum())
        active_days = int((n_active.dropna() > 0).sum())

        logger.info(
            "PCA Stat-Arb results: Sharpe=%.3f, Ann.Return=%.2f%%, "
            "MaxDD=%.2f%%, Trades=%d, Active days=%d/%d",
            sharpe, ann_ret * 100, max_dd * 100,
            n_trades, active_days, len(arr),
        )
    else:
        logger.warning("PCA Stat-Arb produced no meaningful returns")

    return portfolio_returns


# ── Walk-Forward Validation ────────────────────────────────────────────────────


def walk_forward_pca_statarb(
    prices: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 126,
    n_components: int = DEFAULT_N_COMPONENTS,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    tc_bps: float = DEFAULT_TC_BPS,
    min_sharpe_oos: float = 0.5,
) -> dict:
    """
    Walk-forward validation of PCA stat-arb strategy.

    Validation gate for AGENTS.md rule 6.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices panel.
    train_days : int
        In-sample window (2 years).
    test_days : int
        Out-of-sample window (6 months).
    n_components : int
        PCA components.
    entry_z, exit_z : float
        Z-score thresholds.
    tc_bps : float
        Transaction costs.
    min_sharpe_oos : float
        Minimum OOS Sharpe for validation.

    Returns
    -------
    dict
        Walk-forward results with 'validated' flag.
    """
    total_days = len(prices)
    window = train_days + test_days

    if total_days < window:
        return {
            "validated": False,
            "reason": f"Insufficient data: need {window} days, got {total_days}",
            "n_windows": 0,
        }

    oos_sharpes = []
    is_sharpes = []
    n_windows = 0
    start_idx = 0

    while start_idx + window <= total_days:
        train_prices = prices.iloc[start_idx:start_idx + train_days]
        test_prices = prices.iloc[start_idx + train_days:start_idx + window]

        # In-sample
        is_ret = backtest_pca_statarb(
            train_prices, n_components, entry_z, exit_z, tc_bps,
        )
        is_arr = is_ret.values
        is_arr = is_arr[~np.isnan(is_arr)]
        if len(is_arr) > 20 and np.std(is_arr) > 1e-10:
            is_sharpe = (np.mean(is_arr) / np.std(is_arr)) * np.sqrt(252)
            is_sharpes.append(is_sharpe)

        # Out-of-sample: use full data up to test_end for PCA computation,
        # but only measure returns in the OOS window.
        # This avoids the warmup problem (same fix as options_backtest v2).
        full_prices = prices.iloc[start_idx:start_idx + window]
        full_ret = backtest_pca_statarb(
            full_prices, n_components, entry_z, exit_z, tc_bps,
        )
        oos_ret = full_ret.iloc[train_days:]
        oos_arr = oos_ret.values
        oos_arr = oos_arr[~np.isnan(oos_arr)]
        if len(oos_arr) > 20 and np.std(oos_arr) > 1e-10:
            oos_sharpe = (np.mean(oos_arr) / np.std(oos_arr)) * np.sqrt(252)
            oos_sharpes.append(oos_sharpe)

        n_windows += 1
        start_idx += test_days

    if n_windows == 0:
        return {
            "validated": False,
            "reason": "No valid windows produced",
            "n_windows": 0,
        }

    avg_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

    validated = (
        avg_oos_sharpe >= min_sharpe_oos
        and len(oos_sharpes) >= n_windows // 2
    )

    reason = (
        f"OOS Sharpe {avg_oos_sharpe:.3f} "
        f"{'≥' if avg_oos_sharpe >= min_sharpe_oos else '<'} {min_sharpe_oos} "
        f"across {n_windows} windows ({len(oos_sharpes)} with trades). "
        f"IS Sharpe: {avg_is_sharpe:.3f} ({len(is_sharpes)} IS windows)"
    )

    if not validated and is_sharpes and oos_sharpes:
        if avg_is_sharpe > min_sharpe_oos * 2 and avg_oos_sharpe < min_sharpe_oos:
            reason += " — LIKELY OVERFIT (IS >> OOS)"

    logger.info("PCA Stat-Arb walk-forward: %s", reason)

    return {
        "validated": validated,
        "reason": reason,
        "n_windows": n_windows,
        "in_sample_sharpe": round(avg_is_sharpe, 4),
        "out_of_sample_sharpe": round(avg_oos_sharpe, 4),
    }
