"""
Walk-Forward Backtester — Validation Gate
==========================================
The hard gate that decides whether a QuantSignal may be narrated.

A signal is validated ONLY when it passes the walk-forward out-of-sample
test (AGENTS.md rule 6).  An LLM finding a signal "plausible" is NEVER
sufficient.

Design:
  - Anchored walk-forward: train on [0, T], test on [T, T+Δ], slide T forward.
  - Transaction cost model: configurable brokerage + estimated market impact.
  - Metrics: Sharpe, Sortino, max drawdown, Calmar, win rate, total return.
  - Validation thresholds (conservative):
      OOS Sharpe ≥ 1.0
      Max drawdown ≤ 25%
      At least 504 OOS trading days (~2 years)
  - Return-target discipline (AGENTS.md rule 8): we optimize for Sharpe/Sortino
    with bounded max-drawdown, NEVER for a fixed target return percentage.

Reference: docs/ARTHA_ARCHITECTURE.md §5.4
"""

from __future__ import annotations

import logging
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Validation thresholds ───────────────────────────────────────────────────────
# Conservative by design — loosening these is a bug, not a feature (rule 6).

MIN_OOS_SHARPE = 1.0
MAX_OOS_DRAWDOWN = 0.25  # 25%
MIN_OOS_TRADING_DAYS = 504  # ~2 years

# Default transaction costs
DEFAULT_BROKERAGE_BPS = 5  # 0.05% per trade
DEFAULT_IMPACT_BPS = 10    # 0.10% estimated market impact


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest period."""
    total_return: float        # Cumulative return (decimal, not %)
    annualized_return: float   # Annualized return (decimal)
    sharpe_ratio: float        # Annualized Sharpe (excess return / vol)
    sortino_ratio: float       # Using downside deviation
    max_drawdown: float        # Maximum peak-to-trough decline (positive number)
    calmar_ratio: float        # Annualized return / max drawdown
    win_rate: float            # Fraction of positive-return trades
    num_trades: int            # Total number of trades
    num_trading_days: int      # OOS trading days
    volatility: float          # Annualized volatility
    avg_trade_return: float    # Average per-trade return
    profit_factor: float       # Gross profit / gross loss
    is_valid: bool = False     # Set True only by validate()

    def validate(self) -> bool:
        """
        Check if the backtest passes the validation gate.

        AGENTS.md Rule 6: Nothing reaches a report unvalidated.
        AGENTS.md Rule 8: No fixed return targets — Sharpe + drawdown only.
        """
        self.is_valid = bool(
            self.sharpe_ratio >= MIN_OOS_SHARPE
            and self.max_drawdown <= MAX_OOS_DRAWDOWN
            and self.num_trading_days >= MIN_OOS_TRADING_DAYS
        )
        return self.is_valid


@dataclass
class WalkForwardResult:
    """Result of a complete walk-forward backtest."""
    strategy_name: str
    in_sample_metrics: BacktestMetrics
    out_of_sample_metrics: BacktestMetrics
    fold_results: list[BacktestMetrics] = field(default_factory=list)
    fold_boundaries: list[dict] = field(default_factory=list)
    failed_folds: list[dict] = field(default_factory=list)
    expected_oos_days: int = 0
    is_validated: bool = False  # Overall validation flag

    @property
    def validation_id(self) -> str:
        """Return a stable ID for the exact validation result."""
        payload = {
            "strategy_name": self.strategy_name,
            "oos": self.out_of_sample_metrics.__dict__,
            "folds": [fold.__dict__ for fold in self.fold_results],
            "fold_boundaries": self.fold_boundaries,
            "failed_folds": self.failed_folds,
            "expected_oos_days": self.expected_oos_days,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return f"wf-{digest}"

    def evidence(self) -> dict:
        """Return serialisable evidence suitable for a Fact Store signal."""
        return {
            "validation_method": "walk_forward_oos",
            "validation_id": self.validation_id,
            "validation_oos_days": self.out_of_sample_metrics.num_trading_days,
            "backtest_sharpe": self.out_of_sample_metrics.sharpe_ratio,
            "backtest_sortino": self.out_of_sample_metrics.sortino_ratio,
            "backtest_max_drawdown": self.out_of_sample_metrics.max_drawdown,
            "validated": self.is_validated,
            "fold_boundaries": self.fold_boundaries,
            "failed_folds": self.failed_folds,
            "coverage_complete": self.coverage_complete,
        }

    @property
    def coverage_complete(self) -> bool:
        """Whether every requested OOS observation was evaluated exactly once."""
        return (
            not self.failed_folds
            and (
                self.expected_oos_days == 0
                or self.out_of_sample_metrics.num_trading_days == self.expected_oos_days
            )
        )

    def validate(self) -> bool:
        """Validate based on OOS metrics."""
        self.is_validated = self.coverage_complete and self.out_of_sample_metrics.validate()
        return self.is_validated


# ── Core Metrics Computation ────────────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    trades: Optional[pd.Series] = None,
    risk_free_rate: float = 0.06,  # India 10Y ~6%
) -> BacktestMetrics:
    """
    Compute comprehensive backtest metrics from a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (not cumulative), indexed by date.
    trades : pd.Series, optional
        Per-trade returns for win-rate and trade-level stats.
        If None, each day is treated as a "trade".
    risk_free_rate : float
        Annual risk-free rate for Sharpe computation.
    """
    returns = returns.dropna()

    if returns.empty:
        return BacktestMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, num_trades=0, num_trading_days=0,
            volatility=0, avg_trade_return=0, profit_factor=0,
        )

    n_days = len(returns)

    # ── Cumulative return ───────────────────────────────────────────
    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)

    # ── Annualized return ───────────────────────────────────────────
    years = n_days / 252
    if years > 0 and cumulative.iloc[-1] > 0:
        ann_return = float(cumulative.iloc[-1] ** (1 / years) - 1)
    else:
        ann_return = 0.0

    # ── Volatility ──────────────────────────────────────────────────
    daily_vol = float(returns.std())
    ann_vol = daily_vol * np.sqrt(252)

    # ── Sharpe ratio ────────────────────────────────────────────────
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf

    if ann_vol > 1e-10:
        sharpe = float(excess_returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # ── Sortino ratio (downside deviation) ──────────────────────────
    downside = excess_returns[excess_returns < 0]
    if len(downside) > 1:
        downside_std = float(downside.std()) * np.sqrt(252)
        sortino = float(excess_returns.mean() * 252 / downside_std) if downside_std > 1e-10 else 0.0
    else:
        # No downside returns (all positive or single observation)
        # → Sortino is at least as good as Sharpe
        sortino = max(sharpe, 0.0) if sharpe != 0 else 0.0

    # ── Max drawdown ────────────────────────────────────────────────
    cummax = cumulative.cummax()
    drawdown = (cumulative - cummax) / cummax
    max_dd = float(abs(drawdown.min()))

    # ── Calmar ratio ────────────────────────────────────────────────
    calmar = float(ann_return / max_dd) if max_dd > 1e-10 else 0.0

    # ── Trade-level stats ───────────────────────────────────────────
    trade_returns = trades if trades is not None else returns
    trade_returns = trade_returns.dropna()

    n_trades = len(trade_returns)
    wins = (trade_returns > 0).sum()
    win_rate = float(wins / n_trades) if n_trades > 0 else 0.0
    avg_trade = float(trade_returns.mean()) if n_trades > 0 else 0.0

    # Profit factor = sum(positive returns) / abs(sum(negative returns))
    gross_profit = float(trade_returns[trade_returns > 0].sum())
    gross_loss = float(abs(trade_returns[trade_returns < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    return BacktestMetrics(
        total_return=round(total_return, 6),
        annualized_return=round(ann_return, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 6),
        calmar_ratio=round(calmar, 4),
        win_rate=round(win_rate, 4),
        num_trades=n_trades,
        num_trading_days=n_days,
        volatility=round(ann_vol, 6),
        avg_trade_return=round(avg_trade, 8),
        profit_factor=round(profit_factor, 4),
    )


# ── Transaction Cost Model ──────────────────────────────────────────────────────

def apply_transaction_costs(
    returns: pd.Series,
    signals: pd.Series,
    brokerage_bps: float = DEFAULT_BROKERAGE_BPS,
    impact_bps: float = DEFAULT_IMPACT_BPS,
) -> pd.Series:
    """
    Apply transaction costs to a return series based on signal changes.

    A cost is incurred whenever the position changes (entry, exit, or reversal).

    Parameters
    ----------
    returns : pd.Series
        Raw strategy returns.
    signals : pd.Series
        Position signals (e.g., +1, -1, 0). Costs are applied on changes.
    brokerage_bps : float
        Brokerage cost in basis points per trade.
    impact_bps : float
        Estimated market impact in basis points per trade.
    """
    total_cost_per_trade = (brokerage_bps + impact_bps) / 10_000

    # Detect position changes
    position_changes = signals.diff().abs()
    position_changes.iloc[0] = abs(signals.iloc[0])  # Initial entry

    # Scale cost: a full entry/exit costs total_cost, a reversal costs 2x
    costs = position_changes * total_cost_per_trade

    return returns - costs


# ── Walk-Forward Engine ─────────────────────────────────────────────────────────

def _generate_signals(signal_generator, train_data: pd.Series, target_index: pd.Index) -> pd.Series:
    """Invoke a signal generator without masking its internal exceptions."""
    parameters = inspect.signature(signal_generator).parameters
    oos_parameter = parameters.get("oos_index")
    if oos_parameter is not None and oos_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
        return signal_generator(train_data, oos_index=target_index)
    accepts_second_argument = (
        len(parameters) >= 2
        or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters.values())
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    )
    return (
        signal_generator(train_data, target_index)
        if accepts_second_argument else signal_generator(train_data)
    )

def walk_forward_backtest(
    returns: pd.Series,
    signal_generator,
    strategy_name: str = "unnamed",
    n_folds: int = 5,
    train_ratio: float = 0.6,
    risk_free_rate: float = 0.06,
    brokerage_bps: float = DEFAULT_BROKERAGE_BPS,
    impact_bps: float = DEFAULT_IMPACT_BPS,
) -> WalkForwardResult:
    """
    Anchored walk-forward backtest.

    For each fold:
      - Train on [0, T]
      - Generate signals on [T, T+Δ]
      - Compute OOS metrics

    Parameters
    ----------
    returns : pd.Series
        Daily asset returns indexed by date.
    signal_generator : callable
        Function ``(train_returns, oos_index) -> pd.Series`` of signals. The
        returned signals must have exactly the requested OOS index. A legacy
        one-argument callable is accepted only when it already returns the
        exact OOS index; IS-period signals are never forward-filled.
    strategy_name : str
        Label for the strategy.
    n_folds : int
        Number of walk-forward folds.
    train_ratio : float
        Fraction of total data used for initial training.
    risk_free_rate : float
        Risk-free rate for Sharpe computation.
    brokerage_bps, impact_bps : float
        Costs applied on signal changes inside every OOS fold.

    Returns
    -------
    WalkForwardResult
        Contains IS metrics, OOS metrics, and per-fold results.
    """
    returns = returns.dropna()
    n = len(returns)

    if n < 100:
        logger.warning("Insufficient data for walk-forward: %d observations", n)
        empty = BacktestMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, num_trades=0, num_trading_days=0,
            volatility=0, avg_trade_return=0, profit_factor=0,
        )
        return WalkForwardResult(
            strategy_name=strategy_name,
            in_sample_metrics=empty,
            out_of_sample_metrics=empty,
        )

    # Determine fold boundaries. Integer division previously discarded the
    # remainder of the OOS sample; these boundaries account for every date.
    initial_train = int(n * train_ratio)
    total_oos = n - initial_train
    oos_size = total_oos // n_folds

    if oos_size < 20:
        # Not enough data for meaningful folds — use a single IS/OOS split
        n_folds = 1
        oos_size = n - initial_train

    fold_results: list[BacktestMetrics] = []
    fold_boundaries: list[dict] = []
    failed_folds: list[dict] = []
    all_oos_returns: list[pd.Series] = []

    for fold in range(n_folds):
        train_end = initial_train + (total_oos * fold) // n_folds
        oos_start = train_end
        oos_end = initial_train + (total_oos * (fold + 1)) // n_folds

        if oos_start >= n or oos_end <= oos_start:
            break

        train_data = returns.iloc[:train_end]
        oos_data = returns.iloc[oos_start:oos_end]
        boundary = {
            "fold": fold,
            "train_start": str(train_data.index[0]),
            "train_end": str(train_data.index[-1]),
            "oos_start": str(oos_data.index[0]),
            "oos_end": str(oos_data.index[-1]),
            "oos_days": len(oos_data),
        }
        fold_boundaries.append(boundary)

        try:
            # Generate signals using training data. The strict index check is
            # deliberate: an IS signal must never be forward-filled into OOS.
            signals = _generate_signals(signal_generator, train_data, oos_data.index)
            if not isinstance(signals, pd.Series):
                raise TypeError("signal_generator must return a pandas Series")
            if not signals.index.equals(oos_data.index):
                raise ValueError(
                    "signal_generator must return exactly the requested OOS index"
                )
            if signals.isna().any():
                raise ValueError("signal_generator returned NaN OOS signals")

            gross_oos_returns = oos_data * pd.to_numeric(signals, errors="raise")
            oos_returns = apply_transaction_costs(
                gross_oos_returns,
                pd.to_numeric(signals, errors="raise"),
                brokerage_bps=brokerage_bps,
                impact_bps=impact_bps,
            )

            fold_metrics = compute_metrics(oos_returns, risk_free_rate=risk_free_rate)
            fold_results.append(fold_metrics)
            all_oos_returns.append(oos_returns)

        except Exception as e:
            logger.error("Walk-forward fold %d failed: %s", fold, e)
            failed_folds.append({**boundary, "error": str(e)})

    # Aggregate OOS metrics
    if all_oos_returns:
        combined_oos = pd.concat(all_oos_returns)
        oos_metrics = compute_metrics(combined_oos, risk_free_rate=risk_free_rate)
    else:
        oos_metrics = BacktestMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, num_trades=0, num_trading_days=0,
            volatility=0, avg_trade_return=0, profit_factor=0,
        )

    # In-sample metrics use the same strategy contract when the generator can
    # produce an exact training-index signal. If it cannot, leave IS metrics
    # empty rather than reporting raw asset returns as strategy performance.
    is_data = returns.iloc[:initial_train]
    try:
        is_signals = _generate_signals(signal_generator, is_data, is_data.index)
        if not isinstance(is_signals, pd.Series) or not is_signals.index.equals(is_data.index):
            raise ValueError("in-sample signal index is incomplete")
        is_returns = apply_transaction_costs(
            is_data * pd.to_numeric(is_signals, errors="raise"),
            pd.to_numeric(is_signals, errors="raise"),
            brokerage_bps=brokerage_bps,
            impact_bps=impact_bps,
        )
        is_metrics = compute_metrics(is_returns, risk_free_rate=risk_free_rate)
    except Exception as e:
        logger.warning("In-sample strategy metrics unavailable: %s", e)
        is_metrics = compute_metrics(pd.Series(dtype=float), risk_free_rate=risk_free_rate)

    result = WalkForwardResult(
        strategy_name=strategy_name,
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        fold_results=fold_results,
        fold_boundaries=fold_boundaries,
        failed_folds=failed_folds,
        expected_oos_days=total_oos,
    )

    # Validate
    result.validate()

    logger.info(
        "Walk-forward %s: IS Sharpe=%.2f, OOS Sharpe=%.2f, OOS MaxDD=%.1f%%, Validated=%s",
        strategy_name,
        is_metrics.sharpe_ratio,
        oos_metrics.sharpe_ratio,
        oos_metrics.max_drawdown * 100,
        result.is_validated,
    )

    return result
