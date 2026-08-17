"""
Options Strategy Backtester — v2 (rebuilt)
============================================
Walk-forward backtesting engine for options strategies using synthetic
pricing from historical OHLCV data.

Since we don't have historical option chain data (real intraday chains
are expensive and not freely available for NSE), we reconstruct option
prices using:
  1. Historical underlying close prices (daily)
  2. India VIX as a proxy for implied volatility
  3. Black-Scholes pricing with the reconstructed IV
  4. Yang-Zhang realized vol for VRP computation

v2 changes (fixing bugs from v1):
  - CRITICAL: Pre-compute all signals on the FULL series, then slice.
    v1 had warmup=60 inside backtest_strategy, which meant 63-day OOS
    windows had only 3 tradeable days → 0 trades → 0 Sharpe → false
    failure on every OOS window.
  - Signal filtering for SPECIFIC strategy tests: v1 deployed iron
    condors on every entry_interval day regardless of regime/VRP.
    v2 applies regime-appropriate entry conditions even when testing
    a specific strategy type.
  - Walk-forward uses a single continuous backtest with
    date-range-based IS/OOS splitting, not isolated sub-backtests.

Walk-forward protocol:
  - Training window: 252 trading days (1 year)
  - Out-of-sample window: 63 trading days (1 quarter)
  - Strategy must achieve Sharpe > 0.5 OOS to be validated
  - All results reported honestly (AGENTS.md rule 8)

Capital context: ₹5 crore, but backtester works with any capital.
All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.options_pricing import (
    bs_price, get_lot_size, DEFAULT_RISK_FREE_RATE,
)
from app.quant.options_signals import (
    compute_realized_vol, classify_vix_regime,
)
from app.quant.options_strategies import StrategyType, StrategyMetrics
from app.quant.strategy_selector import (
    select_strategies, build_strategy_for_symbol,
)

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────────

DEFAULT_CAPITAL = 5_00_00_000.0    # ₹5 crore
OPTIONS_ALLOCATION_PCT = 0.30      # 30% of capital for options strategies
VRP_PROXY_RATIO = 1.2              # IV ≈ 1.2 × RV when VIX unavailable

WALK_FORWARD_TRAIN = 252           # 1 year in-sample
WALK_FORWARD_TEST = 63             # 1 quarter out-of-sample
MIN_SHARPE_OOS = 0.5               # OOS Sharpe threshold for validation
BROKERAGE_PER_LOT = 40.0           # ₹40 per lot per side (flat fee model)
STT_ON_SELL_PCT = 0.0625 / 100     # STT on sell premium (0.0625%)
THETA_DECAY_ADJ = 1.0              # Theta realization factor (1.0 = full BS theta)

# ── Signal-based entry filters ──────────────────────────────────────────────────
# These prevent blind mechanical entries and are the core fix for the v1 bug
# where strategies were deployed regardless of market conditions.

# Strategy → which regimes it's allowed to enter in
STRATEGY_REGIME_FILTER: dict[StrategyType, set[str]] = {
    StrategyType.IRON_CONDOR:    {"SIDEWAYS", "UNKNOWN"},
    StrategyType.BULL_PUT_SPREAD: {"BULL", "SIDEWAYS", "UNKNOWN"},
    StrategyType.BEAR_CALL_SPREAD: {"BEAR", "SIDEWAYS", "UNKNOWN"},
    StrategyType.SHORT_STRANGLE: {"SIDEWAYS"},
    StrategyType.COVERED_CALL:   {"BULL", "SIDEWAYS", "UNKNOWN"},
    StrategyType.PROTECTIVE_PUT: {"BEAR", "UNKNOWN"},
    StrategyType.COLLAR:         {"BULL", "SIDEWAYS", "BEAR", "UNKNOWN"},
    StrategyType.CALENDAR_SPREAD: {"SIDEWAYS", "UNKNOWN"},
    StrategyType.SYNTHETIC_SHORT: {"BEAR"},
    StrategyType.RATIO_SPREAD:   {"SIDEWAYS", "BULL", "UNKNOWN"},
}

# For premium-selling strategies, require minimum VRP (don't sell cheap vol)
PREMIUM_SELLING_STRATEGIES = {
    StrategyType.IRON_CONDOR,
    StrategyType.SHORT_STRANGLE,
    StrategyType.BULL_PUT_SPREAD,
    StrategyType.BEAR_CALL_SPREAD,
    StrategyType.COVERED_CALL,
    StrategyType.RATIO_SPREAD,
}
MIN_VRP_FOR_SELLING = 0.0          # IV must be ≥ RV (VRP ≥ 0) to sell premium
MIN_IV_PERCENTILE_FOR_SELLING = 25  # At least 25th percentile (not rock-bottom IV)


# ── Data Classes ────────────────────────────────────────────────────────────────


@dataclass
class OptionsTradeRecord:
    """A single options strategy trade event."""
    date: str
    strategy_type: str
    symbol: str
    action: str              # "OPEN" | "CLOSE" | "EXPIRE"
    net_premium: float       # Credit received / debit paid
    max_profit: float
    max_loss: float
    margin_used: float
    realized_pnl: float = 0.0
    days_held: int = 0
    regime: str = "UNKNOWN"
    vix_level: float = 0.0
    vrp_at_entry: float = 0.0


@dataclass
class OptionsBacktestResult:
    """Complete output of an options strategy backtest."""
    strategy_type: str
    n_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    avg_days_held: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_margin_used: float = 0.0
    capital_efficiency: float = 0.0   # return / avg margin
    trades: list[OptionsTradeRecord] = field(default_factory=list)
    daily_pnl: list[float] = field(default_factory=list)
    is_oos: bool = False
    validated: bool = False


@dataclass
class WalkForwardResult:
    """Walk-forward validation result."""
    strategy_type: str
    n_windows: int = 0
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    in_sample_return_pct: float = 0.0
    out_of_sample_return_pct: float = 0.0
    in_sample_max_dd: float = 0.0
    out_of_sample_max_dd: float = 0.0
    validated: bool = False
    reason: str = ""
    window_results: list[OptionsBacktestResult] = field(default_factory=list)


# ── Pre-computed Signal Store ───────────────────────────────────────────────────


@dataclass
class DailySignals:
    """Pre-computed signals for a single day. Avoids recomputation in loops."""
    iv: float
    rv: float
    vrp: float
    iv_percentile: float
    vix_regime: str
    regime: str


def precompute_signals(
    close_prices: pd.Series,
    vix_series: Optional[pd.Series] = None,
    regime_series: Optional[pd.Series] = None,
    rv_window: int = 20,
    iv_lookback: int = 252,
) -> dict[pd.Timestamp, DailySignals]:
    """
    Pre-compute all signals across the full date range ONCE.

    This is the critical fix: v1 recomputed signals per window, which
    meant OOS windows (63 days) had insufficient data for warmup (60 days)
    → 0 tradeable days → 0 trades → false Sharpe=0.

    Now we compute on the full series and look up by date.
    """
    rv_series = compute_realized_vol(close_prices, window=rv_window)

    # Reconstruct IV
    if vix_series is not None and len(vix_series) > 0:
        iv_series = vix_series.reindex(close_prices.index, method="ffill") / 100.0
        iv_series = iv_series.fillna(rv_series * VRP_PROXY_RATIO)
    else:
        iv_series = rv_series * VRP_PROXY_RATIO
    iv_series = iv_series.clip(0.08, 1.0)

    # Compute VIX regime at each point
    dates = close_prices.index.sort_values()
    signals: dict[pd.Timestamp, DailySignals] = {}

    for i, dt in enumerate(dates):
        iv_val = float(iv_series.iloc[i]) if not np.isnan(iv_series.iloc[i]) else 0.18
        rv_val = float(rv_series.iloc[i]) if i < len(rv_series) and not np.isnan(rv_series.iloc[i]) else 0.15

        # VRP
        vrp = (iv_val - rv_val) / max(rv_val, 0.01) if rv_val > 0.001 else 0.0

        # IV percentile
        iv_pctile = 50.0
        if i >= iv_lookback:
            iv_hist = iv_series.iloc[i - iv_lookback:i].dropna()
            if len(iv_hist) >= 30:
                iv_pctile = float((iv_hist < iv_val).sum() / len(iv_hist) * 100)

        # VIX regime
        vix_regime = "MEDIUM"
        if vix_series is not None and len(vix_series) > 0:
            vix_up_to_now = vix_series.loc[:dt].dropna()
            if len(vix_up_to_now) >= 20:
                current_vix = float(vix_up_to_now.iloc[-1])
                vix_regime = classify_vix_regime(current_vix, vix_up_to_now)

        # Market regime
        regime = "UNKNOWN"
        if regime_series is not None and dt in regime_series.index:
            r = regime_series[dt]
            regime = r.value if hasattr(r, "value") else str(r)

        signals[dt] = DailySignals(
            iv=iv_val, rv=rv_val, vrp=vrp,
            iv_percentile=iv_pctile, vix_regime=vix_regime,
            regime=regime,
        )

    return signals


# ── Full BS Repricing (replaces broken Greek approximation) ─────────────────────


def mark_to_market_strategy(
    strategy: StrategyMetrics,
    current_spot: float,
    current_iv: float,
    remaining_dte: int,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """
    Mark-to-market a multi-leg strategy by repricing every leg with BS.

    This replaces the v2 Greek approximation which had THREE bugs:
      1. IV change applied cumulatively every day (20× overcount)
      2. Static entry-time Greeks (delta/gamma/theta/vega never updated)
      3. No convexity (gamma) in the P&L model

    Now: value_today = Σ (sign × BS_price(leg, spot, iv, dte)) × lot_size
    Daily P&L = value_today - value_yesterday
    """
    T = max(remaining_dte / 365.0, 1e-6)
    iv = max(current_iv, 0.01)
    total_value = 0.0

    for leg in strategy.legs:
        if not leg.is_option:
            # Stock leg
            sign = 1.0 if leg.is_long else -1.0
            total_value += sign * abs(leg.quantity) * current_spot
            continue

        opt_type = "call" if leg.is_call else "put"
        price = float(bs_price(current_spot, leg.strike, T, r, iv, opt_type))
        sign = 1.0 if leg.is_long else -1.0
        total_value += sign * abs(leg.quantity) * price * strategy.lot_size

    return total_value


def _max_risk_lots(
    max_loss_per_lot: float,
    options_capital: float,
    risk_per_trade_pct: float = 0.02,
) -> int:
    """
    Compute max lots such that worst-case loss ≤ risk_per_trade_pct of capital.

    Default: risk 2% of options capital per trade.
    This prevents any single trade from creating outsized drawdowns.

    Trade-off: lower risk_per_trade_pct = more trades needed for same return,
    but max drawdown is bounded. This is the correct approach per AGENTS.md
    rule 8 (optimize Sharpe/Sortino with bounded max-drawdown).
    """
    if max_loss_per_lot <= 0:
        return 1
    max_risk = options_capital * risk_per_trade_pct
    lots = int(max_risk / max_loss_per_lot)
    return max(lots, 1)


# Keep for backward compatibility in tests
def synthetic_theta_pnl(
    strategy: StrategyMetrics,
    days_elapsed: int,
    iv_change: float = 0.0,
) -> float:
    """
    DEPRECATED: Use mark_to_market_strategy() instead.
    Kept for backward compatibility with existing tests.
    """
    theta_pnl = strategy.greeks.theta * days_elapsed * strategy.lot_size * THETA_DECAY_ADJ
    vega_pnl = strategy.greeks.vega * iv_change * 100 * strategy.lot_size
    return theta_pnl + vega_pnl


# ── Entry Eligibility ──────────────────────────────────────────────────────────


def _is_entry_eligible(
    strategy_type: StrategyType,
    signals: DailySignals,
) -> bool:
    """
    Check whether market conditions permit opening this strategy.

    This is the core fix for the v1 bug where strategies were deployed
    blindly every N days regardless of conditions. Now:
      - Regime must be appropriate for the strategy
      - Premium-selling strategies require positive VRP
      - Premium-selling strategies require IV not at rock bottom
    """
    # Regime filter
    allowed_regimes = STRATEGY_REGIME_FILTER.get(
        strategy_type, {"BULL", "SIDEWAYS", "BEAR", "UNKNOWN"}
    )
    if signals.regime not in allowed_regimes:
        return False

    # For premium-selling: require VRP ≥ 0 and IV percentile ≥ threshold
    if strategy_type in PREMIUM_SELLING_STRATEGIES:
        if signals.vrp < MIN_VRP_FOR_SELLING:
            return False
        if signals.iv_percentile < MIN_IV_PERCENTILE_FOR_SELLING:
            return False

    return True


# ── Strategy Backtester ─────────────────────────────────────────────────────────


class OptionsBacktester:
    """
    Backtests options strategies on historical data using synthetic pricing.

    v2: Uses pre-computed signals (no warmup period inside the backtest loop).
    v2: Applies signal-based entry filters even for specific strategy tests.
    """

    def __init__(
        self,
        capital: float = DEFAULT_CAPITAL,
        options_pct: float = OPTIONS_ALLOCATION_PCT,
        max_concurrent: int = 5,
    ):
        self.capital = capital
        self.options_capital = capital * options_pct
        self.max_concurrent = max_concurrent

    def backtest_strategy(
        self,
        close_prices: pd.Series,
        regime_series: Optional[pd.Series] = None,
        vix_series: Optional[pd.Series] = None,
        strategy_type: Optional[StrategyType] = None,
        symbol: str = "NIFTY",
        entry_interval: int = 7,       # New trade every N days
        dte_at_entry: int = 30,        # Days to expiry when entering
        early_exit_pct: float = 0.50,  # Close at 50% of max profit
        stop_loss_pct: float = 1.0,    # Close at 100% of max loss (tighter)
        precomputed_signals: Optional[dict] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> OptionsBacktestResult:
        """
        Backtest a specific options strategy type on historical data.

        v2 changes:
          - Accepts pre-computed signals to eliminate the warmup problem
          - Applies signal-based entry filters (regime + VRP + IV percentile)
          - start_date/end_date allow slicing without re-computing signals
        """
        dates = close_prices.index.sort_values()

        # Apply date filters if provided
        if start_date is not None:
            dates = dates[dates >= start_date]
        if end_date is not None:
            dates = dates[dates <= end_date]

        if len(dates) < 10:
            return OptionsBacktestResult(
                strategy_type=strategy_type.value if strategy_type else "MIXED",
            )

        # Use pre-computed signals or compute fresh
        if precomputed_signals is None:
            precomputed_signals = precompute_signals(
                close_prices, vix_series, regime_series,
            )

        # Track state
        trades: list[OptionsTradeRecord] = []
        open_positions: list[dict] = []
        daily_pnl: list[float] = []
        margin_used: list[float] = []
        last_entry_idx = -entry_interval  # Allow immediate first entry
        cum_peak_tracker = 0.0  # For portfolio drawdown circuit breaker

        for i, dt in enumerate(dates):
            date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]
            spot = float(close_prices.loc[dt])
            if np.isnan(spot):
                continue

            # Look up pre-computed signals
            sig = precomputed_signals.get(dt)
            if sig is None:
                # No signals for this date — skip entry but manage positions
                sig = DailySignals(iv=0.18, rv=0.15, vrp=0.2,
                                   iv_percentile=50, vix_regime="MEDIUM",
                                   regime="UNKNOWN")

            day_pnl = 0.0
            day_margin = 0.0

            # ── Manage open positions (FULL BS REPRICING) ─────────────
            positions_to_close = []
            for pos_idx, pos in enumerate(open_positions):
                pos["days_held"] += 1
                pos["remaining_dte"] -= 1

                strategy: StrategyMetrics = pos["strategy"]

                # Check expiry
                if pos["remaining_dte"] <= 0:
                    pnl = strategy.payoff_at_price(spot)
                    pnl -= BROKERAGE_PER_LOT * 2
                    pnl -= abs(pnl) * STT_ON_SELL_PCT

                    trades.append(OptionsTradeRecord(
                        date=date_str,
                        strategy_type=pos["strategy_type"],
                        symbol=symbol, action="EXPIRE",
                        net_premium=pos["net_premium"],
                        max_profit=pos["max_profit"],
                        max_loss=pos["max_loss"],
                        margin_used=pos["margin"],
                        realized_pnl=pnl,
                        days_held=pos["days_held"],
                        regime=sig.regime,
                        vix_level=sig.iv * 100,
                        vrp_at_entry=pos["vrp_at_entry"],
                    ))
                    day_pnl += pnl
                    positions_to_close.append(pos_idx)
                    continue

                # ── FULL BS REPRICING (fixes the 3 Greek-approximation bugs) ──
                # Reprice every leg with current spot, current IV, remaining DTE.
                # This correctly captures: gamma (convexity), theta acceleration
                # near expiry, and vega (applied once, not cumulatively).
                current_value = mark_to_market_strategy(
                    strategy, spot, sig.iv, pos["remaining_dte"],
                )
                pos_pnl = current_value - pos["prev_value"]
                pos["cum_pnl"] += pos_pnl
                pos["prev_value"] = current_value
                day_pnl += pos_pnl
                day_margin += pos["margin"]

                # Early exit: take profit at 50% of max profit
                if pos["max_profit"] > 0 and pos["cum_pnl"] >= pos["max_profit"] * early_exit_pct:
                    pnl = pos["cum_pnl"]
                    pnl -= BROKERAGE_PER_LOT * 2
                    pnl -= abs(pnl) * STT_ON_SELL_PCT

                    trades.append(OptionsTradeRecord(
                        date=date_str,
                        strategy_type=pos["strategy_type"],
                        symbol=symbol, action="CLOSE",
                        net_premium=pos["net_premium"],
                        max_profit=pos["max_profit"],
                        max_loss=pos["max_loss"],
                        margin_used=pos["margin"],
                        realized_pnl=pnl,
                        days_held=pos["days_held"],
                        regime=sig.regime,
                        vix_level=sig.iv * 100,
                        vrp_at_entry=pos["vrp_at_entry"],
                    ))
                    positions_to_close.append(pos_idx)
                    continue

                # Stop loss: close at 1× max loss (not 1.5× — tighter risk control)
                if pos["max_loss"] > 0 and pos["cum_pnl"] <= -pos["max_loss"] * stop_loss_pct:
                    pnl = pos["cum_pnl"]
                    pnl -= BROKERAGE_PER_LOT * 2
                    pnl -= abs(pnl) * STT_ON_SELL_PCT

                    trades.append(OptionsTradeRecord(
                        date=date_str,
                        strategy_type=pos["strategy_type"],
                        symbol=symbol, action="CLOSE",
                        net_premium=pos["net_premium"],
                        max_profit=pos["max_profit"],
                        max_loss=pos["max_loss"],
                        margin_used=pos["margin"],
                        realized_pnl=pnl,
                        days_held=pos["days_held"],
                        regime=sig.regime,
                        vix_level=sig.iv * 100,
                        vrp_at_entry=pos["vrp_at_entry"],
                    ))
                    positions_to_close.append(pos_idx)

            # Remove closed positions
            for idx in sorted(positions_to_close, reverse=True):
                open_positions.pop(idx)

            daily_pnl.append(day_pnl)
            margin_used.append(day_margin)

            # ── Portfolio-level drawdown circuit breaker ────────────────
            # If cumulative P&L drawdown exceeds 10% of options capital,
            # halt new entries (but keep managing existing positions).
            cum_total = sum(daily_pnl) if daily_pnl else 0.0
            cum_peak = max(cum_peak_tracker, cum_total) if i > 0 else cum_total
            cum_peak_tracker = cum_peak
            portfolio_dd = cum_peak - cum_total
            dd_breaker_active = portfolio_dd > self.options_capital * 0.10

            # ── Open new positions ─────────────────────────────────────
            if (
                i - last_entry_idx >= entry_interval
                and len(open_positions) < self.max_concurrent
                and not dd_breaker_active
            ):
                available_margin = self.options_capital - day_margin

                strategy_built = None

                if strategy_type is not None:
                    # SPECIFIC strategy test — BUT with signal filtering
                    if _is_entry_eligible(strategy_type, sig):
                        strategy_built = self._build_specific_strategy(
                            strategy_type, symbol, spot, sig.iv, dte_at_entry,
                        )
                else:
                    # Use the full strategy selector
                    recs = select_strategies(
                        sig.regime, sig.vix_regime,
                        vrp_signal=sig.vrp,
                        iv_percentile=sig.iv_percentile,
                    )
                    for rec in recs:
                        st = StrategyType(rec.strategy_type)
                        if _is_entry_eligible(st, sig):
                            s = build_strategy_for_symbol(rec, symbol, spot, sig.iv, dte_at_entry)
                            if s is not None and s.margin_required <= available_margin:
                                strategy_built = s
                                break

                if (
                    strategy_built is not None
                    and strategy_built.margin_required <= available_margin
                    and strategy_built.margin_required > 0
                ):
                    # Compute initial mark-to-market value for BS repricing
                    entry_value = mark_to_market_strategy(
                        strategy_built, spot, sig.iv, dte_at_entry,
                    )
                    open_positions.append({
                        "strategy": strategy_built,
                        "strategy_type": strategy_built.strategy_type.value,
                        "entry_date": date_str,
                        "entry_spot": spot,
                        "entry_iv": sig.iv,
                        "net_premium": strategy_built.net_premium,
                        "max_profit": strategy_built.max_profit,
                        "max_loss": strategy_built.max_loss,
                        "margin": strategy_built.margin_required,
                        "days_held": 0,
                        "remaining_dte": dte_at_entry,
                        "cum_pnl": 0.0,
                        "prev_value": entry_value,  # For BS repricing
                        "vrp_at_entry": sig.vrp,
                    })
                    last_entry_idx = i

                    trades.append(OptionsTradeRecord(
                        date=date_str,
                        strategy_type=strategy_built.strategy_type.value,
                        symbol=symbol, action="OPEN",
                        net_premium=strategy_built.net_premium,
                        max_profit=strategy_built.max_profit,
                        max_loss=strategy_built.max_loss,
                        margin_used=strategy_built.margin_required,
                        regime=sig.regime,
                        vix_level=sig.iv * 100,
                        vrp_at_entry=sig.vrp,
                    ))

        # Close any remaining open positions at last price
        if open_positions and len(dates) > 0:
            last_spot = float(close_prices.loc[dates[-1]])
            for pos in open_positions:
                pnl = pos["strategy"].payoff_at_price(last_spot)
                pnl -= BROKERAGE_PER_LOT * 2
                trades.append(OptionsTradeRecord(
                    date=str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])[:10],
                    strategy_type=pos["strategy_type"],
                    symbol=symbol, action="CLOSE",
                    net_premium=pos["net_premium"],
                    max_profit=pos["max_profit"],
                    max_loss=pos["max_loss"],
                    margin_used=pos["margin"],
                    realized_pnl=pnl,
                    days_held=pos["days_held"],
                    regime="UNKNOWN",
                    vrp_at_entry=pos["vrp_at_entry"],
                ))
                daily_pnl.append(pnl)

        return self._compute_metrics(
            strategy_type.value if strategy_type else "MIXED",
            trades, daily_pnl, margin_used,
        )

    def _build_specific_strategy(
        self,
        strategy_type: StrategyType,
        symbol: str,
        spot: float,
        iv: float,
        dte: int,
    ) -> Optional[StrategyMetrics]:
        """Build a specific strategy type with default parameters."""
        from app.quant.options_strategies import (
            build_covered_call, build_iron_condor, build_bull_put_spread,
            build_bear_call_spread, build_short_strangle,
        )

        T = dte / 365.0
        otm_pct = iv * np.sqrt(T)  # 1σ OTM distance as % of spot
        otm_dist = spot * otm_pct

        try:
            if strategy_type == StrategyType.IRON_CONDOR:
                return build_iron_condor(
                    symbol, spot,
                    round(spot - otm_dist, -1),
                    round(spot - otm_dist * 1.5, -1),
                    round(spot + otm_dist, -1),
                    round(spot + otm_dist * 1.5, -1),
                    iv, dte,
                )
            elif strategy_type == StrategyType.BULL_PUT_SPREAD:
                return build_bull_put_spread(
                    symbol, spot,
                    round(spot - otm_dist, -1),
                    round(spot - otm_dist * 1.5, -1),
                    iv, dte,
                )
            elif strategy_type == StrategyType.BEAR_CALL_SPREAD:
                return build_bear_call_spread(
                    symbol, spot,
                    round(spot + otm_dist, -1),
                    round(spot + otm_dist * 1.5, -1),
                    iv, dte,
                )
            elif strategy_type == StrategyType.SHORT_STRANGLE:
                return build_short_strangle(
                    symbol, spot,
                    round(spot - otm_dist, -1),
                    round(spot + otm_dist, -1),
                    iv, dte,
                    add_wings=True,
                    wing_width=max(round(otm_dist * 0.5, -1), 50),
                )
            elif strategy_type == StrategyType.COVERED_CALL:
                return build_covered_call(
                    symbol, spot,
                    round(spot + otm_dist, -1),
                    iv, dte,
                )
            else:
                logger.warning("Specific build not implemented for %s", strategy_type)
                return None
        except Exception as e:
            logger.error("Failed to build %s: %s", strategy_type.value, e)
            return None

    @staticmethod
    def _compute_metrics(
        strategy_type: str,
        trades: list[OptionsTradeRecord],
        daily_pnl: list[float],
        margin_used: list[float],
    ) -> OptionsBacktestResult:
        """Compute backtest performance metrics from trade records."""
        closed_trades = [t for t in trades if t.action in ("CLOSE", "EXPIRE")]

        if not closed_trades:
            return OptionsBacktestResult(strategy_type=strategy_type)

        pnls = [t.realized_pnl for t in closed_trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        n_trades = len(closed_trades)
        win_rate = len(winning) / n_trades if n_trades > 0 else 0

        # Sharpe & Sortino from daily P&L
        daily = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        daily = daily[~np.isnan(daily)]

        # Use peak margin as denominator for all ratios.
        # avg_margin can be tiny on sparse windows → inflated metrics.
        # peak_margin reflects the actual capital at risk.
        active_margins = [m for m in margin_used if m > 0]
        avg_margin = np.mean(active_margins) if active_margins else 1.0
        peak_margin = max(active_margins) if active_margins else 1.0
        avg_margin = max(avg_margin, 1.0)
        peak_margin = max(peak_margin, 1.0)

        # Daily returns relative to peak margin (conservative denominator)
        daily_returns = daily / peak_margin
        ann_factor = np.sqrt(252)

        mean_ret = np.mean(daily_returns) if len(daily_returns) > 0 else 0
        std_ret = np.std(daily_returns) if len(daily_returns) > 1 else 1.0
        sharpe = (mean_ret / max(std_ret, 1e-8)) * ann_factor if std_ret > 1e-8 else 0.0

        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else std_ret
        sortino = (mean_ret / max(downside_std, 1e-8)) * ann_factor if downside_std > 1e-8 else 0.0

        # Max drawdown relative to peak margin (fixes the 288% bug)
        cumulative = np.cumsum(daily)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        max_dd_pct = (max_dd / peak_margin) * 100

        avg_days = np.mean([t.days_held for t in closed_trades]) if closed_trades else 0

        # Annualized return
        n_years = max(len(daily) / 252, 0.01)

        return OptionsBacktestResult(
            strategy_type=strategy_type,
            n_trades=n_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=round(total_pnl, 2),
            avg_pnl_per_trade=round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
            win_rate=round(win_rate, 4),
            avg_days_held=round(avg_days, 1),
            total_return_pct=round(total_pnl / peak_margin * 100, 2),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_dd_pct, 2),
            avg_margin_used=round(avg_margin, 2),
            capital_efficiency=round(total_pnl / max(avg_margin, 1) * 100 / n_years, 2),
            trades=trades,
            daily_pnl=list(daily_pnl),
        )


# ── Walk-Forward Validation ─────────────────────────────────────────────────────


def walk_forward_validate(
    close_prices: pd.Series,
    strategy_type: StrategyType,
    symbol: str = "NIFTY",
    vix_series: Optional[pd.Series] = None,
    regime_series: Optional[pd.Series] = None,
    train_days: int = WALK_FORWARD_TRAIN,
    test_days: int = WALK_FORWARD_TEST,
    capital: float = DEFAULT_CAPITAL,
    min_sharpe: float = MIN_SHARPE_OOS,
) -> WalkForwardResult:
    """
    Walk-forward validation of an options strategy.

    v2 CRITICAL FIX: Pre-computes all signals on the FULL series,
    then passes them to each window's backtest. This eliminates the
    warmup-kills-OOS bug where 63-day windows had 0 tradeable days.

    Validation gate for AGENTS.md rule 6.
    """
    dates = close_prices.index.sort_values()
    total_days = len(dates)
    window = train_days + test_days

    if total_days < window:
        return WalkForwardResult(
            strategy_type=strategy_type.value,
            reason=f"Insufficient data: need {window} days, got {total_days}",
        )

    # PRE-COMPUTE ALL SIGNALS ON FULL SERIES (the key fix)
    all_signals = precompute_signals(close_prices, vix_series, regime_series)

    backtester = OptionsBacktester(capital=capital)
    oos_results: list[OptionsBacktestResult] = []
    is_results: list[OptionsBacktestResult] = []

    n_windows = 0
    start_idx = 0

    while start_idx + window <= total_days:
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end_idx = min(start_idx + window - 1, total_days - 1)
        test_end = dates[test_end_idx]

        # In-sample backtest (uses pre-computed signals, no warmup issue)
        is_result = backtester.backtest_strategy(
            close_prices, regime_series, vix_series,
            strategy_type=strategy_type, symbol=symbol,
            precomputed_signals=all_signals,
            start_date=train_start, end_date=train_end,
        )
        is_results.append(is_result)

        # Out-of-sample backtest (uses SAME pre-computed signals — no warmup!)
        oos_result = backtester.backtest_strategy(
            close_prices, regime_series, vix_series,
            strategy_type=strategy_type, symbol=symbol,
            precomputed_signals=all_signals,
            start_date=test_start, end_date=test_end,
        )
        oos_result.is_oos = True
        oos_results.append(oos_result)

        n_windows += 1
        start_idx += test_days

    if n_windows == 0:
        return WalkForwardResult(
            strategy_type=strategy_type.value,
            reason="No valid windows produced",
        )

    # Aggregate results
    is_sharpes = [r.sharpe_ratio for r in is_results if r.n_trades > 0]
    oos_sharpes = [r.sharpe_ratio for r in oos_results if r.n_trades > 0]

    avg_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

    is_returns = [r.total_return_pct for r in is_results if r.n_trades > 0]
    oos_returns = [r.total_return_pct for r in oos_results if r.n_trades > 0]

    is_dds = [r.max_drawdown_pct for r in is_results if r.n_trades > 0]
    oos_dds = [r.max_drawdown_pct for r in oos_results if r.n_trades > 0]

    # Count windows with trades
    is_with_trades = sum(1 for r in is_results if r.n_trades > 0)
    oos_with_trades = sum(1 for r in oos_results if r.n_trades > 0)
    total_oos_trades = sum(r.n_trades for r in oos_results)

    # Validation: OOS Sharpe must exceed threshold
    validated = avg_oos_sharpe >= min_sharpe and oos_with_trades >= n_windows // 2
    reason = (
        f"OOS Sharpe {avg_oos_sharpe:.3f} {'≥' if avg_oos_sharpe >= min_sharpe else '<'} {min_sharpe} "
        f"across {n_windows} windows ({oos_with_trades} with trades, {total_oos_trades} total OOS trades). "
        f"IS Sharpe: {avg_is_sharpe:.3f} ({is_with_trades} IS windows with trades)"
    )

    if not validated and avg_oos_sharpe >= min_sharpe:
        reason += f" — NOT VALIDATED: only {oos_with_trades}/{n_windows} OOS windows had trades"

    if not validated and is_sharpes and oos_sharpes:
        if avg_is_sharpe > min_sharpe * 2 and avg_oos_sharpe < min_sharpe:
            reason += " — LIKELY OVERFIT (IS >> OOS)"

    return WalkForwardResult(
        strategy_type=strategy_type.value,
        n_windows=n_windows,
        in_sample_sharpe=round(avg_is_sharpe, 4),
        out_of_sample_sharpe=round(avg_oos_sharpe, 4),
        in_sample_return_pct=round(float(np.mean(is_returns)) if is_returns else 0, 2),
        out_of_sample_return_pct=round(float(np.mean(oos_returns)) if oos_returns else 0, 2),
        in_sample_max_dd=round(float(np.max(is_dds)) if is_dds else 0, 2),
        out_of_sample_max_dd=round(float(np.max(oos_dds)) if oos_dds else 0, 2),
        validated=validated,
        reason=reason,
        window_results=oos_results,
    )


# ── Convenience: reconstruct_daily_iv (kept for backward compat) ────────────────


def reconstruct_daily_iv(
    close_prices: pd.Series,
    vix_series: Optional[pd.Series] = None,
    window: int = 20,
) -> pd.Series:
    """
    Reconstruct daily implied volatility from either VIX data or
    realized vol with a VRP markup.
    """
    rv = compute_realized_vol(close_prices, window=window)

    if vix_series is not None and len(vix_series) > 0:
        iv = vix_series.reindex(close_prices.index, method="ffill") / 100.0
        iv = iv.fillna(rv * VRP_PROXY_RATIO)
    else:
        iv = rv * VRP_PROXY_RATIO

    iv = iv.clip(0.08, 1.0)
    return iv
