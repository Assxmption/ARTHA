"""
Options Strategy Definitions & Payoff Engine
==============================================
Defines multi-leg options strategies as pure data structures, computes
their P&L profiles, aggregate Greeks, margin requirements, and risk metrics.

Each strategy is a collection of "legs" (individual option positions).
The engine computes:
  - Max profit / max loss / breakeven points
  - Payoff at any underlying price (the payoff diagram)
  - Aggregate portfolio Greeks (delta, gamma, theta, vega)
  - Approximate SPAN margin requirement
  - Risk/reward ratio

Supported Strategies (10):
  1. COVERED_CALL        — Long stock + short OTM call
  2. PROTECTIVE_PUT      — Long stock + long OTM put
  3. SHORT_STRANGLE      — Short OTM call + short OTM put (+ wings for defined risk)
  4. IRON_CONDOR         — Short strangle + long wings
  5. BULL_PUT_SPREAD     — Short put + long lower put
  6. BEAR_CALL_SPREAD    — Short call + long higher call
  7. CALENDAR_SPREAD     — Short near-month + long far-month (same strike)
  8. RATIO_SPREAD        — 1×long + 2×short (volatility selling with directional bias)
  9. SYNTHETIC_SHORT     — Long put + short call (same strike)
 10. COLLAR              — Long stock + long put + short call

Capital context: ₹5 crore starting capital allows simultaneous multi-strategy
deployment across 3-5 index positions and 10-15 stock-level strategies.

All computation is deterministic — LLMs never touch this (AGENTS.md rule 1).

Reference: docs/ARTHA_ARCHITECTURE.md §4.3, Implementation Plan §Component 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from app.quant.options_pricing import (
    bs_price, compute_delta, compute_gamma, compute_theta, compute_vega,
    get_lot_size, DEFAULT_RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)


# ── Enumerations ────────────────────────────────────────────────────────────────


class StrategyType(str, Enum):
    COVERED_CALL = "COVERED_CALL"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    SHORT_STRANGLE = "SHORT_STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    RATIO_SPREAD = "RATIO_SPREAD"
    SYNTHETIC_SHORT = "SYNTHETIC_SHORT"
    COLLAR = "COLLAR"


class LegType(str, Enum):
    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"
    LONG_STOCK = "LONG_STOCK"
    SHORT_STOCK = "SHORT_STOCK"


# ── Data Classes ────────────────────────────────────────────────────────────────


@dataclass
class OptionLeg:
    """A single leg of a multi-leg options strategy."""
    leg_type: LegType
    strike: float = 0.0         # 0 for stock legs
    expiry_days: int = 30       # Days to expiry (0 for stock legs)
    quantity: int = 1           # Number of lots (negative for short)
    premium: float = 0.0        # Per-share premium paid/received
    iv: float = 0.0             # Implied volatility used to price this leg

    @property
    def is_option(self) -> bool:
        return self.leg_type not in (LegType.LONG_STOCK, LegType.SHORT_STOCK)

    @property
    def is_long(self) -> bool:
        return self.leg_type in (LegType.LONG_CALL, LegType.LONG_PUT, LegType.LONG_STOCK)

    @property
    def is_call(self) -> bool:
        return self.leg_type in (LegType.LONG_CALL, LegType.SHORT_CALL)

    @property
    def is_put(self) -> bool:
        return self.leg_type in (LegType.LONG_PUT, LegType.SHORT_PUT)


@dataclass
class StrategyGreeks:
    """Aggregate Greeks for a multi-leg strategy."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # Per calendar day
    vega: float = 0.0     # Per 1% IV move
    net_delta_dollars: float = 0.0  # Delta × spot × lot_size


@dataclass
class StrategyMetrics:
    """Full metrics for a strategy at construction time."""
    strategy_type: StrategyType
    symbol: str
    spot: float
    legs: list[OptionLeg]
    lot_size: int

    # P&L metrics
    net_premium: float = 0.0       # Net credit (+) or debit (-)
    max_profit: float = 0.0        # Per lot
    max_loss: float = 0.0          # Per lot (positive number = loss amount)
    risk_reward_ratio: float = 0.0 # max_loss / max_profit
    breakevens: list[float] = field(default_factory=list)

    # Greeks
    greeks: StrategyGreeks = field(default_factory=StrategyGreeks)

    # Margin
    margin_required: float = 0.0   # Approximate SPAN margin

    # Metadata
    days_to_expiry: int = 30
    iv_at_entry: float = 0.0

    def payoff_at_price(self, price: float) -> float:
        """Compute per-lot P&L at a given underlying price at expiry."""
        return _payoff_at_expiry(self.legs, price, self.lot_size)

    def payoff_curve(
        self, n_points: int = 200, pct_range: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the payoff diagram: prices vs P&L.

        Returns (prices, pnl) arrays for plotting.
        """
        low = self.spot * (1 - pct_range)
        high = self.spot * (1 + pct_range)
        prices = np.linspace(low, high, n_points)
        pnl = np.array([self.payoff_at_price(p) for p in prices])
        return prices, pnl


# ── Payoff Computation ──────────────────────────────────────────────────────────


def _leg_payoff_at_expiry(leg: OptionLeg, price: float) -> float:
    """Compute per-share P&L for a single leg at expiry."""
    if leg.leg_type == LegType.LONG_CALL:
        return max(price - leg.strike, 0) - leg.premium
    elif leg.leg_type == LegType.SHORT_CALL:
        return leg.premium - max(price - leg.strike, 0)
    elif leg.leg_type == LegType.LONG_PUT:
        return max(leg.strike - price, 0) - leg.premium
    elif leg.leg_type == LegType.SHORT_PUT:
        return leg.premium - max(leg.strike - price, 0)
    elif leg.leg_type == LegType.LONG_STOCK:
        return price - leg.premium  # premium = entry price for stock
    elif leg.leg_type == LegType.SHORT_STOCK:
        return leg.premium - price
    return 0.0


def _payoff_at_expiry(
    legs: list[OptionLeg], price: float, lot_size: int,
) -> float:
    """
    Total per-lot P&L at expiry for a multi-leg strategy.

    Each leg's quantity scales its contribution.
    """
    total = 0.0
    for leg in legs:
        per_share = _leg_payoff_at_expiry(leg, price)
        total += per_share * abs(leg.quantity) * lot_size
    return total


# ── Greeks Aggregation ──────────────────────────────────────────────────────────


def _compute_strategy_greeks(
    legs: list[OptionLeg],
    spot: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    lot_size: int = 1,
) -> StrategyGreeks:
    """
    Compute aggregate Greeks across all legs of a strategy.

    Stock legs contribute delta = ±1, gamma = 0, theta = 0, vega = 0.
    """
    total = StrategyGreeks()

    for leg in legs:
        qty = abs(leg.quantity)
        sign = 1.0 if leg.is_long else -1.0

        if not leg.is_option:
            # Stock leg: delta = ±1
            total.delta += sign * qty
            continue

        T = max(leg.expiry_days / 365.0, 1e-6)
        iv = max(leg.iv, 0.01)
        opt_type = "call" if leg.is_call else "put"

        delta = float(compute_delta(spot, leg.strike, T, r, iv, opt_type))
        gamma = float(compute_gamma(spot, leg.strike, T, r, iv))
        theta = float(compute_theta(spot, leg.strike, T, r, iv, opt_type))
        vega = float(compute_vega(spot, leg.strike, T, r, iv))

        total.delta += sign * qty * delta
        total.gamma += sign * qty * gamma
        total.theta += sign * qty * theta
        total.vega += sign * qty * vega

    total.net_delta_dollars = total.delta * spot * lot_size
    return total


# ── Margin Approximation ────────────────────────────────────────────────────────


def _approximate_span_margin(
    legs: list[OptionLeg],
    spot: float,
    lot_size: int,
    is_index: bool = False,
) -> float:
    """
    Approximate SPAN margin for an options strategy.

    This is a simplified model based on NSE's margin framework:
    - For defined-risk strategies (spreads, condors): margin = max loss
    - For naked short options: margin ≈ premium + max(
        span_pct × underlying_value - OTM_amount,
        span_pct_min × underlying_value
      )
    - For covered positions: reduced margin (underlying covers the short)

    The actual SPAN margin is computed by NSE's algorithm and varies
    intraday. This approximation is within ~10% for strategy sizing.

    SEBI 2026: Additional 2% ELM on expiry day for short index options.
    """
    SPAN_PCT = 0.15 if is_index else 0.20   # Base SPAN %
    ELM_PCT = 0.035 if is_index else 0.05   # Extreme Loss Margin

    has_short_options = any(
        not leg.is_long and leg.is_option for leg in legs
    )
    has_long_options = any(
        leg.is_long and leg.is_option for leg in legs
    )
    has_stock = any(not leg.is_option for leg in legs)

    if not has_short_options:
        # Long-only options: margin = premium paid
        total_premium = sum(
            leg.premium * abs(leg.quantity) * lot_size
            for leg in legs if leg.is_option and leg.is_long
        )
        return total_premium

    # Check if it's a defined-risk strategy (every short leg has a covering long)
    short_calls = [l for l in legs if l.leg_type == LegType.SHORT_CALL]
    long_calls = [l for l in legs if l.leg_type == LegType.LONG_CALL]
    short_puts = [l for l in legs if l.leg_type == LegType.SHORT_PUT]
    long_puts = [l for l in legs if l.leg_type == LegType.LONG_PUT]
    long_stock = any(l.leg_type == LegType.LONG_STOCK for l in legs)

    # Covered call: stock covers the short call
    if long_stock and short_calls and not short_puts:
        return spot * lot_size * 0.10  # Reduced margin for covered

    # Spreads: defined risk = max loss
    is_spread = (
        (len(short_calls) > 0 and len(long_calls) > 0 and not short_puts) or
        (len(short_puts) > 0 and len(long_puts) > 0 and not short_calls) or
        (len(short_calls) > 0 and len(long_calls) > 0 and
         len(short_puts) > 0 and len(long_puts) > 0)  # Iron condor
    )

    if is_spread:
        # Max loss calculation for spreads
        prices_test = np.linspace(spot * 0.7, spot * 1.3, 100)
        worst_loss = min(
            _payoff_at_expiry(legs, p, lot_size)
            for p in prices_test
        )
        return abs(worst_loss) if worst_loss < 0 else 0.0

    # Naked short options: full margin
    margin = 0.0
    for leg in legs:
        if not leg.is_long and leg.is_option:
            notional = spot * lot_size * abs(leg.quantity)
            otm_amount = 0.0
            if leg.is_call:
                otm_amount = max(leg.strike - spot, 0) * lot_size * abs(leg.quantity)
            else:
                otm_amount = max(spot - leg.strike, 0) * lot_size * abs(leg.quantity)

            span = max(
                SPAN_PCT * notional - otm_amount + leg.premium * lot_size * abs(leg.quantity),
                (SPAN_PCT / 2) * notional,
            )
            elm = ELM_PCT * notional
            margin += span + elm

    return margin


# ── Breakeven Solver ────────────────────────────────────────────────────────────


def _find_breakevens(
    legs: list[OptionLeg],
    spot: float,
    lot_size: int,
    n_points: int = 1000,
    pct_range: float = 0.30,
) -> list[float]:
    """
    Find breakeven prices where strategy P&L = 0 at expiry.

    Uses numerical scan: evaluates payoff at n_points and finds zero crossings.
    """
    prices = np.linspace(spot * (1 - pct_range), spot * (1 + pct_range), n_points)
    pnls = np.array([_payoff_at_expiry(legs, p, lot_size) for p in prices])

    breakevens = []
    for i in range(len(pnls) - 1):
        if pnls[i] * pnls[i + 1] < 0:  # Sign change → zero crossing
            # Linear interpolation for more precise breakeven
            ratio = abs(pnls[i]) / (abs(pnls[i]) + abs(pnls[i + 1]))
            be = prices[i] + ratio * (prices[i + 1] - prices[i])
            breakevens.append(round(be, 2))

    return breakevens


# ── Strategy Constructors ───────────────────────────────────────────────────────


def build_covered_call(
    symbol: str,
    spot: float,
    call_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Covered Call: Long stock + Short OTM call.

    Best in: BULL regime with Low/Medium VIX.
    Profit source: Theta decay + partial upside.
    Risk: Stock decline (cushioned by premium received).
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    call_premium = float(bs_price(spot, call_strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.LONG_STOCK, premium=spot, quantity=1, expiry_days=0),
        OptionLeg(LegType.SHORT_CALL, call_strike, days_to_expiry, 1, call_premium, iv),
    ]

    max_profit = (call_strike - spot + call_premium) * lot_size
    max_loss = (spot - call_premium) * lot_size  # Stock goes to zero

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.COVERED_CALL,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=call_premium,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_protective_put(
    symbol: str,
    spot: float,
    put_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Protective Put: Long stock + Long OTM put.

    Best in: BEAR regime or transition periods.
    Profit source: Unlimited upside on stock.
    Protection: Put limits downside to (spot - put_strike + put_premium).
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    put_premium = float(bs_price(spot, put_strike, T, r, iv, "put"))

    legs = [
        OptionLeg(LegType.LONG_STOCK, premium=spot, quantity=1, expiry_days=0),
        OptionLeg(LegType.LONG_PUT, put_strike, days_to_expiry, 1, put_premium, iv),
    ]

    max_loss = (spot - put_strike + put_premium) * lot_size
    max_profit = float("inf")  # Unlimited upside

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.PROTECTIVE_PUT,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=-put_premium,  # Debit
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=0.0,  # Unlimited profit → ratio not meaningful
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_iron_condor(
    symbol: str,
    spot: float,
    put_short_strike: float,
    put_long_strike: float,
    call_short_strike: float,
    call_long_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Iron Condor: Short put spread + Short call spread (defined risk).

    Best in: SIDEWAYS regime with Medium/High VIX (rich premiums).
    Profit source: Theta decay within range, VRP harvest.
    Risk: Price moves beyond either wing.

    This is the bread-and-butter premium selling strategy — defined risk,
    clear max profit/loss, benefits from time decay and vol contraction.
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    put_short_p = float(bs_price(spot, put_short_strike, T, r, iv, "put"))
    put_long_p = float(bs_price(spot, put_long_strike, T, r, iv, "put"))
    call_short_p = float(bs_price(spot, call_short_strike, T, r, iv, "call"))
    call_long_p = float(bs_price(spot, call_long_strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.SHORT_PUT, put_short_strike, days_to_expiry, 1, put_short_p, iv),
        OptionLeg(LegType.LONG_PUT, put_long_strike, days_to_expiry, 1, put_long_p, iv),
        OptionLeg(LegType.SHORT_CALL, call_short_strike, days_to_expiry, 1, call_short_p, iv),
        OptionLeg(LegType.LONG_CALL, call_long_strike, days_to_expiry, 1, call_long_p, iv),
    ]

    net_credit = (put_short_p - put_long_p) + (call_short_p - call_long_p)
    put_spread_width = put_short_strike - put_long_strike
    call_spread_width = call_long_strike - call_short_strike
    max_spread_width = max(put_spread_width, call_spread_width)
    max_loss = (max_spread_width - net_credit) * lot_size
    max_profit = net_credit * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.IRON_CONDOR,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_credit,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_bull_put_spread(
    symbol: str,
    spot: float,
    short_strike: float,
    long_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Bull Put Spread: Short put + Long lower put.

    Best in: BULL regime, collecting premium while being moderately bullish.
    Profit source: Theta decay, stock stays above short strike.
    Risk: Defined — limited to spread width minus credit.
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    short_p = float(bs_price(spot, short_strike, T, r, iv, "put"))
    long_p = float(bs_price(spot, long_strike, T, r, iv, "put"))

    legs = [
        OptionLeg(LegType.SHORT_PUT, short_strike, days_to_expiry, 1, short_p, iv),
        OptionLeg(LegType.LONG_PUT, long_strike, days_to_expiry, 1, long_p, iv),
    ]

    net_credit = short_p - long_p
    spread_width = short_strike - long_strike
    max_profit = net_credit * lot_size
    max_loss = (spread_width - net_credit) * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.BULL_PUT_SPREAD,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_credit,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_bear_call_spread(
    symbol: str,
    spot: float,
    short_strike: float,
    long_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Bear Call Spread: Short call + Long higher call.

    Best in: BEAR regime — profit from decline while capping risk.
    Profit source: Theta decay, stock stays below short strike.
    Risk: Defined — limited to spread width minus credit.
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    short_p = float(bs_price(spot, short_strike, T, r, iv, "call"))
    long_p = float(bs_price(spot, long_strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.SHORT_CALL, short_strike, days_to_expiry, 1, short_p, iv),
        OptionLeg(LegType.LONG_CALL, long_strike, days_to_expiry, 1, long_p, iv),
    ]

    net_credit = short_p - long_p
    spread_width = long_strike - short_strike
    max_profit = net_credit * lot_size
    max_loss = (spread_width - net_credit) * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.BEAR_CALL_SPREAD,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_credit,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_short_strangle(
    symbol: str,
    spot: float,
    put_strike: float,
    call_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
    add_wings: bool = True,
    wing_width: float = 0.0,
) -> StrategyMetrics:
    """
    Short Strangle: Short OTM put + Short OTM call.

    Optionally adds wings (becomes an iron condor variant) for defined risk.
    When `add_wings=True` and `wing_width > 0`, a long put below and long
    call above are added, making the risk defined.

    Best in: SIDEWAYS regime with High VIX (maximum VRP harvest).
    Profit source: Massive theta decay, vol contraction.
    Risk: Undefined without wings; defined with wings.
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    put_p = float(bs_price(spot, put_strike, T, r, iv, "put"))
    call_p = float(bs_price(spot, call_strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.SHORT_PUT, put_strike, days_to_expiry, 1, put_p, iv),
        OptionLeg(LegType.SHORT_CALL, call_strike, days_to_expiry, 1, call_p, iv),
    ]

    if add_wings and wing_width > 0:
        put_wing_strike = put_strike - wing_width
        call_wing_strike = call_strike + wing_width
        put_wing_p = float(bs_price(spot, put_wing_strike, T, r, iv, "put"))
        call_wing_p = float(bs_price(spot, call_wing_strike, T, r, iv, "call"))
        legs.append(OptionLeg(LegType.LONG_PUT, put_wing_strike, days_to_expiry, 1, put_wing_p, iv))
        legs.append(OptionLeg(LegType.LONG_CALL, call_wing_strike, days_to_expiry, 1, call_wing_p, iv))

    net_credit = put_p + call_p
    if add_wings and wing_width > 0:
        net_credit -= (put_wing_p + call_wing_p)

    # Max profit/loss
    if add_wings and wing_width > 0:
        max_loss = (wing_width - net_credit) * lot_size
    else:
        # Undefined risk — use a large but finite estimate for sizing
        max_loss = (spot * 0.30 - net_credit) * lot_size  # ~30% move estimate
    max_profit = net_credit * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.SHORT_STRANGLE,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_credit,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_collar(
    symbol: str,
    spot: float,
    put_strike: float,
    call_strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Collar: Long stock + Long OTM put + Short OTM call.

    Best in: BULL → uncertain transition. Protect gains on existing long.
    Profit: Capped at call_strike. Protection: floor at put_strike.
    Often zero-cost (put premium ≈ call premium).
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    put_p = float(bs_price(spot, put_strike, T, r, iv, "put"))
    call_p = float(bs_price(spot, call_strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.LONG_STOCK, premium=spot, quantity=1, expiry_days=0),
        OptionLeg(LegType.LONG_PUT, put_strike, days_to_expiry, 1, put_p, iv),
        OptionLeg(LegType.SHORT_CALL, call_strike, days_to_expiry, 1, call_p, iv),
    ]

    net_premium = call_p - put_p  # Credit if call > put
    max_profit = (call_strike - spot + net_premium) * lot_size
    max_loss = (spot - put_strike - net_premium) * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.COLLAR,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_premium,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_synthetic_short(
    symbol: str,
    spot: float,
    strike: float,
    iv: float,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Synthetic Short: Long put + Short call (same strike).

    Replicates a short stock position using options.
    Best in: BEAR regime — express bearish view without borrowing shares.
    Risk: Unlimited upside risk (like short stock).
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    put_p = float(bs_price(spot, strike, T, r, iv, "put"))
    call_p = float(bs_price(spot, strike, T, r, iv, "call"))

    legs = [
        OptionLeg(LegType.LONG_PUT, strike, days_to_expiry, 1, put_p, iv),
        OptionLeg(LegType.SHORT_CALL, strike, days_to_expiry, 1, call_p, iv),
    ]

    net_premium = call_p - put_p  # Credit if call > put (usually true at ATM)
    # P&L is linear: profit when underlying falls below (strike - net_premium)
    max_profit = (strike + net_premium) * lot_size  # Underlying goes to 0
    max_loss = float("inf")  # Unlimited upside

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.SYNTHETIC_SHORT,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_premium,
        max_profit=max_profit,
        max_loss=max_loss,
        risk_reward_ratio=0.0,  # Infinite max loss
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )


def build_calendar_spread(
    symbol: str,
    spot: float,
    strike: float,
    iv_near: float,
    iv_far: float,
    near_expiry_days: int = 7,
    far_expiry_days: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> StrategyMetrics:
    """
    Calendar Spread: Short near-month + Long far-month (same strike).

    Best in: SIDEWAYS regime with contango term structure.
    Profit source: Near-month decays faster than far-month.
    Risk: Defined — limited to net debit paid.
    """
    lot_size = get_lot_size(symbol)
    T_near = near_expiry_days / 365.0
    T_far = far_expiry_days / 365.0

    # Using calls for the calendar (puts work similarly)
    near_p = float(bs_price(spot, strike, T_near, r, iv_near, "call"))
    far_p = float(bs_price(spot, strike, T_far, r, iv_far, "call"))

    legs = [
        OptionLeg(LegType.SHORT_CALL, strike, near_expiry_days, 1, near_p, iv_near),
        OptionLeg(LegType.LONG_CALL, strike, far_expiry_days, 1, far_p, iv_far),
    ]

    net_debit = far_p - near_p  # Always a debit (far > near)
    max_loss = net_debit * lot_size  # If underlying moves far from strike
    # Max profit is hard to compute analytically for calendars (depends on
    # far-month value when near-month expires). Approximate via payoff scan.
    # At near-month expiry, max profit occurs when underlying = strike.
    far_remaining_T = (far_expiry_days - near_expiry_days) / 365.0
    far_value_at_strike = float(bs_price(strike, strike, far_remaining_T, r, iv_far, "call"))
    max_profit = (far_value_at_strike - net_debit) * lot_size

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.CALENDAR_SPREAD,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=-net_debit,
        max_profit=max(max_profit, 0),
        max_loss=max_loss,
        risk_reward_ratio=max_loss / max(max_profit, 0.01) if max_profit > 0 else float("inf"),
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=near_expiry_days,
        iv_at_entry=(iv_near + iv_far) / 2,
    )


def build_ratio_spread(
    symbol: str,
    spot: float,
    long_strike: float,
    short_strike: float,
    iv: float,
    ratio: int = 2,
    days_to_expiry: int = 30,
    r: float = DEFAULT_RISK_FREE_RATE,
    option_type: str = "call",
) -> StrategyMetrics:
    """
    Ratio Spread: 1×long + N×short (default N=2).

    For calls: Buy 1 lower call, sell 2 higher calls.
    Profit from time decay with directional bias. Has unlimited risk
    beyond a certain point.

    Best in: Mild directional view + high VIX → heavy theta harvest.
    """
    lot_size = get_lot_size(symbol)
    T = days_to_expiry / 365.0

    long_p = float(bs_price(spot, long_strike, T, r, iv, option_type))
    short_p = float(bs_price(spot, short_strike, T, r, iv, option_type))

    if option_type == "call":
        legs = [
            OptionLeg(LegType.LONG_CALL, long_strike, days_to_expiry, 1, long_p, iv),
        ]
        for _ in range(ratio):
            legs.append(
                OptionLeg(LegType.SHORT_CALL, short_strike, days_to_expiry, 1, short_p, iv)
            )
    else:
        legs = [
            OptionLeg(LegType.LONG_PUT, long_strike, days_to_expiry, 1, long_p, iv),
        ]
        for _ in range(ratio):
            legs.append(
                OptionLeg(LegType.SHORT_PUT, short_strike, days_to_expiry, 1, short_p, iv)
            )

    net_premium = short_p * ratio - long_p

    # Max profit/loss via scan
    prices = np.linspace(spot * 0.5, spot * 1.5, 500)
    pnls = np.array([_payoff_at_expiry(legs, p, lot_size) for p in prices])
    max_profit = float(pnls.max())
    max_loss_val = float(pnls.min())

    greeks = _compute_strategy_greeks(legs, spot, r, lot_size)
    margin = _approximate_span_margin(legs, spot, lot_size)
    breakevens = _find_breakevens(legs, spot, lot_size)

    return StrategyMetrics(
        strategy_type=StrategyType.RATIO_SPREAD,
        symbol=symbol, spot=spot, legs=legs, lot_size=lot_size,
        net_premium=net_premium,
        max_profit=max_profit,
        max_loss=abs(max_loss_val) if max_loss_val < 0 else 0,
        risk_reward_ratio=abs(max_loss_val) / max(max_profit, 0.01) if max_loss_val < 0 else 0,
        breakevens=breakevens,
        greeks=greeks,
        margin_required=margin,
        days_to_expiry=days_to_expiry,
        iv_at_entry=iv,
    )
