"""
Regime-Conditioned Strategy Selector
======================================
Maps (HMM regime, VIX regime, signal state) → optimal options strategy set.

This is the core innovation: instead of just adjusting allocation percentages
per regime (the old approach), we now select entirely different strategy
*classes* based on market conditions.

Decision Matrix:
  Regime   × VIX Level → Primary Strategy + Secondary + Hedge

The selector outputs a ranked list of StrategyRecommendation objects.
The RL Allocator (Phase D) will eventually learn to weight these
adaptively; for now, the fixed heuristic matrix captures the domain
knowledge from options literature and practitioner experience.

Capital context: ₹5 crore allows simultaneous deployment of 3-5 index
and 10-15 stock strategies. The selector produces more recommendations
than the position sizer can use, and the sizer picks the top N that fit
the capital budget.

All computation is deterministic — LLMs never touch this (AGENTS.md rule 1).

Reference: docs/ARTHA_ARCHITECTURE.md §4.3, Implementation Plan §Component 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from app.quant.options_strategies import (
    StrategyType, StrategyMetrics,
    build_covered_call, build_protective_put, build_iron_condor,
    build_bull_put_spread, build_bear_call_spread, build_short_strangle,
    build_collar, build_synthetic_short, build_calendar_spread,
    build_ratio_spread,
)
from app.quant.options_pricing import get_lot_size, DEFAULT_RISK_FREE_RATE

logger = logging.getLogger(__name__)


# ── Types ───────────────────────────────────────────────────────────────────────


class RegimeLabel(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class VIXRegime(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class StrategyRole(str, Enum):
    """Role of a strategy in the portfolio."""
    PRIMARY = "PRIMARY"       # Core alpha generator
    SECONDARY = "SECONDARY"   # Complementary, diversifying
    HEDGE = "HEDGE"           # Protective, capital preservation
    DEFENSE = "DEFENSE"       # Tail risk insurance


@dataclass
class StrategyRecommendation:
    """A recommended strategy with context about why it's selected."""
    strategy_type: StrategyType
    role: StrategyRole
    regime: RegimeLabel
    vix_regime: VIXRegime
    rationale: str
    confidence: float = 0.5   # [0, 1] — how strongly the selector recommends this
    sizing_hint_pct: float = 0.0  # Suggested % of options capital to allocate

    # Parameters to pass to the strategy builder
    strike_selection: str = "auto"  # How to pick strikes
    target_delta: float = 0.0      # Target delta for strike selection
    min_credit_pct: float = 0.0    # Minimum credit as % of margin for entry


# ── The Decision Matrix ─────────────────────────────────────────────────────────

# Each (RegimeLabel, VIXRegime) pair maps to a list of strategy recommendations.
# The VIX thresholds are dynamic (percentile-based), not fixed VIX levels.
# See classify_vix_regime() in options_signals.py.

STRATEGY_MATRIX: dict[tuple[RegimeLabel, VIXRegime], list[StrategyRecommendation]] = {

    # ── BULL + LOW VIX ───────────────────────────────────────────────────
    # Steady uptrend, low vol. Premium is cheap — not ideal for selling.
    # Focus: harvest some theta via covered calls, be bullish via spreads.
    (RegimeLabel.BULL, VIXRegime.LOW): [
        StrategyRecommendation(
            strategy_type=StrategyType.COVERED_CALL,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.LOW,
            rationale="Sell OTM calls against long stock holdings. Low premium "
                      "but safe in steady bull. Pick 1σ OTM (15-delta) calls to "
                      "avoid being called away too often.",
            confidence=0.7,
            sizing_hint_pct=40.0,
            target_delta=0.15,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.LOW,
            rationale="Collect small credit, high probability of profit in "
                      "trending market. Use 10-delta short puts.",
            confidence=0.5,
            sizing_hint_pct=20.0,
            target_delta=0.10,
        ),
    ],

    # ── BULL + MEDIUM VIX ────────────────────────────────────────────────
    # Bull trend with normal volatility. Decent premium available.
    (RegimeLabel.BULL, VIXRegime.MEDIUM): [
        StrategyRecommendation(
            strategy_type=StrategyType.COVERED_CALL,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.MEDIUM,
            rationale="Good premium on OTM calls in moderate-vol bull market. "
                      "Use 25-delta calls for higher credit.",
            confidence=0.8,
            sizing_hint_pct=35.0,
            target_delta=0.25,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.MEDIUM,
            rationale="Decent credit with high POP in bullish environment.",
            confidence=0.7,
            sizing_hint_pct=25.0,
            target_delta=0.15,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.COLLAR,
            role=StrategyRole.HEDGE,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.MEDIUM,
            rationale="Protect concentrated positions. Medium VIX makes puts "
                      "affordable; call premium offsets some cost.",
            confidence=0.4,
            sizing_hint_pct=10.0,
        ),
    ],

    # ── BULL + HIGH VIX ──────────────────────────────────────────────────
    # Bull trend but elevated fear. Caution — possible transition.
    # Rich premiums but could be a bull trap.
    (RegimeLabel.BULL, VIXRegime.HIGH): [
        StrategyRecommendation(
            strategy_type=StrategyType.COVERED_CALL,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.HIGH,
            rationale="Very rich call premiums. Sell wider OTM calls (10-delta) "
                      "for large credit. High VIX may signal transition — don't "
                      "sell too close to ATM.",
            confidence=0.7,
            sizing_hint_pct=25.0,
            target_delta=0.10,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.COLLAR,
            role=StrategyRole.HEDGE,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.HIGH,
            rationale="High VIX in a bull market can precede a correction. "
                      "Lock in profits on winners with zero-cost collars.",
            confidence=0.8,
            sizing_hint_pct=30.0,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            role=StrategyRole.DEFENSE,
            regime=RegimeLabel.BULL, vix_regime=VIXRegime.HIGH,
            rationale="Tail insurance — puts are expensive in high VIX but "
                      "that's when you need them most.",
            confidence=0.5,
            sizing_hint_pct=10.0,
        ),
    ],

    # ── SIDEWAYS + LOW VIX ───────────────────────────────────────────────
    # Range-bound, low vol. Premium is low — calendar spreads exploit
    # term structure contango (far-month decays slower than near-month).
    (RegimeLabel.SIDEWAYS, VIXRegime.LOW): [
        StrategyRecommendation(
            strategy_type=StrategyType.IRON_CONDOR,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.LOW,
            rationale="Range-bound market — sell both sides. Low VIX means "
                      "tighter wings needed. Use 1σ strikes (15-delta).",
            confidence=0.6,
            sizing_hint_pct=30.0,
            target_delta=0.15,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.LOW,
            rationale="Exploit contango term structure. Near-month decays faster. "
                      "Best when vol is low and expected to stay low.",
            confidence=0.6,
            sizing_hint_pct=25.0,
        ),
    ],

    # ── SIDEWAYS + MEDIUM VIX ────────────────────────────────────────────
    # Sweet spot for premium selling — range-bound with decent premiums.
    (RegimeLabel.SIDEWAYS, VIXRegime.MEDIUM): [
        StrategyRecommendation(
            strategy_type=StrategyType.IRON_CONDOR,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.MEDIUM,
            rationale="Optimal conditions for iron condors — range-bound with "
                      "enough premium to make the trade worthwhile. Use 1σ wings.",
            confidence=0.85,
            sizing_hint_pct=35.0,
            target_delta=0.20,
            min_credit_pct=0.30,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.SHORT_STRANGLE,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.MEDIUM,
            rationale="Higher credit than condor (no wings) but add wings for "
                      "defined risk. Use 20-delta strikes.",
            confidence=0.7,
            sizing_hint_pct=20.0,
            target_delta=0.20,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.MEDIUM,
            rationale="Diversify premium selling with time-spread strategies.",
            confidence=0.5,
            sizing_hint_pct=15.0,
        ),
    ],

    # ── SIDEWAYS + HIGH VIX ──────────────────────────────────────────────
    # Best VRP harvest opportunity — high vol + range-bound = massive theta.
    # This is where the pandemic-era premium sellers made crores.
    (RegimeLabel.SIDEWAYS, VIXRegime.HIGH): [
        StrategyRecommendation(
            strategy_type=StrategyType.SHORT_STRANGLE,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.HIGH,
            rationale="Maximum VRP harvest — implied vol far exceeds realized in "
                      "range-bound high-VIX markets. Add wings for defined risk. "
                      "Use 1.5σ strikes for wider range.",
            confidence=0.9,
            sizing_hint_pct=30.0,
            target_delta=0.15,
            min_credit_pct=0.35,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.IRON_CONDOR,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.HIGH,
            rationale="Rich premiums on wide iron condors. Set wings 2σ apart.",
            confidence=0.85,
            sizing_hint_pct=30.0,
            target_delta=0.15,
            min_credit_pct=0.40,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            role=StrategyRole.DEFENSE,
            regime=RegimeLabel.SIDEWAYS, vix_regime=VIXRegime.HIGH,
            rationale="High VIX can transition to BEAR. Small tail hedge.",
            confidence=0.4,
            sizing_hint_pct=5.0,
        ),
    ],

    # ── BEAR + LOW VIX ───────────────────────────────────────────────────
    # Unusual combo — suggests slow grind down or early bear (vol hasn't
    # expanded yet). Be cautious — vol expansion may follow.
    (RegimeLabel.BEAR, VIXRegime.LOW): [
        StrategyRecommendation(
            strategy_type=StrategyType.BEAR_CALL_SPREAD,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.LOW,
            rationale="Express bearish view with defined risk. Low VIX means "
                      "less premium but trend is down.",
            confidence=0.7,
            sizing_hint_pct=30.0,
            target_delta=0.20,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            role=StrategyRole.HEDGE,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.LOW,
            rationale="Puts are cheap in low VIX — buy protection before vol "
                      "expands. This is the cheapest insurance you'll get.",
            confidence=0.8,
            sizing_hint_pct=15.0,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.SYNTHETIC_SHORT,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.LOW,
            rationale="Synthetic short on weakest sector names. Low VIX makes "
                      "the put leg cheap.",
            confidence=0.5,
            sizing_hint_pct=15.0,
        ),
    ],

    # ── BEAR + MEDIUM VIX ────────────────────────────────────────────────
    # Confirmed bear with moderate fear. Classic bearish strategy territory.
    (RegimeLabel.BEAR, VIXRegime.MEDIUM): [
        StrategyRecommendation(
            strategy_type=StrategyType.BEAR_CALL_SPREAD,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.MEDIUM,
            rationale="Bear call spreads profit from decline with defined risk. "
                      "Medium VIX provides decent credit.",
            confidence=0.85,
            sizing_hint_pct=30.0,
            target_delta=0.25,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.SYNTHETIC_SHORT,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.MEDIUM,
            rationale="Synthetic shorts on names with weakest fundamentals. "
                      "Higher conviction bearish bet.",
            confidence=0.6,
            sizing_hint_pct=15.0,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            role=StrategyRole.HEDGE,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.MEDIUM,
            rationale="Protect remaining longs with puts.",
            confidence=0.7,
            sizing_hint_pct=15.0,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.COLLAR,
            role=StrategyRole.DEFENSE,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.MEDIUM,
            rationale="Collar positions you're holding through the downturn.",
            confidence=0.6,
            sizing_hint_pct=10.0,
        ),
    ],

    # ── BEAR + HIGH VIX ──────────────────────────────────────────────────
    # Crisis mode — COVID March 2020, 2008, etc.
    # Extreme VRP but also extreme risk. Cautious premium selling only
    # on defined-risk structures. Priority: survival.
    (RegimeLabel.BEAR, VIXRegime.HIGH): [
        StrategyRecommendation(
            strategy_type=StrategyType.BEAR_CALL_SPREAD,
            role=StrategyRole.PRIMARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.HIGH,
            rationale="High VIX + Bear = very rich call premiums on the upside. "
                      "Sell call spreads — the upside is capped by the bear trend, "
                      "and you collect massive credit.",
            confidence=0.9,
            sizing_hint_pct=25.0,
            target_delta=0.20,
            min_credit_pct=0.40,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            role=StrategyRole.DEFENSE,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.HIGH,
            rationale="Puts are expensive but this is when you NEED them. "
                      "Protect all remaining long equity.",
            confidence=0.9,
            sizing_hint_pct=20.0,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.IRON_CONDOR,
            role=StrategyRole.SECONDARY,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.HIGH,
            rationale="Very wide iron condors — 2σ+ wings. Massive credit but "
                      "ONLY if you believe the worst is priced in. Defined risk.",
            confidence=0.4,  # Lower confidence — risky in bear+high VIX
            sizing_hint_pct=10.0,
            target_delta=0.10,
            min_credit_pct=0.45,
        ),
        StrategyRecommendation(
            strategy_type=StrategyType.COLLAR,
            role=StrategyRole.DEFENSE,
            regime=RegimeLabel.BEAR, vix_regime=VIXRegime.HIGH,
            rationale="Zero-cost collars on everything still held long.",
            confidence=0.8,
            sizing_hint_pct=15.0,
        ),
    ],
}


# ── Strategy Selection Logic ────────────────────────────────────────────────────


def select_strategies(
    regime: str,
    vix_regime: str,
    vrp_signal: float = 0.0,
    iv_percentile: float = 50.0,
    portfolio_delta: float = 0.0,
) -> list[StrategyRecommendation]:
    """
    Select options strategies for the current market conditions.

    Parameters
    ----------
    regime : "BULL", "BEAR", or "SIDEWAYS" (from HMM regime detector)
    vix_regime : "LOW", "MEDIUM", or "HIGH" (from VIX percentile classifier)
    vrp_signal : current VRP (IV - RV) / RV — positive means options are rich
    iv_percentile : current IV percentile rank (0-100)
    portfolio_delta : current portfolio net delta (for hedging decisions)

    Returns
    -------
    List of StrategyRecommendation objects, sorted by confidence descending.
    """
    try:
        regime_key = RegimeLabel(regime.upper())
    except ValueError:
        logger.warning("Unknown regime '%s', defaulting to SIDEWAYS", regime)
        regime_key = RegimeLabel.SIDEWAYS

    try:
        vix_key = VIXRegime(vix_regime.upper())
    except ValueError:
        logger.warning("Unknown VIX regime '%s', defaulting to MEDIUM", vix_regime)
        vix_key = VIXRegime.MEDIUM

    # Look up base recommendations
    recommendations = list(STRATEGY_MATRIX.get(
        (regime_key, vix_key), []
    ))

    if not recommendations:
        logger.warning(
            "No strategies for regime=%s, vix=%s",
            regime_key.value, vix_key.value,
        )
        return []

    # Adjust confidence based on signal context
    for rec in recommendations:
        # VRP boost: if VRP is very high, premium-selling strategies get a boost
        if vrp_signal > 0.3 and rec.strategy_type in (
            StrategyType.SHORT_STRANGLE, StrategyType.IRON_CONDOR,
            StrategyType.COVERED_CALL, StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
        ):
            rec.confidence = min(rec.confidence + 0.1, 1.0)

        # VRP negative: don't sell premium when options are cheap
        if vrp_signal < -0.1 and rec.strategy_type in (
            StrategyType.SHORT_STRANGLE, StrategyType.IRON_CONDOR,
        ):
            rec.confidence = max(rec.confidence - 0.2, 0.1)

        # IV percentile extremes
        if iv_percentile > 80 and rec.role in (StrategyRole.PRIMARY, StrategyRole.SECONDARY):
            # Very high IV — selling premium is more attractive
            if rec.strategy_type in (
                StrategyType.SHORT_STRANGLE, StrategyType.IRON_CONDOR,
                StrategyType.BEAR_CALL_SPREAD,
            ):
                rec.confidence = min(rec.confidence + 0.05, 1.0)

        if iv_percentile < 20 and rec.strategy_type in (
            StrategyType.PROTECTIVE_PUT,
        ):
            # Very low IV — puts are cheap, great for hedging
            rec.confidence = min(rec.confidence + 0.15, 1.0)

        # Delta hedging: if portfolio is already very long, boost hedges
        if portfolio_delta > 0.5 and rec.role in (StrategyRole.HEDGE, StrategyRole.DEFENSE):
            rec.confidence = min(rec.confidence + 0.1, 1.0)

    # Sort by confidence (primary sort) then by role priority
    role_priority = {
        StrategyRole.PRIMARY: 0,
        StrategyRole.SECONDARY: 1,
        StrategyRole.HEDGE: 2,
        StrategyRole.DEFENSE: 3,
    }
    recommendations.sort(
        key=lambda r: (-r.confidence, role_priority.get(r.role, 9)),
    )

    return recommendations


def get_strategy_summary(recommendations: list[StrategyRecommendation]) -> str:
    """Format strategy recommendations as a human-readable summary."""
    if not recommendations:
        return "No strategies recommended for current conditions."

    lines = []
    for i, rec in enumerate(recommendations, 1):
        lines.append(
            f"{i}. [{rec.role.value}] {rec.strategy_type.value} "
            f"(confidence: {rec.confidence:.0%}, sizing: {rec.sizing_hint_pct:.0f}%)\n"
            f"   {rec.rationale}"
        )
    return "\n\n".join(lines)


def build_strategy_for_symbol(
    recommendation: StrategyRecommendation,
    symbol: str,
    spot: float,
    iv: float,
    days_to_expiry: int = 30,
) -> Optional[StrategyMetrics]:
    """
    Construct a concrete StrategyMetrics from a StrategyRecommendation.

    Automatically selects strikes based on the recommendation's target_delta
    and the current spot/IV.

    Parameters
    ----------
    recommendation : the selected strategy recommendation
    symbol : NSE symbol
    spot : current spot price
    iv : current implied volatility (annualized, decimal)
    days_to_expiry : DTE for the options

    Returns
    -------
    StrategyMetrics or None if the strategy can't be constructed.
    """
    r = DEFAULT_RISK_FREE_RATE
    target_d = recommendation.target_delta or 0.20

    # Strike distance from spot based on target delta
    # Approximate: for a normal distribution, delta≈N(d1), so
    # strike ≈ spot * exp(±z * sigma * sqrt(T)) where z = N^-1(target_d)
    T = days_to_expiry / 365.0
    from scipy.stats import norm
    z = norm.ppf(1 - target_d)  # OTM distance in std devs
    otm_distance = spot * iv * np.sqrt(T) * z

    try:
        if recommendation.strategy_type == StrategyType.COVERED_CALL:
            call_strike = round(spot + otm_distance, -1)  # Round to nearest 10
            return build_covered_call(symbol, spot, call_strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.PROTECTIVE_PUT:
            put_strike = round(spot - otm_distance, -1)
            return build_protective_put(symbol, spot, put_strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.IRON_CONDOR:
            put_short = round(spot - otm_distance, -1)
            put_long = round(put_short - otm_distance * 0.5, -1)
            call_short = round(spot + otm_distance, -1)
            call_long = round(call_short + otm_distance * 0.5, -1)
            return build_iron_condor(
                symbol, spot, put_short, put_long, call_short, call_long,
                iv, days_to_expiry, r,
            )

        elif recommendation.strategy_type == StrategyType.BULL_PUT_SPREAD:
            short_strike = round(spot - otm_distance, -1)
            long_strike = round(short_strike - otm_distance * 0.5, -1)
            return build_bull_put_spread(symbol, spot, short_strike, long_strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.BEAR_CALL_SPREAD:
            short_strike = round(spot + otm_distance, -1)
            long_strike = round(short_strike + otm_distance * 0.5, -1)
            return build_bear_call_spread(symbol, spot, short_strike, long_strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.SHORT_STRANGLE:
            put_strike = round(spot - otm_distance, -1)
            call_strike = round(spot + otm_distance, -1)
            wing_width = round(otm_distance * 0.5, -1)
            return build_short_strangle(
                symbol, spot, put_strike, call_strike, iv,
                days_to_expiry, r, add_wings=True, wing_width=max(wing_width, 50),
            )

        elif recommendation.strategy_type == StrategyType.COLLAR:
            put_strike = round(spot - otm_distance, -1)
            call_strike = round(spot + otm_distance, -1)
            return build_collar(symbol, spot, put_strike, call_strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.SYNTHETIC_SHORT:
            strike = round(spot, -1)  # ATM
            return build_synthetic_short(symbol, spot, strike, iv, days_to_expiry, r)

        elif recommendation.strategy_type == StrategyType.CALENDAR_SPREAD:
            strike = round(spot, -1)  # ATM
            return build_calendar_spread(
                symbol, spot, strike, iv, iv * 0.95,  # Far-month slightly lower IV
                near_expiry_days=min(days_to_expiry, 7),
                far_expiry_days=max(days_to_expiry, 30),
                r=r,
            )

        elif recommendation.strategy_type == StrategyType.RATIO_SPREAD:
            long_strike = round(spot, -1)  # ATM
            short_strike = round(spot + otm_distance, -1)
            return build_ratio_spread(symbol, spot, long_strike, short_strike, iv, 2, days_to_expiry, r)

        else:
            logger.warning("Unknown strategy type: %s", recommendation.strategy_type)
            return None

    except Exception as e:
        logger.error(
            "Failed to build %s for %s: %s",
            recommendation.strategy_type.value, symbol, e,
        )
        return None
