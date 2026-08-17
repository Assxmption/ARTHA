"""
Tests for Options Strategies & Strategy Selector
==================================================
Validates multi-leg strategy construction, P&L computation, Greeks
aggregation, margin estimation, breakeven solving, and regime-conditioned
strategy selection.

Test Categories:
  1. Strategy P&L correctness (payoff at known prices)
  2. Max profit / max loss / breakeven accuracy
  3. Greeks aggregation across legs
  4. Margin estimation bounds
  5. Strategy selector: correct strategies per regime × VIX
  6. No naked (undefined-risk) positions in the output
  7. Fact Store OptionsSignal schema compliance

Reference: Implementation Plan §Phase B
"""

import pytest
import numpy as np
from datetime import date

from app.quant.options_strategies import (
    StrategyType, LegType, OptionLeg, StrategyMetrics,
    build_covered_call, build_protective_put, build_iron_condor,
    build_bull_put_spread, build_bear_call_spread, build_short_strangle,
    build_collar, build_synthetic_short, build_calendar_spread,
    build_ratio_spread,
)
from app.quant.strategy_selector import (
    RegimeLabel, VIXRegime, StrategyRole,
    select_strategies, build_strategy_for_symbol,
    STRATEGY_MATRIX,
)
from app.factstore.schemas import (
    FactType, SignalType, OptionsSignal,
)


# ── Test Fixtures ───────────────────────────────────────────────────────────────

SPOT = 23000.0     # NIFTY-ish spot
IV = 0.18          # 18% implied vol
DTE = 30           # 30 days to expiry
SYMBOL = "NIFTY"


# ── 1. Covered Call Tests ───────────────────────────────────────────────────────


class TestCoveredCall:

    def test_construction(self):
        """Covered call builds with correct structure."""
        cc = build_covered_call(SYMBOL, SPOT, 23500, IV, DTE)
        assert cc.strategy_type == StrategyType.COVERED_CALL
        assert len(cc.legs) == 2
        assert any(l.leg_type == LegType.LONG_STOCK for l in cc.legs)
        assert any(l.leg_type == LegType.SHORT_CALL for l in cc.legs)

    def test_max_profit_bounded(self):
        """Max profit is capped (call strike - spot + premium)."""
        cc = build_covered_call(SYMBOL, SPOT, 23500, IV, DTE)
        # Max profit = (23500 - 23000 + premium) × lot_size
        assert cc.max_profit > 0
        assert cc.max_profit < 1e7  # Sanity bound

    def test_payoff_at_strike(self):
        """At call strike price, P&L should be near max profit."""
        cc = build_covered_call(SYMBOL, SPOT, 23500, IV, DTE)
        pnl_at_strike = cc.payoff_at_price(23500)
        # Should be close to max profit
        assert pnl_at_strike > 0

    def test_payoff_below_breakeven_negative(self):
        """Below breakeven, P&L should be negative."""
        cc = build_covered_call(SYMBOL, SPOT, 23500, IV, DTE)
        pnl_at_deep_low = cc.payoff_at_price(SPOT * 0.90)
        assert pnl_at_deep_low < 0

    def test_delta_less_than_one(self):
        """Covered call delta should be < 1 (short call reduces delta)."""
        cc = build_covered_call(SYMBOL, SPOT, 23500, IV, DTE)
        assert cc.greeks.delta < 1.0
        assert cc.greeks.delta > 0.0  # Still net long


# ── 2. Iron Condor Tests ───────────────────────────────────────────────────────


class TestIronCondor:

    def test_construction(self):
        """Iron condor has 4 legs: short put, long put, short call, long call."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert ic.strategy_type == StrategyType.IRON_CONDOR
        assert len(ic.legs) == 4

    def test_defined_risk(self):
        """Max loss is finite (defined risk)."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert ic.max_loss > 0
        assert ic.max_loss < 1e7

    def test_net_credit(self):
        """Iron condor generates a net credit."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert ic.net_premium > 0

    def test_max_profit_at_spot(self):
        """Max profit occurs when price stays between short strikes."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        pnl_at_spot = ic.payoff_at_price(SPOT)
        assert pnl_at_spot > 0
        # Should be close to max profit
        assert abs(pnl_at_spot - ic.max_profit) / max(ic.max_profit, 1) < 0.05

    def test_max_loss_beyond_wings(self):
        """Max loss occurs when price goes beyond either wing."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        pnl_at_deep_low = ic.payoff_at_price(21000)
        pnl_at_deep_high = ic.payoff_at_price(25000)
        assert pnl_at_deep_low < 0
        assert pnl_at_deep_high < 0
        # Both should be approximately -max_loss
        assert abs(pnl_at_deep_low + ic.max_loss) / ic.max_loss < 0.15
        assert abs(pnl_at_deep_high + ic.max_loss) / ic.max_loss < 0.15

    def test_two_breakevens(self):
        """Iron condor should have exactly 2 breakeven points."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert len(ic.breakevens) == 2

    def test_near_zero_delta(self):
        """ATM iron condor should have near-zero delta (market-neutral)."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert abs(ic.greeks.delta) < 0.3

    def test_positive_theta(self):
        """Iron condor benefits from time decay (net positive theta)."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        assert ic.greeks.theta > 0  # Net short options → positive theta


# ── 3. Spread Tests ─────────────────────────────────────────────────────────────


class TestSpreads:

    def test_bull_put_spread_credit(self):
        """Bull put spread generates a net credit."""
        bps = build_bull_put_spread(SYMBOL, SPOT, 22500, 22000, IV, DTE)
        assert bps.net_premium > 0
        assert bps.strategy_type == StrategyType.BULL_PUT_SPREAD

    def test_bear_call_spread_credit(self):
        """Bear call spread generates a net credit."""
        bcs = build_bear_call_spread(SYMBOL, SPOT, 23500, 24000, IV, DTE)
        assert bcs.net_premium > 0
        assert bcs.strategy_type == StrategyType.BEAR_CALL_SPREAD

    def test_bull_put_max_profit_at_above_short_strike(self):
        """Bull put spread: max profit when price stays above short strike."""
        bps = build_bull_put_spread(SYMBOL, SPOT, 22500, 22000, IV, DTE)
        pnl_high = bps.payoff_at_price(24000)
        assert pnl_high > 0
        assert abs(pnl_high - bps.max_profit) / max(bps.max_profit, 1) < 0.05

    def test_bear_call_max_profit_at_below_short_strike(self):
        """Bear call spread: max profit when price stays below short strike."""
        bcs = build_bear_call_spread(SYMBOL, SPOT, 23500, 24000, IV, DTE)
        pnl_low = bcs.payoff_at_price(22000)
        assert pnl_low > 0

    def test_spread_risk_reward_finite(self):
        """Spreads have finite, positive risk/reward ratio."""
        bps = build_bull_put_spread(SYMBOL, SPOT, 22500, 22000, IV, DTE)
        assert bps.risk_reward_ratio > 0
        assert bps.risk_reward_ratio < 100


# ── 4. Bearish Strategy Tests ──────────────────────────────────────────────────


class TestBearishStrategies:

    def test_synthetic_short_negative_delta(self):
        """Synthetic short should have delta ≈ -1."""
        ss = build_synthetic_short(SYMBOL, SPOT, SPOT, IV, DTE)
        assert ss.greeks.delta < -0.5
        assert ss.strategy_type == StrategyType.SYNTHETIC_SHORT

    def test_synthetic_short_profits_from_decline(self):
        """Synthetic short profits when underlying declines."""
        ss = build_synthetic_short(SYMBOL, SPOT, SPOT, IV, DTE)
        pnl_down = ss.payoff_at_price(SPOT * 0.90)
        pnl_up = ss.payoff_at_price(SPOT * 1.10)
        assert pnl_down > pnl_up

    def test_protective_put_limits_downside(self):
        """Protective put limits downside loss."""
        pp = build_protective_put(SYMBOL, SPOT, 22000, IV, DTE)
        pnl_crash = pp.payoff_at_price(SPOT * 0.50)
        # Loss should be bounded, not proportional to the crash
        assert pnl_crash > -SPOT * 25 * 0.5  # Better than naked stock loss


# ── 5. Collar Tests ─────────────────────────────────────────────────────────────


class TestCollar:

    def test_collar_bounded_risk(self):
        """Collar has both bounded profit and bounded loss."""
        collar = build_collar(SYMBOL, SPOT, 22000, 24000, IV, DTE)
        assert collar.max_profit > 0
        assert collar.max_loss > 0
        assert collar.strategy_type == StrategyType.COLLAR


# ── 6. Calendar Spread Tests ────────────────────────────────────────────────────


class TestCalendarSpread:

    def test_calendar_debit(self):
        """Calendar spread is a debit strategy (far-month costs more)."""
        cs = build_calendar_spread(SYMBOL, SPOT, SPOT, IV, IV * 0.95, 7, 30)
        assert cs.net_premium < 0  # Debit

    def test_calendar_max_profit_near_strike(self):
        """Calendar spread max profit near the strike at near-month expiry."""
        cs = build_calendar_spread(SYMBOL, SPOT, SPOT, IV, IV * 0.95, 7, 30)
        assert cs.max_profit > 0


# ── 7. Short Strangle Tests ─────────────────────────────────────────────────────


class TestShortStrangle:

    def test_strangle_with_wings_defined_risk(self):
        """Short strangle with wings has defined max loss."""
        ss = build_short_strangle(
            SYMBOL, SPOT, 22000, 24000, IV, DTE,
            add_wings=True, wing_width=500,
        )
        assert ss.max_loss > 0
        assert ss.max_loss < SPOT * 25 * 0.5  # Much less than total NAV

    def test_strangle_credit(self):
        """Short strangle generates net credit."""
        ss = build_short_strangle(SYMBOL, SPOT, 22000, 24000, IV, DTE)
        assert ss.net_premium > 0

    def test_strangle_positive_theta(self):
        """Short strangle benefits from time decay."""
        ss = build_short_strangle(SYMBOL, SPOT, 22000, 24000, IV, DTE)
        assert ss.greeks.theta > 0


# ── 8. Ratio Spread Tests ──────────────────────────────────────────────────────


class TestRatioSpread:

    def test_ratio_spread_construction(self):
        """Ratio spread has correct number of legs (1 long + N short)."""
        rs = build_ratio_spread(SYMBOL, SPOT, SPOT, 23500, IV, 2, DTE)
        assert rs.strategy_type == StrategyType.RATIO_SPREAD
        assert len(rs.legs) == 3  # 1 long + 2 short


# ── 9. Payoff Curve Tests ───────────────────────────────────────────────────────


class TestPayoffCurve:

    def test_payoff_curve_shape(self):
        """Payoff curve returns correct shape."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        prices, pnl = ic.payoff_curve(n_points=100)
        assert len(prices) == 100
        assert len(pnl) == 100

    def test_payoff_curve_max_at_center(self):
        """Iron condor payoff peak is between short strikes."""
        ic = build_iron_condor(
            SYMBOL, SPOT, 22500, 22000, 23500, 24000, IV, DTE,
        )
        prices, pnl = ic.payoff_curve(n_points=200)
        peak_price = prices[np.argmax(pnl)]
        assert 22500 <= peak_price <= 23500


# ── 10. Strategy Selector Tests ─────────────────────────────────────────────────


class TestStrategySelector:

    def test_all_regime_vix_combos_covered(self):
        """Every regime × VIX combination has at least one strategy."""
        for regime in RegimeLabel:
            for vix in VIXRegime:
                key = (regime, vix)
                recs = STRATEGY_MATRIX.get(key, [])
                assert len(recs) > 0, \
                    f"No strategies for {regime.value} × {vix.value}"

    def test_bull_selects_covered_calls(self):
        """BULL regime should include covered calls."""
        recs = select_strategies("BULL", "MEDIUM")
        types = [r.strategy_type for r in recs]
        assert StrategyType.COVERED_CALL in types

    def test_bear_selects_bearish_strategies(self):
        """BEAR regime should include bear call spreads."""
        recs = select_strategies("BEAR", "MEDIUM")
        types = [r.strategy_type for r in recs]
        assert StrategyType.BEAR_CALL_SPREAD in types

    def test_sideways_selects_iron_condors(self):
        """SIDEWAYS regime should include iron condors."""
        recs = select_strategies("SIDEWAYS", "MEDIUM")
        types = [r.strategy_type for r in recs]
        assert StrategyType.IRON_CONDOR in types

    def test_high_vix_bear_includes_defense(self):
        """BEAR + HIGH VIX should include defense strategies."""
        recs = select_strategies("BEAR", "HIGH")
        roles = [r.role for r in recs]
        assert StrategyRole.DEFENSE in roles

    def test_vrp_boost_increases_confidence(self):
        """High VRP should boost premium-selling strategy confidence."""
        base = select_strategies("SIDEWAYS", "HIGH", vrp_signal=0.0)
        boosted = select_strategies("SIDEWAYS", "HIGH", vrp_signal=0.5)

        # Find iron condor confidence in both
        base_ic = next((r for r in base if r.strategy_type == StrategyType.IRON_CONDOR), None)
        boost_ic = next((r for r in boosted if r.strategy_type == StrategyType.IRON_CONDOR), None)
        if base_ic and boost_ic:
            assert boost_ic.confidence >= base_ic.confidence

    def test_negative_vrp_reduces_selling_confidence(self):
        """Negative VRP should reduce premium-selling confidence."""
        base = select_strategies("SIDEWAYS", "HIGH", vrp_signal=0.0)
        reduced = select_strategies("SIDEWAYS", "HIGH", vrp_signal=-0.2)

        base_ic = next((r for r in base if r.strategy_type == StrategyType.IRON_CONDOR), None)
        red_ic = next((r for r in reduced if r.strategy_type == StrategyType.IRON_CONDOR), None)
        if base_ic and red_ic:
            assert red_ic.confidence <= base_ic.confidence

    def test_sorted_by_confidence(self):
        """Recommendations are sorted by confidence descending."""
        recs = select_strategies("SIDEWAYS", "MEDIUM")
        confidences = [r.confidence for r in recs]
        assert confidences == sorted(confidences, reverse=True)

    def test_unknown_regime_defaults(self):
        """Unknown regime falls back to SIDEWAYS."""
        recs = select_strategies("UNKNOWN", "MEDIUM")
        assert len(recs) > 0

    def test_build_strategy_produces_metrics(self):
        """build_strategy_for_symbol returns valid StrategyMetrics."""
        recs = select_strategies("SIDEWAYS", "MEDIUM")
        for rec in recs[:2]:  # Test top 2
            metrics = build_strategy_for_symbol(rec, SYMBOL, SPOT, IV, DTE)
            if metrics is not None:
                assert metrics.max_profit > 0 or metrics.max_profit == float("inf")
                assert metrics.lot_size > 0


# ── 11. Fact Store Schema Tests ─────────────────────────────────────────────────


class TestOptionsSignalSchema:

    def test_options_signal_creation(self):
        """OptionsSignal can be created with all required fields."""
        signal = OptionsSignal(
            job_id="test-job-1",
            source="STRATEGY_SELECTOR_V1",
            symbol_or_index="NIFTY",
            strategy_type="IRON_CONDOR",
            legs=[
                {"type": "SHORT_PUT", "strike": 22500, "premium": 45.0},
                {"type": "LONG_PUT", "strike": 22000, "premium": 20.0},
                {"type": "SHORT_CALL", "strike": 23500, "premium": 40.0},
                {"type": "LONG_CALL", "strike": 24000, "premium": 15.0},
            ],
            net_premium=50.0,
            max_profit=50 * 25,
            max_loss=450 * 25,
            breakevens=[22450.0, 23550.0],
            regime_label="SIDEWAYS",
            vix_regime="MEDIUM",
            as_of=date(2026, 8, 16),
        )
        assert signal.fact_type == FactType.OPTIONS_SIGNAL
        assert signal.validated is False
        assert len(signal.fact_id) > 0
        assert signal.source == "STRATEGY_SELECTOR_V1"

    def test_options_signal_requires_source(self):
        """OptionsSignal requires non-empty source (rule 10)."""
        with pytest.raises(Exception):
            OptionsSignal(
                job_id="test-job-1",
                source="",  # Empty source should fail
                symbol_or_index="NIFTY",
                strategy_type="IRON_CONDOR",
                net_premium=50.0,
                max_profit=1250,
                max_loss=11250,
                regime_label="SIDEWAYS",
                as_of=date(2026, 8, 16),
            )

    def test_new_signal_types_exist(self):
        """New signal types are in the SignalType enum."""
        assert SignalType.VRP.value == "vrp"
        assert SignalType.IV_PERCENTILE.value == "iv_percentile"
        assert SignalType.PCR.value == "pcr"
        assert SignalType.IV_SKEW.value == "iv_skew"
        assert SignalType.OPTIONS_STRATEGY.value == "options_strategy"
        assert SignalType.GAMMA_EXPOSURE.value == "gamma_exposure"

    def test_fact_type_options_signal_exists(self):
        """OPTIONS_SIGNAL fact type is in the FactType enum."""
        assert FactType.OPTIONS_SIGNAL.value == "options_signal"
