"""
Options & Volatility Signal Generator
=======================================
Deterministic volatility-derived alpha signals for the ARTHA options engine.

These signals quantify the state of the options/volatility market and feed
into the Strategy Selector to decide which options strategies to deploy.

Signal Families:
  1. Volatility Risk Premium (VRP)   — the core alpha of options selling
  2. IV Percentile Rank              — historical context for current vol
  3. Put-Call Ratio (PCR)            — contrarian sentiment from OI data
  4. IV Skew                         — fear gauge from put vs call vol
  5. IV Term Structure               — panic detector (backwardation = stress)
  6. Gamma Exposure (GEX)            — dealer positioning → move amplification
  7. Max Pain                        — expiry gravitational pull
  8. OI Concentration                — support/resistance from options OI

All computation is deterministic — LLMs never touch this (AGENTS.md rule 1).
All signals use MAD-robust z-scoring where applicable (AGENTS.md rule 7).

Reference: docs/ARTHA_ARCHITECTURE.md §4.3, Implementation Plan §Component 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Signal Names ────────────────────────────────────────────────────────────────

OPTIONS_SIGNAL_NAMES = [
    "vrp",              # Volatility Risk Premium: (IV - RV) / RV
    "iv_percentile",    # IV percentile rank over trailing 252 days
    "pcr_oi",           # Put-Call ratio (OI-weighted)
    "pcr_volume",       # Put-Call ratio (volume-weighted)
    "iv_skew",          # 25-delta put IV / 25-delta call IV
    "iv_term_slope",    # Near-month IV - Far-month IV (backwardation < 0)
    "gex_normalized",   # Normalized gamma exposure
    "max_pain_dist",    # Distance to max pain as % of spot
    "oi_concentration", # Herfindahl index of OI across strikes
]


@dataclass
class VolatilityState:
    """Snapshot of the volatility regime at a point in time."""
    date: pd.Timestamp
    symbol: str
    realized_vol_20d: float   # 20-day realized vol (annualized)
    realized_vol_60d: float   # 60-day realized vol (annualized)
    implied_vol: float        # ATM implied vol (or VIX for index)
    vrp: float                # IV - RV_20d
    vrp_zscore: float         # Z-score of VRP vs trailing history
    iv_percentile: float      # 0-100 percentile rank
    vix_regime: str           # "LOW" | "MEDIUM" | "HIGH"


# ── Realized Volatility ────────────────────────────────────────────────────────


def compute_realized_vol(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility from close prices.

    Uses log returns and the standard deviation estimator.
    Annualized by default (× sqrt(250)).

    Parameters
    ----------
    prices : close price series (DatetimeIndex)
    window : rolling window in trading days
    annualize : if True, multiply by sqrt(250)

    Returns
    -------
    Rolling realized volatility series.
    """
    log_returns = np.log(prices / prices.shift(1))
    rv = log_returns.rolling(window=window, min_periods=max(window // 2, 5)).std()

    if annualize:
        rv = rv * np.sqrt(250)

    return rv


def compute_yang_zhang_vol(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator — uses OHLC data for a more
    efficient estimate than close-to-close.

    This is superior to simple close-to-close vol because it captures
    intraday volatility information. Efficiency ratio ~7× vs close-to-close.

    The trade-off is that it assumes no drift — acceptable for short windows
    (20d) but less so for longer ones. We use close-to-close for 60d+ windows.

    Reference: Yang, D. & Zhang, Q. (2000), "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices", Journal of
    Business, 73(3).
    """
    # Overnight returns
    log_oc_prev = np.log(open_prices / close_prices.shift(1))
    # Open to close returns
    log_co = np.log(close_prices / open_prices)
    # Rogers-Satchell component
    log_hi = np.log(high_prices / open_prices)
    log_lo = np.log(low_prices / open_prices)
    rs = log_hi * (log_hi - log_co) + log_lo * (log_lo - log_co)

    k = 0.34 / (1 + (window + 1) / (window - 1))

    overnight_var = log_oc_prev.rolling(window).var()
    close_var = log_co.rolling(window).var()
    rs_var = rs.rolling(window).mean()

    yz_var = overnight_var + k * close_var + (1 - k) * rs_var
    yz_vol = np.sqrt(yz_var.clip(lower=0)) * np.sqrt(250)

    return yz_vol


# ── Volatility Risk Premium ────────────────────────────────────────────────────


def compute_vrp(
    implied_vol: pd.Series,
    realized_vol: pd.Series,
) -> pd.Series:
    """
    Volatility Risk Premium: (IV - RV) / RV.

    When VRP > 0, options are "expensive" relative to realized moves →
    selling premium is favorable. When VRP < 0, options are "cheap" →
    buying protection is favorable.

    The VRP is the core alpha source for systematic options selling.
    Academic literature reports it averaging ~20-30% for equity indices
    (i.e., IV is typically 1.2-1.3× RV), with significant variation
    across regimes.

    Parameters
    ----------
    implied_vol : IV series (ATM or VIX)
    realized_vol : realized vol series (same frequency)

    Returns
    -------
    VRP series. Positive = options expensive, negative = options cheap.
    """
    # Avoid division by zero
    rv_safe = realized_vol.clip(lower=0.01)
    return (implied_vol - realized_vol) / rv_safe


def compute_vrp_zscore(
    vrp: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    MAD-robust z-score of VRP vs its trailing distribution.

    Uses Median Absolute Deviation instead of mean/std to be robust to
    the fat-tailed nature of VRP (AGENTS.md rule 7: use MAD for robust
    detection).

    A VRP z-score > 2 means the premium is unusually rich → strong
    sell signal for options sellers. A z-score < -1 means options are
    unusually cheap → consider buying protection.
    """
    median = vrp.rolling(lookback, min_periods=60).median()
    mad = (vrp - median).abs().rolling(lookback, min_periods=60).median()
    # MAD to standard deviation: multiply by 1.4826
    mad_std = mad * 1.4826
    return (vrp - median) / mad_std.clip(lower=0.001)


# ── IV Percentile Rank ──────────────────────────────────────────────────────────


def compute_iv_percentile(
    iv_series: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    IV percentile rank: what fraction of the trailing `lookback` days
    had a lower IV than today?

    0 = IV at its lowest in a year.
    100 = IV at its highest in a year.

    This is more useful than raw IV because the same IV number means
    different things for different stocks. RELIANCE IV=25% is high;
    TATAMOTORS IV=25% is low.

    Parameters
    ----------
    iv_series : implied volatility series
    lookback : trailing window in trading days (default 252 = 1 year)

    Returns
    -------
    Series of percentile ranks [0, 100].
    """
    def _percentile_rank(window):
        if len(window) < 2:
            return 50.0  # Insufficient data
        current = window.iloc[-1]
        rank = (window.iloc[:-1] < current).sum()
        return (rank / (len(window) - 1)) * 100

    return iv_series.rolling(lookback, min_periods=30).apply(
        _percentile_rank, raw=False,
    )


# ── Put-Call Ratio ──────────────────────────────────────────────────────────────


def compute_pcr(
    put_oi: pd.Series | float,
    call_oi: pd.Series | float,
) -> pd.Series | float:
    """
    Put-Call Ratio from open interest.

    PCR > 1.0: more puts than calls → bearish sentiment (contrarian bullish)
    PCR < 0.7: more calls than puts → bullish sentiment (contrarian bearish)
    PCR 0.7-1.0: neutral zone

    Used as a contrarian indicator: extreme PCR values often precede reversals.
    """
    if isinstance(call_oi, pd.Series):
        call_safe = call_oi.clip(lower=1)
        return put_oi / call_safe
    else:
        return put_oi / max(call_oi, 1)


def compute_pcr_from_chain(option_chain: dict) -> dict[str, float]:
    """
    Compute PCR metrics from a raw option chain dict (as returned by nse_fno.py).

    Returns dict with:
      pcr_oi: OI-weighted PCR
      pcr_volume: volume-weighted PCR
      total_call_oi: total call OI
      total_put_oi: total put OI
    """
    calls = option_chain.get("calls", [])
    puts = option_chain.get("puts", [])

    total_call_oi = sum(c.get("openInterest", 0) for c in calls)
    total_put_oi = sum(p.get("openInterest", 0) for p in puts)
    total_call_vol = sum(c.get("volume", 0) for c in calls)
    total_put_vol = sum(p.get("volume", 0) for p in puts)

    return {
        "pcr_oi": total_put_oi / max(total_call_oi, 1),
        "pcr_volume": total_put_vol / max(total_call_vol, 1),
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_call_volume": total_call_vol,
        "total_put_volume": total_put_vol,
    }


# ── IV Skew ─────────────────────────────────────────────────────────────────────


def compute_iv_skew(
    option_chain: dict,
    spot: float,
) -> float:
    """
    IV Skew: ratio of OTM put IV to OTM call IV.

    Specifically: average IV of puts 3-7% OTM / average IV of calls 3-7% OTM.

    Elevated skew (> 1.3) = market is paying a premium for downside protection
    → fear is elevated. This is a signal to be cautious about selling puts,
    and potentially to sell calls instead (or buy put spreads for protection).

    Low skew (< 1.0) = calls are more expensive than puts → bullish market,
    or complacency about downside risk.

    Parameters
    ----------
    option_chain : dict with "calls", "puts", "spotPrice" keys
    spot : current spot price

    Returns
    -------
    IV skew ratio. NaN if insufficient data.
    """
    puts = option_chain.get("puts", [])
    calls = option_chain.get("calls", [])

    if not puts or not calls or spot <= 0:
        return np.nan

    # OTM puts: strikes 3-7% below spot
    otm_put_ivs = [
        p["impliedVolatility"]
        for p in puts
        if 0.93 * spot <= p["strike"] <= 0.97 * spot
        and p.get("impliedVolatility", 0) > 0
    ]

    # OTM calls: strikes 3-7% above spot
    otm_call_ivs = [
        c["impliedVolatility"]
        for c in calls
        if 1.03 * spot <= c["strike"] <= 1.07 * spot
        and c.get("impliedVolatility", 0) > 0
    ]

    if not otm_put_ivs or not otm_call_ivs:
        return np.nan

    avg_put_iv = np.mean(otm_put_ivs)
    avg_call_iv = np.mean(otm_call_ivs)

    if avg_call_iv <= 0:
        return np.nan

    return avg_put_iv / avg_call_iv


# ── IV Term Structure ───────────────────────────────────────────────────────────


def compute_iv_term_slope(
    near_month_iv: float,
    far_month_iv: float,
) -> float:
    """
    IV term structure slope: near_month_iv - far_month_iv.

    Negative = contango (normal market, calm)
    Positive = backwardation (panic, near-term fear)

    Backwardation is a strong signal that the market is pricing in a
    near-term shock. In backwardation:
    - Don't sell near-term options (they're expensive for a reason)
    - Consider selling far-month options or calendar spreads
    - Protective strategies become more important

    Parameters
    ----------
    near_month_iv : ATM IV for the nearest expiry
    far_month_iv : ATM IV for the next expiry

    Returns
    -------
    Term slope. Positive = backwardation (stress), negative = contango (calm).
    """
    if far_month_iv <= 0:
        return np.nan
    return near_month_iv - far_month_iv


# ── Gamma Exposure ──────────────────────────────────────────────────────────────


def compute_gex(
    option_chain: dict,
    spot: float,
    lot_size: int = 25,
) -> dict[str, float]:
    """
    Gamma Exposure (GEX): net dealer gamma across all strikes.

    When dealers are long gamma (positive GEX):
    - They delta-hedge by selling into rallies and buying dips
    - This DAMPENS volatility → good for premium sellers

    When dealers are short gamma (negative GEX):
    - They delta-hedge by buying into rallies and selling dips
    - This AMPLIFIES volatility → bad for premium sellers, good for directional

    This is a simplified model that assumes dealers are net short calls and
    net long puts (the standard retail-sells-to-dealer model). In reality,
    the dealer positioning is more complex, but this captures the dominant effect.

    Parameters
    ----------
    option_chain : dict with "calls", "puts" lists
    spot : current spot price
    lot_size : contract multiplier

    Returns
    -------
    dict with:
      total_gex: total gamma exposure (in spot-dollar terms)
      gex_normalized: GEX / spot (dimensionless, for cross-asset comparison)
      flip_point: spot level where GEX flips from positive to negative
    """
    from app.quant.options_pricing import compute_gamma, DEFAULT_RISK_FREE_RATE

    calls = option_chain.get("calls", [])
    puts = option_chain.get("puts", [])
    r = DEFAULT_RISK_FREE_RATE

    # Estimate T from expiry (crude — 30 days default if no expiry info)
    T = 30.0 / 365.0  # Will be overridden if expiry date is available

    total_gex = 0.0
    strike_gex = {}

    for c in calls:
        strike = c.get("strike", 0)
        oi = c.get("openInterest", 0)
        iv = c.get("impliedVolatility", 0) / 100.0  # Convert from % to decimal
        if strike <= 0 or oi <= 0 or iv <= 0:
            continue

        gamma = float(compute_gamma(spot, strike, T, r, iv))
        # Dealers are short calls → their gamma from calls is negative
        gex_contribution = -gamma * oi * lot_size * spot * 0.01  # per 1% move
        total_gex += gex_contribution
        strike_gex[strike] = strike_gex.get(strike, 0) + gex_contribution

    for p in puts:
        strike = p.get("strike", 0)
        oi = p.get("openInterest", 0)
        iv = p.get("impliedVolatility", 0) / 100.0
        if strike <= 0 or oi <= 0 or iv <= 0:
            continue

        gamma = float(compute_gamma(spot, strike, T, r, iv))
        # Dealers are long puts → their gamma from puts is positive
        gex_contribution = gamma * oi * lot_size * spot * 0.01
        total_gex += gex_contribution
        strike_gex[strike] = strike_gex.get(strike, 0) + gex_contribution

    # Find GEX flip point
    flip_point = spot  # Default
    sorted_strikes = sorted(strike_gex.keys())
    cumulative = 0.0
    for s in sorted_strikes:
        cumulative += strike_gex[s]
        if cumulative <= 0:
            flip_point = s
            break

    return {
        "total_gex": total_gex,
        "gex_normalized": total_gex / max(spot, 1),
        "flip_point": flip_point,
        "n_strikes": len(strike_gex),
    }


# ── Max Pain ────────────────────────────────────────────────────────────────────


def compute_max_pain(option_chain: dict) -> dict[str, float]:
    """
    Max Pain: the strike price at which total option buyer losses are maximized
    (equivalently, where total option writer profits are maximized).

    Near expiry, the underlying price tends to gravitate toward max pain —
    this is the "pinning" effect. It's not a strong enough signal to trade
    alone, but it provides useful context for strike selection.

    Parameters
    ----------
    option_chain : dict with "calls", "puts" lists and "spotPrice"

    Returns
    -------
    dict with:
      max_pain_strike: the strike where total buyer losses are highest
      distance_pct: (max_pain - spot) / spot as a percentage
    """
    calls = option_chain.get("calls", [])
    puts = option_chain.get("puts", [])
    spot = option_chain.get("spotPrice", 0)

    if not calls or not puts or spot <= 0:
        return {"max_pain_strike": 0, "distance_pct": 0}

    all_strikes = sorted(set(
        [c["strike"] for c in calls if c.get("strike", 0) > 0] +
        [p["strike"] for p in puts if p.get("strike", 0) > 0]
    ))

    if not all_strikes:
        return {"max_pain_strike": 0, "distance_pct": 0}

    # Build OI lookup
    call_oi = {c["strike"]: c.get("openInterest", 0) for c in calls}
    put_oi = {p["strike"]: p.get("openInterest", 0) for p in puts}

    min_pain = float("inf")
    max_pain_strike = all_strikes[len(all_strikes) // 2]

    for test_price in all_strikes:
        total_pain = 0.0

        # Call buyer pain: max(0, strike - test_price) doesn't apply;
        # Call buyer pays premium, loses money when underlying < strike
        for strike in all_strikes:
            oi = call_oi.get(strike, 0)
            if oi > 0:
                # Call buyer P&L at test_price (ignoring premium):
                # max(test_price - strike, 0)
                # Call buyer loss = -max(test_price - strike, 0) if OTM at test_price
                # Total call pain = sum of intrinsic values calls would have
                call_itm = max(test_price - strike, 0)
                total_pain += call_itm * oi

        for strike in all_strikes:
            oi = put_oi.get(strike, 0)
            if oi > 0:
                put_itm = max(strike - test_price, 0)
                total_pain += put_itm * oi

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_price

    return {
        "max_pain_strike": max_pain_strike,
        "distance_pct": (max_pain_strike - spot) / spot * 100 if spot > 0 else 0,
    }


# ── OI Concentration ───────────────────────────────────────────────────────────


def compute_oi_concentration(option_chain: dict) -> dict[str, float | list]:
    """
    OI Concentration: identifies strikes with unusually high open interest.

    These strikes act as support (high put OI) and resistance (high call OI)
    levels, especially near expiry. Options sellers want to place strikes
    beyond these levels.

    Returns the Herfindahl-Hirschman Index (HHI) of OI distribution —
    higher HHI means OI is concentrated at fewer strikes.

    Parameters
    ----------
    option_chain : dict with "calls", "puts" lists

    Returns
    -------
    dict with:
      hhi_calls: HHI for call OI distribution
      hhi_puts: HHI for put OI distribution
      top_call_strikes: top 3 strikes by call OI
      top_put_strikes: top 3 strikes by put OI
      call_wall: strike with highest call OI (resistance)
      put_wall: strike with highest put OI (support)
    """
    calls = option_chain.get("calls", [])
    puts = option_chain.get("puts", [])

    # Call OI distribution
    call_ois = [(c["strike"], c.get("openInterest", 0)) for c in calls if c.get("openInterest", 0) > 0]
    put_ois = [(p["strike"], p.get("openInterest", 0)) for p in puts if p.get("openInterest", 0) > 0]

    def _hhi(ois: list[tuple]) -> float:
        if not ois:
            return 0.0
        total = sum(oi for _, oi in ois)
        if total == 0:
            return 0.0
        shares = [(oi / total) ** 2 for _, oi in ois]
        return sum(shares)

    call_ois_sorted = sorted(call_ois, key=lambda x: x[1], reverse=True)
    put_ois_sorted = sorted(put_ois, key=lambda x: x[1], reverse=True)

    return {
        "hhi_calls": _hhi(call_ois),
        "hhi_puts": _hhi(put_ois),
        "top_call_strikes": [s for s, _ in call_ois_sorted[:3]],
        "top_put_strikes": [s for s, _ in put_ois_sorted[:3]],
        "call_wall": call_ois_sorted[0][0] if call_ois_sorted else 0,
        "put_wall": put_ois_sorted[0][0] if put_ois_sorted else 0,
    }


# ── VIX Regime Classifier ──────────────────────────────────────────────────────


def classify_vix_regime(
    vix_current: float,
    vix_history: pd.Series,
    lookback: int = 252,
) -> str:
    """
    Classify VIX into LOW / MEDIUM / HIGH regime based on trailing percentile.

    LOW:    VIX below 33rd percentile → complacency, good for selling premium
    MEDIUM: VIX 33rd-67th percentile → normal, standard strategies
    HIGH:   VIX above 67th percentile → fear, be cautious selling premium

    This is a dynamic threshold, not hardcoded VIX levels, because what
    counts as "high" VIX changes over time (India VIX has ranged from
    ~10 to ~85 historically).

    Parameters
    ----------
    vix_current : current VIX value
    vix_history : trailing VIX history (at least lookback days)
    lookback : trailing window for percentile computation

    Returns
    -------
    "LOW", "MEDIUM", or "HIGH"
    """
    if len(vix_history) < 30:
        return "MEDIUM"  # Insufficient data

    recent = vix_history.tail(lookback)
    percentile = (recent < vix_current).sum() / len(recent) * 100

    if percentile < 33:
        return "LOW"
    elif percentile < 67:
        return "MEDIUM"
    else:
        return "HIGH"


# ── Aggregate Signal Generator ──────────────────────────────────────────────────


def compute_all_options_signals(
    close_prices: pd.Series,
    iv_series: Optional[pd.Series] = None,
    option_chain: Optional[dict] = None,
    vix_series: Optional[pd.Series] = None,
) -> dict[str, float]:
    """
    Compute all options signals for a given symbol at a point in time.

    Parameters
    ----------
    close_prices : historical close prices (at least 252 days)
    iv_series : historical implied volatility (optional; synthetic if absent)
    option_chain : current option chain dict (optional)
    vix_series : historical VIX series (optional)

    Returns
    -------
    dict of signal_name → signal_value for all OPTIONS_SIGNAL_NAMES.
    Missing signals are NaN.
    """
    from app.quant.options_pricing import synthetic_iv_from_realized_vol

    signals: dict[str, float] = {name: np.nan for name in OPTIONS_SIGNAL_NAMES}

    if len(close_prices) < 30:
        return signals

    # Realized vol
    rv_20d = compute_realized_vol(close_prices, window=20)
    rv_60d = compute_realized_vol(close_prices, window=60)
    current_rv_20 = rv_20d.iloc[-1] if not rv_20d.empty else np.nan

    # IV: use provided series or synthesize from RV
    if iv_series is not None and len(iv_series) > 0:
        current_iv = iv_series.iloc[-1]
    elif not np.isnan(current_rv_20):
        current_iv = synthetic_iv_from_realized_vol(current_rv_20)
        # Build a synthetic IV series for percentile calculation
        iv_series = rv_20d.apply(lambda rv: synthetic_iv_from_realized_vol(rv) if not np.isnan(rv) else np.nan)
    else:
        current_iv = np.nan

    # VRP
    if not np.isnan(current_iv) and not np.isnan(current_rv_20):
        vrp_series = compute_vrp(
            pd.Series([current_iv], dtype=float),
            pd.Series([current_rv_20], dtype=float),
        )
        signals["vrp"] = float(vrp_series.iloc[0])

    # IV Percentile
    if iv_series is not None and len(iv_series) >= 30:
        iv_pct = compute_iv_percentile(iv_series)
        if not iv_pct.empty and not np.isnan(iv_pct.iloc[-1]):
            signals["iv_percentile"] = float(iv_pct.iloc[-1])

    # Option chain-dependent signals
    if option_chain is not None:
        spot = option_chain.get("spotPrice", close_prices.iloc[-1])

        # PCR
        pcr_data = compute_pcr_from_chain(option_chain)
        signals["pcr_oi"] = pcr_data["pcr_oi"]
        signals["pcr_volume"] = pcr_data["pcr_volume"]

        # IV Skew
        signals["iv_skew"] = compute_iv_skew(option_chain, spot)

        # Max Pain
        mp = compute_max_pain(option_chain)
        signals["max_pain_dist"] = mp["distance_pct"]

        # OI Concentration
        oi_conc = compute_oi_concentration(option_chain)
        signals["oi_concentration"] = oi_conc["hhi_calls"] + oi_conc["hhi_puts"]

        # GEX
        gex = compute_gex(option_chain, spot)
        signals["gex_normalized"] = gex["gex_normalized"]

    # IV Term Structure (needs near and far month IVs — skip if not available)
    # This would come from multi-expiry option chain data
    # signals["iv_term_slope"] = computed elsewhere

    return signals
