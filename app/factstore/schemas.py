"""
Fact Store Schemas
==================
Typed Pydantic models for all fact types in the ARTHA Fact Store.

Design principles (from AGENTS.md):
  - Every fact has a mandatory `source` field — no fact enters the store
    without provenance.
  - QuantSignal.validated defaults to False and can ONLY be set True by
    the walk-forward backtester (app/quant/backtest.py), never by an LLM.
  - LLM agents narrate facts that already exist here; they never compute
    or assert a number (rule 1).

Reference: docs/ARTHA_ARCHITECTURE.md §4.4
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────────────────────


class FactType(str, Enum):
    """Discriminator for polymorphic fact storage."""

    FUNDAMENTAL = "fundamental"
    QUANT_SIGNAL = "quant_signal"
    OPTIONS_SIGNAL = "options_signal"  # New: options strategy signals
    RISK_FLAG = "risk_flag"
    NEWS_SIGNAL = "news_signal"
    NARRATIVE = "narrative"
    REPORT_CITATION = "report_citation"


class Severity(str, Enum):
    """Risk-flag severity levels."""

    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"  # hard veto — Risk Agent can block a signal from the report


class SignalType(str, Enum):
    """Categories for quant signals."""

    REGIME = "regime"
    STAT_ARB_ZSCORE = "stat_arb_zscore"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    SEASONALITY = "seasonality"
    TERM_STRUCTURE = "term_structure"
    FACTOR = "factor"
    COMPOSITE = "composite"
    # Options-specific signal types
    VRP = "vrp"                        # Volatility risk premium
    IV_PERCENTILE = "iv_percentile"    # IV percentile rank
    PCR = "pcr"                        # Put-call ratio
    IV_SKEW = "iv_skew"               # IV skew
    OPTIONS_STRATEGY = "options_strategy"  # Strategy recommendation
    GAMMA_EXPOSURE = "gamma_exposure"  # Market-wide GEX


class EventType(str, Enum):
    """Coarse categorisation for news events."""

    EARNINGS = "earnings"
    CORPORATE_ACTION = "corporate_action"  # split, bonus, buyback, etc.
    POLICY = "policy"  # RBI, SEBI, government policy
    SUPPLY_SHOCK = "supply_shock"  # commodity-specific
    MACRO = "macro"  # GDP, inflation, employment
    SECTOR = "sector"
    GENERAL = "general"


# ── Base Fact ───────────────────────────────────────────────────────────────────


class BaseFact(BaseModel):
    """
    Common fields shared by every fact in the store.

    `fact_id` is auto-generated.  `source` is mandatory and must be
    non-empty — this is the traceability anchor for AGENTS.md rule 10.
    """

    fact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str = Field(
        ..., description="ID of the research/analysis job that produced this fact."
    )
    fact_type: FactType
    source: str = Field(
        ...,
        min_length=1,
        description=(
            "Provenance string — e.g. 'NSE_BHAVCOPY_2025-03-31', "
            "'BSE_ANNUAL_REPORT_2025_PG34', 'FINBERT_LOCAL'. "
            "Must never be empty (AGENTS.md rule 10)."
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Concrete Fact Types ─────────────────────────────────────────────────────────


class FundamentalRow(BaseFact):
    """
    A single row-level fundamental metric for a company/fiscal-year.

    This is the Screener++ building block: one metric, one year, one source.
    The Fundamentals Agent writes these; the Writer narrates them.

    Edge-case notes:
      - `value` may be NaN for years where the metric is unavailable
        (e.g. a company hasn't reported yet).  Represent as None.
      - Corporate actions (splits, bonuses) that break per-share series:
        the data layer must adjust *before* writing the fact.
    """

    fact_type: FactType = FactType.FUNDAMENTAL

    symbol: str = Field(..., description="NSE/BSE symbol, e.g. 'HDFCBANK'.")
    fiscal_year: int = Field(..., description="Fiscal year ending, e.g. 2025.")
    metric: str = Field(
        ...,
        description=(
            "Metric name — e.g. 'revenue', 'operating_margin', "
            "'debt_to_equity', 'eps', 'roe', 'roce'."
        ),
    )
    value: Optional[float] = Field(
        None,
        description="Metric value.  None if not yet reported / unavailable.",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit, e.g. 'INR_crore', 'percent', 'ratio'.",
    )


class QuantSignal(BaseFact):
    """
    A quantitative signal produced by the deterministic Quant Engine.

    `validated` is False by default and may ONLY be set to True by the
    walk-forward backtester after the signal clears the out-of-sample
    gate (AGENTS.md rule 6).  An LLM finding a signal "plausible"
    is never sufficient.
    """

    fact_type: FactType = FactType.QUANT_SIGNAL

    symbol_or_pair: str = Field(
        ...,
        description=(
            "Single symbol ('HDFCBANK') or pair/spread "
            "('HDFCBANK-ICICIBANK', 'GOLD-SILVER')."
        ),
    )
    signal_type: SignalType
    value: float = Field(..., description="Signal value (z-score, regime label index, etc.).")
    regime_label: Optional[str] = Field(
        None,
        description="Current regime label if signal is regime-conditioned.",
    )
    backtest_sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio from walk-forward backtest, if available.",
    )
    backtest_sortino: Optional[float] = Field(
        None,
        description="Sortino ratio from walk-forward backtest, if available.",
    )
    backtest_max_drawdown: Optional[float] = Field(
        None,
        description="Max drawdown (as a negative fraction) from walk-forward backtest.",
    )
    validated: bool = Field(
        False,
        description=(
            "ONLY True if the signal cleared the walk-forward out-of-sample "
            "gate in app/quant/backtest.py.  Never set by an LLM."
        ),
    )
    as_of: date = Field(..., description="Date the signal was computed for.")


class RiskFlag(BaseFact):
    """
    A risk or compliance flag raised by the Risk Agent.

    Flags with severity='block' prevent the flagged signal/symbol
    from appearing in the final report.
    """

    fact_type: FactType = FactType.RISK_FLAG

    symbol_or_pair: str
    flag_type: str = Field(
        ...,
        description=(
            "E.g. 'concentration', 'drawdown_breach', 'circuit_limit', "
            "'liquidity_low', 'corporate_action_pending'."
        ),
    )
    detail: str = Field(..., description="Human-readable explanation of the flag.")
    severity: Severity


class NewsSignal(BaseFact):
    """
    Sentiment signal from the local FinBERT pass over ticker-scoped news.

    The sentiment_score is computed by FinBERT locally — no LLM tokens
    spent on scoring.  Only the narration of what the score means touches
    the LLM quota (and only via the Sentiment Agent, using a cheap model).
    """

    fact_type: FactType = FactType.NEWS_SIGNAL

    symbol_or_commodity: str
    headline_count: int = Field(
        ..., ge=0, description="Number of headlines in the scoring window."
    )
    sentiment_score: float = Field(
        ...,
        description=(
            "Aggregate sentiment from FinBERT.  Range roughly [-1, +1]. "
            "Negative = bearish, positive = bullish."
        ),
    )
    event_type: Optional[EventType] = None
    as_of: date = Field(..., description="Date of the news window.")
    headlines_sample: list[str] = Field(
        default_factory=list,
        description="Up to 5 representative headlines (for narration context).",
    )


class NarrativeFact(BaseFact):
    """
    A narrative paragraph produced by an LLM agent, grounded in other facts.

    Every NarrativeFact must reference the `fact_id`s of the facts it
    is narrating — this is how the traceability gate
    (tests/test_factstore_citations.py) verifies that no numbers are
    invented.
    """

    fact_type: FactType = FactType.NARRATIVE

    agent_name: str = Field(
        ...,
        description="Which agent produced this narrative (e.g. 'fundamentals_agent').",
    )
    section: str = Field(
        ...,
        description="Report section this belongs to (e.g. 'fundamentals_analysis').",
    )
    content: str = Field(..., description="The narrative text.")
    referenced_fact_ids: list[str] = Field(
        default_factory=list,
        description="fact_id values of the facts this narrative is grounded in.",
    )


class ReportCitation(BaseFact):
    """
    A resolved citation linking a specific claim in the final report
    back to a Fact Store entry.

    Built during the post-generation citation check (AGENTS.md rule 10).
    """

    fact_type: FactType = FactType.REPORT_CITATION

    claim_text: str = Field(..., description="The exact claim text from the report.")
    referenced_fact_id: str = Field(
        ..., description="fact_id of the Fact Store entry backing this claim."
    )
    resolved: bool = Field(
        False,
        description="True if the referenced fact was found and its source is non-empty.",
    )


class OptionsSignal(BaseFact):
    """
    A signal from the options strategy selector / options engine.

    Captures the full state of a recommended options strategy: legs,
    Greeks, P&L bounds, margin, and the regime/VIX context that
    triggered the recommendation.

    Like QuantSignal, `validated` defaults to False and may ONLY be set
    True after the strategy clears the walk-forward options backtest gate
    (AGENTS.md rule 6).
    """

    fact_type: FactType = FactType.OPTIONS_SIGNAL

    symbol_or_index: str = Field(
        ...,
        description="NSE symbol or index name, e.g. 'RELIANCE' or 'NIFTY'.",
    )
    strategy_type: str = Field(
        ...,
        description=(
            "Strategy type, e.g. 'IRON_CONDOR', 'COVERED_CALL'. "
            "Values from app.quant.options_strategies.StrategyType."
        ),
    )
    legs: list[dict] = Field(
        default_factory=list,
        description=(
            "List of option legs. Each dict: "
            "{'type': 'SHORT_PUT', 'strike': 22800, 'expiry_days': 30, "
            "'quantity': 1, 'premium': 45.5, 'iv': 0.18}"
        ),
    )
    net_premium: float = Field(
        ...,
        description="Net credit (positive) or debit (negative) per lot.",
    )
    max_profit: float = Field(..., description="Maximum profit per lot in INR.")
    max_loss: float = Field(..., description="Maximum loss per lot in INR (positive = loss).")
    breakevens: list[float] = Field(
        default_factory=list,
        description="Breakeven price(s) at expiry.",
    )
    portfolio_delta: float = Field(
        0.0, description="Aggregate delta of the strategy."
    )
    portfolio_gamma: float = Field(
        0.0, description="Aggregate gamma of the strategy."
    )
    portfolio_theta: float = Field(
        0.0, description="Aggregate theta (per day) of the strategy."
    )
    portfolio_vega: float = Field(
        0.0, description="Aggregate vega (per 1% IV) of the strategy."
    )
    margin_required: float = Field(
        0.0, description="Approximate SPAN margin required in INR."
    )
    regime_label: str = Field(
        ..., description="Regime at time of recommendation: BULL/BEAR/SIDEWAYS."
    )
    vix_level: float = Field(
        0.0, description="India VIX level at time of recommendation."
    )
    vix_regime: str = Field(
        "MEDIUM", description="VIX regime: LOW/MEDIUM/HIGH."
    )
    vrp_signal: float = Field(
        0.0, description="VRP at time of recommendation: (IV-RV)/RV."
    )
    strategy_role: str = Field(
        "PRIMARY", description="Role: PRIMARY/SECONDARY/HEDGE/DEFENSE."
    )
    confidence: float = Field(
        0.5, description="Strategy selector confidence [0, 1]."
    )
    validated: bool = Field(
        False,
        description=(
            "ONLY True if the strategy cleared the walk-forward out-of-sample "
            "gate in app/quant/options_backtest.py. Never set by an LLM."
        ),
    )
    as_of: date = Field(..., description="Date the signal was computed for.")
