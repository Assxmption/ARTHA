"""
Risk Agent — Position-Sizing, Drawdown, and Compliance Checks
===============================================================
Reads ALL facts in the Fact Store for a given job, checks for risk
conditions, and writes RiskFlag facts with severity levels.

The Risk Agent has VETO POWER: a RiskFlag with severity='block'
prevents the flagged signal/symbol from being used in the final
report recommendation.

Non-negotiable constraints (AGENTS.md):
  - Rule 1: Never computes a number — only checks facts against
    pre-defined risk rules.
  - Rule 10: Every risk flag must reference the fact_id that triggered it.

Uses mid-tier model (AGENTS.md rule 3).

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Risk & Compliance Agent row)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from app.factstore.schemas import (
    BaseFact,
    FactType,
    FundamentalRow,
    NarrativeFact,
    QuantSignal,
    RiskFlag,
    Severity,
    SignalType,
)
from app.factstore.store import FactStore
from app.llm.router import get_mid_llm

logger = logging.getLogger(__name__)

# ── Risk Rules (deterministic, no LLM needed) ──────────────────────────────────

# Sector concentration: flag if > 30% of signals/fundamentals are in one sector
SECTOR_CONCENTRATION_LIMIT = 0.30

# Drawdown: flag if backtest max drawdown exceeds thresholds
DD_WARNING_THRESHOLD = -0.10   # -10%
DD_BLOCK_THRESHOLD = -0.20     # -20%

# Sharpe: flag if backtest Sharpe is below threshold
SHARPE_WARNING_THRESHOLD = 0.5
SHARPE_BLOCK_THRESHOLD = 0.0

# Debt-to-equity: flag high leverage
DE_WARNING_THRESHOLD = 2.0
DE_BLOCK_THRESHOLD = 5.0

# Volatility: flag extreme vol
VOL_WARNING_THRESHOLD = 50.0   # 50% annualised
VOL_BLOCK_THRESHOLD = 80.0    # 80% annualised


def run_risk_checks(
    job_id: str,
    store: FactStore,
    symbol: Optional[str] = None,
) -> list[RiskFlag]:
    """
    Run deterministic risk checks against all facts in the Fact Store.

    This is a pure rule-engine — no LLM needed for the checks themselves.
    The LLM (mid-tier) is only used for the optional narration of risk
    findings.

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    symbol : str, optional
        Focus symbol. If None, checks all symbols.

    Returns
    -------
    list[RiskFlag]
        Risk flags written to the Fact Store.
    """
    all_facts = store.get_job_facts(job_id)
    flags: list[RiskFlag] = []

    # ── Check 1: Fundamental risk (high leverage, low margins) ──────────
    fund_facts = [f for f in all_facts if isinstance(f, FundamentalRow)]

    for f in fund_facts:
        if symbol and f.symbol != symbol:
            continue

        # Debt-to-equity check
        if f.metric == "debt_to_equity" and f.value is not None:
            if f.value >= DE_BLOCK_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_DE_CHECK_{f.symbol}_FY{f.fiscal_year}",
                    symbol_or_pair=f.symbol,
                    flag_type="high_leverage",
                    detail=(
                        f"Debt-to-equity ratio of {f.value:.2f} in FY{f.fiscal_year} "
                        f"exceeds block threshold ({DE_BLOCK_THRESHOLD}). "
                        f"[ref: fact_id {f.fact_id[:8]}…]"
                    ),
                    severity=Severity.BLOCK,
                ))
            elif f.value >= DE_WARNING_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_DE_CHECK_{f.symbol}_FY{f.fiscal_year}",
                    symbol_or_pair=f.symbol,
                    flag_type="elevated_leverage",
                    detail=(
                        f"Debt-to-equity ratio of {f.value:.2f} in FY{f.fiscal_year} "
                        f"is above warning threshold ({DE_WARNING_THRESHOLD}). "
                        f"[ref: fact_id {f.fact_id[:8]}…]"
                    ),
                    severity=Severity.WARNING,
                ))

        # Negative margins check
        if f.metric == "operating_margin" and f.value is not None and f.value < 0:
            flags.append(RiskFlag(
                job_id=job_id,
                source=f"RISK_AGENT_MARGIN_CHECK_{f.symbol}_FY{f.fiscal_year}",
                symbol_or_pair=f.symbol,
                flag_type="negative_operating_margin",
                detail=(
                    f"Operating margin of {f.value:.1f}% in FY{f.fiscal_year} "
                    f"indicates operating losses. [ref: fact_id {f.fact_id[:8]}…]"
                ),
                severity=Severity.WARNING,
            ))

        # Volatility check
        if f.metric == "annual_volatility_pct" and f.value is not None:
            if f.value >= VOL_BLOCK_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_VOL_CHECK_{f.symbol}_FY{f.fiscal_year}",
                    symbol_or_pair=f.symbol,
                    flag_type="extreme_volatility",
                    detail=(
                        f"Annual volatility of {f.value:.1f}% in FY{f.fiscal_year} "
                        f"exceeds block threshold ({VOL_BLOCK_THRESHOLD}%). "
                        f"[ref: fact_id {f.fact_id[:8]}…]"
                    ),
                    severity=Severity.BLOCK,
                ))
            elif f.value >= VOL_WARNING_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_VOL_CHECK_{f.symbol}_FY{f.fiscal_year}",
                    symbol_or_pair=f.symbol,
                    flag_type="high_volatility",
                    detail=(
                        f"Annual volatility of {f.value:.1f}% in FY{f.fiscal_year} "
                        f"exceeds warning threshold ({VOL_WARNING_THRESHOLD}%). "
                        f"[ref: fact_id {f.fact_id[:8]}…]"
                    ),
                    severity=Severity.WARNING,
                ))

    # ── Check 2: Quant signal risk (poor backtest, extreme signals) ─────
    quant_facts = [f for f in all_facts if isinstance(f, QuantSignal)]

    for q in quant_facts:
        if symbol and symbol not in q.symbol_or_pair:
            continue

        # Sharpe check
        if q.backtest_sharpe is not None:
            if q.backtest_sharpe <= SHARPE_BLOCK_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_SHARPE_CHECK_{q.symbol_or_pair}",
                    symbol_or_pair=q.symbol_or_pair,
                    flag_type="negative_sharpe",
                    detail=(
                        f"Signal {q.signal_type.value} has backtest Sharpe "
                        f"of {q.backtest_sharpe:.2f} — destroying value. "
                        f"[ref: fact_id {q.fact_id[:8]}…]"
                    ),
                    severity=Severity.BLOCK,
                ))
            elif q.backtest_sharpe <= SHARPE_WARNING_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_SHARPE_CHECK_{q.symbol_or_pair}",
                    symbol_or_pair=q.symbol_or_pair,
                    flag_type="low_sharpe",
                    detail=(
                        f"Signal {q.signal_type.value} has backtest Sharpe "
                        f"of {q.backtest_sharpe:.2f} — weak risk-adjusted return. "
                        f"[ref: fact_id {q.fact_id[:8]}…]"
                    ),
                    severity=Severity.WARNING,
                ))

        # Drawdown check
        if q.backtest_max_drawdown is not None:
            if q.backtest_max_drawdown <= DD_BLOCK_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_DD_CHECK_{q.symbol_or_pair}",
                    symbol_or_pair=q.symbol_or_pair,
                    flag_type="excessive_drawdown",
                    detail=(
                        f"Max drawdown of {q.backtest_max_drawdown*100:.1f}% "
                        f"exceeds block threshold ({DD_BLOCK_THRESHOLD*100}%). "
                        f"[ref: fact_id {q.fact_id[:8]}…]"
                    ),
                    severity=Severity.BLOCK,
                ))
            elif q.backtest_max_drawdown <= DD_WARNING_THRESHOLD:
                flags.append(RiskFlag(
                    job_id=job_id,
                    source=f"RISK_AGENT_DD_CHECK_{q.symbol_or_pair}",
                    symbol_or_pair=q.symbol_or_pair,
                    flag_type="elevated_drawdown",
                    detail=(
                        f"Max drawdown of {q.backtest_max_drawdown*100:.1f}% "
                        f"exceeds warning threshold ({DD_WARNING_THRESHOLD*100}%). "
                        f"[ref: fact_id {q.fact_id[:8]}…]"
                    ),
                    severity=Severity.WARNING,
                ))

        # Bear regime warning
        if q.signal_type == SignalType.REGIME and q.regime_label == "BEAR":
            flags.append(RiskFlag(
                job_id=job_id,
                source=f"RISK_AGENT_REGIME_CHECK",
                symbol_or_pair=q.symbol_or_pair,
                flag_type="bear_regime",
                detail=(
                    f"Market regime is BEAR — defensive positioning recommended. "
                    f"[ref: fact_id {q.fact_id[:8]}…]"
                ),
                severity=Severity.WARNING,
            ))

    # ── Check 3: SEBI compliance (informational) ───────────────────────
    flags.append(RiskFlag(
        job_id=job_id,
        source="RISK_AGENT_SEBI_COMPLIANCE",
        symbol_or_pair=symbol or "PORTFOLIO",
        flag_type="sebi_compliance_note",
        detail=(
            "SEBI retail algo-trading framework (April 2026): "
            "Personal-use API trading requires Generic Algo-ID, "
            "registered static IP, and daily 2FA. "
            "This report is not investment advice."
        ),
        severity=Severity.INFO,
    ))

    # ── Write all flags to Fact Store (checkpoint) ─────────────────────
    if flags:
        store.put_facts(flags)
        logger.info(
            "Risk Agent wrote %d flags (%d block, %d warning, %d info)",
            len(flags),
            sum(1 for f in flags if f.severity == Severity.BLOCK),
            sum(1 for f in flags if f.severity == Severity.WARNING),
            sum(1 for f in flags if f.severity == Severity.INFO),
        )

    return flags


def run_risk_narration(
    job_id: str,
    store: FactStore,
    flags: list[RiskFlag],
) -> Optional[NarrativeFact]:
    """
    Produce an LLM narrative summarising the risk findings.

    Uses mid-tier model since risk assessment needs nuance.

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    flags : list[RiskFlag]
        Risk flags to narrate.

    Returns
    -------
    NarrativeFact or None
        The risk narrative, or None if no meaningful flags.
    """
    # Filter out info-only flags for narration
    meaningful = [f for f in flags if f.severity in (Severity.WARNING, Severity.BLOCK)]
    if not meaningful:
        logger.info("No meaningful risk flags to narrate")
        return None

    flag_text = "\n".join(
        f"- [{f.severity.value.upper()}] {f.flag_type}: {f.detail}"
        for f in meaningful
    )

    prompt = f"""You are the ARTHA Risk & Compliance Agent — a senior risk manager
specialising in Indian equity markets (NSE) and SEBI regulatory compliance.

Given the following risk flags identified from the analysis, write a 2-3 paragraph
risk assessment. Be direct and actionable. For BLOCK-level flags, clearly state
that the signal/symbol should be excluded from any portfolio consideration.

RISK FLAGS:
{flag_text}

RULES:
- Be concise and direct — no filler.
- Reference the specific flag types and severity levels.
- For BLOCK flags: clearly state the signal is vetoed.
- For WARNING flags: note the concern and suggest monitoring.
- End with a one-line SEBI compliance reminder.
- Do NOT invent any numbers not present in the flags above."""

    try:
        llm = get_mid_llm()
        response = llm.call([{"role": "user", "content": prompt}])
        narrative_text = str(response)
    except Exception as e:
        logger.error("LLM call failed for risk narration: %s", e)
        # Fallback: summarise flags without LLM
        narrative_text = _fallback_risk_narration(meaningful)

    fact = NarrativeFact(
        job_id=job_id,
        source="RISK_AGENT_NARRATION",
        agent_name="risk_agent",
        section="risk_assessment",
        content=narrative_text,
        referenced_fact_ids=[f.fact_id for f in flags],
    )
    store.put_fact(fact)
    logger.info("Risk Agent wrote risk narrative (%d chars)", len(narrative_text))
    return fact


def _fallback_risk_narration(flags: list[RiskFlag]) -> str:
    """Template-based fallback when LLM is unavailable."""
    lines = ["## Risk Assessment\n"]

    blocks = [f for f in flags if f.severity == Severity.BLOCK]
    warnings = [f for f in flags if f.severity == Severity.WARNING]

    if blocks:
        lines.append("### 🛑 Blocked Signals\n")
        for f in blocks:
            lines.append(f"- **{f.flag_type}** ({f.symbol_or_pair}): {f.detail}")

    if warnings:
        lines.append("\n### ⚠️ Warnings\n")
        for f in warnings:
            lines.append(f"- **{f.flag_type}** ({f.symbol_or_pair}): {f.detail}")

    lines.append(
        "\n*Note: This report is not investment advice. SEBI retail algo-trading "
        "framework (April 2026) requires Generic Algo-ID and registered static IP "
        "for automated order placement.*"
    )
    return "\n".join(lines)
