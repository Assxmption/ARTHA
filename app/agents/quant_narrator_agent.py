"""
Quant Narrator Agent — Explains Quant Engine Outputs in Plain Language
=======================================================================
Reads validated QuantSignal facts from the Fact Store and the latest
backtest report, then produces a Harvey-style narrative explaining what
the numbers mean for the portfolio.

Non-negotiable constraints (AGENTS.md):
  - Rule 1: NEVER computes or asserts a number.  All numeric values
    come from QuantSignal facts or backtest reports.
  - Rule 10: Every narrative claim must cite a specific fact_id.

Uses cheap model (AGENTS.md rule 3).

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Quant Narrator Agent row)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from app.factstore.schemas import (
    NarrativeFact,
    QuantSignal,
    SignalType,
)
from app.factstore.store import FactStore
from app.llm.router import get_cheap_llm

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("docs/backtest_reports")


def _load_latest_backtest_report() -> Optional[dict]:
    """Load the most recent backtest report JSON."""
    if not REPORTS_DIR.exists():
        return None
    reports = sorted(REPORTS_DIR.glob("artha_v5*.json"), key=lambda f: f.stat().st_mtime)
    if not reports:
        return None
    try:
        with open(reports[-1]) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load backtest report: %s", e)
        return None


def _format_quant_signals_for_prompt(signals: list[QuantSignal]) -> str:
    """Format QuantSignal facts into structured text for the LLM prompt."""
    if not signals:
        return "No validated quant signals available."

    lines = ["## Validated Quant Signals\n"]
    lines.append("| Fact ID | Symbol/Pair | Signal Type | Value | Regime | Backtest Sharpe |")
    lines.append("|---------|-------------|-------------|-------|--------|----------------|")

    for s in signals:
        lines.append(
            f"| {s.fact_id[:8]}… | {s.symbol_or_pair} | {s.signal_type.value} "
            f"| {s.value:.4f} | {s.regime_label or 'N/A'} "
            f"| {s.backtest_sharpe or 'N/A'} |"
        )

    lines.append(
        "\n**CRITICAL**: When narrating these signals, you MUST reference "
        "the Fact ID of each value you mention."
    )
    return "\n".join(lines)


def _format_backtest_for_prompt(report: dict) -> str:
    """Format backtest report into structured text for the LLM prompt."""
    if not report:
        return "No backtest report available."

    lines = ["## Latest Backtest Report\n"]
    lines.append(f"- **Version**: {report.get('version', 'N/A')}")
    lines.append(f"- **Universe**: {report.get('universe', 'N/A')} stocks")
    lines.append(f"- **Trades**: {report.get('trades', 'N/A')}")

    for strategy_key in ["raw", "hedged", "hvt"]:
        s = report.get(strategy_key, {})
        if s:
            lines.append(f"\n### {s.get('label', strategy_key)}")
            lines.append(f"- Sharpe: {s.get('sharpe', 'N/A')}")
            lines.append(f"- Sortino: {s.get('sortino', 'N/A')}")
            lines.append(f"- Ann. Return: {s.get('ann_return', 'N/A')}")
            lines.append(f"- Max Drawdown: {s.get('max_dd', 'N/A')}")
            lines.append(f"- Win Rate: {s.get('win_rate', 'N/A')}")

    return "\n".join(lines)


def run_quant_narration(
    job_id: str,
    store: FactStore,
    symbol: Optional[str] = None,
) -> NarrativeFact:
    """
    Run the quant narration pipeline:
      1. Read validated QuantSignal facts from the Fact Store.
      2. Load the latest backtest report.
      3. Call the cheap LLM to produce a narrative.
      4. Write the NarrativeFact to the Fact Store (checkpoint).

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    symbol : str, optional
        Focus symbol (filters signals). If None, narrates all signals.

    Returns
    -------
    NarrativeFact
        The narrative explaining quant engine state.
    """
    # 1. Read validated signals
    all_signals = store.get_validated_signals(job_id)
    if symbol:
        relevant = [s for s in all_signals if symbol in s.symbol_or_pair]
        # Also include portfolio-level signals (COMPOSITE, REGIME)
        relevant.extend(
            s for s in all_signals
            if s.signal_type in (SignalType.REGIME, SignalType.COMPOSITE)
            and s not in relevant
        )
    else:
        relevant = all_signals

    # 2. Load backtest report
    backtest = _load_latest_backtest_report()

    # 3. Build prompt context
    signal_context = _format_quant_signals_for_prompt(relevant)
    backtest_context = _format_backtest_for_prompt(backtest)

    prompt = f"""You are the ARTHA Quant Narrator — a senior quantitative analyst
specialising in Indian markets (NSE). Your job is to read validated quant engine
outputs and explain what they mean in plain, professional language.

RULES:
- NEVER compute or estimate any number. Only narrate values from the data below.
- Every numeric claim MUST reference the Fact ID of the signal it comes from.
- Explain the regime state, top signals, and portfolio positioning.
- Use language appropriate for an informed investor, not a mathematician.
- If a signal type is unclear, say so rather than guessing.

{signal_context}

{backtest_context}

Write a 3-4 paragraph narrative titled "Quantitative Engine Assessment" that covers:
1. Current market regime and what it means for positioning
2. The strongest signals and their implications
3. Portfolio-level performance (Sharpe, drawdown) and risk posture
4. Any regime transitions or signal divergences to watch

Reference Fact IDs as [fact_id: XXXXXXXX…] for traceability."""

    # 4. Call LLM
    try:
        llm = get_cheap_llm()
        response = llm.call([{"role": "user", "content": prompt}])
        narrative_text = str(response)
    except Exception as e:
        logger.error("LLM call failed for quant narration: %s", e)
        # Fallback: produce a template-based narrative without LLM
        narrative_text = _fallback_narration(relevant, backtest)

    # 5. Write NarrativeFact to Fact Store (checkpoint)
    fact = NarrativeFact(
        job_id=job_id,
        source="QUANT_NARRATOR_AGENT",
        agent_name="quant_narrator",
        section="quantitative_assessment",
        content=narrative_text,
        referenced_fact_ids=[s.fact_id for s in relevant],
    )
    store.put_fact(fact)
    logger.info(
        "Quant Narrator wrote NarrativeFact (%d chars, %d signal refs)",
        len(narrative_text), len(relevant),
    )
    return fact


def _fallback_narration(signals: list[QuantSignal], backtest: Optional[dict]) -> str:
    """Template-based fallback when LLM is unavailable."""
    lines = ["## Quantitative Engine Assessment\n"]

    # Regime
    regime_signals = [s for s in signals if s.signal_type == SignalType.REGIME]
    if regime_signals:
        r = regime_signals[0]
        lines.append(
            f"The HMM regime detector currently classifies the market as "
            f"**{r.regime_label or 'UNKNOWN'}** [fact_id: {r.fact_id[:8]}…]. "
        )

    # Backtest
    if backtest:
        raw = backtest.get("raw", {})
        if raw:
            lines.append(
                f"\nThe latest backtest ({backtest.get('version', 'N/A')}) shows "
                f"a Sharpe ratio of {raw.get('sharpe', 'N/A'):.2f} over "
                f"{raw.get('n', 'N/A')} trading days. "
            )

    # Signals
    factor_signals = [s for s in signals if s.signal_type == SignalType.FACTOR]
    if factor_signals:
        top = sorted(factor_signals, key=lambda s: abs(s.value), reverse=True)[:3]
        lines.append("\nStrongest factor signals:")
        for s in top:
            lines.append(
                f"- {s.symbol_or_pair}: {s.value:+.4f} [fact_id: {s.fact_id[:8]}…]"
            )

    if not any(lines[1:]):
        lines.append("No validated quant signals available for narration.")

    return "\n".join(lines)
