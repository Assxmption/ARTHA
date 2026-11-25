"""
Writer Agent — Final Report Synthesis
======================================
Produces the structured + narrative report from Fact Store entries.
This is the ONLY agent that uses the deep (expensive) model, and
it is called ONLY ONCE per job.

Non-negotiable constraints (AGENTS.md):
  - Rule 1: Never computes a number — only narrates facts from the store.
  - Rule 10: Every numeric claim must resolve to a Fact Store entry.
    Post-generation citation check enforces this.
  - Rule 11: This is not investment advice.  Reports describe validated
    signals and historical backtest performance; they do not tell a user
    what to buy or sell.

Reference: docs/ARTHA_ARCHITECTURE.md §4.5 (Portfolio Writer row)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from crewai import Agent

from app.factstore.schemas import (
    BaseFact,
    FactType,
    FundamentalRow,
    NarrativeFact,
    NewsSignal,
    QuantSignal,
    ReportCitation,
    RiskFlag,
)
from app.factstore.store import FactStore
from app.llm.router import get_deep_llm

logger = logging.getLogger(__name__)


def _blocked_targets(store: FactStore, job_id: str) -> set[str]:
    """Return normalized symbols/pairs vetoed by BLOCK risk facts."""
    return {
        fact.symbol_or_pair.strip().upper()
        for fact in store.get_facts_by_type(job_id, FactType.RISK_FLAG)
        if isinstance(fact, RiskFlag) and fact.severity.value == "block"
    }


def _is_blocked_target(target: str, blocked: set[str]) -> bool:
    """Match a blocked symbol against a symbol or a multi-leg pair/spread."""
    normalized = target.strip().upper()
    if normalized in blocked:
        return True
    tokens = {
        token
        for token in re.split(r"[^A-Z0-9_.-]+", normalized)
        if token
    }
    return any(
        blocked_target in tokens
        or blocked_target in normalized.split("-")
        or blocked_target in normalized.split("/")
        for blocked_target in blocked
    )


def create_writer_agent() -> Agent:
    """
    Create the ARTHA Writer Agent.

    The Writer synthesises all Fact Store entries for the current job
    into a structured, professional report.  Only uses the deep model
    tier, and only once per job.
    """
    llm = get_deep_llm()

    return Agent(
        role="ARTHA Portfolio Report Writer",
        goal=(
            "Produce a comprehensive, professionally structured analytical "
            "report in Markdown format. The report must include: Executive "
            "Summary, Fundamental Analysis, any available Quantitative "
            "Signals, Risk Flags, and a References section. EVERY numeric "
            "claim must cite a Fact Store entry by its fact_id. If a metric "
            "is unavailable, say so — never invent a number. End with the "
            "mandatory disclaimer: 'This is not investment advice.'"
        ),
        backstory=(
            "You are an expert financial report writer specialising in "
            "Indian equity and commodity markets. Your reports are valued "
            "for their analytical depth, precise language, and rigorous "
            "source attribution. You write for an audience that includes "
            "both technical analysts and informed investors. You always "
            "maintain intellectual honesty about uncertainties and never "
            "present analysis as investment advice. Your style is "
            "authoritative yet accessible, with every number traceable "
            "to a specific data source."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ── Report context builder ─────────────────────────────────────────────────────

def build_writer_context(
    job_id: str,
    store: FactStore,
    symbol: str,
) -> str:
    """
    Build the Writer agent's context from the Fact Store.

    Only includes facts relevant to the current job — NOT a growing
    chat transcript (AGENTS.md rule 2: no sequential prompt-stacking).

    Parameters
    ----------
    job_id : str
        Current job ID.
    store : FactStore
        Fact Store instance.
    symbol : str
        Primary symbol being analysed.

    Returns
    -------
    str
        Structured context text for the Writer's prompt.
    """
    sections = []

    # ── Fundamentals ────────────────────────────────────────────────────
    fund_facts = store.get_facts_by_type(job_id, FactType.FUNDAMENTAL)
    if fund_facts:
        sections.append("## FUNDAMENTAL DATA (from Fact Store)\n")
        sections.append(
            "| Fact ID | Symbol | FY | Metric | Value | Unit | Source |"
        )
        sections.append(
            "|---------|--------|----|--------|-------|------|--------|"
        )
        for f in sorted(
            fund_facts,
            key=lambda x: (x.symbol, x.fiscal_year, x.metric)
            if isinstance(x, FundamentalRow)
            else ("", 0, ""),
        ):
            if isinstance(f, FundamentalRow):
                val = f"{f.value:.2f}" if f.value is not None else "N/A"
                sections.append(
                    f"| {f.fact_id[:8]}… | {f.symbol} | {f.fiscal_year} "
                    f"| {f.metric} | {val} | {f.unit or '—'} | {f.source} |"
                )
        sections.append("")

    # ── Quant Signals (validated only) ──────────────────────────────────
    blocked_targets = _blocked_targets(store, job_id)
    quant_facts = [
        fact for fact in store.get_validated_signals(job_id)
        if not _is_blocked_target(fact.symbol_or_pair, blocked_targets)
    ]
    if quant_facts:
        sections.append("## VALIDATED QUANT SIGNALS (from Fact Store)\n")
        for q in quant_facts:
            sections.append(
                f"- **{q.signal_type.value}** for {q.symbol_or_pair}: "
                f"value={q.value:.4f}, regime={q.regime_label or 'N/A'}, "
                f"backtest Sharpe={q.backtest_sharpe or 'N/A'} "
                f"[fact_id: {q.fact_id[:8]}…, source: {q.source}]"
            )
        sections.append("")

    # ── Risk Flags ──────────────────────────────────────────────────────
    risk_facts = store.get_facts_by_type(job_id, FactType.RISK_FLAG)
    if risk_facts:
        sections.append("## RISK FLAGS (from Fact Store)\n")
        for r in risk_facts:
            if isinstance(r, RiskFlag):
                icon = {"info": "ℹ️", "warning": "⚠️", "block": "🛑"}.get(
                    r.severity.value, "❓"
                )
                sections.append(
                    f"- {icon} **{r.flag_type}** ({r.severity.value}) — "
                    f"{r.symbol_or_pair}: {r.detail} "
                    f"[fact_id: {r.fact_id[:8]}…]"
                )
        sections.append("")

    # ── News / Sentiment ────────────────────────────────────────────────
    news_facts = store.get_facts_by_type(job_id, FactType.NEWS_SIGNAL)
    if news_facts:
        sections.append("## NEWS SENTIMENT (from Fact Store)\n")
        for n in news_facts:
            if isinstance(n, NewsSignal):
                sections.append(
                    f"- {n.symbol_or_commodity}: sentiment={n.sentiment_score:.2f}, "
                    f"headlines={n.headline_count}, event={n.event_type or 'general'} "
                    f"[fact_id: {n.fact_id[:8]}…]"
                )
        sections.append("")

    # ── Narratives from prior agents ────────────────────────────────────
    narrative_facts = store.get_facts_by_type(job_id, FactType.NARRATIVE)
    if narrative_facts:
        sections.append("## AGENT NARRATIVES (from Fact Store)\n")
        for nf in narrative_facts:
            if isinstance(nf, NarrativeFact):
                sections.append(
                    f"### {nf.agent_name} — {nf.section}\n{nf.content}\n"
                )

    # ── Instructions ────────────────────────────────────────────────────
    sections.append("---")
    sections.append(
        "**INSTRUCTIONS**: Write a comprehensive report about "
        f"{symbol} using ONLY the data above. Every line that contains a "
        "number must cite, on that same line, the Fact ID(s) the number "
        "comes from, and the number must be copied exactly as shown above "
        "(you may round it, but do not convert units). Do not compute "
        "growth rates, differences, averages or counts, and do not invent "
        "or estimate any value not present above — an automated check "
        "rejects any number that does not match a cited fact. Do not "
        "number your headings. End the report with: "
        "'**Disclaimer**: This report is not investment advice. It "
        "describes validated signals and historical data; it does not "
        "tell you what to buy or sell.'"
    )

    return "\n".join(sections)


# ── Post-generation citation check ─────────────────────────────────────────────

def check_citations(
    report_text: str,
    job_id: str,
    store: FactStore,
) -> dict:
    """
    Post-generation citation check (AGENTS.md rule 10).

    Scans the report for fact_id references and verifies each one
    exists in the Fact Store with a non-empty source.

    Returns
    -------
    dict
        'valid': bool — True if all citations resolve.
        'total_citations': int
        'resolved': int
        'unresolved': list[str] — fact_ids that failed to resolve.
    """
    report_text = _ascii_fold(report_text)
    all_facts = store.get_job_facts(job_id)
    fact_id_map = {f.fact_id: f for f in all_facts}

    def resolve(ref_id: str) -> Optional[BaseFact]:
        # Reports may cite truncated 8-char prefixes of the full id.
        ref_id = ref_id.lower()
        for full_id, fact in fact_id_map.items():
            if full_id.startswith(ref_id) and fact.source:
                return fact
        return None

    unique_ids = list(dict.fromkeys(ref.lower() for ref in _CITATION_ID_RE.findall(report_text)))
    resolved = [ref for ref in unique_ids if resolve(ref) is not None]
    unresolved = [ref for ref in unique_ids if resolve(ref) is None]

    # A citation somewhere in a report is not enough:
    #   (a) every line containing a numeric claim must carry a citation, and
    #   (b) every number on that line must equal a value held by one of the
    #       facts cited on the same line (to its displayed precision).
    # (b) is what stops "Revenue grew 9999% [fact_id: <real id>]": the id
    # resolves, but 9999 is not in that fact.  Numbers derived by the LLM
    # (growth rates, differences, counts) are rejected by design — rule 1.
    uncited_numeric_claims: list[str] = []
    unsupported_numeric_claims: list[dict] = []
    for line in report_text.splitlines():
        line_ids = _CITATION_ID_RE.findall(line)
        numbers = _claim_numbers(line)
        if not numbers:
            continue
        if not line_ids:
            uncited_numeric_claims.append(line.strip())
            continue
        candidates: list[float] = []
        for ref in line_ids:
            fact = resolve(ref)
            if fact is not None:
                candidates.extend(_fact_numeric_values(fact))
        unmatched = [
            token for token, value, decimals in numbers
            if not any(_number_matches(value, decimals, c) for c in candidates)
        ]
        if unmatched:
            unsupported_numeric_claims.append({"line": line.strip(), "numbers": unmatched})

    return {
        "valid": (
            len(unresolved) == 0
            and len(resolved) > 0
            and len(uncited_numeric_claims) == 0
            and len(unsupported_numeric_claims) == 0
        ),
        "total_citations": len(unique_ids),
        "resolved": len(resolved),
        "unresolved": unresolved,
        "uncited_numeric_claims": uncited_numeric_claims,
        "unsupported_numeric_claims": unsupported_numeric_claims,
    }


# ── Numeric-claim helpers for the citation gate ─────────────────────────────────

# Accept the spellings LLMs actually produce when copying the context table
# ("fact_id: ab12…", "Fact ID ab12…", "fact-id #ab12…").  The *numbers* on the
# line are still verified against the resolved fact, so a looser id syntax
# does not loosen the gate.
_CITATION_KEY = r"fact[\s_-]?id"
# One key may introduce several ids: "[Fact IDs: ab12cd34…; ef56ab78…]".
_CITATION_GROUP_RE = re.compile(
    _CITATION_KEY + r"s?[:\s#]*"
    r"([a-f0-9]{8,}…?(?:\s*(?:[;,/&]|and)\s*#?[a-f0-9]{8,}…?)*)",
    re.IGNORECASE,
)
_HEX_ID_RE = re.compile(r"[a-f0-9]{8,}", re.IGNORECASE)


class _CitationIds:
    """`findall`-compatible extractor that expands multi-id citation groups."""

    @staticmethod
    def findall(text: str) -> list[str]:
        ids: list[str] = []
        for group in _CITATION_GROUP_RE.findall(text):
            ids.extend(_HEX_ID_RE.findall(group))
        return ids


_CITATION_ID_RE = _CitationIds()
# Whole citation markup, including any "source: NSE_X_2025" text inside the
# brackets, so provenance strings are never mistaken for claims.
_CITATION_SPAN_RES = (
    re.compile(r"\[[^\]\n]*" + _CITATION_KEY + r"[^\]\n]*\]", re.IGNORECASE),
    re.compile(r"\([^)\n]*" + _CITATION_KEY + r"[^)\n]*\)", re.IGNORECASE),
    _CITATION_GROUP_RE,
)
# Markdown ordered-list / numbered-heading markers are structure, not claims.
_LIST_MARKER_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:[-*+]\s+)?\d+[.)]\s+")
# A number not glued to a preceding letter/digit/underscore, so labels such as
# FY2025, Q2, H1, return_12m are not treated as claims.  Commas are accepted
# in both Western (250,000) and Indian (2,50,000) grouping.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:,\d+)*(?:\.\d+)?")
# Lookback-window names ("52-week high", "200-day average", "12-month return")
# are metric labels, not asserted values.
# Fiscal-year labels and ranges ("FY 2025", "FY2023-2026", "FY24–FY26") name
# the period a fact belongs to; they are not values.
_FY_LABEL_RE = re.compile(
    r"\bFY\s?'?\d{2,4}(?:\s?[-/]\s?(?:FY\s?'?)?\d{2,4})?", re.IGNORECASE
)
_PERIOD_LABEL_RE = re.compile(r"(?<![A-Za-z0-9_.,])\d+[- ](?:week|day|month|year)s?\b", re.IGNORECASE)
_NON_NUMERIC_FACT_TYPES = {FactType.NARRATIVE, FactType.REPORT_CITATION}
_EXCLUDED_FACT_FIELDS = {"fact_id", "job_id", "created_at", "fact_type"}


# LLMs emit typographic characters (U+2011 non-breaking hyphen, U+2212 minus,
# U+202F narrow no-break space, …); fold them to ASCII before parsing so
# "52‑week" and "‑24.98 %" are read the same as their ASCII spellings.
_TYPOGRAPHIC = str.maketrans({
    **{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d"},
    **{c: " " for c in "\u00a0\u2007\u2009\u200a\u202f\u205f\u3000"},
    "\u2009": " ",
})


def _ascii_fold(text: str) -> str:
    return text.translate(_TYPOGRAPHIC)


def _parse_numbers(text: str) -> list[tuple[str, float, int]]:
    """Return (token, absolute value, decimal places) for numbers in text."""
    out = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        plain = token.replace(",", "")
        decimals = len(plain.split(".", 1)[1]) if "." in plain else 0
        out.append((token, abs(float(plain)), decimals))
    return out


def _claim_numbers(line: str) -> list[tuple[str, float, int]]:
    """Numbers on a report line that count as claims."""
    stripped = line
    for pattern in _CITATION_SPAN_RES:
        stripped = pattern.sub(" ", stripped)
    stripped = _LIST_MARKER_RE.sub("", stripped)
    stripped = _PERIOD_LABEL_RE.sub(" ", stripped)
    stripped = _FY_LABEL_RE.sub(" ", stripped)
    return _parse_numbers(stripped)


def _fact_numeric_values(fact: BaseFact) -> list[float]:
    """
    Every number a fact can legitimately support: numeric fields, date parts,
    and numbers embedded in its string fields (e.g. RiskFlag.detail, news
    headlines).  Narrative and citation facts contribute nothing — their
    numbers were written by an LLM, and citing them must not launder a claim.
    """
    if fact.fact_type in _NON_NUMERIC_FACT_TYPES:
        return []
    values: list[float] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            if math.isfinite(obj):
                values.append(abs(float(obj)))
        elif isinstance(obj, str):
            values.extend(v for _, v, _ in _parse_numbers(obj))
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(fact.model_dump(mode="json", exclude=_EXCLUDED_FACT_FIELDS))
    return values


def _number_matches(value: float, decimals: int, candidate: float) -> bool:
    """True if `value` is `candidate` rounded to `decimals` places."""
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9 * max(1.0, candidate)
    return abs(value - candidate) <= tolerance


def persist_report_citations(
    report_text: str,
    job_id: str,
    store: FactStore,
) -> dict:
    """Persist resolved and unresolved report citations as Fact Store facts."""
    result = check_citations(report_text, job_id, store)
    all_facts = store.get_job_facts(job_id)
    fact_id_map = {fact.fact_id: fact for fact in all_facts}
    citations: list[ReportCitation] = []
    for ref_id in (ref.lower() for ref in _CITATION_ID_RE.findall(report_text)):
        matched_id = next(
            (full_id for full_id in fact_id_map if full_id.startswith(ref_id)),
            ref_id,
        )
        claim = next(
            (line.strip() for line in report_text.splitlines() if ref_id in line.lower()),
            ref_id,
        )
        citations.append(ReportCitation(
            job_id=job_id,
            source="writer_agent.citation_check",
            claim_text=claim,
            referenced_fact_id=matched_id,
            resolved=matched_id in fact_id_map and bool(fact_id_map[matched_id].source),
        ))
    if citations:
        store.put_facts(citations)
    return result
