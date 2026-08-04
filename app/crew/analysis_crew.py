"""
Analysis Crew — Direct Python Orchestrator
=============================================
Replaces the old CrewAI-based research_crew.py with a Fact Store–centric
pipeline that avoids sequential context-stacking.

Pipeline (fundamentals only — quant engine is a separate system):
  Stage 1: Fundamentals Agent (financial statements + market data)
  Stage 2: Fundamentals Narration + Risk Agent
  Stage 3: Writer Agent (synthesizes all facts into final report)

Per-stage checkpointing: each agent writes to the Fact Store immediately.
A failure at stage N never discards completed work from stages 1..N-1.

Reference: docs/ARTHA_ARCHITECTURE.md §4.5, §4.6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.fundamentals_agent import (
    format_facts_for_narration,
    run_fundamentals_analysis,
)
from app.agents.quant_narrator_agent import run_quant_narration
from app.agents.risk_agent import run_risk_checks, run_risk_narration
from app.agents.writer_agent import build_writer_context, check_citations
from app.factstore.schemas import NarrativeFact, Severity
from app.factstore.store import FactStore
from app.llm.router import get_cheap_llm, get_mid_llm, get_deep_llm
from app.quant.signal_injector import inject_backtest_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("artha.analysis_crew")

REPORTS_DIR = Path("data/Reports")


class AnalysisCrew:
    """
    Orchestrates the ARTHA multi-agent analysis pipeline using direct
    Python calls through the Fact Store — no CrewAI context-stacking.

    Usage::

        crew = AnalysisCrew()
        result = crew.run("RELIANCE")
        print(result["report"])
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.store = FactStore(db_url=db_url)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        symbol: str,
        job_id: Optional[str] = None,
        on_progress: Optional[Callable[[str, str], None]] = None,
        skip_llm: bool = False,
    ) -> dict:
        """
        Execute the full analysis pipeline for a symbol.

        Parameters
        ----------
        symbol : str
            NSE symbol to analyse (e.g., 'RELIANCE', 'HDFCBANK').
        job_id : str, optional
            Job ID. Auto-generated if not provided.
        on_progress : callable, optional
            Callback(stage, message) for live progress updates.
        skip_llm : bool
            If True, skip LLM calls (useful for testing data pipeline).

        Returns
        -------
        dict
            Keys: job_id, symbol, report, status, stages, duration_seconds
        """
        if job_id is None:
            job_id = f"analysis_{symbol}_{uuid.uuid4().hex[:8]}"

        symbol = symbol.upper().strip()
        started_at = time.time()
        stages = {}

        def progress(stage: str, msg: str):
            logger.info("[%s] %s", stage, msg)
            if on_progress:
                on_progress(stage, msg)

        progress("init", f"Starting analysis for {symbol} (job: {job_id})")

        # ════════════════════════════════════════════════════════════════
        # STAGE 1: Data Collection (Fundamentals + Quant Signals)
        # ════════════════════════════════════════════════════════════════
        progress("stage1", "📊 Collecting fundamentals and quant signals...")

        # 1a. Fundamentals (financial statements + market data)
        try:
            fund_result = run_fundamentals_analysis(
                symbol=symbol,
                job_id=job_id,
                store=self.store,
                start_fy=2022,
            )
            fund_count = len(fund_result.get("facts", []))
            stmt_count = fund_result.get("statement_count", 0)
            mkt_count = fund_result.get("market_count", 0)
            stages["fundamentals"] = {
                "status": "ok",
                "facts_written": fund_count,
                "statement_metrics": stmt_count,
                "market_metrics": mkt_count,
                "fiscal_years": fund_result.get("fiscal_years", []),
            }
            progress("stage1", f"✓ {fund_count} facts ({stmt_count} statement + {mkt_count} market)")
        except Exception as e:
            stages["fundamentals"] = {"status": "error", "error": str(e)}
            progress("stage1", f"⚠ Fundamentals failed: {e}")

        # NOTE: Quant signal injection has been REMOVED from the analysis pipeline.
        # The quant engine is an independent automated worker (see app/quant/simulator.py).
        # Fundamental analysis is purely about financial statements + market data.

        # ════════════════════════════════════════════════════════════════
        # STAGE 2: Agent Reasoning (Quant Narrator + Risk)
        # ════════════════════════════════════════════════════════════════
        progress("stage2", "🧠 Running agent reasoning...")

        # 2a. Fundamentals narration (LLM)
        if not skip_llm:
            try:
                fund_facts = fund_result.get("facts", []) if "fund_result" in dir() else []
                if fund_facts:
                    fund_narrative = _run_fundamentals_narration(
                        job_id, self.store, symbol, fund_facts
                    )
                    stages["fundamentals_narration"] = {"status": "ok"}
                    progress("stage2", "✓ Fundamentals narrative written")
                else:
                    stages["fundamentals_narration"] = {"status": "skipped", "reason": "no facts"}
            except Exception as e:
                stages["fundamentals_narration"] = {"status": "error", "error": str(e)}
                progress("stage2", f"⚠ Fundamentals narration failed: {e}")

        # NOTE: Quant narration removed — quant engine is independent.
        # The fundamental report narrates financial data only.

        # 2c. Risk checks (deterministic + LLM narration)
        try:
            risk_flags = run_risk_checks(
                job_id=job_id,
                store=self.store,
                symbol=symbol,
            )
            blocks = sum(1 for f in risk_flags if f.severity == Severity.BLOCK)
            warnings = sum(1 for f in risk_flags if f.severity == Severity.WARNING)
            stages["risk_checks"] = {
                "status": "ok",
                "flags": len(risk_flags),
                "blocks": blocks,
                "warnings": warnings,
            }
            progress("stage2", f"✓ Risk: {blocks} blocks, {warnings} warnings")

            if not skip_llm and risk_flags:
                risk_narrative = run_risk_narration(job_id, self.store, risk_flags)
                stages["risk_narration"] = {"status": "ok" if risk_narrative else "skipped"}
        except Exception as e:
            stages["risk_checks"] = {"status": "error", "error": str(e)}
            progress("stage2", f"⚠ Risk checks failed: {e}")

        # ════════════════════════════════════════════════════════════════
        # STAGE 3: Final Report Synthesis (Writer Agent)
        # ════════════════════════════════════════════════════════════════
        report_text = ""
        if not skip_llm:
            progress("stage3", "📝 Synthesizing final report...")
            try:
                # Build context from Fact Store
                writer_context = build_writer_context(job_id, self.store, symbol)

                # Call deep-tier LLM
                llm = get_deep_llm()
                response = llm.call([{"role": "user", "content": writer_context}])
                report_text = str(response)

                # Citation check
                citation_result = check_citations(report_text, job_id, self.store)
                stages["writer"] = {
                    "status": "ok",
                    "report_length": len(report_text),
                    "citations": citation_result,
                }
                progress(
                    "stage3",
                    f"✓ Report: {len(report_text)} chars, "
                    f"{citation_result['resolved']}/{citation_result['total_citations']} citations resolved",
                )
            except Exception as e:
                stages["writer"] = {"status": "error", "error": str(e)}
                progress("stage3", f"⚠ Writer failed: {e}")
                # Fallback: build a report from Fact Store data directly
                report_text = _build_fallback_report(job_id, self.store, symbol)
                stages["writer"]["fallback"] = True
        else:
            # skip_llm mode: build a data-only report
            report_text = _build_fallback_report(job_id, self.store, symbol)
            stages["writer"] = {"status": "skipped_llm", "fallback": True}

        # ════════════════════════════════════════════════════════════════
        # SAVE & RETURN
        # ════════════════════════════════════════════════════════════════
        duration = time.time() - started_at

        # Save report
        report_file = REPORTS_DIR / f"{job_id}.md"
        header = (
            f"# ARTHA Analysis Report — {symbol}\n\n"
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
            f"**Job ID**: {job_id}  \n"
            f"**Duration**: {duration:.1f}s  \n\n---\n\n"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(header + report_text)

        # Save metadata
        meta_file = REPORTS_DIR / f"{job_id}.json"
        meta = {
            "job_id": job_id,
            "symbol": symbol,
            "status": "completed",
            "duration_seconds": round(duration, 1),
            "stages": stages,
            "fact_counts": self.store.count_facts(job_id),
            "report_file": str(report_file),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        progress("done", f"✅ Analysis complete in {duration:.1f}s → {report_file}")

        return {
            "job_id": job_id,
            "symbol": symbol,
            "status": "completed",
            "report": report_text,
            "report_file": str(report_file),
            "stages": stages,
            "duration_seconds": round(duration, 1),
        }

    def list_reports(self) -> list[dict]:
        """List all saved analysis reports."""
        reports = []
        for f in sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                reports.append({
                    "job_id": data.get("job_id"),
                    "symbol": data.get("symbol"),
                    "status": data.get("status"),
                    "duration_seconds": data.get("duration_seconds"),
                })
            except Exception:
                continue
        return reports

    def get_report(self, job_id: str) -> Optional[str]:
        """Get a report by job ID."""
        report_file = REPORTS_DIR / f"{job_id}.md"
        if report_file.exists():
            return report_file.read_text()
        return None


# ── Fundamentals narration helper ───────────────────────────────────────────────

def _run_fundamentals_narration(
    job_id: str,
    store: FactStore,
    symbol: str,
    facts: list,
) -> NarrativeFact:
    """
    Run the LLM to narrate the fundamental data.
    
    Uses a condensed pivot table (one row per FY, key metrics as columns)
    instead of dumping all individual facts — keeps the prompt under
    Groq's 6000 TPM limit even with 100+ facts.
    
    Falls back to mid-tier if cheap-tier rate-limits.
    """
    # ── Build condensed pivot table ──────────────────────────────────────
    # Group facts by FY, pick the most important metrics
    from collections import defaultdict
    by_fy: dict[int, dict] = defaultdict(dict)
    fact_id_map: dict[str, str] = {}  # metric_fy -> fact_id (for citation)
    
    for f in facts:
        key = f"{f.metric}_{f.fiscal_year}"
        by_fy[f.fiscal_year][f.metric] = f.value
        fact_id_map[key] = f.fact_id[:8]

    # Key metrics to include (in order of importance)
    key_metrics = [
        ("revenue", "Revenue (₹Cr)"),
        ("net_income", "Net Income (₹Cr)"),
        ("ebitda", "EBITDA (₹Cr)"),
        ("eps", "EPS (₹)"),
        ("operating_margin", "OPM (%)"),
        ("net_profit_margin", "NPM (%)"),
        ("roe", "ROE (%)"),
        ("total_debt", "Total Debt (₹Cr)"),
        ("total_equity", "Equity (₹Cr)"),
        ("debt_to_equity", "D/E Ratio"),
        ("current_ratio", "Current Ratio"),
        ("operating_cashflow", "Op. Cash Flow (₹Cr)"),
        ("free_cashflow", "FCF (₹Cr)"),
        ("annual_return_pct", "Annual Return (%)"),
        ("annual_volatility_pct", "Volatility (%)"),
        ("price_close_fy_end", "FY Close (₹)"),
    ]
    
    # Build the table
    fys = sorted(by_fy.keys())
    header_cols = ["Metric"] + [f"FY{fy}" for fy in fys] + ["Fact IDs"]
    table_lines = [" | ".join(header_cols), " | ".join(["---"] * len(header_cols))]
    
    included_fact_ids = []
    for metric_key, label in key_metrics:
        row_vals = []
        row_fact_ids = []
        has_data = False
        for fy in fys:
            val = by_fy[fy].get(metric_key)
            fid = fact_id_map.get(f"{metric_key}_{fy}", "")
            if val is not None:
                row_vals.append(f"{val:,.2f}")
                row_fact_ids.append(fid)
                has_data = True
            else:
                row_vals.append("—")
        if has_data:
            ids_str = ", ".join(f"[{fid}…]" for fid in row_fact_ids if fid)
            table_lines.append(f"{label} | " + " | ".join(row_vals) + f" | {ids_str}")
            included_fact_ids.extend(row_fact_ids)

    facts_text = "\n".join(table_lines)
    
    # ── Count what's available ──────────────────────────────────────────
    stmt_metrics = sum(1 for f in facts if "YFINANCE" in f.source)
    mkt_metrics = sum(1 for f in facts if "OHLCV" in f.source)

    prompt = f"""You are the ARTHA Fundamentals Analyst — a senior equity research
analyst specialising in Indian markets (NSE). Analyse {symbol}'s fundamentals.

DATA COVERAGE: {stmt_metrics} financial statement metrics + {mkt_metrics} market metrics across FY{fys[0]}–FY{fys[-1]}.

{facts_text}

Write a 4-5 paragraph analysis titled "Fundamental Analysis — {symbol}":
1. **Revenue & Profitability**: Revenue growth trajectory, EBITDA/OPM/NPM trends, EPS progression
2. **Balance Sheet**: Debt levels, D/E ratio changes, current ratio, equity growth
3. **Cash Flow**: Operating cash flow trends, FCF generation, capex signals
4. **Market Performance**: Stock returns, volatility, FY-end price trajectory
5. **Assessment**: Overall health verdict, key strengths, risks to watch

RULES:
- NEVER invent numbers. Only narrate values from the table.
- Reference Fact IDs as [fact_id: XXXXXXXX…] for traceability.
- Highlight year-over-year changes and inflection points.
- If a metric is missing (shown as —), note the gap explicitly."""

    # Try cheap first, fall back to mid on rate limit
    for tier_fn, tier_name in [(get_cheap_llm, "cheap"), (get_mid_llm, "mid")]:
        try:
            llm = tier_fn()
            response = llm.call([{"role": "user", "content": prompt}])
            narrative_text = str(response)
            logger.info("Fundamentals narration succeeded on %s tier", tier_name)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str:
                logger.warning("Fundamentals narration rate-limited on %s, trying next tier", tier_name)
                continue
            raise
    else:
        # All tiers failed — build template narrative
        narrative_text = f"## Fundamental Analysis — {symbol}\n\n{facts_text}\n\n*Auto-generated data summary (LLM narration unavailable).*"

    fact = NarrativeFact(
        job_id=job_id,
        source="FUNDAMENTALS_AGENT_NARRATION",
        agent_name="fundamentals_agent",
        section="fundamental_analysis",
        content=narrative_text,
        referenced_fact_ids=[f.fact_id for f in facts],
    )
    store.put_fact(fact)
    return fact


# ── Fallback report builder ────────────────────────────────────────────────────

def _build_fallback_report(job_id: str, store: FactStore, symbol: str) -> str:
    """Build a data-only report when LLM is unavailable."""
    from app.factstore.schemas import (
        FactType,
        FundamentalRow,
        NarrativeFact,
        QuantSignal,
        RiskFlag,
    )

    lines = [f"## Data Summary — {symbol}\n"]

    # Fundamentals table
    fund_facts = store.get_facts_by_type(job_id, FactType.FUNDAMENTAL)
    if fund_facts:
        lines.append("### Fundamentals\n")
        lines.append("| FY | Metric | Value | Unit |")
        lines.append("|----|--------|-------|------|")
        for f in fund_facts:
            if isinstance(f, FundamentalRow):
                val = f"{f.value:.2f}" if f.value is not None else "N/A"
                lines.append(f"| {f.fiscal_year} | {f.metric} | {val} | {f.unit or '—'} |")

    # Quant signals
    quant_facts = store.get_validated_signals(job_id)
    if quant_facts:
        lines.append("\n### Quant Signals\n")
        for q in quant_facts:
            lines.append(
                f"- **{q.signal_type.value}** ({q.symbol_or_pair}): "
                f"value={q.value:.4f}, Sharpe={q.backtest_sharpe or 'N/A'}"
            )

    # Risk flags
    risk_facts = store.get_facts_by_type(job_id, FactType.RISK_FLAG)
    if risk_facts:
        lines.append("\n### Risk Flags\n")
        for r in risk_facts:
            if isinstance(r, RiskFlag):
                icon = {"info": "ℹ️", "warning": "⚠️", "block": "🛑"}.get(r.severity.value, "❓")
                lines.append(f"- {icon} **{r.flag_type}**: {r.detail}")

    # Narratives (if any agents ran before writer failed)
    narrative_facts = store.get_facts_by_type(job_id, FactType.NARRATIVE)
    if narrative_facts:
        lines.append("\n### Agent Analysis\n")
        for n in narrative_facts:
            if isinstance(n, NarrativeFact):
                lines.append(f"#### {n.agent_name} — {n.section}\n{n.content}\n")

    lines.append(
        "\n---\n\n**Disclaimer**: This report is not investment advice. "
        "It describes validated signals and historical data; it does not "
        "tell you what to buy or sell."
    )
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARTHA Analysis Crew")
    parser.add_argument("--symbol", required=True, help="NSE symbol (e.g., RELIANCE)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls (data-only)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    crew = AnalysisCrew()

    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ARTHA Analysis Crew — {args.symbol:<37s} ║")
    print("║  Fact Store → Agent Crew → Narrative Report                ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    result = crew.run(
        symbol=args.symbol,
        skip_llm=args.skip_llm,
        on_progress=lambda stage, msg: print(f"  [{stage}] {msg}"),
    )

    print(f"\n{'═' * 60}")
    print(f"  Status: {result['status']}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")
    print(f"  Report: {result['report_file']}")
    print(f"\n  Stages:")
    for name, info in result["stages"].items():
        status = info.get("status", "unknown")
        icon = "✓" if status == "ok" else "⚠" if status == "error" else "—"
        print(f"    {icon} {name}: {status}")
    print(f"{'═' * 60}\n")

    if result["report"]:
        print("─── REPORT PREVIEW (first 2000 chars) ───")
        print(result["report"][:2000])
        if len(result["report"]) > 2000:
            print(f"\n... ({len(result['report']) - 2000} more chars)")
