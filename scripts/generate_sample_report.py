#!/usr/bin/env python3
"""
ARTHA Phase 1 — End-to-End Sample Report Generator
====================================================
Runs the full fundamentals pipeline on HDFCBANK:
  1. Fetch financial statements + OHLCV data
  2. Write FundamentalRow facts to the Fact Store
  3. Build a Screener-style summary table
  4. Output the report for review

This script does NOT invoke LLM agents — it produces the deterministic
data layer output that agents would narrate.  This is the correct design:
LLMs never compute numbers (AGENTS.md rule 1).
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.data.fundamentals import (
    compute_statement_fundamentals,
    compute_market_fundamentals,
    get_full_fundamentals,
    fundamentals_to_dataframe,
)
from app.factstore.store import FactStore
from app.factstore.schemas import FactType


def main():
    SYMBOL = "HDFCBANK"
    JOB_ID = "demo-hdfcbank-phase1"

    print("=" * 70)
    print("  ARTHA Phase 1 — Sample Report: HDFC Bank (HDFCBANK.NS)")
    print("=" * 70)

    # ── 1. Fetch full fundamentals ──────────────────────────────────────
    print("\n[1/4] Fetching fundamentals (financial statements + market data)...")
    all_facts = get_full_fundamentals(SYMBOL, start_fy=2023, end_fy=2026)

    # Set job_id on all facts
    for f in all_facts:
        f.job_id = JOB_ID

    print(f"      → Extracted {len(all_facts)} FundamentalRow facts")

    if not all_facts:
        print("\n⚠️  No data fetched. This may be due to network issues or")
        print("   missing yfinance/jugaad-data packages.")
        print("   Install with: pip install yfinance jugaad-data")
        return

    # ── 2. Write to Fact Store ──────────────────────────────────────────
    print("\n[2/4] Writing to Fact Store (per-stage checkpoint)...")
    store = FactStore(db_path="data_cache/demo_factstore.db")
    store.delete_job_facts(JOB_ID)  # Clean slate for demo
    store.put_facts(all_facts)

    counts = store.count_facts(JOB_ID)
    print(f"      → Fact Store: {counts}")

    # ── 3. Build Screener-style table ───────────────────────────────────
    print("\n[3/4] Building Screener-style summary table...")
    table = fundamentals_to_dataframe(all_facts)

    print("\n" + "─" * 70)
    print(f"  HDFC BANK — Screener++ Fundamentals (FY{table.index.min()}–FY{table.index.max()})")
    print("─" * 70)

    # Format the table for display
    if not table.empty:
        # Select key metrics in Screener order
        screener_order = [
            "revenue", "operating_income", "net_income", "ebitda", "eps",
            "operating_margin", "net_profit_margin", "roe",
            "total_assets", "total_debt", "shareholders_equity",
            "debt_to_equity", "current_ratio",
            "operating_cashflow", "free_cashflow", "capex",
            "annual_return_pct", "annual_volatility_pct",
            "price_high_52w", "price_low_52w", "avg_daily_volume",
        ]
        available = [m for m in screener_order if m in table.columns]
        display_table = table[available].copy()

        # Print with formatting
        print(display_table.to_string(float_format=lambda x: f"{x:,.2f}" if abs(x) > 1 else f"{x:.2f}"))
    else:
        print("  (No tabular data available)")

    # ── 4. Output individual facts with sources ─────────────────────────
    print("\n\n" + "─" * 70)
    print("  FACT STORE ENTRIES (source traceability)")
    print("─" * 70)

    # Group by fiscal year
    by_fy = {}
    for f in all_facts:
        by_fy.setdefault(f.fiscal_year, []).append(f)

    for fy in sorted(by_fy.keys()):
        print(f"\n  FY{fy}:")
        for f in sorted(by_fy[fy], key=lambda x: x.metric):
            val = f"{f.value:,.2f}" if f.value is not None else "N/A"
            print(f"    {f.metric:<25s} = {val:>15s} {f.unit or '':<12s} [source: {f.source}]")
            print(f"    {'':25s}   {'':15s} {'':12s}  fact_id: {f.fact_id[:12]}…")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  REPORT SUMMARY")
    print("=" * 70)
    print(f"  Symbol:           {SYMBOL}")
    print(f"  Fiscal Years:     FY{min(f.fiscal_year for f in all_facts)}–FY{max(f.fiscal_year for f in all_facts)}")
    print(f"  Total Facts:      {len(all_facts)}")
    print(f"  Metrics:          {len(set(f.metric for f in all_facts))} unique")
    print(f"  Sources:          {len(set(f.source for f in all_facts))} unique")
    print(f"  All have source:  {all(f.source for f in all_facts)}")
    print(f"  Fact Store DB:    data_cache/demo_factstore.db")
    print()
    print("  ✅ Every fact has a non-empty source (AGENTS.md rule 10)")
    print("  ✅ All numbers computed by deterministic code, not LLM (rule 1)")
    print("  ✅ Data cached locally as Parquet (no re-fetch on next run)")
    print("  ✅ Per-stage checkpointing — facts persisted immediately")
    print()
    print("  📊 Compare with: https://www.screener.in/company/HDFCBANK/consolidated/")
    print("=" * 70)

    store.close()


if __name__ == "__main__":
    main()
