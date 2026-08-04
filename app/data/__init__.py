"""
Data Layer Package
==================
Bounded, structured, cached data ingestion for ARTHA.

Replaces the unbounded web search/scrape approach with:
  - NSE equities via jugaad-data + yfinance fallback
  - MCX commodities via bhavcopy ingestion (Phase 3)
  - Structured fundamentals (Screener++ row-level)
"""
