# Data: Fundamentals

Located in `app/data/fundamentals.py`.

This module acts as the "Screener.in" backend. It builds a row-level fundamental schema for the [[Fundamentals_Agent]] to narrate.

## Data Sources
1. **Primary**: `yfinance` financials API. Extracts the Income Statement, Balance Sheet, and Cash Flow statement.
2. **Fallback**: Market-derived metrics from OHLCV data (e.g., 52-week highs, annual volatility).

## Extracted Metrics
The module extracts specific line items and normalizes them into INR Crores:
- **Income**: Revenue, Gross Profit, Operating Income, Net Income, EBITDA, EPS.
- **Balance Sheet**: Total Assets, Total Debt, Stockholders Equity, Cash Equivalents.
- **Cash Flow**: Operating Cash Flow, Free Cash Flow, Capex.

## Derived Ratios
If raw line items exist, the module computes classical fundamental ratios on the fly:
- Operating Profit Margin (OPM)
- Net Profit Margin (NPM)
- Debt to Equity
- Current Ratio
- Return on Equity (ROE)

## Fiscal Year Mapping
Since Indian financial reporting runs April 1st to March 31st, the module maps statements dated e.g., December 2024 to `FY2025` using the `_date_to_indian_fy` helper function.

## Traceability
Every metric is bundled into a `FundamentalRow` Pydantic object and appended with a precise `source` string (e.g., `YFINANCE_INCOME_STMT_HDFCBANK_FY2025`). This enforces the traceability rule so the LLM can cite its numbers.
