"""
Fundamentals Data — Screener++ Row-Level Schema Builder
========================================================
Structured, row-per-year fundamental metrics for NSE companies,
replicating the Screener.in data presentation model.

Data sources (in priority order):
  1. yfinance financials API — provides income statements, balance sheets,
     and cash flow statements for .NS (NSE) equities.
  2. Market-derived metrics from OHLCV data (always available as fallback).

Design:
  - Every FundamentalRow written to the Fact Store has a non-empty `source`.
  - Corporate actions (splits, bonuses) are handled by yfinance's auto_adjust.
  - Indian fiscal year convention: April 1 to March 31.
  - No LLM computes or asserts any number (AGENTS.md rule 1).

Reference: docs/ARTHA_ARCHITECTURE.md §4.2 (Corporate Filings row)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from app.data.nse import fetch_nse_ohlcv, normalize_nse_symbol
from app.factstore.schemas import FundamentalRow

logger = logging.getLogger(__name__)


# ── Core Screener-style metrics we extract ──────────────────────────────────────

# From income statement
_INCOME_METRICS = {
    "Total Revenue": ("revenue", "INR_crore"),
    "Gross Profit": ("gross_profit", "INR_crore"),
    "Operating Income": ("operating_income", "INR_crore"),
    "Net Income": ("net_income", "INR_crore"),
    "EBITDA": ("ebitda", "INR_crore"),
    "Operating Expense": ("operating_expense", "INR_crore"),
    "Interest Expense": ("interest_expense", "INR_crore"),
    "Tax Provision": ("tax_expense", "INR_crore"),
    "Basic EPS": ("eps", "INR"),
    "Diluted EPS": ("eps_diluted", "INR"),
}

# From balance sheet
_BALANCE_METRICS = {
    "Total Assets": ("total_assets", "INR_crore"),
    "Total Debt": ("total_debt", "INR_crore"),
    "Total Stockholder Equity": ("total_equity", "INR_crore"),
    "Net Debt": ("net_debt", "INR_crore"),
    "Cash And Cash Equivalents": ("cash", "INR_crore"),
    "Total Current Assets": ("current_assets", "INR_crore"),
    "Total Current Liabilities": ("current_liabilities", "INR_crore"),
    "Stockholders Equity": ("shareholders_equity", "INR_crore"),
}

# From cash flow statement
_CASHFLOW_METRICS = {
    "Operating Cash Flow": ("operating_cashflow", "INR_crore"),
    "Free Cash Flow": ("free_cashflow", "INR_crore"),
    "Capital Expenditure": ("capex", "INR_crore"),
}


# ── Financial statements via yfinance ───────────────────────────────────────────

def _fetch_yfinance_financials(symbol: str) -> dict:
    """
    Fetch financial statements from yfinance for an NSE equity.

    Returns a dict with keys: 'income', 'balance', 'cashflow', each
    containing a DataFrame (columns = dates, rows = line items).
    Returns empty DataFrames on failure.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        result = {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
        }

        try:
            income = ticker.financials
            if income is not None and not income.empty:
                result["income"] = income
        except Exception as e:
            logger.debug("No income statement for %s: %s", symbol, e)

        try:
            balance = ticker.balance_sheet
            if balance is not None and not balance.empty:
                result["balance"] = balance
        except Exception as e:
            logger.debug("No balance sheet for %s: %s", symbol, e)

        try:
            cashflow = ticker.cashflow
            if cashflow is not None and not cashflow.empty:
                result["cashflow"] = cashflow
        except Exception as e:
            logger.debug("No cash flow for %s: %s", symbol, e)

        return result

    except ImportError:
        logger.info("yfinance not installed, financial statements unavailable")
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}
    except Exception as e:
        logger.warning("Failed to fetch financials for %s: %s", symbol, e)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def _extract_metric_from_statement(
    statement: pd.DataFrame,
    metric_name: str,
    col_date: pd.Timestamp,
) -> Optional[float]:
    """
    Extract a single metric value from a financial statement DataFrame.

    yfinance financial statements have line items as row index and
    dates as columns.
    """
    if statement.empty:
        return None

    # Try exact match first
    if metric_name in statement.index:
        val = statement.loc[metric_name, col_date]
        if pd.notna(val):
            return float(val)

    # Try case-insensitive partial match
    for idx in statement.index:
        if isinstance(idx, str) and metric_name.lower() in idx.lower():
            val = statement.loc[idx, col_date]
            if pd.notna(val):
                return float(val)

    return None


def _date_to_indian_fy(dt: pd.Timestamp) -> int:
    """
    Convert a financial statement date to Indian fiscal year.

    A statement dated March 2025 → FY2025 (April 2024–March 2025).
    A statement dated December 2024 → FY2025 (Q3 of FY2025).
    """
    month = dt.month
    year = dt.year
    if month >= 4:
        return year + 1  # April onwards = next FY
    return year


def compute_statement_fundamentals(
    symbol: str,
) -> list[FundamentalRow]:
    """
    Compute Screener-style fundamentals from yfinance financial statements.

    Returns a list of FundamentalRow objects with real reported financials:
    revenue, net income, EBITDA, EPS, debt, equity, cash flow, etc.

    Each fact has a source like 'YFINANCE_INCOME_STMT_HDFCBANK_FY2025'
    for full traceability.
    """
    symbol = normalize_nse_symbol(symbol)
    financials = _fetch_yfinance_financials(symbol)
    rows: list[FundamentalRow] = []

    job_id_placeholder = "pending"  # Set by the agent before writing to store

    # ── Income statement metrics ────────────────────────────────────────
    income = financials["income"]
    if not income.empty:
        for col_date in income.columns:
            fy = _date_to_indian_fy(pd.Timestamp(col_date))
            source = f"YFINANCE_INCOME_STMT_{symbol}_FY{fy}"

            for yf_name, (metric, unit) in _INCOME_METRICS.items():
                value = _extract_metric_from_statement(income, yf_name, col_date)
                if value is not None:
                    # Convert from raw (typically in local currency) to crore
                    # yfinance reports in the stock's currency (INR for .NS)
                    display_value = value / 1e7 if unit == "INR_crore" else value
                    rows.append(FundamentalRow(
                        job_id=job_id_placeholder,
                        symbol=symbol,
                        fiscal_year=fy,
                        metric=metric,
                        value=round(display_value, 2),
                        unit=unit,
                        source=source,
                    ))

            # ── Derived ratios from income statement ────────────────────
            rev = _extract_metric_from_statement(income, "Total Revenue", col_date)
            op_inc = _extract_metric_from_statement(income, "Operating Income", col_date)
            net_inc = _extract_metric_from_statement(income, "Net Income", col_date)

            if rev and rev != 0 and op_inc is not None:
                opm = (op_inc / rev) * 100
                rows.append(FundamentalRow(
                    job_id=job_id_placeholder,
                    symbol=symbol,
                    fiscal_year=fy,
                    metric="operating_margin",
                    value=round(opm, 2),
                    unit="percent",
                    source=source,
                ))

            if rev and rev != 0 and net_inc is not None:
                npm = (net_inc / rev) * 100
                rows.append(FundamentalRow(
                    job_id=job_id_placeholder,
                    symbol=symbol,
                    fiscal_year=fy,
                    metric="net_profit_margin",
                    value=round(npm, 2),
                    unit="percent",
                    source=source,
                ))

    # ── Balance sheet metrics ───────────────────────────────────────────
    balance = financials["balance"]
    if not balance.empty:
        for col_date in balance.columns:
            fy = _date_to_indian_fy(pd.Timestamp(col_date))
            source = f"YFINANCE_BALANCE_SHEET_{symbol}_FY{fy}"

            for yf_name, (metric, unit) in _BALANCE_METRICS.items():
                value = _extract_metric_from_statement(balance, yf_name, col_date)
                if value is not None:
                    display_value = value / 1e7 if unit == "INR_crore" else value
                    rows.append(FundamentalRow(
                        job_id=job_id_placeholder,
                        symbol=symbol,
                        fiscal_year=fy,
                        metric=metric,
                        value=round(display_value, 2),
                        unit=unit,
                        source=source,
                    ))

            # ── Derived ratios ──────────────────────────────────────────
            total_debt = _extract_metric_from_statement(balance, "Total Debt", col_date)
            total_equity = _extract_metric_from_statement(balance, "Stockholders Equity", col_date)
            current_assets = _extract_metric_from_statement(balance, "Total Current Assets", col_date)
            current_liab = _extract_metric_from_statement(balance, "Total Current Liabilities", col_date)

            if total_equity and total_equity != 0 and total_debt is not None:
                d2e = total_debt / total_equity
                rows.append(FundamentalRow(
                    job_id=job_id_placeholder,
                    symbol=symbol,
                    fiscal_year=fy,
                    metric="debt_to_equity",
                    value=round(d2e, 2),
                    unit="ratio",
                    source=source,
                ))

            if current_liab and current_liab != 0 and current_assets is not None:
                cr = current_assets / current_liab
                rows.append(FundamentalRow(
                    job_id=job_id_placeholder,
                    symbol=symbol,
                    fiscal_year=fy,
                    metric="current_ratio",
                    value=round(cr, 2),
                    unit="ratio",
                    source=source,
                ))

            # ROE = Net Income / Shareholders' Equity
            net_inc_for_roe = None
            if not income.empty:
                # Find the income statement column closest to this balance sheet date
                for inc_date in income.columns:
                    if _date_to_indian_fy(pd.Timestamp(inc_date)) == fy:
                        net_inc_for_roe = _extract_metric_from_statement(
                            income, "Net Income", inc_date
                        )
                        break

            if net_inc_for_roe is not None and total_equity and total_equity != 0:
                roe = (net_inc_for_roe / total_equity) * 100
                rows.append(FundamentalRow(
                    job_id=job_id_placeholder,
                    symbol=symbol,
                    fiscal_year=fy,
                    metric="roe",
                    value=round(roe, 2),
                    unit="percent",
                    source=source,
                ))

    # ── Cash flow metrics ───────────────────────────────────────────────
    cashflow = financials["cashflow"]
    if not cashflow.empty:
        for col_date in cashflow.columns:
            fy = _date_to_indian_fy(pd.Timestamp(col_date))
            source = f"YFINANCE_CASHFLOW_{symbol}_FY{fy}"

            for yf_name, (metric, unit) in _CASHFLOW_METRICS.items():
                value = _extract_metric_from_statement(cashflow, yf_name, col_date)
                if value is not None:
                    display_value = value / 1e7 if unit == "INR_crore" else value
                    rows.append(FundamentalRow(
                        job_id=job_id_placeholder,
                        symbol=symbol,
                        fiscal_year=fy,
                        metric=metric,
                        value=round(display_value, 2),
                        unit=unit,
                        source=source,
                    ))

    if rows:
        logger.info(
            "Extracted %d fundamental metrics for %s from financial statements",
            len(rows), symbol,
        )
    else:
        logger.info("No financial statements available for %s, falling back to market data", symbol)

    return rows


# ── Market-derived fundamentals (always available) ──────────────────────────────

def compute_market_fundamentals(
    symbol: str,
    fiscal_year: int,
    end_date: Optional[date] = None,
) -> list[FundamentalRow]:
    """
    Compute market-derived fundamental metrics for a symbol and fiscal year.

    These metrics are always available from OHLCV data, even when financial
    statements are not.  They supplement the statement-based metrics.

    Indian fiscal years run April 1 to March 31.  FY2025 = April 2024–March 2025.
    """
    symbol = normalize_nse_symbol(symbol)

    fy_start = date(fiscal_year - 1, 4, 1)
    fy_end = end_date or date(fiscal_year, 3, 31)

    today = date.today()
    if fy_end > today:
        fy_end = today

    df = fetch_nse_ohlcv(symbol, fy_start, fy_end)

    if df.empty or len(df) < 2:
        logger.warning(
            "Insufficient OHLCV data for %s FY%d (%d rows)", symbol, fiscal_year, len(df)
        )
        return []

    source = f"NSE_OHLCV_{symbol}_FY{fiscal_year}"
    job_id_placeholder = "pending"

    rows: list[FundamentalRow] = []

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    if first_close > 0:
        annual_return = ((last_close / first_close) - 1) * 100
        rows.append(FundamentalRow(
            job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
            metric="annual_return_pct", value=round(annual_return, 2),
            unit="percent", source=source,
        ))

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    if len(log_returns) >= 5:
        daily_vol = log_returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        rows.append(FundamentalRow(
            job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
            metric="annual_volatility_pct", value=round(annual_vol, 2),
            unit="percent", source=source,
        ))

    avg_vol = df["volume"].mean()
    rows.append(FundamentalRow(
        job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
        metric="avg_daily_volume", value=round(avg_vol, 0),
        unit="shares", source=source,
    ))

    rows.append(FundamentalRow(
        job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
        metric="price_high_52w", value=round(float(df["high"].max()), 2),
        unit="INR", source=source,
    ))
    rows.append(FundamentalRow(
        job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
        metric="price_low_52w", value=round(float(df["low"].min()), 2),
        unit="INR", source=source,
    ))
    rows.append(FundamentalRow(
        job_id=job_id_placeholder, symbol=symbol, fiscal_year=fiscal_year,
        metric="price_close_fy_end", value=round(float(last_close), 2),
        unit="INR", source=source,
    ))

    return rows


# ── Combined pipeline ──────────────────────────────────────────────────────────

def get_full_fundamentals(
    symbol: str,
    start_fy: int = 2023,
    end_fy: Optional[int] = None,
) -> list[FundamentalRow]:
    """
    Get the full Screener++ fundamentals profile for a symbol.

    Combines:
      1. Financial statement data (income, balance sheet, cash flow)
         from yfinance — this is the Screener.in-style data.
      2. Market-derived metrics (return, vol, volume, price range)
         from OHLCV data — always available.

    Returns all FundamentalRow objects, sorted by fiscal year and metric.
    """
    symbol = normalize_nse_symbol(symbol)
    if end_fy is None:
        today = date.today()
        end_fy = today.year if today.month >= 4 else today.year

    all_rows: list[FundamentalRow] = []

    # 1. Financial statements (covers all available years automatically)
    statement_rows = compute_statement_fundamentals(symbol)
    all_rows.extend(statement_rows)

    # 2. Market-derived metrics (per fiscal year)
    for fy in range(start_fy, end_fy + 1):
        market_rows = compute_market_fundamentals(symbol, fy)
        all_rows.extend(market_rows)

    # De-duplicate: if a metric appears in both sources for the same FY,
    # prefer the financial statement version (more authoritative).
    seen = set()
    deduped: list[FundamentalRow] = []
    for row in all_rows:
        key = (row.symbol, row.fiscal_year, row.metric)
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return sorted(deduped, key=lambda r: (r.fiscal_year, r.metric))


def get_multi_year_fundamentals(
    symbol: str,
    start_fy: int,
    end_fy: int,
) -> list[FundamentalRow]:
    """Alias for get_full_fundamentals with explicit FY range."""
    return get_full_fundamentals(symbol, start_fy, end_fy)


def fundamentals_to_dataframe(rows: list[FundamentalRow]) -> pd.DataFrame:
    """
    Pivot FundamentalRow facts into a Screener-style table:
    rows = fiscal years, columns = metrics.
    """
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "symbol": r.symbol,
            "fiscal_year": r.fiscal_year,
            "metric": r.metric,
            "value": r.value,
            "unit": r.unit,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index="fiscal_year",
        columns="metric",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.sort_index()
    return pivot
