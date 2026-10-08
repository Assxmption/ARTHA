"""
Tests for Screener Cache & Query Engine
========================================
Covers: cache building, filter logic (gt, lt, gte, lte, eq, between),
multi-filter AND combination, sorting, sector filter, edge cases.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from app.data.screener_cache import (
    MetricInfo,
    ScreenerFilter,
    METRIC_BY_KEY,
    METRIC_CATALOG,
    VALID_OPERATORS,
    _ensure_table,
    _get_connection,
    _transform_value,
    build_screener_cache,
    get_available_metrics,
    get_cache_info,
    get_sectors,
    query_screener,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_db(tmp_path):
    """Create a test SQLite database with sample data."""
    db_path = tmp_path / "test_screener.db"
    conn = _get_connection(db_path)
    _ensure_table(conn)

    now = datetime.now().isoformat()
    stocks = [
        {
            "symbol": "HDFCBANK", "cached_at": now, "sector": "Financial Services",
            "industry": "Banks", "market_cap": 120000.0, "current_price": 1650.0,
            "pe": 22.5, "forward_pe": 20.0, "pb": 3.2, "eps": 73.3,
            "book_value": 515.6, "dividend_yield": 1.1, "roe": 16.5,
            "roa": 1.8, "debt_to_equity": 0.0, "revenue_growth": 18.0,
            "profit_margin": 22.0, "operating_margin": 28.0, "free_cashflow": 5000.0,
            "beta": 0.95, "fifty_two_wk_high": 1800.0, "fifty_two_wk_low": 1400.0,
        },
        {
            "symbol": "TCS", "cached_at": now, "sector": "Technology",
            "industry": "IT Services", "market_cap": 150000.0, "current_price": 3500.0,
            "pe": 30.0, "forward_pe": 28.0, "pb": 12.0, "eps": 116.7,
            "book_value": 291.7, "dividend_yield": 1.5, "roe": 45.0,
            "roa": 25.0, "debt_to_equity": 0.08, "revenue_growth": 12.0,
            "profit_margin": 20.0, "operating_margin": 25.0, "free_cashflow": 8000.0,
            "beta": 0.6, "fifty_two_wk_high": 4000.0, "fifty_two_wk_low": 3100.0,
        },
        {
            "symbol": "TATASTEEL", "cached_at": now, "sector": "Materials",
            "industry": "Steel", "market_cap": 18000.0, "current_price": 140.0,
            "pe": 8.0, "forward_pe": 7.5, "pb": 1.2, "eps": 17.5,
            "book_value": 116.7, "dividend_yield": 3.5, "roe": 12.0,
            "roa": 5.0, "debt_to_equity": 1.5, "revenue_growth": -5.0,
            "profit_margin": 6.0, "operating_margin": 10.0, "free_cashflow": -1000.0,
            "beta": 1.6, "fifty_two_wk_high": 160.0, "fifty_two_wk_low": 100.0,
        },
        {
            "symbol": "ITC", "cached_at": now, "sector": "Consumer Staples",
            "industry": "Tobacco", "market_cap": 60000.0, "current_price": 480.0,
            "pe": 28.0, "forward_pe": 25.0, "pb": 8.0, "eps": 17.1,
            "book_value": 60.0, "dividend_yield": 3.0, "roe": 28.0,
            "roa": 22.0, "debt_to_equity": 0.01, "revenue_growth": 8.0,
            "profit_margin": 26.0, "operating_margin": 35.0, "free_cashflow": 12000.0,
            "beta": 0.5, "fifty_two_wk_high": 510.0, "fifty_two_wk_low": 380.0,
        },
        {
            "symbol": "RELIANCE", "cached_at": now, "sector": "Energy",
            "industry": "Oil & Gas", "market_cap": 200000.0, "current_price": 2900.0,
            "pe": 25.0, "forward_pe": 22.0, "pb": 2.5, "eps": 116.0,
            "book_value": 1160.0, "dividend_yield": 0.3, "roe": 10.0,
            "roa": 6.0, "debt_to_equity": 0.4, "revenue_growth": 25.0,
            "profit_margin": 8.0, "operating_margin": 14.0, "free_cashflow": 20000.0,
            "beta": 0.85, "fifty_two_wk_high": 3200.0, "fifty_two_wk_low": 2400.0,
        },
    ]

    for stock in stocks:
        cols = list(stock.keys())
        placeholders = ", ".join(["?"] * len(cols))
        conn.execute(
            f"INSERT INTO screener_data ({', '.join(cols)}) VALUES ({placeholders})",
            [stock[c] for c in cols],
        )
    conn.commit()
    conn.close()
    return db_path


# ── Metric Catalog Tests ────────────────────────────────────────────────────────

class TestMetricCatalog:
    def test_metric_catalog_not_empty(self):
        assert len(METRIC_CATALOG) > 10

    def test_metric_keys_unique(self):
        keys = [m.key for m in METRIC_CATALOG]
        assert len(keys) == len(set(keys))

    def test_all_metrics_have_yfinance_field(self):
        for m in METRIC_CATALOG:
            assert m.yfinance_field, f"Metric {m.key} missing yfinance_field"

    def test_valid_operators_set(self):
        assert "gt" in VALID_OPERATORS
        assert "lt" in VALID_OPERATORS
        assert "between" in VALID_OPERATORS

    def test_get_available_metrics(self):
        result = get_available_metrics()
        assert len(result) > 0
        assert all("key" in m and "display_name" in m for m in result)


# ── Transform Tests ─────────────────────────────────────────────────────────────

class TestTransform:
    def test_none_returns_none(self):
        m = MetricInfo("test", "Test", "test", "ratio")
        assert _transform_value(None, m) is None

    def test_no_transform(self):
        m = MetricInfo("test", "Test", "test", "ratio")
        assert _transform_value(42.5, m) == 42.5

    def test_pct_to_100(self):
        m = MetricInfo("test", "Test", "test", "percent", "pct_to_100")
        assert _transform_value(0.15, m) == 15.0

    def test_to_crore(self):
        m = MetricInfo("test", "Test", "test", "inr_crore", "to_crore")
        result = _transform_value(1e9, m)
        assert abs(result - 100.0) < 0.01  # 1e9 / 1e7 = 100

    def test_nan_returns_none(self):
        import math
        m = MetricInfo("test", "Test", "test", "ratio")
        assert _transform_value(float("nan"), m) is None

    def test_inf_returns_none(self):
        m = MetricInfo("test", "Test", "test", "ratio")
        assert _transform_value(float("inf"), m) is None


# ── Query Engine Tests ──────────────────────────────────────────────────────────

class TestQueryNoFilter:
    def test_returns_all_when_no_filters(self, test_db):
        results, total = query_screener(db_path=test_db)
        assert total == 5
        assert len(results) == 5

    def test_default_sort_by_market_cap_desc(self, test_db):
        results, _ = query_screener(db_path=test_db)
        caps = [r["market_cap"] for r in results]
        assert caps == sorted(caps, reverse=True)

    def test_sort_ascending(self, test_db):
        results, _ = query_screener(sort_by="pe", sort_desc=False, db_path=test_db)
        pes = [r["pe"] for r in results]
        assert pes == sorted(pes)

    def test_limit(self, test_db):
        results, total = query_screener(limit=2, db_path=test_db)
        assert len(results) == 2
        assert total == 5  # Total matches still 5


class TestSingleFilter:
    def test_filter_pe_lt(self, test_db):
        filters = [ScreenerFilter("pe", "lt", 25.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 2  # HDFCBANK (22.5), TATASTEEL (8.0)
        assert all(r["pe"] < 25.0 for r in results)

    def test_filter_pe_gt(self, test_db):
        filters = [ScreenerFilter("pe", "gt", 25.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 2  # TCS (30.0), ITC (28.0)
        assert all(r["pe"] > 25.0 for r in results)

    def test_filter_roe_gte(self, test_db):
        filters = [ScreenerFilter("roe", "gte", 28.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 2  # TCS (45.0), ITC (28.0)
        assert all(r["roe"] >= 28.0 for r in results)

    def test_filter_roe_lte(self, test_db):
        filters = [ScreenerFilter("roe", "lte", 12.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 2  # TATASTEEL (12.0), RELIANCE (10.0)
        assert all(r["roe"] <= 12.0 for r in results)

    def test_filter_eq(self, test_db):
        filters = [ScreenerFilter("pe", "eq", 25.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 1  # RELIANCE
        assert results[0]["symbol"] == "RELIANCE"

    def test_filter_between(self, test_db):
        filters = [ScreenerFilter("pe", "between", 20.0, 28.0)]
        results, total = query_screener(filters=filters, db_path=test_db)
        # HDFCBANK (22.5), RELIANCE (25.0), ITC (28.0) — BETWEEN is inclusive
        assert total == 3
        assert all(20.0 <= r["pe"] <= 28.0 for r in results)


class TestMultiFilter:
    def test_and_combination(self, test_db):
        """PE < 25 AND ROE > 15 — only HDFCBANK (PE=22.5, ROE=16.5)."""
        filters = [
            ScreenerFilter("pe", "lt", 25.0),
            ScreenerFilter("roe", "gt", 15.0),
        ]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 1
        assert results[0]["symbol"] == "HDFCBANK"

    def test_value_picks_pattern(self, test_db):
        """PE < 15 AND ROE > 10 AND D/E < 1 — TATASTEEL fails D/E."""
        filters = [
            ScreenerFilter("pe", "lt", 15.0),
            ScreenerFilter("roe", "gt", 10.0),
            ScreenerFilter("debt_to_equity", "lt", 1.0),
        ]
        results, total = query_screener(filters=filters, db_path=test_db)
        # TATASTEEL has PE=8 and ROE=12 but D/E=1.5, so no matches
        assert total == 0

    def test_quality_largecap_pattern(self, test_db):
        """Market cap > 50k AND ROE > 15 AND D/E < 1."""
        filters = [
            ScreenerFilter("market_cap", "gt", 50000),
            ScreenerFilter("roe", "gt", 15.0),
            ScreenerFilter("debt_to_equity", "lt", 1.0),
        ]
        results, total = query_screener(filters=filters, db_path=test_db)
        # HDFCBANK (120k, 16.5, 0.0), TCS (150k, 45.0, 0.08), ITC (60k, 28.0, 0.01)
        assert total == 3
        symbols = {r["symbol"] for r in results}
        assert symbols == {"HDFCBANK", "TCS", "ITC"}


class TestSectorFilter:
    def test_sector_exact_match(self, test_db):
        results, total = query_screener(sector="Technology", db_path=test_db)
        assert total == 1
        assert results[0]["symbol"] == "TCS"

    def test_sector_with_metric_filter(self, test_db):
        filters = [ScreenerFilter("pe", "lt", 30)]
        results, total = query_screener(
            filters=filters, sector="Financial Services", db_path=test_db
        )
        assert total == 1
        assert results[0]["symbol"] == "HDFCBANK"

    def test_sector_no_match(self, test_db):
        results, total = query_screener(sector="Healthcare", db_path=test_db)
        assert total == 0

    def test_get_sectors(self, test_db):
        sectors = get_sectors(db_path=test_db)
        assert len(sectors) == 5
        assert "Technology" in sectors
        assert "Energy" in sectors


class TestSorting:
    def test_sort_by_roe_desc(self, test_db):
        results, _ = query_screener(sort_by="roe", sort_desc=True, db_path=test_db)
        roes = [r["roe"] for r in results]
        assert roes == sorted(roes, reverse=True)
        assert results[0]["symbol"] == "TCS"  # ROE 45.0

    def test_sort_by_pe_asc(self, test_db):
        results, _ = query_screener(sort_by="pe", sort_desc=False, db_path=test_db)
        pes = [r["pe"] for r in results]
        assert pes == sorted(pes)
        assert results[0]["symbol"] == "TATASTEEL"  # PE 8.0


class TestCacheInfo:
    def test_cache_info_populated(self, test_db):
        info = get_cache_info(db_path=test_db)
        assert info["total_symbols"] == 5
        assert info["cache_age_hours"] >= 0

    def test_cache_info_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        info = get_cache_info(db_path=db_path)
        assert info["total_symbols"] == 0


class TestEdgeCases:
    def test_empty_filters_list(self, test_db):
        results, total = query_screener(filters=[], db_path=test_db)
        assert total == 5

    def test_unknown_metric_ignored(self, test_db):
        filters = [ScreenerFilter("nonexistent_metric", "gt", 10)]
        results, total = query_screener(filters=filters, db_path=test_db)
        # Unknown metric is skipped, so all stocks returned
        assert total == 5

    def test_invalid_operator_ignored(self, test_db):
        filters = [ScreenerFilter("pe", "invalid_op", 10)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 5  # Invalid operator skipped

    def test_between_without_value2_ignored(self, test_db):
        filters = [ScreenerFilter("pe", "between", 10)]
        results, total = query_screener(filters=filters, db_path=test_db)
        assert total == 5  # between without value2 is skipped

    def test_null_values_excluded_from_filter(self, test_db):
        """Stocks with NULL metric values should not match > or < filters."""
        conn = _get_connection(test_db)
        conn.execute(
            "INSERT INTO screener_data (symbol, cached_at, pe, market_cap) "
            "VALUES (?, ?, NULL, 1000)",
            ("NULLSTOCK", datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

        filters = [ScreenerFilter("pe", "lt", 100)]
        results, total = query_screener(filters=filters, db_path=test_db)
        symbols = {r["symbol"] for r in results}
        assert "NULLSTOCK" not in symbols  # NULL PE shouldn't match pe < 100


class TestBuildCache:
    def test_build_with_mock_symbols(self, tmp_path, monkeypatch):
        """Test cache building with mocked yfinance data."""
        db_path = tmp_path / "build_test.db"

        mock_info = {
            "regularMarketPrice": 100.0,
            "marketCap": 1e11,
            "currentPrice": 100.0,
            "trailingPE": 15.0,
            "forwardPE": 14.0,
            "priceToBook": 2.0,
            "trailingEps": 6.67,
            "bookValue": 50.0,
            "dividendYield": 0.02,
            "returnOnEquity": 0.2,
            "returnOnAssets": 0.1,
            "debtToEquity": 0.5,
            "revenueGrowth": 0.15,
            "profitMargins": 0.12,
            "operatingMargins": 0.18,
            "freeCashflow": 5e9,
            "beta": 1.1,
            "sector": "TestSector",
            "industry": "TestIndustry",
            "fiftyTwoWeekHigh": 120.0,
            "fiftyTwoWeekLow": 80.0,
        }

        def mock_fetch(symbol):
            return mock_info

        monkeypatch.setattr(
            "app.data.screener_cache._fetch_single_info", mock_fetch
        )

        count = build_screener_cache(
            symbols=["TESTSTOCK1", "TESTSTOCK2"],
            db_path=db_path,
            delay_between_batches=0,
        )
        assert count == 2

        results, total = query_screener(db_path=db_path)
        assert total == 2
        # Check transformed values
        r = results[0]
        assert r["pe"] == 15.0
        assert abs(r["roe"] - 20.0) < 0.1  # 0.2 * 100
        assert abs(r["dividend_yield"] - 0.02) < 0.01  # raw 0.02 stored directly (no pct transform)
        assert abs(r["market_cap"] - 10000.0) < 1  # 1e11 / 1e7
