"""Former-ticker resolution and local instrument search."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.data import symbols
from app.data.security_master import PointInTimeSecurityMaster

MASTER = {
    "schema_version": "nse_security_master_v1",
    "records": [
        {"security_id": "NSE-ZOMATO", "canonical_symbol": "ETERNAL", "listed_from": "2021-07-23",
         "aliases": "ZOMATO;ETERNAL LTD"},
        {"security_id": "NSE-TATAMOTORS", "canonical_symbol": "TMPV", "listed_from": "1998-07-22",
         "aliases": "TATAMOTORS;TATA MOTORS PASSENGER VEHICLES"},
        {"security_id": "NSE-TMCV", "canonical_symbol": "TMCV", "listed_from": "2025-11-12"},
        {"security_id": "NSE-TATASTEEL", "canonical_symbol": "TATASTEEL", "listed_from": "1998-01-01"},
        {"security_id": "NSE-HDFC-MERGED", "canonical_symbol": "HDFC", "listed_from": "1996-02-21",
         "delisted_on": "2023-07-13", "aliases": "HDFC LTD"},
    ],
}

KITE_NSE = pd.DataFrame([
    # tradingsymbol, name, segment, instrument_type
    ("ETERNAL", "ETERNAL - ZOMATO", "NSE", "EQ"),
    ("TMPV", "TATA MOTORS PASS VEH", "NSE", "EQ"),
    ("TMCV", "TATA MOTORS", "NSE", "EQ"),
    ("TATASTEEL", "TATA STEEL", "NSE", "EQ"),
    ("TATACAP", "TATA CAPITAL", "NSE", "EQ"),
    ("765TMCV27-N0", "", "NSE", "EQ"),             # debt series: must be excluded
    ("SGBJUN31I-GB", "SGB JUN31", "NSE", "EQ"),     # gold bond series: excluded
    ("BAJAJ-AUTO", "BAJAJ AUTO", "NSE", "EQ"),      # real ticker with a hyphen: kept
    ("NIFTY 50", "NIFTY 50", "INDICES", "EQ"),
    ("NIFTY BANK", "NIFTY BANK", "INDICES", "EQ"),
    ("NIFTY IT", "NIFTY IT", "INDICES", "EQ"),
], columns=["tradingsymbol", "name", "segment", "instrument_type"])


@pytest.fixture
def master_path(tmp_path: Path) -> Path:
    p = tmp_path / "master.json"
    p.write_text(json.dumps(MASTER))
    return p


@pytest.fixture
def index(master_path, monkeypatch):
    idx = symbols.InstrumentIndex(master_path)
    monkeypatch.setattr(idx, "_kite_frame", lambda: (KITE_NSE, "2026-11-24"))
    return idx


def test_alias_map_only_ticker_shaped_aliases(master_path):
    assert symbols._alias_map(master_path) == {"ZOMATO": "ETERNAL", "TATAMOTORS": "TMPV"}


def test_resolve_symbol_passthrough_and_case(master_path, monkeypatch):
    monkeypatch.setattr(symbols, "MASTER_PATH", master_path)
    assert symbols.resolve_symbol("zomato") == "ETERNAL"
    assert symbols.resolve_symbol("TATAMOTORS.NS") == "TMPV"
    assert symbols.resolve_symbol("INFY") == "INFY"
    # Name-shaped aliases (with spaces) are never treated as tickers.
    assert symbols.resolve_symbol("ETERNAL LTD") == "ETERNAL LTD"


def test_missing_master_resolves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(symbols, "MASTER_PATH", tmp_path / "missing.json")
    assert symbols.resolve_symbol("ZOMATO") == "ZOMATO"


def test_kite_and_yahoo_mapping_use_current_ticker(master_path, monkeypatch):
    from app.data import kite
    from app.data.nse import _yfinance_symbol

    monkeypatch.setattr(symbols, "MASTER_PATH", master_path)
    assert kite.kite_symbol("ZOMATO") == ("NSE", "ETERNAL")
    assert _yfinance_symbol("TATAMOTORS") == "TMPV.NS"
    assert _yfinance_symbol("NIFTY") == "^NSEI"


def test_former_ticker_finds_current_listing(index):
    top = index.search("zomato")[0]
    assert top["symbol"] == "ETERNAL" and top["matched_former"] == "ZOMATO" and top["in_universe"]
    assert index.search("ZOM")[0]["symbol"] == "ETERNAL"
    assert index.search("tatamotors")[0]["symbol"] == "TMPV"


def test_company_name_finds_both_demerged_listings(index):
    found = [r["symbol"] for r in index.search("tata motors")]
    assert set(found[:2]) == {"TMPV", "TMCV"}


def test_universe_ranks_ahead_of_other_prefix_matches(index):
    found = [r["symbol"] for r in index.search("tata")]
    assert found.index("TMPV") < found.index("TATACAP")
    assert found.index("TATASTEEL") < found.index("TATACAP")


def test_debt_series_excluded_and_hyphen_tickers_kept(index):
    syms = {r["symbol"] for r in index.search("tmcv", limit=20)}
    assert syms == {"TMCV"}
    assert index.search("sgb") == []
    assert index.search("bajaj-auto")[0]["symbol"] == "BAJAJ-AUTO"


def test_indices_use_chartable_aliases(index):
    found = [r["symbol"] for r in index.search("nifty")]
    assert found[:2] == ["NIFTY", "BANKNIFTY"]
    assert all(r["type"] == "INDEX" for r in index.search("nifty"))


def test_delisted_names_not_offered(index):
    assert all(r["symbol"] != "HDFC" for r in index.search("hdfc", limit=20))


def test_empty_query_and_no_kite(master_path, monkeypatch):
    idx = symbols.InstrumentIndex(master_path)
    monkeypatch.setattr(idx, "_kite_frame", lambda: (None, None))
    assert idx.search("   ") == []
    # Without Kite the security master alone still answers, former tickers included.
    assert idx.search("zomato")[0]["symbol"] == "ETERNAL"
    assert idx.sources == ["security-master"]


def test_live_master_has_current_tickers():
    path = Path("data_cache/nse/security_master.json")
    if not path.exists():
        pytest.skip("security master not present")
    m = PointInTimeSecurityMaster.from_file(path)
    assert m.canonicalize("ZOMATO") == "ETERNAL"
    assert m.canonicalize("TATAMOTORS") == "TMPV"
    assert m.record_for("TMCV").listed_from == date(2025, 11, 12)
    active = set(m.symbols_at(date(2026, 11, 24)))
    assert {"ETERNAL", "TMPV", "TMCV"} <= active
    assert not {"ZOMATO", "TATAMOTORS"} & active
    assert "TMCV" not in m.symbols_at(date(2025, 11, 11))  # point-in-time: not listed yet
