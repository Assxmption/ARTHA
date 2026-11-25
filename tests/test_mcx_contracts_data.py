"""
Tests for MCX Contract Master data file
========================================
Validates that data_cache/mcx/contracts.json loads correctly through
the MCXContractMaster class and that contract specs match official
MCX circular values.

AGENTS.md Rule 10: Every number resolves to an authoritative source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.data.mcx_contracts import (
    MCXContractMaster,
    MCXContractMetadataError,
    MCXContractSpec,
    contract_metadata_for_frame,
)

CONTRACTS_PATH = Path("data_cache/mcx/contracts.json")


@pytest.fixture(scope="module")
def master() -> MCXContractMaster:
    """Load the production MCX contracts file."""
    if not CONTRACTS_PATH.exists():
        pytest.skip(f"MCX contracts file not found at {CONTRACTS_PATH}")
    return MCXContractMaster()


# ── Loading & Schema ────────────────────────────────────────────────────────────

class TestMCXContractLoading:
    """Verify the contracts file loads and contains reasonable data."""

    def test_loads_successfully(self, master: MCXContractMaster):
        assert len(master.list()) > 0

    def test_minimum_coverage(self, master: MCXContractMaster):
        """Should have at least 10 contracts (major energy + bullion + base metals)."""
        assert len(master.list()) >= 10

    def test_json_schema_version(self):
        """Master file should declare its schema version."""
        payload = json.loads(CONTRACTS_PATH.read_text())
        assert payload.get("schema_version") == "mcx_contract_master_v1"

    def test_all_specs_valid(self, master: MCXContractMaster):
        """Every spec should pass its own validation."""
        for spec in master.list():
            assert spec.lot_size > 0
            assert spec.tick_size > 0
            assert spec.quote_unit.strip()
            assert spec.currency.strip()


# ── Individual Contract Verification ─────────────────────────────────────────────

class TestCrudeOilContracts:
    """Verify crude oil contract specs match MCX circulars."""

    def test_crude_oil_lot_size(self, master: MCXContractMaster):
        spec = master.require("CRUDEOIL")
        assert spec.lot_size == 100  # 100 barrels per lot

    def test_crude_oil_mini_lot_size(self, master: MCXContractMaster):
        spec = master.require("CRUDEOILM")
        assert spec.lot_size == 10  # Mini contract = 10 barrels

    def test_crude_oil_quote_unit(self, master: MCXContractMaster):
        spec = master.require("CRUDEOIL")
        assert spec.quote_unit == "BBL"  # Barrels

    def test_crude_oil_currency(self, master: MCXContractMaster):
        spec = master.require("CRUDEOIL")
        assert spec.currency == "INR"


class TestGoldContracts:
    """Verify gold contract specs."""

    def test_gold_lot_size(self, master: MCXContractMaster):
        spec = master.require("GOLD")
        assert spec.lot_size == 1000  # 1 kg = 1000 grams

    def test_gold_mini_lot_size(self, master: MCXContractMaster):
        spec = master.require("GOLDM")
        assert spec.lot_size == 100  # 100 grams

    def test_gold_petal_lot_size(self, master: MCXContractMaster):
        spec = master.require("GOLDPETAL")
        assert spec.lot_size == 1  # 1 gram

    def test_gold_guinea_lot_size(self, master: MCXContractMaster):
        spec = master.require("GOLDGUINEA")
        assert spec.lot_size == 8  # 8 grams

    def test_gold_quote_unit(self, master: MCXContractMaster):
        spec = master.require("GOLD")
        assert spec.quote_unit == "GMS"  # Grams


class TestSilverContracts:
    """Verify silver contract specs."""

    def test_silver_lot_size(self, master: MCXContractMaster):
        spec = master.require("SILVER")
        assert spec.lot_size == 30000  # 30 kg = 30000 grams

    def test_silver_mini_lot_size(self, master: MCXContractMaster):
        spec = master.require("SILVERM")
        assert spec.lot_size == 5000  # 5 kg

    def test_silver_micro_lot_size(self, master: MCXContractMaster):
        spec = master.require("SILVERMIC")
        assert spec.lot_size == 1000  # 1 kg


class TestBaseMetalContracts:
    """Verify base metal contracts are present and have valid specs."""

    @pytest.mark.parametrize("commodity", [
        "COPPER", "ZINC", "LEAD", "NICKEL", "ALUMINIUM",
    ])
    def test_base_metal_present(self, master: MCXContractMaster, commodity: str):
        spec = master.require(commodity)
        assert spec.lot_size > 0
        assert spec.currency == "INR"


class TestNaturalGasContract:
    """Verify natural gas contract spec."""

    def test_natural_gas_present(self, master: MCXContractMaster):
        spec = master.require("NATURALGAS")
        assert spec.lot_size > 0
        assert spec.quote_unit == "MMBTU"


# ── Lookup Edge Cases ────────────────────────────────────────────────────────────

class TestLookupEdgeCases:
    """Test lookup, require, and error handling."""

    def test_case_insensitive_lookup(self, master: MCXContractMaster):
        """Commodity lookup should be case-insensitive."""
        assert master.get("crudeoil") is not None
        assert master.get("CrudeOil") is not None

    def test_require_raises_for_unknown(self, master: MCXContractMaster):
        """require() should raise for unknown commodities."""
        with pytest.raises(MCXContractMetadataError):
            master.require("NONEXISTENT_COMMODITY_XYZ")

    def test_get_returns_none_for_unknown(self, master: MCXContractMaster):
        """get() should return None, not raise, for unknown commodities."""
        assert master.get("NONEXISTENT_COMMODITY_XYZ") is None

    def test_list_by_commodity_filter(self, master: MCXContractMaster):
        """Filtering by commodity should return only that commodity's specs."""
        gold_specs = master.list("GOLD")
        assert len(gold_specs) == 1  # Only GOLD|FUTCOM, not GOLDM/GOLDPETAL
        assert gold_specs[0].commodity == "GOLD"


# ── Frame Enrichment ─────────────────────────────────────────────────────────────

class TestFrameEnrichment:
    """Test contract_metadata_for_frame helper."""

    def test_enriches_valid_frame(self, master: MCXContractMaster):
        """Should add lot_size, tick_size, etc. columns."""
        df = pd.DataFrame([
            {"commodity": "CRUDEOIL", "instrument": "FUTCOM", "close": 5000.0},
            {"commodity": "GOLD", "instrument": "FUTCOM", "close": 60000.0},
        ])
        enriched = contract_metadata_for_frame(df, master)
        assert "lot_size" in enriched.columns
        assert "tick_size" in enriched.columns
        assert enriched.iloc[0]["lot_size"] == 100   # CRUDEOIL
        assert enriched.iloc[1]["lot_size"] == 1000   # GOLD

    def test_empty_frame_returns_empty(self, master: MCXContractMaster):
        """Empty DataFrame should not crash."""
        df = pd.DataFrame()
        result = contract_metadata_for_frame(df, master)
        assert result.empty

    def test_unknown_commodity_raises(self, master: MCXContractMaster):
        """Frame with unknown commodity should raise, not silently skip."""
        df = pd.DataFrame([
            {"commodity": "UNOBTANIUM", "instrument": "FUTCOM", "close": 999.0},
        ])
        with pytest.raises(MCXContractMetadataError):
            contract_metadata_for_frame(df, master)
