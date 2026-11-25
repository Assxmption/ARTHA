"""MCX ingestion tests using deterministic provider fixtures."""

from datetime import date

import pandas as pd
import pytest

from app.data.mcx import (
    MCX_COLUMNS,
    fetch_mcx_bhavcopy,
    fetch_mcx_history,
    normalize_mcx_bhavcopy,
)
from app.data.mcx_contracts import (
    MCXContractMaster,
    MCXContractMetadataError,
    MCXContractSpec,
    contract_metadata_for_frame,
)


ROWS = [
    {
        "Date": "03/11/2026",
        "Instrument": "FUTCOM",
        "Commodity": "GOLD",
        "Expiry Date": "05 Feb 2027",
        "Open": "123,000",
        "High": "123500",
        "Low": "122500",
        "Close": "123250",
        "PCP": "122900",
        "Vol (Lots)": "1200",
        "Volume (000's)": "12.000 KGS",
        "Value (Lakhs)": "12345.5",
        "OI (Lots)": "999",
    },
    {
        "Date": "03/11/2026",
        "Instrument": "FUTCOM",
        "Commodity": "SILVER",
        "Expiry Date": "31 Dec 2026",
        "Open": "150000",
        "High": "151000",
        "Low": "149000",
        "Close": "150500",
        "PCP": "149500",
        "Vol (Lots)": "10",
        "OI (Lots)": "20",
    },
]


class FixtureProvider:
    def __init__(self, by_date):
        self.by_date = by_date
        self.calls = []

    def fetch_bhavcopy(self, trade_date, instrument="ALL"):
        self.calls.append((trade_date, instrument))
        return self.by_date.get(trade_date, [])


def test_normalize_mcx_rows_preserves_contract_identity():
    frame = normalize_mcx_bhavcopy(ROWS)
    assert list(frame.columns) == MCX_COLUMNS
    assert len(frame) == 2
    assert frame.loc[0, "trade_date"] == date(2026, 11, 3)
    assert frame.loc[0, "expiry_date"] == date(2027, 2, 5)
    assert frame.loc[0, "close"] == 123250.0
    assert frame.loc[0, "volume_thousands"] == 12.0
    assert "GOLD|FUTCOM|2027-02-05" in frame.loc[0, "contract_key"]


def test_normalize_csv_payload_and_drop_unusable_rows():
    csv = """Date,Instrument,Commodity,Expiry Date,Close\n03/11/2026,FUTCOM,GOLD,05 Feb 2027,123250\n03/11/2026,FUTCOM,GOLD,05 Feb 2027,-\n"""
    frame = normalize_mcx_bhavcopy(csv)
    assert len(frame) == 1
    assert frame.iloc[0]["commodity"] == "GOLD"


def test_empty_holiday_is_cached_and_not_forward_filled(tmp_path):
    provider = FixtureProvider({date(2026, 11, 2): [], date(2026, 11, 3): ROWS})
    first = fetch_mcx_history(
        date(2026, 11, 2), date(2026, 11, 3),
        instrument="FUTCOM", cache_dir=tmp_path, provider=provider,
    )
    assert len(first) == 2
    assert set(first["trade_date"]) == {date(2026, 11, 3)}
    assert (tmp_path / "bhavcopy_2026-11-02.parquet").exists()

    second = fetch_mcx_history(
        date(2026, 11, 2), date(2026, 11, 3),
        instrument="FUTCOM", cache_dir=tmp_path, provider=provider,
    )
    assert len(second) == 2
    assert len(provider.calls) == 2


def test_cache_prevents_refetch(tmp_path):
    provider = FixtureProvider({date(2026, 11, 3): ROWS})
    one = fetch_mcx_bhavcopy(date(2026, 11, 3), cache_dir=tmp_path, provider=provider)
    two = fetch_mcx_bhavcopy(date(2026, 11, 3), cache_dir=tmp_path, provider=provider)
    pd.testing.assert_frame_equal(one, two)
    assert len(provider.calls) == 1


def test_future_date_rejected(tmp_path):
    with pytest.raises(ValueError, match="future"):
        fetch_mcx_bhavcopy(date(2099, 1, 1), cache_dir=tmp_path, provider=FixtureProvider({}))


def test_contract_master_fails_closed_without_sizing_metadata():
    master = MCXContractMaster(specs=[])
    with pytest.raises(MCXContractMetadataError, match="No contract metadata"):
        master.require("GOLD", "FUTCOM")


def test_contract_metadata_enriches_frame_for_sizing():
    frame = normalize_mcx_bhavcopy(ROWS)
    master = MCXContractMaster(specs=[
        MCXContractSpec(
            commodity="GOLD",
            instrument="FUTCOM",
            lot_size=100.0,
            tick_size=1.0,
            quote_unit="INR per 10 grams",
        ),
        MCXContractSpec(
            commodity="SILVER",
            instrument="FUTCOM",
            lot_size=30.0,
            tick_size=1.0,
            quote_unit="INR per kilogram",
        ),
    ])
    enriched = contract_metadata_for_frame(frame, master)
    assert enriched["lot_size"].tolist() == [100.0, 30.0]
    assert enriched["tick_size"].tolist() == [1.0, 1.0]


def test_invalid_contract_metadata_is_rejected():
    with pytest.raises(MCXContractMetadataError, match="positive"):
        MCXContractSpec(
            commodity="GOLD", instrument="FUTCOM", lot_size=0,
            tick_size=1, quote_unit="INR"
        ).validate()
