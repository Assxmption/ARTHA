"""
MCX contract master and sizing metadata.

The exchange's contract master is deliberately data-driven.  This module does
not guess lot sizes, tick sizes, quote units, or expiry rules from a commodity
name.  A contract is eligible for sizing only when its metadata is present and
valid in the configured master file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CONTRACT_MASTER = Path("data_cache/mcx/contracts.json")


class MCXContractMetadataError(ValueError):
    """Raised when contract metadata is incomplete or invalid."""


@dataclass(frozen=True)
class MCXContractSpec:
    commodity: str
    instrument: str
    lot_size: float
    tick_size: float
    quote_unit: str
    currency: str = "INR"
    expiry_rule: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.commodity.upper()}|{self.instrument.upper()}"

    def validate(self) -> "MCXContractSpec":
        if not self.commodity.strip() or not self.instrument.strip():
            raise MCXContractMetadataError("commodity and instrument are required")
        if self.lot_size <= 0 or self.tick_size <= 0:
            raise MCXContractMetadataError("lot_size and tick_size must be positive")
        if not self.quote_unit.strip() or not self.currency.strip():
            raise MCXContractMetadataError("quote_unit and currency are required")
        return self

    def to_dict(self) -> dict:
        return asdict(self) | {"key": self.key}


class MCXContractMaster:
    """Read-only contract metadata registry loaded from a JSON master file."""

    def __init__(self, path: Optional[Path] = None, specs: Optional[list[MCXContractSpec]] = None):
        self.path = path or Path(os.getenv("MCX_CONTRACT_MASTER", DEFAULT_CONTRACT_MASTER))
        self._specs: dict[str, MCXContractSpec] = {}
        if specs is not None:
            for spec in specs:
                self.add(spec)
        elif self.path.exists():
            self.load()

    def add(self, spec: MCXContractSpec) -> None:
        spec.validate()
        self._specs[spec.key] = spec

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MCXContractMetadataError(
                f"Unable to load MCX contract master: {self.path}"
            ) from exc
        rows = payload.get("contracts", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise MCXContractMetadataError("MCX contract master must contain a contracts list")
        for row in rows:
            if not isinstance(row, dict):
                raise MCXContractMetadataError("MCX contract rows must be objects")
            self.add(MCXContractSpec(**{key: row[key] for key in (
                "commodity", "instrument", "lot_size", "tick_size", "quote_unit",
            )}, currency=row.get("currency", "INR"), expiry_rule=row.get("expiry_rule")))

    def get(self, commodity: str, instrument: str = "FUTCOM") -> Optional[MCXContractSpec]:
        return self._specs.get(f"{commodity.upper()}|{instrument.upper()}")

    def require(self, commodity: str, instrument: str = "FUTCOM") -> MCXContractSpec:
        spec = self.get(commodity, instrument)
        if spec is None:
            raise MCXContractMetadataError(
                f"No contract metadata for {commodity.upper()} {instrument.upper()}"
            )
        return spec

    def list(self, commodity: Optional[str] = None) -> list[MCXContractSpec]:
        values = list(self._specs.values())
        if commodity:
            values = [spec for spec in values if spec.commodity.upper() == commodity.upper()]
        return sorted(values, key=lambda spec: spec.key)

    def to_dict(self, commodity: Optional[str] = None) -> list[dict]:
        return [spec.to_dict() for spec in self.list(commodity)]


def contract_metadata_for_frame(frame, master: MCXContractMaster):
    """Attach sizing metadata to an MCX frame; fail closed when unavailable."""
    if frame.empty:
        return frame.copy()
    enriched = frame.copy()
    specs = []
    for row in enriched.itertuples(index=False):
        spec = master.require(row.commodity, row.instrument)
        specs.append(spec)
    enriched["lot_size"] = [spec.lot_size for spec in specs]
    enriched["tick_size"] = [spec.tick_size for spec in specs]
    enriched["quote_unit"] = [spec.quote_unit for spec in specs]
    enriched["currency"] = [spec.currency for spec in specs]
    return enriched
