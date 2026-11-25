"""Point-in-time security master for historical NSE universe construction.

Price files alone cannot reconstruct an investable historical universe. This
module deliberately requires an external, authoritative security-master file
for validation runs; it does not infer listing or delisting dates from cache
coverage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional


class SecurityMasterError(ValueError):
    """Raised when security-master identity or date intervals are invalid."""


def _parse_date(value: object, *, field: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise SecurityMasterError(f"{field} is required")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise SecurityMasterError(f"Invalid {field}: {text!r}") from exc


@dataclass(frozen=True)
class SecurityRecord:
    """One canonical security identity and its investable interval."""

    security_id: str
    canonical_symbol: str
    listed_from: date
    delisted_on: Optional[date] = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.security_id.strip():
            raise SecurityMasterError("security_id is required")
        if not self.canonical_symbol.strip():
            raise SecurityMasterError("canonical_symbol is required")
        if self.delisted_on is not None and self.delisted_on < self.listed_from:
            raise SecurityMasterError(
                f"{self.security_id}: delisted_on precedes listed_from"
            )

    def is_active(self, as_of: date) -> bool:
        """Return whether the security was listed at the as-of date."""
        return self.listed_from <= as_of and (
            self.delisted_on is None or as_of <= self.delisted_on
        )


class PointInTimeSecurityMaster:
    """Validated canonical identities with date-aware membership queries."""

    def __init__(self, records: Iterable[SecurityRecord]):
        ordered = tuple(records)
        if not ordered:
            raise SecurityMasterError("Security master cannot be empty")

        by_id: dict[str, SecurityRecord] = {}
        by_symbol: dict[str, str] = {}
        for record in ordered:
            if record.security_id in by_id:
                raise SecurityMasterError(
                    f"Duplicate security_id: {record.security_id}"
                )
            by_id[record.security_id] = record
            names = (record.canonical_symbol, *record.aliases)
            for name in names:
                normalized = self.normalize_symbol(name)
                previous = by_symbol.get(normalized)
                if previous is not None and previous != record.security_id:
                    raise SecurityMasterError(
                        f"Symbol alias {name!r} maps to multiple securities: "
                        f"{previous}, {record.security_id}"
                    )
                by_symbol[normalized] = record.security_id

        self.records = tuple(sorted(ordered, key=lambda item: item.security_id))
        self._by_id = by_id
        self._by_symbol = by_symbol

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return str(symbol).strip().upper().removesuffix(".NS")

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "PointInTimeSecurityMaster":
        parsed = []
        for raw in records:
            aliases = raw.get("aliases", ())
            if isinstance(aliases, str):
                aliases = tuple(item.strip() for item in aliases.split(";") if item.strip())
            parsed.append(SecurityRecord(
                security_id=str(raw.get("security_id", "")).strip(),
                canonical_symbol=str(raw.get("canonical_symbol", "")).strip().upper(),
                listed_from=_parse_date(raw.get("listed_from"), field="listed_from"),
                delisted_on=(
                    _parse_date(raw["delisted_on"], field="delisted_on")
                    if raw.get("delisted_on") else None
                ),
                aliases=tuple(str(item).strip().upper() for item in aliases),
            ))
        return cls(parsed)

    @classmethod
    def from_file(cls, path: str | Path) -> "PointInTimeSecurityMaster":
        """Load JSON records or a CSV with the documented column names."""
        file_path = Path(path)
        if not file_path.exists():
            raise SecurityMasterError(f"Security-master file not found: {file_path}")
        try:
            if file_path.suffix.lower() == ".json":
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                records = payload.get("records", payload) if isinstance(payload, dict) else payload
            else:
                with file_path.open(newline="", encoding="utf-8") as handle:
                    records = list(csv.DictReader(handle))
            if not isinstance(records, list):
                raise SecurityMasterError("Security-master payload must contain records")
            return cls.from_records(records)
        except SecurityMasterError:
            raise
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise SecurityMasterError(f"Could not read security master: {file_path}") from exc

    def canonicalize(self, symbol: str) -> str:
        normalized = self.normalize_symbol(symbol)
        security_id = self._by_symbol.get(normalized)
        if security_id is None:
            raise SecurityMasterError(f"Unknown security symbol: {symbol}")
        return self._by_id[security_id].canonical_symbol

    def record_for(self, symbol: str) -> SecurityRecord:
        normalized = self.normalize_symbol(symbol)
        security_id = self._by_symbol.get(normalized)
        if security_id is None:
            raise SecurityMasterError(f"Unknown security symbol: {symbol}")
        return self._by_id[security_id]

    def symbols_at(self, as_of: date) -> tuple[str, ...]:
        """Return canonical symbols active on the exact as-of date."""
        return tuple(
            record.canonical_symbol
            for record in self.records
            if record.is_active(as_of)
        )

    def snapshot(self, as_of: date, available_symbols: Iterable[str] = ()) -> dict:
        """Return auditable active/missing/unrecognised membership metadata."""
        active = set(self.symbols_at(as_of))
        available = {self.canonicalize(symbol) for symbol in available_symbols}
        return {
            "as_of": as_of.isoformat(),
            "active_symbols": sorted(active),
            "available_symbols": sorted(active & available),
            "missing_price_symbols": sorted(active - available),
            "security_count": len(active),
        }

    @property
    def sha256(self) -> str:
        payload = [asdict(record) for record in self.records]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
