"""Point-in-time fundamentals for leakage-safe historical research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable, Optional


class FundamentalHistoryError(ValueError):
    """Raised when as-reported fundamental observations are ambiguous."""


def _date(value: object, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise FundamentalHistoryError(f"Invalid {field}: {value!r}") from exc


@dataclass(frozen=True)
class FundamentalObservation:
    """A value as it was available to an investor on ``available_at``."""

    symbol: str
    metric: str
    value: float
    period_end: date
    available_at: date
    source: str
    restated: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.metric.strip() or not self.source.strip():
            raise FundamentalHistoryError("symbol, metric, and source are required")


class PointInTimeFundamentals:
    """As-of lookup that never exposes a future filing or later restatement."""

    def __init__(self, observations: Iterable[FundamentalObservation]):
        ordered = tuple(observations)
        if not ordered:
            raise FundamentalHistoryError("Fundamental history cannot be empty")
        seen: set[tuple[str, str, date, date]] = set()
        for observation in ordered:
            key = (
                observation.symbol.upper(), observation.metric,
                observation.period_end, observation.available_at,
            )
            if key in seen:
                raise FundamentalHistoryError(
                    f"Duplicate observation for {observation.symbol}/{observation.metric}"
                )
            seen.add(key)
        self.observations = tuple(sorted(
            ordered,
            key=lambda item: (item.symbol, item.metric, item.available_at, item.period_end),
        ))

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "PointInTimeFundamentals":
        observations = []
        for raw in records:
            observations.append(FundamentalObservation(
                symbol=str(raw.get("symbol", "")).strip().upper(),
                metric=str(raw.get("metric", "")).strip(),
                value=float(raw["value"]),
                period_end=_date(raw.get("period_end"), "period_end"),
                available_at=_date(raw.get("available_at"), "available_at"),
                source=str(raw.get("source", "")).strip(),
                restated=bool(raw.get("restated", False)),
            ))
        return cls(observations)

    def as_of(self, as_of: date) -> dict[str, dict[str, float]]:
        """Return latest as-reported values available on ``as_of``."""
        selected: dict[tuple[str, str], FundamentalObservation] = {}
        for observation in self.observations:
            if observation.available_at > as_of:
                continue
            key = (observation.symbol, observation.metric)
            previous = selected.get(key)
            if previous is None or (
                observation.available_at,
                observation.period_end,
            ) > (previous.available_at, previous.period_end):
                selected[key] = observation
        result: dict[str, dict[str, float]] = {}
        for (symbol, metric), observation in selected.items():
            result.setdefault(symbol, {})[metric] = observation.value
        return result

    def source_for(self, symbol: str, metric: str, as_of: date) -> Optional[str]:
        """Return the source of the exact value selected by ``as_of``."""
        values = self.as_of(as_of)
        if symbol.upper() not in values or metric not in values[symbol.upper()]:
            return None
        candidates = [
            item for item in self.observations
            if item.symbol == symbol.upper()
            and item.metric == metric
            and item.available_at <= as_of
        ]
        return max(candidates, key=lambda item: (item.available_at, item.period_end)).source

    @property
    def sha256(self) -> str:
        payload = [asdict(item) for item in self.observations]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
