"""
MCX End-of-Day Bhavcopy Ingestion
==================================

Normalises official MCX date-wise bhavcopy rows into a contract-aware table.
The HTTP transport is deliberately separated from parsing and persistence:
tests and future licensed providers can inject a provider without touching the
quant engine.

The public MCX Bhavcopy page exposes a date-wise report.  The current web
application backs that report with a JSON endpoint; its URL is configurable
because exchange web endpoints can change without notice.  No live broker or
order API is used here.

Output columns:
    trade_date, instrument, commodity, expiry_date, option_type, strike,
    open, high, low, close, previous_close, volume_lots, volume_thousands,
    value_lakhs, open_interest_lots, contract_key, source

Important data rules:
  - Contract identity includes commodity, expiry, instrument, option type, and
    strike.  Combining contracts by commodity alone would create false series.
  - Missing/holiday dates return an empty frame; no forward filling is done.
  - Rows with no usable close are retained only in the raw provider response,
    not in the quant-ready output.
  - Source and trade date are attached to every returned row.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

import pandas as pd

logger = logging.getLogger("artha.data.mcx")

MCX_BHAVCOPY_PAGE_URL = "https://www.mcxindia.com/market-data/bhavcopy"
MCX_BHAVCOPY_API_URL = (
    "https://www.mcxindia.com/market-data/bhavcopy/GetDateWiseBhavCopy"
)
_DEFAULT_CACHE_DIR = Path("data_cache/mcx")

MCX_COLUMNS = [
    "trade_date",
    "instrument",
    "commodity",
    "expiry_date",
    "option_type",
    "strike",
    "open",
    "high",
    "low",
    "close",
    "previous_close",
    "volume_lots",
    "volume_thousands",
    "value_lakhs",
    "open_interest_lots",
    "contract_key",
    "source",
]


class MCXProvider(Protocol):
    """Transport contract for one MCX bhavcopy date."""

    def fetch_bhavcopy(
        self,
        trade_date: date,
        instrument: str = "ALL",
    ) -> Any:
        """Return JSON, CSV bytes, a mapping, a dataframe, or an empty result."""


class MCXDataError(RuntimeError):
    """Raised when a non-empty MCX response cannot be normalised safely."""


def _empty_frame() -> pd.DataFrame:
    """Return an empty frame with the stable MCX schema."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in MCX_COLUMNS})


def _parse_date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none", "-"}:
        return None
    for fmt in ("%d/%m/%Y", "%d %b %Y", "%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _clean_number(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "null", "none", "-", "na", "n/a"}:
        return None
    # MCX volume fields may contain units such as "12.000 KGS".
    numeric = pd.to_numeric(text.split()[0], errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _field(row: Mapping[str, Any], *names: str) -> Any:
    """Case/spacing-insensitive lookup across MCX response versions."""
    normalized = {
        "".join(ch.lower() for ch in str(key) if ch.isalnum()): value
        for key, value in row.items()
    }
    for name in names:
        key = "".join(ch.lower() for ch in name if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def _rows_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    """Extract rows from common MCX JSON/CSV/provider response shapes."""
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return payload.to_dict(orient="records")
    if isinstance(payload, Mapping):
        for key in ("Data", "data", "rows", "Rows", "result", "Result"):
            if key in payload:
                return _rows_from_payload(payload[key])
        return [payload]
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        try:
            return _rows_from_payload(json.loads(raw.decode("utf-8-sig")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                frame = pd.read_csv(BytesIO(raw))
            except Exception as exc:
                raise MCXDataError("MCX response is neither JSON nor CSV") from exc
            return frame.to_dict(orient="records")
    if isinstance(payload, str):
        try:
            return _rows_from_payload(json.loads(payload))
        except json.JSONDecodeError:
            try:
                return pd.read_csv(StringIO(payload)).to_dict(orient="records")
            except Exception as exc:
                raise MCXDataError("MCX response is neither JSON nor CSV") from exc
    if isinstance(payload, Iterable):
        rows = list(payload)
        if not rows:
            return []
        if not all(isinstance(row, Mapping) for row in rows):
            raise MCXDataError("MCX provider returned non-mapping rows")
        return rows
    raise MCXDataError(f"Unsupported MCX provider response: {type(payload).__name__}")


def normalize_mcx_bhavcopy(
    payload: Any,
    trade_date: Optional[date] = None,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Normalise one raw MCX bhavcopy response into contract-aware rows."""
    rows = _rows_from_payload(payload)
    if not rows:
        return _empty_frame()

    records: list[dict[str, Any]] = []
    for row in rows:
        row_date = _parse_date(_field(row, "Date", "TradeDate", "Trade Date")) or trade_date
        if row_date is None:
            raise MCXDataError("MCX row has no trade date")

        instrument = str(_field(row, "Instrument", "InstrumentName") or "").strip().upper()
        commodity = str(_field(row, "Commodity", "CommodityName") or "").strip().upper()
        expiry = _parse_date(_field(row, "ExpiryDate", "Expiry Date", "ContractExpiry"))
        option_type = str(_field(row, "OptionType", "Option Type") or "").strip().upper()
        strike = _clean_number(_field(row, "StrikePrice", "Strike Price", "Strike"))
        close = _clean_number(_field(row, "Close", "ClosingPrice", "ClosePrice"))
        if not instrument or not commodity or close is None:
            # Headers, option metadata rows, and malformed records must not
            # become a false price observation.
            continue

        expiry_text = expiry.isoformat() if expiry else "-"
        strike_text = "" if strike is None else f"{strike:g}"
        contract_key = "|".join(
            (commodity, instrument, expiry_text, option_type or "-", strike_text)
        )
        records.append({
            "trade_date": row_date,
            "instrument": instrument,
            "commodity": commodity,
            "expiry_date": expiry,
            "option_type": option_type or None,
            "strike": strike,
            "open": _clean_number(_field(row, "Open", "OpeningPrice")),
            "high": _clean_number(_field(row, "High", "HighPrice")),
            "low": _clean_number(_field(row, "Low", "LowPrice")),
            "close": close,
            "previous_close": _clean_number(_field(row, "PCP", "PreviousClose", "PrevClose")),
            "volume_lots": _clean_number(_field(row, "VolLots", "VolumeLots", "Vol (Lots)", "Volume")),
            "volume_thousands": _clean_number(_field(row, "Volume000", "Volume (000s)", "VolumeThousands")),
            "value_lakhs": _clean_number(_field(row, "ValueLakhs", "Value (Lakhs)", "Value")),
            "open_interest_lots": _clean_number(_field(row, "OILots", "OI (Lots)", "OpenInterest")),
            "contract_key": contract_key,
            "source": source or f"MCX_BHAVCOPY_{row_date.isoformat()}",
        })

    result = pd.DataFrame(records, columns=MCX_COLUMNS)
    if result.empty:
        return _empty_frame()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date
    result["expiry_date"] = pd.to_datetime(result["expiry_date"], errors="coerce").dt.date
    return result.drop_duplicates(subset=["trade_date", "contract_key"], keep="last")


class HTTPMCXProvider:
    """Small official-endpoint transport with no parsing policy."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        session: Any = None,
        timeout: float = 20.0,
    ) -> None:
        self.endpoint = endpoint or os.getenv("MCX_BHAVCOPY_API_URL", MCX_BHAVCOPY_API_URL)
        self.session = session
        self.timeout = timeout

    def fetch_bhavcopy(self, trade_date: date, instrument: str = "ALL") -> Any:
        import requests

        session = self.session or requests.Session()
        response = session.get(
            self.endpoint,
            params={"InstrumentName": instrument, "fromDate": trade_date.strftime("%d/%m/%Y")},
            headers={
                "Accept": "application/json,text/plain,*/*",
                "Referer": MCX_BHAVCOPY_PAGE_URL,
                "User-Agent": "ARTHA/1.0 (research; contact configured operator)",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            return response.json()
        return response.content


def _cache_file(cache_dir: Path, trade_date: date) -> Path:
    return cache_dir / f"bhavcopy_{trade_date.isoformat()}.parquet"


def fetch_mcx_bhavcopy(
    trade_date: date,
    instrument: str = "ALL",
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    provider: Optional[MCXProvider] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch, normalise, and cache one MCX trading date."""
    if not isinstance(trade_date, date):
        raise TypeError("trade_date must be a datetime.date")
    if trade_date > date.today():
        raise ValueError("trade_date cannot be in the future")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file(cache_dir, trade_date)
    if not force_refresh and cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if set(MCX_COLUMNS).issubset(cached.columns):
            return cached[MCX_COLUMNS]

    active_provider = provider or HTTPMCXProvider()
    payload = active_provider.fetch_bhavcopy(trade_date, instrument)
    result = normalize_mcx_bhavcopy(
        payload,
        trade_date=trade_date,
        source=f"MCX_BHAVCOPY_{trade_date.isoformat()}",
    )
    if instrument != "ALL" and not result.empty:
        result = result[result["instrument"].eq(instrument.upper())].reset_index(drop=True)
    # Empty holiday frames are cached too, preventing repeated requests for a
    # known non-trading date.
    result.to_parquet(cache_file, index=False)
    return result[MCX_COLUMNS]


def fetch_mcx_history(
    start_date: date,
    end_date: Optional[date] = None,
    commodity: Optional[str] = None,
    instrument: str = "FUTCOM",
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    provider: Optional[MCXProvider] = None,
) -> pd.DataFrame:
    """Fetch trading-day rows without forward filling holidays or expiries."""
    end_date = end_date or date.today()
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    frames = []
    current = start_date
    while current <= end_date:
        # Weekends are known non-sessions; weekday exchange holidays remain
        # provider-resolved and are cached as empty frames.
        frame = _empty_frame() if current.weekday() >= 5 else fetch_mcx_bhavcopy(
            current,
            instrument=instrument,
            cache_dir=cache_dir,
            provider=provider,
        )
        if commodity and not frame.empty:
            frame = frame[frame["commodity"].eq(commodity.upper())]
        if not frame.empty:
            frames.append(frame)
        current += timedelta(days=1)

    if not frames:
        return _empty_frame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["trade_date", "contract_key"], keep="last")
        .sort_values(["trade_date", "contract_key"])
        .reset_index(drop=True)[MCX_COLUMNS]
    )
