"""
Who pays for an LLM call: the visitor's own key, or the owner's (rate-capped).

Bring-your-own-key (BYOK)
    A visitor can send their own provider keys with an LLM request in the
    ``X-Groq-Api-Key`` / ``X-Gemini-Api-Key`` headers.  They are used for that
    request only: never written to disk, a database or a log, never returned
    in a response, and never mixed with the owner's keys (a BYOK request does
    not fall back to the owner's quota).

Owner keys (``LLM_OWNER_KEYS``)
    ``unlimited`` — owner's keys for everyone (local development default).
    ``demo``      — owner's keys, capped per visitor and per day
                    (``DEMO_JOBS_PER_VISITOR`` / ``DEMO_JOBS_PER_DAY``); the
                    production default, for a showcase.
    ``off``       — owner's keys are never used for visitors; everyone must
                    bring a key.  Going fully public = set this (or simply
                    remove GROQ_API_KEY / GEMINI_API_KEY from the host).
"""

from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Mapping, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

HEADER = {"groq": "x-groq-api-key", "gemini": "x-gemini-api-key"}
_FORMAT = {
    "groq": re.compile(r"^gsk_[A-Za-z0-9]{20,120}$"),
    "gemini": re.compile(r"^AIza[0-9A-Za-z_\-]{30,60}$"),
}


class KeyFormatError(ValueError):
    """A supplied key is malformed (message never contains the key)."""


@dataclass(frozen=True)
class UserKeys:
    groq: Optional[str] = None
    gemini: Optional[str] = None

    def get(self, provider: str) -> Optional[str]:
        return getattr(self, provider, None)

    @property
    def any(self) -> bool:
        return bool(self.groq or self.gemini)

    @property
    def providers(self) -> list[str]:
        return [p for p in HEADER if self.get(p)]

    def __repr__(self) -> str:  # keys must never reach logs via repr()
        return f"UserKeys(providers={self.providers})"

    __str__ = __repr__


def keys_from_headers(headers: Mapping[str, str]) -> UserKeys:
    found: dict[str, str] = {}
    for provider, header in HEADER.items():
        raw = (headers.get(header) or "").strip()
        if not raw:
            continue
        if not _FORMAT[provider].match(raw):
            raise KeyFormatError(f"The {provider} key is not in the expected format")
        found[provider] = raw
    return UserKeys(**found)


_current: ContextVar[Optional[UserKeys]] = ContextVar("artha_user_llm_keys", default=None)


def current_keys() -> Optional[UserKeys]:
    return _current.get()


@contextmanager
def use_keys(keys: Optional[UserKeys]) -> Iterator[None]:
    """Make ``keys`` the LLM credentials for code run inside this block (this thread)."""
    token = _current.set(keys if keys and keys.any else None)
    try:
        yield
    finally:
        _current.reset(token)


def redact(text: str, keys: Optional[UserKeys] = None) -> str:
    """Remove any provider key (the visitor's, the owner's, or key-shaped strings) from text."""
    if not text:
        return text
    secrets = [k for k in ((keys.groq, keys.gemini) if keys else ()) if k]
    secrets += [v for v in (os.getenv("GROQ_API_KEY", ""), os.getenv("GEMINI_API_KEY", "")) if v]
    for s in secrets:
        text = text.replace(s, "***")
    text = re.sub(r"gsk_[A-Za-z0-9]{8,}", "gsk_***", text)
    return re.sub(r"AIza[0-9A-Za-z_\-]{8,}", "AIza***", text)


# ── Owner-key policy and demo quota ───────────────────────────────────────────

def owner_keys_mode() -> str:
    default = "demo" if os.getenv("APP_ENV", "development") == "production" else "unlimited"
    mode = os.getenv("LLM_OWNER_KEYS", default).strip().lower()
    return mode if mode in ("unlimited", "demo", "off") else "off"


def owner_keys_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


class DemoQuota:
    """
    Daily (IST) caps on owner-paid LLM jobs: per visitor and in total.

    In memory on purpose (no visitor data is stored anywhere).  A restart
    resets the counters, so the true worst case is ``DEMO_JOBS_PER_DAY`` jobs
    per process start; keep it low on hosts that restart often.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: Optional[str] = None
        self._by_visitor: dict[str, int] = {}
        self._total = 0

    @staticmethod
    def limits() -> tuple[int, int]:
        return _int_env("DEMO_JOBS_PER_VISITOR", 3), _int_env("DEMO_JOBS_PER_DAY", 25)

    def _roll(self) -> None:
        today = datetime.now(IST).date().isoformat()
        if today != self._day:
            self._day, self._by_visitor, self._total = today, {}, 0

    def remaining(self, visitor: str) -> dict:
        per_visitor, per_day = self.limits()
        with self._lock:
            self._roll()
            return {
                "visitor_remaining": max(0, per_visitor - self._by_visitor.get(visitor, 0)),
                "day_remaining": max(0, per_day - self._total),
                "per_visitor": per_visitor, "per_day": per_day,
            }

    def consume(self, visitor: str) -> Optional[str]:
        """Take one job from the quota; returns a refusal reason, or None if allowed."""
        per_visitor, per_day = self.limits()
        with self._lock:
            self._roll()
            if self._total >= per_day:
                return "The shared demo quota for today is used up"
            if self._by_visitor.get(visitor, 0) >= per_visitor:
                return f"You have used your {per_visitor} free AI reports for today"
            self._by_visitor[visitor] = self._by_visitor.get(visitor, 0) + 1
            self._total += 1
            return None

    def reset(self) -> None:
        with self._lock:
            self._day, self._by_visitor, self._total = None, {}, 0


demo_quota = DemoQuota()


@dataclass(frozen=True)
class LLMAccess:
    mode: str                     # "byok" | "owner" | "demo"
    keys: Optional[UserKeys]


class LLMKeyRequired(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def resolve_access(user_keys: UserKeys, visitor: str, consume: bool = True) -> LLMAccess:
    """Decide whose keys pay for an LLM job, enforcing the owner-key policy."""
    if user_keys.any:
        return LLMAccess("byok", user_keys)
    mode = owner_keys_mode()
    if not owner_keys_configured() or mode == "off":
        raise LLMKeyRequired("This deployment needs your own Groq or Gemini API key for AI reports")
    if mode == "unlimited":
        return LLMAccess("owner", None)
    refusal = demo_quota.consume(visitor) if consume else None
    if refusal:
        raise LLMKeyRequired(f"{refusal}. Add your own free Groq or Gemini key in Settings to continue")
    return LLMAccess("demo", None)


def policy_status(visitor: str) -> dict:
    mode = owner_keys_mode() if owner_keys_configured() else "off"
    out = {"owner_keys": mode, "byok_supported": True, "byok_providers": list(HEADER)}
    if mode == "demo":
        out["demo"] = demo_quota.remaining(visitor)
    return out
