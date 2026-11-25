"""
Tiered LLM Router — Honest Provider Failover
================================================
Three tiers (cheap/mid/deep) with cross-provider failover that happens at
*call time*, where rate limits actually surface.

Design (AGENTS.md rules 3, 4):
  - Cheap: Orchestrator, Fundamentals, Quant Narrator, Sentiment.
  - Mid: Risk Agent.
  - Deep: Writer Agent — final synthesis, once per job.
  - One Groq key per account; no quota multiplication through key rotation.
  - On a real 429 the provider is put in cooldown (Retry-After if the
    provider sends one, otherwise a jittered default) and the call moves to
    the next provider in the tier's chain, then to a local Ollama server if
    one is actually reachable.
  - If every provider is cooling down, the call waits for the soonest
    cooldown to expire (bounded by LLM_MAX_WAIT_SECONDS) and makes one more
    pass; otherwise it raises AllProvidersFailedError so the caller can use
    its deterministic fallback.

Why the wrapper exists: constructing a crewai/litellm LLM object performs no
network I/O, so catching 429s at construction (the previous design) could
never trigger failover.  `FailoverLLM.call()` owns the retry loop instead.

Model IDs are overridable per tier via env vars because hosted model
catalogues change (Groq retired llama3-8b-8192 on 2025-08-30 and
llama-3.1-8b-instant / llama-3.3-70b-versatile on 2026-08-16; see
https://console.groq.com/docs/deprecations).

Reference: docs/ARTHA_ARCHITECTURE.md §4.6
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from app.llm.keys import UserKeys, current_keys, owner_keys_mode, redact

load_dotenv()

# litellm's Gemini provider reads GOOGLE_API_KEY or GEMINI_API_KEY.
_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
if _gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = _gemini_key

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model cost/capability tiers."""

    CHEAP = "cheap"
    MID = "mid"
    DEEP = "deep"


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in a tier's chain failed for one call."""

    def __init__(self, tier: "ModelTier", errors: list[tuple[str, Exception]]):
        self.tier = tier
        self.errors = errors
        detail = "; ".join(f"{model}: {type(e).__name__}: {e}" for model, e in errors)
        super().__init__(
            f"No LLM provider succeeded for tier '{tier.value}'"
            + (f" ({detail})" if detail else " (no provider configured or all cooling down)")
        )


# ── Cooldown bookkeeping ───────────────────────────────────────────────────────

DEFAULT_COOLDOWN_SECONDS = 65.0  # per-minute quotas reset after ~60s


def _cooldown_with_jitter(base: float) -> float:
    """±20% jitter so concurrent workers don't all retry in the same second."""
    return base * random.uniform(0.8, 1.2)


class GroqKeyManager:
    """
    Holds the single Groq account key and its cooldown.

    A comma-separated value is accepted for backwards compatibility, but only
    the first key is used: keys from the same Groq organization share one
    quota, so rotating them is not a rate-limit workaround (AGENTS.md rule 4).
    Thread-safe.
    """

    KEY_COOLDOWN_SECONDS = DEFAULT_COOLDOWN_SECONDS

    def __init__(self) -> None:
        raw = os.getenv("GROQ_API_KEY", "").strip()
        configured_keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._keys = configured_keys[:1]
        self._current_index = 0
        self._cooldowns: dict[int, datetime] = {}
        self._lock = threading.Lock()

        if self._keys:
            if len(configured_keys) > 1:
                logger.warning(
                    "Multiple GROQ_API_KEY values configured; using one key only "
                    "because provider quotas are organization-scoped"
                )
            logger.info("Groq provider configured with one account key")
        else:
            logger.warning("GroqKeyManager: no GROQ_API_KEY found")

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_current_key(self) -> Optional[str]:
        """Return the key, or None if it is missing or cooling down."""
        if not self._keys:
            return None
        with self._lock:
            until = self._cooldowns.get(0)
            if until is None or datetime.now() >= until:
                return self._keys[0]
            return None

    def cooldown_remaining(self) -> Optional[float]:
        """Seconds until the key is usable again (None if not cooling)."""
        with self._lock:
            until = self._cooldowns.get(0)
        if until is None:
            return None
        remaining = (until - datetime.now()).total_seconds()
        return remaining if remaining > 0 else None

    def mark_rate_limited(self, key: Optional[str] = None, seconds: Optional[float] = None) -> Optional[str]:
        """Put the Groq key in cooldown. Returns the key if still usable (never, now)."""
        if not self._keys:
            return None
        if key is not None and key not in self._keys:
            return self.get_current_key()
        duration = seconds if seconds is not None else _cooldown_with_jitter(self.KEY_COOLDOWN_SECONDS)
        with self._lock:
            self._cooldowns[0] = datetime.now() + timedelta(seconds=duration)
        logger.info("Groq provider hit rate limit → cooling down for %.0fs", duration)
        return self.get_current_key()

    def get_all_available_keys(self) -> list[str]:
        key = self.get_current_key()
        return [key] if key else []


class _ProviderCooldowns:
    """Cooldowns for non-Groq providers (Gemini, Ollama, ...)."""

    def __init__(self) -> None:
        self._until: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def cool(self, provider: str, seconds: float) -> None:
        with self._lock:
            self._until[provider] = datetime.now() + timedelta(seconds=seconds)

    def remaining(self, provider: str) -> Optional[float]:
        with self._lock:
            until = self._until.get(provider)
        if until is None:
            return None
        rem = (until - datetime.now()).total_seconds()
        return rem if rem > 0 else None

    def reset(self) -> None:
        with self._lock:
            self._until.clear()


_groq_keys = GroqKeyManager()
_cooldowns = _ProviderCooldowns()


# ── Provider configuration ──────────────────────────────────────────────────────
# Order within a tier = failover order.  Model IDs are env-overridable.

_GEMINI_FLASH = os.getenv("ARTHA_GEMINI_MODEL", "gemini/gemini-3.6-flash")
_GROQ_SMALL = os.getenv("ARTHA_GROQ_SMALL_MODEL", "groq/openai/gpt-oss-20b")
_GROQ_LARGE = os.getenv("ARTHA_GROQ_LARGE_MODEL", "groq/openai/gpt-oss-120b")

# gpt-oss models on Groq are reasoning models: hidden reasoning tokens count
# against max_tokens, so with default effort the answer can be cut off
# mid-sentence (observed live: a writer report ended at "- FY 2025: **16").
# Raising max_tokens is not an option on the free tier — Groq counts the
# requested completion budget against an 8k tokens-per-minute limit (observed:
# "Limit 8000, Requested 11349").  Narration needs little reasoning, so
# reasoning effort is set low instead.
_GPT_OSS_EXTRA = {"reasoning_effort": "low"}
_PROVIDER_CHAINS: dict[ModelTier, list[dict]] = {
    ModelTier.CHEAP: [
        {"model": _GEMINI_FLASH, "provider": "gemini", "max_tokens": 2048, "temperature": 0.1},
        {"model": _GROQ_SMALL, "provider": "groq", "max_tokens": 4096, "temperature": 0.1,
         "extra": _GPT_OSS_EXTRA},
    ],
    ModelTier.MID: [
        # Groq first so a cheap-tier Gemini 429 doesn't immediately hit the
        # same Gemini quota again when callers escalate cheap → mid.
        {"model": _GROQ_LARGE, "provider": "groq", "max_tokens": 4096, "temperature": 0.1,
         "extra": _GPT_OSS_EXTRA},
        {"model": _GEMINI_FLASH, "provider": "gemini", "max_tokens": 4096, "temperature": 0.1},
    ],
    ModelTier.DEEP: [
        {"model": _GROQ_LARGE, "provider": "groq", "max_tokens": 4096, "temperature": 0.2,
         "extra": _GPT_OSS_EXTRA},
        {"model": _GEMINI_FLASH, "provider": "gemini", "max_tokens": 8192, "temperature": 0.2},
    ],
}

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.1:8b")
_OLLAMA_CONFIG = {
    "model": _OLLAMA_MODEL,
    "provider": "ollama",
    "max_tokens": 2048,
    "temperature": 0.1,
    "base_url": _OLLAMA_BASE_URL,
}

# ── Retry configuration ─────────────────────────────────────────────────────────

_MAX_PASSES = 2
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
_REQUEST_TIMEOUT_S = 180


def _max_wait_seconds() -> float:
    try:
        return float(os.getenv("LLM_MAX_WAIT_SECONDS", "70"))
    except ValueError:
        return 70.0


def _jittered_backoff(attempt: int) -> float:
    """Exponential backoff with full jitter (AWS-recommended pattern)."""
    cap = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
    return random.uniform(0, cap)


# Indirection so tests can avoid real sleeping.
_sleep: Callable[[float], None] = time.sleep


# ── Error classification ───────────────────────────────────────────────────────

def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` is a provider rate-limit / quota (HTTP 429) error."""
    try:
        import litellm

        if isinstance(exc, litellm.RateLimitError):
            return True
    except Exception:  # pragma: no cover - litellm is a hard dependency
        pass
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "ratelimit", "rate_limit", "resource_exhausted", "too many requests")
    )


_MAX_COOLDOWN_S = 24 * 3600
# Providers put the real wait in the error text, e.g.
#   Gemini: "Please retry in 20h14m29.11s"  /  "retryDelay": "73s"
#   Groq:   "Please try again in 1m30.5s"
# A daily-quota 429 must cool the provider for hours, not 65 seconds —
# otherwise every call re-hits an exhausted quota (observed live with the
# Gemini free tier's 20 requests/day limit).
_RETRY_HINT_RE = re.compile(
    r"(?:retry|try again)\s+in\s+((?:\d+(?:\.\d+)?\s*[hms]\s*)+)"
    r"|retryDelay\W+(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])", re.IGNORECASE)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Provider-stated wait: Retry-After header, else a hint in the message."""
    seconds: Optional[float] = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is not None:
                seconds = float(value)
        except (TypeError, ValueError, AttributeError):
            seconds = None
    if seconds is None:
        match = _RETRY_HINT_RE.search(str(exc))
        if match:
            if match.group(1):
                unit = {"h": 3600.0, "m": 60.0, "s": 1.0}
                seconds = sum(
                    float(n) * unit[u.lower()] for n, u in _DURATION_PART_RE.findall(match.group(1))
                )
            else:
                seconds = float(match.group(2))
    if seconds is None or seconds <= 0:
        return None
    return min(seconds, _MAX_COOLDOWN_S)


# ── Provider availability ──────────────────────────────────────────────────────

def _get_api_key(provider: str) -> Optional[str]:
    """Resolve an API key for a provider (None if missing or cooling down)."""
    if provider == "groq":
        return _groq_keys.get_current_key()
    if provider == "gemini":
        if _cooldowns.remaining("gemini"):
            return None
        key = os.getenv("GEMINI_API_KEY", "").strip()
        return key or None
    return None


_ollama_probe_lock = threading.Lock()
_ollama_probe: dict[str, Any] = {"checked_at": 0.0, "ok": False}
_OLLAMA_PROBE_TTL_S = 30.0


def _ollama_reachable() -> bool:
    """Cheap cached probe; an unreachable Ollama must not count as a provider."""
    if os.getenv("ARTHA_DISABLE_OLLAMA", "").lower() in ("1", "true", "yes"):
        return False
    now = time.monotonic()
    with _ollama_probe_lock:
        if now - _ollama_probe["checked_at"] < _OLLAMA_PROBE_TTL_S:
            return bool(_ollama_probe["ok"])
    ok = False
    try:
        import httpx

        ok = httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=0.5).status_code == 200
    except Exception:
        ok = False
    with _ollama_probe_lock:
        _ollama_probe.update(checked_at=now, ok=ok)
    return ok


def mark_provider_rate_limited(
    provider: str,
    api_key: Optional[str] = None,
    seconds: Optional[float] = None,
) -> None:
    """Put a provider in cooldown after a real 429."""
    duration = seconds if seconds is not None else _cooldown_with_jitter(DEFAULT_COOLDOWN_SECONDS)
    if provider == "groq":
        _groq_keys.mark_rate_limited(api_key, seconds=duration)
    else:
        _cooldowns.cool(provider, duration)
        logger.info("%s provider hit rate limit → cooling down for %.0fs", provider, duration)


def _provider_cooldown_remaining(provider: str) -> Optional[float]:
    if provider == "groq":
        return _groq_keys.cooldown_remaining()
    return _cooldowns.remaining(provider)


def _provider_configured(provider: str) -> bool:
    if provider == "groq":
        return _groq_keys.key_count > 0
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY", "").strip())
    if provider == "ollama":
        return _ollama_reachable()
    return False


@dataclass(frozen=True)
class ProviderAttempt:
    """One provider the router may try right now, with its resolved key."""

    config: dict
    api_key: Optional[str] = field(repr=False)
    byok: bool = False

    @property
    def provider(self) -> str:
        return self.config["provider"]

    @property
    def model(self) -> str:
        return self.config["model"]


def provider_attempts(tier: ModelTier, user_keys: Optional[UserKeys] = None) -> list[ProviderAttempt]:
    """
    Providers in failover order that are configured and not cooling down.

    With visitor keys (BYOK) only those keys are used — never the owner's
    keys, the owner's cooldowns or the owner's local Ollama.  Without them,
    owner keys are used unless the owner-key policy is ``off``.
    """
    chain = _PROVIDER_CHAINS.get(tier, _PROVIDER_CHAINS[ModelTier.CHEAP])
    if user_keys is not None and user_keys.any:
        return [ProviderAttempt(cfg, user_keys.get(cfg["provider"]), byok=True)
                for cfg in chain if user_keys.get(cfg["provider"])]
    attempts: list[ProviderAttempt] = []
    if owner_keys_mode() != "off":
        for cfg in chain:
            key = _get_api_key(cfg["provider"])
            if key:
                attempts.append(ProviderAttempt(cfg, key))
    if not _cooldowns.remaining("ollama") and _ollama_reachable():
        attempts.append(ProviderAttempt(_OLLAMA_CONFIG, None))
    return attempts


def _soonest_cooldown(tier: ModelTier) -> Optional[float]:
    """Seconds until the earliest configured provider in `tier` frees up."""
    providers = {cfg["provider"] for cfg in _PROVIDER_CHAINS.get(tier, [])} | {"ollama"}
    remaining = [
        rem
        for p in providers
        if _provider_configured(p) and (rem := _provider_cooldown_remaining(p)) is not None
    ]
    return min(remaining) if remaining else None


def _build_llm(attempt: ProviderAttempt):
    """Construct a crewai LLM for one attempt. No network I/O happens here."""
    from crewai import LLM

    cfg = attempt.config
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        # litellm's own retries would sit on a 429 instead of failing over.
        "num_retries": 0,
        "timeout": _REQUEST_TIMEOUT_S,
    }
    if attempt.api_key:
        kwargs["api_key"] = attempt.api_key
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    # Only apply model-specific extras to the model they were written for
    # (an env override to a non-reasoning model must not get reasoning_effort).
    if "gpt-oss" in cfg["model"]:
        kwargs.update(cfg.get("extra", {}))
    return LLM(**kwargs)


# ── Public LLM wrapper ─────────────────────────────────────────────────────────

@dataclass
class FailoverLLM:
    """
    Tier-bound LLM handle whose `.call()` fails over across providers.

    Exposes `model`, `temperature`, `max_tokens` (of the primary provider) so
    code that only inspects those attributes keeps working.
    """

    tier: ModelTier
    max_passes: int = _MAX_PASSES
    max_wait_seconds: Optional[float] = None
    # Visitor keys captured when the handle is created (inside the job), so
    # they still apply if a framework calls the LLM from another thread.
    user_keys: Optional[UserKeys] = field(default=None, repr=False)
    last_model: Optional[str] = field(default=None, init=False)

    @property
    def _primary(self) -> dict:
        return _PROVIDER_CHAINS.get(self.tier, _PROVIDER_CHAINS[ModelTier.CHEAP])[0]

    @property
    def model(self) -> str:
        return self.last_model or self._primary["model"]

    @property
    def temperature(self) -> float:
        return self._primary["temperature"]

    @property
    def max_tokens(self) -> int:
        return self._primary["max_tokens"]

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the first healthy provider; fail over on errors."""
        max_wait = self.max_wait_seconds if self.max_wait_seconds is not None else _max_wait_seconds()
        errors: list[tuple[str, Exception]] = []

        for pass_idx in range(max(1, self.max_passes)):
            attempts = provider_attempts(self.tier, self.user_keys)
            if not attempts:
                if self.user_keys is not None and self.user_keys.any:
                    break   # a visitor's own key never waits on the owner's cooldowns
                wait = _soonest_cooldown(self.tier)
                last_pass = pass_idx == self.max_passes - 1
                if wait is None or wait > max_wait or last_pass:
                    break
                logger.info("All providers for tier=%s cooling down; waiting %.0fs", self.tier.value, wait)
                _sleep(wait + random.uniform(0, 1.0))
                continue

            saw_rate_limit = False
            for attempt in attempts:
                try:
                    llm = _build_llm(attempt)
                    result = llm.call(messages, *args, **kwargs)
                    self.last_model = attempt.model
                    logger.info("LLM tier=%s answered by %s", self.tier.value, attempt.model)
                    return result
                except Exception as exc:  # noqa: BLE001 - classify, then fail over
                    text = str(exc)
                    clean = redact(text, self.user_keys)
                    # Provider errors can echo the key; keep the original exception only if it doesn't.
                    safe = exc if clean == text else RuntimeError(f"{type(exc).__name__}: {clean}")
                    errors.append((attempt.model, safe))
                    if is_rate_limit_error(exc):
                        saw_rate_limit = True
                        if not attempt.byok:   # a visitor's 429 must not cool the owner's account
                            mark_provider_rate_limited(
                                attempt.provider, attempt.api_key, _retry_after_seconds(exc)
                            )
                        logger.warning("Rate limited on %s; failing over", attempt.model)
                    else:
                        logger.warning("LLM %s failed (%s); failing over", attempt.model, safe)

            if not saw_rate_limit:
                # Auth/model/connection errors won't fix themselves by waiting.
                break
            if pass_idx < self.max_passes - 1:
                _sleep(_jittered_backoff(pass_idx))

        raise AllProvidersFailedError(self.tier, errors)


def get_llm(tier: ModelTier = ModelTier.CHEAP) -> FailoverLLM:
    """Return a failover-capable LLM handle for `tier` (no network I/O)."""
    return FailoverLLM(tier=tier, user_keys=current_keys())


def mark_key_exhausted(api_key: str) -> None:
    """Backwards-compatible: cool down the Groq provider after a runtime 429."""
    mark_provider_rate_limited("groq", api_key)


def get_key_status() -> dict:
    """Provider cooldown status for debugging/monitoring (no secrets)."""
    statuses = {}
    for provider in ("groq", "gemini", "ollama"):
        rem = _provider_cooldown_remaining(provider)
        statuses[provider] = {
            "configured": _provider_configured(provider),
            "status": "cooling_down" if rem else "available",
            **({"available_in_seconds": round(rem)} if rem else {}),
        }
    return {"providers": statuses}


# ── Convenience functions ──────────────────────────────────────────────────────

def get_cheap_llm() -> FailoverLLM:
    """Cheap tier (Orchestrator, Fundamentals, Quant Narrator, Sentiment)."""
    return get_llm(ModelTier.CHEAP)


def get_mid_llm() -> FailoverLLM:
    """Mid tier (Risk Agent)."""
    return get_llm(ModelTier.MID)


def get_deep_llm() -> FailoverLLM:
    """Deep tier (Writer Agent — final synthesis only)."""
    return get_llm(ModelTier.DEEP)
