"""
Tiered LLM Router — With Per-Key Groq Rotation
================================================
Three tiers (cheap/mid/deep) with real per-key rotation on Groq:
when key N hits a 429, rotate to key N+1 from the comma-separated
GROQ_API_KEY list. Only after ALL keys are exhausted does the
system failover to the next provider (Gemini → Ollama).

Design (AGENTS.md rules 3, 4):
  - Cheap (8B-class): Orchestrator, Fundamentals Agent, Quant Narrator,
    Sentiment Agent — high-quota, low-cost.
  - Mid: Risk Agent — slightly more capable for nuanced compliance checks.
  - Deep: Writer Agent — final synthesis, used once per job.
  - Per-key rotation: each Groq key is tried individually on 429.
  - Cross-provider failover: Groq → Gemini → local Ollama.
  - Jittered exponential backoff on real 429s.

Reference: docs/ARTHA_ARCHITECTURE.md §4.6
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure GOOGLE_API_KEY is set for litellm's Gemini provider
# litellm looks for GOOGLE_API_KEY or GEMINI_API_KEY
_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
if _gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = _gemini_key

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model cost/capability tiers."""

    CHEAP = "cheap"
    MID = "mid"
    DEEP = "deep"


# ── Groq Key Manager — per-key rotation on rate limit ───────────────────────────

class GroqKeyManager:
    """
    Manages multiple Groq API keys from a comma-separated env var.
    Rotates to the next key when the current one hits a rate limit.
    Tracks cooldown per key so exhausted keys get retried after a wait.
    
    Thread-safe — multiple agents can request keys concurrently.
    """

    # Cooldown after a key hits 429 (Groq resets per-minute)
    KEY_COOLDOWN_SECONDS = 65  # slightly over 1 minute to be safe

    def __init__(self):
        raw = os.getenv("GROQ_API_KEY", "").strip()
        self._keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._current_index = 0
        self._cooldowns: dict[int, datetime] = {}  # key_index → cooldown_until
        self._lock = threading.Lock()

        if self._keys:
            logger.info(
                "GroqKeyManager: loaded %d keys, rotating on 429",
                len(self._keys),
            )
        else:
            logger.warning("GroqKeyManager: no GROQ_API_KEY found")

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_current_key(self) -> Optional[str]:
        """Get the current active key (skipping cooled-down ones)."""
        if not self._keys:
            return None

        with self._lock:
            now = datetime.now()
            # Try each key starting from current index
            for offset in range(len(self._keys)):
                idx = (self._current_index + offset) % len(self._keys)
                cooldown_until = self._cooldowns.get(idx)

                if cooldown_until is None or now >= cooldown_until:
                    self._current_index = idx
                    return self._keys[idx]

            # All keys are in cooldown — return the one that expires soonest
            soonest_idx = min(
                self._cooldowns, key=lambda i: self._cooldowns[i]
            )
            wait_seconds = (self._cooldowns[soonest_idx] - now).total_seconds()
            logger.warning(
                "All %d Groq keys in cooldown. Soonest available in %.0fs (key %d)",
                len(self._keys), max(0, wait_seconds), soonest_idx + 1,
            )
            return None  # Signal to try next provider

    def mark_rate_limited(self, key: str) -> Optional[str]:
        """
        Mark a key as rate-limited and rotate to the next one.
        
        Returns the next available key, or None if all keys are cooling down.
        """
        with self._lock:
            # Find which index this key is
            try:
                idx = self._keys.index(key)
            except ValueError:
                return self.get_current_key()

            # Put this key on cooldown
            self._cooldowns[idx] = datetime.now() + timedelta(
                seconds=self.KEY_COOLDOWN_SECONDS
            )
            logger.info(
                "Groq key %d/%d hit rate limit → cooling down for %ds",
                idx + 1, len(self._keys), self.KEY_COOLDOWN_SECONDS,
            )

            # Move to next key
            self._current_index = (idx + 1) % len(self._keys)

        return self.get_current_key()

    def get_all_available_keys(self) -> list[str]:
        """Get all keys not currently in cooldown (for provider expansion)."""
        now = datetime.now()
        with self._lock:
            return [
                self._keys[i]
                for i in range(len(self._keys))
                if self._cooldowns.get(i) is None or now >= self._cooldowns[i]
            ]


# Singleton
_groq_keys = GroqKeyManager()


# ── Provider configuration ──────────────────────────────────────────────────────
# Each entry: (litellm_model_string, api_key_env_var, max_tokens_default)
# Order within a tier = failover order.

_PROVIDER_CHAINS: dict[ModelTier, list[dict]] = {
    ModelTier.CHEAP: [
        {
            "model": "groq/llama-3.1-8b-instant",
            "provider": "groq",
            "max_tokens": 2048,
            "temperature": 0.1,
        },
        {
            "model": "gemini/gemini-2.0-flash",
            "provider": "gemini",
            "max_tokens": 2048,
            "temperature": 0.1,
        },
    ],
    ModelTier.MID: [
        {
            "model": "gemini/gemini-2.0-flash",
            "provider": "gemini",
            "max_tokens": 4096,
            "temperature": 0.1,
        },
        {
            "model": "groq/llama-3.3-70b-versatile",
            "provider": "groq",
            "max_tokens": 2048,
            "temperature": 0.1,
        },
    ],
    ModelTier.DEEP: [
        {
            "model": "groq/llama-3.3-70b-versatile",
            "provider": "groq",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
        {
            "model": "gemini/gemini-2.0-flash",
            "provider": "gemini",
            "max_tokens": 8192,
            "temperature": 0.2,
        },
    ],
}

# ── Backoff configuration ───────────────────────────────────────────────────────

_MAX_RETRIES = 5
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0


def _jittered_backoff(attempt: int) -> float:
    """Exponential backoff with full jitter (AWS-recommended pattern)."""
    cap = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
    return random.uniform(0, cap)


# ── Key resolver ────────────────────────────────────────────────────────────────

def _get_api_key(provider: str) -> Optional[str]:
    """
    Resolve an API key for a provider.
    For Groq: uses the key manager (per-key rotation).
    For Gemini: reads GEMINI_API_KEY directly.
    """
    if provider == "groq":
        return _groq_keys.get_current_key()
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        return key if key else None
    return None


# ── Public API ──────────────────────────────────────────────────────────────────


def get_llm(tier: ModelTier = ModelTier.CHEAP):
    """
    Build and return a CrewAI-compatible LLM instance for the given tier.

    For Groq providers: tries each comma-separated key individually.
    When a key 429s, the GroqKeyManager rotates to the next key.
    Only after all Groq keys are exhausted does it failover to Gemini.

    Parameters
    ----------
    tier : ModelTier
        Which capability tier to use.

    Returns
    -------
    crewai.LLM
        A configured LLM instance.
    """
    from crewai import LLM

    chain = _PROVIDER_CHAINS.get(tier, _PROVIDER_CHAINS[ModelTier.CHEAP])

    for provider_config in chain:
        provider = provider_config["provider"]
        model = provider_config["model"]

        if provider == "groq":
            # Try each available Groq key
            available_keys = _groq_keys.get_all_available_keys()
            if not available_keys:
                logger.info(
                    "All Groq keys in cooldown for tier=%s, trying next provider",
                    tier.value,
                )
                continue

            for key in available_keys:
                try:
                    llm = LLM(
                        model=model,
                        api_key=key,
                        max_tokens=provider_config["max_tokens"],
                        temperature=provider_config["temperature"],
                        num_retries=2,  # Low retries per key — rotate instead
                        timeout=180,
                    )
                    key_idx = _groq_keys._keys.index(key) + 1
                    logger.info(
                        "LLM tier=%s using %s (Groq key %d/%d)",
                        tier.value, model, key_idx, _groq_keys.key_count,
                    )
                    return llm
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate" in err_str or "429" in err_str or "limit" in err_str:
                        _groq_keys.mark_rate_limited(key)
                        continue
                    logger.warning(
                        "Failed to construct LLM %s (key %d): %s",
                        model, _groq_keys._keys.index(key) + 1, e,
                    )
                    continue
        else:
            # Non-Groq provider (Gemini)
            api_key = _get_api_key(provider)
            if not api_key:
                logger.debug("No key for %s, skipping", provider)
                continue

            try:
                llm = LLM(
                    model=model,
                    api_key=api_key,
                    max_tokens=provider_config["max_tokens"],
                    temperature=provider_config["temperature"],
                    num_retries=_MAX_RETRIES,
                    timeout=180,
                )
                logger.info(
                    "LLM tier=%s using %s (Gemini)", tier.value, model
                )
                return llm
            except Exception as e:
                logger.warning(
                    "Failed to construct LLM %s for tier %s: %s",
                    model, tier.value, e,
                )
                continue

    # Last resort: try Ollama local if available
    try:
        llm = LLM(
            model="ollama/llama3.1:8b",
            base_url="http://localhost:11434",
            max_tokens=2048,
            temperature=0.1,
            num_retries=3,
            timeout=120,
        )
        logger.info("LLM tier=%s falling back to local Ollama", tier.value)
        return llm
    except Exception:
        pass

    raise RuntimeError(
        f"No LLM provider available for tier '{tier.value}'. "
        f"Set at least one of GROQ_API_KEY or GEMINI_API_KEY in .env."
    )


def mark_key_exhausted(api_key: str):
    """
    Call this when a runtime 429 occurs during an LLM call.
    Rotates to the next Groq key for subsequent calls.
    """
    _groq_keys.mark_rate_limited(api_key)


def get_key_status() -> dict:
    """Get current status of all Groq keys (for debugging/monitoring)."""
    now = datetime.now()
    statuses = []
    for i, key in enumerate(_groq_keys._keys):
        cooldown = _groq_keys._cooldowns.get(i)
        if cooldown and now < cooldown:
            remaining = (cooldown - now).total_seconds()
            statuses.append({
                "key_index": i + 1,
                "key_prefix": key[:8] + "…",
                "status": "cooling_down",
                "available_in_seconds": round(remaining),
            })
        else:
            statuses.append({
                "key_index": i + 1,
                "key_prefix": key[:8] + "…",
                "status": "available",
            })
    return {
        "total_keys": len(_groq_keys._keys),
        "current_index": _groq_keys._current_index + 1,
        "keys": statuses,
    }


# ── Convenience functions ──────────────────────────────────────────────────────

def get_cheap_llm():
    """Convenience: get a cheap-tier LLM (Orchestrator, Fundamentals, etc.)."""
    return get_llm(ModelTier.CHEAP)


def get_mid_llm():
    """Convenience: get a mid-tier LLM (Risk Agent)."""
    return get_llm(ModelTier.MID)


def get_deep_llm():
    """Convenience: get a deep-tier LLM (Writer Agent — final synthesis only)."""
    return get_llm(ModelTier.DEEP)
