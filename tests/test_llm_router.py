"""
Tests — Tiered LLM router failover (AGENTS.md rules 3 and 4).

Rate limits surface when a model is *called*, not when the client object is
built, so failover must happen inside FailoverLLM.call().  No network I/O:
`_build_llm` is replaced with fakes.
"""

from __future__ import annotations

import pytest

import app.llm.router as router
from app.llm.router import (
    AllProvidersFailedError,
    FailoverLLM,
    GroqKeyManager,
    ModelTier,
    _ProviderCooldowns,
)


class FakeRateLimit(Exception):
    status_code = 429


class FakeLLM:
    def __init__(self, model: str, behaviour):
        self.model = model
        self.behaviour = behaviour

    def call(self, messages, *args, **kwargs):
        outcome = self.behaviour(self.model)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def isolated_router(monkeypatch):
    """Fresh key/cooldown state, both cloud providers configured, no Ollama."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test_key")
    monkeypatch.setenv("ARTHA_DISABLE_OLLAMA", "1")
    monkeypatch.setattr(router, "_groq_keys", GroqKeyManager())
    monkeypatch.setattr(router, "_cooldowns", _ProviderCooldowns())
    sleeps: list[float] = []
    monkeypatch.setattr(router, "_sleep", sleeps.append)
    calls: list[str] = []

    def install(behaviour):
        def fake_build(attempt):
            calls.append(attempt.model)
            return FakeLLM(attempt.model, behaviour)

        monkeypatch.setattr(router, "_build_llm", fake_build)

    return install, calls, sleeps


def _chain_models(tier: ModelTier) -> list[str]:
    return [cfg["model"] for cfg in router._PROVIDER_CHAINS[tier]]


def test_rate_limit_at_call_time_fails_over_to_next_provider(isolated_router):
    install, calls, _ = isolated_router
    first, second = _chain_models(ModelTier.CHEAP)
    install(lambda model: FakeRateLimit("429 Too Many Requests") if model == first else "ok")

    llm = FailoverLLM(tier=ModelTier.CHEAP)
    assert llm.call([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == [first, second]
    assert llm.model == second


def test_rate_limited_provider_is_skipped_on_next_call(isolated_router):
    install, calls, _ = isolated_router
    first, second = _chain_models(ModelTier.CHEAP)
    install(lambda model: FakeRateLimit("rate limit") if model == first else "ok")

    FailoverLLM(tier=ModelTier.CHEAP).call("a")
    calls.clear()
    FailoverLLM(tier=ModelTier.CHEAP).call("b")
    assert calls == [second], "provider in cooldown must not be retried immediately"


def test_non_rate_limit_error_fails_over_without_cooldown(isolated_router):
    install, calls, _ = isolated_router
    first, second = _chain_models(ModelTier.CHEAP)
    install(lambda model: ValueError("bad request") if model == first else "ok")

    assert FailoverLLM(tier=ModelTier.CHEAP).call("x") == "ok"
    calls.clear()
    FailoverLLM(tier=ModelTier.CHEAP).call("y")
    assert calls[0] == first, "a non-429 error must not cool the provider down"


def test_all_providers_rate_limited_raises_without_long_wait(isolated_router):
    install, _, sleeps = isolated_router
    install(lambda model: FakeRateLimit("429"))

    llm = FailoverLLM(tier=ModelTier.DEEP, max_wait_seconds=0)
    with pytest.raises(AllProvidersFailedError) as excinfo:
        llm.call("x")
    assert len(excinfo.value.errors) == 2
    assert all(wait < 60 for wait in sleeps)


def test_auth_errors_do_not_trigger_retry_passes(isolated_router):
    install, calls, sleeps = isolated_router
    install(lambda model: PermissionError("invalid api key"))

    with pytest.raises(AllProvidersFailedError):
        FailoverLLM(tier=ModelTier.MID).call("x")
    assert len(calls) == 2 and sleeps == []


def test_no_providers_configured_raises(monkeypatch, isolated_router):
    install, calls, _ = isolated_router
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setattr(router, "_groq_keys", GroqKeyManager())
    install(lambda model: "unreachable")

    with pytest.raises(AllProvidersFailedError):
        FailoverLLM(tier=ModelTier.CHEAP).call("x")
    assert calls == []


def test_unreachable_ollama_is_not_counted_as_a_provider(monkeypatch, isolated_router):
    monkeypatch.delenv("ARTHA_DISABLE_OLLAMA")
    monkeypatch.setattr(router, "_ollama_probe", {"checked_at": 0.0, "ok": False})
    monkeypatch.setattr(router, "_OLLAMA_BASE_URL", "http://127.0.0.1:9")  # discard port
    models = [a.model for a in router.provider_attempts(ModelTier.CHEAP)]
    assert not any(m.startswith("ollama/") for m in models)


def test_retry_after_header_sets_cooldown(isolated_router):
    class WithRetryAfter(FakeRateLimit):
        class response:  # noqa: N801 - mimics httpx.Response shape
            status_code = 429
            headers = {"retry-after": "5"}

    install, _, _ = isolated_router
    first, _ = _chain_models(ModelTier.CHEAP)
    install(lambda model: WithRetryAfter("429") if model == first else "ok")
    FailoverLLM(tier=ModelTier.CHEAP).call("x")
    remaining = router._provider_cooldown_remaining(
        router._PROVIDER_CHAINS[ModelTier.CHEAP][0]["provider"]
    )
    assert remaining is not None and remaining <= 5


def test_only_one_groq_key_is_used(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key_a,key_b,key_c")
    manager = GroqKeyManager()
    assert manager.key_count == 1
    manager.mark_rate_limited("key_a")
    assert manager.get_current_key() is None, "no rotation to another key of the same org"


def test_retired_groq_models_are_not_configured():
    retired = {"groq/llama3-8b-8192", "groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile"}
    configured = {cfg["model"] for chain in router._PROVIDER_CHAINS.values() for cfg in chain}
    assert not configured & retired


@pytest.mark.parametrize(
    "exc, expected",
    [
        (FakeRateLimit("x"), True),
        (Exception("Error code: 429 - quota"), True),
        (Exception("RESOURCE_EXHAUSTED: per-minute quota"), True),
        (Exception("invalid api key"), False),
        (TimeoutError("read timeout"), False),
    ],
)
def test_is_rate_limit_error(exc, expected):
    assert router.is_rate_limit_error(exc) is expected


@pytest.mark.parametrize(
    "message, expected",
    [
        ("429 quota exceeded ... Please retry in 20h14m29.112813279s.", 20 * 3600 + 14 * 60 + 29.112813279),
        ('{"error": {"details": [{"retryDelay": "73s"}]}}', 73.0),
        ("Rate limit reached. Please try again in 1m30.5s.", 90.5),
        ("Please retry in 50h", 24 * 3600.0),  # capped
        ("rate limited", None),
    ],
)
def test_retry_hint_parsed_from_provider_message(message, expected):
    got = router._retry_after_seconds(Exception(message))
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_daily_quota_429_cools_provider_for_hours(isolated_router):
    install, calls, _ = isolated_router
    first, second = _chain_models(ModelTier.CHEAP)
    install(
        lambda model: FakeRateLimit("Generate content requests per day exceeded. Please retry in 20h14m29s")
        if model == first else "ok"
    )
    FailoverLLM(tier=ModelTier.CHEAP).call("x")
    provider = router._PROVIDER_CHAINS[ModelTier.CHEAP][0]["provider"]
    assert router._provider_cooldown_remaining(provider) > 20 * 3600
