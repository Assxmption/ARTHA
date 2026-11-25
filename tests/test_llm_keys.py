"""Bring-your-own-key and the owner-key demo quota (app/llm/keys.py)."""

from __future__ import annotations

import pytest

from app.llm import keys as k

GROQ = "gsk_" + "A" * 52
GEMINI = "AIza" + "B" * 35


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    k.demo_quota.reset()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "O" * 52)   # the owner's key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    k.demo_quota.reset()


def test_headers_parse_validate_and_never_repr_the_key():
    uk = k.keys_from_headers({"x-groq-api-key": GROQ, "x-gemini-api-key": GEMINI})
    assert uk.providers == ["groq", "gemini"]
    assert GROQ not in repr(uk) and GEMINI not in str(uk)
    assert not k.keys_from_headers({}).any
    with pytest.raises(k.KeyFormatError) as e:
        k.keys_from_headers({"x-groq-api-key": "sk-not-groq"})
    assert "sk-not-groq" not in str(e.value)


def test_redact_removes_visitor_owner_and_key_shaped_strings():
    uk = k.UserKeys(groq=GROQ)
    text = f"401 invalid key {GROQ}; owner {'gsk_' + 'O' * 52}; other AIzaSyXXXXXXXXXXXXXXXX"
    out = k.redact(text, uk)
    assert GROQ not in out and "O" * 52 not in out and "AIzaSyXXXX" not in out


@pytest.mark.parametrize("mode", ["unlimited", "demo", "off"])
def test_byok_always_allowed_and_never_spends_quota(monkeypatch, mode):
    monkeypatch.setenv("LLM_OWNER_KEYS", mode)
    access = k.resolve_access(k.UserKeys(groq=GROQ), "ip:1.2.3.4")
    assert access.mode == "byok" and access.keys.groq == GROQ
    assert k.demo_quota.remaining("ip:1.2.3.4")["visitor_remaining"] == k.demo_quota.limits()[0]


def test_off_mode_requires_a_key(monkeypatch):
    monkeypatch.setenv("LLM_OWNER_KEYS", "off")
    with pytest.raises(k.LLMKeyRequired):
        k.resolve_access(k.UserKeys(), "ip:1.2.3.4")


def test_no_owner_keys_means_byok_only(monkeypatch):
    monkeypatch.setenv("LLM_OWNER_KEYS", "unlimited")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(k.LLMKeyRequired):
        k.resolve_access(k.UserKeys(), "v")
    assert k.policy_status("v")["owner_keys"] == "off"


def test_demo_quota_per_visitor_and_per_day(monkeypatch):
    monkeypatch.setenv("LLM_OWNER_KEYS", "demo")
    monkeypatch.setenv("DEMO_JOBS_PER_VISITOR", "2")
    monkeypatch.setenv("DEMO_JOBS_PER_DAY", "3")
    assert k.resolve_access(k.UserKeys(), "a").mode == "demo"
    k.resolve_access(k.UserKeys(), "a")
    with pytest.raises(k.LLMKeyRequired) as e:
        k.resolve_access(k.UserKeys(), "a")
    assert "free AI reports" in e.value.reason
    k.resolve_access(k.UserKeys(), "b")
    with pytest.raises(k.LLMKeyRequired) as e:             # day total reached
        k.resolve_access(k.UserKeys(), "c")
    assert "shared demo quota" in e.value.reason
    assert k.policy_status("c")["demo"]["day_remaining"] == 0


def test_production_defaults_to_demo(monkeypatch):
    monkeypatch.delenv("LLM_OWNER_KEYS", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    assert k.owner_keys_mode() == "demo"
    monkeypatch.setenv("APP_ENV", "development")
    assert k.owner_keys_mode() == "unlimited"
    monkeypatch.setenv("LLM_OWNER_KEYS", "nonsense")
    assert k.owner_keys_mode() == "off"                     # unknown values fail closed


def test_router_uses_only_visitor_keys(monkeypatch):
    from app.llm import router

    monkeypatch.setenv("LLM_OWNER_KEYS", "unlimited")
    monkeypatch.setattr(router, "_ollama_reachable", lambda: True)
    attempts = router.provider_attempts(router.ModelTier.CHEAP, k.UserKeys(gemini=GEMINI))
    assert attempts and all(a.byok and a.provider == "gemini" and a.api_key == GEMINI for a in attempts)
    assert GEMINI not in repr(attempts[0])
    # Without visitor keys and with owner keys off: nothing but a local model.
    monkeypatch.setenv("LLM_OWNER_KEYS", "off")
    assert {a.provider for a in router.provider_attempts(router.ModelTier.CHEAP)} == {"ollama"}


def test_visitor_rate_limit_does_not_cool_owner(monkeypatch):
    from app.llm import router

    calls = []
    monkeypatch.setattr(router, "mark_provider_rate_limited", lambda *a, **kw: calls.append(a))

    class RateLimited(Exception):
        status_code = 429

    class FakeLLM:
        def call(self, *a, **kw):
            raise RateLimited(f"429 rate limit for key {GROQ}")

    monkeypatch.setattr(router, "_build_llm", lambda attempt: FakeLLM())
    monkeypatch.setattr(router, "is_rate_limit_error", lambda e: True)
    llm = router.FailoverLLM(tier=router.ModelTier.CHEAP, user_keys=k.UserKeys(groq=GROQ), max_passes=1)
    with pytest.raises(router.AllProvidersFailedError) as e:
        llm.call([{"role": "user", "content": "x"}])
    assert calls == []                                       # owner's cooldown untouched
    assert GROQ not in str(e.value)                          # key never echoed


def test_get_llm_captures_context_keys():
    from app.llm import router

    with k.use_keys(k.UserKeys(groq=GROQ)):
        llm = router.get_cheap_llm()
    assert llm.user_keys.groq == GROQ
    assert router.get_cheap_llm().user_keys is None


def test_analyze_endpoint_requires_key_when_owner_keys_off(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("LLM_OWNER_KEYS", "off")
    c = TestClient(app)
    r = c.post("/api/analyze", json={"symbol": "TCS"})
    assert r.status_code == 402 and r.json()["detail"]["code"] == "LLM_KEY_REQUIRED"
    r = c.post("/api/analyze", json={"symbol": "TCS"}, headers={"X-Groq-Api-Key": "bad"})
    assert r.status_code == 400
    assert c.get("/api/system/llm-policy").json()["owner_keys"] == "off"


def test_narrative_cache_hit_spends_nothing(monkeypatch):
    from fastapi.testclient import TestClient

    import app.api.quant_routes as qr
    from app.main import app

    monkeypatch.setenv("LLM_OWNER_KEYS", "off")
    key = qr._narrative_key("TCS")
    monkeypatch.setitem(qr._narrative_cache, key, (__import__("time").time(), "cached narrative"))
    c = TestClient(app)
    r = c.get("/api/quant/fundamentals/TCS/narrative_stream")
    assert r.status_code == 200 and r.text == "cached narrative"
    assert r.headers["x-narrative-cache"] == "hit"
    assert c.get("/api/quant/fundamentals/INFY/narrative_stream").status_code == 402
    assert c.get("/api/quant/fundamentals/..%2Fx/narrative_stream").status_code in (400, 404)
