"""Unit tests for internal-config fetch + effective_api_key / effective_base_url.

Covers:
1. fetch_internal_config — sends X-Internal-Token header, caches, returns {} on error
   and when internal token not configured.
2. effective_api_key — internal-config value beats env fallback; empty/absent → env.
3. effective_base_url — internal-config > public-config > None precedence.
4. build_effective_client — effective api_key / base_url reach openrouter/openai/anthropic
   provider clients correctly.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from agent.integrations.backend import (
    _invalidate_internal_config_cache,
    _invalidate_public_config_cache,
    fetch_internal_config,
)
from agent.integrations.dynamic_config import (
    effective_api_key,
    effective_base_url,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def _mock_httpx_response(data: dict[str, Any], status: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=data)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return mock_client


def _mock_httpx_error():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("network error"))
    return mock_client


# ── fetch_internal_config ─────────────────────────────────────────────────────


def test_fetch_internal_config_sends_internal_token_header(monkeypatch):
    """X-Internal-Token must be present in the request headers."""
    _invalidate_internal_config_cache()

    payload = {"ai_agent.api_key": "sk-test-key", "ai_agent.base_url": "https://example.com/v1"}

    captured_headers: list[dict] = []

    async def _fake_get(url, headers=None, **kw):
        captured_headers.append(headers or {})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=payload)
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = _fake_get

    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)

    # Provide internal token via settings override
    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "backend_internal_token", "my-secret-internal-token", raising=False)

    result = _run(fetch_internal_config())

    assert result == payload
    assert len(captured_headers) == 1
    assert captured_headers[0].get("X-Internal-Token") == "my-secret-internal-token"


def test_fetch_internal_config_returns_empty_when_no_token(monkeypatch):
    """When BACKEND_INTERNAL_TOKEN is not set, must return {} without making a request."""
    _invalidate_internal_config_cache()

    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "backend_internal_token", None, raising=False)

    call_count = [0]

    async def _fake_get(*a, **kw):
        call_count[0] += 1
        raise AssertionError("should not be called")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = _fake_get
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)

    result = _run(fetch_internal_config())
    assert result == {}
    assert call_count[0] == 0


def test_fetch_internal_config_returns_empty_on_network_error(monkeypatch):
    _invalidate_internal_config_cache()

    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "backend_internal_token", "token", raising=False)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: _mock_httpx_error())

    result = _run(fetch_internal_config())
    assert result == {}


def test_fetch_internal_config_caches_within_ttl(monkeypatch):
    """Second call within TTL must not hit the network again."""
    _invalidate_internal_config_cache()

    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "backend_internal_token", "token", raising=False)

    payload = {"ai_agent.api_key": "cached-key"}
    mock_client = _mock_httpx_response(payload)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)

    r1 = _run(fetch_internal_config())
    r2 = _run(fetch_internal_config())

    assert r1 == payload
    assert r2 == payload
    assert mock_client.get.call_count == 1


def test_fetch_internal_config_refetches_after_ttl(monkeypatch):
    """After TTL expires the cache is refreshed."""
    import agent.integrations.backend as backend_mod

    _invalidate_internal_config_cache()

    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "backend_internal_token", "token", raising=False)

    payload1 = {"ai_agent.api_key": "key-v1"}
    payload2 = {"ai_agent.api_key": "key-v2"}
    calls = [0]

    async def _fake_get(url, **kw):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=payload1 if calls[0] == 0 else payload2)
        calls[0] += 1
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = _fake_get
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)

    _run(fetch_internal_config())
    # Manually expire the cache
    backend_mod._internal_config_fetched_at = (
        time.monotonic() - backend_mod._INTERNAL_CONFIG_TTL - 1
    )
    result2 = _run(fetch_internal_config())
    assert result2["ai_agent.api_key"] == "key-v2"
    assert calls[0] == 2


# ── effective_api_key ─────────────────────────────────────────────────────────


def test_effective_api_key_uses_internal_config(monkeypatch):
    """Internal config value must beat env-based key."""
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={"ai_agent.api_key": "internal-sk-key"}),
    )
    result = _run(effective_api_key("openai"))
    assert result == "internal-sk-key"


def test_effective_api_key_falls_back_to_openai_env(monkeypatch):
    """When internal config has no key, fall back to settings.openai_api_key."""
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "env-openai-key", raising=False)

    result = _run(effective_api_key("openai"))
    assert result == "env-openai-key"


def test_effective_api_key_falls_back_to_openrouter_env(monkeypatch):
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "openrouter_api_key", "env-or-key", raising=False)

    result = _run(effective_api_key("openrouter"))
    assert result == "env-or-key"


def test_effective_api_key_falls_back_to_anthropic_env(monkeypatch):
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "anthropic_api_key", "env-anthropic-key", raising=False)

    result = _run(effective_api_key("anthropic"))
    assert result == "env-anthropic-key"


def test_effective_api_key_ignores_empty_string(monkeypatch):
    """Empty string from internal config must fall back to env."""
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={"ai_agent.api_key": "   "}),
    )
    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "env-fallback-key", raising=False)

    result = _run(effective_api_key("openai"))
    assert result == "env-fallback-key"


def test_effective_api_key_bedrock_returns_none(monkeypatch):
    """For bedrock provider, effective_api_key returns None (uses AWS credentials)."""
    _invalidate_internal_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    result = _run(effective_api_key("bedrock"))
    assert result is None


# ── effective_base_url ────────────────────────────────────────────────────────


def test_effective_base_url_prefers_internal_config(monkeypatch):
    """Internal config base_url beats public config."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={"ai_agent.base_url": "https://internal.example.com/v1"}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.base_url": "https://public.example.com/v1"}),
    )
    result = _run(effective_base_url())
    assert result == "https://internal.example.com/v1"


def test_effective_base_url_falls_back_to_public_config(monkeypatch):
    """When internal config has no base_url, fall back to public config."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.base_url": "https://public.example.com/v1"}),
    )
    result = _run(effective_base_url())
    assert result == "https://public.example.com/v1"


def test_effective_base_url_returns_none_when_absent(monkeypatch):
    """When neither internal nor public config has base_url, returns None."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={}),
    )
    result = _run(effective_base_url())
    assert result is None


def test_effective_base_url_ignores_empty_string(monkeypatch):
    """Empty string in both configs must produce None (fall through to provider default)."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={"ai_agent.base_url": ""}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.base_url": "  "}),
    )
    result = _run(effective_base_url())
    assert result is None


# ── build_effective_client — api_key/base_url wired correctly ─────────────────


def test_build_effective_client_openrouter_uses_internal_api_key(monkeypatch):
    """OpenRouterClient must receive the effective api_key from internal config."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()

    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.provider": "openrouter"}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={"ai_agent.api_key": "internal-or-key"}),
    )

    from agent.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "openrouter_site_url", None, raising=False)
    monkeypatch.setattr(s, "openrouter_app_name", None, raising=False)

    captured: dict = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url=None, default_headers=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    # AsyncOpenAI is imported lazily inside build_effective_client via
    # `from openai import AsyncOpenAI` — patch it at the source module level.
    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    # Also patch agent.llm.openai.AsyncOpenAI so _wrap_for_langsmith path works
    monkeypatch.setattr("agent.llm.openai.AsyncOpenAI", FakeAsyncOpenAI)
    # _wrap_for_langsmith just returns the client as-is when LangSmith not configured
    import agent.llm.openai as openai_mod
    monkeypatch.setattr(openai_mod, "_wrap_for_langsmith", lambda c: c)

    from agent.integrations.dynamic_config import build_effective_client
    _run(build_effective_client())

    assert captured.get("api_key") == "internal-or-key"


def test_build_effective_client_openai_with_base_url_override(monkeypatch):
    """OpenAIClient must use base_url from internal config when provided."""
    _invalidate_internal_config_cache()
    _invalidate_public_config_cache()

    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.provider": "openai"}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={
            "ai_agent.api_key": "internal-openai-key",
            "ai_agent.base_url": "https://custom-openai.example.com/v1",
        }),
    )

    captured: dict = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url=None, default_headers=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("agent.llm.openai.AsyncOpenAI", FakeAsyncOpenAI)
    import agent.llm.openai as openai_mod
    monkeypatch.setattr(openai_mod, "_wrap_for_langsmith", lambda c: c)

    from agent.integrations.dynamic_config import build_effective_client
    _run(build_effective_client())

    assert captured.get("api_key") == "internal-openai-key"
    assert captured.get("base_url") == "https://custom-openai.example.com/v1"
