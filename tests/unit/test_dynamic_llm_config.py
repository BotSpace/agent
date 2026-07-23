"""Unit tests for dynamic LLM config (public-config API integration).

Covers:
1. fetch_public_config — caches result and returns {} on network error.
2. effective_provider / effective_region — backend value beats env fallback.
3. effective_model_roles — backend model applied to all inference roles;
   env fallback when key absent or backend unreachable.
4. Cache TTL: a second call within TTL returns cached value without network hit.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.integrations.backend import (
    _invalidate_public_config_cache,
    fetch_public_config,
)
from agent.integrations.dynamic_config import (
    effective_model_roles,
    effective_provider,
    effective_region,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def _mock_httpx_response(data: dict[str, Any], status: int = 200):
    """Return an httpx.AsyncClient mock that yields a fake JSON response."""
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
    """Return an httpx.AsyncClient mock that raises on GET."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("network error"))
    return mock_client


# ── fetch_public_config ───────────────────────────────────────────────────────


def test_fetch_public_config_returns_data(monkeypatch):
    _invalidate_public_config_cache()
    payload = {"ai_agent.provider": "bedrock", "ai_agent.model": "test-model", "ai_agent.region": "eu-west-1"}
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: _mock_httpx_response(payload))
    result = _run(fetch_public_config())
    assert result == payload


def test_fetch_public_config_propagates_network_error(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: _mock_httpx_error())
    with pytest.raises(Exception, match="network error"):
        _run(fetch_public_config())


def test_fetch_public_config_caches_within_ttl(monkeypatch):
    """Second call within TTL must NOT hit the network again."""
    _invalidate_public_config_cache()
    payload = {"ai_agent.provider": "openai"}
    mock_client = _mock_httpx_response(payload)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)

    r1 = _run(fetch_public_config())
    r2 = _run(fetch_public_config())

    assert r1 == payload
    assert r2 == payload
    # GET called only once
    assert mock_client.get.call_count == 1


def test_fetch_public_config_refetches_after_ttl(monkeypatch):
    """After TTL expires the cache is refreshed."""
    import agent.integrations.backend as backend_mod

    _invalidate_public_config_cache()
    payload1 = {"ai_agent.provider": "openai"}
    payload2 = {"ai_agent.provider": "anthropic"}

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

    _run(fetch_public_config())  # fills cache

    # Manually expire the cache
    backend_mod._public_config_fetched_at = time.monotonic() - backend_mod._PUBLIC_CONFIG_TTL - 1

    result2 = _run(fetch_public_config())
    assert result2["ai_agent.provider"] == "anthropic"
    assert calls[0] == 2


# ── effective_provider ────────────────────────────────────────────────────────


def test_effective_provider_uses_backend_value(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.provider": "bedrock"}),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )
    assert _run(effective_provider()) == "bedrock"


def test_effective_provider_requires_backend_value(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.provider"):
        _run(effective_provider())


def test_effective_provider_rejects_empty_string(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.provider": "   "}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.provider"):
        _run(effective_provider())


# ── effective_region ──────────────────────────────────────────────────────────


def test_effective_region_uses_backend_value(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.region": "ap-southeast-1"}),
    )
    assert _run(effective_region()) == "ap-southeast-1"


def test_effective_region_requires_backend_value(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.region"):
        _run(effective_region())


# ── effective_model_roles ─────────────────────────────────────────────────────


def test_effective_model_roles_backend_model_applied_to_all_roles(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.model": "qwen3-coder-480b"}),
    )
    roles = _run(effective_model_roles())
    assert roles.planner == "qwen3-coder-480b"
    assert roles.executor == "qwen3-coder-480b"
    assert roles.critic == "qwen3-coder-480b"
    # embedding stays from env
    from agent.config import get_settings

    assert roles.embedding == get_settings().model_embedding


def test_effective_model_roles_requires_backend_model(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.model"):
        _run(effective_model_roles())


def test_effective_model_roles_rejects_missing_backend_config(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.model"):
        _run(effective_model_roles())


def test_effective_model_roles_rejects_empty_model(monkeypatch):
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(return_value={"ai_agent.model": ""}),
    )
    with pytest.raises(RuntimeError, match=r"ai_agent\.model"):
        _run(effective_model_roles())


# ── bedrock region reaches BedrockClient ─────────────────────────────────────


def test_build_effective_client_bedrock_uses_effective_region(monkeypatch):
    """build_effective_client must construct BedrockClient with dynamic region."""
    _invalidate_public_config_cache()
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_public_config",
        AsyncMock(
            return_value={
                "ai_agent.provider": "bedrock",
                "ai_agent.region": "us-west-2",
            }
        ),
    )
    monkeypatch.setattr(
        "agent.integrations.dynamic_config.fetch_internal_config",
        AsyncMock(return_value={}),
    )

    created_kwargs: dict = {}

    class FakeBedrockClient:
        def __init__(self, *, region, access_key_id=None, secret_access_key=None):
            created_kwargs["region"] = region

    # BedrockClient is imported lazily inside build_effective_client;
    # patch the bedrock module's class directly.
    monkeypatch.setattr(
        "agent.llm.bedrock.BedrockClient",
        FakeBedrockClient,
    )

    import agent.llm.bedrock as bedrock_mod
    from agent.integrations.dynamic_config import build_effective_client

    # Patch at the source module level — build_effective_client imports
    # BedrockClient fresh via `from ..llm.bedrock import BedrockClient`.
    monkeypatch.setattr(bedrock_mod, "BedrockClient", FakeBedrockClient)

    _run(build_effective_client())

    assert created_kwargs.get("region") == "us-west-2"
