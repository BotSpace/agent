"""Tests for module/admin credential-TYPE awareness (teaching change).

credentials.list_types must surface code/admin/module type sources, and
credentials.create must accept any NON-OAuth type_key (including module types
like `openrouter.apikey`) while still rejecting OAuth (mode oauth2) types.
"""
from __future__ import annotations

import pytest

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.credentials import build_credentials_tools


class _FakeBackend(BackendClient):
    def __init__(self, *, types: list[dict] | None = None):
        self._types = types or []
        self.created: list[dict] = []

    async def list_credential_types(self, *, auth_claims=None):
        return self._types

    async def create_credential(self, body, *, auth_claims=None):
        self.created.append(body)
        return {"id": 99, **body}


_TYPES = [
    {"key": "openai", "label": "OpenAI", "mode": "bearer", "source": "code",
     "fields": [{"name": "api_key", "type": "password"}]},
    {"key": "google_sheets", "label": "Google Sheets", "mode": "oauth2",
     "source": "code", "fields": []},
    {"key": "openrouter.apikey", "label": "OpenRouter API key", "mode": "apikey",
     "source": "module:openrouter",
     "fields": [{"name": "api_key", "type": "password"}]},
]


def _tools():
    return {t.name: t for t in build_credentials_tools()}


def _ctx(backend):
    return ToolContext(run_id="r", project_id="42", user_id="u",
                       services={"backend": backend})


def test_list_types_is_registered():
    assert "credentials.list_types" in _tools()


@pytest.mark.asyncio
async def test_list_types_surfaces_three_sources():
    backend = _FakeBackend(types=_TYPES)
    res = await _tools()["credentials.list_types"].run({}, _ctx(backend))
    assert not res.is_error
    keys = {t["key"] for t in res.data["types"]}
    assert {"openai", "google_sheets", "openrouter.apikey"} <= keys
    by_key = {t["key"]: t for t in res.data["types"]}
    assert by_key["google_sheets"]["oauth"] is True
    assert by_key["openrouter.apikey"]["source"] == "module:openrouter"
    assert by_key["openrouter.apikey"]["oauth"] is False


@pytest.mark.asyncio
async def test_create_accepts_module_type():
    backend = _FakeBackend(types=_TYPES)
    res = await _tools()["credentials.create"].run(
        {
            "type_key": "openrouter.apikey",
            "name": "My OpenRouter",
            "data": {"api_key": "sk-or-xxx"},
        },
        _ctx(backend),
    )
    assert not res.is_error, res.content
    assert backend.created and backend.created[0]["type_key"] == "openrouter.apikey"


@pytest.mark.asyncio
async def test_create_rejects_oauth_type_dynamically():
    backend = _FakeBackend(types=_TYPES)
    res = await _tools()["credentials.create"].run(
        {"type_key": "google_sheets", "name": "x", "data": {}},
        _ctx(backend),
    )
    assert res.is_error
    assert "connect_oauth" in res.content
    assert not backend.created
