"""Tests for the modules.* discovery tools (search / list_available / enable).

These tools let the agent find an external module in the platform catalog
(`modules.search` via RAG, `modules.list_available` straight off the backend)
and turn one on for the active project (`modules.enable`, side-effectful).

We don't touch pgvector or the backend here: the RAG retriever, the catalog
loader and the backend client are all monkeypatched so the tests stay fast and
hermetic and assert the tool *contract* (guards, messaging, data envelope).
"""
from __future__ import annotations

import pytest

import agent.tools.modules as mod
from agent.integrations.backend import BackendError
from agent.tools.base import ToolContext
from agent.tools.modules import build_modules_tools


def _tools():
    return {t.name: t for t in build_modules_tools()}


def _ctx(project_id: str | None = "42") -> ToolContext:
    ctx = ToolContext(run_id="r", project_id=project_id, user_id="u")
    # The tools read the active project from state (not ctx.project_id).
    if project_id is not None:
        ctx.state["project_id"] = project_id
    return ctx


class _FakeRetriever:
    def __init__(self, hits=None, *, raises: Exception | None = None):
        self._hits = hits or []
        self._raises = raises
        self.calls: list[dict] = []

    async def recall(self, *, project_id, query, kind, top_k):
        self.calls.append(
            {"project_id": project_id, "query": query, "kind": kind, "top_k": top_k}
        )
        if self._raises is not None:
            raise self._raises
        return self._hits


class _FakeBackend:
    """Stand-in for BackendClient — only enable_module is exercised here."""

    def __init__(self, result=None, *, raises: Exception | None = None):
        self._result = result if result is not None else {}
        self._raises = raises
        self.calls: list[tuple] = []

    async def enable_module(self, project_id, module_slug, *, auth_claims=None):
        self.calls.append((project_id, module_slug, auth_claims))
        if self._raises is not None:
            raise self._raises
        return self._result


def _hit(slug: str, sim: float, name: str | None = None, node_types=None):
    return {
        "similarity": sim,
        "metadata": {
            "module_slug": slug,
            "name": name or slug,
            "node_types": node_types or [f"{slug}.Chat"],
            "has_trigger": False,
        },
    }


# ── modules.search ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_strong_match(monkeypatch):
    retriever = _FakeRetriever([_hit("openrouter", 0.71, "OpenRouter")])
    monkeypatch.setattr("agent.memory.get_retriever", lambda: retriever)

    res = await _tools()["modules.search"].run({"query": "another LLM provider"}, _ctx())

    assert not res.is_error
    # Searches the shared _global / module namespace, not the project pool.
    assert retriever.calls[0]["project_id"] == mod.MODULE_PROJECT_ID
    assert retriever.calls[0]["kind"] == mod.MODULE_KIND
    assert "GOOD" in res.content
    assert "modules.enable" in res.content
    assert res.data["matches"][0]["slug"] == "openrouter"
    assert res.data["matches"][0]["node_types"] == ["openrouter.Chat"]


@pytest.mark.asyncio
async def test_search_weak_match_advises_browse(monkeypatch):
    retriever = _FakeRetriever([_hit("misc", 0.20)])
    monkeypatch.setattr("agent.memory.get_retriever", lambda: retriever)

    res = await _tools()["modules.search"].run({"query": "weird thing"}, _ctx())

    assert not res.is_error
    assert "weak" in res.content.lower()
    assert "modules.list_available" in res.content


@pytest.mark.asyncio
async def test_search_no_hits(monkeypatch):
    monkeypatch.setattr("agent.memory.get_retriever", lambda: _FakeRetriever([]))

    res = await _tools()["modules.search"].run({"query": "x"}, _ctx())

    assert not res.is_error
    assert res.data["matches"] == []
    assert "modules.list_available" in res.content


@pytest.mark.asyncio
async def test_search_vector_store_unavailable(monkeypatch):
    retriever = _FakeRetriever(raises=RuntimeError("vector extension not available"))
    monkeypatch.setattr("agent.memory.get_retriever", lambda: retriever)

    res = await _tools()["modules.search"].run({"query": "x"}, _ctx())

    # Degrades gracefully — not an error, points at the offline browse tool.
    assert not res.is_error
    assert res.data["vector_store_unavailable"] is True
    assert "modules.list_available" in res.content


@pytest.mark.asyncio
async def test_search_generic_failure_is_soft(monkeypatch):
    retriever = _FakeRetriever(raises=ValueError("boom"))
    monkeypatch.setattr("agent.memory.get_retriever", lambda: retriever)

    res = await _tools()["modules.search"].run({"query": "x"}, _ctx())

    assert not res.is_error
    assert res.data["matches"] == []
    assert "modules.list_available" in res.content


# ── modules.list_available ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_available_returns_catalog(monkeypatch):
    catalog = [
        {
            "slug": "openrouter",
            "name": "OpenRouter",
            "description": "Call any LLM via OpenRouter",
            "nodes": [{"type": "openrouter.Chat"}],
        }
    ]

    async def fake_loader(*a, **kw):
        return catalog

    monkeypatch.setattr(mod, "load_modules_catalog", fake_loader)

    res = await _tools()["modules.list_available"].run({}, _ctx())

    assert not res.is_error
    assert res.data["modules"][0]["slug"] == "openrouter"
    assert res.data["modules"][0]["node_types"] == ["openrouter.Chat"]
    assert "modules.enable" in res.content


@pytest.mark.asyncio
async def test_list_available_empty(monkeypatch):
    async def fake_loader(*a, **kw):
        return []

    monkeypatch.setattr(mod, "load_modules_catalog", fake_loader)

    res = await _tools()["modules.list_available"].run({"search": "zzz"}, _ctx())

    assert not res.is_error
    assert res.data["modules"] == []
    assert "zzz" in res.content


@pytest.mark.asyncio
async def test_list_available_backend_error(monkeypatch):
    async def fake_loader(*a, **kw):
        raise BackendError("down", status=503)

    monkeypatch.setattr(mod, "load_modules_catalog", fake_loader)

    res = await _tools()["modules.list_available"].run({}, _ctx())

    assert res.is_error


# ── modules.enable ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enable_success(monkeypatch):
    backend = _FakeBackend({"status": "deploying"})
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run({"slug": "openrouter"}, _ctx())

    assert not res.is_error
    assert backend.calls == [("42", "openrouter", None)]
    assert res.data["slug"] == "openrouter"
    assert res.data["already_enabled"] is False
    assert res.data["status"] == "deploying"
    # Must warn that nodes aren't instantly queryable (pod still booting).
    assert "session_use_project" in res.content


@pytest.mark.asyncio
async def test_enable_already_enabled(monkeypatch):
    backend = _FakeBackend({"detail": "already enabled"})
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run({"slug": "openrouter"}, _ctx())

    assert not res.is_error
    assert res.data["already_enabled"] is True
    assert "already enabled" in res.content.lower()


@pytest.mark.asyncio
async def test_enable_requires_active_project(monkeypatch):
    # Should never reach the backend without a project.
    backend = _FakeBackend()
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run({"slug": "openrouter"}, _ctx(project_id=None))

    assert res.is_error
    assert "session_use_project" in res.content
    assert backend.calls == []


@pytest.mark.asyncio
async def test_enable_project_override(monkeypatch):
    backend = _FakeBackend({"status": "ok"})
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run(
        {"slug": "openrouter", "project_id": "99"}, _ctx(project_id=None)
    )

    assert not res.is_error
    assert backend.calls[0][0] == "99"


@pytest.mark.asyncio
async def test_enable_blank_slug_rejected(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run({"slug": "   "}, _ctx())

    assert res.is_error
    assert "slug" in res.content.lower()
    assert backend.calls == []


@pytest.mark.asyncio
async def test_enable_backend_error(monkeypatch):
    backend = _FakeBackend(raises=BackendError("nope", status=500))
    monkeypatch.setattr(mod, "get_backend_client", lambda: backend)

    res = await _tools()["modules.enable"].run({"slug": "openrouter"}, _ctx())

    assert res.is_error
