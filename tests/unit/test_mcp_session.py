"""MCP session tool'lari — session_use_project seed, session_status."""

from __future__ import annotations

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp.auth import AuthedUser
from agent.mcp.draft_store import set_draft_store
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store


def _user(uid: str = "42") -> AuthedUser:
    return AuthedUser(
        user_id=uid,
        claims={"user_id": uid, "access_token": "acc", "refresh_token": "ref"},
    )


class FakeDraftStore:
    """In-memory stand-in for DraftStore — no real Postgres in unit tests."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict] = {}

    async def upsert(self, *, user_id, project_id, project_name, working_flow,
                     baseline_flow, flow_dirty, collections, snapshots) -> None:
        self.rows[(user_id, project_id)] = {
            "project_name": project_name,
            "working_flow": working_flow,
            "baseline_flow": baseline_flow,
            "flow_dirty": flow_dirty,
            "collections": collections or [],
            "snapshots": list(snapshots or [])[-5:],
        }

    async def load(self, *, user_id, project_id):
        return self.rows.get((user_id, project_id))

    async def clear(self, *, user_id, project_id) -> None:
        self.rows.pop((user_id, project_id), None)


class _FakeBackend:
    async def get_project(self, project_id, *, auth_claims=None):
        return {"id": project_id, "name": f"Project {project_id}"}


@pytest.fixture(autouse=True)
def _clear_workspace(monkeypatch):
    get_workspace_store().clear()
    set_draft_store(FakeDraftStore())
    # Avoid real backend HTTP for project-name lookup in session_use_project.
    monkeypatch.setattr(mcp_server, "get_backend_client", lambda: _FakeBackend())
    yield
    get_workspace_store().clear()
    set_draft_store(None)


async def _call(srv, name: str, args: dict):
    handler = srv.request_handlers[mtypes.CallToolRequest]
    req = mtypes.CallToolRequest(
        method="tools/call",
        params=mtypes.CallToolRequestParams(name=name, arguments=args),
    )
    result = await handler(req)
    return result.root


@pytest.mark.asyncio
async def test_session_use_project_seeds_flow(monkeypatch):
    saved_flow = {
        "nodes": [{"id": "start", "type": "command_trigger"}, {"id": "msg", "type": "send_text"}],
        "edges": [{"source": "start", "target": "msg"}],
    }
    collections = [{"name": "orders", "fields": ["id", "status"]}]

    async def fake_load_flow(project_id, **kwargs):
        assert str(project_id) == "7"
        return saved_flow

    async def fake_load_collections(project_id, **kwargs):
        assert str(project_id) == "7"
        return collections

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(srv, "session_use_project", {"project_id": "7"})
    finally:
        mcp_server._current_user.reset(token)

    assert result.isError is False
    assert "2 node" in result.content[0].text
    assert "orders" in result.content[0].text

    ws = get_workspace_store().get("42")
    assert ws.project_id == "7"
    assert ws.working_flow == saved_flow
    assert ws.baseline_flow == saved_flow
    assert ws.flow_dirty is False
    assert ws.collections == collections


@pytest.mark.asyncio
async def test_session_status_no_project():
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(token)
    assert result.isError is False
    assert "Aktiv loyiha yo'q" in result.content[0].text


@pytest.mark.asyncio
async def test_session_status_after_use_project(monkeypatch):
    async def fake_load_flow(project_id, **kwargs):
        return {"nodes": [{"id": "a", "type": "x"}], "edges": []}

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "3"})
        result = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(token)

    assert result.isError is False
    assert "Aktiv loyiha: 3" in result.content[0].text
    assert result.structuredContent["nodes"] == 1


@pytest.mark.asyncio
async def test_session_use_project_requires_id():
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(srv, "session_use_project", {"project_id": ""})
    finally:
        mcp_server._current_user.reset(token)
    assert result.isError is True


# ── #1: draft persistence + restore ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_dirty_draft_is_persisted_and_restored(monkeypatch):
    saved_flow = {"nodes": [{"id": "start", "type": "CommandTriggerNode",
                             "data": {"command": "/start"}}], "edges": []}

    async def fake_load_flow(project_id, **kwargs):
        return saved_flow

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    store = get_workspace_store()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "9"})
        # Make the draft dirty via a real flow tool (add a node).
        await _call(srv, "flow_add_send_text_message",
                    {"id": "m1", "data": {"messageText": "hi"}, "after": "start"})
        ws = store.get("42")
        assert ws.flow_dirty is True
        draft_store = mcp_server.get_draft_store()
        assert ("42", "9") in draft_store.rows
        assert draft_store.rows[("42", "9")]["flow_dirty"] is True

        # Simulate pod restart: wipe in-memory workspaces, keep the draft store.
        store.clear()
        result = await _call(srv, "session_use_project", {"project_id": "9"})
    finally:
        mcp_server._current_user.reset(token)

    assert result.isError is False
    assert "draft tiklandi" in result.content[0].text
    assert result.structuredContent["draft_restored"] is True
    ws2 = store.get("42")
    assert ws2.flow_dirty is True
    assert any(n["id"] == "m1" for n in ws2.working_flow["nodes"])


# ── #2: parallel tool calls never wipe the workspace ─────────────────────────


@pytest.mark.asyncio
async def test_parallel_flow_patches_share_active_project(monkeypatch):
    import asyncio

    async def fake_load_flow(project_id, **kwargs):
        return {"nodes": [
            {"id": "m1", "type": "SendTextMessageNode", "data": {"messageText": "a"}},
            {"id": "m2", "type": "SendTextMessageNode", "data": {"messageText": "b"}},
        ], "edges": []}

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "5"})
        r1, r2 = await asyncio.gather(
            _call(srv, "flow_patch", {"id": "m1", "data": {"messageText": "A"}}),
            _call(srv, "flow_patch", {"id": "m2", "data": {"messageText": "B"}}),
        )
    finally:
        mcp_server._current_user.reset(token)

    assert r1.isError is False
    assert r2.isError is False
    ws = get_workspace_store().get("42")
    texts = {n["id"]: n["data"]["messageText"] for n in ws.working_flow["nodes"]}
    assert texts == {"m1": "A", "m2": "B"}


# ── #7: session_status returns full data ─────────────────────────────────────


@pytest.mark.asyncio
async def test_session_status_returns_collections_and_dirty(monkeypatch):
    async def fake_load_flow(project_id, **kwargs):
        return {"nodes": [{"id": "a", "type": "x"}], "edges": []}

    async def fake_load_collections(project_id, **kwargs):
        return [{"name": "orders"}, {"name": "users"}]

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "3"})
        result = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(token)

    sc = result.structuredContent
    assert sc["project_id"] == "3"
    assert sc["collections"] == ["orders", "users"]
    assert sc["flow_dirty"] is False


# ── #11: session_undo restores pre-clear snapshot ────────────────────────────


@pytest.mark.asyncio
async def test_session_undo_restores_flow_after_clear(monkeypatch):
    async def fake_load_flow(project_id, **kwargs):
        return {"nodes": [
            {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            {"id": "m1", "type": "SendTextMessageNode", "data": {"messageText": "hi"}},
        ], "edges": [{"id": "e1", "source": "start", "target": "m1"}]}

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "1"})
        cleared = await _call(srv, "flow_clear", {"confirm": True})
        assert "session_undo" in cleared.content[0].text
        ws = get_workspace_store().get("42")
        assert len(ws.working_flow["nodes"]) == 0
        undone = await _call(srv, "session_undo", {})
    finally:
        mcp_server._current_user.reset(token)

    assert undone.isError is False
    assert undone.structuredContent["restored"] is True
    ws = get_workspace_store().get("42")
    assert len(ws.working_flow["nodes"]) == 2
