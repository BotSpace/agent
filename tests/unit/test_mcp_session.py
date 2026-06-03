"""MCP session tool'lari — session_use_project seed, session_status."""

from __future__ import annotations

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp.auth import AuthedUser
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store


def _user(uid: str = "42") -> AuthedUser:
    return AuthedUser(
        user_id=uid,
        claims={"user_id": uid, "access_token": "acc", "refresh_token": "ref"},
    )


@pytest.fixture(autouse=True)
def _clear_workspace():
    get_workspace_store().clear()
    yield
    get_workspace_store().clear()


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
