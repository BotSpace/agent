"""MCP tool mapping + exec — nom konversiyasi, exclude, state merge, session."""

from __future__ import annotations

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp.auth import AuthedUser
from agent.mcp.server import (
    build_mcp_server,
    exported_tools,
    is_excluded,
    mcp_tool_name,
)
from agent.mcp.workspace import get_workspace_store
from agent.tools import build_default_registry

# ── nom mapping + exclude ────────────────────────────────────────────────────


def test_mcp_tool_name_dot_to_underscore():
    assert mcp_tool_name("flow.add_node") == "flow_add_node"
    assert mcp_tool_name("cms.insert") == "cms_insert"
    assert mcp_tool_name("ask_user") == "ask_user"


def test_excluded_namespaces():
    assert is_excluded("plan.set")
    assert is_excluded("task.run")
    assert is_excluded("ask_user")


def test_flow_ask_tools_not_excluded():
    # `flow.ask_buttons` ask_user'ga o'xshasa-da, flow.* namespace'da — qolishi kerak.
    assert not is_excluded("flow.ask_buttons")
    assert not is_excluded("flow.ask_text")


def test_exported_tools_excludes_agent_internal():
    reg = build_default_registry()
    exp = exported_tools(reg)
    assert "flow_add_node" in exp
    assert exp["flow_add_node"] == "flow.add_node"
    assert "plan_set" not in exp
    assert "task_run" not in exp
    assert "ask_user" not in exp
    # flow.ask_* qoladi
    assert "flow_ask_buttons" in exp


@pytest.mark.asyncio
async def test_list_tools_includes_session_tools():
    srv = build_mcp_server()
    handler = srv.request_handlers[mtypes.ListToolsRequest]
    result = await handler(mtypes.ListToolsRequest(method="tools/list"))
    names = {t.name for t in result.root.tools}
    assert "session_use_project" in names
    assert "session_status" in names
    assert "flow_save" in names
    assert "plan_set" not in names


@pytest.mark.asyncio
async def test_destructive_annotation():
    srv = build_mcp_server()
    handler = srv.request_handlers[mtypes.ListToolsRequest]
    result = await handler(mtypes.ListToolsRequest(method="tools/list"))
    by_name = {t.name: t for t in result.root.tools}
    assert by_name["flow_save"].annotations.destructiveHint is True
    assert by_name["flow_get"].annotations.destructiveHint is False
    assert by_name["flow_get"].annotations.readOnlyHint is True


# ── tool exec: state merge + project guard ───────────────────────────────────


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
    return result.root  # CallToolResult


@pytest.mark.asyncio
async def test_flow_tool_requires_active_project():
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(srv, "flow_get", {})
    finally:
        mcp_server._current_user.reset(token)
    assert result.isError is True
    assert "session_use_project" in result.content[0].text


@pytest.mark.asyncio
async def test_state_updates_merge_into_workspace():
    """flow.add_command_trigger → working_flow workspace'ga singadi."""
    srv = build_mcp_server()
    ws = get_workspace_store().get("42")
    ws.project_id = "7"  # loyiha tanlangan deb hisoblaymiz
    ws.working_flow = {"nodes": [], "edges": []}
    ws.baseline_flow = {"nodes": [], "edges": []}

    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(
            srv,
            "flow_add_command_trigger",
            {"id": "start", "data": {"command": "/start"}},
        )
    finally:
        mcp_server._current_user.reset(token)

    assert result.isError is False
    ws2 = get_workspace_store().get("42")
    assert ws2.flow_dirty is True
    node_ids = [n["id"] for n in (ws2.working_flow or {}).get("nodes", [])]
    assert "start" in node_ids


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        result = await _call(srv, "does_not_exist", {})
    finally:
        mcp_server._current_user.reset(token)
    assert result.isError is True


@pytest.mark.asyncio
async def test_call_without_auth_errors():
    srv = build_mcp_server()
    # contextvar o'rnatilmagan — auth yo'q
    result = await _call(srv, "session_status", {})
    assert result.isError is True
    assert "Autentifikatsiya" in result.content[0].text


# ── path normalizer (trailing-slash redirect bo'lmasligi) ────────────────────


def test_mcp_path_without_slash_no_redirect():
    """`/mcp` (slashsiz) 307 redirect EMAS, to'g'ridan-to'g'ri MCP'ga tushadi.

    TLS gateway ortida Mount'ning 307 Location'i `http://` bo'lib qolardi va
    MCP klientlar (Claude Code) downgrade redirect'ni rad etardi.
    """
    from starlette.testclient import TestClient

    from agent.mcp.server import create_standalone_app

    app = create_standalone_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        for path in ("/mcp", "/mcp/"):
            resp = client.post(
                path,
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
                headers={"Accept": "application/json, text/event-stream"},
                follow_redirects=False,
            )
            # Auth'siz — 401 kutiladi; muhimi: 3xx redirect EMAS.
            assert resp.status_code == 401, (path, resp.status_code)


# ── docs_guide + instructions + prompts ──────────────────────────────────────


@pytest.mark.asyncio
async def test_docs_guide_returns_flow_guide():
    srv = build_mcp_server()
    handler = srv.request_handlers[mtypes.CallToolRequest]
    from agent.mcp.server import _current_user

    token = _current_user.set(AuthedUser(user_id="u1", claims={"user_id": "u1"}))
    try:
        req = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(
                name="docs_guide", arguments={"topic": "flow"}
            ),
        )
        result = (await handler(req)).root
        assert not result.isError
        assert "flow" in result.content[0].text.lower()
        # node_catalog ham ishlaydi
        req2 = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(
                name="docs_guide", arguments={"topic": "node_catalog"}
            ),
        )
        result2 = (await handler(req2)).root
        assert not result2.isError
        assert len(result2.content[0].text) > 1000
        # all — agentlar yangi task boshida bitta chaqiriqda o'qiydigan umumiy guide
        req_all = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(
                name="docs_guide", arguments={"topic": "all"}
            ),
        )
        result_all = (await handler(req_all)).root
        text_all = result_all.content[0].text.lower()
        assert not result_all.isError
        assert "# flow" in text_all
        assert "# node_catalog" in text_all
        assert "# deploy" in text_all
        # noma'lum topic — xato
        req3 = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(
                name="docs_guide", arguments={"topic": "yo'q"}
            ),
        )
        result3 = (await handler(req3)).root
        assert result3.isError
    finally:
        _current_user.reset(token)


def test_server_has_instructions():
    srv = build_mcp_server()
    opts = srv.create_initialization_options()
    assert opts.instructions and "session_use_project" in opts.instructions


@pytest.mark.asyncio
async def test_build_bot_prompt():
    srv = build_mcp_server()
    handler = srv.request_handlers[mtypes.GetPromptRequest]
    req = mtypes.GetPromptRequest(
        method="prompts/get",
        params=mtypes.GetPromptRequestParams(
            name="build_bot", arguments={"task": "pizza buyurtma bot"}
        ),
    )
    result = (await handler(req)).root
    text = result.messages[0].content.text
    assert "pizza buyurtma bot" in text
    assert len(text) > 10_000  # guide + katalog + misollar


# ── OAuth resource server (RFC 9728) ─────────────────────────────────────────


def test_protected_resource_metadata_and_www_authenticate():
    from starlette.testclient import TestClient

    from agent.mcp.server import create_standalone_app, protected_resource_metadata

    meta = protected_resource_metadata()
    assert meta["resource"].endswith("/mcp")
    assert meta["authorization_servers"]

    app = create_standalone_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Metadata endpoint ochiq
        resp = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resp.status_code == 200
        assert resp.json()["resource"] == meta["resource"]
        # 401'da WWW-Authenticate resource_metadata'ga ko'rsatadi
        resp2 = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert resp2.status_code == 401
        www = resp2.headers.get("www-authenticate", "")
        assert 'resource_metadata="' in www
        assert "/.well-known/oauth-protected-resource/mcp" in www
