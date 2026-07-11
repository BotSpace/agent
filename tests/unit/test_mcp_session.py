"""MCP session tool'lari — session_use_project seed, session_status."""

from __future__ import annotations

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp.auth import AuthedUser
from agent.mcp.draft_store import set_draft_store
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store


def _user(uid: str = "42", client_key: str = "") -> AuthedUser:
    return AuthedUser(
        user_id=uid,
        claims={"user_id": uid, "access_token": "acc", "refresh_token": "ref"},
        client_key=client_key,
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

    async def touch_active(self, *, user_id, project_id) -> None:
        # Real DraftStore: (user,project)ni oxirgi-tanlangan deb belgilaydi. Fake — no-op yozib qo'yamiz.
        self.active = (user_id, project_id)


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
    assert _structured(result)["nodes"] == 1


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
    assert _structured(result)["draft_restored"] is True
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

    sc = _structured(result)
    assert sc["project_id"] == "3"
    assert sc["collections"] == ["orders", "users"]
    assert sc["flow_dirty"] is False


# ── #11: session_undo restores pre-clear snapshot ────────────────────────────


@pytest.mark.asyncio
async def test_session_undo_restores_flow_after_import(monkeypatch):
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
        # flow.import replaces the whole working flow (destructive) and takes a
        # pre-op snapshot, exactly like the removed flow.clear did.
        imported = await _call(
            srv,
            "flow_import",
            {"flow_json": {
                "nodes": [{"id": "only", "type": "CommandTriggerNode", "data": {"command": "/only"}}],
                "edges": [],
            }},
        )
        assert imported.isError is False
        ws = get_workspace_store().get("42")
        assert len(ws.working_flow["nodes"]) == 1
        undone = await _call(srv, "session_undo", {})
    finally:
        mcp_server._current_user.reset(token)

    assert undone.isError is False
    assert _structured(undone)["restored"] is True
    ws = get_workspace_store().get("42")
    assert len(ws.working_flow["nodes"]) == 2


# ── per-CLIENT active project pointer ────────────────────────────────────────


def _seed_flow_loaders(monkeypatch):
    """load_saved_flow returns a 1-node flow tagged with the project id so we can
    tell which project a flow tool actually operated on."""

    async def fake_load_flow(project_id, **kwargs):
        return {
            "nodes": [{"id": f"n{project_id}", "type": "SendTextMessageNode",
                       "data": {"messageText": str(project_id)}}],
            "edges": [],
        }

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)


@pytest.mark.asyncio
async def test_two_clients_same_user_independent_active_project(monkeypatch):
    """Same user_id, two different client_key — A use_project(1) and B
    use_project(2): A's flow tools target project 1, B's target project 2."""
    _seed_flow_loaders(monkeypatch)
    srv = build_mcp_server()
    store = get_workspace_store()

    user_a = _user("42", client_key="clientAAAA")
    user_b = _user("42", client_key="clientBBBB")

    # A selects project 1.
    tok = mcp_server._current_user.set(user_a)
    try:
        await _call(srv, "session_use_project", {"project_id": "1"})
    finally:
        mcp_server._current_user.reset(tok)

    # B selects project 2 — must NOT clobber A's pointer.
    tok = mcp_server._current_user.set(user_b)
    try:
        await _call(srv, "session_use_project", {"project_id": "2"})
    finally:
        mcp_server._current_user.reset(tok)

    # Pointers are independent per (user, client).
    assert store.active_project_id("42", client_key="clientAAAA") == "1"
    assert store.active_project_id("42", client_key="clientBBBB") == "2"

    # A's session_status still reports project 1 (the live bug: it used to be 2).
    tok = mcp_server._current_user.set(user_a)
    try:
        status_a = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(tok)
    assert _structured(status_a)["project_id"] == "1"

    # B's session_status reports project 2.
    tok = mcp_server._current_user.set(user_b)
    try:
        status_b = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(tok)
    assert _structured(status_b)["project_id"] == "2"

    # A flow tool from client A lands on project 1's shared workspace.
    tok = mcp_server._current_user.set(user_a)
    try:
        r = await _call(srv, "flow_patch", {"id": "n1", "data": {"messageText": "A!"}})
    finally:
        mcp_server._current_user.reset(tok)
    assert r.isError is False
    ws1 = store.get("42", "1")
    assert ws1.working_flow["nodes"][0]["data"]["messageText"] == "A!"
    # Project 2's workspace is untouched by A.
    ws2 = store.get("42", "2")
    assert ws2.working_flow["nodes"][0]["data"]["messageText"] == "2"


@pytest.mark.asyncio
async def test_same_client_sequential_use_project_last_wins(monkeypatch):
    """Single client calling use_project twice — the last selection wins
    (regression of the simple single-client case)."""
    _seed_flow_loaders(monkeypatch)
    srv = build_mcp_server()
    store = get_workspace_store()
    user = _user("42", client_key="solo")

    tok = mcp_server._current_user.set(user)
    try:
        await _call(srv, "session_use_project", {"project_id": "1"})
        await _call(srv, "session_use_project", {"project_id": "2"})
        status = await _call(srv, "session_status", {})
    finally:
        mcp_server._current_user.reset(tok)

    assert _structured(status)["project_id"] == "2"
    assert store.active_project_id("42", client_key="solo") == "2"


@pytest.mark.asyncio
async def test_explicit_project_id_overrides_active_pointer(monkeypatch):
    """A tool argument project_id always wins over the client's active pointer."""
    _seed_flow_loaders(monkeypatch)
    srv = build_mcp_server()
    store = get_workspace_store()
    user = _user("42", client_key="ovr")

    # Active pointer → project 1.
    tok = mcp_server._current_user.set(user)
    try:
        await _call(srv, "session_use_project", {"project_id": "1"})
        # Make project 2's workspace exist too (so flow_patch can run on it).
        await _call(srv, "session_use_project", {"project_id": "2"})
        # Re-point active back to 1.
        await _call(srv, "session_use_project", {"project_id": "1"})
        # Explicit project_id=2 on the tool overrides the active pointer (1).
        r = await _call(
            srv, "flow_patch", {"id": "n2", "data": {"messageText": "OVR"}, "project_id": "2"}
        )
    finally:
        mcp_server._current_user.reset(tok)

    assert r.isError is False
    # Project 2 got the edit despite the active pointer being 1.
    ws2 = store.get("42", "2")
    assert ws2.working_flow["nodes"][0]["data"]["messageText"] == "OVR"
    # Project 1 untouched.
    ws1 = store.get("42", "1")
    assert ws1.working_flow["nodes"][0]["data"]["messageText"] == "1"
    # And the active pointer is still 1.
    assert store.active_project_id("42", client_key="ovr") == "1"


def _structured(_res):
    """MCP javob content'idagi YAML blokni dict qilib qaytaradi (structuredContent o'chirilgan)."""
    import yaml as _yaml
    _text = _res.content[0].text if _res.content else ""
    for _chunk in (_text.rsplit("\n\n", 1)[-1], _text):
        try:
            _p = _yaml.safe_load(_chunk)
            if isinstance(_p, dict):
                return _p
        except Exception:
            pass
    return {}
