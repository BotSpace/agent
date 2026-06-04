"""MCP power tools — bot_simulate (#1), flow_export/import (#2),
flow_history/rollback (#3), {{env.*}} validator whitelist (#4), transparent
session auto-restore (#5)."""

from __future__ import annotations

import mcp.types as mtypes
import pytest

from agent.flow.schema import FlowDocument
from agent.flow.validator import validate
from agent.mcp import server as mcp_server
from agent.mcp.auth import AuthedUser
from agent.mcp.draft_store import set_draft_store
from agent.mcp.flow_versions import set_flow_version_store
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store

# ── helpers ──────────────────────────────────────────────────────────────────


def _user(uid: str = "42", client_key: str = "") -> AuthedUser:
    return AuthedUser(
        user_id=uid,
        claims={"user_id": uid, "access_token": "acc", "refresh_token": "ref"},
        client_key=client_key,
    )


_FLOW = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
        {"id": "msg", "type": "SendTextMessageNode", "data": {"messageText": "Salom"}},
    ],
    "edges": [{"id": "e", "source": "start", "target": "msg"}],
}


class FakeDraftStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict] = {}
        self.order: list[tuple[str, str]] = []

    async def upsert(self, *, user_id, project_id, project_name, working_flow,
                     baseline_flow, flow_dirty, collections, snapshots) -> None:
        key = (user_id, project_id)
        self.rows[key] = {
            "project_name": project_name,
            "working_flow": working_flow,
            "baseline_flow": baseline_flow,
            "flow_dirty": flow_dirty,
            "collections": collections or [],
            "snapshots": list(snapshots or [])[-5:],
        }
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)  # most-recent last

    async def load(self, *, user_id, project_id):
        return self.rows.get((user_id, project_id))

    async def load_latest(self, *, user_id):
        for key in reversed(self.order):
            if key[0] == user_id:
                row = dict(self.rows[key])
                row["project_id"] = key[1]
                return row
        return None

    async def clear(self, *, user_id, project_id) -> None:
        key = (user_id, project_id)
        self.rows.pop(key, None)
        if key in self.order:
            self.order.remove(key)


class FakeVersionStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {}

    async def append(self, *, project_id, user_id, flow, node_count, edge_count):
        lst = self.rows.setdefault(project_id, [])
        version = (lst[-1]["version"] + 1) if lst else 1
        lst.append({
            "version": version,
            "flow": flow,
            "node_count": node_count,
            "edge_count": edge_count,
            "saved_at": "2026-06-04T00:00:00+00:00",
        })
        # keep last 20
        self.rows[project_id] = lst[-20:]
        return version

    async def list(self, *, project_id):
        lst = self.rows.get(project_id, [])
        return [
            {"version": r["version"], "saved_at": r["saved_at"],
             "node_count": r["node_count"], "edge_count": r["edge_count"]}
            for r in reversed(lst)
        ]

    async def get(self, *, project_id, version):
        for r in self.rows.get(project_id, []):
            if r["version"] == version:
                return {"flow": r["flow"], "node_count": r["node_count"],
                        "edge_count": r["edge_count"], "saved_at": r["saved_at"]}
        return None


class _FakeBackend:
    def __init__(self, docs=None):
        self._docs = docs or {}

    async def get_project(self, project_id, *, auth_claims=None):
        return {"id": project_id, "name": f"Project {project_id}", "flow": _FLOW}

    async def cms_list_documents(self, project_id, name, *, limit=50, skip=0,
                                 filter=None, sort=None, auth_claims=None):
        return {"results": self._docs.get(name, []), "total": len(self._docs.get(name, []))}


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    get_workspace_store().clear()
    set_draft_store(FakeDraftStore())
    set_flow_version_store(FakeVersionStore())
    monkeypatch.setattr(mcp_server, "get_backend_client", lambda: _FakeBackend())
    yield
    get_workspace_store().clear()
    set_draft_store(None)
    set_flow_version_store(None)


async def _call(srv, name: str, args: dict, user=None):
    user = user or _user()
    token = mcp_server._current_user.set(user)
    try:
        handler = srv.request_handlers[mtypes.CallToolRequest]
        req = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(name=name, arguments=args),
        )
        result = await handler(req)
        return result.root
    finally:
        mcp_server._current_user.reset(token)


def _text(res) -> str:
    return "\n".join(
        b.text for b in (res.content or []) if getattr(b, "text", None) is not None
    )


async def _activate(srv, project_id="7", user=None, flow=None):
    """Set the active project with a working flow in the workspace."""
    user = user or _user()
    store = get_workspace_store()
    ws = store.set_active(user.user_id, project_id, client_key=user.client_key)
    ws.project_id = project_id
    ws.working_flow = flow or dict(_FLOW)
    ws.baseline_flow = flow or dict(_FLOW)
    ws.collections = [{"name": "orders", "fields": ["id", "status"]}]
    return ws


# ── #4: {{env.*}} validator whitelist ────────────────────────────────────────


def test_env_variable_is_whitelisted_no_warning():
    flow = FlowDocument.model_validate({
        "nodes": [
            {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            {"id": "m", "type": "SendTextMessageNode",
             "data": {"messageText": "Token {{env.SECRET_TOKEN}} and {{env.API_KEY}}"}},
        ],
        "edges": [{"id": "e", "source": "t", "target": "m"}],
    })
    res = validate(flow)
    assert not any(i.code == "unknown_variable" for i in res.issues), res.render()


def test_bare_env_root_no_warning_but_unknown_still_warns():
    flow = FlowDocument.model_validate({
        "nodes": [
            {"id": "m", "type": "SendTextMessageNode",
             "data": {"messageText": "{{env}} {{totallyunknownvar}}"}},
        ],
        "edges": [],
    })
    res = validate(flow)
    codes = [i.message for i in res.issues if i.code == "unknown_variable"]
    assert any("totallyunknownvar" in m for m in codes)
    assert not any("{{env}}" in m for m in codes)


# ── #2: flow_export / flow_import ────────────────────────────────────────────


async def test_flow_export_returns_full_flow():
    srv = build_mcp_server()
    await _activate(srv)
    res = await _call(srv, "flow_export", {})
    assert not res.isError
    assert res.structuredContent["nodes"] == _FLOW["nodes"]
    assert "```json" in _text(res)


async def test_flow_import_replaces_after_validation():
    srv = build_mcp_server()
    await _activate(srv)
    new_flow = {
        "nodes": [
            {"id": "t2", "type": "CommandTriggerNode", "data": {"command": "/help"}},
            {"id": "m2", "type": "SendTextMessageNode", "data": {"messageText": "Yordam"}},
        ],
        "edges": [{"id": "e2", "source": "t2", "target": "m2"}],
    }
    res = await _call(srv, "flow_import", {"flow_json": new_flow})
    assert not res.isError, _text(res)
    ws = get_workspace_store().get("42", client_key="")
    assert [n["id"] for n in ws.working_flow["nodes"]] == ["t2", "m2"]
    assert ws.flow_dirty is True
    # snapshot taken → undo-able
    assert len(ws.snapshots) == 1


async def test_flow_import_rejects_invalid_flow():
    srv = build_mcp_server()
    await _activate(srv)
    bad = {
        "nodes": [{"id": "x", "type": "NotARealNodeType", "data": {}}],
        "edges": [{"id": "e", "source": "x", "target": "missing"}],
    }
    res = await _call(srv, "flow_import", {"flow_json": bad})
    assert res.isError
    assert "RAD ETILDI" in _text(res)
    # original flow untouched
    ws = get_workspace_store().get("42", client_key="")
    assert [n["id"] for n in ws.working_flow["nodes"]] == ["start", "msg"]


async def test_flow_import_export_roundtrip():
    srv = build_mcp_server()
    await _activate(srv)
    exported = (await _call(srv, "flow_export", {})).structuredContent
    res = await _call(srv, "flow_import", {"flow_json": exported})
    assert not res.isError, _text(res)


# ── #3: flow_history / flow_rollback ─────────────────────────────────────────


async def test_flow_save_records_version_and_history_lists_it():
    srv = build_mcp_server()
    ws = await _activate(srv)
    # simulate a successful flow.save by calling the version append path directly
    await mcp_server._append_flow_version(ws, _user())
    res = await _call(srv, "flow_history", {})
    assert not res.isError, _text(res)
    versions = res.structuredContent["versions"]
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert versions[0]["node_count"] == 2


async def test_flow_rollback_loads_version_into_workspace():
    srv = build_mcp_server()
    ws = await _activate(srv)
    # v1 = current 2-node flow
    await mcp_server._append_flow_version(ws, _user())
    # mutate + save v2 (1 node)
    ws.working_flow = {"nodes": [{"id": "only", "type": "CommandTriggerNode",
                                  "data": {"command": "/x"}}], "edges": []}
    await mcp_server._append_flow_version(ws, _user())

    res = await _call(srv, "flow_rollback", {"version": 1})
    assert not res.isError, _text(res)
    ws2 = get_workspace_store().get("42", client_key="")
    assert [n["id"] for n in ws2.working_flow["nodes"]] == ["start", "msg"]
    assert ws2.flow_dirty is True
    assert "flow_save" in _text(res)


async def test_flow_rollback_unknown_version_errors():
    srv = build_mcp_server()
    ws = await _activate(srv)
    await mcp_server._append_flow_version(ws, _user())
    res = await _call(srv, "flow_rollback", {"version": 99})
    assert res.isError
    assert "topilmadi" in _text(res)


async def test_flow_history_empty_when_never_saved():
    srv = build_mcp_server()
    await _activate(srv)
    res = await _call(srv, "flow_history", {})
    assert not res.isError
    assert res.structuredContent["versions"] == []


# ── #1: bot_simulate ─────────────────────────────────────────────────────────


async def test_bot_simulate_formats_steps(monkeypatch):
    srv = build_mcp_server()
    await _activate(srv)

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat):
        assert flow["nodes"]  # working draft passed
        return {
            "ok": True,
            "steps": [
                {
                    "input": {"type": "command", "text": "/start"},
                    "bot_messages": [{"message_id": 1, "text": "Salom"}],
                    "nodes": [{"id": "start", "type": "CommandTriggerNode"},
                              {"id": "msg", "type": "SendTextMessageNode"}],
                    "edits": [], "deletes": [], "errors": [],
                }
            ],
            "final_state": {"state": "msg", "context": {}},
            "collections_after": {},
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _call(srv, "bot_simulate",
                      {"steps": [{"type": "command", "text": "/start"}],
                       "seed_collections": False})
    assert not res.isError, _text(res)
    body = _text(res)
    assert "➡️" in body and "🤖" in body
    assert "start → msg" in body
    assert res.structuredContent["ok"] is True


async def test_bot_simulate_seeds_collections(monkeypatch):
    srv = build_mcp_server()
    await _activate(srv)
    captured = {}

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat):
        captured["collections"] = collections
        return {"ok": True, "steps": [], "errors": []}

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    monkeypatch.setattr(
        mcp_server, "get_backend_client",
        lambda: _FakeBackend(docs={"orders": [{"id": 1, "status": "new"}]}),
    )
    res = await _call(srv, "bot_simulate",
                      {"steps": [{"type": "command", "text": "/start"}],
                       "seed_collections": True})
    assert not res.isError, _text(res)
    assert captured["collections"]["orders"] == [{"id": 1, "status": "new"}]


async def test_bot_simulate_service_down(monkeypatch):
    srv = build_mcp_server()
    await _activate(srv)
    from agent.integrations.simulator import SimulatorUnavailableError

    async def fake_simulate(**kwargs):
        raise SimulatorUnavailableError("connection refused")

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _call(srv, "bot_simulate",
                      {"steps": [{"type": "command", "text": "/start"}],
                       "seed_collections": False})
    assert res.isError
    assert "Simulyator servisi hozircha ishga tushmagan" in _text(res)


async def test_bot_simulate_requires_steps():
    srv = build_mcp_server()
    await _activate(srv)
    res = await _call(srv, "bot_simulate", {"steps": []})
    assert res.isError
    assert "steps majburiy" in _text(res)


# ── #5: transparent session auto-restore ─────────────────────────────────────


async def test_auto_restore_on_flow_tool_after_pointer_lost():
    srv = build_mcp_server()
    user = _user()
    # Persist a dirty draft, then wipe the in-memory store (pod restart).
    store = get_workspace_store()
    ws = store.set_active(user.user_id, "55", client_key=user.client_key)
    ws.project_id = "55"
    ws.working_flow = dict(_FLOW)
    ws.flow_dirty = True
    await mcp_server._persist_draft(ws)
    store.clear()  # pointer + workspace gone

    # A flow tool should auto-restore and succeed.
    res = await _call(srv, "flow_get", {"summary_only": True}, user=user)
    assert not res.isError, _text(res)
    assert "Sessiya avtomatik tiklandi (loyiha 55)" in _text(res)


async def test_session_status_auto_restores():
    srv = build_mcp_server()
    user = _user()
    store = get_workspace_store()
    ws = store.set_active(user.user_id, "77", client_key=user.client_key)
    ws.project_id = "77"
    ws.working_flow = dict(_FLOW)
    ws.flow_dirty = True
    await mcp_server._persist_draft(ws)
    store.clear()

    res = await _call(srv, "session_status", {}, user=user)
    assert not res.isError
    assert res.structuredContent["project_id"] == "77"
    assert res.structuredContent["session_restored"] is True


async def test_no_restore_when_nothing_persisted():
    srv = build_mcp_server()
    user = _user("999")
    res = await _call(srv, "flow_get", {"summary_only": True}, user=user)
    # No draft → old "no active project" error remains.
    assert res.isError
    assert "Aktiv loyiha yo'q" in _text(res)


# ── store SQL paths via a fake psycopg connection ────────────────────────────


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        self._result = self._conn.responder(sql, params)

    async def fetchone(self):
        return self._result

    async def fetchall(self):
        return self._result if isinstance(self._result, list) else []


class _FakeConn:
    def __init__(self, responder):
        self.responder = responder
        self.executed: list = []
        self.committed = False

    def cursor(self):
        return _FakeCursor(self)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_psycopg(monkeypatch, module, responder):
    class _AC:
        @staticmethod
        async def connect(dsn):
            return _FakeConn(responder)

    monkeypatch.setattr(module.psycopg, "AsyncConnection", _AC)


async def test_flow_version_store_append_and_list(monkeypatch):
    from agent.mcp import flow_versions as fv

    rows: list[dict] = []

    def responder(sql, params):
        s = " ".join(sql.split())
        if s.startswith("CREATE TABLE"):
            return None
        if s.startswith("SELECT COALESCE(MAX(version)"):
            return (max((r["version"] for r in rows), default=0),)
        if s.startswith("INSERT INTO mcp_flow_versions"):
            _pid, _uid, version, _flow, nc, ec = params
            rows.append({"version": version, "node_count": nc, "edge_count": ec})
            return None
        if s.startswith("DELETE FROM mcp_flow_versions"):
            return None
        if s.startswith("SELECT version, saved_at"):
            return [
                (r["version"], None, r["node_count"], r["edge_count"])
                for r in sorted(rows, key=lambda r: -r["version"])
            ]
        return None

    store = fv.FlowVersionStore("postgresql://x")
    _patch_psycopg(monkeypatch, fv, responder)

    v1 = await store.append(project_id="9", user_id="u", flow=_FLOW, node_count=2, edge_count=1)
    v2 = await store.append(project_id="9", user_id="u", flow=_FLOW, node_count=3, edge_count=2)
    assert (v1, v2) == (1, 2)
    listed = await store.list(project_id="9")
    assert [r["version"] for r in listed] == [2, 1]


async def test_draft_store_load_latest(monkeypatch):
    from agent.mcp import draft_store as ds

    def responder(sql, params):
        s = " ".join(sql.split())
        if s.startswith("CREATE TABLE"):
            return None
        if "ORDER BY updated_at DESC" in s:
            # project_id, project_name, working_flow, baseline_flow,
            # flow_dirty, collections, snapshots
            return ("88", "Proj 88", _FLOW, _FLOW, True, [], [])
        return None

    store = ds.DraftStore("postgresql://x")
    _patch_psycopg(monkeypatch, ds, responder)
    row = await store.load_latest(user_id="u")
    assert row is not None
    assert row["project_id"] == "88"
    assert row["flow_dirty"] is True
