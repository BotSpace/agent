"""MCP A-band trace — Mongo write shape, redaction, graceful degrade, wiring.

Two groups:
  - main (562a0bd): redaction, graceful degrade, write shape, call_tool /
    get_prompt wiring — a `_FakeCol` stands in for Mongo so no DB is needed.
  - feedback batch: `session_log_prompt` opens a trace, returns trace_id in
    structuredContent, and links every subsequent tool call to that trace
    (orphan when no prompt opened one). Ported from the old Postgres variant.
"""

from __future__ import annotations

import asyncio

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp import trace
from agent.mcp.auth import AuthedUser
from agent.mcp.draft_store import set_draft_store
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store
from tests.unit.test_mcp_session import FakeDraftStore, _FakeBackend


class _FakeCol:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    async def insert_one(self, doc: dict) -> None:
        self.docs.append(doc)


@pytest.fixture(autouse=True)
def _reset_trace():
    trace._reset()
    yield
    trace._reset()


# ── redaction ────────────────────────────────────────────────────────────────


def test_redact_masks_token_and_auth_fields():
    out = trace._redact(
        {
            "project_id": "42",
            "access_token": "secret-abc",
            "auth_claims": {"refresh": "r"},
            "nested": {"api_key": "k", "keep": "ok"},
            "list": [{"password": "p"}, "plain"],
        }
    )
    assert out["project_id"] == "42"
    assert out["access_token"] == trace._REDACTED
    assert out["auth_claims"] == trace._REDACTED
    assert out["nested"]["api_key"] == trace._REDACTED
    assert out["nested"]["keep"] == "ok"
    assert out["list"][0]["password"] == trace._REDACTED
    assert out["list"][1] == "plain"


# ── graceful degrade ─────────────────────────────────────────────────────────


def test_log_tool_call_no_mongo_is_noop(monkeypatch):
    monkeypatch.setattr(trace, "_get_col", lambda: None)
    # Must not raise even with no event loop / no Mongo.
    trace.log_tool_call(
        user_id="u",
        client_key="c",
        project_id="1",
        tool_name="flow_save",
        arguments={"token": "x"},
        result_summary="ok",
        is_error=False,
        duration_ms=5,
    )


# ── write shape ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_tool_call_writes_redacted_doc(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)
    trace.log_tool_call(
        user_id="u1",
        client_key="ck",
        project_id="7",
        tool_name="cms_insert",
        arguments={"project_id": "7", "access_token": "zzz", "doc": {"a": 1}},
        result_summary="inserted",
        is_error=False,
        duration_ms=12,
    )
    await asyncio.sleep(0)  # let the fire-and-forget task run
    assert len(col.docs) == 1
    doc = col.docs[0]
    assert doc["tool_name"] == "cms_insert"
    assert doc["user_id"] == "u1"
    assert doc["client_key"] == "ck"
    assert doc["project_id"] == "7"
    assert doc["prompt"] is None
    assert doc["arguments"]["access_token"] == trace._REDACTED
    assert doc["arguments"]["project_id"] == "7"
    assert doc["is_error"] is False
    assert doc["duration_ms"] == 12
    assert "trace_id" in doc and "created_at" in doc


@pytest.mark.asyncio
async def test_log_prompt_writes_prompt_record(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)
    tid = trace.log_prompt(
        user_id="u", client_key="c", project_id="3", prompt="build a shop bot"
    )
    await asyncio.sleep(0)
    assert len(col.docs) == 1
    doc = col.docs[0]
    assert doc["tool_name"] == trace.PROMPT_TOOL
    assert doc["prompt"] == "build a shop bot"
    assert doc["project_id"] == "3"
    # log_prompt opens a trace and returns its id (also written on the record).
    assert doc["trace_id"] == tid
    assert trace.current_trace_id("u", "c") == tid


# ── current-trace pointer ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_tool_call_attaches_to_open_trace(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)
    tid = trace.start_trace("u", "c")
    trace.log_tool_call(
        user_id="u",
        client_key="c",
        project_id=None,
        tool_name="flow_save",
        arguments={},
        result_summary="ok",
        is_error=False,
        duration_ms=1,
    )
    await asyncio.sleep(0)
    # trace_id falls back to the per-client pointer when not passed explicitly.
    assert col.docs[0]["trace_id"] == tid


@pytest.mark.asyncio
async def test_log_tool_call_auto_starts_trace_without_prompt(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)
    # No prompt opened a trace for (u2, c2). Auto-trace (#13): the first tool
    # call mints a trace so the record is NOT an orphan and later calls link.
    trace.log_tool_call(
        user_id="u2",
        client_key="c2",
        project_id=None,
        tool_name="flow_save",
        arguments={},
        result_summary="ok",
        is_error=False,
        duration_ms=1,
    )
    await asyncio.sleep(0)
    tid = col.docs[0]["trace_id"]
    assert tid is not None
    # The minted trace becomes the client's current trace for the next calls.
    assert trace.current_trace_id("u2", "c2") == tid
    trace.log_tool_call(
        user_id="u2",
        client_key="c2",
        project_id=None,
        tool_name="flow_validate",
        arguments={},
        result_summary="ok",
        is_error=False,
        duration_ms=1,
    )
    await asyncio.sleep(0)
    assert col.docs[1]["trace_id"] == tid


@pytest.mark.asyncio
async def test_ensure_trace_is_idempotent(monkeypatch):
    trace.clear_active()
    tid = trace.ensure_trace("uu", "cc")
    assert trace.ensure_trace("uu", "cc") == tid
    # A fresh prompt opens a NEW trace, replacing the auto-started one.
    new = trace.start_trace("uu", "cc")
    assert new != tid
    assert trace.ensure_trace("uu", "cc") == new


# ── server wiring ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_emits_trace(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)

    srv = build_mcp_server()
    user = AuthedUser(user_id="u9", claims={"user_id": "u9"}, client_key="ck9")
    token = mcp_server._current_user.set(user)
    try:
        handler = srv.request_handlers[mtypes.CallToolRequest]
        # docs_guide is a self-contained session tool (no backend needed).
        await handler(
            mtypes.CallToolRequest(
                method="tools/call",
                params=mtypes.CallToolRequestParams(
                    name="docs_guide", arguments={"topic": "flow"}
                ),
            )
        )
    finally:
        mcp_server._current_user.reset(token)
    await asyncio.sleep(0)
    assert any(d["tool_name"] == "docs_guide" for d in col.docs)
    doc = next(d for d in col.docs if d["tool_name"] == "docs_guide")
    assert doc["user_id"] == "u9"
    assert doc["is_error"] is False


@pytest.mark.asyncio
async def test_get_prompt_emits_prompt_trace(monkeypatch):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)

    srv = build_mcp_server()
    user = AuthedUser(user_id="u10", claims={"user_id": "u10"}, client_key="ck10")
    token = mcp_server._current_user.set(user)
    try:
        handler = srv.request_handlers[mtypes.GetPromptRequest]
        await handler(
            mtypes.GetPromptRequest(
                method="prompts/get",
                params=mtypes.GetPromptRequestParams(
                    name="build_bot", arguments={"task": "loyaliti bot"}
                ),
            )
        )
    finally:
        mcp_server._current_user.reset(token)
    await asyncio.sleep(0)
    assert any(
        d["tool_name"] == trace.PROMPT_TOOL and d["prompt"] == "loyaliti bot"
        for d in col.docs
    )


# ── session_log_prompt tool + trace linkage (feedback batch) ─────────────────


def _user(uid: str = "42", client_key: str = "ck") -> AuthedUser:
    return AuthedUser(
        user_id=uid,
        claims={"user_id": uid, "access_token": "acc", "refresh_token": "ref"},
        client_key=client_key,
    )


@pytest.fixture
def wired(monkeypatch):
    """Session store + draft store + backend wired so registry tools run."""
    get_workspace_store().clear()
    set_draft_store(FakeDraftStore())
    monkeypatch.setattr(mcp_server, "get_backend_client", lambda: _FakeBackend())
    yield
    get_workspace_store().clear()
    set_draft_store(None)


async def _call(srv, name, args):
    handler = srv.request_handlers[mtypes.CallToolRequest]
    req = mtypes.CallToolRequest(
        method="tools/call",
        params=mtypes.CallToolRequestParams(name=name, arguments=args),
    )
    return (await handler(req)).root


async def _drain():
    # Let fire-and-forget trace writes complete.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_session_log_prompt_opens_trace_and_returns_id(monkeypatch, wired):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        res = await _call(srv, "session_log_prompt", {"prompt": "/start bot qur"})
        await _drain()
    finally:
        mcp_server._current_user.reset(token)
    assert res.isError is False
    tid = res.structuredContent["trace_id"]
    assert tid
    assert trace.current_trace_id("42", "ck") == tid
    # The __prompt__ record carries the verbatim prompt and this trace_id.
    prompt_docs = [d for d in col.docs if d["tool_name"] == trace.PROMPT_TOOL]
    assert prompt_docs[0]["prompt"] == "/start bot qur"
    assert prompt_docs[0]["trace_id"] == tid


@pytest.mark.asyncio
async def test_session_log_prompt_requires_prompt(monkeypatch, wired):
    monkeypatch.setattr(trace, "_get_col", lambda: _FakeCol())
    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        res = await _call(srv, "session_log_prompt", {"prompt": "  "})
    finally:
        mcp_server._current_user.reset(token)
    assert res.isError is True


@pytest.mark.asyncio
async def test_tool_calls_attach_to_trace(monkeypatch, wired):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)

    async def fake_load_flow(project_id, **kwargs):
        return {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                }
            ],
            "edges": [],
        }

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        res = await _call(srv, "session_log_prompt", {"prompt": "build"})
        tid = res.structuredContent["trace_id"]
        await _call(srv, "session_use_project", {"project_id": "9"})
        await _call(
            srv,
            "flow_add_send_text_message",
            {"id": "m1", "data": {"messageText": "hi"}, "after": "start"},
        )
        await _drain()
    finally:
        mcp_server._current_user.reset(token)

    names = [d["tool_name"] for d in col.docs]
    assert "session_use_project" in names
    assert "flow_add_send_text_message" in names
    # Every tool-call record links to the single open trace.
    tool_docs = [d for d in col.docs if d["tool_name"] != trace.PROMPT_TOOL]
    assert tool_docs
    for d in tool_docs:
        assert d["trace_id"] == tid
        assert isinstance(d["duration_ms"], int)


@pytest.mark.asyncio
async def test_tool_call_without_prompt_auto_starts_trace(monkeypatch, wired):
    col = _FakeCol()
    monkeypatch.setattr(trace, "_get_col", lambda: col)

    async def fake_load_flow(project_id, **kwargs):
        return {"nodes": [], "edges": []}

    async def fake_load_collections(project_id, **kwargs):
        return []

    monkeypatch.setattr(mcp_server, "load_saved_flow", fake_load_flow)
    monkeypatch.setattr(mcp_server, "load_project_collections", fake_load_collections)

    srv = build_mcp_server()
    token = mcp_server._current_user.set(_user())
    try:
        await _call(srv, "session_use_project", {"project_id": "3"})
        await _call(srv, "session_status", {})
        await _drain()
    finally:
        mcp_server._current_user.reset(token)
    # No prompt opened a trace → auto-trace (#13): both calls share one minted id.
    tool_docs = [d for d in col.docs if d["tool_name"] != trace.PROMPT_TOOL]
    assert tool_docs
    ids = {d["trace_id"] for d in tool_docs}
    assert len(ids) == 1
    assert None not in ids


def teardown_function() -> None:
    # Keep the shared workspace store clean between tests.
    get_workspace_store()._store.clear()
    get_workspace_store()._active.clear()
