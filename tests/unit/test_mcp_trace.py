"""MCP A-band trace — Mongo write shape, redaction, graceful degrade, wiring."""

from __future__ import annotations

import asyncio

import mcp.types as mtypes
import pytest

from agent.mcp import server as mcp_server
from agent.mcp import trace
from agent.mcp.auth import AuthedUser
from agent.mcp.server import build_mcp_server
from agent.mcp.workspace import get_workspace_store


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
    trace.log_prompt(user_id="u", client_key="c", project_id="3", prompt="build a shop bot")
    await asyncio.sleep(0)
    assert len(col.docs) == 1
    doc = col.docs[0]
    assert doc["tool_name"] == trace.PROMPT_TOOL
    assert doc["prompt"] == "build a shop bot"
    assert doc["project_id"] == "3"


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


def teardown_function() -> None:
    # Keep the shared workspace store clean between tests.
    get_workspace_store()._store.clear()
    get_workspace_store()._active.clear()
