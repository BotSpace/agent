"""Feedback-batch fixes: validator warnings (#4, #9), stale-session errors (#1),
debug-log cleaning (#8), project.create compact result (#10)."""

from __future__ import annotations

import pytest

from agent.flow.schema import FlowDocument
from agent.flow.validator import validate
from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools
from agent.tools.release import _clean_log_lines, _filter_log_lines


def _flow_tools():
    return {t.name: t for t in build_flow_tools()}


def _ctx_with(working_flow, collections=None):
    state = {"project_id": "1", "working_flow": working_flow}
    if collections is not None:
        state["collections"] = collections
    return ToolContext(run_id="r", project_id="1", user_id="u", state=state)


# ── #4: filter value type vs collection field kind ───────────────────────────


def test_filter_kind_mismatch_warns():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "load", "type": "LoadCollectionItemNode",
                 "data": {"collection_name": "users", "collection": "users", "contextKey": "u",
                          "filters": [{"field": "is_active", "operator": "equals",
                                       "valueSource": "static", "value": "true"}]}},
            ],
            "edges": [{"id": "e", "source": "t", "target": "load"}],
        }
    )
    schemas = {"users": {"is_active": "bool", "age": "int"}}
    res = validate(flow, collection_schemas=schemas)
    assert any(i.code == "filter_kind_mismatch" for i in res.warnings), res.render()
    # No error escalation.
    assert res.ok


def test_filter_kind_match_no_warning():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "load", "type": "LoadCollectionItemNode",
                 "data": {"collection": {"name": "users"}, "contextKey": "u",
                          "filters": [{"field": "is_active", "operator": "equals",
                                       "valueSource": "static", "value": True}]}},
            ],
            "edges": [{"id": "e", "source": "t", "target": "load"}],
        }
    )
    schemas = {"users": {"is_active": "bool"}}
    res = validate(flow, collection_schemas=schemas)
    assert not any(i.code == "filter_kind_mismatch" for i in res.warnings)


def test_filter_kind_state_value_skipped():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "load", "type": "LoadCollectionItemNode",
                 "data": {"collection": {"name": "users"}, "contextKey": "u",
                          "filters": [{"field": "is_active", "operator": "equals",
                                       "valueSource": "state", "stateKey": "flag"}]}},
            ],
            "edges": [{"id": "e", "source": "t", "target": "load"}],
        }
    )
    res = validate(flow, collection_schemas={"users": {"is_active": "bool"}})
    assert not any(i.code == "filter_kind_mismatch" for i in res.warnings)


# ── #9a: CustomCode declared-but-unused outputs ──────────────────────────────


def test_custom_code_unused_output_warns():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "code", "type": "CustomCodeNode",
                 "data": {"jsCode": "exit('ok')",
                          "outputs": [{"name": "ok"}, {"name": "error"}]}},
                {"id": "after", "type": "SendTextMessageNode",
                 "data": {"messageText": "done"}},
            ],
            "edges": [
                {"id": "e1", "source": "t", "target": "code"},
                {"id": "e2", "source": "code", "target": "after",
                 "sourceHandle": "output-ok"},
            ],
        }
    )
    res = validate(flow)
    unused = [i for i in res.warnings if i.code == "unused_custom_output"]
    assert len(unused) == 1
    assert "error" in unused[0].message


def test_custom_code_all_outputs_wired_no_warning():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "code", "type": "CustomCodeNode",
                 "data": {"jsCode": "exit('ok')", "outputs": [{"name": "ok"}]}},
                {"id": "after", "type": "SendTextMessageNode",
                 "data": {"messageText": "done"}},
            ],
            "edges": [
                {"id": "e1", "source": "t", "target": "code"},
                {"id": "e2", "source": "code", "target": "after",
                 "sourceHandle": "output-ok"},
            ],
        }
    )
    res = validate(flow)
    assert not any(i.code == "unused_custom_output" for i in res.warnings)


# ── #9b: unknown {{var}} references (conservative) ───────────────────────────


def test_unknown_variable_warns():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "msg", "type": "SendTextMessageNode",
                 "data": {"messageText": "Salom {{not_a_real_var}}"}},
            ],
            "edges": [{"id": "e", "source": "t", "target": "msg"}],
        }
    )
    res = validate(flow)
    assert any(i.code == "unknown_variable" for i in res.warnings), res.render()


def test_known_variable_no_warning():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "cap", "type": "StateNode", "data": {"key": "city", "type": "text"}},
                {"id": "msg", "type": "SendTextMessageNode",
                 "data": {"messageText": "Hi {{first_name}}, {{city}}, {{state.x}}"}},
            ],
            "edges": [
                {"id": "e1", "source": "t", "target": "cap"},
                {"id": "e2", "source": "cap", "target": "msg"},
            ],
        }
    )
    res = validate(flow)
    assert not any(i.code == "unknown_variable" for i in res.warnings), res.render()


# ── #1: stale-session node-missing error ─────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_missing_node_empty_flow_hints_stale_session():
    ctx = _ctx_with({"nodes": [], "edges": []})
    res = await _flow_tools()["flow.patch"].run({"id": "x", "data": {"a": 1}}, ctx)
    assert res.is_error
    assert "sessiya eskirgan" in res.content
    assert "session_use_project" in res.content


@pytest.mark.asyncio
async def test_patch_missing_node_nonempty_lists_ids():
    flow = {"nodes": [{"id": "real", "type": "SendTextMessageNode",
                       "data": {"messageText": "x"}}], "edges": []}
    ctx = _ctx_with(flow)
    res = await _flow_tools()["flow.patch"].run({"id": "ghost", "data": {"a": 1}}, ctx)
    assert res.is_error
    assert "real" in res.content
    assert "topilmadi" in res.content


# ── #8: debug-log line cleaning + filtering ──────────────────────────────────


def test_clean_log_lines_decodes_and_splits():
    raw = [b"line1\nline2", "b'line3\\nline4'", "  line5  "]
    out = _clean_log_lines(raw)
    assert out == ["line1", "line2", "line3", "line4", "line5"]


def test_filter_log_lines_by_level_and_node():
    lines = [
        '{"level":"info","node_id":"a","msg":"ok"}',
        '{"level":"error","node_id":"b","msg":"boom"}',
        '{"level":"error","node_id":"a","msg":"bad"}',
    ]
    only_err = _filter_log_lines(lines, level="error", node_id=None)
    assert len(only_err) == 2
    err_a = _filter_log_lines(lines, level="error", node_id="\"node_id\":\"a\"")
    assert len(err_a) == 1
    assert "bad" in err_a[0]


# ── #7: flow.save debug reminder + restart_debug ─────────────────────────────


class _SaveBackend(BackendClient):
    def __init__(self) -> None:
        self.debug_started = False

    async def update_project(self, project_id, body, *, auth_claims=None):
        return {"id": project_id}

    async def start_debug(self, project_id, *, auth_claims=None):
        self.debug_started = True
        return {}

    async def get_project(self, project_id, *, auth_claims=None):
        return {"id": project_id, "debug_uuid": "dbg-123"}


def _save_ctx(backend):
    flow = {
        "nodes": [
            {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            {"id": "w", "type": "SendTextMessageNode", "data": {"messageText": "hi"}},
        ],
        "edges": [{"id": "e", "source": "start", "target": "w"}],
    }
    return ToolContext(run_id="r", project_id="1", user_id="u",
                       state={"project_id": "1", "working_flow": flow},
                       services={"backend": backend})


@pytest.mark.asyncio
async def test_flow_save_static_debug_reminder():
    # restart_debug endi default TRUE (avto-debug) — reminder faqat false'da.
    backend = _SaveBackend()
    res = await _flow_tools()["flow.save"].run({"restart_debug": False}, _save_ctx(backend))
    assert not res.is_error, res.content
    assert "release_debug" in res.content
    assert backend.debug_started is False


@pytest.mark.asyncio
async def test_flow_save_auto_debug_by_default():
    # Foydalanuvchi so'ramasa ham: default save'da debug avtomatik ishga tushadi.
    backend = _SaveBackend()
    res = await _flow_tools()["flow.save"].run({}, _save_ctx(backend))
    assert not res.is_error, res.content
    assert backend.debug_started is True
    assert res.data.get("debug_link", "").endswith("start=dbg-123")


@pytest.mark.asyncio
async def test_flow_save_restart_debug_restarts_and_returns_link():
    backend = _SaveBackend()
    res = await _flow_tools()["flow.save"].run({"restart_debug": True}, _save_ctx(backend))
    assert not res.is_error, res.content
    assert backend.debug_started is True
    assert res.data.get("debug_link", "").endswith("start=dbg-123")
