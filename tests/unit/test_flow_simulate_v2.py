"""Batch2 #1 — flow.simulate v2: per-step chat/expect, advance_time, mode=assert.

Exercised through ToolContext exactly like the LangGraph agent / MCP layer.
The simulator transport is mocked so no engine service is needed."""

from __future__ import annotations

from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools


def _tools():
    return {t.name: t for t in build_flow_tools()}


_FLOW = {
    "nodes": [
        {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
        {"id": "msg", "type": "SendTextMessageNode", "data": {"messageText": "Salom"}},
    ],
    "edges": [{"id": "e", "source": "start", "target": "msg"}],
}


def _ctx(working_flow=_FLOW, project_id="7"):
    state = {"project_id": project_id, "auth_claims": {}, "working_flow": working_flow}
    return ToolContext(run_id="r", project_id=project_id, user_id="u", state=state, services={})


async def test_simulate_forwards_chat_expect_advance_time_and_mode(monkeypatch):
    captured = {}

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        captured["steps"] = steps
        captured["mode"] = mode
        return {"ok": True, "steps": [], "errors": []}

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {
            "steps": [
                {
                    "type": "command",
                    "text": "/start",
                    "chat": {"id": 99, "first_name": "B", "username": "b"},
                    "expect": {"contains": "Salom", "no_errors": True},
                },
                {"type": "advance_time", "seconds": 3600},
            ],
            "seed_collections": False,
            "mode": "assert",
        },
        _ctx(),
    )
    assert not res.is_error
    assert captured["mode"] == "assert"
    # chat + expect ride inside the step dicts; advance_time carries seconds.
    s0 = captured["steps"][0]
    assert s0["chat"] == {"id": 99, "first_name": "B", "username": "b"}
    assert s0["expect"] == {"contains": "Salom", "no_errors": True}
    assert captured["steps"][1] == {"type": "advance_time", "seconds": 3600}


async def test_simulate_forwards_media_step(monkeypatch):
    captured = {}

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        captured["steps"] = steps
        return {"ok": True, "steps": [], "errors": []}

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {
            "steps": [
                {
                    "type": "video",
                    "file_id": "BAAC_SIM_VIDEO",
                    "caption": "1-qism",
                    "file_name": "ep1.mp4",
                    "mime_type": "video/mp4",
                }
            ],
            "seed_collections": False,
        },
        _ctx(),
    )

    assert not res.is_error
    assert captured["steps"][0]["file_id"] == "BAAC_SIM_VIDEO"
    assert captured["steps"][0]["caption"] == "1-qism"


async def test_assert_mode_compact_pass(monkeypatch):
    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": True,
            "steps": [
                {"input": {"type": "command", "text": "/start"}, "expect_passed": True, "errors": []},
                {"input": {"type": "text", "text": "hi"}, "expect_passed": True, "errors": []},
            ],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start"}], "seed_collections": False, "mode": "assert"},
        _ctx(),
    )
    assert not res.is_error
    assert "✅ 2/2 o'tdi" in res.content
    assert res.data["passed"] == 2
    assert res.data["failed"] == []


async def test_assert_mode_reports_only_failed(monkeypatch):
    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": False,
            "steps": [
                {"input": {"type": "command", "text": "/start"}, "expect_passed": True, "errors": []},
                {
                    "input": {"type": "text", "text": "oops"},
                    "expect_passed": False,
                    "expect_failures": ["contains 'Tugma' topilmadi"],
                    "errors": [],
                },
                {"input": {"type": "text", "text": "z"}, "expect_passed": True, "errors": ["boom"]},
            ],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start"}], "seed_collections": False, "mode": "assert"},
        _ctx(),
    )
    assert res.is_error  # there were failures
    assert "1/3 o'tdi" in res.content
    # Only failed steps are detailed.
    assert "step 2" in res.content
    assert "topilmadi" in res.content
    assert "step 3" in res.content  # engine error counts as failure
    failed_steps = {f["step"] for f in res.data["failed"]}
    assert failed_steps == {2, 3}


async def test_full_mode_marks_expect(monkeypatch):
    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": True,
            "steps": [
                {
                    "input": {"type": "command", "text": "/start"},
                    "bot_messages": [{"message_id": 1, "text": "Salom"}],
                    "nodes": [{"id": "start"}, {"id": "msg"}],
                    "expect_passed": True,
                    "errors": [],
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start", "expect": {"contains": "Salom"}}],
         "seed_collections": False},
        _ctx(),
    )
    assert not res.is_error
    assert "✅ expect: o'tdi" in res.content
