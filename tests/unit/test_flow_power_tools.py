"""Registry-level tests for the flow power tools (flow.simulate / export /
import / history / rollback) — exercised through ToolContext exactly as the
LangGraph agent would, independent of the MCP layer."""

from __future__ import annotations

import pytest

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


def _ctx(*, working_flow=_FLOW, project_id="7", services=None, collections=None):
    state = {"project_id": project_id, "auth_claims": {}}
    if working_flow is not None:
        state["working_flow"] = working_flow
    if collections is not None:
        state["collections"] = collections
    return ToolContext(
        run_id="r",
        project_id=project_id,
        user_id="u",
        state=state,
        services=services or {},
    )


# ── annotations / metadata ────────────────────────────────────────────────────


def test_power_tools_present_and_annotated():
    tools = _tools()
    for name in (
        "flow.simulate",
        "flow.export",
        "flow.import",
        "flow.history",
        "flow.rollback",
    ):
        assert name in tools
    assert tools["flow.import"].destructive is True
    assert tools["flow.rollback"].destructive is True
    assert tools["flow.simulate"].destructive is False
    assert tools["flow.export"].destructive is False


# ── flow.simulate (simulator client mocked) ──────────────────────────────────


async def test_flow_simulate_passes_working_flow_and_formats(monkeypatch):
    captured = {}

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        captured["flow"] = flow
        captured["steps"] = steps
        captured["mode"] = mode
        return {
            "ok": True,
            "steps": [
                {
                    "input": {"type": "command", "text": "/start"},
                    "bot_messages": [{"message_id": 1, "text": "Salom"}],
                    "nodes": [
                        {"id": "start", "type": "CommandTriggerNode"},
                        {"id": "msg", "type": "SendTextMessageNode"},
                    ],
                    "edits": [],
                    "deletes": [],
                    "errors": [],
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    ctx = _ctx()
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start"}], "seed_collections": False}, ctx
    )
    assert not res.is_error, res.content
    assert captured["flow"]["nodes"]  # working draft was passed
    assert captured["steps"] == [{"type": "command", "text": "/start"}]
    assert "start → msg" in res.content
    assert res.data["ok"] is True


async def test_flow_simulate_seeds_collections_from_backend(monkeypatch):
    captured = {}

    async def fake_simulate(*, simulator_url, flow, steps, collections, chat, mode="full"):
        captured["collections"] = collections
        return {"ok": True, "steps": [], "errors": []}

    class _Backend:
        async def cms_list_documents(self, project_id, name, *, limit=50, auth_claims=None):
            return {"results": [{"id": 1, "status": "new"}]}

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    ctx = _ctx(
        services={"backend": _Backend()},
        collections=[{"name": "orders", "fields": ["id", "status"]}],
    )
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start"}], "seed_collections": True}, ctx
    )
    assert not res.is_error, res.content
    assert captured["collections"]["orders"] == [{"id": 1, "status": "new"}]


async def test_flow_simulate_service_down(monkeypatch):
    from agent.integrations.simulator import SimulatorUnavailableError

    async def fake_simulate(**kwargs):
        raise SimulatorUnavailableError("refused")

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake_simulate)
    res = await _tools()["flow.simulate"].run(
        {"steps": [{"type": "command", "text": "/start"}], "seed_collections": False}, _ctx()
    )
    assert res.is_error
    assert "Simulyator servisi hozircha ishga tushmagan" in res.content


# ── flow.export / flow.import ─────────────────────────────────────────────────


async def test_flow_export_returns_full_flow():
    res = await _tools()["flow.export"].run({}, _ctx())
    assert not res.is_error
    assert res.data["nodes"] == _FLOW["nodes"]
    assert "```json" in res.content


async def test_flow_import_replaces_after_validation():
    new_flow = {
        "nodes": [
            {"id": "t2", "type": "CommandTriggerNode", "data": {"command": "/help"}},
            {"id": "m2", "type": "SendTextMessageNode", "data": {"messageText": "Yordam"}},
        ],
        "edges": [{"id": "e2", "source": "t2", "target": "m2"}],
    }
    res = await _tools()["flow.import"].run({"flow_json": new_flow}, _ctx())
    assert not res.is_error, res.content
    assert res.state_updates["flow_dirty"] is True
    ids = [n["id"] for n in res.state_updates["working_flow"]["nodes"]]
    assert ids == ["t2", "m2"]


async def test_flow_import_rejects_invalid_flow():
    bad = {
        "nodes": [{"id": "x", "type": "NotARealNodeType", "data": {}}],
        "edges": [{"id": "e", "source": "x", "target": "missing"}],
    }
    res = await _tools()["flow.import"].run({"flow_json": bad}, _ctx())
    assert res.is_error
    assert "RAD ETILDI" in res.content
    # No state mutation on rejection.
    assert "working_flow" not in res.state_updates


# ── flow.history / flow.rollback (fake version store) ────────────────────────


class _FakeVersionStore:
    def __init__(self):
        self.rows = {}

    async def append(self, *, project_id, user_id, flow, node_count, edge_count):
        lst = self.rows.setdefault(project_id, [])
        version = (lst[-1]["version"] + 1) if lst else 1
        lst.append({
            "version": version, "flow": flow, "node_count": node_count,
            "edge_count": edge_count, "saved_at": "2026-06-04T00:00:00+00:00",
        })
        return version

    async def list(self, *, project_id):
        return [
            {"version": r["version"], "saved_at": r["saved_at"],
             "node_count": r["node_count"], "edge_count": r["edge_count"]}
            for r in reversed(self.rows.get(project_id, []))
        ]

    async def get(self, *, project_id, version):
        for r in self.rows.get(project_id, []):
            if r["version"] == version:
                return {"flow": r["flow"], "node_count": r["node_count"],
                        "edge_count": r["edge_count"], "saved_at": r["saved_at"]}
        return None


@pytest.fixture
def _fake_versions(monkeypatch):
    from agent.mcp import flow_versions as fv

    store = _FakeVersionStore()
    monkeypatch.setattr(fv._store, "_value", store)
    return store


async def test_flow_history_lists_versions(_fake_versions):
    await _fake_versions.append(
        project_id="7", user_id="u", flow=_FLOW, node_count=2, edge_count=1
    )
    res = await _tools()["flow.history"].run({}, _ctx())
    assert not res.is_error, res.content
    versions = res.data["versions"]
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert versions[0]["node_count"] == 2


async def test_flow_history_empty_when_never_saved(_fake_versions):
    res = await _tools()["flow.history"].run({}, _ctx())
    assert not res.is_error
    assert res.data["versions"] == []


async def test_flow_rollback_loads_version(_fake_versions):
    await _fake_versions.append(
        project_id="7", user_id="u", flow=_FLOW, node_count=2, edge_count=1
    )
    res = await _tools()["flow.rollback"].run({"version": 1}, _ctx())
    assert not res.is_error, res.content
    ids = [n["id"] for n in res.state_updates["working_flow"]["nodes"]]
    assert ids == ["start", "msg"]
    assert res.state_updates["flow_dirty"] is True
    assert "flow.save" in res.content


async def test_flow_rollback_unknown_version_errors(_fake_versions):
    await _fake_versions.append(
        project_id="7", user_id="u", flow=_FLOW, node_count=2, edge_count=1
    )
    res = await _tools()["flow.rollback"].run({"version": 99}, _ctx())
    assert res.is_error
    assert "topilmadi" in res.content


# ── flow.save now records a version (registry-level) ─────────────────────────


async def test_flow_save_records_version(monkeypatch, _fake_versions):
    from agent.integrations import backend as backend_mod

    class _Backend:
        async def update_project(self, project_id, body, *, auth_claims=None):
            return {"id": project_id}

    monkeypatch.setattr(backend_mod, "get_backend_client", lambda *a, **k: _Backend())
    res = await _tools()["flow.save"].run({"project_id": "7"}, _ctx())
    assert not res.is_error, res.content
    listed = await _fake_versions.list(project_id="7")
    assert len(listed) == 1
    assert listed[0]["node_count"] == 2


# ── MCP export surface ───────────────────────────────────────────────────────


def test_mcp_export_includes_power_tools():
    from agent.mcp.server import exported_tools
    from agent.tools import build_default_registry

    exported = exported_tools(build_default_registry())
    for mcp_name in (
        "flow_simulate",
        "flow_export",
        "flow_import",
        "flow_history",
        "flow_rollback",
    ):
        assert mcp_name in exported, mcp_name


def test_fmt_simulate_step_surfaces_alerts():
    """CustomCode alert.* per-step render'da level ikonkasi bilan ko'rinadi."""
    from agent.tools.flow.power import _fmt_simulate_step

    step = {
        "input": {"type": "text", "text": "salom"},
        "bot_messages": [{"text": "javob"}],
        "nodes": [{"id": "n1", "type": "CustomCodeNode"}],
        "alerts": [
            {"level": "warning", "message": "narx yuqori", "detail": "amount=999"},
            {"level": "info", "message": "ok"},
        ],
    }
    out = "\n".join(_fmt_simulate_step(1, step))
    assert "🟡 alert[warning]: narx yuqori — amount=999" in out
    assert "🔵 alert[info]: ok" in out
