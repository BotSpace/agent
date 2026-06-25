"""Targeted tests for MCP-feedback fixes #4 (template normalize) and
#13 (clean default project flow)."""

from __future__ import annotations

import pytest

from agent.flow.schema import FlowDocument
from agent.flow.validator import validate
from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.flow.templates import normalize_legacy_flow
from agent.tools.project import build_project_tools

# ── #4: legacy template normalization ────────────────────────────────────────


def test_normalize_legacy_collection_object_to_name():
    raw = {
        "nodes": [
            {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            # legacy shape: collection:{name} instead of collection_name
            {"id": "list", "type": "LoadCollectionListNode",
             "data": {"collection": {"name": "products"}}},
        ],
        "edges": [{"id": "e", "source": "t", "target": "list"}],
    }
    # Before: validator flags the missing required collection_name.
    before = validate(FlowDocument.model_validate(raw))
    assert any(i.code == "missing_required_field" for i in before.errors)

    changed = normalize_legacy_flow(raw)
    assert changed == 1
    assert raw["nodes"][1]["data"]["collection_name"] == "products"

    after = validate(FlowDocument.model_validate(raw))
    assert not any(i.code == "missing_required_field" for i in after.errors), after.render()


def test_normalize_legacy_string_collection():
    raw = {
        "nodes": [
            {"id": "save", "type": "CollectionNode",
             "data": {"collection": "orders", "fieldMappings": {"a": "b"}}},
        ],
        "edges": [],
    }
    changed = normalize_legacy_flow(raw)
    assert changed == 1
    assert raw["nodes"][0]["data"]["collection_name"] == "orders"
    assert "collection" not in raw["nodes"][0]["data"]


def test_normalize_noop_when_already_flat():
    raw = {
        "nodes": [
            {"id": "list", "type": "LoadCollectionListNode",
             "data": {"collection_name": "products"}},
        ],
        "edges": [],
    }
    assert normalize_legacy_flow(raw) == 0


# ── #13: project.create installs a clean default flow ────────────────────────


class _FakeBackend(BackendClient):
    def __init__(self) -> None:
        self.created = []
        self.updated = []

    async def create_project(self, body, *, auth_claims=None):
        self.created.append(body)
        return {"id": 555, "name": body.get("name")}

    async def update_project(self, project_id, body, *, auth_claims=None):
        self.updated.append((str(project_id), body))
        return {"id": project_id}


def _project_tools():
    return {t.name: t for t in build_project_tools()}


@pytest.mark.asyncio
async def test_project_create_clean_overwrites_default_flow():
    backend = _FakeBackend()
    ctx = ToolContext(run_id="r", project_id=None, user_id="u",
                      services={"backend": backend})
    res = await _project_tools()["project.create"].run(
        {"name": "MyBot", "default_flow": "clean"}, ctx
    )
    assert not res.is_error, res.content
    # The agent sent ONLY name to create (no flow), then overwrote with a flow.
    assert "flow" not in backend.created[0]
    assert backend.updated and "flow" in backend.updated[0][1]
    flow = FlowDocument.model_validate(backend.updated[0][1]["flow"])
    assert {n.type for n in flow.nodes} == {"CommandTriggerNode", "SendTextMessageNode"}
    result = validate(flow)
    assert result.ok, result.render()
    # Working state seeded for follow-up flow.* edits.
    assert res.state_updates["working_flow"]["nodes"][0]["id"] == "start"


@pytest.mark.asyncio
async def test_project_create_returns_compact_result_no_flow():
    # #10 — project.create must NOT echo a flow JSON; only a compact summary
    # and a pointer to session_use_project.
    class _BackendWithFlow(_FakeBackend):
        async def create_project(self, body, *, auth_claims=None):
            self.created.append(body)
            return {"id": 777, "name": body.get("name"),
                    "flow": {"nodes": [{"id": "x"}], "edges": []}}

    backend = _BackendWithFlow()
    ctx = ToolContext(run_id="r", project_id=None, user_id="u",
                      services={"backend": backend})
    res = await _project_tools()["project.create"].run(
        {"name": "MyBot", "default_flow": "clean"}, ctx
    )
    assert not res.is_error, res.content
    assert "flow" not in res.data["project"]
    assert res.data["project"]["id"] == "777"
    assert "session_use_project" in res.content


@pytest.mark.asyncio
async def test_project_create_keep_skips_overwrite():
    backend = _FakeBackend()
    ctx = ToolContext(run_id="r", project_id=None, user_id="u",
                      services={"backend": backend})
    res = await _project_tools()["project.create"].run(
        {"name": "MyBot", "default_flow": "keep"}, ctx
    )
    assert not res.is_error
    assert backend.updated == []  # backend default flow untouched
