"""Batch2 #4 — project.snapshot: one-call overview, parallel + partial-failure
isolation. #6 — destructive confirm gating for project.delete /
cms.delete_collection."""

from __future__ import annotations

import pytest

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.cms import build_cms_tools
from agent.tools.project import build_project_tools


def _project_tools():
    return {t.name: t for t in build_project_tools()}


def _cms_tools():
    return {t.name: t for t in build_cms_tools()}


_PROJECT = {
    "id": "7",
    "name": "ShopBot",
    "debug_running": True,
    "flow": {
        "nodes": [
            {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            {"id": "m", "type": "SendTextMessageNode", "data": {"messageText": "hi"}},
        ],
        "edges": [{"id": "e", "source": "start", "target": "m"}],
    },
}


class _Backend(BackendClient):
    def __init__(self, *, fail_cms=False, fail_releases=False):
        # Skip BackendClient.__init__ (no real HTTP client needed).
        self.fail_cms = fail_cms
        self.fail_releases = fail_releases

    async def get_project(self, pid, *, auth_claims=None):
        return _PROJECT

    async def cms_list_collections(self, pid, *, auth_claims=None):
        if self.fail_cms:
            raise RuntimeError("cms down")
        return [{"name": "orders"}, {"name": "users"}]

    async def cms_list_documents(self, pid, name, *, limit=1, auth_claims=None):
        return {"count": 5 if name == "orders" else 12, "results": []}

    async def list_releases(self, pid, *, auth_claims=None):
        if self.fail_releases:
            raise RuntimeError("rel down")
        return [{"id": 3, "status": "running"}]


def _ctx(backend, *, dirty=True):
    state = {
        "project_id": "7",
        "auth_claims": {},
        "working_flow": _PROJECT["flow"],
        "flow_dirty": dirty,
    }
    return ToolContext(
        run_id="r", project_id="7", user_id="u", state=state,
        services={"backend": backend},
    )


@pytest.fixture
def _no_versions(monkeypatch):
    from agent.mcp import flow_versions as fv

    class _V:
        async def list(self, *, project_id):
            return [
                {"version": 2, "node_count": 4, "edge_count": 3, "saved_at": None},
                {"version": 1, "node_count": 2, "edge_count": 1, "saved_at": None},
            ]

    monkeypatch.setattr(fv._store, "_value", _V())


async def test_snapshot_aggregates_everything(_no_versions):
    res = await _project_tools()["project.snapshot"].run({}, _ctx(_Backend()))
    assert not res.is_error, res.content
    d = res.data
    assert d["name"] == "ShopBot"
    assert d["flow"]["nodes"] == 2
    assert any("CommandTriggerNode" in t for t in d["flow"]["triggers"])
    names = {c["name"]: c["count"] for c in d["collections"]}
    assert names == {"orders": 5, "users": 12}
    assert d["debug_status"] == "ishlayapti"
    assert d["releases"][0]["status"] == "running"
    assert d["flow_dirty"] is True
    assert len(d["versions"]) == 2
    # content carries a human overview
    assert "ShopBot" in res.content
    assert "Kolleksiyalar" in res.content


async def test_snapshot_partial_failure_isolated(_no_versions):
    # CMS down — the rest of the snapshot must still render.
    res = await _project_tools()["project.snapshot"].run(
        {}, _ctx(_Backend(fail_cms=True))
    )
    assert not res.is_error, res.content
    assert res.data["name"] == "ShopBot"
    assert res.data["collections"] is None  # isolated failure
    assert res.data["releases"][0]["id"] == 3  # others survived


async def test_snapshot_release_failure_isolated(_no_versions):
    res = await _project_tools()["project.snapshot"].run(
        {}, _ctx(_Backend(fail_releases=True))
    )
    assert not res.is_error
    assert res.data["releases"] is None
    assert res.data["collections"] is not None


async def test_snapshot_no_project():
    state = {"auth_claims": {}}
    ctx = ToolContext(run_id="r", project_id=None, user_id="u", state=state, services={})
    res = await _project_tools()["project.snapshot"].run({}, ctx)
    assert res.is_error
    assert "Aktiv loyiha yo'q" in res.content


async def test_snapshot_is_read_only_annotation():
    from agent.mcp.server import _is_read_only

    assert _is_read_only("project.snapshot") is True


# ── #6 — confirm gating ───────────────────────────────────────────────────────


async def test_project_delete_requires_confirm():
    calls = {"deleted": False}

    class _B(BackendClient):
        def __init__(self):
            pass

        async def get_project(self, pid, *, auth_claims=None):
            return {"id": pid, "name": "ShopBot"}

        async def delete_project(self, pid, *, auth_claims=None):
            calls["deleted"] = True

    ctx = ToolContext(
        run_id="r", project_id="7", user_id="u",
        state={"auth_claims": {}}, services={"backend": _B()},
    )
    res = await _project_tools()["project.delete"].run({"id": "7"}, ctx)
    assert res.is_error
    assert "confirm=true" in res.content
    assert "ShopBot" in res.content  # summary of what's deleted
    assert calls["deleted"] is False

    res = await _project_tools()["project.delete"].run({"id": "7", "confirm": True}, ctx)
    assert not res.is_error
    assert calls["deleted"] is True


async def test_cms_delete_collection_requires_confirm():
    calls = {"deleted": False}

    class _B(BackendClient):
        def __init__(self):
            pass

        async def cms_delete_collection(self, pid, name, *, auth_claims=None):
            calls["deleted"] = True

    ctx = ToolContext(
        run_id="r", project_id="7", user_id="u",
        state={"project_id": "7", "auth_claims": {}}, services={"backend": _B()},
    )
    res = await _cms_tools()["cms.delete_collection"].run({"name": "orders"}, ctx)
    assert res.is_error
    assert "confirm=true" in res.content
    assert "orders" in res.content
    assert calls["deleted"] is False

    res = await _cms_tools()["cms.delete_collection"].run(
        {"name": "orders", "confirm": True}, ctx
    )
    assert not res.is_error
    assert calls["deleted"] is True


async def test_destructive_tools_keep_destructive_hint():
    ptools = _project_tools()
    ctools = _cms_tools()
    assert ptools["project.delete"].destructive is True
    assert ctools["cms.delete_collection"].destructive is True
    ftools = {t.name: t for t in __import__("agent.tools.flow", fromlist=["build_flow_tools"]).build_flow_tools()}
    assert ftools["flow.test_delete"].destructive is True
