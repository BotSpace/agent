"""Batch2 #3 — saved regression tests (flow.test_save/list/delete/run) and the
flow.save auto-regression integration. Fake test store + mocked simulator."""

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


def _ctx(project_id="7"):
    state = {"project_id": project_id, "auth_claims": {}, "working_flow": _FLOW}
    return ToolContext(run_id="r", project_id=project_id, user_id="u", state=state, services={})


class _FakeTestStore:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}
        self.available = True

    async def save(self, *, project_id, user_id, name, steps, enabled=True):
        if not self.available:
            return False
        self.rows[(project_id, name)] = {"name": name, "steps": steps, "enabled": enabled}
        return True

    async def list(self, *, project_id, only_enabled=False):
        out = [v for (p, _), v in self.rows.items() if p == project_id]
        if only_enabled:
            out = [v for v in out if v["enabled"]]
        return [dict(v) for v in out]

    async def get(self, *, project_id, name):
        v = self.rows.get((project_id, name))
        return dict(v) if v else None

    async def delete(self, *, project_id, name):
        return self.rows.pop((project_id, name), None) is not None


@pytest.fixture
def store(monkeypatch):
    from agent.mcp import flow_tests as ft

    s = _FakeTestStore()
    monkeypatch.setattr(ft._store, "_value", s)
    return s


@pytest.fixture
def sim_ok(monkeypatch):
    """Simulator that passes every step (expect_passed True)."""
    async def fake(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": True,
            "steps": [{"input": s, "expect_passed": True, "errors": []} for s in steps],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake)


_STEPS = [{"type": "command", "text": "/start", "expect": {"contains": "Salom"}}]


# ── save / list / delete ──────────────────────────────────────────────────────


async def test_test_save_and_list(store):
    res = await _tools()["flow.test_save"].run({"name": "happy", "steps": _STEPS}, _ctx())
    assert not res.is_error, res.content
    assert ("7", "happy") in store.rows

    res = await _tools()["flow.test_list"].run({}, _ctx())
    assert not res.is_error
    assert res.data["tests"][0]["name"] == "happy"


async def test_test_save_rejects_bad_steps(store):
    # missing required step model field shape — empty steps rejected at schema level
    res = await _tools()["flow.test_save"].run({"name": "x", "steps": []}, _ctx())
    assert res.is_error


async def test_test_save_store_unavailable(store):
    store.available = False
    res = await _tools()["flow.test_save"].run({"name": "x", "steps": _STEPS}, _ctx())
    assert res.is_error
    assert "mavjud emas" in res.content


async def test_test_delete_requires_confirm(store):
    await store.save(project_id="7", user_id="u", name="happy", steps=_STEPS)
    res = await _tools()["flow.test_delete"].run({"name": "happy"}, _ctx())
    assert res.is_error
    assert "confirm=true" in res.content
    assert ("7", "happy") in store.rows  # not deleted

    res = await _tools()["flow.test_delete"].run({"name": "happy", "confirm": True}, _ctx())
    assert not res.is_error
    assert ("7", "happy") not in store.rows


# ── run ───────────────────────────────────────────────────────────────────────


async def test_test_run_all_pass(store, sim_ok):
    await store.save(project_id="7", user_id="u", name="a", steps=_STEPS)
    await store.save(project_id="7", user_id="u", name="b", steps=_STEPS)
    res = await _tools()["flow.test_run"].run({}, _ctx())
    assert not res.is_error, res.content
    assert "2/2 test o'tdi" in res.content
    assert res.data["passed"] == 2


async def test_test_run_reports_failure(store, monkeypatch):
    await store.save(project_id="7", user_id="u", name="referral", steps=_STEPS)

    async def fake(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": False,
            "steps": [
                {"input": steps[0], "expect_passed": False,
                 "expect_failures": ["contains topilmadi"], "errors": []}
            ],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake)
    res = await _tools()["flow.test_run"].run({}, _ctx())
    assert res.is_error
    assert "0/1 test o'tdi" in res.content
    assert "referral" in res.content


async def test_test_run_simulator_down(store, monkeypatch):
    from agent.integrations.simulator import SimulatorUnavailableError

    await store.save(project_id="7", user_id="u", name="a", steps=_STEPS)

    async def fake(**kwargs):
        raise SimulatorUnavailableError("refused")

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake)
    res = await _tools()["flow.test_run"].run({}, _ctx())
    assert res.is_error
    assert "Simulyator" in res.content


async def test_test_run_no_tests(store):
    res = await _tools()["flow.test_run"].run({}, _ctx())
    assert not res.is_error
    assert "enabled test yo'q" in res.content


# ── flow.save regression integration ──────────────────────────────────────────


@pytest.fixture
def _versions(monkeypatch):
    from agent.mcp import flow_versions as fv

    class _V:
        async def append(self, **kwargs):
            return 1

        async def list(self, **kwargs):
            return []

        async def get(self, **kwargs):
            return None

    monkeypatch.setattr(fv._store, "_value", _V())


@pytest.fixture
def _backend(monkeypatch):
    from agent.integrations import backend as backend_mod

    class _B:
        async def update_project(self, project_id, body, *, auth_claims=None):
            return {"id": project_id}

    monkeypatch.setattr(backend_mod, "get_backend_client", lambda *a, **k: _B())


async def test_flow_save_runs_regression_pass(store, sim_ok, _versions, _backend):
    await store.save(project_id="7", user_id="u", name="happy", steps=_STEPS)
    res = await _tools()["flow.save"].run({"project_id": "7"}, _ctx())
    assert not res.is_error, res.content
    assert "🧪 Regression: 1/1 o'tdi" in res.content
    assert res.data["regression"]["passed"] == 1


async def test_flow_save_regression_failure_does_not_break_save(store, monkeypatch, _versions, _backend):
    await store.save(project_id="7", user_id="u", name="referral", steps=_STEPS)

    async def fake(*, simulator_url, flow, steps, collections, chat, mode="full"):
        return {
            "ok": False,
            "steps": [{"input": steps[0], "expect_passed": False,
                       "expect_failures": ["nope"], "errors": []}],
            "errors": [],
        }

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake)
    res = await _tools()["flow.save"].run({"project_id": "7"}, _ctx())
    # Save itself succeeded (flow persisted); regression failure only surfaced.
    assert not res.is_error, res.content
    assert "🧪 Regression" in res.content
    assert "referral" in res.content
    assert "flow_rollback" in res.content


async def test_flow_save_regression_simulator_down(store, monkeypatch, _versions, _backend):
    from agent.integrations.simulator import SimulatorUnavailableError

    await store.save(project_id="7", user_id="u", name="a", steps=_STEPS)

    async def fake(**kwargs):
        raise SimulatorUnavailableError("down")

    monkeypatch.setattr("agent.integrations.simulator.simulate", fake)
    res = await _tools()["flow.save"].run({"project_id": "7"}, _ctx())
    assert not res.is_error, res.content
    assert "testlar o'tkazilmadi" in res.content


async def test_flow_save_no_tests_no_regression_line(store, sim_ok, _versions, _backend):
    res = await _tools()["flow.save"].run({"project_id": "7"}, _ctx())
    assert not res.is_error
    assert "Regression" not in res.content
