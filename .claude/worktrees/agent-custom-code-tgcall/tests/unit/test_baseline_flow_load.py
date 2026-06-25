"""Tests for baseline-flow seeding on a new run (agent.integrations.backend.load_saved_flow).

A fresh conversation must inspect/edit the SAVED project flow instead of
rebuilding from scratch. load_saved_flow loads project["flow"] from the backend
when the client omits a baseline, and must NEVER raise on backend failure.
"""
import pytest

from agent.integrations.backend import load_saved_flow


class _FakeBackend:
    def __init__(self, project=None, raise_exc=None):
        self._project = project or {}
        self._raise = raise_exc
        self.calls = []

    async def get_project(self, project_id, *, auth_claims=None):
        self.calls.append((project_id, auth_claims))
        if self._raise is not None:
            raise self._raise
        return self._project


@pytest.mark.asyncio
async def test_uses_client_baseline_when_nonempty():
    """A client-provided non-empty baseline is honoured; backend is NOT queried."""
    backend = _FakeBackend(project={"flow": {"nodes": [{"id": "saved"}]}})
    body = {"nodes": [{"id": "client"}], "edges": []}
    result = await load_saved_flow("42", body_baseline=body, client=backend)
    assert result == body
    assert backend.calls == []


@pytest.mark.asyncio
async def test_loads_saved_flow_when_baseline_missing():
    saved = {"nodes": [{"id": "n1"}], "edges": [{"id": "e1"}]}
    backend = _FakeBackend(project={"flow": saved})
    result = await load_saved_flow("42", body_baseline=None, client=backend, auth_claims={"sub": "u"})
    assert result == saved
    assert backend.calls == [("42", {"sub": "u"})]


@pytest.mark.asyncio
async def test_loads_saved_flow_when_baseline_empty():
    saved = {"nodes": [{"id": "n1"}], "edges": []}
    backend = _FakeBackend(project={"flow": saved})
    result = await load_saved_flow("7", body_baseline={"nodes": [], "edges": []}, client=backend)
    assert result == saved


@pytest.mark.asyncio
async def test_falls_back_to_baseline_on_backend_error():
    """Backend failure must NOT raise — fall back to the original (empty) baseline."""
    backend = _FakeBackend(raise_exc=RuntimeError("backend down"))
    body = {"nodes": [], "edges": []}
    result = await load_saved_flow("42", body_baseline=body, client=backend)
    assert result == body


@pytest.mark.asyncio
async def test_skips_global_or_missing_project():
    backend = _FakeBackend(project={"flow": {"nodes": [{"id": "x"}]}})
    assert await load_saved_flow("global", body_baseline=None, client=backend) is None
    assert await load_saved_flow(None, body_baseline=None, client=backend) is None
    assert await load_saved_flow("", body_baseline=None, client=backend) is None
    assert backend.calls == []


@pytest.mark.asyncio
async def test_returns_baseline_when_saved_flow_empty():
    backend = _FakeBackend(project={"flow": {"nodes": [], "edges": []}})
    assert await load_saved_flow("42", body_baseline=None, client=backend) is None
