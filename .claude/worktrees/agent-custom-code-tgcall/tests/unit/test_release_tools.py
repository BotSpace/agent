import pytest

from agent.tools.base import ToolContext
from agent.tools.release import build_release_tools


class _FakeBackend:
    def __init__(self):
        self.list_release_project_ids: list[str | int] = []
        self.get_release_ids: list[str | int] = []

    async def list_releases(self, project_id, *, auth_claims=None):
        self.list_release_project_ids.append(project_id)
        return [{"id": 7, "version": 1, "status": "running"}]

    async def get_release(self, release_id, *, auth_claims=None):
        self.get_release_ids.append(release_id)
        return {"id": release_id, "version": 1, "status": "running"}

    async def get_release_logs(self, release_id, *, tail=200, since_seconds=None, auth_claims=None):
        return {
            "pod_name": "engine-test-pod",
            "lines": [
                "2026-05-28T10:00:00Z starting bot",
                "2026-05-28T10:00:01Z connected to mongo",
            ],
            "truncated": False,
        }

    async def subscription_current(self, *, auth_claims=None):
        return {"plan": {"id": 1, "name": "free"}, "status": "active"}

    async def subscription_upgrade(self, *, plan_id: int, months: int = 1, auth_claims=None):
        return {"new_plan": f"plan-{plan_id}", "months": months}

    async def create_payment_link(self, *, method: str, amount: int, auth_claims=None):
        return {"link": f"https://pay.local/{method}/{amount}"}


def _tools():
    return {t.name: t for t in build_release_tools()}


@pytest.mark.asyncio
async def test_subscription_current_tool():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _FakeBackend()},
    )
    res = await tools["subscription.current"].run({}, ctx)
    assert not res.is_error
    assert "Current plan" in res.content
    assert res.data and res.data["subscription"]["plan"]["name"] == "free"


class _BackendWithIntPlan(_FakeBackend):
    async def subscription_current(self, *, auth_claims=None):
        # Real prod response sometimes returns plan as an int (plan_id)
        return {"plan": 7, "plan_name": "pro", "status": "active"}


@pytest.mark.asyncio
async def test_subscription_current_handles_int_plan():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _BackendWithIntPlan()},
    )
    res = await tools["subscription.current"].run({}, ctx)
    assert not res.is_error
    assert "pro" in res.content
    assert res.data and res.data["subscription"]["plan"] == 7


@pytest.mark.asyncio
async def test_subscription_upgrade_tool():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _FakeBackend()},
    )
    res = await tools["subscription.upgrade"].run({"plan_id": 3, "months": 6}, ctx)
    assert not res.is_error
    assert "updated" in res.content.lower()
    assert res.data and res.data["subscription_upgrade"]["new_plan"] == "plan-3"


@pytest.mark.asyncio
async def test_payment_link_create_tool():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _FakeBackend()},
    )
    res = await tools["payment.link.create"].run({"method": "click", "amount": 50000}, ctx)
    assert not res.is_error
    assert "https://pay.local/click/50000" in res.content
    assert res.data and res.data["payment"]["method"] == "click"


@pytest.mark.asyncio
async def test_release_logs_tool_returns_snapshot():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": backend},
    )
    res = await tools["release.logs"].run({"release_id": 42, "tail": 100}, ctx)
    assert not res.is_error
    assert "engine-test-pod" in res.content
    assert res.data and res.data["logs"]["lines"][0].startswith("2026-05-28")


@pytest.mark.asyncio
async def test_release_logs_empty():
    class _EmptyBackend(_FakeBackend):
        async def get_release_logs(self, release_id, *, tail=200, since_seconds=None, auth_claims=None):
            return {"pod_name": None, "lines": [], "truncated": False, "detail": "Service is not deployed"}

    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _EmptyBackend()},
    )
    res = await tools["release.logs"].run({"release_id": 1}, ctx)
    assert not res.is_error
    assert "not deployed" in res.content.lower() or "no log" in res.content.lower()


@pytest.mark.asyncio
async def test_release_list_rejects_global_project_context():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id="global",
        user_id="u",
        state={"project_id": "global"},
        services={"backend": backend},
    )

    res = await tools["release.list"].run({}, ctx)

    assert res.is_error
    assert "No project_id available" in res.content
    assert backend.list_release_project_ids == []


@pytest.mark.asyncio
async def test_release_status_uses_release_id_without_project_context():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": backend},
    )

    res = await tools["release.status"].run({"release_id": 7}, ctx)

    assert not res.is_error
    assert "running" in res.content
    assert backend.get_release_ids == [7]
    assert backend.list_release_project_ids == []


class _BackendWithDebugLogs(_FakeBackend):
    async def get_debug_logs(self, project_id, *, tail=200, since_seconds=None, auth_claims=None):
        self.debug_logs_project = project_id
        return {
            "pod_name": "debug-pod-1",
            "lines": ["debug line 1", "debug line 2"],
            "truncated": False,
        }


@pytest.mark.asyncio
async def test_release_debug_logs_tool():
    tools = _tools()
    backend = _BackendWithDebugLogs()
    ctx = ToolContext(
        run_id="r",
        project_id="42",
        user_id="u",
        services={"backend": backend},
    )
    res = await tools["release.debug_logs"].run({}, ctx)
    assert not res.is_error
    assert "debug-pod-1" in res.content
    assert "debug line 2" in res.content
    assert backend.debug_logs_project == "42"


@pytest.mark.asyncio
async def test_release_debug_logs_no_project():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _BackendWithDebugLogs()},
    )
    res = await tools["release.debug_logs"].run({}, ctx)
    assert res.is_error
