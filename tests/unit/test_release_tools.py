import pytest

from agent.tools.base import ToolContext
from agent.tools.release import build_release_tools


class _FakeBackend:
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

