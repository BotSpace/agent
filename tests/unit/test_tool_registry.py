import pytest
from pydantic import BaseModel

from agentv3.tools.base import ToolContext, ToolResult, tool
from agentv3.tools.registry import ToolRegistry


class _In(BaseModel):
    n: int


@tool(name="t.add", description="echo +1", input_model=_In)
async def add(input: _In, ctx: ToolContext) -> ToolResult:
    return ToolResult.ok(str(input.n + 1), data={"result": input.n + 1})


@pytest.mark.asyncio
async def test_registry_execute_validates_args():
    reg = ToolRegistry([add])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    res = await reg.execute("t.add", {"n": 4}, ctx)
    assert not res.is_error
    assert res.content == "5"
    assert res.data == {"result": 5}


@pytest.mark.asyncio
async def test_registry_rejects_invalid_args():
    reg = ToolRegistry([add])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    res = await reg.execute("t.add", {"n": "not-a-number"}, ctx)
    assert res.is_error


def test_registry_exposes_specs():
    reg = ToolRegistry([add])
    specs = reg.specs()
    assert len(specs) == 1
    assert specs[0].name == "t.add"
    schema = specs[0].input_schema
    assert "properties" in schema and "n" in schema["properties"]
