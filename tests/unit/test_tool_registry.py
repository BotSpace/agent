import pytest
from pydantic import BaseModel

from agent.tools.base import ToolContext, ToolResult, tool
from agent.tools.registry import ToolRegistry


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


class _NestedIn(BaseModel):
    name: str
    data: dict
    items: list[dict] | None = None


@tool(name="t.nested", description="nested fields", input_model=_NestedIn)
async def nested_tool(input: _NestedIn, ctx: ToolContext) -> ToolResult:
    return ToolResult.ok(
        f"{input.name}:{input.data.get('k')}:{len(input.items or [])}"
    )


@pytest.mark.asyncio
async def test_tool_unwraps_stringified_dict_field():
    """Qwen/Gemma stilida — dict field string sifatida kelsa, parse qilinadi."""
    reg = ToolRegistry([nested_tool])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    res = await reg.execute(
        "t.nested",
        {"name": "x", "data": '{"k": "v"}'},
        ctx,
    )
    assert not res.is_error
    assert res.content == "x:v:0"


@pytest.mark.asyncio
async def test_tool_unwraps_stringified_list_field():
    reg = ToolRegistry([nested_tool])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    res = await reg.execute(
        "t.nested",
        {
            "name": "y",
            "data": '{"k": "v"}',
            "items": '[{"a": 1}, {"a": 2}]',
        },
        ctx,
    )
    assert not res.is_error
    assert res.content == "y:v:2"


@pytest.mark.asyncio
async def test_tool_does_not_unwrap_genuine_strings():
    """Oddiy string field (str-typed) hech qachon parse qilinmaydi."""
    reg = ToolRegistry([add])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    # `n` is int, not a JSON candidate — should fail normally with type error.
    res = await reg.execute("t.add", {"n": "not a number"}, ctx)
    assert res.is_error
