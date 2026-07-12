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
async def test_tool_unwraps_stringified_dict_with_invalid_apostrophe_escape():
    """OpenRouter models escape apostrophes as \\' inside a stringified JSON
    field — INVALID JSON. Uzbek text is full of apostrophes (to'lov, ko'ring),
    so this must still coerce instead of failing 'not a valid dictionary'."""
    reg = ToolRegistry([nested_tool])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    # data value contains a backslash-apostrophe (invalid JSON escape).
    bad = '{"k": "to\\\'lov ko\\\'ring"}'
    res = await reg.execute("t.nested", {"name": "z", "data": bad}, ctx)
    assert not res.is_error, res.content
    assert res.content == "z:to'lov ko'ring:0"


@pytest.mark.asyncio
async def test_tool_does_not_unwrap_genuine_strings():
    """Oddiy string field (str-typed) hech qachon parse qilinmaydi."""
    reg = ToolRegistry([add])
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    # `n` is int, not a JSON candidate — should fail normally with type error.
    res = await reg.execute("t.add", {"n": "not a number"}, ctx)
    assert res.is_error


# ── surface flags: sleep is agent-only (not exposed to MCP) ──────────────────


def test_sleep_tool_is_agent_only():
    from agent.mcp.server import exported_tools
    from agent.tools import build_default_registry, build_main_registry

    sleep = build_default_registry().get("sleep")
    assert sleep is not None
    assert sleep.expose_to_agent is True
    assert sleep.expose_to_mcp is False

    # In the agent registry, absent from the MCP surface.
    assert build_main_registry().get("sleep") is not None
    assert "sleep" not in exported_tools(build_default_registry()).values()


def test_surface_flags_default_and_reverse():
    """Defaults expose to both; the reverse (mcp-only) flag is honoured."""
    class _In2(BaseModel):
        x: int

    @tool(name="both", description="d", input_model=_In2)
    async def _both(input: _In2, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("ok")

    @tool(name="mcp.only", description="d", input_model=_In2, expose_to_agent=False)
    async def _mcp_only(input: _In2, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("ok")

    assert _both.expose_to_agent and _both.expose_to_mcp  # default both
    assert _mcp_only.expose_to_mcp and not _mcp_only.expose_to_agent


# ── lazy tools: core always, rest via tools.search ───────────────────────────


def test_core_and_lazy_partition():
    from agent.tools import build_default_registry
    from agent.tools.lazy import active_tool_specs, is_core

    tools = build_default_registry().all()
    assert is_core("flow.add_node") and is_core("plan.set") and is_core("tools.search")
    assert not is_core("broadcast.send") and not is_core("flow.simulate")
    # With nothing activated, only CORE tools are sent.
    names = {s.name for s in active_tool_specs(tools, set())}
    assert "flow.add_node" in names and "broadcast.send" not in names
    # After activation, the lazy tool appears.
    names2 = {s.name for s in active_tool_specs(tools, {"broadcast.send"})}
    assert "broadcast.send" in names2


@pytest.mark.asyncio
async def test_tools_search_activates_lazy_tools():
    from agent.tools import build_default_registry

    reg = build_default_registry()
    search = reg.get("tools.search")
    assert search is not None and search.expose_to_mcp is False
    ctx = ToolContext(run_id="r", project_id="1", user_id="u")
    res = await search.run({"query": "broadcast"}, ctx)
    assert not res.is_error
    activated = res.state_updates.get("activated_tools") or []
    assert "broadcast.send" in activated
