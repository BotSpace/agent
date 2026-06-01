"""Tests for per-node Pydantic data validators."""
import pytest

from agent.flow.catalog import NODE_TYPES
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools
from agent.tools.flow.node_data_models import (
    NODE_DATA_MODELS,
    render_node_schema,
    render_summary_table,
)


def _tools():
    return {t.name: t for t in build_flow_tools()}


def test_every_engine_node_has_data_model():
    """Each NodeType in the catalog must have a corresponding Pydantic model."""
    for type_name in NODE_TYPES:
        assert type_name in NODE_DATA_MODELS, f"{type_name} missing a data model"


def test_data_models_forbid_extra_fields():
    """Unknown fields must be rejected by the per-node validator."""
    Model = NODE_DATA_MODELS["SendTextMessageNode"]
    with pytest.raises(Exception):  # pydantic ValidationError
        Model.model_validate({"messageText": "hi", "unknown_field": 1})


def test_data_models_reject_wrong_enum():
    Model = NODE_DATA_MODELS["HTTPRequestNode"]
    with pytest.raises(Exception):
        Model.model_validate({"url": "https://x", "method": "INVALID"})


def test_data_models_accept_alias_for_reserved_word():
    """`global` is a Python keyword — field is `global_` with alias='global'."""
    Model = NODE_DATA_MODELS["CommandTriggerNode"]
    inst = Model.model_validate({"command": "/start", "global": True})
    dumped = inst.model_dump(by_alias=True)
    assert dumped["global"] is True
    assert "global_" not in dumped


@pytest.mark.asyncio
async def test_add_node_reports_unknown_field():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.add_node"].run(
        {
            "id": "x",
            "type": "SendTextMessageNode",
            "data": {"messageText": "hi", "typo_field": 1},
        },
        ctx,
    )
    assert res.is_error
    assert "typo_field" in res.content.lower() or "extra" in res.content.lower()


@pytest.mark.asyncio
async def test_add_node_reports_missing_required():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.add_node"].run(
        {"id": "x", "type": "SendTextMessageNode", "data": {}},
        ctx,
    )
    assert res.is_error
    assert "messageText" in res.content


@pytest.mark.asyncio
async def test_add_node_reports_wrong_enum():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.add_node"].run(
        {
            "id": "h",
            "type": "HTTPRequestNode",
            "data": {"url": "https://api.test", "method": "NOPE"},
        },
        ctx,
    )
    assert res.is_error
    assert "method" in res.content.lower()


@pytest.mark.asyncio
async def test_add_node_accepts_valid_data():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.add_node"].run(
        {
            "id": "delay-1",
            "type": "DelayNode",
            "data": {"seconds": 3},
        },
        ctx,
    )
    assert not res.is_error, res.content


def test_collection_tool_data_defaults_and_keys():
    """AICollectionToolNode mirrors the engine/constructor AICollectionToolData contract."""
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    inst = Model.model_validate({"collection": "students"})
    dumped = inst.model_dump()
    assert dumped["tool"] == "collection"
    assert dumped["toolName"] == "collection"
    assert dumped["collection"] == "students"
    assert dumped["operations"] == ["create", "read", "update", "delete"]
    assert dumped["searchFields"] is None


def test_collection_tool_data_operations_allow_list():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    inst = Model.model_validate(
        {
            "collection": "orders",
            "operations": ["read"],
            "searchFields": [{"name": "email", "type": "string"}],
        }
    )
    dumped = inst.model_dump()
    assert dumped["operations"] == ["read"]
    assert dumped["searchFields"] == [{"name": "email", "type": "string"}]


def test_collection_tool_data_forbids_extra_fields():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({"collection": "x", "operation": "find"})


def test_mongo_tool_node_removed():
    assert "AIMongoToolNode" not in NODE_DATA_MODELS
    assert "AIMongoToolNode" not in NODE_TYPES
    assert "AICollectionToolNode" in NODE_TYPES


def test_render_node_schema_has_types_and_required():
    schema = render_node_schema("CommandTriggerNode")
    assert schema["validator"] is True
    fields_by_name = {f["name"]: f for f in schema["fields"]}
    assert fields_by_name["command"]["required"]
    assert fields_by_name["global"]["required"] is False  # has default


def test_render_summary_table_mentions_all_categories():
    table = render_summary_table()
    for cat in ("trigger", "send", "edit", "flow_control", "data", "integration"):
        assert f"### {cat}" in table
    # Spot-check: required field shown
    assert "messageText" in table
    assert "command" in table
