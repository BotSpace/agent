"""Tests for per-node Pydantic data validators."""

import pytest

from agent.flow.manifest_compiler.node_catalog import NODE_TYPES
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


def test_string_fields_coerce_numbers():
    """The LLM keeps sending PaymentNode.amount as a NUMBER (1000) while the
    field is a string. The engine reads both, so the validator must coerce
    int/float → str instead of failing with 'Input should be a valid string'."""
    Model = NODE_DATA_MODELS["PaymentNode"]
    inst = Model.model_validate(
        {"amount": 1000, "paymentType": "click", "comment": "x", "buttonText": "Pay"}
    )
    assert inst.amount == "1000"
    # A real string still validates unchanged.
    inst2 = Model.model_validate(
        {"amount": "2000", "paymentType": "click", "comment": "x"}
    )
    assert inst2.amount == "2000"


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


def test_new_runtime_node_models_accept_constructor_shapes():
    NODE_DATA_MODELS["MessageTriggerNode"].model_validate(
        {"type": "voice", "upload_file": True, "file_key": "user_audio", "ttl": 86400}
    )
    NODE_DATA_MODELS["WebhookTriggerNode"].model_validate(
        {"chat_id": "{{webhook.body.user_id}}", "wait_response": True, "response_variable": "response"}
    )
    NODE_DATA_MODELS["DownloadFileNode"].model_validate({"key": "uploaded_file"})
    NODE_DATA_MODELS["SetLanguageNode"].model_validate({"language": "{{callback_data}}"})
    NODE_DATA_MODELS["SubFlowExitNode"].model_validate({"exitName": "success"})


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
    """AICollectionToolNode manifestdan: engine-kontrakt default'lari injektsiya."""
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    dumped = Model.model_validate(
        {"toolName": "debts", "description": "qarz tool", "collection": "debts"}
    ).model_dump(by_alias=True, exclude_none=True)
    assert dumped["tool"] == "collection"  # engine tool identifikatori (defaults)
    assert dumped["toolName"] == "debts"
    assert dumped["collection"] == "debts"
    assert dumped["action"] == "create"  # manifest default


def test_collection_tool_data_forbids_extra_fields():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({"toolName": "x", "description": "d", "bogusField": 1})


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


# ── AIKnowledgeToolNode tests ─────────────────────────────────────────────────


def test_knowledge_tool_data_defaults():
    """AIKnowledgeToolNode manifestdan: embedding default'lari injektsiya."""
    Model = NODE_DATA_MODELS["AIKnowledgeToolNode"]
    dumped = Model.model_validate(
        {"toolName": "kb", "description": "bilim bazasi", "collection": "products"}
    ).model_dump(by_alias=True, exclude_none=True)
    assert dumped["tool"] == "knowledge"
    assert dumped["collection"] == "products"
    assert dumped["embeddingProvider"] == "openai"
    assert dumped["embeddingModel"] == "text-embedding-3-small"
    assert dumped["topK"] == 5


def test_knowledge_tool_data_custom_fields():
    Model = NODE_DATA_MODELS["AIKnowledgeToolNode"]
    dumped = Model.model_validate(
        {
            "toolName": "kb", "description": "d", "collection": "articles",
            "textFields": ["title", "body"], "embeddingProvider": "ollama",
            "embeddingModel": "nomic-embed-text", "topK": 10,
        }
    ).model_dump(by_alias=True, exclude_none=True)
    assert dumped["textFields"] == ["title", "body"]
    assert dumped["embeddingProvider"] == "ollama"
    assert dumped["topK"] == 10


def test_knowledge_tool_data_forbids_extra_fields():
    Model = NODE_DATA_MODELS["AIKnowledgeToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({"toolName": "x", "description": "d", "unknown_key": "y"})


def test_knowledge_tool_data_requires_content_fields():
    """toolName/description — asosiy content field'lar, talab qilinadi."""
    Model = NODE_DATA_MODELS["AIKnowledgeToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({})


def test_knowledge_tool_node_in_registry():
    assert "AIKnowledgeToolNode" in NODE_DATA_MODELS
    assert "AIKnowledgeToolNode" in NODE_TYPES


# ── Payment tools ────────────────────────────────────────────────────────────


def test_payment_link_tool_data_keys_and_max_amount_default():
    """tool/maxAmount manifest defaults'dan; maxAmount xavfsizlik shifti ENGINE'da."""
    Model = NODE_DATA_MODELS["AIPaymentLinkToolNode"]
    dumped = Model.model_validate({"paymentType": "click"}).model_dump(by_alias=True, exclude_none=True)
    assert dumped["tool"] == "payment_link"
    assert dumped["paymentType"] == "click"
    assert dumped["maxAmount"] == 100000000  # manifest default (engine clamp qiladi)


def test_payment_link_tool_data_accepts_custom_max_amount():
    Model = NODE_DATA_MODELS["AIPaymentLinkToolNode"]
    dumped = Model.model_validate({"paymentType": "paycue", "maxAmount": 50000}).model_dump(by_alias=True)
    assert dumped["maxAmount"] == 50000


def test_payment_link_tool_data_rejects_bad_provider():
    Model = NODE_DATA_MODELS["AIPaymentLinkToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({"paymentType": "payme"})


def test_payment_status_tool_data_keys_and_no_required_config():
    Model = NODE_DATA_MODELS["AIPaymentStatusToolNode"]
    dumped = Model.model_validate({}).model_dump(by_alias=True, exclude_none=True)
    assert dumped["tool"] == "payment_status"


def test_payment_status_tool_data_forbids_extra_fields():
    Model = NODE_DATA_MODELS["AIPaymentStatusToolNode"]
    with pytest.raises(Exception):
        Model.model_validate({"unknown_key": "y"})


def test_payment_status_tool_node_in_registry_and_catalog():
    assert "AIPaymentStatusToolNode" in NODE_DATA_MODELS
    assert "AIPaymentStatusToolNode" in NODE_TYPES


def test_payment_status_tool_has_named_mcp_action():
    tools = _tools()
    assert "flow.add_ai_payment_status_tool" in tools
