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
    # Engine (AICollectionToolNode / mongoQueryTool) reads searchFields as
    # [{"field": str, "description"?: str}]. Legacy {"name","type"} keys are
    # NOT understood by the engine, so the model normalises name→field and drops
    # the unknown `type` key. This keeps agent / engine in lockstep (#4).
    inst = Model.model_validate(
        {
            "collection": "orders",
            "operations": ["read"],
            "searchFields": [{"name": "email", "type": "string"}],
        }
    )
    dumped = inst.model_dump()
    assert dumped["operations"] == ["read"]
    assert dumped["searchFields"] == [{"field": "email"}]


def test_collection_tool_data_search_fields_engine_shape():
    """searchFields accepts the engine shape, a string array, and name-alias —
    all normalised to [{"field", "description"?}] (#4)."""
    Model = NODE_DATA_MODELS["AICollectionToolNode"]

    # Canonical engine shape passes through unchanged.
    canon = Model.model_validate(
        {"collection": "c", "searchFields": [{"field": "status", "description": "order status"}]}
    ).model_dump()
    assert canon["searchFields"] == [{"field": "status", "description": "order status"}]

    # Plain string array → coerced to {field}.
    arr = Model.model_validate({"collection": "c", "searchFields": ["chat_id", "kind"]}).model_dump()
    assert arr["searchFields"] == [{"field": "chat_id"}, {"field": "kind"}]


def test_collection_tool_data_forbids_extra_fields():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    # Unknown field rejected (extra='forbid').
    with pytest.raises(Exception):
        Model.model_validate({"collection": "x", "bogusField": 1})
    # `operation` is now a real Literal field, but an invalid value is rejected.
    with pytest.raises(Exception):
        Model.model_validate({"collection": "x", "operation": "find"})


def test_collection_tool_template_read_with_fixed_and_ai_slots():
    """TEMPLATE (hybrid) read: fixed {{chat_id}} slot + ai status slot."""
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    inst = Model.model_validate(
        {
            "collection": "debts",
            "operation": "read",
            "filterMatch": "and",
            "filterConditions": [
                {"field": "user_id", "op": "eq", "source": "fixed", "value": "{{chat_id}}"},
                {
                    "field": "status",
                    "op": "eq",
                    "source": "ai",
                    "aiDescription": "qaysi holatdagi qarzlar (masalan active)",
                },
            ],
        }
    )
    dumped = inst.model_dump()
    assert dumped["operation"] == "read"
    assert dumped["filterMatch"] == "and"
    fixed, ai = dumped["filterConditions"]
    assert fixed["source"] == "fixed" and fixed["value"] == "{{chat_id}}"
    assert ai["source"] == "ai" and ai["aiDescription"]
    # Legacy fields keep their backward-compat defaults.
    assert dumped["operations"] == ["create", "read", "update", "delete"]
    assert dumped["documentFields"] is None


def test_collection_tool_template_create_document_fields():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    inst = Model.model_validate(
        {
            "collection": "debts",
            "operation": "create",
            "documentFields": [
                {"field": "user_id", "source": "fixed", "value": "{{chat_id}}"},
                {"field": "amount", "source": "ai", "aiDescription": "qarz summasi"},
                {"field": "text", "source": "ai", "aiDescription": "qarz izohi"},
            ],
        }
    )
    dumped = inst.model_dump()
    assert dumped["operation"] == "create"
    assert [f["field"] for f in dumped["documentFields"]] == ["user_id", "amount", "text"]
    assert dumped["documentFields"][0]["value"] == "{{chat_id}}"


def test_collection_tool_template_invalid_op_rejected():
    Model = NODE_DATA_MODELS["AICollectionToolNode"]
    with pytest.raises(Exception):
        Model.model_validate(
            {
                "collection": "c",
                "operation": "read",
                "filterConditions": [{"field": "x", "op": "between", "source": "ai"}],
            }
        )


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
    """AIKnowledgeToolNode mirrors the engine/constructor AIKnowledgeToolData contract."""
    from agent.tools.flow.node_data_models import AIKnowledgeToolData

    inst = AIKnowledgeToolData.model_validate({"collection": "products"})
    dumped = inst.model_dump()
    assert dumped["tool"] == "knowledge"
    assert dumped["toolName"] == "knowledge"
    assert dumped["collection"] == "products"
    assert dumped["textFields"] is None
    assert dumped["embeddingProvider"] == "openai"
    assert dumped["embeddingModel"] == "text-embedding-3-small"
    assert dumped["embeddingApiKey"] == ""
    assert dumped["embeddingBaseURL"] == ""
    assert dumped["vectorDbUrl"] == ""
    assert dumped["vectorDbApiKey"] == ""
    assert dumped["topK"] == 5


def test_knowledge_tool_data_custom_fields():
    from agent.tools.flow.node_data_models import AIKnowledgeToolData

    inst = AIKnowledgeToolData.model_validate(
        {
            "collection": "articles",
            "textFields": ["title", "body"],
            "embeddingProvider": "ollama",
            "embeddingModel": "nomic-embed-text",
            "topK": 10,
        }
    )
    dumped = inst.model_dump()
    assert dumped["textFields"] == ["title", "body"]
    assert dumped["embeddingProvider"] == "ollama"
    assert dumped["embeddingModel"] == "nomic-embed-text"
    assert dumped["topK"] == 10


def test_knowledge_tool_data_forbids_extra_fields():
    from agent.tools.flow.node_data_models import AIKnowledgeToolData

    with pytest.raises(Exception):
        AIKnowledgeToolData.model_validate({"collection": "x", "unknown_key": "y"})


def test_knowledge_tool_data_requires_collection():
    from agent.tools.flow.node_data_models import AIKnowledgeToolData

    with pytest.raises(Exception):
        AIKnowledgeToolData.model_validate({})


def test_knowledge_tool_node_in_registry():
    assert "AIKnowledgeToolNode" in NODE_DATA_MODELS
    assert "AIKnowledgeToolNode" in NODE_TYPES


# ── Payment tools ────────────────────────────────────────────────────────────


def test_payment_link_tool_data_keys_and_max_amount_default():
    from agent.tools.flow.node_data_models import AIPaymentLinkToolData

    inst = AIPaymentLinkToolData.model_validate({"paymentType": "click"})
    dumped = inst.model_dump(by_alias=True)
    # Cross-stream keys the engine reads.
    assert dumped["tool"] == "payment_link"
    assert dumped["toolName"] == "payment_link"
    assert dumped["paymentType"] == "click"
    # Injection-guard default ceiling.
    assert dumped["maxAmount"] == 100000000


def test_payment_link_tool_data_accepts_custom_max_amount():
    from agent.tools.flow.node_data_models import AIPaymentLinkToolData

    inst = AIPaymentLinkToolData.model_validate({"paymentType": "paycue", "maxAmount": 50000})
    assert inst.model_dump(by_alias=True)["maxAmount"] == 50000


def test_payment_link_tool_data_rejects_bad_provider():
    from agent.tools.flow.node_data_models import AIPaymentLinkToolData

    with pytest.raises(Exception):
        AIPaymentLinkToolData.model_validate({"paymentType": "payme"})


def test_payment_status_tool_data_keys_and_no_required_config():
    from agent.tools.flow.node_data_models import AIPaymentStatusToolData

    # No build-time config required (order_id is a runtime arg, project from engine).
    inst = AIPaymentStatusToolData.model_validate({})
    dumped = inst.model_dump(by_alias=True)
    assert dumped["tool"] == "payment_status"
    assert dumped["toolName"] == "payment_status"


def test_payment_status_tool_data_forbids_extra_fields():
    from agent.tools.flow.node_data_models import AIPaymentStatusToolData

    with pytest.raises(Exception):
        AIPaymentStatusToolData.model_validate({"unknown_key": "y"})


def test_payment_status_tool_node_in_registry_and_catalog():
    assert "AIPaymentStatusToolNode" in NODE_DATA_MODELS
    assert "AIPaymentStatusToolNode" in NODE_TYPES


def test_payment_status_tool_has_named_mcp_action():
    tools = _tools()
    assert "flow.add_ai_payment_status_tool" in tools
