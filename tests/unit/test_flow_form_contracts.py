from __future__ import annotations

import pytest

from agent.flow import FlowDocument, validate
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools
from agent.tools.flow.node_data_models import get_data_model


def _tools():
    return {tool.name: tool for tool in build_flow_tools()}


def _ctx() -> ToolContext:
    return ToolContext(
        run_id="r",
        project_id="p",
        user_id="u",
        state={"working_flow": {"nodes": [], "edges": []}},
        services={},
    )


@pytest.mark.asyncio
async def test_ai_agent_accepts_json_schema_object_and_stores_string() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    result = await _tools()["flow.add_node"].run(
        {
            "id": "agent",
            "type": "AIAgentNode",
            "data": {
                "systemPrompt": "Return a structured answer.",
                "userInputText": "{{message.text}}",
                "outputKey": "agent_output",
                "jsonSchema": schema,
            },
        },
        _ctx(),
    )

    assert not result.is_error, result.content
    node = result.state_updates["working_flow"]["nodes"][0]
    assert isinstance(node["data"]["jsonSchema"], str)
    assert '"type":"object"' in node["data"]["jsonSchema"]
    assert '"required":["answer"]' in node["data"]["jsonSchema"]


def test_ai_agent_schema_contract_exposes_object_or_string() -> None:
    model = get_data_model("AIAgentNode")
    assert model is not None
    schema = model.model_json_schema()
    json_schema = schema["properties"]["jsonSchema"]
    variants = json_schema.get("anyOf", [])
    variant_types = {variant.get("type") for variant in variants}

    assert {"string", "object"} <= variant_types


@pytest.mark.asyncio
async def test_ask_buttons_builds_canonical_collection_callback() -> None:
    result = await _tools()["flow.ask_buttons"].run(
        {
            "id_prefix": "course",
            "question": "Kursni tanlang",
            "state_key": "course_id",
            "style": "inline",
            "state_type": "collection_id",
            "collection_name": "courses",
            "options": [{"text": "Matematika", "value": "507f1f77bcf86cd799439011"}],
        },
        _ctx(),
    )

    assert not result.is_error, result.content
    flow = result.state_updates["working_flow"]
    ask = next(node for node in flow["nodes"] if node["id"] == "course-ask")
    capture = next(node for node in flow["nodes"] if node["id"] == "course-capture")
    assert ask["data"]["keyboard"]["inline"][0][0]["value"] == ("collection:courses:507f1f77bcf86cd799439011")
    assert capture["data"]["type"] == "collection_id"


@pytest.mark.asyncio
async def test_save_to_collection_normalizes_runtime_context_mappings() -> None:
    result = await _tools()["flow.save_to_collection"].run(
        {
            "id": "save",
            "collection_name": "enrollments",
            "field_mappings": {
                "full_name": "full_name",
                "user_id": "user_id",
                "chat_id": "chat_id",
            },
        },
        _ctx(),
    )

    assert not result.is_error, result.content
    save = next(node for node in result.state_updates["working_flow"]["nodes"] if node["id"] == "save")
    assert save["data"]["fieldMappings"] == {
        "full_name": "full_name",
        "user_id": "{{user_id}}",
        "chat_id": "{{chat_id}}",
    }


def test_collection_mapping_is_a_consumer_not_a_state_producer() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "enrollments",
                        "fieldMappings": {"name": "missing_name"},
                    },
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "save"}],
        }
    )

    result = validate(flow)
    assert any(issue.code == "unresolved_collection_mapping_source" for issue in result.issues)


def test_unknown_template_mapping_is_an_error() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "enrollments",
                        "fieldMappings": {"name": "{{missing_name}}"},
                    },
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "save"}],
        }
    )

    result = validate(flow)
    assert not result.ok
    assert any(issue.code == "unknown_collection_mapping_variable" for issue in result.errors)


def test_custom_code_rejects_async_await() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "load",
                    "type": "CustomCodeNode",
                    "data": {"jsCode": "const rows = await db_find('courses', {});"},
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "load"}],
        }
    )

    result = validate(flow)
    assert any(issue.code == "custom_code_async_unsupported" for issue in result.errors)


def test_collection_trigger_requires_canonical_callback_prefix() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "pick",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "filter": "starts_with",
                        "context": {"value": "course:"},
                        "state_key": "selected_course",
                        "state_type": "full_data",
                    },
                }
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "invalid_collection_callback_prefix" for issue in result.errors)


def test_nested_collection_mapping_must_be_a_template() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "pick",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "filter": "starts_with",
                        "context": {"value": "collection:courses:"},
                        "state_key": "selected_course",
                        "state_type": "full_data",
                    },
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "registrations",
                        "fieldMappings": {"course_id": "selected_course._id"},
                    },
                },
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "unwrapped_collection_mapping_source" for issue in result.errors)


def test_static_keyboard_rejects_collection_row_templates() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "courses",
                    "type": "EditOrSendMessageNode",
                    "data": {
                        "messageText": "Kursni tanlang",
                        "rows": [
                            {
                                "columns": [
                                    {
                                        "text": "{{name}}",
                                        "type": "callback",
                                        "value": "course:{{_id}}",
                                    }
                                ]
                            }
                        ],
                    },
                },
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "dynamic_keyboard_template_unsupported" for issue in result.errors)


def test_collection_data_keyboard_allows_row_templates() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "courses",
                    "type": "EditOrSendMessageNode",
                    "data": {
                        "messageText": "Kursni tanlang",
                        "rows": [
                            {
                                "columns": [
                                    {
                                        "collectionData": {
                                            "collection": "courses",
                                            "field": "name",
                                            "text": "{{name}} — {{price}}",
                                            "callbackDataPattern": "collection:courses:{id}",
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                },
                {
                    "id": "pick",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "global": True,
                        "filter": "starts_with",
                        "context": {"value": "collection:courses:"},
                        "state_key": "selected_course",
                        "state_type": "full_data",
                    },
                },
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert not any(issue.code == "dynamic_keyboard_template_unsupported" for issue in result.errors)
    assert not any(issue.code == "invalid_collection_data_text" for issue in result.errors)


def test_collection_data_rejects_single_brace_field_placeholders() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "courses",
                    "type": "EditOrSendMessageNode",
                    "data": {
                        "messageText": "Kursni tanlang",
                        "rows": [
                            {
                                "columns": [
                                    {
                                        "collectionData": {
                                            "collection": "courses",
                                            "text": "{emoji} {name}",
                                            "callbackDataPattern": "collection:courses:{id}",
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                }
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "invalid_collection_data_text" for issue in result.errors)


def test_collection_data_requires_global_matching_trigger() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "courses",
                    "type": "EditOrSendMessageNode",
                    "data": {
                        "messageText": "Kursni tanlang",
                        "rows": [
                            {
                                "columns": [
                                    {
                                        "collectionData": {
                                            "collection": "courses",
                                            "field": "name",
                                            "text": "{{name}}",
                                            "callbackDataPattern": "collection:courses:{id}",
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                },
                {
                    "id": "pick",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "global": False,
                        "filter": "starts_with",
                        "context": {"value": "collection:courses:"},
                        "state_key": "selected_course",
                        "state_type": "full_data",
                    },
                },
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "collection_data_trigger_must_be_global" for issue in result.errors)


def test_any_message_trigger_cannot_capture_contact_phone() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "phone",
                    "type": "MessageTriggerNode",
                    "data": {
                        "type": "any",
                        "filter": "any",
                        "global": False,
                        "state_key": "phone",
                        "state_type": "phone_number",
                    },
                }
            ],
            "edges": [],
        }
    )

    result = validate(flow)
    assert any(issue.code == "incompatible_trigger_state_type" for issue in result.errors)
