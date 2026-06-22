"""Tests for external MODULE node-type awareness (teaching change).

Modules contribute per-project node types `moduleId.NodeName` (e.g.
`openrouter.Chat`) loaded into ctx.state["module_nodes"]. flow.add_node, the
validator, flow.list_node_types and the credential tooling must treat them as
first-class — without these the agent silently rejects every module node.
"""
from __future__ import annotations

import pytest

from agent.flow.schema import FlowDocument, FlowNode
from agent.flow.validator import validate
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools


def _tools():
    return {t.name: t for t in build_flow_tools()}


# A normalized manifest as produced by backend.load_project_module_nodes.
_OPENROUTER_CHAT = {
    "type": "openrouter.Chat",
    "module": "openrouter",
    "module_name": "OpenRouter",
    "category": "ai",
    "description": "OpenRouter LLM chat",
    "is_trigger": False,
    "trigger_mode": None,
    "required_fields": ["prompt"],
    "optional_fields": ["model", "system", "temperature"],
    "credential_fields": [{"key": "api_credential", "type_key": "openrouter.apikey"}],
    "produces_state": ["llm_output", "llm_error"],
}

_ON_KEYWORD_TRIGGER = {
    "type": "mymodule.OnKeyword",
    "module": "mymodule",
    "module_name": "My Module",
    "category": "triggers",
    "description": "Keyword trigger",
    "is_trigger": True,
    "trigger_mode": "event-match",
    "required_fields": ["keyword"],
    "optional_fields": ["global"],
    "credential_fields": [],
    "produces_state": [],
}


@pytest.mark.asyncio
async def test_add_module_node_accepted_with_credential():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state["module_nodes"] = [_OPENROUTER_CHAT]

    res = await tools["flow.add_node"].run(
        {
            "id": "chat1",
            "type": "openrouter.Chat",
            "data": {
                "api_credential": 7,
                "model": "openai/gpt-4o-mini",
                "prompt": "{{message.text}}",
            },
        },
        ctx,
    )
    assert not res.is_error, res.content
    assert res.data["module"] == "openrouter"
    ctx.state.update(res.state_updates)
    flow = FlowDocument.model_validate(ctx.state["working_flow"])
    node = flow.get_node("chat1")
    assert node is not None and node.type == "openrouter.Chat"
    assert node.data["api_credential"] == 7


@pytest.mark.asyncio
async def test_add_module_node_unknown_without_session_list():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    # No module_nodes in state → module type must be rejected as unknown.
    res = await tools["flow.add_node"].run(
        {"id": "x", "type": "openrouter.Chat", "data": {"prompt": "hi"}}, ctx
    )
    assert res.is_error
    assert "Unknown node type" in res.content


@pytest.mark.asyncio
async def test_add_module_node_missing_required_field():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state["module_nodes"] = [_OPENROUTER_CHAT]
    res = await tools["flow.add_node"].run(
        {"id": "c", "type": "openrouter.Chat", "data": {"api_credential": 1}}, ctx
    )
    assert res.is_error
    assert "prompt" in res.content


@pytest.mark.asyncio
async def test_add_module_node_missing_credential_blocks():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state["module_nodes"] = [_OPENROUTER_CHAT]
    res = await tools["flow.add_node"].run(
        {"id": "c", "type": "openrouter.Chat", "data": {"prompt": "hi"}}, ctx
    )
    assert res.is_error
    assert "openrouter.apikey" in res.content


def test_validator_accepts_module_node():
    flow = FlowDocument(
        nodes=[
            FlowNode(
                id="t",
                type="mymodule.OnKeyword",
                data={"keyword": "hi", "global": True},
            ),
            FlowNode(
                id="c",
                type="openrouter.Chat",
                data={"api_credential": 1, "prompt": "{{keyword}}"},
            ),
        ],
        edges=[],
    )
    # Without module_nodes: both flagged unknown_node_type.
    res_blind = validate(flow)
    codes_blind = {i.code for i in res_blind.issues}
    assert "unknown_node_type" in codes_blind

    # With module_nodes: no unknown_node_type, module trigger satisfies "no_trigger".
    res = validate(flow, module_nodes=[_OPENROUTER_CHAT, _ON_KEYWORD_TRIGGER])
    codes = {i.code for i in res.issues}
    assert "unknown_node_type" not in codes
    assert "no_trigger" not in codes


def test_validator_module_required_field_error():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="c", type="openrouter.Chat", data={"api_credential": 1}),
        ],
        edges=[],
    )
    res = validate(flow, module_nodes=[_OPENROUTER_CHAT])
    msgs = " ".join(i.message for i in res.issues)
    assert "openrouter.Chat" in msgs and "prompt" in msgs


@pytest.mark.asyncio
async def test_list_node_types_includes_module_nodes():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state["module_nodes"] = [_OPENROUTER_CHAT, _ON_KEYWORD_TRIGGER]
    res = await tools["flow.list_node_types"].run({}, ctx)
    assert not res.is_error
    types = {t["type"] for t in res.data["types"]}
    assert "openrouter.Chat" in types
    assert "mymodule.OnKeyword" in types
    chat = next(t for t in res.data["types"] if t["type"] == "openrouter.Chat")
    assert chat.get("module") == "openrouter"
    # credential field surfaced with its type_key
    cred_fields = [
        f for f in chat["fields"] if f.get("type") == "credential"
    ]
    assert cred_fields and cred_fields[0]["credential_type_key"] == "openrouter.apikey"
