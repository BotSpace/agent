"""Production-safe flow action tools.

These tools keep node type selection deterministic while still exposing the
engine's full node catalog. The agent can call `flow.add_send_text_message`
instead of a raw `flow.upsert_node(type=...)`, and higher-level helpers build
common prompt/answer patterns with engine-compatible trigger/state wiring.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...flow import FlowDocument, FlowEdge, FlowNode, Position, auto_layout, get_node_spec
from ...flow.catalog import NODE_TYPES, NodeTypeSpec
from ..base import Tool, ToolContext, ToolResult
from .node_data_models import (
    CallbackQueryTriggerData,
    CollectionData,
    CommandTriggerData,
    CustomCodeData,
    HTTPRequestData,
    IfConditionData,
    MessageTriggerData,
    SendTextMessageData,
    StateData,
)


def _load_working_flow(ctx: ToolContext) -> FlowDocument:
    raw = ctx.state.get("working_flow")
    if isinstance(raw, dict):
        return FlowDocument.model_validate(raw)
    if isinstance(raw, str) and raw:
        return FlowDocument.from_json(raw)
    return FlowDocument()


def _set_working_flow(flow: FlowDocument) -> dict[str, Any]:
    return {"working_flow": flow.model_dump(), "flow_dirty": True}


def _snake_node_name(type_name: str) -> str:
    name = type_name.removesuffix("Node")
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    return "_".join(part.lower() for part in parts)


def _edge_id(source: str, target: str, source_handle: str | None = None) -> str:
    handle = (source_handle or "source-handler").replace(" ", "_")
    return f"e-{source}-{target}-{handle}"


def _connect(
    flow: FlowDocument,
    *,
    source: str | None,
    target: str,
    source_handle: str | None = "source-handler",
    target_handle: str | None = "target-handler",
    edge_id: str | None = None,
) -> None:
    if not source:
        return
    flow.upsert_edge(
        FlowEdge(
            id=edge_id or _edge_id(source, target, source_handle),
            source=source,
            target=target,
            sourceHandle=source_handle,
            targetHandle=target_handle,
        )
    )


def _button_handle(source: str, text: str) -> str:
    return f"target-handler-{source}-{text}"


def _node(flow: FlowDocument, node_id: str, type_name: str, data: dict[str, Any], position: dict[str, float] | None = None) -> None:
    flow.upsert_node(
        FlowNode(
            id=node_id,
            type=type_name,
            data=data,
            position=Position(**position) if position else Position(),
        )
    )


# NOTE: The old per-type `flow.add_*` family (53 tools, ~18K tokens) was
# replaced. Now:
#   - 9 POPULAR nodes have their own typed flow.add_* tool below (strict
#     Pydantic validation per node type — wrong/extra fields rejected).
#   - The remaining ~46 node types go through generic `flow.add_node`
#     (`flow/__init__.py`) which checks required fields + unknown fields.


def _ok_node(node_id: str, type_name: str, connected_from: str | None = None) -> str:
    if connected_from:
        return f"Added `{node_id}` ({type_name}) and connected `{connected_from}` -> `{node_id}`."
    return f"Added `{node_id}` ({type_name})."


# ─────────────────────────────────────────────────────────────────────────────
# Popular-node tool factory (data models are in node_data_models.py).
# ─────────────────────────────────────────────────────────────────────────────


def _make_popular_node_tool(
    *,
    name: str,
    type_name: str,
    data_model: type[BaseModel],
    description: str,
) -> Tool:
    """Build a flow.add_<X> tool with strict typed data validation."""

    class TypedNodeInput(BaseModel):
        id: str = Field(description="Stable kebab-case node id.")
        data: data_model  # type: ignore[valid-type]
        after: str | None = Field(
            default=None,
            description="Previous node id to auto-connect from (optional).",
        )
        source_handle: str | None = Field(
            default="source-handler",
            description="Source handle on the `after` node (use branch handles like 'true'/'false' if applicable).",
        )
        target_handle: str | None = "target-handler"
        edge_id: str | None = None
        position: dict[str, float] | None = None

    async def _handler(input: TypedNodeInput, ctx: ToolContext) -> ToolResult:  # noqa: ARG001
        flow = _load_working_flow(ctx)
        if input.after and flow.get_node(input.after) is None:
            return ToolResult.error(f"`after` node `{input.after}` does not exist.")
        data_dict = input.data.model_dump(by_alias=True, exclude_none=True)
        _node(flow, input.id, type_name, data_dict, input.position)
        _connect(
            flow,
            source=input.after,
            target=input.id,
            source_handle=input.source_handle,
            target_handle=input.target_handle,
            edge_id=input.edge_id,
        )
        return ToolResult.ok(
            _ok_node(input.id, type_name, input.after),
            state_updates=_set_working_flow(flow),
            data={"node_id": input.id, "type": type_name, "connected_from": input.after},
        )

    return Tool(
        name=name,
        description=description,
        input_model=TypedNodeInput,
        handler=_handler,
        tags=["flow", "popular-node"],
    )


class ConnectButtonInput(BaseModel):
    source: str = Field(description="SendTextMessageNode id that owns the keyboard button.")
    target: str = Field(description="Target node id.")
    button_text: str = Field(description="Exact button text from the source node keyboard.")
    target_handle: str | None = "target-handler"
    edge_id: str | None = None


async def flow_connect_button(input: ConnectButtonInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    source = flow.get_node(input.source)
    if source is None:
        return ToolResult.error(f"Source node `{input.source}` does not exist.")
    if source.type != "SendTextMessageNode":
        return ToolResult.error("Button edges must start from a SendTextMessageNode.")
    if flow.get_node(input.target) is None:
        return ToolResult.error(f"Target node `{input.target}` does not exist.")
    handle = _button_handle(input.source, input.button_text)
    _connect(flow, source=input.source, target=input.target, source_handle=handle, target_handle=input.target_handle, edge_id=input.edge_id)
    return ToolResult.ok(
        f"Button `{input.button_text}` connected `{input.source}` -> `{input.target}`.",
        state_updates=_set_working_flow(flow),
        data={"source_handle": handle},
    )


class ConnectBranchInput(BaseModel):
    source: str
    target: str
    handle: str = Field(description="Exact engine source handle, e.g. true, false, found, not_found, is-member, option_0.")
    target_handle: str | None = "target-handler"
    edge_id: str | None = None


async def flow_connect_branch(input: ConnectBranchInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    if flow.get_node(input.source) is None:
        return ToolResult.error(f"Source node `{input.source}` does not exist.")
    if flow.get_node(input.target) is None:
        return ToolResult.error(f"Target node `{input.target}` does not exist.")
    _connect(flow, source=input.source, target=input.target, source_handle=input.handle, target_handle=input.target_handle, edge_id=input.edge_id)
    return ToolResult.ok(
        f"Branch `{input.handle}` connected `{input.source}` -> `{input.target}`.",
        state_updates=_set_working_flow(flow),
        data={"source_handle": input.handle},
    )


class AskTextInput(BaseModel):
    id_prefix: str = Field(description="Stable prefix, e.g. full-name.")
    question: str
    state_key: str = Field(description="State key to save the answer under.")
    after: str | None = None
    message_type: Literal["text", "any"] = "text"
    state_type: str = "text"


async def flow_ask_text(input: AskTextInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    ask_id = f"{input.id_prefix}-ask"
    capture_id = f"{input.id_prefix}-capture"
    _node(flow, ask_id, "SendTextMessageNode", {"messageText": input.question})
    _node(
        flow,
        capture_id,
        "MessageTriggerNode",
        {
            "type": input.message_type,
            "filter": "any",
            "global": False,
            "state": {"key": input.state_key, "type": input.state_type},
        },
    )
    _connect(flow, source=input.after, target=ask_id)
    _connect(flow, source=ask_id, target=capture_id)
    return ToolResult.ok(
        f"Text question `{input.state_key}` added.",
        state_updates=_set_working_flow(flow),
        data={"ask_node_id": ask_id, "capture_node_id": capture_id, "state_key": input.state_key},
    )


class AskContactInput(BaseModel):
    id_prefix: str
    question: str
    state_key: str = "phone"
    button_text: str = "Telefon yuborish"
    after: str | None = None


async def flow_ask_contact(input: AskContactInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    ask_id = f"{input.id_prefix}-ask"
    capture_id = f"{input.id_prefix}-capture"
    _node(
        flow,
        ask_id,
        "SendTextMessageNode",
        {
            "messageText": input.question,
            "keyboard": {
                "active": "reply",
                "reply": [[{"id": f"{input.id_prefix}-contact", "text": input.button_text, "type": "request_contact"}]],
            },
        },
    )
    _node(
        flow,
        capture_id,
        "MessageTriggerNode",
        {
            "type": "contact",
            "filter": "any",
            "global": False,
            "state": {"key": input.state_key, "type": "phone_number"},
        },
    )
    _connect(flow, source=input.after, target=ask_id)
    _connect(flow, source=ask_id, target=capture_id)
    return ToolResult.ok(
        f"Contact question `{input.state_key}` added.",
        state_updates=_set_working_flow(flow),
        data={"ask_node_id": ask_id, "capture_node_id": capture_id, "state_key": input.state_key},
    )


class AskLocationInput(BaseModel):
    id_prefix: str
    question: str
    latitude_key: str = "latitude"
    longitude_key: str = "longitude"
    button_text: str = "Lokatsiya yuborish"
    after: str | None = None


async def flow_ask_location(input: AskLocationInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    ask_id = f"{input.id_prefix}-ask"
    lat_id = f"{input.id_prefix}-capture-latitude"
    lon_id = f"{input.id_prefix}-capture-longitude"
    _node(
        flow,
        ask_id,
        "SendTextMessageNode",
        {
            "messageText": input.question,
            "keyboard": {
                "active": "reply",
                "reply": [[{"id": f"{input.id_prefix}-location", "text": input.button_text, "type": "request_location"}]],
            },
        },
    )
    _node(flow, lat_id, "MessageTriggerNode", {"type": "location", "filter": "any", "global": False, "state": {"key": input.latitude_key, "type": "latitude"}})
    _node(flow, lon_id, "StateNode", {"key": input.longitude_key, "type": "longitude"})
    _connect(flow, source=input.after, target=ask_id)
    _connect(flow, source=ask_id, target=lat_id)
    _connect(flow, source=lat_id, target=lon_id)
    return ToolResult.ok(
        "Location question added.",
        state_updates=_set_working_flow(flow),
        data={"ask_node_id": ask_id, "latitude_node_id": lat_id, "longitude_node_id": lon_id},
    )


class ButtonOption(BaseModel):
    text: str
    value: str | None = None


class AskButtonsInput(BaseModel):
    id_prefix: str
    question: str
    state_key: str
    options: list[ButtonOption] = Field(min_length=1)
    after: str | None = None
    style: Literal["reply", "inline"] = "reply"
    columns: int = Field(default=2, ge=1, le=4)


async def flow_ask_buttons(input: AskButtonsInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    ask_id = f"{input.id_prefix}-ask"
    capture_id = f"{input.id_prefix}-capture"
    rows: list[list[dict[str, Any]]] = []
    for idx in range(0, len(input.options), input.columns):
        row: list[dict[str, Any]] = []
        for option in input.options[idx : idx + input.columns]:
            if input.style == "inline":
                row.append({"text": option.text, "type": "callback", "value": option.value or option.text})
            else:
                row.append({"id": f"{input.id_prefix}-{_slug(option.text)}", "text": option.text, "type": "reply"})
        rows.append(row)

    keyboard = {"active": input.style, input.style: rows}
    _node(flow, ask_id, "SendTextMessageNode", {"messageText": input.question, "keyboard": keyboard})
    _node(
        flow,
        capture_id,
        "StateNode",
        {"key": input.state_key, "type": "data" if input.style == "inline" else "text"},
    )
    _connect(flow, source=input.after, target=ask_id)
    for option in input.options:
        _connect(flow, source=ask_id, target=capture_id, source_handle=_button_handle(ask_id, option.text))
    return ToolResult.ok(
        f"Button question `{input.state_key}` added with {len(input.options)} option(s).",
        state_updates=_set_working_flow(flow),
        data={"ask_node_id": ask_id, "capture_node_id": capture_id, "state_key": input.state_key},
    )


class SaveToCollectionInput(BaseModel):
    id: str = "save"
    collection_name: str
    field_mappings: dict[str, str] = Field(description="Database field -> state key or template string.")
    after: str | None = None


async def flow_save_to_collection(input: SaveToCollectionInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    _node(flow, input.id, "CollectionNode", {"collection_name": input.collection_name, "fieldMappings": input.field_mappings})
    _connect(flow, source=input.after, target=input.id)
    return ToolResult.ok(
        f"Save-to-collection node `{input.id}` added.",
        state_updates=_set_working_flow(flow),
        data={"node_id": input.id, "collection_name": input.collection_name},
    )


class FinishInput(BaseModel):
    id: str = "finish"
    message: str
    after: str | None = None


async def flow_finish(input: FinishInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    _node(flow, input.id, "SendTextMessageNode", {"messageText": input.message})
    _connect(flow, source=input.after, target=input.id)
    auto_layout(flow)
    return ToolResult.ok(
        f"Finish message `{input.id}` added.",
        state_updates=_set_working_flow(flow),
        data={"node_id": input.id},
    )


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "option"


def build_flow_action_tools() -> list[Tool]:
    # The 9 most-used node types keep their own typed tool with strict Pydantic
    # validation (extra='forbid' rejects unknown fields, enums validate values).
    # The other ~46 node types go through the generic `flow.add_node` in
    # `flow/__init__.py`.
    popular_tools: list[Tool] = [
        _make_popular_node_tool(
            name="flow.add_command_trigger",
            type_name="CommandTriggerNode",
            data_model=CommandTriggerData,
            description=(
                "Add a command trigger like /start, /help. Required: command. "
                "Use `state={key,type}` to auto-capture command args."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_send_text_message",
            type_name="SendTextMessageNode",
            data_model=SendTextMessageData,
            description=(
                "Send a text message. Required: messageText (supports {{var}}). "
                "Optional: keyboard (reply/inline), recipients (admin group)."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_message_trigger",
            type_name="MessageTriggerNode",
            data_model=MessageTriggerData,
            description=(
                "Wait for a user message of a given media type (text/photo/contact/...). "
                "Use `state={key,type}` to auto-save the value. global=False (default) "
                "means it fires only while engine is paused after a Send node."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_callback_query_trigger",
            type_name="CallbackQueryTriggerNode",
            data_model=CallbackQueryTriggerData,
            description=(
                "Handle inline keyboard callback presses. filter='starts_with' with "
                "context={value:'accept:'} routes only callbacks starting with that prefix."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_if_condition",
            type_name="IfConditionNode",
            data_model=IfConditionData,
            description=(
                "Branching node. Source handles: 'true'/'false' (or 'branch_0'/'branch_1' "
                "for multi-branch). Connect each handle with flow.connect_branch."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_collection",
            type_name="CollectionNode",
            data_model=CollectionData,
            description=(
                "Insert collected state values into a MongoDB collection. "
                "fieldMappings={collection_field: state_key}. After save, "
                "{{inserted_id}} holds the new document id."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_state",
            type_name="StateNode",
            data_model=StateData,
            description=(
                "Save a value to state. Use type='Custom' + customType='{{expr}}' to "
                "store any expression (e.g. {{inserted_id}}, {{message.text}})."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_custom_code",
            type_name="CustomCodeNode",
            data_model=CustomCodeData,
            description=(
                "Add a CustomCodeNode — JavaScript executed by engine-golang Goja VM. "
                "Use one CustomCodeNode per logical step (validation, DB op, "
                "computation, dynamic message composition) and chain them via "
                "named outputs + exit() routing — do NOT put unrelated steps in "
                "a single block, and do NOT fragment trivial 2-line ops into "
                "separate nodes. When the node has multiple downstream branches, "
                "declare them in `outputs: [{name: 'ok'}, {name: 'invalid'}]` and "
                "branch via exit('ok'). Engine routes through the edge with "
                "source_handle='output-<name>'. Single linear next-step: omit "
                "outputs, omit exit(). Available globals: exit, wait_for_input, "
                "set_context/get_context, send_message/send_photo/send_to_chat/"
                "edit_message_text/delete_message/answer_callback_query, "
                "db_find_one/db_find/db_insert/db_update/db_upsert/db_delete/db_count, "
                "http_get/http_post/http_put/http_delete/http_request, date_*/"
                "hash_*/base64_*/uuid/random_int, schedule_message. Auto-vars: "
                "chat_id, message.text, message.contact.phone_number, "
                "callback_query.data, user_first_name, state_*."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_http_request",
            type_name="HTTPRequestNode",
            data_model=HTTPRequestData,
            description=(
                "Make an HTTP call. url/method required. saveAs stores parsed JSON "
                "response into state[saveAs]."
            ),
        ),
    ]
    tools: list[Tool] = list(popular_tools)
    tools += [
            Tool(
                name="flow.connect_button",
                description="Connect an exact reply/inline keyboard button using target-handler-{nodeId}-{buttonText}.",
                input_model=ConnectButtonInput,
                handler=flow_connect_button,
                tags=["flow", "safe-edge"],
            ),
            Tool(
                name="flow.connect_branch",
                description="Connect a branch edge with an exact engine handle such as true, false, found, not_found, is-member, not-member, loop-body, no-items, loop-continue, loop-done, option_0.",
                input_model=ConnectBranchInput,
                handler=flow_connect_branch,
                tags=["flow", "safe-edge"],
            ),
            Tool(
                name="flow.ask_text",
                description="Create a safe text question pair: SendTextMessageNode -> MessageTriggerNode(global=false,state).",
                input_model=AskTextInput,
                handler=flow_ask_text,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_contact",
                description="Create a safe contact question: text with request_contact keyboard -> MessageTriggerNode(type=contact,state phone_number).",
                input_model=AskContactInput,
                handler=flow_ask_contact,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_location",
                description="Create a safe location question and capture latitude/longitude state keys.",
                input_model=AskLocationInput,
                handler=flow_ask_location,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_buttons",
                description="Create a safe button question with reply/inline keyboard and exact button-handle edges into a StateNode.",
                input_model=AskButtonsInput,
                handler=flow_ask_buttons,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.save_to_collection",
                description="Create a CollectionNode with fieldMappings and optional connection from previous node.",
                input_model=SaveToCollectionInput,
                handler=flow_save_to_collection,
                tags=["flow", "safe-pattern", "data"],
            ),
            Tool(
                name="flow.finish",
                description="Create the final SendTextMessageNode, optionally connected from previous node, and auto-layout the flow.",
                input_model=FinishInput,
                handler=flow_finish,
                tags=["flow", "safe-pattern"],
            ),
    ]
    return tools
