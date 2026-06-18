"""Production-safe flow action tools.

These tools keep node type selection deterministic while still exposing the
engine's full node catalog. The agent can call `flow.add_send_text_message`
instead of a raw `flow.upsert_node(type=...)`, and higher-level helpers build
common prompt/answer patterns with engine-compatible trigger/state wiring.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...flow import FlowDocument, FlowEdge, FlowNode, Position, auto_layout
from ..base import Tool, ToolContext, ToolResult
from .node_data_models import (
    AIAgentData,
    AIAnthropicProviderData,
    AICollectionToolData,
    AICustomCodeToolData,
    AIGeminiProviderData,
    AIHTTPToolData,
    AIKnowledgeToolData,
    AIOllamaProviderData,
    AIOpenAIProviderData,
    CallbackQueryTriggerData,
    CollectionData,
    CommandTriggerData,
    CustomCodeData,
    EditOrSendMessageData,
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


def _node_missing(node_id: str, flow: FlowDocument, *, role: str = "Node") -> ToolResult:
    """Stale-session-aware 'node not found' error (feedback #1)."""
    if not flow.nodes:
        return ToolResult.error(
            f"`{node_id}` topilmadi — sessiya eskirgan yoki flow bo'sh bo'lishi "
            "mumkin. session_use_project(project_id) bilan loyihani qayta yuklang "
            "yoki flow_get bilan joriy holatni tekshiring."
        )
    return ToolResult.error(
        f"{role} `{node_id}` topilmadi. Mavjud node id'lari: "
        f"{[n.id for n in flow.nodes]}. flow_get(summary_only=true) bilan tekshiring."
    )


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


# ─────────────────────────────────────────────────────────────────────────────
# Button-text → edge-handle remap (#3): when a keyboard button keeps its `id`
# but changes its `text`, the engine's edge sourceHandle (encoded as
# `target-handler-{nodeId}-{text}`) silently breaks. We compare the OLD and NEW
# keyboard of a node, build a map of stable-id → (old_text, new_text), then
# rewrite every outgoing edge sourceHandle on that node to the new text.
# ─────────────────────────────────────────────────────────────────────────────


def _iter_keyboard_buttons(data: dict[str, Any]):
    """Yield every button dict from both the new (`rows`) and legacy
    (`keyboard.{reply,inline}`) keyboard formats stored in a node's `data`."""
    rows_new = data.get("rows")
    if isinstance(rows_new, list):
        for row in rows_new:
            if isinstance(row, dict):
                for btn in row.get("columns") or []:
                    if isinstance(btn, dict):
                        yield btn
            elif isinstance(row, list):  # tolerate bare 2-D rows
                for btn in row:
                    if isinstance(btn, dict):
                        yield btn
    keyboard = data.get("keyboard")
    if isinstance(keyboard, dict):
        for group in ("reply", "inline"):
            for row in keyboard.get(group) or []:
                if isinstance(row, list):
                    for btn in row:
                        if isinstance(btn, dict):
                            yield btn


def _button_text_by_id(data: dict[str, Any]) -> dict[str, str]:
    """{button_id: button_text} for buttons that carry a stable `id`."""
    out: dict[str, str] = {}
    for btn in _iter_keyboard_buttons(data):
        bid = btn.get("id")
        text = btn.get("text")
        if isinstance(bid, str) and bid and isinstance(text, str) and text:
            out[bid] = text
    return out


def remap_button_edge_handles(
    flow: FlowDocument, node_id: str, old_data: dict[str, Any], new_data: dict[str, Any]
) -> int:
    """Remap outgoing button-edge handles after a keyboard text change.

    Buttons matched by stable `id`: if a button kept its id but changed its
    `text`, rewrite every edge whose source == node_id and whose sourceHandle
    was `target-handler-{node_id}-{old_text}` to use the new text. Returns the
    number of edge handles remapped. Backward-compatible, migration-free: only
    same-id/different-text pairs are touched; buttons without ids are ignored.
    """
    old_by_id = _button_text_by_id(old_data)
    new_by_id = _button_text_by_id(new_data)
    # old_handle → new_handle, only where text actually changed.
    renames: dict[str, str] = {}
    for bid, old_text in old_by_id.items():
        new_text = new_by_id.get(bid)
        if new_text is not None and new_text != old_text:
            renames[_button_handle(node_id, old_text)] = _button_handle(node_id, new_text)
    if not renames:
        return 0
    remapped = 0
    for edge in flow.edges:
        if edge.source != node_id:
            continue
        new_handle = renames.get(edge.sourceHandle or "")
        if new_handle is not None:
            edge.sourceHandle = new_handle
            remapped += 1
    return remapped


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

    async def _handler(input: TypedNodeInput, ctx: ToolContext) -> ToolResult:
        flow = _load_working_flow(ctx)
        if input.after and flow.get_node(input.after) is None:
            return _node_missing(input.after, flow, role="`after` node")
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


# Node types whose keyboard buttons route via button-edges
# (`target-handler-{nodeId}-{text}`). Mirrors the engine controller pause list
# and the validator's `_BUTTON_EDGE_NODE_TYPES` — keep in sync.
_BUTTON_EDGE_NODE_TYPES = frozenset(
    {
        "SendTextMessageNode",
        "EditOrSendMessageNode",
        "EditMessageNode",
        "MessageNode",
    }
)


class ConnectButtonInput(BaseModel):
    source: str = Field(description="Id of a keyboard-bearing node (SendTextMessageNode, EditOrSendMessageNode, EditMessageNode, MessageNode) that owns the button.")
    target: str = Field(description="Target node id.")
    button_text: str = Field(description="Exact button text from the source node keyboard.")
    target_handle: str | None = "target-handler"
    edge_id: str | None = None


async def flow_connect_button(input: ConnectButtonInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    source = flow.get_node(input.source)
    if source is None:
        return _node_missing(input.source, flow, role="Source node")
    if source.type not in _BUTTON_EDGE_NODE_TYPES:
        return ToolResult.error(
            f"`{input.source}` is `{source.type}`; button edges start from a "
            f"keyboard-bearing node ({', '.join(sorted(_BUTTON_EDGE_NODE_TYPES))}). "
            "Alternative: use flow.connect(source, target, "
            "source_handle='target-handler-{nodeId}-{buttonText}') directly."
        )
    if flow.get_node(input.target) is None:
        return _node_missing(input.target, flow, role="Target node")
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
        return _node_missing(input.source, flow, role="Source node")
    if flow.get_node(input.target) is None:
        return _node_missing(input.target, flow, role="Target node")
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


# ─────────────────────────────────────────────────────────────────────────────
# Accessory edge helper: connect provider/tool sub-nodes to AIAgentNode ports.
# ─────────────────────────────────────────────────────────────────────────────

_ACC_PORT_ACCEPTS: dict[str, frozenset[str]] = {
    "acc:ai_model": frozenset({
        "AIOpenAIProviderNode", "AIOllamaProviderNode",
        "AIGeminiProviderNode", "AIAnthropicProviderNode",
    }),
    "acc:ai_tool": frozenset({
        "AIHTTPToolNode", "AICollectionToolNode", "AICustomCodeToolNode",
        "AIKnowledgeToolNode",
    }),
}


class ConnectAccessoryInput(BaseModel):
    accessory_id: str = Field(description="Id of the accessory sub-node (provider or tool).")
    agent_id: str = Field(description="Id of the AIAgentNode this sub-node attaches to.")
    port: str = Field(
        description="Target accessory port: 'acc:ai_model' (exactly one provider) or 'acc:ai_tool' (zero or more tools).",
    )
    edge_id: str | None = None


async def flow_connect_accessory(input: ConnectAccessoryInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    acc_node = flow.get_node(input.accessory_id)
    if acc_node is None:
        return _node_missing(input.accessory_id, flow, role="Accessory node")
    agent_node = flow.get_node(input.agent_id)
    if agent_node is None:
        return _node_missing(input.agent_id, flow, role="AIAgentNode")
    if agent_node.type != "AIAgentNode":
        return ToolResult.error(f"`{input.agent_id}` is `{agent_node.type}`, expected AIAgentNode.")
    accepted = _ACC_PORT_ACCEPTS.get(input.port)
    if accepted is None:
        return ToolResult.error(
            f"Unknown accessory port `{input.port}`. Valid ports: {sorted(_ACC_PORT_ACCEPTS)}."
        )
    if acc_node.type not in accepted:
        return ToolResult.error(
            f"Node type `{acc_node.type}` is not accepted on port `{input.port}`. "
            f"Accepted: {sorted(accepted)}."
        )
    eid = input.edge_id or f"e-acc-{input.accessory_id}-{input.agent_id}-{input.port.replace(':', '_')}"
    _connect(
        flow,
        source=input.accessory_id,
        target=input.agent_id,
        source_handle="source-default",
        target_handle=input.port,
        edge_id=eid,
    )
    return ToolResult.ok(
        f"Accessory edge: `{input.accessory_id}` → `{input.agent_id}` on `{input.port}`.",
        state_updates=_set_working_flow(flow),
        data={"source": input.accessory_id, "target": input.agent_id, "port": input.port, "edge_id": eid},
    )


def build_flow_action_tools() -> list[Tool]:
    # The 9 most-used node types keep their own typed tool with strict Pydantic
    # validation (extra='forbid' rejects unknown fields, enums validate values).
    # The other ~46 node types go through the generic `flow.add_node` in
    # `flow/__init__.py`.
    popular_tools: list[Tool] = [
        # ── AI Agent nodes (typed, strict validation) ───────────────────
        _make_popular_node_tool(
            name="flow.add_ai_agent",
            type_name="AIAgentNode",
            data_model=AIAgentData,
            description=(
                "Add an AIAgentNode (LLM agent). After adding, attach a provider with "
                "flow.connect_accessory(port='acc:ai_model') and optionally tools on "
                "'acc:ai_tool'. The agent output is stored in outputKey (default: agent_output). "
                "Use {{agent_output}} in downstream nodes. "
                "For guaranteed JSON, set structuredOutput=true + jsonSchema (schema as a "
                "string) — do NOT ask for JSON in systemPrompt. For real data/actions, attach "
                "tools (collection/http/custom_code/knowledge) instead of relying on plain "
                "text generation. On provider error errorOutputKey is set and the flow "
                "continues (branch on it); on success errorOutputKey is cleared."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_openai_provider",
            type_name="AIOpenAIProviderNode",
            data_model=AIOpenAIProviderData,
            description=(
                "Add an OpenAI provider sub-node. Then connect: "
                "flow.connect_accessory(accessory_id=<this>, agent_id=<agent>, port='acc:ai_model'). "
                "Default model: gpt-4o-mini."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_ollama_provider",
            type_name="AIOllamaProviderNode",
            data_model=AIOllamaProviderData,
            description=(
                "Add an Ollama provider sub-node (local LLM). Connect to acc:ai_model. "
                "Default model: llama3.1, baseURL: http://localhost:11434."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_gemini_provider",
            type_name="AIGeminiProviderNode",
            data_model=AIGeminiProviderData,
            description=(
                "Add a Google Gemini provider sub-node. Connect to acc:ai_model. "
                "Default model: gemini-1.5-flash."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_anthropic_provider",
            type_name="AIAnthropicProviderNode",
            data_model=AIAnthropicProviderData,
            description=(
                "Add an Anthropic (Claude) provider sub-node. Connect to acc:ai_model. "
                "Default model: claude-3-5-haiku-latest."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_http_tool",
            type_name="AIHTTPToolNode",
            data_model=AIHTTPToolData,
            description=(
                "Add an HTTP request tool sub-node for an AIAgentNode (tool='http_request'). "
                "Use for calling an external REST API. Connect to acc:ai_tool. The LLM calls "
                "this tool with its own args. Result is status-aware: on non-2xx the model "
                "sees the status and ok=false, so it can branch on 4xx/5xx. Internal/private "
                "addresses are SSRF-blocked by default; set allowInternal=true or allowedHosts "
                "only to deliberately reach an internal service."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_collection_tool",
            type_name="AICollectionToolNode",
            data_model=AICollectionToolData,
            description=(
                "Add a Collection CRUD tool sub-node for an AIAgentNode (tool='collection'). "
                "Full CRUD on a bot `collection`; restrict via the `operations` allow-list "
                "(create/read/update/delete). Connect to acc:ai_tool. "
                "searchFields (optional) = ARRAY OF OBJECTS, each "
                "{\"field\": \"<name>\", \"description\": \"<hint>\"} (`field` required) — "
                "NOT a string array, keys are `field`/`description` (not name/type)."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_custom_code_tool",
            type_name="AICustomCodeToolNode",
            data_model=AICustomCodeToolData,
            description=(
                "Add a custom JavaScript tool sub-node for an AIAgentNode (tool='custom_code'). "
                "Use only for complex logic/computation or multi-step actions the "
                "collection/http tools can't express. Connect to acc:ai_tool. jsCode return "
                "value becomes the tool result. POWERFUL: full engine access (Mongo write, "
                "Telegram send, http) — write only the needed operation, avoid unintended "
                "destructive DB ops and infinite loops (30s timeout)."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_ai_knowledge_tool",
            type_name="AIKnowledgeToolNode",
            data_model=AIKnowledgeToolData,
            description=(
                "Add a RAG knowledge retrieval tool sub-node for an AIAgentNode "
                "(tool='knowledge'). Use ONLY for semantic (RAG) search over documents. "
                "Embeds the bot `collection` into a vector DB (lazy) and lets the agent "
                "retrieve relevant docs via semantic search; the agent calls it with a "
                "plain-text `query`. Connect to acc:ai_tool after adding. "
                "HEAVY INFRA — requires an EXTERNAL Qdrant URL plus an embedding "
                "provider/key. There is NO built-in vector DB: if vectorDbUrl is empty the "
                "engine fails to build the tool and silently skips it. Do NOT add this tool "
                "unless you have a real Qdrant URL and embedding key; otherwise use the "
                "collection (read) or http tool instead. "
                "Required: collection, vectorDbUrl, embeddingProvider "
                "(openai|ollama|gemini), embeddingModel, embeddingApiKey. "
                "Optional: textFields, embeddingBaseURL, vectorDbApiKey, topK."
            ),
        ),
        # ── Core popular nodes ──────────────────────────────────────────
        _make_popular_node_tool(
            name="flow.add_command_trigger",
            type_name="CommandTriggerNode",
            data_model=CommandTriggerData,
            description="Add a command trigger (/start etc.). state={key,type} captures args.",
        ),
        _make_popular_node_tool(
            name="flow.add_edit_or_send",
            type_name="EditOrSendMessageNode",
            data_model=EditOrSendMessageData,
            description=(
                "PREFERRED message node: edits the previous bot message when possible, "
                "sends a new one otherwise — menus/pages don't pile up in the chat. "
                "messageText ({{var}} ok), inline keyboard via keyboard_kind+rows. "
                "Use flow.add_send_text_message ONLY for reply keyboards "
                "(request_contact/request_location)."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_send_text_message",
            type_name="SendTextMessageNode",
            data_model=SendTextMessageData,
            description=(
                "Send a NEW text message. Use ONLY when a reply keyboard is needed "
                "(request_contact/request_location) or a separate message is explicitly "
                "required — otherwise PREFER flow.add_edit_or_send (edits in place, "
                "keeps the chat clean)."
            ),
        ),
        _make_popular_node_tool(
            name="flow.add_message_trigger",
            type_name="MessageTriggerNode",
            data_model=MessageTriggerData,
            description="Wait for user message of given type. state={key,type} auto-saves. global=false → fires only after Send.",
        ),
        _make_popular_node_tool(
            name="flow.add_callback_query_trigger",
            type_name="CallbackQueryTriggerNode",
            data_model=CallbackQueryTriggerData,
            description="Inline-button callback. filter='starts_with'+context={value:'accept:'} matches prefix.",
        ),
        _make_popular_node_tool(
            name="flow.add_if_condition",
            type_name="IfConditionNode",
            data_model=IfConditionData,
            description="Branch node. Handles: true/false (or branch_0/1). Use flow.connect_branch.",
        ),
        _make_popular_node_tool(
            name="flow.add_collection",
            type_name="CollectionNode",
            data_model=CollectionData,
            description="Insert state into MongoDB. fieldMappings={col:state_key}. {{inserted_id}} after.",
        ),
        _make_popular_node_tool(
            name="flow.add_state",
            type_name="StateNode",
            data_model=StateData,
            description="Save to state. type='Custom'+customType='{{expr}}' stores any expression.",
        ),
        _make_popular_node_tool(
            name="flow.add_custom_code",
            type_name="CustomCodeNode",
            data_model=CustomCodeData,
            description="CustomCodeNode (JS, Goja VM). One node per logical step. outputs=[{name:'ok'}]+exit('ok') for branches. JS API in prompt Pattern 7.",
        ),
        _make_popular_node_tool(
            name="flow.add_http_request",
            type_name="HTTPRequestNode",
            data_model=HTTPRequestData,
            description="HTTP call. url/method required. saveAs writes JSON response to state.",
        ),
    ]
    tools: list[Tool] = list(popular_tools)
    tools += [
            Tool(
                name="flow.connect_accessory",
                description=(
                    "Wire a provider/tool sub-node to an AIAgentNode's accessory port. "
                    "port='acc:ai_model' (one provider, e.g. AIOpenAIProviderNode) or "
                    "'acc:ai_tool' (any number of tool nodes). "
                    "This creates an edge: sourceHandle='source-default' → targetHandle=port."
                ),
                input_model=ConnectAccessoryInput,
                handler=flow_connect_accessory,
                tags=["flow", "ai", "safe-edge"],
            ),
            Tool(
                name="flow.connect_button",
                description="Connect an exact reply/inline keyboard button using target-handler-{nodeId}-{buttonText}. Works for any keyboard-bearing node: SendTextMessageNode, EditOrSendMessageNode, EditMessageNode, MessageNode.",
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
                description=(
                    "Create a safe text question pair: SendTextMessageNode -> "
                    "MessageTriggerNode(global=false,state). Creates nodes "
                    "`<id_prefix>-ask` and `<id_prefix>-capture` — connect follow-ups "
                    "from `<id_prefix>-capture`."
                ),
                input_model=AskTextInput,
                handler=flow_ask_text,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_contact",
                description=(
                    "Create a safe contact question: text with request_contact keyboard "
                    "-> MessageTriggerNode(type=contact,state phone_number). Creates "
                    "`<id_prefix>-ask` and `<id_prefix>-capture` — connect follow-ups "
                    "from `<id_prefix>-capture`."
                ),
                input_model=AskContactInput,
                handler=flow_ask_contact,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_location",
                description=(
                    "Create a safe location question and capture latitude/longitude "
                    "state keys. Creates `<id_prefix>-ask`, "
                    "`<id_prefix>-capture-latitude` and `<id_prefix>-capture-longitude` "
                    "— connect follow-ups from `<id_prefix>-capture-longitude`."
                ),
                input_model=AskLocationInput,
                handler=flow_ask_location,
                tags=["flow", "safe-pattern", "form"],
            ),
            Tool(
                name="flow.ask_buttons",
                description=(
                    "Create a safe button question with reply/inline keyboard and exact "
                    "button-handle edges into a StateNode. Creates `<id_prefix>-ask` and "
                    "`<id_prefix>-capture` — connect follow-ups from `<id_prefix>-capture`."
                ),
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
