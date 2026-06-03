"""flow.* tools — flow CRUD, validation, layout, list node types.

Agent state ichida ikkita asosiy maydon ishlatiladi:
  - `working_flow`     : dict   — joriy o'zgartirishlar (agent shu yerda yozadi)
  - `baseline_flow`    : dict   — boshlang'ich (diff uchun)
  - `flow_dirty`       : bool   — saqlanmagan o'zgarishlar bormi

Saqlash (`flow.save`) — backend'ga POST qiladi. Destructive.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pydantic import ValidationError

from ...flow import (
    FlowDocument,
    FlowEdge,
    FlowNode,
    Position,
    all_node_types,
    auto_layout,
    diff,
    get_node_spec,
    validate,
)
from ..base import Tool, ToolContext, ToolResult, tool
from .actions import build_flow_action_tools
from .templates import build_flow_template_tools
from .node_data_models import NODE_DATA_MODELS, get_data_model

# ── State helpers ────────────────────────────────────────────────────────────


def _load_working_flow(ctx: ToolContext) -> FlowDocument:
    raw = ctx.state.get("working_flow")
    if isinstance(raw, dict):
        return FlowDocument.model_validate(raw)
    if isinstance(raw, str) and raw:
        return FlowDocument.from_json(raw)
    return FlowDocument()


def _set_working_flow(flow: FlowDocument) -> dict[str, Any]:
    return {"working_flow": flow.model_dump(), "flow_dirty": True}


def _concrete_project_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "global", "none", "null"}:
        return None
    return text


def _project_id(ctx: ToolContext) -> str | None:
    # Chat sessions may start in a global context, then select/create a project
    # via tools. Save must use that concrete state project, never `/global/`.
    return _concrete_project_id(ctx.state.get("project_id")) or _concrete_project_id(
        ctx.project_id
    )


# ── flow.get ─────────────────────────────────────────────────────────────────


class FlowGetInput(BaseModel):
    summary_only: bool = Field(
        default=True,
        description=(
            "True (default) = return only node count, edge count, type breakdown and trigger ids. "
            "False = return full flow JSON (expensive — only use when you need to inspect a specific "
            "node's data or edge list and flow.diff does not cover it)."
        ),
    )


@tool(
    name="flow.get",
    description=(
        "Inspect the current working flow. "
        "CALL SPARINGLY — the flow state is updated automatically after every "
        "add/update/connect/delete call, so you do NOT need to call this before "
        "each modification. Use summary_only=true (default) for a cheap overview; "
        "summary_only=false only when you must read a specific node's stored data."
    ),
    input_model=FlowGetInput,
)
async def flow_get(input: FlowGetInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    if input.summary_only:
        type_counts: dict[str, int] = {}
        for n in flow.nodes:
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
        node_lines = [f"  {t}×{c}" for t, c in sorted(type_counts.items())]
        summary = {
            "nodes": len(flow.nodes),
            "edges": len(flow.edges),
            "type_counts": type_counts,
            "triggers": [t.id for t in flow.triggers()],
            "node_ids": [n.id for n in flow.nodes],
        }
        text = (
            f"Flow: {len(flow.nodes)} nodes, {len(flow.edges)} edges\n"
            + "\n".join(node_lines)
            + f"\nnode_ids: {[n.id for n in flow.nodes]}"
            + f"\ntriggers: {[t.id for t in flow.triggers()]}"
        )
        return ToolResult.ok(text, data=summary)
    payload = flow.model_dump()
    return ToolResult.ok(flow.to_json(indent=2), data=payload)


# ── flow.add_node (generic, replaces 53 per-type tools) ─────────────────────


class FlowAddNodeInput(BaseModel):
    id: str = Field(description="Stable kebab-case id (re-using = replace).")
    type: str = Field(description="Engine node type (from node-spec table or flow.list_node_types).")
    data: dict[str, Any] = Field(default_factory=dict, description="Type-specific fields.")
    after: str | None = Field(default=None, description="Previous node id; auto-creates an edge.")
    source_handle: str | None = Field(
        default="source-handler",
        description="Default 'source-handler'. Branch: true/false/found/not_found/is-member/loop-body/option_0. Button: target-handler-{nodeId}-{btn.text}.",
    )
    target_handle: str | None = Field(default="target-handler")
    edge_id: str | None = Field(default=None)
    position: dict[str, float] | None = Field(default=None, description="{x,y} — omit, run flow.layout_auto last.")


def _edge_id(source: str, target: str, source_handle: str | None = None) -> str:
    handle = (source_handle or "source-handler").replace(" ", "_")
    return f"e-{source}-{target}-{handle}"


@tool(
    name="flow.add_node",
    description=(
        "Generic add/replace for any node type that lacks a typed flow.add_* "
        "tool (send_photo, delay, cron, for_loop, ...). For required fields see "
        "node-spec table or flow.list_node_types."
    ),
    input_model=FlowAddNodeInput,
)
async def flow_add_node(input: FlowAddNodeInput, ctx: ToolContext) -> ToolResult:
    spec = get_node_spec(input.type)
    if spec is None:
        return ToolResult.error(
            f"Unknown node type `{input.type}`. Use flow.list_node_types to discover types."
        )

    # Per-type strict Pydantic validation (rejects unknown fields, type errors,
    # enum mismatches). Falls back to catalog-based check if no model is registered.
    data_model = get_data_model(input.type)
    if data_model is not None:
        try:
            validated = data_model.model_validate(input.data)
        except ValidationError as e:
            errs = e.errors()
            lines: list[str] = []
            for err in errs[:5]:
                loc = ".".join(str(p) for p in err["loc"]) or "(root)"
                lines.append(f"  - `{loc}` — {err['msg']}")
            extra = f"\n  ...and {len(errs) - 5} more" if len(errs) > 5 else ""
            return ToolResult.error(
                f"Invalid data for `{input.type}`:\n" + "\n".join(lines) + extra
                + f"\nSchema: {sorted(data_model.model_fields.keys())}."
            )
        # Convert back to dict for storage in the flow (drop None, use aliases).
        data_dict = validated.model_dump(by_alias=True, exclude_none=True)
    else:
        # Fallback: catalog-based required + unknown-field check.
        missing = [r for r in spec.required_data if r not in input.data]
        if missing:
            return ToolResult.error(
                f"`{input.type}` requires data fields: {', '.join(missing)}. Provided: {list(input.data)}"
            )
        known = set(spec.required_data) | set(spec.optional_data)
        unknown = [k for k in input.data if k not in known]
        if unknown:
            return ToolResult.error(
                f"Unknown data fields for `{input.type}`: {unknown}. "
                f"Allowed: {sorted(known)}."
            )
        data_dict = dict(input.data)

    flow = _load_working_flow(ctx)

    if input.after and flow.get_node(input.after) is None:
        return ToolResult.error(f"`after` node `{input.after}` does not exist.")

    pos = Position(**input.position) if input.position else Position()
    flow.upsert_node(FlowNode(id=input.id, type=input.type, data=data_dict, position=pos))

    connected_from: str | None = None
    if input.after:
        flow.upsert_edge(
            FlowEdge(
                id=input.edge_id or _edge_id(input.after, input.id, input.source_handle),
                source=input.after,
                target=input.id,
                sourceHandle=input.source_handle,
                targetHandle=input.target_handle,
            )
        )
        connected_from = input.after

    msg = (
        f"Node `{input.id}` ({input.type}) saved"
        + (f", connected from `{connected_from}`." if connected_from else ".")
    )
    return ToolResult.ok(
        msg,
        state_updates=_set_working_flow(flow),
        data={
            "node_id": input.id,
            "type": input.type,
            "connected_from": connected_from,
        },
    )


# ── flow.clear ────────────────────────────────────────────────────────────────


class FlowClearInput(BaseModel):
    confirm: bool = Field(
        default=True,
        description=(
            "Tasdiqlash flag. Foydalanuvchi yangi bot quryapti yoki tozalashni "
            "aniq xohlasagina true qiling."
        ),
    )


@tool(
    name="flow.clear",
    description=(
        "Working flow'dan BARCHA node va edge'larni bir chaqiruvda o'chiradi. "
        "DIQQAT: butun mavjud flow'ni yo'q qiladi — faqat foydalanuvchi AYNAN "
        "'qaytadan boshla / hammasini o'chir / yangidan qur' deb so'rasagina chaqir. "
        "Bu amal foydalanuvchidan APPROVAL (tasdiq) talab qiladi va u rad etsa "
        "bajarilmaydi. Oddiy tahrir uchun ishlatma — node qo'shish/o'zgartirish "
        "uchun flow.add_*/flow.patch ishlat."
    ),
    input_model=FlowClearInput,
    destructive=True,
    always_approve=True,
)
async def flow_clear(input: FlowClearInput, ctx: ToolContext) -> ToolResult:
    if not input.confirm:
        return ToolResult.error(
            "flow.clear destructive — tasdiqlash uchun confirm=true bering."
        )
    flow = _load_working_flow(ctx)
    n_nodes = len(flow.nodes)
    n_edges = len(flow.edges)
    if n_nodes == 0 and n_edges == 0:
        return ToolResult.ok("Flow allaqachon bo'sh.")
    empty = FlowDocument()
    return ToolResult.ok(
        f"Flow tozalandi: {n_nodes} node + {n_edges} edge o'chirildi. "
        f"Endi flow.add_* tool'lari bilan yangi node'lar qo'shing.",
        state_updates={
            "working_flow": empty.model_dump(),
            "flow_dirty": True,
            "last_validation": None,
        },
    )


# ── flow.remove_node ─────────────────────────────────────────────────────────


class FlowRemoveNodeInput(BaseModel):
    id: str


@tool(
    name="flow.remove_node",
    description="Remove a node by id. Also removes all edges connected to it.",
    input_model=FlowRemoveNodeInput,
)
async def flow_remove_node(input: FlowRemoveNodeInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    if flow.get_node(input.id) is None:
        return ToolResult.error(f"Node `{input.id}` does not exist")
    flow.remove_node(input.id)
    return ToolResult.ok(
        f"Removed `{input.id}` and its edges.",
        state_updates=_set_working_flow(flow),
    )


# ── flow.patch ───────────────────────────────────────────────────────────────


class FlowPatchInput(BaseModel):
    id: str = Field(description="Node id to patch")
    data: dict[str, Any] = Field(description="Fields to merge into node.data (shallow)")


@tool(
    name="flow.patch",
    description=(
        "Patch a node's `data` fields without rewriting the entire node. "
        "Useful when changing a single attribute like `messageText` or `keyboard`."
    ),
    input_model=FlowPatchInput,
)
async def flow_patch(input: FlowPatchInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    node = flow.get_node(input.id)
    if node is None:
        return ToolResult.error(f"Node `{input.id}` does not exist")
    node.data = {**node.data, **input.data}
    return ToolResult.ok(
        f"Patched `{input.id}` fields: {', '.join(input.data)}",
        state_updates=_set_working_flow(flow),
    )


# ── flow.connect ─────────────────────────────────────────────────────────────


class FlowConnectInput(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    source_handle: str | None = Field(
        default="source-handler",
        description="For branching nodes (IfCondition, ForLoop, LoadCollection*): branch handle",
    )
    target_handle: str | None = "target-handler"
    edge_id: str | None = Field(default=None, description="Auto-generated if omitted")


@tool(
    name="flow.connect",
    description=(
        "Connect two nodes with an edge. Default source_handle='source-handler'. "
        "For branching nodes use exact engine handles: "
        "IfConditionNode → 'true'/'false' (or 'branch_0','branch_1' for multi-branch), "
        "LoadCollectionItemNode → 'found'/'not_found', "
        "CheckMembershipNode → 'is-member'/'not-member', "
        "ForLoopNode → 'loop-body'/'no-items', "
        "RandomNode → 'option_0'/'option_1'/... "
        "For AI accessory wiring, prefer flow.connect_accessory which validates port types."
    ),
    input_model=FlowConnectInput,
)
async def flow_connect(input: FlowConnectInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    if flow.get_node(input.source) is None:
        return ToolResult.error(f"Source node `{input.source}` does not exist")
    if flow.get_node(input.target) is None:
        return ToolResult.error(f"Target node `{input.target}` does not exist")
    edge_id = input.edge_id or f"e-{input.source}-{input.target}-{input.source_handle or 'src'}"
    flow.upsert_edge(
        FlowEdge(
            id=edge_id,
            source=input.source,
            target=input.target,
            sourceHandle=input.source_handle,
            targetHandle=input.target_handle,
        )
    )
    return ToolResult.ok(
        f"Edge `{input.source}` → `{input.target}` (handle={input.source_handle}) added.",
        state_updates=_set_working_flow(flow),
    )


# ── flow.disconnect ──────────────────────────────────────────────────────────


class FlowDisconnectInput(BaseModel):
    edge_id: str


@tool(
    name="flow.disconnect",
    description="Remove an edge by id.",
    input_model=FlowDisconnectInput,
)
async def flow_disconnect(input: FlowDisconnectInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    flow.remove_edge(input.edge_id)
    return ToolResult.ok(
        f"Edge `{input.edge_id}` removed.",
        state_updates=_set_working_flow(flow),
    )


# ── flow.validate ────────────────────────────────────────────────────────────


class FlowValidateInput(BaseModel):
    pass


@tool(
    name="flow.validate",
    description=(
        "Validate the current working flow — checks schema, required fields, edge handles, "
        "reachability. Always call this before flow.save."
    ),
    input_model=FlowValidateInput,
)
async def flow_validate(_: FlowValidateInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    result = validate(flow)
    summary = {
        "ok": result.ok,
        "errors": [i.message for i in result.errors],
        "warnings": [i.message for i in result.warnings],
    }
    return ToolResult.ok(result.render(), data=summary, state_updates={"last_validation": summary})


# ── flow.layout_auto ─────────────────────────────────────────────────────────


class FlowLayoutInput(BaseModel):
    pass


@tool(
    name="flow.layout_auto",
    description="Auto-position all nodes using a layered layout (left-to-right).",
    input_model=FlowLayoutInput,
)
async def flow_layout_auto(_: FlowLayoutInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    auto_layout(flow)
    return ToolResult.ok("Auto-layout applied.", state_updates=_set_working_flow(flow))


# ── flow.list_node_types ─────────────────────────────────────────────────────


class FlowListNodeTypesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional filter: trigger | send | edit | flow_control | data | integration | interaction | subflow | ai",
    )
    type: str | None = Field(
        default=None,
        description="If set, return only that single node type with full field schema.",
    )


@tool(
    name="flow.list_node_types",
    description=(
        "Return the FULL per-type field/validator schema for one or more node types. "
        "The compact catalog (type names, categories, descriptions, handles) is already "
        "in the system context — you do NOT need to call this tool to discover available "
        "node types. Call it only when you need the detailed field schema before adding "
        "a specific node: `type=<NodeName>` for a single type's full schema (types, "
        "enums, required vs optional). `category=<name>` filters to a category."
    ),
    input_model=FlowListNodeTypesInput,
)
async def flow_list_node_types(input: FlowListNodeTypesInput, ctx: ToolContext) -> ToolResult:
    from .node_data_models import render_node_schema

    specs = all_node_types()
    if input.type:
        specs = [s for s in specs if s.type == input.type]
    elif input.category:
        specs = [s for s in specs if s.category == input.category]

    payload = []
    for s in specs:
        schema = render_node_schema(s.type)
        payload.append(
            {
                "type": s.type,
                "category": s.category,
                "description": s.description,
                "source_handles": s.source_handles,
                "is_trigger": s.is_trigger,
                "fields": schema["fields"],
                "validator": schema["validator"],
            }
        )
    # Group by category so the LLM gets an overview without drowning in schema detail.
    # Full per-type schema (fields + validators) is still in data["types"].
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for p in payload:
        marker = "[trigger]" if p.get("is_trigger") else ""
        by_cat[p["category"]].append(f"{p['type']}{marker}")
    lines = [f"{len(payload)} node type(s) in {len(by_cat)} categories:"]
    for cat, types in sorted(by_cat.items()):
        lines.append(f"  {cat}: {', '.join(types)}")
    if input.type and payload:
        # Single-type lookup: show full schema in content
        import json as _json
        lines.append("")
        lines.append(_json.dumps(payload[0], ensure_ascii=False, indent=2))
    return ToolResult.ok("\n".join(lines), data={"types": payload})


# ── flow.diff ────────────────────────────────────────────────────────────────


class FlowDiffInput(BaseModel):
    pass


@tool(
    name="flow.diff",
    description="Show what has changed in the working flow compared to the baseline (saved version).",
    input_model=FlowDiffInput,
)
async def flow_diff(_: FlowDiffInput, ctx: ToolContext) -> ToolResult:
    baseline_raw = ctx.state.get("baseline_flow") or {}
    baseline = FlowDocument.model_validate(baseline_raw) if baseline_raw else FlowDocument()
    current = _load_working_flow(ctx)
    d = diff(baseline, current)
    return ToolResult.ok(
        d.render(),
        data={
            "added_nodes": d.added_nodes,
            "removed_nodes": d.removed_nodes,
            "modified_nodes": d.modified_nodes,
            "added_edges": d.added_edges,
            "removed_edges": d.removed_edges,
        },
    )


# ── flow.save (DESTRUCTIVE) ──────────────────────────────────────────────────


class FlowSaveInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Optional project id override. Useful if the run started in global context.",
    )
    apply_layout: bool = Field(default=True, description="Run auto-layout before saving.")


@tool(
    name="flow.save",
    description=(
        "Persist the working flow to the backend (project.flow). Triggers approval interrupt "
        "unless `auto_approve` is configured. After save, baseline_flow is updated and "
        "flow_dirty is cleared."
    ),
    input_model=FlowSaveInput,
    destructive=True,
)
async def flow_save(input: FlowSaveInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    result = validate(flow)
    if not result.ok:
        return ToolResult.error(
            "Flow has validation errors — fix them before saving:\n" + result.render()
        )
    if input.apply_layout:
        auto_layout(flow)

    project_id = _concrete_project_id(input.project_id) or _project_id(ctx)
    if not project_id:
        return ToolResult.error(
            "Cannot save flow: no concrete project_id is set. "
            "Use project.create, project.get, or choose a project first."
        )

    from ...integrations.backend import BackendClient, BackendError, get_backend_client

    backend = ctx.services.get("backend")
    if not isinstance(backend, BackendClient):
        backend = get_backend_client()
    claims = ctx.state.get("auth_claims")
    flow_dict = flow.model_dump()
    try:
        await backend.update_project(project_id, {"flow": flow_dict}, auth_claims=claims)
    except BackendError as e:
        from ...integrations.backend import format_backend_error
        # 401 — refresh logikasi backend client'da bo'lishi kerak, lekin agar bu
        # yerga yetib kelgan bo'lsa, refresh muvaffaqiyatsiz tugagan. Agentga
        # aniq sabab beramiz.
        extra = ""
        if e.status == 401:
            has_refresh = bool(claims and (claims.get("refresh_token") or claims.get("refresh")))
            if not has_refresh:
                extra = (
                    "\nFoydalanuvchi refresh_token'i mavjud emas — agent uzoq vaqt "
                    "ishlagan va access muddati tugagan bo'lishi mumkin. Foydalanuvchi "
                    "qaytadan kirishi kerak."
                )
            else:
                extra = (
                    "\nAccess token muddati tugagan, refresh ham muvaffaqiyatsiz "
                    "(refresh ham eskirgan yoki backend rad etdi). Foydalanuvchidan "
                    "qaytadan login qilishni so'rang."
                )
        return ToolResult.error(format_backend_error("Flow saqlash", e) + extra)

    return ToolResult.ok(
        f"Flow saved to project `{project_id}`.",
        state_updates={
            "working_flow": flow_dict,
            "baseline_flow": flow_dict,
            "flow_dirty": False,
            "pending_save": False,
        },
        data={"flow": flow_dict},
    )


def build_flow_tools() -> list[Tool]:
    return [
        flow_get,
        *build_flow_action_tools(),
        *build_flow_template_tools(),
        flow_add_node,
        flow_remove_node,
        flow_clear,
        flow_patch,
        flow_connect,
        flow_disconnect,
        flow_validate,
        flow_layout_auto,
        flow_list_node_types,
        flow_diff,
        flow_save,
    ]
