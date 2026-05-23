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


# ── flow.get ─────────────────────────────────────────────────────────────────


class FlowGetInput(BaseModel):
    summary_only: bool = Field(
        default=False,
        description="If true, return only node/edge counts + types instead of full JSON.",
    )


@tool(
    name="flow.get",
    description=(
        "Return the current working flow (the in-memory version, not the saved one). "
        "Use this before any modification to confirm the current state."
    ),
    input_model=FlowGetInput,
)
async def flow_get(input: FlowGetInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    if input.summary_only:
        type_counts: dict[str, int] = {}
        for n in flow.nodes:
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
        summary = {
            "nodes": len(flow.nodes),
            "edges": len(flow.edges),
            "type_counts": type_counts,
            "triggers": [t.id for t in flow.triggers()],
        }
        return ToolResult.ok(str(summary), data=summary)
    payload = flow.model_dump()
    return ToolResult.ok(flow.to_json(indent=2), data=payload)


# ── flow.upsert_node ─────────────────────────────────────────────────────────


class FlowUpsertNodeInput(BaseModel):
    id: str = Field(description="Node id (stable, kebab-case)")
    type: str = Field(description="Node type from the catalog (see flow.list_node_types)")
    data: dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    position: dict[str, float] | None = Field(
        default=None, description="Optional {x,y}. If omitted, auto-layout will assign."
    )


@tool(
    name="flow.upsert_node",
    description=(
        "Create or replace a node. Use this for every node you add to the flow. "
        "`type` must come from flow.list_node_types. Provide all required `data` fields. "
        "Skip `position` to let auto-layout place it."
    ),
    input_model=FlowUpsertNodeInput,
)
async def flow_upsert_node(input: FlowUpsertNodeInput, ctx: ToolContext) -> ToolResult:
    spec = get_node_spec(input.type)
    if spec is None:
        return ToolResult.error(
            f"Unknown node type `{input.type}`. Use flow.list_node_types to discover types."
        )
    missing = [r for r in spec.required_data if r not in input.data]
    if missing:
        return ToolResult.error(
            f"`{input.type}` requires data fields: {', '.join(missing)}. Provided: {list(input.data)}"
        )

    flow = _load_working_flow(ctx)
    pos = Position(**input.position) if input.position else Position()
    flow.upsert_node(FlowNode(id=input.id, type=input.type, data=input.data, position=pos))
    return ToolResult.ok(
        f"Node `{input.id}` ({input.type}) saved.",
        state_updates=_set_working_flow(flow),
        data={"node_id": input.id, "type": input.type},
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
        "RandomNode → 'option_0'/'option_1'/..."
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
        description="Optional filter: trigger | send | edit | flow_control | data | integration | interaction | subflow",
    )


@tool(
    name="flow.list_node_types",
    description=(
        "List available node types from the engine catalog. Returns each type's name, "
        "category, description, and required data fields."
    ),
    input_model=FlowListNodeTypesInput,
)
async def flow_list_node_types(input: FlowListNodeTypesInput, ctx: ToolContext) -> ToolResult:
    specs = all_node_types()
    if input.category:
        specs = [s for s in specs if s.category == input.category]
    payload = [
        {
            "type": s.type,
            "category": s.category,
            "description": s.description,
            "required_data": s.required_data,
            "optional_data": s.optional_data,
            "source_handles": s.source_handles,
            "is_trigger": s.is_trigger,
        }
        for s in specs
    ]
    return ToolResult.ok(f"{len(payload)} node type(s)", data={"types": payload})


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

    project_id = ctx.project_id
    if not project_id:
        return ToolResult.error(
            "Cannot save flow: no project_id is set. "
            "Use project.create or project.get first."
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
        return ToolResult.error(f"Backend save failed: {e}")

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
        flow_upsert_node,
        flow_remove_node,
        flow_patch,
        flow_connect,
        flow_disconnect,
        flow_validate,
        flow_layout_auto,
        flow_list_node_types,
        flow_diff,
        flow_save,
    ]
