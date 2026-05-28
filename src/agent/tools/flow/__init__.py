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


# ── flow.add_node (generic, replaces 53 per-type tools) ─────────────────────


class FlowAddNodeInput(BaseModel):
    id: str = Field(description="Stable kebab-case node id. Same id replaces existing node (idempotent).")
    type: str = Field(
        description=(
            "Engine node type from flow.list_node_types. Generic fallback — for the 9 "
            "most-used types use the named tools (flow.add_send_text_message, "
            "flow.add_command_trigger, etc.) which have strict typed data validation."
        )
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific configuration. See flow.list_node_types for required/optional fields per type.",
    )
    after: str | None = Field(
        default=None,
        description="Optional previous node id. If set, auto-create an edge from `after` to this node.",
    )
    source_handle: str | None = Field(
        default="source-handler",
        description=(
            "Source handle on the `after` node. 'source-handler' (default) for normal flow; "
            "branch handles like 'true'/'false', 'found'/'not_found', 'is-member'/'not-member', "
            "'loop-body'/'no-items', 'option_0' for branching nodes; "
            "'target-handler-{nodeId}-{button.text}' for keyboard buttons."
        ),
    )
    target_handle: str | None = Field(default="target-handler")
    edge_id: str | None = Field(
        default=None,
        description="Optional explicit edge id (auto-generated if omitted).",
    )
    position: dict[str, float] | None = Field(
        default=None,
        description="Optional {x,y}. Omit and call flow.layout_auto at the end.",
    )


def _edge_id(source: str, target: str, source_handle: str | None = None) -> str:
    handle = (source_handle or "source-handler").replace(" ", "_")
    return f"e-{source}-{target}-{handle}"


@tool(
    name="flow.add_node",
    description=(
        "Generic add-or-replace node for ANY engine type that does not have its own "
        "named flow.add_* tool. Use the named tools first (they have strict typed "
        "validation). Use this generic tool for rare types like send_photo, send_audio, "
        "delay, cron, for_loop, edit_message, etc. Call flow.list_node_types(category) "
        "to discover required `data` fields."
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
    type: str | None = Field(
        default=None,
        description="If set, return only that single node type with full field schema.",
    )


@tool(
    name="flow.list_node_types",
    description=(
        "Return engine node catalog with strict per-type field schemas (types, enums, "
        "required vs optional). Use `type=<NodeName>` to get the full schema for a "
        "single type (recommended when adding a node). `category=<name>` filters."
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


# ── flow.design + flow.bulk_apply (planner→executor split) ────────────────────


class FlowDesignInput(BaseModel):
    task: str = Field(
        description=(
            "User-facing task: what the bot should do. Be specific (commands, "
            "form fields, branches, integrations). Planner reasons over this."
        )
    )


@tool(
    name="flow.design",
    description=(
        "Strong planner LLM (planner role) bilan to'liq flow blueprint tuzadi: "
        "nodes + edges ro'yxati JSON tarzida. Bitta LLM chaqiruvi — recursion "
        "yo'q. Natija `state.flow_blueprint` ga yoziladi. Keyin `flow.bulk_apply` "
        "bilan bir chaqiruvda qo'llanadi. Bittadan `flow.add_node` chaqirishdan "
        "OLDIN birinchi shu tool'ni chaqir."
    ),
    input_model=FlowDesignInput,
)
async def flow_design(input: FlowDesignInput, ctx: ToolContext) -> ToolResult:
    from ...config import get_settings
    from ...llm import GenerateOptions, Message, get_client
    from .blueprint import FlowBlueprint, parse_blueprint
    from .node_data_models import render_summary_table

    s = get_settings()
    llm = get_client()

    current_flow = ctx.state.get("working_flow")
    current_summary = ""
    if isinstance(current_flow, dict):
        nodes = current_flow.get("nodes") or []
        edges = current_flow.get("edges") or []
        current_summary = (
            f"\n\n## Joriy flow (o'zgartirish kerak bo'lsa)\n"
            f"{len(nodes)} node, {len(edges)} edge.\n"
            f"Node id'lari: {[n.get('id') for n in nodes][:30]}"
        )

    catalog = render_summary_table()
    system_prompt = f"""\
Sen — Botmother **Flow Architect**. Vazifang: foydalanuvchi tavsifidan to'liq
visual bot flow blueprint'ini yaratish. Natija sifatida **faqat JSON** qaytar
(``` kod-blok ichida).

## JSON sxema
```json
{{
  "summary": "qisqa tavsif",
  "nodes": [
    {{"id": "start", "type": "CommandTriggerNode", "data": {{"command": "/start"}}}},
    {{"id": "welcome", "type": "SendTextMessageNode", "data": {{"messageText": "Salom!"}}}}
  ],
  "edges": [
    {{"source": "start", "target": "welcome"}}
  ]
}}
```

## Qoidalar
1. **Har trigger node yagona kirish** — `CommandTriggerNode` /start uchun, va h.k.
2. Reply keyboard menyu: `SendTextMessageNode` ichida `keyboard.reply` belgila;
   har tugma uchun edge: `source_handle: "target-handler-{{nodeId}}-{{button text}}"`.
3. Form (ma'lumot yig'ish): har savol = `SendTextMessageNode` + `MessageTriggerNode`
   ketma-ketligi. Trigger'da `state: {{key, type}}` qo'sh — avtomatik saqlanadi.
4. Collection'ga yozish: `CollectionNode` bilan `fieldMappings`.
5. Branching: `IfConditionNode` uchun `source_handle: "true"` yoki `"false"`.
6. Expression syntax: `{{{{state.var}}}}`, `{{{{first_name}}}}`, `{{{{inserted_id}}}}`.
7. Node id'lari **kebab-case**, takrorlanmasin.
8. **JSON'dan boshqa hech narsa qaytarma** — kommentariy yoki tushuntirish yo'q.

## Engine node kataligi (qisqa)
{catalog}
"""

    user_prompt = f"Task: {input.task}{current_summary}\n\nQuyidagi JSON sxemada blueprint yarat."

    try:
        result = await llm.generate(
            [Message.system_text(system_prompt), Message.user_text(user_prompt)],
            GenerateOptions(
                model=s.model_planner,
                max_tokens=8000,
                temperature=0.1,
            ),
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult.error(f"Planner LLM xatosi: {e}")

    try:
        bp: FlowBlueprint = parse_blueprint(result.message.text)
    except ValueError as e:
        return ToolResult.error(
            f"Blueprint parse qilinmadi: {e}\nPlanner javobining boshi:\n{result.message.text[:300]}"
        )

    bp_dict = bp.model_dump()
    return ToolResult.ok(
        f"Blueprint tayyor: {len(bp.nodes)} node, {len(bp.edges)} edge. "
        f"Endi `flow.bulk_apply` chaqir.",
        data={
            "blueprint": bp_dict,
            "summary": bp.summary or "",
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        state_updates={"flow_blueprint": bp_dict},
    )


class FlowBulkApplyInput(BaseModel):
    nodes: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Nodes ro'yxati [{id, type, data}, ...]. Berilmasa, "
            "state.flow_blueprint dan olinadi (`flow.design` natijasi)."
        ),
    )
    edges: list[dict[str, Any]] | None = Field(
        default=None,
        description="Edges ro'yxati [{source, target, source_handle?, target_handle?}, ...].",
    )
    validate_flow: bool = Field(
        default=True, description="Apply'dan keyin flow.validate chaqirilsinmi?",
    )
    layout: bool = Field(
        default=True, description="layout_auto chaqirilsinmi?",
    )
    save: bool = Field(
        default=False, description="Backend'ga save qilinsinmi (destructive).",
    )


@tool(
    name="flow.bulk_apply",
    description=(
        "Bir chaqiruvda barcha nodes/edges ni working flow'ga qo'llaydi: "
        "per-type Pydantic validatorlar bilan, ixtiyoriy validate + layout + save. "
        "Argumentlar berilmasa state.flow_blueprint dan o'qiydi (flow.design natijasi). "
        "Bittadan `flow.add_node` o'rniga shu tool'ni ishlat — 30+ qadam bitta "
        "chaqiruvga aylanadi."
    ),
    input_model=FlowBulkApplyInput,
    destructive=False,
)
async def flow_bulk_apply(input: FlowBulkApplyInput, ctx: ToolContext) -> ToolResult:
    from .node_data_models import get_data_model

    # State'dan blueprint'ni o'qish
    nodes_raw = input.nodes
    edges_raw = input.edges
    if nodes_raw is None or edges_raw is None:
        bp = ctx.state.get("flow_blueprint")
        if isinstance(bp, dict):
            if nodes_raw is None:
                nodes_raw = bp.get("nodes") or []
            if edges_raw is None:
                edges_raw = bp.get("edges") or []

    nodes_raw = nodes_raw or []
    edges_raw = edges_raw or []

    if not nodes_raw:
        return ToolResult.error(
            "Hech qanday node yo'q. Avval `flow.design` chaqir yoki nodes argument ber."
        )

    # Per-node validation
    errors: list[str] = []
    validated_nodes: list[tuple[str, str, dict[str, Any]]] = []
    for i, n in enumerate(nodes_raw):
        try:
            nid = n["id"]
            ntype = n["type"]
            ndata = n.get("data") or {}
        except (KeyError, TypeError):
            errors.append(f"node[{i}]: id/type yo'q")
            continue
        spec = get_node_spec(ntype)
        if spec is None:
            errors.append(f"node[{i}] `{nid}`: noma'lum type `{ntype}`")
            continue
        data_model = get_data_model(ntype)
        if data_model is not None:
            try:
                validated = data_model.model_validate(ndata)
                ndata = validated.model_dump(by_alias=True, exclude_none=True)
            except Exception as e:  # noqa: BLE001
                errors.append(f"node[{i}] `{nid}` ({ntype}): {e}")
                continue
        else:
            missing = [r for r in spec.required_data if r not in ndata]
            if missing:
                errors.append(
                    f"node[{i}] `{nid}` ({ntype}) majburiy field yo'q: {missing}"
                )
                continue
        validated_nodes.append((nid, ntype, ndata))

    if errors:
        return ToolResult.error(
            "Blueprint xatolar:\n" + "\n".join(f"  - {e}" for e in errors[:10])
            + (f"\n  ...va yana {len(errors) - 10}" if len(errors) > 10 else "")
        )

    # Qo'llaymiz
    flow = _load_working_flow(ctx)
    node_ids_set = {n.id for n in flow.nodes}
    for nid, ntype, ndata in validated_nodes:
        flow.upsert_node(FlowNode(id=nid, type=ntype, data=ndata, position=Position()))
        node_ids_set.add(nid)

    # Edges
    edge_errors: list[str] = []
    applied_edges = 0
    for i, e in enumerate(edges_raw):
        if not isinstance(e, dict):
            edge_errors.append(f"edge[{i}]: not a dict")
            continue
        src = e.get("source")
        tgt = e.get("target")
        if not src or not tgt:
            edge_errors.append(f"edge[{i}]: source/target yo'q")
            continue
        if src not in node_ids_set or tgt not in node_ids_set:
            edge_errors.append(
                f"edge[{i}] `{src}→{tgt}`: source yoki target node mavjud emas"
            )
            continue
        sh = e.get("source_handle") or "source-handler"
        th = e.get("target_handle") or "target-handler"
        flow.upsert_edge(
            FlowEdge(
                id=e.get("id") or _edge_id(src, tgt, sh),
                source=src,
                target=tgt,
                sourceHandle=sh,
                targetHandle=th,
            )
        )
        applied_edges += 1

    updates = _set_working_flow(flow)

    # Optional validate
    val_status = ""
    if input.validate_flow:
        result = validate(flow)
        if not result.ok:
            return ToolResult.error(
                f"Apply qilindi ({len(validated_nodes)} node, {applied_edges} edge), "
                f"lekin VALIDATE muvaffaqiyatsiz: {result.errors[:5]}"
            )
        val_status = " · validate OK"

    if input.layout:
        auto_layout(flow)
        updates = _set_working_flow(flow)

    # Optional save
    save_status = ""
    if input.save:
        save_result = await flow_save.run({}, ctx)
        if save_result.is_error:
            return ToolResult.error(
                f"Apply + validate o'tdi, save xatosi: {save_result.content}"
            )
        save_status = " · saved"
        if save_result.state_updates:
            updates.update(save_result.state_updates)

    msg = (
        f"Apply: {len(validated_nodes)} node, {applied_edges} edge"
        f"{val_status}{save_status}."
    )
    if edge_errors:
        msg += f" (Edge ogohlantirishlari: {len(edge_errors)})"
    return ToolResult.ok(
        msg,
        data={
            "nodes_applied": len(validated_nodes),
            "edges_applied": applied_edges,
            "edge_warnings": edge_errors[:10],
        },
        state_updates=updates,
    )


def build_flow_tools() -> list[Tool]:
    return [
        flow_get,
        *build_flow_action_tools(),
        flow_add_node,
        flow_remove_node,
        flow_patch,
        flow_connect,
        flow_disconnect,
        flow_validate,
        flow_layout_auto,
        flow_list_node_types,
        flow_diff,
        flow_save,
        flow_design,
        flow_bulk_apply,
    ]
