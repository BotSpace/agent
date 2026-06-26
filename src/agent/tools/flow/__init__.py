"""flow.* tools — flow CRUD, validation, layout, list node types.

Agent state ichida ikkita asosiy maydon ishlatiladi:
  - `working_flow`     : dict   — joriy o'zgartirishlar (agent shu yerda yozadi)
  - `baseline_flow`    : dict   — boshlang'ich (diff uchun)
  - `flow_dirty`       : bool   — saqlanmagan o'zgarishlar bormi

Saqlash (`flow.save`) — backend'ga POST qiladi. Destructive.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

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
from .actions import build_flow_action_tools, remap_button_edge_handles
from .node_data_models import NODE_DATA_MODELS, get_data_model
from .power import build_flow_power_tools
from .templates import build_flow_template_tools

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


def node_missing_error(node_id: str, flow: FlowDocument, *, role: str = "Node") -> ToolResult:
    """Uniform 'node not found' error (feedback #1).

    When the looked-up node is missing AND the working flow is empty/just
    loaded, the likely cause is a STALE session — point the model at recovery
    (session_use_project / flow_get) instead of a bare 'does not exist'.
    """
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


def _collection_schemas(ctx: ToolContext) -> dict[str, dict[str, str]] | None:
    """{collection_name: {field_name: kind}} from the workspace CMS list (#4).

    None when collections were never loaded (skip the filter-kind cross-check).
    Empty inner dicts are fine — the validator just can't type-check that one."""
    raw = ctx.state.get("collections")
    if not isinstance(raw, list):
        return None
    out: dict[str, dict[str, str]] = {}
    for c in raw:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str) or not name:
            continue
        kinds = c.get("field_kinds")
        out[name] = {
            str(k): str(v)
            for k, v in (kinds.items() if isinstance(kinds, dict) else [])
        }
    return out or None


def _known_collections(ctx: ToolContext) -> list[str] | None:
    """Collection names from the workspace's loaded CMS list (#12). None = the
    list was never loaded, so skip the cross-check entirely."""
    raw = ctx.state.get("collections")
    if not isinstance(raw, list):
        return None
    names: list[str] = []
    for c in raw:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and c["name"]:
            names.append(c["name"])
        elif isinstance(c, str) and c:
            names.append(c)
    return names


# ── Module (external) node types — dynamic, per-project ───────────────────────
#
# External modules contribute extra node types (namespaced `moduleId.NodeName`,
# e.g. `openrouter.Chat`) loaded at session_use_project into ctx.state
# ["module_nodes"]. They are NOT in the static catalog, so flow.add_node and the
# validator must consult this session list as a fallback. Each entry is a
# normalized manifest (see backend.load_project_module_nodes).


def _module_nodes(ctx: ToolContext) -> list[dict[str, Any]]:
    raw = ctx.state.get("module_nodes")
    return raw if isinstance(raw, list) else []


def _module_node_spec(ctx: ToolContext, node_type: str) -> dict[str, Any] | None:
    """Return the normalized manifest for a module node type, or None.

    Used as a fallback when get_node_spec() (static catalog) returns None — a
    namespaced `moduleId.NodeName` type may be a per-project enabled module node.
    """
    if "." not in node_type:
        return None
    for m in _module_nodes(ctx):
        if isinstance(m, dict) and m.get("type") == node_type:
            return m
    return None


def _module_node_types(ctx: ToolContext) -> list[str]:
    return [m["type"] for m in _module_nodes(ctx) if isinstance(m, dict) and m.get("type")]


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


async def _add_module_node(
    input: FlowAddNodeInput, ctx: ToolContext, mod: dict[str, Any]
) -> ToolResult:
    """Add/replace an external MODULE node type (`moduleId.NodeName`).

    Module nodes are validated against the per-project manifest (not a static
    Pydantic model): required non-credential fields must be present, and each
    credential field needs a `<key>` referencing a credential of the manifest's
    `type_key` (engine resolves it at runtime). Data is stored verbatim — the
    engine routes unknown types to the module's `node.execute`.
    """
    data = dict(input.data)
    required = [str(r) for r in (mod.get("required_fields") or [])]
    missing = [r for r in required if r not in data or data.get(r) in (None, "")]
    if missing:
        return ToolResult.error(
            f"Module node `{input.type}` ({mod.get('module')}) requires field(s): "
            f"{', '.join(missing)}. Optional: {mod.get('optional_fields') or '—'}."
        )

    # Credential fields: each must carry a credential reference. The manifest key
    # is the data key (e.g. `api_credential`); its `type_key` (e.g.
    # 'openrouter.apikey') is what credentials.list(type_key=...) must match.
    for cf in mod.get("credential_fields") or []:
        key = cf.get("key")
        if key and not data.get(key):
            return ToolResult.error(
                f"Module node `{input.type}` needs a credential in `{key}` "
                f"(credential type_key `{cf.get('type_key')}`). "
                f"credentials.list(type_key='{cf.get('type_key')}') → put the id "
                f"in data['{key}']. If none exists, create/connect one first."
            )

    flow = _load_working_flow(ctx)
    if input.after and flow.get_node(input.after) is None:
        return ToolResult.error(f"`after` node `{input.after}` does not exist.")

    pos = Position(**input.position) if input.position else Position()
    flow.upsert_node(FlowNode(id=input.id, type=input.type, data=data, position=pos))

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
        f"Module node `{input.id}` ({input.type}, module={mod.get('module')}) saved"
        + (f", connected from `{connected_from}`." if connected_from else ".")
    )
    return ToolResult.ok(
        msg,
        state_updates=_set_working_flow(flow),
        data={
            "node_id": input.id,
            "type": input.type,
            "module": mod.get("module"),
            "connected_from": connected_from,
        },
    )


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
        # Fallback: a namespaced `moduleId.NodeName` may be a per-project module
        # node type (dynamic catalog, loaded at session_use_project). Modules have
        # no static Pydantic model — validate required fields against the manifest
        # and pass data through verbatim.
        mod = _module_node_spec(ctx, input.type)
        if mod is not None:
            return await _add_module_node(input, ctx, mod)
        known_mods = _module_node_types(ctx)
        hint = (
            f" This project's module node types: {', '.join(known_mods)}."
            if known_mods
            else " No module node types are enabled for this project."
        )
        return ToolResult.error(
            f"Unknown node type `{input.type}`. Use flow.list_node_types to discover "
            f"built-in types.{hint}"
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
        return node_missing_error(input.id, flow)
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
        return node_missing_error(input.id, flow)
    old_data = dict(node.data)
    node.data = {**node.data, **input.data}
    # #3 — if the keyboard text changed for a same-id button, the engine's
    # edge sourceHandle (target-handler-{id}-{text}) breaks; auto-remap.
    remapped = remap_button_edge_handles(flow, input.id, old_data, node.data)
    msg = f"Patched `{input.id}` fields: {', '.join(input.data)}"
    if remapped:
        msg += f". {remapped} edge handle(s) remapped to the new button text."
    return ToolResult.ok(msg, state_updates=_set_working_flow(flow))


# ── flow.patch_many (atomic multi-patch) ─────────────────────────────────────


class FlowPatchManyItem(BaseModel):
    id: str = Field(description="Node id to patch")
    data: dict[str, Any] = Field(description="Fields to merge into node.data (shallow)")


class FlowPatchManyInput(BaseModel):
    patches: list[FlowPatchManyItem] = Field(
        min_length=1,
        description="List of {id, data} patches. Applied ATOMICALLY — if any id is "
        "missing, NOTHING is applied.",
    )


@tool(
    name="flow.patch_many",
    description=(
        "Apply MANY node-data patches atomically in one call. Input: "
        "patches=[{id, data}, ...]. Either all patches validate and apply, or "
        "none do (the failing item is reported by index). Keyboard button-text "
        "changes auto-remap their edge handles, same as flow.patch."
    ),
    input_model=FlowPatchManyInput,
)
async def flow_patch_many(input: FlowPatchManyInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    # Pre-flight: every id must exist before we mutate anything (atomicity).
    for idx, item in enumerate(input.patches):
        if flow.get_node(item.id) is None:
            if not flow.nodes:
                return ToolResult.error(
                    f"patches[{idx}] `{item.id}` topilmadi va flow bo'sh — sessiya "
                    "eskirgan bo'lishi mumkin. session_use_project(project_id) bilan "
                    "qayta yuklang, keyin butun batch'ni qayta yuboring. Hech narsa "
                    "qo'llanmadi."
                )
            return ToolResult.error(
                f"patches[{idx}] node `{item.id}` topilmadi — HECH NARSA qo'llanmadi. "
                f"Mavjud id'lar: {[n.id for n in flow.nodes]}. flow_get(summary_only=true) "
                "bilan tekshirib, butun batch'ni qayta yuboring."
            )
    total_remapped = 0
    patched_ids: list[str] = []
    for item in input.patches:
        node = flow.get_node(item.id)
        assert node is not None
        old_data = dict(node.data)
        node.data = {**node.data, **item.data}
        total_remapped += remap_button_edge_handles(flow, item.id, old_data, node.data)
        patched_ids.append(item.id)
    msg = f"Patched {len(patched_ids)} node(s): {', '.join(patched_ids)}"
    if total_remapped:
        msg += f". {total_remapped} edge handle(s) remapped."
    return ToolResult.ok(
        msg,
        state_updates=_set_working_flow(flow),
        data={"patched": patched_ids, "edges_remapped": total_remapped},
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
        return node_missing_error(input.source, flow, role="Source node")
    if flow.get_node(input.target) is None:
        return node_missing_error(input.target, flow, role="Target node")
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
    result = validate(
        flow,
        known_collections=_known_collections(ctx),
        collection_schemas=_collection_schemas(ctx),
        module_nodes=_module_nodes(ctx),
    )
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


# ── flow.set_languages ───────────────────────────────────────────────────────


class FlowSetLanguagesInput(BaseModel):
    languages: list[str] = Field(
        min_length=1,
        description=(
            "Bot tillari — ISO kodlari, masalan ['uz','ru','en']. Engine faqat shu "
            "ro'yxatdagi tilni public_lang / Telegram language_code sifatida tanlaydi — "
            "i18n marker ({\"__i18n\":...}) umuman ishlashi uchun MAJBURIY."
        ),
    )
    default_language: str = Field(
        description=(
            "Asosiy til kodi (languages ichidan). Saqlangan public_lang yoki Telegram "
            "language_code ro'yxatda bo'lmasa shu til ishlatiladi."
        ),
    )


@tool(
    name="flow.set_languages",
    description=(
        "Set the flow's supported languages + default language (top-level "
        "`languages` / `defaultLanguage`). REQUIRED before any i18n marker "
        '({"__i18n": true, "en": "..", "uz": ".."}) works — call it FIRST when '
        "building a multilingual bot. Runtime active language: saved public_lang → "
        "Telegram language_code → defaultLanguage → 'en' (only codes listed in "
        "`languages` are ever picked). See docs_guide('i18n')."
    ),
    input_model=FlowSetLanguagesInput,
)
async def flow_set_languages(input: FlowSetLanguagesInput, ctx: ToolContext) -> ToolResult:
    flow = _load_working_flow(ctx)
    langs = [code.strip() for code in input.languages if code and code.strip()]
    if not langs:
        return ToolResult.error("languages bo'sh bo'lmasin — kamida bitta til kodi bering.")
    default = input.default_language.strip()
    if default and default not in langs:
        langs = [default, *langs]
    default = default or langs[0]
    # FlowDocument `extra='allow'` — top-level languages/defaultLanguage saqlanadi
    # va flow.save (model_dump) orqali saqlangan flow JSON'ga yetib boradi.
    data = flow.model_dump()
    data["languages"] = langs
    data["defaultLanguage"] = default
    return ToolResult.ok(
        f"Tillar o'rnatildi: {langs} (default: {default}). Endi foydalanuvchiga "
        "ko'rinadigan matnlarni i18n marker obyekt bilan yoz.",
        state_updates={"working_flow": data, "flow_dirty": True},
        data={"languages": langs, "defaultLanguage": default},
    )


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

    # External MODULE node types (per-project, dynamic). Built from the session
    # manifest rather than a static Pydantic model. Filtered the same way.
    for mod in _module_nodes(ctx):
        if not isinstance(mod, dict) or not mod.get("type"):
            continue
        if input.type and mod["type"] != input.type:
            continue
        if input.category and mod.get("category") != input.category:
            continue
        fields = [
            {"name": f, "required": True, "source": "module"}
            for f in (mod.get("required_fields") or [])
        ] + [
            {"name": f, "required": False, "source": "module"}
            for f in (mod.get("optional_fields") or [])
        ] + [
            {
                "name": cf.get("key"),
                "required": True,
                "type": "credential",
                "credential_type_key": cf.get("type_key"),
            }
            for cf in (mod.get("credential_fields") or [])
        ]
        payload.append(
            {
                "type": mod["type"],
                "category": mod.get("category") or "integration",
                "description": (
                    f"[MODULE {mod.get('module')}] {mod.get('description')}"
                    + (
                        f"  trigger:{mod.get('trigger_mode')}"
                        if mod.get("is_trigger")
                        else ""
                    )
                ),
                "source_handles": ["source-default"],
                "is_trigger": bool(mod.get("is_trigger")),
                "module": mod.get("module"),
                "fields": fields,
                "validator": None,
            }
        )

    # Group by category so the LLM gets an overview without drowning in schema detail.
    # Full per-type schema (fields + validators) is still in data["types"].
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for p in payload:
        marker = "[trigger]" if p.get("is_trigger") else ""
        mod_marker = "[module]" if p.get("module") else ""
        by_cat[p["category"]].append(f"{p['type']}{marker}{mod_marker}")
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


# ── #10 — regression run after flow.save ─────────────────────────────────────


async def _run_regression_after_save(
    ctx: ToolContext, project_id: str, saved_flow: dict[str, Any]
) -> dict[str, Any] | None:
    """Run the project's enabled saved tests against the just-saved flow.

    Returns {"text", "data"} to append to the save result, or None if there are
    no tests. NEVER raises and NEVER breaks the save — simulator/store outages
    degrade to an informational line. The save is already persisted at this
    point, so a regression FAILURE only warns (rollback/fix is the user's call).
    """
    try:
        from ...mcp.flow_tests import get_flow_test_store
        from .power import _fmt_test_results, _run_saved_tests

        tests = await get_flow_test_store().list(
            project_id=str(project_id), only_enabled=True
        )
        if not tests:
            return None
        # `ok` is the load-bearing flag for callers: True only when every enabled
        # test passed. `ran` distinguishes "didn't run" (simulator down) from
        # "ran and failed" so a save with no working simulator is not treated as a
        # regression failure.
        # Simulate against the flow we just saved (clone ctx state so we don't
        # mutate the live workspace; collections come from the same state).
        sim_state = dict(ctx.state)
        sim_state["working_flow"] = saved_flow
        sim_ctx = ToolContext(
            run_id=ctx.run_id,
            project_id=ctx.project_id,
            user_id=ctx.user_id,
            state=sim_state,
            services=ctx.services,
        )
        agg = await _run_saved_tests(
            sim_ctx, flow=saved_flow, tests=tests, project_id=str(project_id)
        )
        if not agg.get("ran"):
            return {
                "ok": True,  # nothing ran → not a failure
                "ran": False,
                "text": "🧪 Regression: testlar o'tkazilmadi (simulyator mavjud emas).",
                "data": {"ran": False},
            }
        passed = agg["passed"]
        total = agg["total"]
        if passed == total:
            return {
                "ok": True,
                "ran": True,
                "text": f"🧪 Regression: {passed}/{total} o'tdi",
                "data": agg,
            }
        detail = _fmt_test_results(agg)
        return {
            "ok": False,
            "ran": True,
            "text": (
                f"🧪 Regression FAIL: {detail}\n"
                "Tuzatish yoki flow_rollback bilan oldingi versiyaga qaytish mumkin."
            ),
            "data": agg,
        }
    except Exception:  # pragma: no cover — regression must never break save
        return None


# ── flow.save (DESTRUCTIVE) ──────────────────────────────────────────────────


class FlowSaveInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Optional project id override. Useful if the run started in global context.",
    )
    apply_layout: bool = Field(default=True, description="Run auto-layout before saving.")
    verbose: bool = Field(
        default=False,
        description="False (default) returns a compact summary {ok, project_id, nodes, "
        "edges, flow_dirty}. True also returns the full saved flow JSON (~30k tokens) — "
        "only set when you must echo the exact saved document.",
    )
    restart_debug: bool = Field(
        default=True,
        description="Default TRUE: after a successful save the project's DEBUG bot is "
        "automatically (re)started with the new flow and a test link is returned — "
        "the user gets a working bot to try WITHOUT asking. Set false only if the "
        "user explicitly does not want a debug bot.",
    )
    fail_on_regression: bool = Field(
        default=False,
        description="Default FALSE: regression test failures only WARN (the flow is "
        "already persisted, so the save still reports ok=true with regression_ok=false). "
        "Set TRUE to make the tool result an ERROR when any saved regression test fails — "
        "use this when you want the save step to hard-fail so the failure can't be "
        "overlooked. NOTE: the flow is saved either way; this only changes how the "
        "result is reported.",
    )


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
    result = validate(
        flow,
        known_collections=_known_collections(ctx),
        collection_schemas=_collection_schemas(ctx),
        module_nodes=_module_nodes(ctx),
    )
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

    n_nodes = len(flow_dict.get("nodes") or [])
    n_edges = len(flow_dict.get("edges") or [])

    # Versiyalash — har muvaffaqiyatli save yangi versiya yozadi (flow.history /
    # flow.rollback shu store'dan o'qiydi). LangGraph save'lari ham, MCP
    # save'lari ham shu yo'l orqali versiyalanadi. Graceful degrade store ichida;
    # bu guard versiyalash xatosi save natijasiga sizib chiqmasligini ta'minlaydi.
    try:
        from ...mcp.flow_versions import get_flow_version_store

        await get_flow_version_store().append(
            project_id=str(project_id),
            user_id=str(ctx.user_id) if ctx.user_id is not None else None,
            flow=flow_dict,
            node_count=n_nodes,
            edge_count=n_edges,
        )
    except Exception:  # pragma: no cover — defensive
        pass

    # #6 — default to a compact summary so flow.save (and its MCP
    # structuredContent) does NOT echo the full ~30k-token flow back to the
    # model. Full flow only when verbose=true.
    data: dict[str, Any] = {
        "ok": True,
        "project_id": project_id,
        "nodes": n_nodes,
        "edges": n_edges,
        "flow_dirty": False,
    }
    if input.verbose:
        data["flow"] = flow_dict

    msg = f"Flow saved to project `{project_id}` ({n_nodes} nodes, {n_edges} edges)."

    # #10 / #9 — auto-run saved regression tests against the JUST-SAVED flow. A
    # failure must NOT undo the save (already persisted) — we surface the result
    # AND an explicit `regression_ok` flag so the agent can distinguish a clean
    # save (ok=true, regression_ok=true) from a save whose regression failed
    # (ok=true, regression_ok=false). Simulator/store outages degrade to a notice
    # and count as NOT-failed (regression_ok=true, ran=false).
    regression_failed = False
    regression = await _run_regression_after_save(ctx, project_id, flow_dict)
    if regression:
        data["regression"] = regression.get("data")
        data["regression_ok"] = bool(regression.get("ok", True))
        data["regression_ran"] = bool(regression.get("ran", False))
        regression_failed = not data["regression_ok"]
        msg += "\n" + regression["text"]
    else:
        # No tests defined for this project → nothing to regress against.
        data["regression_ok"] = True
        data["regression_ran"] = False

    # #7 — a running debug bot keeps serving the OLD flow until restarted.
    if input.restart_debug:
        try:
            await backend.start_debug(project_id, auth_claims=claims)
            project = await backend.get_project(project_id, auth_claims=claims)
            debug_uuid = (
                project.get("debug_uuid") or project.get("debug_webhook_uuid")
                if isinstance(project, dict)
                else None
            )
            if debug_uuid:
                link = f"https://t.me/botmother_test_robot?start={debug_uuid}"
                data["debug_link"] = link
                msg += f" 🤖 Debug bot avtomatik ishga tushirildi — foydalanuvchiga shu havolani BER: {link}"
            else:
                msg += " Debug bot qayta ishga tushirildi (test havolasi topilmadi)."
        except Exception as e:  # avto-debug HECH QACHON save'ni buzmasligi kerak
            detail = str(e)
            if isinstance(e, BackendError):
                from ...integrations.backend import format_backend_error

                detail = format_backend_error(None, e)
            msg += (
                " (Diqqat: debug botni avtomatik ishga tushirib bo'lmadi: "
                + detail
                + " — release_debug bilan qo'lda ishga tushiring.)"
            )
    else:
        # Static reminder: we cannot cheaply tell whether a debug bot is live,
        # so always nudge the model to refresh it if one is.
        msg += (
            " Eslatma: agar debug bot ishlayotgan bo'lsa, u hali ESKI flow bilan "
            "ishlaydi — yangilash uchun release_debug ni qayta chaqiring "
            "(yoki flow_save'ni restart_debug=true bilan chaqiring)."
        )

    # #9 — the flow is ALWAYS persisted by this point; state must reflect that
    # even when fail_on_regression turns the result into an error.
    state_updates = {
        "working_flow": flow_dict,
        "baseline_flow": flow_dict,
        "flow_dirty": False,
        "pending_save": False,
    }

    if input.fail_on_regression and regression_failed:
        return ToolResult.error(
            msg
            + "\n⚠️ fail_on_regression=true — save PERSISTED, lekin regressiya "
            "yiqildi. Tuzating va qayta save qiling, yoki flow_rollback bilan "
            "oldingi versiyaga qayting.",
            state_updates=state_updates,
            data=data,
        )

    return ToolResult.ok(
        msg,
        state_updates=state_updates,
        data=data,
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
        flow_patch_many,
        flow_connect,
        flow_disconnect,
        flow_validate,
        flow_layout_auto,
        flow_set_languages,
        flow_list_node_types,
        flow_diff,
        flow_save,
        *build_flow_power_tools(),
    ]
