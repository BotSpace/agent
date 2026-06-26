"""flow.search_template + flow.apply_template — template RAG tools.

The flow-builder sub-agent should ALWAYS try `flow.search_template`
first when the user asks to build a new bot. If a template's
similarity score is high enough, `flow.apply_template` drops its nodes
and edges into working_flow as a starting point; subsequent flow.*
edits modify that baseline instead of building from scratch.

The vector pool is shared across all projects (`project_id="_global"`,
`kind="template"`), populated by `python -m agent.scripts.ingest_templates`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...flow.catalog import get_node_spec
from ...integrations.backend import get_backend_client
from ..base import Tool, ToolContext, ToolResult, tool

SIMILARITY_GOOD = 0.55  # below this, the agent should probably build from scratch
TEMPLATE_PROJECT_ID = "_global"
TEMPLATE_KIND = "template"


# ─────────────────────────────────────────────────────────────────────────────
# Legacy template normalization (#4).
#
# Older templates store some fields in shapes the current node-spec/validator no
# longer accepts. The most common one: a collection name given as
# `data.collection: {name: "..."}` while the engine + validator now require a
# flat `data.collection_name`. We normalize such legacy shapes IN PLACE when a
# template is applied so the loaded flow passes flow.validate without manual
# patching. Only fields we can map unambiguously (by comparing against the
# node's catalog spec) are touched.
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_collection_name(data: dict[str, Any]) -> bool:
    """Legacy `collection: {name}` (or `collectionName`) → `collection_name`.
    Returns True if anything changed."""
    if "collection_name" in data and isinstance(data["collection_name"], str):
        return False
    legacy = data.get("collection")
    name: str | None = None
    if isinstance(legacy, dict) and isinstance(legacy.get("name"), str):
        name = legacy["name"]
    elif isinstance(legacy, str) and legacy:
        name = legacy
    elif isinstance(data.get("collectionName"), str):
        name = data["collectionName"]
    if not name:
        return False
    data["collection_name"] = name
    data.pop("collectionName", None)
    # Only drop the nested `collection` if THIS node type does not itself
    # require a `collection` object (LoadCollectionItemNode keeps it).
    return True


def _normalize_required_envs(tpl: dict[str, Any]) -> list[dict[str, str]]:
    """Extract `required_envs` from a template into a clean
    `[{"key", "description"}, ...]` list. Skips RELEASE_BOT_TOKEN (handled by
    its own release flow) and entries without a key. Never raises."""
    raw = tpl.get("required_envs")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key or key == "RELEASE_BOT_TOKEN":
            continue
        out.append({"key": key, "description": str(item.get("description") or "").strip()})
    return out


def normalize_legacy_flow(flow: dict[str, Any]) -> int:
    """Normalize a raw template flow dict in place. Returns the number of nodes
    whose data was changed."""
    nodes = flow.get("nodes")
    if not isinstance(nodes, list):
        return 0
    changed = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_type = node.get("type")
        spec = get_node_spec(node_type) if isinstance(node_type, str) else None
        requires_collection_name = bool(
            spec and "collection_name" in spec.required_data
        )
        node_changed = False
        if requires_collection_name and "collection_name" not in data:
            if _normalize_collection_name(data):
                # Safe to drop the nested legacy `collection` for these nodes —
                # they require collection_name, not the object form.
                data.pop("collection", None)
                node_changed = True
        # LoadCollectionListNode now requires an output key (contextKey). Legacy
        # nodes omitted it; default it to the collection name so the migrated
        # node is valid and the list lands at {{<collection_name>}}.
        if node_type == "LoadCollectionListNode" and not (
            data.get("contextKey") or data.get("resultKey")
        ):
            cn = data.get("collection_name")
            if isinstance(cn, str) and cn:
                data["contextKey"] = cn
                node_changed = True
        if node_changed:
            changed += 1
    return changed


class FlowSearchTemplateInput(BaseModel):
    query: str = Field(
        description=(
            "User's intent in 1-2 sentences (what the bot should do). "
            "Example: 'Restoran uchun menyu va buyurtma qabul qiluvchi bot.'"
        )
    )
    top_k: int = Field(default=3, ge=1, le=10)


@tool(
    name="flow.search_template",
    description=(
        "Semantic search over the global template catalog. Returns top-k "
        "matches with similarity scores and metadata (id, slug, name, "
        "categories, node/edge counts). Use BEFORE building a flow from "
        "scratch — if the best match scores > 0.55 the agent should call "
        "flow.apply_template with its id and then edit the loaded flow."
    ),
    input_model=FlowSearchTemplateInput,
)
async def flow_search_template(input: FlowSearchTemplateInput, ctx: ToolContext) -> ToolResult:
    # Lazy import — get_retriever pulls embedder + pgvector connection, which
    # we don't want at module load (breaks tests without DB).
    from ...memory import get_retriever

    retriever = get_retriever()
    try:
        hits = await retriever.recall(
            project_id=TEMPLATE_PROJECT_ID,
            query=input.query,
            kind=TEMPLATE_KIND,
            top_k=input.top_k,
        )
    except Exception as e:
        # pgvector extension missing on the Postgres server is the most
        # common cause here. Degrade gracefully so the agent just builds
        # from scratch instead of bubbling a stack trace to the user.
        msg = str(e).lower()
        if "vector" in msg and ("not available" in msg or "control file" in msg):
            return ToolResult.ok(
                "Template catalog not available right now (vector store not "
                "provisioned). Proceed by building the flow from scratch.",
                data={"matches": [], "vector_store_unavailable": True},
            )
        return ToolResult.ok(
            f"Template search unavailable ({type(e).__name__}). "
            "Proceed by building from scratch.",
            data={"matches": [], "error": str(e)[:200]},
        )

    if not hits:
        return ToolResult.ok(
            "No templates indexed yet, or no semantic matches. Proceed by building the flow from scratch.",
            data={"matches": []},
        )

    summary_lines = []
    for h in hits:
        meta = h.get("metadata") or {}
        sim = h.get("similarity", 0.0)
        verdict = "GOOD" if sim >= SIMILARITY_GOOD else "weak"
        env_keys = [str(k) for k in (meta.get("required_envs") or []) if k]
        env_note = ""
        if env_keys:
            shown = ",".join(env_keys[:4])
            if len(env_keys) > 4:
                shown += f",+{len(env_keys) - 4}"
            env_note = f" env: {shown}"
        summary_lines.append(
            f"  • id={meta.get('template_id')} slug={meta.get('slug')} "
            f"sim={sim:.2f} ({verdict}) — {meta.get('name')} "
            f"[{meta.get('n_nodes', 0)} nodes, {meta.get('n_edges', 0)} edges]{env_note}"
        )
    best = hits[0]
    best_sim = best.get("similarity", 0.0)
    advice = (
        f"Best match similarity={best_sim:.2f}. "
        + (
            "Strong match — call flow.apply_template with this id and then edit."
            if best_sim >= SIMILARITY_GOOD
            else "Weak match — building from scratch is likely better."
        )
    )
    return ToolResult.ok(
        f"{len(hits)} candidate(s):\n" + "\n".join(summary_lines) + f"\n\n{advice}",
        data={
            "matches": [
                {
                    "template_id": (h.get("metadata") or {}).get("template_id"),
                    "slug": (h.get("metadata") or {}).get("slug"),
                    "name": (h.get("metadata") or {}).get("name"),
                    "similarity": h.get("similarity"),
                    "n_nodes": (h.get("metadata") or {}).get("n_nodes"),
                    "n_edges": (h.get("metadata") or {}).get("n_edges"),
                    "categories": (h.get("metadata") or {}).get("categories"),
                    "required_envs": (h.get("metadata") or {}).get("required_envs"),
                }
                for h in hits
            ]
        },
    )


class FlowApplyTemplateInput(BaseModel):
    template_id: int | None = Field(
        default=None,
        description="Template DB id from flow.search_template results.",
    )
    slug: str | None = Field(
        default=None,
        description="Template slug. Pass either template_id OR slug.",
    )


@tool(
    name="flow.apply_template",
    description=(
        "Load a template's flow (nodes + edges) into working_flow as a "
        "starting baseline. Overwrites any existing working_flow — the "
        "agent should normally only call this on an empty or freshly "
        "cleared flow. After applying, edit the loaded flow with the "
        "usual flow.add_*, flow.connect, flow.patch tools."
    ),
    input_model=FlowApplyTemplateInput,
)
async def flow_apply_template(input: FlowApplyTemplateInput, ctx: ToolContext) -> ToolResult:
    identifier: str | int | None = input.template_id if input.template_id is not None else input.slug
    if identifier is None:
        return ToolResult.error("Either template_id or slug is required.")

    backend = get_backend_client()
    try:
        tpl = await backend.get_internal_template(identifier)
    except Exception as e:
        return ToolResult.error(f"Backend lookup failed: {type(e).__name__}: {e}")
    if tpl is None:
        return ToolResult.error(f"Template `{identifier}` not found.")

    flow: dict[str, Any] = tpl.get("flow") or {"nodes": [], "edges": []}
    nodes = flow.get("nodes") or []
    edges = flow.get("edges") or []
    if not nodes:
        return ToolResult.error(
            f"Template `{tpl.get('name')}` has an empty flow — cannot apply."
        )

    working = {"nodes": list(nodes), "edges": list(edges)}
    # #4 — normalize legacy field shapes (e.g. collection:{name} →
    # collection_name) so the loaded flow passes flow.validate.
    normalized = normalize_legacy_flow(working)
    norm_note = (
        f" Normalized {normalized} legacy node(s) (e.g. collection_name)."
        if normalized
        else ""
    )

    required_envs = _normalize_required_envs(tpl)
    env_note = ""
    if required_envs:
        listing = ", ".join(
            f"{e['key']} ({e['description']})" if e["description"] else e["key"]
            for e in required_envs
        )
        env_note = (
            f" Bu template quyidagi MAJBURIY env'larni talab qiladi: {listing} — "
            "foydalanuvchidan qiymatlarini SO'RA va env.set bilan o'rnat; "
            "o'rnatilmaguncha bot to'liq ishlamaydi."
        )

    data: dict[str, Any] = {
        "template_id": tpl.get("id"),
        "slug": tpl.get("slug"),
        "name": tpl.get("name"),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "normalized_nodes": normalized,
    }
    if required_envs:
        data["required_envs"] = required_envs
    return ToolResult.ok(
        f"Applied template `{tpl.get('name')}` (id={tpl.get('id')}): "
        f"{len(nodes)} nodes + {len(edges)} edges loaded.{norm_note} "
        f"Edit with flow.add_*, flow.connect, flow.patch as needed.{env_note}",
        data=data,
        state_updates={
            "working_flow": working,
            "flow_dirty": True,
            "last_validation": None,
        },
    )


class FlowListRecipesInput(BaseModel):
    pass


@tool(
    name="flow.list_recipes",
    description=(
        "List the bundled starter recipes — complete, pre-validated flows for "
        "the most common bot types (info/menu, lead-capture form, shop/order). "
        "ALWAYS works offline (no backend/vector store). Use this when building a "
        "NEW bot from scratch and flow.search_template found no good match: pick "
        "the closest recipe and call flow.apply_recipe, then adapt texts."
    ),
    input_model=FlowListRecipesInput,
)
async def flow_list_recipes(_: FlowListRecipesInput, ctx: ToolContext) -> ToolResult:
    from ...flow.recipes import RECIPES

    lines = []
    for r in RECIPES:
        flow = r["flow"]
        needs = r.get("needs_collections") or []
        needs_note = f" — collection: {', '.join(needs)}" if needs else ""
        lines.append(
            f"  • id={r['id']} ({r['difficulty']}) — {r['name']}: {r['use_cases']} "
            f"[{len(flow['nodes'])} nodes]{needs_note}"
        )
    return ToolResult.ok(
        "Bundled starter recipes (apply with flow.apply_recipe(recipe_id=...)):\n"
        + "\n".join(lines),
        data={
            "recipes": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "use_cases": r["use_cases"],
                    "difficulty": r["difficulty"],
                    "needs_collections": r.get("needs_collections") or [],
                    "n_nodes": len(r["flow"]["nodes"]),
                }
                for r in RECIPES
            ]
        },
    )


class FlowApplyRecipeInput(BaseModel):
    recipe_id: str = Field(description="Recipe id from flow.list_recipes (e.g. 'info-menu').")


@tool(
    name="flow.apply_recipe",
    description=(
        "Load a bundled starter recipe's flow (nodes + edges) into working_flow "
        "as a complete, valid baseline. Overwrites working_flow — call only on an "
        "empty/cleared flow. After applying: adapt the texts/buttons with "
        "flow.patch/flow.add_*, create any needed collections, then flow.validate "
        "→ flow.save → flow.simulate."
    ),
    input_model=FlowApplyRecipeInput,
)
async def flow_apply_recipe(input: FlowApplyRecipeInput, ctx: ToolContext) -> ToolResult:
    from ...flow.recipes import get_recipe

    recipe = get_recipe(input.recipe_id)
    if recipe is None:
        from ...flow.recipes import RECIPES

        ids = ", ".join(r["id"] for r in RECIPES)
        return ToolResult.error(f"Recipe `{input.recipe_id}` not found. Available: {ids}")

    flow = recipe["flow"]
    working = {"nodes": list(flow["nodes"]), "edges": list(flow["edges"])}
    needs = recipe.get("needs_collections") or []
    needs_note = (
        " Avval shu collection'larni yarating (yo'q bo'lsa cms-manager'ga delegate qiling): "
        + "; ".join(needs)
        + "."
        if needs
        else ""
    )
    return ToolResult.ok(
        f"Applied recipe `{recipe['name']}` ({len(working['nodes'])} nodes + "
        f"{len(working['edges'])} edges) — to'liq, validatsiyadan o'tgan flow yuklandi. "
        f"Moslashtirish: {recipe['adapt']}{needs_note} "
        "Keyin: matnlarni o'zgartiring → flow.validate → flow.save → flow.simulate.",
        data={
            "recipe_id": recipe["id"],
            "name": recipe["name"],
            "n_nodes": len(working["nodes"]),
            "n_edges": len(working["edges"]),
            "needs_collections": needs,
        },
        state_updates={
            "working_flow": working,
            "flow_dirty": True,
            "last_validation": None,
        },
    )


def build_flow_template_tools() -> list[Tool]:
    return [flow_search_template, flow_apply_template, flow_list_recipes, flow_apply_recipe]
