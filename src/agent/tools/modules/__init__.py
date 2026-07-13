"""modules.* tools — discover and enable external modules for a project.

A module is an external service that contributes extra `moduleId.NodeName` node
types (and sometimes triggers / credential types) to a project. The static node
catalog does NOT include them — they appear only once a module is ENABLED for the
project (then `session_use_project` / `flow_list_node_types` surface them).

These tools let the agent:
  - `modules.search`        — semantic search over the platform-wide module
                              CATALOG (RAG, kind="module"). Use when a user wants
                              a capability the built-in nodes don't cover ("can it
                              call OpenRouter / a custom LLM / my own service?").
  - `modules.list_available`— browse / keyword-filter the catalog directly
                              (offline-friendly: always hits the backend, no
                              vector store needed).
  - `modules.enable`        — ENABLE a catalog module for the active project.
                              SIDE-EFFECTFUL: the backend deploys/attaches the
                              module's service. ASK the user before calling it.

Enable is intentionally a "prompt-asks" gate, not a hard approval interrupt: the
description tells the agent to confirm with the user first, and the tool itself
states the side effect plainly in its result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendClient,
    BackendError,
    format_backend_error,
    get_backend_client,
    load_modules_catalog,
)
from .._common import resolve_project_id
from ..base import Tool, ToolContext, ToolResult, tool

MODULE_PROJECT_ID = "_global"
MODULE_KIND = "module"
SIMILARITY_GOOD = 0.45  # modules have short descriptions → looser bar than templates


# ponytail: local _backend saqlanadi — testlar module-global get_backend_client'ni
# patch qiladi (ctx.services orqali emas), umumiy resolver bu seam'ni buzadi.
def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def _claims(ctx: ToolContext) -> dict | None:
    claims = ctx.state.get("auth_claims")
    return claims if isinstance(claims, dict) else None


# ── modules.search ───────────────────────────────────────────────────────────


class ModulesSearchInput(BaseModel):
    query: str = Field(
        description=(
            "The capability the user wants in 1-2 sentences. "
            "Example: 'OpenRouter orqali boshqa LLM modeliga ulanish'."
        )
    )
    top_k: int = Field(default=4, ge=1, le=10)


@tool(
    name="modules.search",
    description=(
        "Semantic search over the platform's external MODULE catalog. Use this "
        "FIRST when a user asks for a capability the built-in nodes don't have "
        "(e.g. another LLM provider, a third-party integration, their own "
        "service). Returns top-k catalog modules with slug, the node types they "
        "add, and a similarity score. If a match looks right, tell the user what "
        "it does and (after they agree) call modules.enable with its slug."
    ),
    input_model=ModulesSearchInput,
)
async def modules_search(input: ModulesSearchInput, ctx: ToolContext) -> ToolResult:
    # Lazy import — get_retriever pulls embedder + pgvector connection.
    from ...memory import get_retriever

    retriever = get_retriever()
    try:
        hits = await retriever.recall(
            project_id=MODULE_PROJECT_ID,
            query=input.query,
            kind=MODULE_KIND,
            top_k=input.top_k,
        )
    except Exception as e:
        msg = str(e).lower()
        if "vector" in msg and ("not available" in msg or "control file" in msg):
            return ToolResult.ok(
                "Module catalog search not available right now (vector store not "
                "provisioned). Use modules.list_available to browse the catalog instead.",
                data={"matches": [], "vector_store_unavailable": True},
            )
        return ToolResult.ok(
            f"Module search unavailable ({type(e).__name__}). "
            "Use modules.list_available to browse the catalog instead.",
            data={"matches": [], "error": str(e)[:200]},
        )

    if not hits:
        return ToolResult.ok(
            "No modules indexed yet, or no semantic matches. Try "
            "modules.list_available to browse the full catalog.",
            data={"matches": []},
        )

    lines: list[str] = []
    for h in hits:
        meta = h.get("metadata") or {}
        sim = h.get("similarity", 0.0)
        verdict = "GOOD" if sim >= SIMILARITY_GOOD else "weak"
        node_types = [str(t) for t in (meta.get("node_types") or []) if t]
        nodes_note = (" nodes: " + ",".join(node_types[:4])) if node_types else ""
        lines.append(
            f"  • slug={meta.get('module_slug')} sim={sim:.2f} ({verdict}) — "
            f"{meta.get('name')}{nodes_note}"
        )
    best = hits[0]
    best_sim = best.get("similarity", 0.0)
    advice = (
        f"Best match similarity={best_sim:.2f}. "
        + (
            "Strong match — confirm with the user, then modules.enable(slug=...)."
            if best_sim >= SIMILARITY_GOOD
            else "Weak match — double-check with modules.list_available before enabling."
        )
    )
    return ToolResult.ok(
        f"{len(hits)} candidate module(s):\n" + "\n".join(lines) + f"\n\n{advice}",
        data={
            "matches": [
                {
                    "slug": (h.get("metadata") or {}).get("module_slug"),
                    "name": (h.get("metadata") or {}).get("name"),
                    "similarity": h.get("similarity"),
                    "node_types": (h.get("metadata") or {}).get("node_types"),
                    "has_trigger": (h.get("metadata") or {}).get("has_trigger"),
                }
                for h in hits
            ]
        },
    )


# ── modules.list_available ───────────────────────────────────────────────────


class ModulesListAvailableInput(BaseModel):
    search: str | None = Field(
        default=None,
        description="Optional keyword filter over name/slug/description.",
    )


@tool(
    name="modules.list_available",
    description=(
        "List the platform's external MODULE catalog (active modules that CAN be "
        "enabled per-project). Optional keyword `search`. Always hits the backend "
        "directly (no vector store needed) — use it to browse, or to verify a "
        "modules.search hit. To turn one on for the project, call modules.enable."
    ),
    input_model=ModulesListAvailableInput,
)
async def modules_list_available(
    input: ModulesListAvailableInput, ctx: ToolContext
) -> ToolResult:
    try:
        modules = await load_modules_catalog(
            search=input.search,
            auth_claims=_claims(ctx),
            client=_backend(ctx),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("Modul katalogi", e))

    if not modules:
        return ToolResult.ok(
            "No modules in the catalog"
            + (f" matching '{input.search}'." if input.search else ".")
            + " (Nothing to enable.)",
            data={"modules": []},
        )

    lines: list[str] = []
    compact: list[dict] = []
    for m in modules:
        slug = m.get("slug")
        name = m.get("name") or slug
        desc = (m.get("description") or "").strip()
        nodes = m.get("nodes") or []
        node_types = [str(n.get("type")) for n in nodes if isinstance(n, dict) and n.get("type")]
        desc_short = (desc[:80] + "…") if len(desc) > 80 else desc
        nodes_note = (" — nodes: " + ",".join(node_types[:4])) if node_types else ""
        lines.append(f"  • slug={slug} — {name}: {desc_short}{nodes_note}")
        compact.append(
            {"slug": slug, "name": name, "description": desc, "node_types": node_types}
        )
    return ToolResult.ok(
        f"{len(modules)} module(s) available "
        "(enable one with modules.enable(slug=...)):\n" + "\n".join(lines),
        data={"modules": compact},
    )


# ── modules.enable ───────────────────────────────────────────────────────────


class ModulesEnableInput(BaseModel):
    slug: str = Field(description="Module slug from modules.search / modules.list_available.")
    project_id: str | int | None = Field(
        default=None,
        description="Target project; omit to use the active project.",
    )


@tool(
    name="modules.enable",
    description=(
        "Enable a catalog module FOR THE ACTIVE PROJECT so its node types become "
        "usable. SIDE-EFFECTFUL — the backend deploys/attaches the module's "
        "service; this may take a moment and (for billed modules) has cost "
        "implications. ALWAYS ask the user to confirm BEFORE calling this, naming "
        "the module. After it succeeds the new node types are NOT instantly ready: "
        "the module pod is still booting, so re-run session_use_project (or "
        "flow_list_node_types) after a short wait to load its manifest/nodes."
    ),
    input_model=ModulesEnableInput,
)
async def modules_enable(input: ModulesEnableInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if not pid or str(pid).lower() in {"global", "none", "null"}:
        return ToolResult.error(
            "No active project — call session_use_project (or pass project_id) first."
        )
    slug = input.slug.strip()
    if not slug:
        return ToolResult.error("`slug` is required.")
    try:
        result = await _backend(ctx).enable_module(pid, slug, auth_claims=_claims(ctx))
    except BackendError as e:
        return ToolResult.error(format_backend_error(f"Modul '{slug}' yoqish", e))

    already = isinstance(result, dict) and str(result.get("detail", "")).lower() == "already enabled"
    status = result.get("status") if isinstance(result, dict) else None
    head = (
        f"Module '{slug}' is already enabled for project {pid}."
        if already
        else f"Enabled module '{slug}' for project {pid}"
        + (f" (status: {status})." if status else ".")
    )

    # Refresh the module node manifests into state RIGHT NOW. The manifests are
    # available as soon as the module is enabled (the backend returns them for
    # every enabled=TRUE project module, pending or active — its runtime pod
    # deploy is separate), so the agent can query/add moduleId.NodeName types
    # immediately without a sleep + reload dance.
    state_updates: dict | None = None
    node_note = ""
    try:
        from ...integrations.backend import load_project_module_nodes

        module_nodes = await load_project_module_nodes(
            pid, auth_claims=_claims(ctx), client=_backend(ctx)
        )
        if module_nodes:
            state_updates = {"module_nodes": module_nodes}
            mine = [
                n.get("type")
                for n in module_nodes
                if isinstance(n, dict) and str(n.get("module")) == slug
            ]
            node_note = (
                f" Node types now available: {', '.join(str(t) for t in mine if t)}."
                if mine
                else ""
            )
    except Exception:
        node_note = ""

    return ToolResult.ok(
        head
        + (
            node_note
            or " The module's node types should now be queryable via "
            "flow_list_node_types(module=...) — if they aren't, re-run "
            "session_use_project."
        ),
        data={"slug": slug, "project_id": str(pid), "already_enabled": already, "status": status},
        state_updates=state_updates,
    )


def build_modules_tools() -> list[Tool]:
    return [modules_search, modules_list_available, modules_enable]
