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

from ...integrations.backend import get_backend_client
from ..base import Tool, ToolContext, ToolResult, tool


SIMILARITY_GOOD = 0.55  # below this, the agent should probably build from scratch
TEMPLATE_PROJECT_ID = "_global"
TEMPLATE_KIND = "template"


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
    except Exception as e:  # noqa: BLE001
        return ToolResult.error(f"Template search failed: {type(e).__name__}: {e}")

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
        summary_lines.append(
            f"  • id={meta.get('template_id')} slug={meta.get('slug')} "
            f"sim={sim:.2f} ({verdict}) — {meta.get('name')} "
            f"[{meta.get('n_nodes', 0)} nodes, {meta.get('n_edges', 0)} edges]"
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
    except Exception as e:  # noqa: BLE001
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
    return ToolResult.ok(
        f"Applied template `{tpl.get('name')}` (id={tpl.get('id')}): "
        f"{len(nodes)} nodes + {len(edges)} edges loaded. "
        f"Edit with flow.add_*, flow.connect, flow.patch as needed.",
        data={
            "template_id": tpl.get("id"),
            "slug": tpl.get("slug"),
            "name": tpl.get("name"),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        },
        state_updates={
            "working_flow": working,
            "flow_dirty": True,
            "last_validation": None,
        },
    )


def build_flow_template_tools() -> list[Tool]:
    return [flow_search_template, flow_apply_template]
