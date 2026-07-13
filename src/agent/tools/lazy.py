"""Lazy tool loading — send only CORE tool schemas up front; the rest are shown
as a compact name+summary catalog and their full schemas are pulled on demand
with `tools.search`.

99 tool schemas were ~22K tokens/request. Core (common building/flow/plan tools)
stay full; specialized/rare tools (cms.*, broadcast.*, flow.simulate/test_*, ...)
become summary-only until the agent searches for them — cutting the per-request
tool payload roughly in half.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Tool, ToolContext, ToolResult, tool

# Tools ALWAYS sent with their full schema (common path: build/edit a flow, plan,
# pick a project, ask the user, deploy). Everything else is lazy.
_CORE_NAMES: frozenset[str] = frozenset(
    {
        # flow building essentials
        "flow.get", "flow.add_node", "flow.remove_node", "flow.patch",
        "flow.patch_many", "flow.connect", "flow.connect_button",
        "flow.connect_branch", "flow.connect_accessory", "flow.disconnect",
        "flow.validate", "flow.save", "flow.layout_auto", "flow.list_node_types",
        "flow.diff", "flow.set_languages", "flow.finish", "flow.save_to_collection",
        "flow.ask_text", "flow.ask_contact", "flow.ask_buttons", "flow.ask_location",
        # orchestration / project / user
        "ask_user", "task.run", "sleep", "search.semantic", "tools.search",
        "project.list", "project.get", "project.create", "project.update",
        "project.snapshot",
        # common deploy + checks
        "credentials.list", "env.list",
        "release.deploy", "release.debug", "release.status",
    }
)
# Whole namespaces that are always core.
_CORE_NAMESPACES: frozenset[str] = frozenset({"plan"})


def is_core(name: str) -> bool:
    if name in _CORE_NAMES:
        return True
    ns = name.split(".", 1)[0] if "." in name else name
    return ns in _CORE_NAMESPACES


def active_tool_specs(tools: list[Tool], activated: set[str]):
    """Specs to send this turn: every CORE tool + any lazy tool the agent has
    already loaded via tools.search (tracked in `activated`)."""
    return [t.to_spec() for t in tools if is_core(t.name) or t.name in activated]


def lazy_catalog(tools: list[Tool]) -> str:
    """Compact `name — summary` catalog of the LAZY tools for the system prompt."""
    lazy = sorted((t for t in tools if not is_core(t.name)), key=lambda t: t.name)
    if not lazy:
        return ""
    lines = [
        "## Qo'shimcha tool'lar (lazy)\n"
        "Quyidagilar DEFAULT'da to'liq berilmaydi — faqat nomi+qisqa tavsifi. "
        "Kerak bo'lsa `tools.search(query=\"...\")` bilan to'liq schemasini yukla "
        "(bir yoki bir nechta query/nom bilan), keyin chaqir.\n"
    ]
    for t in lazy:
        lines.append(f"- `{t.name}` — {t.summary}")
    return "\n".join(lines) + "\n"


class ToolsSearchInput(BaseModel):
    query: str | list[str] = Field(
        description=(
            "Qidiruv — tool nomi yoki tavsifidagi so'z(lar). Bir yoki bir nechta: "
            "`query=\"broadcast\"` yoki `query=[\"to'lov\",\"collection\"]`. Qisman mos "
            "kelsa ham topadi; mos tool'larning TO'LIQ schemasi qaytadi va shu suhbatda "
            "chaqirish uchun FAOLLASHADI (keyingi javobdan chaqira olasan)."
        )
    )


def _matches(t: Tool, terms: list[str]) -> bool:
    hay = f"{t.name} {t.description}".lower()
    return any(term in hay for term in terms)


def build_tools_search(all_tools_getter) -> Tool:
    """Build the tools.search tool. `all_tools_getter()` returns the full tool
    list (so search sees lazy tools that aren't in the active set)."""

    @tool(
        name="tools.search",
        description=(
            "Load the FULL schema of specialized/rarely-used tools that are shown "
            "only as name+summary in the '## Qo'shimcha tool'lar (lazy)' catalog. "
            "Search by name/keyword (partial match, one or more queries). The matched "
            "tools become callable from your NEXT reply. Use it before calling any tool "
            "not already in your active tool list (cms.*, broadcast.*, chat.*, file.*, "
            "flow.simulate/test_*, credentials.create, etc.)."
        ),
        input_model=ToolsSearchInput,
        core=True,
        expose_to_mcp=False,  # MCP sends all tools directly — no lazy loading there
    )
    async def tools_search(input: ToolsSearchInput, ctx: ToolContext) -> ToolResult:
        terms = [input.query] if isinstance(input.query, str) else list(input.query)
        terms = [str(x).strip().lower() for x in terms if str(x).strip()]
        if not terms:
            return ToolResult.error("tools.search: query bo'sh.")
        lazy = [t for t in all_tools_getter() if not is_core(t.name)]
        matched = [t for t in lazy if _matches(t, terms)]
        if not matched:
            return ToolResult.ok(f"Hech qanday lazy tool topilmadi: {terms}")

        import json as _json

        activated = set(ctx.state.get("activated_tools") or [])
        activated.update(t.name for t in matched)
        payload = [
            {"name": t.name, "description": t.description, "schema": t.to_spec().input_schema}
            for t in matched[:20]
        ]
        names = ", ".join(t.name for t in matched[:20])
        return ToolResult.ok(
            f"{len(matched)} tool faollashtirildi (keyingi javobdan chaqir): {names}\n"
            + _json.dumps(payload, ensure_ascii=False),
            state_updates={"activated_tools": sorted(activated)},
            data_in_content=True,
        )

    return tools_search
