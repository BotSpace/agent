"""Tool registry assembly — barcha namespace'lar shu yerda yig'iladi."""

from __future__ import annotations

from .ask_user import build_ask_user_tools
from .base import Tool, ToolContext, ToolResult, tool
from .broadcast import build_broadcast_tools
from .chat import build_chat_tools
from .cms import build_cms_tools
from .credentials import build_credentials_tools
from .env import build_env_tools
from .file import build_file_tools
from .flow import build_flow_tools
from .modules import build_modules_tools
from .plan import has_incomplete_plan
from .project import build_project_tools
from .registry import ToolRegistry
from .release import build_release_tools
from .search import build_search_tools
from .sleep import build_sleep_tools
from .task import build_task_tools


def build_default_registry() -> ToolRegistry:
    """Full registry — barcha domen tool'larini o'z ichiga oladi.

    Bu registry sub-agent (analyzer) uchun manba sifatida ishlatiladi
    (`task.run` ichida, `allowed_tools` bo'yicha filtrlanadi — analyzer'da
    task.run YO'Q). Main agent ham AYNAN shu to'plamni oladi
    (`build_main_registry()` shu funksiyani qaytaradi).
    """
    reg = ToolRegistry()
    # NOTE: the old step-checklist plan tools (plan.set/advance/complete/update)
    # are intentionally NOT registered — planning is now a read-only "plan mode"
    # that produces a detailed plan document, not a tracked checklist.
    reg.extend(build_ask_user_tools())
    reg.extend(build_flow_tools())
    reg.extend(build_project_tools())
    reg.extend(build_env_tools())
    reg.extend(build_release_tools())
    reg.extend(build_file_tools())
    reg.extend(build_cms_tools())
    reg.extend(build_credentials_tools())
    reg.extend(build_modules_tools())
    reg.extend(build_broadcast_tools())
    reg.extend(build_chat_tools())
    reg.extend(build_search_tools())
    reg.extend(build_sleep_tools())
    reg.extend(build_task_tools())
    # tools.search — loads full schemas of lazy tools on demand. Its getter sees
    # the full registry (incl. itself; is_core excludes it from the lazy set).
    from .lazy import build_tools_search

    reg.register(build_tools_search(lambda: reg.all()))
    return reg


# Read-only toolset for PLAN MODE — research/inspect + clarify only. An ALLOWLIST
# (not a denylist) so no mutating tool (build/save/create/deploy/broadcast) can
# ever leak into plan mode.
_PLAN_MODE_TOOLS = frozenset({
    "ask_user",
    "chat.list_users", "chat.user_history", "chat.user_stats",
    "cms.find", "cms.get_collection", "cms.get_document",
    "cms.list_collections", "cms.aggregate", "cms.audit_log",
    "credentials.list", "credentials.list_types",
    "env.list", "file.list",
    "flow.get", "flow.diff", "flow.export", "flow.history",
    "flow.list_node_types", "flow.list_recipes", "flow.search_template",
    "flow.validate", "flow.test_list",
    "modules.list_available", "modules.search",
    "plans.list",
    "project.list", "project.get", "project.errors", "project.snapshot",
    "release.debug_logs", "release.list", "release.logs", "release.status",
    "search.semantic", "sleep", "subscription.current", "tools.search",
})


def build_plan_mode_registry() -> ToolRegistry:
    """Read-only registry used when a run is started in plan mode."""
    reg = ToolRegistry()
    for t in build_main_registry().all():
        if t.name in _PLAN_MODE_TOOLS and t.name != "tools.search":
            reg.register(t)
    # Rebuild tools.search scoped to THIS registry so a search can only ever
    # surface plan-mode (read) tools — never project.create / cms.insert /
    # flow.add_* which would let the agent try to build in plan mode.
    from .lazy import build_tools_search

    reg.register(build_tools_search(lambda: reg.all()))
    return reg


def build_main_registry() -> ToolRegistry:
    """Main agent uchun registry — agent surface (TO'LIQ toolset minus MCP-only).

    Main agent endi domen ishini (flow qurish, CRM, CMS, deploy, ...) O'ZI
    bevosita bajaradi — chunki u chat tarixini to'liq ko'radi. Sub-agentlar
    tarixni ko'rmagani uchun chalkashardi, shuning uchun ish main'da qoladi.

    `task.run` ham bor: main faqat TAHLIL/TEKSHIRUV uchun bitta `analyzer`
    sub-agent chaqira oladi (bot qurish uchun EMAS). Analyzer'ning o'zida
    task.run yo'q — ichma-ich sub-agent bo'lmaydi.

    `expose_to_agent=False` tool'lar (MCP-only) bu registrga kirmaydi.
    """
    reg = ToolRegistry()
    for t in build_default_registry().all():
        if t.expose_to_agent:
            reg.register(t)
    return reg


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "build_main_registry",
    "build_plan_mode_registry",
    "has_incomplete_plan",
    "tool",
]

