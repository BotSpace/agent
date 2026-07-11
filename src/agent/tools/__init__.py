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
from .plan import build_plan_tools, has_incomplete_plan
from .project import build_project_tools
from .registry import ToolRegistry
from .release import build_release_tools
from .search import build_search_tools
from .task import build_task_tools


def build_default_registry() -> ToolRegistry:
    """Full registry — barcha domen tool'larini o'z ichiga oladi.

    Bu registry sub-agent (analyzer) uchun manba sifatida ishlatiladi
    (`task.run` ichida, `allowed_tools` bo'yicha filtrlanadi — analyzer'da
    task.run YO'Q). Main agent ham AYNAN shu to'plamni oladi
    (`build_main_registry()` shu funksiyani qaytaradi).
    """
    reg = ToolRegistry()
    reg.extend(build_plan_tools())
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
    reg.extend(build_task_tools())
    return reg


def build_main_registry() -> ToolRegistry:
    """Main agent uchun registry — TO'LIQ toolset.

    Main agent endi domen ishini (flow qurish, CRM, CMS, deploy, ...) O'ZI
    bevosita bajaradi — chunki u chat tarixini to'liq ko'radi. Sub-agentlar
    tarixni ko'rmagani uchun chalkashardi, shuning uchun ish main'da qoladi.

    `task.run` ham bor: main faqat TAHLIL/TEKSHIRUV uchun bitta `analyzer`
    sub-agent chaqira oladi (bot qurish uchun EMAS). Analyzer'ning o'zida
    task.run yo'q — ichma-ich sub-agent bo'lmaydi.
    """
    return build_default_registry()


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "build_main_registry",
    "has_incomplete_plan",
    "tool",
]

