"""Tool registry assembly — barcha namespace'lar shu yerda yig'iladi."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolResult, tool
from .broadcast import build_broadcast_tools
from .chat import build_chat_tools
from .cms import build_cms_tools
from .crm import build_crm_tools
from .env import build_env_tools
from .file import build_file_tools
from .flow import build_flow_tools
from .plan import build_plan_tools, has_incomplete_plan
from .project import build_project_tools
from .registry import ToolRegistry
from .release import build_release_tools
from .search import build_search_tools
from .task import build_task_tools


def build_default_registry() -> ToolRegistry:
    """Full registry — Phase 0-7 inclusive."""
    reg = ToolRegistry()
    reg.extend(build_plan_tools())
    reg.extend(build_flow_tools())
    reg.extend(build_project_tools())
    reg.extend(build_env_tools())
    reg.extend(build_release_tools())
    reg.extend(build_file_tools())
    reg.extend(build_crm_tools())
    reg.extend(build_cms_tools())
    reg.extend(build_broadcast_tools())
    reg.extend(build_chat_tools())
    reg.extend(build_search_tools())
    reg.extend(build_task_tools())
    return reg


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "has_incomplete_plan",
    "tool",
]
