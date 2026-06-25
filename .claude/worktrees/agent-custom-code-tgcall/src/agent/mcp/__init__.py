"""agent MCP (Model Context Protocol) server moduli.

Botmother agent tool registry'sini MCP orqali tashqi klientlarga (Claude Code,
boshqa MCP host'lar) ochadi. Transport: Streamable HTTP, stateless + JSON
response (LB-friendly). Auth: `bsp_` personal access token yoki oddiy JWT.

Ishga tushirish:
  - `create_app()` (api/app.py) `/mcp`ga mount qiladi (AGENT_MCP_ENABLED).
  - `python -m agent.mcp` — standalone uvicorn (AGENT_MCP_PORT, default 8200).
"""

from __future__ import annotations

from .auth import AuthedUser, MCPAuthError, authenticate
from .server import (
    build_mcp_asgi_app,
    build_mcp_server,
    build_session_manager,
    create_standalone_app,
    exported_tools,
    mcp_lifespan,
    mcp_tool_name,
)
from .workspace import Workspace, WorkspaceStore, get_workspace_store

__all__ = [
    "AuthedUser",
    "MCPAuthError",
    "Workspace",
    "WorkspaceStore",
    "authenticate",
    "build_mcp_asgi_app",
    "build_mcp_server",
    "build_session_manager",
    "create_standalone_app",
    "exported_tools",
    "get_workspace_store",
    "mcp_lifespan",
    "mcp_tool_name",
]
