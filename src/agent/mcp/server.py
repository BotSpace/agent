"""MCP server — agentv3 tool registry'sini Model Context Protocol orqali ochadi.

Gibrid transport: rasmiy `mcp` SDK low-level `Server` + `StreamableHTTPSessionManager`
(`stateless_http=True`, `json_response=True` — LB-friendly). Ikki ishga tushirish:
  - `create_app()` ichiga `/mcp` path'ga mount (api/app.py)
  - `python -m agent.mcp` — standalone uvicorn

Tool manbasi: `build_default_registry()`. Agent-ichki namespace'lar
(`plan.*`, `task.*`, `ask_user`) MCP'da ma'nosiz — chiqarib tashlanadi.
MCP tool nomi: nuqta → pastki chiziq (`flow.add_node` → `flow_add_node`).

Auth contextvar orqali har request'ning AuthedUser'i tool handler'ga yetadi.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mcp.types as mtypes
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..integrations.backend import (
    get_backend_client,
    load_project_collections,
    load_saved_flow,
)
from ..tools import ToolContext, build_default_registry
from ..tools.registry import ToolRegistry
from .auth import AuthedUser, MCPAuthError, authenticate
from .workspace import get_workspace_store

logger = logging.getLogger(__name__)

# MCP'da ko'rsatilmaydigan agent-ichki namespace/tool'lar.
_EXCLUDED_NAMESPACES = frozenset({"plan", "task"})
_EXCLUDED_NAMES = frozenset({"ask_user"})

# Read-only deb hisoblanadigan operatsiya nomlari (oxirgi segment bo'yicha).
_READ_ONLY_VERBS = ("get", "list", "diff", "validate", "search")

# Joriy request'ning autentifikatsiyalangan foydalanuvchisi (Starlette middleware
# o'rnatadi, tool handler o'qiydi).
_current_user: contextvars.ContextVar[AuthedUser | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)

# MCP-only session tool'lari — registry'da yo'q, server o'zi boshqaradi.
_SESSION_USE_PROJECT = "session_use_project"
_SESSION_STATUS = "session_status"


def mcp_tool_name(registry_name: str) -> str:
    """`flow.add_node` → `flow_add_node` (nuqta → pastki chiziq)."""
    return registry_name.replace(".", "_")


def is_excluded(registry_name: str) -> bool:
    if registry_name in _EXCLUDED_NAMES:
        return True
    ns = registry_name.split(".", 1)[0] if "." in registry_name else registry_name
    return ns in _EXCLUDED_NAMESPACES


def exported_tools(registry: ToolRegistry) -> dict[str, str]:
    """MCP tool nomi → registry tool nomi (exclude qilinganlar tashqari)."""
    out: dict[str, str] = {}
    for t in registry.all():
        if is_excluded(t.name):
            continue
        out[mcp_tool_name(t.name)] = t.name
    return out


def _is_read_only(registry_name: str) -> bool:
    verb = registry_name.rsplit(".", 1)[-1]
    return any(verb == v or verb.startswith(v + "_") for v in _READ_ONLY_VERBS)


def _to_mcp_tool(reg_name: str, registry: ToolRegistry) -> mtypes.Tool:
    tool = registry.get(reg_name)
    assert tool is not None
    spec = tool.to_spec()
    destructive = bool(tool.destructive or tool.always_approve)
    annotations = mtypes.ToolAnnotations(
        destructiveHint=destructive,
        readOnlyHint=_is_read_only(reg_name) or None,
    )
    return mtypes.Tool(
        name=mcp_tool_name(reg_name),
        description=tool.description,
        inputSchema=spec.input_schema,
        annotations=annotations,
    )


# ── Session (MCP-only) tool definitions ──────────────────────────────────────


def _session_tools() -> list[mtypes.Tool]:
    return [
        mtypes.Tool(
            name=_SESSION_USE_PROJECT,
            description=(
                "Aktiv loyihani tanlaydi va uning saqlangan flow'i hamda CMS "
                "kolleksiyalarini sessiyaga yuklaydi. flow.* tool'laridan OLDIN "
                "chaqiring — aks holda qaysi loyiha ustida ishlash noma'lum."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Loyiha id'si (raqamli).",
                    }
                },
                "required": ["project_id"],
            },
            annotations=mtypes.ToolAnnotations(readOnlyHint=True),
        ),
        mtypes.Tool(
            name=_SESSION_STATUS,
            description=(
                "Joriy sessiya holati: aktiv loyiha, flow node/edge soni, "
                "saqlanmagan o'zgarishlar (flow_dirty) bormi."
            ),
            inputSchema={"type": "object", "properties": {}},
            annotations=mtypes.ToolAnnotations(readOnlyHint=True),
        ),
    ]


# ── Tool execution ───────────────────────────────────────────────────────────


def _result_to_mcp(content: str, *, is_error: bool, data: dict | None) -> mtypes.CallToolResult:
    blocks: list[mtypes.ContentBlock] = [mtypes.TextContent(type="text", text=content or "")]
    structured = data if isinstance(data, dict) else None
    return mtypes.CallToolResult(
        content=blocks,
        isError=is_error,
        structuredContent=structured,
    )


async def _handle_session_use_project(
    user: AuthedUser, arguments: dict
) -> mtypes.CallToolResult:
    project_id = str(arguments.get("project_id") or "").strip()
    if not project_id:
        return _result_to_mcp("project_id majburiy.", is_error=True, data=None)

    ws = get_workspace_store().get(user.user_id)
    ws.sync_auth_claims(user.claims)

    backend = get_backend_client()
    flow = await load_saved_flow(project_id, auth_claims=ws.auth_claims, client=backend)
    collections = await load_project_collections(
        project_id, auth_claims=ws.auth_claims, client=backend
    )

    flow = flow if isinstance(flow, dict) else {"nodes": [], "edges": []}
    ws.project_id = project_id
    ws.working_flow = flow
    ws.baseline_flow = flow
    ws.flow_dirty = False
    ws.collections = collections

    nodes = len(flow.get("nodes") or [])
    edges = len(flow.get("edges") or [])
    col_names = [c.get("name") for c in collections if isinstance(c, dict) and c.get("name")]
    text = (
        f"Loyiha {project_id} tanlandi.\n"
        f"Flow: {nodes} node, {edges} edge.\n"
        f"Kolleksiyalar ({len(col_names)}): {', '.join(col_names) if col_names else '—'}"
    )
    return _result_to_mcp(
        text,
        is_error=False,
        data={
            "project_id": project_id,
            "nodes": nodes,
            "edges": edges,
            "collections": col_names,
        },
    )


async def _handle_session_status(user: AuthedUser) -> mtypes.CallToolResult:
    ws = get_workspace_store().get(user.user_id)
    if not ws.project_id:
        return _result_to_mcp(
            "Aktiv loyiha yo'q. session_use_project bilan loyiha tanlang.",
            is_error=False,
            data={"project_id": None},
        )
    flow = ws.working_flow or {}
    nodes = len(flow.get("nodes") or [])
    edges = len(flow.get("edges") or [])
    text = (
        f"Aktiv loyiha: {ws.project_id}\n"
        f"Flow: {nodes} node, {edges} edge\n"
        f"Saqlanmagan o'zgarishlar: {'BOR' if ws.flow_dirty else 'yoq'}"
    )
    return _result_to_mcp(
        text,
        is_error=False,
        data={
            "project_id": ws.project_id,
            "nodes": nodes,
            "edges": edges,
            "flow_dirty": ws.flow_dirty,
            "collections": [c.get("name") for c in ws.collections if isinstance(c, dict)],
        },
    )


async def _execute_registry_tool(
    reg_name: str,
    arguments: dict,
    user: AuthedUser,
    registry: ToolRegistry,
) -> mtypes.CallToolResult:
    ws = get_workspace_store().get(user.user_id)
    ws.sync_auth_claims(user.claims)

    # Flow tool'lari aktiv loyihani talab qiladi.
    if reg_name.startswith("flow.") and not ws.project_id:
        return _result_to_mcp(
            "Aktiv loyiha yo'q — avval session_use_project bilan loyiha tanlang.",
            is_error=True,
            data=None,
        )

    state = ws.to_tool_state()
    ctx = ToolContext(
        run_id=f"mcp-{uuid.uuid4().hex[:12]}",
        project_id=ws.project_id,
        user_id=user.user_id,
        state=state,
        services={"backend": get_backend_client()},
    )
    result = await registry.execute(reg_name, arguments, ctx)
    # state_updates'ni workspace'ga qaytarib singdiramiz (working_flow, flow_dirty, ...).
    ws.merge_state_updates(result.state_updates)
    return _result_to_mcp(result.content, is_error=result.is_error, data=result.data)


# ── Low-level MCP Server qurish ──────────────────────────────────────────────


def build_mcp_server(registry: ToolRegistry | None = None) -> Server:
    registry = registry or build_default_registry()
    name_map = exported_tools(registry)  # mcp_name → reg_name

    server: Server = Server("botmother-agent")

    @server.list_tools()
    async def list_tools() -> list[mtypes.Tool]:
        tools = _session_tools()
        tools.extend(_to_mcp_tool(reg_name, registry) for reg_name in name_map.values())
        return tools

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> mtypes.CallToolResult:
        user = _current_user.get()
        if user is None:
            # Auth middleware har doim o'rnatadi; bu himoya chizig'i.
            return _result_to_mcp("Autentifikatsiya talab qilinadi.", is_error=True, data=None)
        if name == _SESSION_USE_PROJECT:
            return await _handle_session_use_project(user, arguments)
        if name == _SESSION_STATUS:
            return await _handle_session_status(user)
        reg_name = name_map.get(name)
        if reg_name is None:
            return _result_to_mcp(
                f"Noma'lum tool `{name}`. /tools ro'yxatiga qarang.",
                is_error=True,
                data=None,
            )
        return await _execute_registry_tool(reg_name, arguments, user, registry)

    return server


# ── Starlette ASGI app (auth middleware + StreamableHTTP mount) ───────────────


class _MCPApp:
    """`/mcp`ga mount qilinadigan ASGI app — auth + StreamableHTTP handler.

    Streamable HTTP'da har request `session_manager.handle_request` orqali
    yo'naltiriladi. Biz oldin `Authorization` header'ini tekshirib, AuthedUser'ni
    contextvar'ga o'rnatamiz; auth yiqilsa 401 qaytaramiz (MCP so'rovi backend'ga
    umuman bormaydi).
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._session_manager.handle_request(scope, receive, send)
            return
        request = Request(scope, receive)
        authorization = request.headers.get("authorization")
        try:
            user = await authenticate(authorization)
        except MCPAuthError as exc:
            response = JSONResponse(
                {"error": "unauthorized", "detail": str(exc)},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        token = _current_user.set(user)
        try:
            await self._session_manager.handle_request(scope, receive, send)
        finally:
            _current_user.reset(token)


def build_session_manager(registry: ToolRegistry | None = None) -> StreamableHTTPSessionManager:
    server = build_mcp_server(registry)
    return StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
    )


@asynccontextmanager
async def mcp_lifespan(session_manager: StreamableHTTPSessionManager) -> AsyncIterator[None]:
    """StreamableHTTPSessionManager.run() lifespan'ga ulanishi SHART.

    Aks holda task group ishga tushmaydi va handle_request RuntimeError beradi.
    Bu context manager mount-host app'ning lifespan'iga (yoki standalone app'ga)
    ulanadi.
    """
    async with session_manager.run():
        yield


def build_mcp_asgi_app(registry: ToolRegistry | None = None) -> tuple[_MCPApp, StreamableHTTPSessionManager]:
    """`(asgi_app, session_manager)` — host app mount qiladi, lifespan ulaydi."""
    session_manager = build_session_manager(registry)
    return _MCPApp(session_manager), session_manager


def create_standalone_app(registry: ToolRegistry | None = None) -> Starlette:
    """`python -m agent.mcp` uchun mustaqil Starlette app — `/mcp` mount."""
    asgi_app, session_manager = build_mcp_asgi_app(registry)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with mcp_lifespan(session_manager):
            yield

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "agent-mcp"})

    from starlette.routing import Route

    return Starlette(
        debug=False,
        routes=[
            Route("/health", health),
            Mount("/mcp", app=asgi_app),
        ],
        lifespan=lifespan,
    )


__all__ = [
    "build_mcp_asgi_app",
    "build_mcp_server",
    "build_session_manager",
    "create_standalone_app",
    "exported_tools",
    "is_excluded",
    "mcp_lifespan",
    "mcp_tool_name",
]
