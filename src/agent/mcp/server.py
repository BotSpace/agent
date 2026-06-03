"""MCP server — agent tool registry'sini Model Context Protocol orqali ochadi.

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
from .draft_store import get_draft_store
from .workspace import Workspace, get_workspace_store

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
_SESSION_UNDO = "session_undo"
_DOCS_GUIDE = "docs_guide"

# Registry tool'lar — bajarilishidan OLDIN avtomatik undo snapshot olinadi (#11).
_SNAPSHOT_BEFORE = frozenset({"flow.apply_template", "flow.clear", "flow.save"})


async def _persist_draft(ws: Workspace) -> None:
    """Write-through persist of the workspace draft to agent Postgres (#1).

    Graceful degrade lives inside DraftStore — on DB failure this is a no-op
    plus a warning, and the MCP keeps working in-memory.
    """
    if not ws.project_id:
        return
    await get_draft_store().upsert(
        user_id=ws.user_id,
        project_id=ws.project_id,
        project_name=ws.project_name,
        working_flow=ws.working_flow,
        baseline_flow=ws.baseline_flow,
        flow_dirty=ws.flow_dirty,
        collections=ws.collections,
        snapshots=ws.snapshots,
    )


def _hydrate_from_draft(ws: Workspace, draft: dict) -> None:
    """Load a persisted draft row into the in-memory workspace."""
    ws.project_name = draft.get("project_name")
    ws.working_flow = draft.get("working_flow")
    ws.baseline_flow = draft.get("baseline_flow")
    ws.flow_dirty = bool(draft.get("flow_dirty"))
    ws.collections = draft.get("collections") or []
    snaps = draft.get("snapshots")
    ws.snapshots = list(snaps) if isinstance(snaps, list) else []

# docs_guide topic → (sarlavha, kontent loader). Kontent agent'ning ICHKI
# agentlariga beriladigan system prompt'lar — MCP klient modeli ham xuddi shu
# bilimga ega bo'lishi uchun tool orqali ochib beriladi.
_GUIDE_TOPICS = {
    "flow": "Flow-builder qo'llanmasi (handles, trigger, AI nodes, keyboard, state)",
    "flow_examples": "To'liq bot misollari (flow JSON'lari bilan)",
    "node_catalog": "Barcha node turlari katalogi (kategoriya, handle, maydonlar)",
    "cms": "CMS kolleksiyalari bilan ishlash qo'llanmasi",
    "crm": "CRM (chat tarixi, broadcast) qo'llanmasi",
    "deploy": "Deploy/release qo'llanmasi",
}


def _load_guide(topic: str) -> str:
    from ..flow.catalog import render_compact_catalog
    from ..prompts.loader import load_prompt

    if topic == "node_catalog":
        return render_compact_catalog()
    files = {
        "flow": "subagents/flow_builder.txt",
        "flow_examples": "subagents/flow_builder_examples.txt",
        "cms": "subagents/cms_manager.txt",
        "crm": "subagents/crm_manager.txt",
        "deploy": "subagents/deploy_manager.txt",
    }
    return load_prompt(files[topic])


def _load_instructions() -> str:
    """MCP initialize natijasidagi `instructions` — klient modeliga kontekst."""
    from ..prompts.loader import load_prompt

    return load_prompt("mcp/instructions.txt")


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
                "Joriy sessiya holati: project_id, project_name, flow node/edge "
                "soni, saqlanmagan o'zgarishlar (flow_dirty), kolleksiyalar ro'yxati."
            ),
            inputSchema={"type": "object", "properties": {}},
            annotations=mtypes.ToolAnnotations(readOnlyHint=True),
        ),
        mtypes.Tool(
            name=_SESSION_UNDO,
            description=(
                "Oxirgi avtomatik snapshot'ga qaytaradi (flow_apply_template / "
                "flow_clear / flow_save dan OLDIN olingan). Oxirgi 5 ta snapshot "
                "saqlanadi. Destructive amalni qaytarish uchun ishlating."
            ),
            inputSchema={"type": "object", "properties": {}},
            annotations=mtypes.ToolAnnotations(),
        ),
        mtypes.Tool(
            name=_DOCS_GUIDE,
            description=(
                "Botmother bot qurish qo'llanmalarini qaytaradi — agent'ning "
                "ichki agentlari ishlatadigan to'liq yo'riqnomalar. Notrivial bot "
                "qurishdan OLDIN o'qing: topic='flow' (qoidalar/handles/AI nodes), "
                "topic='node_catalog' (barcha node turlari), topic='flow_examples' "
                "(to'liq misollar). Boshqa topic'lar: cms, crm, deploy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": sorted(_GUIDE_TOPICS),
                        "description": "; ".join(
                            f"{k} — {v}" for k, v in sorted(_GUIDE_TOPICS.items())
                        ),
                    }
                },
                "required": ["topic"],
            },
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

    store = get_workspace_store()
    async with store.lock(user.user_id):
        ws = store.set_active(user.user_id, project_id)
        ws.sync_auth_claims(user.claims)

        backend = get_backend_client()
        # #1 — restore an unsaved draft if one was persisted for this
        # (user, project) and is still dirty. Otherwise load the saved flow.
        draft = await get_draft_store().load(user_id=user.user_id, project_id=project_id)
        restored = bool(draft and draft.get("flow_dirty") and draft.get("working_flow"))

        if restored:
            _hydrate_from_draft(ws, draft)
            # Refresh collections from backend (cheap, keeps CMS list current);
            # fall back to the persisted list on failure.
            try:
                collections = await load_project_collections(
                    project_id, auth_claims=ws.auth_claims, client=backend
                )
                ws.collections = collections
            except Exception:
                collections = ws.collections
            ws.project_id = project_id
            flow = ws.working_flow or {"nodes": [], "edges": []}
        else:
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
            ws.snapshots = []

        # Remember the project's display name for session_status (#7).
        try:
            project = await backend.get_project(project_id, auth_claims=ws.auth_claims)
            if isinstance(project, dict) and project.get("name"):
                ws.project_name = str(project["name"])
        except Exception:
            pass

        await _persist_draft(ws)

        nodes = len(flow.get("nodes") or [])
        edges = len(flow.get("edges") or [])
        col_names = [
            c.get("name") for c in collections if isinstance(c, dict) and c.get("name")
        ]
        if restored:
            head = (
                f"Loyiha {project_id} tanlandi. Saqlanmagan draft tiklandi "
                f"(flow_dirty=true), {nodes} node, {edges} edge."
            )
        else:
            head = f"Loyiha {project_id} tanlandi.\nFlow: {nodes} node, {edges} edge."
        text = (
            f"{head}\n"
            f"Kolleksiyalar ({len(col_names)}): {', '.join(col_names) if col_names else '—'}"
        )
        return _result_to_mcp(
            text,
            is_error=False,
            data={
                "project_id": project_id,
                "project_name": ws.project_name,
                "nodes": nodes,
                "edges": edges,
                "flow_dirty": ws.flow_dirty,
                "draft_restored": restored,
                "collections": col_names,
            },
        )


async def _handle_session_status(user: AuthedUser) -> mtypes.CallToolResult:
    store = get_workspace_store()
    async with store.lock(user.user_id):
        ws = store.get(user.user_id)
        if not ws.project_id:
            return _result_to_mcp(
                "Aktiv loyiha yo'q. session_use_project(project_id) bilan loyiha "
                "tanlang — saqlanmagan draft bo'lsa avtomatik tiklanadi.",
                is_error=False,
                data={"project_id": None},
            )
        flow = ws.working_flow or {}
        nodes = len(flow.get("nodes") or [])
        edges = len(flow.get("edges") or [])
        col_names = [
            c.get("name") for c in ws.collections if isinstance(c, dict) and c.get("name")
        ]
        text = (
            f"Aktiv loyiha: {ws.project_id}"
            + (f" ({ws.project_name})" if ws.project_name else "")
            + f"\nFlow: {nodes} node, {edges} edge"
            f"\nSaqlanmagan o'zgarishlar: {'BOR' if ws.flow_dirty else 'yoq'}"
            f"\nKolleksiyalar ({len(col_names)}): {', '.join(col_names) if col_names else '—'}"
        )
        return _result_to_mcp(
            text,
            is_error=False,
            data={
                "project_id": ws.project_id,
                "project_name": ws.project_name,
                "nodes": nodes,
                "edges": edges,
                "flow_dirty": ws.flow_dirty,
                "collections": col_names,
            },
        )


async def _handle_session_undo(user: AuthedUser) -> mtypes.CallToolResult:
    store = get_workspace_store()
    async with store.lock(user.user_id):
        ws = store.get(user.user_id)
        if not ws.project_id:
            return _result_to_mcp(
                "Aktiv loyiha yo'q — session_use_project(project_id) chaqiring.",
                is_error=True,
                data=None,
            )
        snap = ws.pop_snapshot()
        if snap is None:
            return _result_to_mcp(
                "Qaytariladigan snapshot yo'q (flow_apply_template/flow_clear/"
                "flow_save dan keyin paydo bo'ladi).",
                is_error=False,
                data={"restored": False},
            )
        await _persist_draft(ws)
        flow = ws.working_flow or {}
        nodes = len(flow.get("nodes") or [])
        edges = len(flow.get("edges") or [])
        return _result_to_mcp(
            f"`{snap.get('label')}` amalidan oldingi holatga qaytarildi: "
            f"{nodes} node, {edges} edge. ({len(ws.snapshots)} snapshot qoldi.)",
            is_error=False,
            data={"restored": True, "label": snap.get("label"), "nodes": nodes, "edges": edges},
        )


async def _execute_registry_tool(
    reg_name: str,
    arguments: dict,
    user: AuthedUser,
    registry: ToolRegistry,
) -> mtypes.CallToolResult:
    store = get_workspace_store()
    # #2 — entire lookup + execute + merge is atomic per user so parallel
    # flow_patch calls can't see a half-built workspace ("Aktiv loyiha yo'q").
    async with store.lock(user.user_id):
        ws = store.get(user.user_id)
        ws.sync_auth_claims(user.claims)

        # Flow tool'lari aktiv loyihani talab qiladi.
        if reg_name.startswith("flow.") and not ws.project_id:
            return _result_to_mcp(
                "Aktiv loyiha yo'q. session_use_project(project_id) chaqiring — "
                "saqlanmagan draft bo'lsa avtomatik tiklanadi.",
                is_error=True,
                data=None,
            )

        # #11 — snapshot before destructive ops so session_undo can restore.
        if reg_name in _SNAPSHOT_BEFORE and ws.working_flow is not None:
            ws.push_snapshot(reg_name)

        state = ws.to_tool_state()
        ctx = ToolContext(
            run_id=f"mcp-{uuid.uuid4().hex[:12]}",
            project_id=ws.project_id,
            user_id=user.user_id,
            state=state,
            services={"backend": get_backend_client()},
        )
        result = await registry.execute(reg_name, arguments, ctx)
        # state_updates'ni workspace'ga qaytarib singdiramiz.
        mutated = ws.is_mutated(result.state_updates)
        ws.merge_state_updates(result.state_updates)

        # #1 — write-through persist whenever the draft changed (and on
        # successful flow.save clear the persisted draft / dirty flag).
        if reg_name == "flow.save" and not result.is_error:
            ws.flow_dirty = False
            await get_draft_store().clear(user_id=user.user_id, project_id=ws.project_id)
        elif mutated and not result.is_error:
            await _persist_draft(ws)

        content = result.content
        # #14 — recovery hint for destructive ops.
        if reg_name in ("flow.apply_template", "flow.clear") and not result.is_error:
            content = (content or "") + "\n(session_undo bilan qaytarish mumkin.)"
        return _result_to_mcp(content, is_error=result.is_error, data=result.data)


# ── Low-level MCP Server qurish ──────────────────────────────────────────────


def build_mcp_server(registry: ToolRegistry | None = None) -> Server:
    registry = registry or build_default_registry()
    name_map = exported_tools(registry)  # mcp_name → reg_name

    server: Server = Server("botmother-agent", instructions=_load_instructions())

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
        if name == _SESSION_UNDO:
            return await _handle_session_undo(user)
        if name == _DOCS_GUIDE:
            topic = str(arguments.get("topic") or "").strip()
            if topic not in _GUIDE_TOPICS:
                return _result_to_mcp(
                    f"Noma'lum topic `{topic}`. Mavjud: {sorted(_GUIDE_TOPICS)}",
                    is_error=True,
                    data=None,
                )
            return _result_to_mcp(_load_guide(topic), is_error=False, data=None)
        reg_name = name_map.get(name)
        if reg_name is None:
            return _result_to_mcp(
                f"Noma'lum tool `{name}`. Keyingi qadam: mavjud tool'lar ro'yxatini "
                "(tools/list) ko'ring; flow tahriri uchun avval "
                "session_use_project(project_id) chaqiring.",
                is_error=True,
                data=None,
            )
        return await _execute_registry_tool(reg_name, arguments, user, registry)

    # ── Prompts — klient foydalanuvchisi /mcp__botspace__build_bot bilan to'liq
    # qo'llanmani kontekstga tortadi (tool chaqirig'ini kutmasdan).
    @server.list_prompts()
    async def list_prompts() -> list[mtypes.Prompt]:
        return [
            mtypes.Prompt(
                name="build_bot",
                description=(
                    "Botmother'da Telegram bot qurish uchun to'liq qo'llanma "
                    "(flow qoidalari + node katalogi + misollar) bilan vazifani "
                    "boshlaydi."
                ),
                arguments=[
                    mtypes.PromptArgument(
                        name="task",
                        description="Qanday bot kerak — qisqacha tavsif",
                        required=False,
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> mtypes.GetPromptResult:
        if name != "build_bot":
            raise ValueError(f"Noma'lum prompt: {name}")
        task = (arguments or {}).get("task", "").strip()
        guide = "\n\n---\n\n".join(
            [_load_guide("flow"), _load_guide("node_catalog"), _load_guide("flow_examples")]
        )
        text = (
            "Quyida Botmother bot-builder platformasining to'liq qo'llanmasi. "
            "Shu qoidalarga amal qilib bot qur.\n\n" + guide
        )
        if task:
            text += f"\n\n---\n\n## Vazifa\n{task}"
        return mtypes.GetPromptResult(
            description="Botmother bot qurish qo'llanmasi",
            messages=[
                mtypes.PromptMessage(
                    role="user",
                    content=mtypes.TextContent(type="text", text=text),
                )
            ],
        )

    return server


# ── Starlette ASGI app (auth middleware + StreamableHTTP mount) ───────────────


def protected_resource_metadata() -> dict:
    """RFC 9728 Protected Resource Metadata — MCP klient OAuth discovery uchun.

    Klient 401'dagi `WWW-Authenticate: ... resource_metadata=...` URL'idan shu
    hujjatni olib, `authorization_servers`dagi backend bilan OAuth flow boshlaydi.
    """
    from ..config import get_settings

    s = get_settings()
    base = s.public_base_url.rstrip("/")
    return {
        "resource": f"{base}/mcp",
        "authorization_servers": [s.oauth_issuer.rstrip("/")],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp"],
    }


def _www_authenticate_value() -> str:
    """401 uchun RFC 9728 §5.1 bo'yicha `WWW-Authenticate` qiymati."""
    from ..config import get_settings

    base = get_settings().public_base_url.rstrip("/")
    return f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'


class _MCPApp:
    """`/mcp`ga mount qilinadigan ASGI app — auth + StreamableHTTP handler.

    Streamable HTTP'da har request `session_manager.handle_request` orqali
    yo'naltiriladi. Biz oldin `Authorization` header'ini tekshirib, AuthedUser'ni
    contextvar'ga o'rnatamiz; auth yiqilsa 401 qaytaramiz (MCP so'rovi backend'ga
    umuman bormaydi). 401'dagi `WWW-Authenticate` MCP klientda OAuth discovery'ni
    boshlaydi (Authenticate tugmasi).
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
                headers={"WWW-Authenticate": _www_authenticate_value()},
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


class MCPPathNormalizer:
    """`/mcp` (slashsiz) → `/mcp/` ASGI darajasida rewrite — redirect'siz.

    Starlette Mount slashsiz path'ga 307 redirect qaytaradi va TLS-terminating
    gateway ortida `Location` scheme'i `http://` bo'lib qoladi — MCP klientlar
    (Claude Code) https→http downgrade redirect'ni rad etadi. Path'ni routing'dan
    OLDIN to'g'irlash redirect'ni butunlay yo'q qiladi.
    """

    def __init__(self, app, prefix: str = "/mcp") -> None:
        self.app = app
        self._prefix = prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") == self._prefix:
            scope = dict(scope)
            scope["path"] = self._prefix + "/"
            raw = scope.get("raw_path")
            if isinstance(raw, bytes):
                scope["raw_path"] = raw + b"/"
        await self.app(scope, receive, send)


def create_standalone_app(registry: ToolRegistry | None = None) -> Starlette:
    """`python -m agent.mcp` uchun mustaqil Starlette app — `/mcp` mount."""
    asgi_app, session_manager = build_mcp_asgi_app(registry)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with mcp_lifespan(session_manager):
            yield

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "agent-mcp"})

    async def resource_metadata(_: Request) -> Response:
        return JSONResponse(protected_resource_metadata())

    from starlette.routing import Route

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health),
            Route("/.well-known/oauth-protected-resource/mcp", resource_metadata),
            Mount("/mcp", app=asgi_app),
        ],
        lifespan=lifespan,
    )
    app.add_middleware(MCPPathNormalizer)
    return app


__all__ = [
    "MCPPathNormalizer",
    "build_mcp_asgi_app",
    "build_mcp_server",
    "build_session_manager",
    "create_standalone_app",
    "exported_tools",
    "is_excluded",
    "mcp_lifespan",
    "mcp_tool_name",
    "protected_resource_metadata",
]
