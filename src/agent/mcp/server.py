"""MCP server — agent tool registry'sini Model Context Protocol orqali ochadi.

Gibrid transport: rasmiy `mcp` SDK low-level `Server` + `StreamableHTTPSessionManager`
(`stateless=False` — per-connection Mcp-Session-Id; `json_response=True` — LB-friendly). Ikki ishga tushirish:
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
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mcp.types as mtypes
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import request_ctx as sdk_request_ctx
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..integrations.backend import (
    get_backend_client,
    load_project_collections,
    load_project_module_nodes,
    load_saved_flow,
)
from ..tools import ToolContext, build_default_registry
from ..tools.registry import ToolRegistry
from . import trace
from .auth import AuthedUser, MCPAuthError, authenticate
from .draft_store import get_draft_store
from .workspace import Workspace, WorkspaceStore, get_workspace_store

logger = logging.getLogger(__name__)

# MCP'da ko'rsatilmaydigan agent-ichki namespace/tool'lar.
_EXCLUDED_NAMESPACES = frozenset({"plan", "task"})
_EXCLUDED_NAMES = frozenset({"ask_user"})

# Read-only deb hisoblanadigan operatsiya nomlari (oxirgi segment bo'yicha).
_READ_ONLY_VERBS = (
    "get", "list", "diff", "validate", "search", "export", "history", "snapshot",
    "errors",
)

# Joriy request'ning autentifikatsiyalangan foydalanuvchisi (Starlette middleware
# o'rnatadi, tool handler o'qiydi).
_current_user: contextvars.ContextVar[AuthedUser | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)

# Stateful transportda har ulanish (session) uchun SDK BITTA ServerSession
# obyektini yaratadi va tool handler'lar aynan shu sessiya kontekstida ishlaydi.
# Shu obyektdan barqaror per-connection client kalitini yasaymiz — ikki terminal /
# sub-agent bir xil token bilan ulansa ham har ulanish alohida kalit oladi.
# WeakKeyDictionary: sessiya obyekti o'chsa yozuv ham o'chadi (id() qayta
# ishlatilish xavfisiz).
_SESSION_KEYS: weakref.WeakKeyDictionary[object, str] = weakref.WeakKeyDictionary()


def _session_client_key() -> str | None:
    """SDK request kontekstidagi sessiya obyektidan per-connection kalit.

    Tool/prompt handler ichida chaqirilganda SDK `request_ctx`ni o'rnatgan
    bo'ladi (sessiya task'i ichida) — bu HTTP-header yondashuvidagi contextvar
    tuzog'iga uchramaydi. Kontekst tashqarisida (yoki sessiz holatda) None —
    chaqiruvchi token-hash fallback'ka qaytadi."""
    try:
        sess = sdk_request_ctx.get().session
    except LookupError:
        return None
    if sess is None:
        return None
    key = _SESSION_KEYS.get(sess)
    if key is None:
        key = f"sid-{uuid.uuid4().hex[:16]}"
        _SESSION_KEYS[sess] = key
    return key


# MCP-only session tool'lari — registry'da yo'q, server o'zi boshqaradi.
_SESSION_USE_PROJECT = "session_use_project"
_SESSION_STATUS = "session_status"
_SESSION_UNDO = "session_undo"
_SESSION_LOG_PROMPT = "session_log_prompt"
_DOCS_GUIDE = "docs_guide"
# Power tool'lar (flow.simulate / export / import / history / rollback) endi
# ToolRegistry darajasida — registry-export orqali MCP'ga avtomatik chiqadi
# (flow_simulate, flow_export, ...). Ular uchun alohida MCP handler YO'Q.

# Transparent session-restore notice (#5) — prepended to the result text.
_RESTORE_NOTICE = "ℹ️ Sessiya avtomatik tiklandi (loyiha {pid})"  # noqa: RUF001

# Registry tool'lar — bajarilishidan OLDIN avtomatik undo snapshot olinadi (#11).
# flow.import / flow.rollback ham working_flow'ni butunlay almashtiradi — shu
# sabab ular ham snapshot oladi (session_undo bilan qaytarish mumkin).
_SNAPSHOT_BEFORE = frozenset(
    {"flow.apply_template", "flow.apply_recipe", "flow.clear", "flow.save", "flow.import", "flow.rollback"}
)


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


async def _auto_restore_session(
    store: WorkspaceStore, user: AuthedUser
) -> str | None:
    """Transparent session recovery (#5).

    After a pod restart / TTL expiry the per-client active-project pointer is
    gone, so flow tools fail with "Aktiv loyiha yo'q". Here we look up THIS
    user's most-recently-updated persisted draft, re-activate that project for
    this client and hydrate the workspace from the draft, so work continues
    seamlessly. Returns the recovered project_id (caller prepends a one-line
    notice), or None if nothing to restore. MUST be called with the user lock
    held. Explicit session_use_project always wins (it sets the pointer, so this
    only fires when the pointer is empty)."""
    try:
        draft = await get_draft_store().load_latest(user_id=user.user_id)
    except AttributeError:
        # Older/fake stores without load_latest — degrade silently.
        return None
    except Exception:  # pragma: no cover — defensive
        return None
    if not draft or not draft.get("project_id"):
        return None
    project_id = str(draft["project_id"])
    ws = store.set_active(user.user_id, project_id, client_key=user.client_key)
    ws.sync_auth_claims(user.claims)
    _hydrate_from_draft(ws, draft)
    ws.project_id = project_id
    return project_id

# docs_guide topic → (sarlavha, kontent loader). Kontent agent'ning ICHKI
# agentlariga beriladigan system prompt'lar — MCP klient modeli ham xuddi shu
# bilimga ega bo'lishi uchun tool orqali ochib beriladi.
_GUIDE_TOPICS = {
    "all": "Barcha asosiy qo'llanmalar bitta javobda (har yangi bot/non-trivial edit oldidan majburiy)",
    "flow": "Flow-builder qo'llanmasi (handles, trigger, AI nodes, keyboard, state)",
    "flow_examples": "To'liq bot misollari (flow JSON'lari bilan)",
    "node_catalog": "Barcha node turlari katalogi (kategoriya, handle, maydonlar)",
    "cms": "CMS kolleksiyalari bilan ishlash qo'llanmasi",
    "crm": "CRM (chat tarixi, broadcast) qo'llanmasi",
    "deploy": "Deploy/release qo'llanmasi",
    "simulate": "Bot simulyatsiyasi qo'llanmasi (flow_simulate step turlari, expect/assert, multi-user, regression testlar)",
    "custom_code": "CustomCodeNode JS API to'liq ma'lumotnoma (send/edit_or_send, db_*, callback_query.message.message_id, pagination edit-in-place namunasi)",
    "modules": "Tashqi MODULE tizimi: modul node turlari (moduleId.NodeName), modul triggerlari (+global), dinamik credential type'lar, Result.Error xato surface, self-hosted modullar",
    "i18n": "Ko'p tilli (i18n) botlar: __i18n marker obyekt, languages/defaultLanguage, flow_set_languages, engine til tanlash tartibi, custom code t()/lang",
}


def _load_guide(topic: str) -> str:
    from ..flow.manifest_compiler.node_catalog import render_compact_catalog
    from ..prompts.loader import load_prompt

    if topic == "all":
        # MCP klient natija limiti (~25k token) uchun IXCHAM "start here":
        # kritik qoidalar digest'i + node katalog + kichik guide'lar inline.
        # 3 ta KATTA deep-dive (flow 18k, custom_code 4k, flow_examples 4k token)
        # shu yerga sig'maydi → ko'rsatkich orqali alohida docs_guide bilan o'qiladi.
        parts = [
            "# Botmother bot qurish — START HERE (ixcham).\n"
            "Asosiy qoidalar + node katalog + kichik bo'limlar shu yerda. KATTA "
            "bo'limlar (flow/custom_code/flow_examples) limitga sig'maydi — pastdagi "
            "ko'rsatkich bo'yicha alohida `docs_guide(topic=...)` bilan o'qing.",
            load_prompt("mcp/build_rules_digest.txt").strip(),
            "# node_catalog\n\n" + _load_guide("node_catalog"),
        ]
        for name in ("simulate", "modules", "i18n", "cms", "crm", "deploy"):
            parts.append(f"# {name}\n\n{_load_guide(name)}")
        parts.append(
            "# CHUQURROQ — alohida o'qing (katta, sig'maydi):\n"
            "- `docs_guide(topic='flow')` — to'liq qurish qoidalari: handle'lar, "
            "trigger ustunligi, AI nodes, keyboard/button-edge, state, patternlar.\n"
            "- `docs_guide(topic='custom_code')` — CustomCodeNode JS API "
            "(send_*/db_*/http_*/exit/format), media, pagination namunasi.\n"
            "- `docs_guide(topic='flow_examples')` — to'liq bot misollari (flow JSON)."
        )
        return "\n\n---\n\n".join(parts)
    if topic == "node_catalog":
        return render_compact_catalog() + _MODULE_CATALOG_NOTE + _I18N_CATALOG_NOTE
    files = {
        "flow": "subagents/flow_builder.txt",
        "flow_examples": "subagents/flow_builder_examples.txt",
        "cms": "subagents/cms_manager.txt",
        "crm": "subagents/crm_manager.txt",
        "deploy": "subagents/deploy_manager.txt",
        "simulate": "mcp/guide_simulate.txt",
        "custom_code": "mcp/guide_custom_code.txt",
        "modules": "mcp/guide_modules.txt",
        "i18n": "mcp/guide_i18n.txt",
    }
    return load_prompt(files[topic])


# Appended to the static node_catalog so the agent learns that, beyond the
# built-in types listed above, per-project ENABLED modules contribute extra
# `moduleId.NodeName` node types (the live list is in the session context after
# session_use_project, and via flow_list_node_types). See docs_guide("modules").
_MODULE_CATALOG_NOTE = (
    "\n\n### MODUL NODE TURLARI (dinamik, per-project)\n"
    "Yuqoridagi built-in turlardan tashqari, loyihada YOQILGAN tashqi modullar "
    "qo'shimcha node turlari beradi — namespace `moduleId.NodeName` "
    "(masalan `openrouter.Chat`, `ollama.Chat`). Bu turlar STATIK emas: ular "
    "loyihaga bog'liq va `session_use_project` paytida yuklanadi. Joriy ro'yxat "
    "session javobida (`module_node_types`), kontekst blokida va "
    "`flow_list_node_types` natijasida ko'rinadi. Qo'shish: "
    "`flow_add_node(type='moduleId.NodeName', data={...})`. Batafsil: "
    "`docs_guide(topic='modules')`.\n"
)

# Lokalizatsiya eslatmasi — yuqoridagi node'lardagi user-facing matn maydonlari
# (messageText/newText, caption, tugma yorliqlari, poll question, venue
# title/address, PaymentNode buttonText) i18n marker obyektni qabul qiladi.
_I18N_CATALOG_NOTE = (
    "\n\n### KO'P TILLI (i18n) MATN\n"
    "Yuqoridagi node'lardagi foydalanuvchiga ko'rinadigan matn maydonlari "
    "(messageText / newText, caption, klaviatura tugma yorliqlari, poll question, "
    'venue title/address, PaymentNode buttonText) marker obyektni qabul qiladi: '
    '{"__i18n": true, "en": "..", "uz": "..", "ru": ".."}. Engine ishga tushganda '
    "aktiv tilga yassilaydi. Ishlashi uchun AVVAL `flow_set_languages(languages, "
    "default_language)` chaqir. Batafsil: `docs_guide(topic='i18n')`.\n"
)


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


def _explicit_project_id(arguments: dict) -> str | None:
    """Tool argumentlaridagi konkret `project_id` (override). Yo'q / placeholder
    (`global`/`none`/bo'sh) bo'lsa None — aktiv-loyiha pointer'i ishlaydi.

    Mos `tools/flow._concrete_project_id` bilan bir xil placeholder ro'yxati."""
    if not isinstance(arguments, dict):
        return None
    raw = arguments.get("project_id")
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in {"", "global", "none", "null"}:
        return None
    return text


def _trace_project_id(user: AuthedUser, arguments: dict) -> str | None:
    """Best-effort active project for a trace record: explicit arg override, else
    the per-client active-project pointer. Read-only and never raises — tracing
    must not affect tool execution."""
    explicit = _explicit_project_id(arguments)
    if explicit is not None:
        return explicit
    try:
        return get_workspace_store().active_project_id(
            user.user_id, client_key=user.client_key
        )
    except Exception:
        return None


def _result_text(result: mtypes.CallToolResult) -> str:
    """Flatten a CallToolResult's text blocks for the trace result_summary."""
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


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
            name=_SESSION_LOG_PROMPT,
            description=(
                "Foydalanuvchining ASL tabiiy-til so'rovini (verbatim, o'zgartirmasdan) "
                "yozib qo'yadi va yangi 'trace' boshlaydi. Har YANGI foydalanuvchi "
                "vazifasini boshlashda BIRINCHI bo'lib chaqir — shundan keyingi barcha "
                "tool chaqiriqlari shu trace'ga bog'lanadi (AI o'qitish ma'lumoti uchun). "
                "trace_id qaytaradi. Yangi chaqiruv — yangi trace (eskisi yopiladi)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Foydalanuvchi so'rovi — verbatim, qisqartirilmagan.",
                    }
                },
                "required": ["prompt"],
            },
            annotations=mtypes.ToolAnnotations(readOnlyHint=True),
        ),
        mtypes.Tool(
            name=_DOCS_GUIDE,
            description=(
                "Botmother bot qurish qo'llanmalarini qaytaradi. Notrivial bot "
                "qurishdan OLDIN topic='all' o'qing — bu IXCHAM start: asosiy "
                "qoidalar + node katalog + kichik bo'limlar. KEYIN, qurayotgan "
                "narsangizga qarab KATTA deep-dive'larni alohida o'qing: "
                "topic='flow' (to'liq qurish qoidalari/handles/AI nodes/patternlar), "
                "topic='custom_code' (CustomCode JS API), topic='flow_examples' "
                "(to'liq misollar). Boshqa: node_catalog, simulate, cms, crm, deploy, modules, i18n."
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
        # Aktiv loyiha pointer'i per-CLIENT (user_id, client_key) — bir user'ning
        # boshqa klienti boshqa loyihada ishlayotgan bo'lsa, unga tegmaymiz.
        ws = store.set_active(user.user_id, project_id, client_key=user.client_key)
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
            # Module node types are dynamic (per-project enabled modules) — always
            # refresh so a freshly-enabled module's nodes appear; keep old list on
            # failure (never raises).
            ws.module_nodes = await load_project_module_nodes(
                project_id, auth_claims=ws.auth_claims, client=backend
            )
            ws.project_id = project_id
            flow = ws.working_flow or {"nodes": [], "edges": []}
        else:
            flow = await load_saved_flow(project_id, auth_claims=ws.auth_claims, client=backend)
            collections = await load_project_collections(
                project_id, auth_claims=ws.auth_claims, client=backend
            )
            module_nodes = await load_project_module_nodes(
                project_id, auth_claims=ws.auth_claims, client=backend
            )
            flow = flow if isinstance(flow, dict) else {"nodes": [], "edges": []}
            ws.project_id = project_id
            ws.working_flow = flow
            ws.baseline_flow = flow
            ws.flow_dirty = False
            ws.collections = collections
            ws.module_nodes = module_nodes
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
        module_types = [
            m.get("type")
            for m in ws.module_nodes
            if isinstance(m, dict) and m.get("type")
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
        if module_types:
            text += (
                f"\nModul node turlari ({len(module_types)}): {', '.join(module_types)}"
                " — bular flow.add_node bilan to'g'ridan-to'g'ri qo'shiladi "
                "(docs_guide(topic='modules'))."
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
                "module_node_types": module_types,
            },
        )


async def _handle_session_status(user: AuthedUser) -> mtypes.CallToolResult:
    store = get_workspace_store()
    # Diagnostika uchun klient identifikatorining qisqartmasi (xom token EMAS).
    client_short = user.client_key[:8] if user.client_key else "—"
    async with store.lock(user.user_id):
        ws = store.get(user.user_id, client_key=user.client_key)
        restored_pid: str | None = None
        if not ws.project_id:
            # #5 — try transparent recovery from the last-edited draft before
            # reporting "no active project".
            restored_pid = await _auto_restore_session(store, user)
            if restored_pid is not None:
                ws = store.get(user.user_id, client_key=user.client_key)
            else:
                return _result_to_mcp(
                    "Aktiv loyiha yo'q. session_use_project(project_id) bilan loyiha "
                    "tanlang — saqlanmagan draft bo'lsa avtomatik tiklanadi.\n"
                    f"(klient: {client_short})",
                    is_error=False,
                    data={"project_id": None, "client_key": client_short},
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
            f"\nKlient: {client_short}"
        )
        if restored_pid is not None:
            text = _RESTORE_NOTICE.format(pid=restored_pid) + "\n" + text
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
                "client_key": client_short,
                "session_restored": restored_pid is not None,
            },
        )


async def _handle_session_undo(user: AuthedUser) -> mtypes.CallToolResult:
    store = get_workspace_store()
    async with store.lock(user.user_id):
        ws = store.get(user.user_id, client_key=user.client_key)
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


async def _handle_session_log_prompt(
    user: AuthedUser, arguments: dict
) -> mtypes.CallToolResult:
    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt:
        return _result_to_mcp("prompt majburiy.", is_error=True, data=None)
    # Open a new Mongo trace and set the per-(user, client) current-trace
    # pointer; every subsequent tool call attaches to this trace_id. The write
    # is fire-and-forget + graceful degrade inside trace.log_prompt, but the
    # pointer (and trace_id) are returned even if Mongo is down.
    trace_id = trace.log_prompt(
        user_id=user.user_id,
        client_key=user.client_key,
        project_id=_trace_project_id(user, {}),
        prompt=prompt,
    )
    return _result_to_mcp(
        f"So'rov yozildi, trace boshlandi (trace_id={trace_id}). Keyingi tool "
        "chaqiriqlari shu trace'ga bog'lanadi.",
        is_error=False,
        data={"trace_id": trace_id},
    )



# ── build-rules digest (server-side guarantee) ───────────────────────────────

# Qurish (flow mutatsiya) tool'lari — birinchisida qoidalar digest'i beriladi.
_BUILD_TOOL_PREFIXES = (
    "flow.add_",
    "flow.ask_",
    "flow.apply_template",
    "flow.apply_recipe",
    "flow.connect",
    "flow.import",
)

# (user_id, client_key) → True. In-memory: pod restartda qayta beriladi — OK.
_rules_served: dict[tuple[str, str], bool] = {}


def _is_build_tool(reg_name: str) -> bool:
    return reg_name.startswith(_BUILD_TOOL_PREFIXES)


def _mark_rules_served(user_id: str, client_key: str) -> bool:
    """True — hali berilmagan (endi beriladi); False — allaqachon berilgan."""
    key = (user_id, client_key)
    if _rules_served.get(key):
        return False
    _rules_served[key] = True
    return True


def _load_build_rules_digest() -> str:
    from ..prompts.loader import load_prompt

    return load_prompt("mcp/build_rules_digest.txt").strip()


async def _execute_registry_tool(
    reg_name: str,
    arguments: dict,
    user: AuthedUser,
    registry: ToolRegistry,
) -> mtypes.CallToolResult:
    store = get_workspace_store()
    # Explicit `project_id` argumenti DOIM per-client aktiv-loyiha pointer'idan
    # ustun (#4) — ko'p tool'larda (cms_*, env_*, release_*, flow_save, ...)
    # optional project_id bor. Berilgan bo'lsa o'sha loyiha workspace'iga
    # to'g'ridan-to'g'ri yo'naltiramiz; aks holda pointer ishlaydi.
    explicit_pid = _explicit_project_id(arguments)
    # #2 — entire lookup + execute + merge is atomic per user so parallel
    # flow_patch calls can't see a half-built workspace ("Aktiv loyiha yo'q").
    restored_pid: str | None = None
    async with store.lock(user.user_id):
        if explicit_pid is not None:
            ws = store.get(user.user_id, explicit_pid, client_key=user.client_key)
        else:
            ws = store.get(user.user_id, client_key=user.client_key)
        ws.sync_auth_claims(user.claims)

        # #5 — transparent session recovery: if no active project (pointer lost
        # after a restart/TTL), restore the user's last-edited draft and carry on
        # instead of erroring. Only when project_id wasn't explicit.
        if not ws.project_id and explicit_pid is None:
            restored_pid = await _auto_restore_session(store, user)
            if restored_pid is not None:
                ws = store.get(user.user_id, client_key=user.client_key)
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
            # Versiyalash endi registry flow.save tool'ining O'ZIDA — LangGraph
            # va MCP save'lari bir xil yo'ldan versiyalanadi (mcp.flow_versions
            # store). Bu yerda takror yozmaymiz.
        elif mutated and not result.is_error:
            await _persist_draft(ws)

        content = result.content
        # #14 — recovery hint for destructive ops.
        if reg_name in ("flow.apply_template", "flow.apply_recipe", "flow.clear") and not result.is_error:
            content = (content or "") + "\n(session_undo bilan qaytarish mumkin.)"
        # #5 — surface the transparent restore on the result.
        if restored_pid is not None:
            content = _RESTORE_NOTICE.format(pid=restored_pid) + "\n" + (content or "")
        # Klient docs_guide o'qimasa ham qoidalar yetib borsin: shu klientning
        # BIRINCHI qurish chaqirig'i natijasiga qisqa qoidalar digest'i qo'shiladi
        # (server-side guarantee — instructions/docs maslahatiga tayanmaymiz).
        if (
            not result.is_error
            and _is_build_tool(reg_name)
            and _mark_rules_served(user.user_id, user.client_key)
        ):
            content = _load_build_rules_digest() + "\n\n" + (content or "")
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

    async def _dispatch(
        name: str, arguments: dict, user: AuthedUser
    ) -> mtypes.CallToolResult:
        if name == _SESSION_USE_PROJECT:
            return await _handle_session_use_project(user, arguments)
        if name == _SESSION_STATUS:
            return await _handle_session_status(user)
        if name == _SESSION_UNDO:
            return await _handle_session_undo(user)
        if name == _SESSION_LOG_PROMPT:
            return await _handle_session_log_prompt(user, arguments)
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

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> mtypes.CallToolResult:
        user = _current_user.get()
        if user is None:
            # Auth middleware har doim o'rnatadi; bu himoya chizig'i.
            return _result_to_mcp("Autentifikatsiya talab qilinadi.", is_error=True, data=None)
        # Per-connection kimlik: SDK sessiya obyektidan (stateful). Token-hash
        # faqat fallback (sessiyasiz chaqiruvlar uchun).
        sid_key = _session_client_key()
        if sid_key:
            user.client_key = sid_key

        started = time.monotonic()
        result = await _dispatch(name, arguments, user)
        # A-band trace (AI training data) — append this call to the open trace
        # (the one session_log_prompt started; orphan if none). Fire-and-forget
        # + graceful degrade inside trace.log_tool_call, so tracing never adds
        # latency to or affects the tool result.
        try:
            trace.log_tool_call(
                user_id=user.user_id,
                client_key=user.client_key,
                project_id=_trace_project_id(user, arguments),
                tool_name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                result_summary=_result_text(result),
                is_error=bool(result.isError),
                duration_ms=int((time.monotonic() - started) * 1000),
                trace_id=trace.current_trace_id(user.user_id, user.client_key),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("MCP trace skipped for %s: %s", name, exc)
        return result

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
        # A-band: record the user's intent as the `__prompt__` trace (the only
        # record carrying `prompt`). Fire-and-forget, graceful degrade.
        if task:
            user = _current_user.get()
            sid_key = _session_client_key()
            if user and sid_key:
                user.client_key = sid_key
            try:
                trace.log_prompt(
                    user_id=user.user_id if user else None,
                    client_key=user.client_key if user else None,
                    project_id=_trace_project_id(user, {}) if user else None,
                    prompt=task,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("MCP prompt trace skipped: %s", exc)
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

        # Per-connection kimlik bu yerda EMAS: stateful rejimda tool handler
        # alohida sessiya task'ida ishlaydi va bu request-kontekst contextvar'i
        # unga yetib bormaydi. client_key sessiyadan handler ichida olinadi —
        # `_session_client_key()` (SDK request_ctx.session). Bu yerdagi user
        # token-hash fallback bilan qoladi.
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
        # Stateful: per-connection Mcp-Session-Id → true client isolation even on
        # a shared token. SAFE while the agent runs a single replica; multi-replica
        # would need session affinity (route a session id to the same pod).
        stateless=False,
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
