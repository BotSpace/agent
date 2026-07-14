"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from ..config import Settings, get_settings, validate_security_config
from ..logging import configure_logging
from ..runtime.store import THINKING_EFFORTS, RunStore, get_run_store
from ..tools import build_default_registry
from .deps import CurrentUser, get_current_user, get_store


class ProjectSettingsUpdate(BaseModel):
    thinking_effort: str | None = None


async def _max_thinking_allowed(user: "CurrentUser") -> bool:
    """Max reasoning effort is a PAID-plan feature. Returns True for paid plans
    (pro/max) and for internal callers; fail-open (True) on a backend hiccup so a
    transient error never blocks a legitimately-paid user."""
    if user.user_id == "__internal__":
        return True
    try:
        from ..integrations.backend import get_backend_client

        pu = await get_backend_client().subscription_usage(auth_claims=user.claims)
        plan = (pu.get("ai_token_usage") or {}).get("plan") or {}
        name = (plan.get("name") if isinstance(plan, dict) else str(plan)) or ""
        return name.strip().lower() != "free"
    except Exception:
        logger.warning("plan check failed; allowing max (fail-open)")
        return True
from .routes.runs import router as runs_router
from .routes.threads import router as threads_router

logger = logging.getLogger(__name__)


async def _template_sync_loop(initial_delay: int, interval_hours: float) -> None:
    """Run agent.scripts.ingest_templates on a repeating schedule.

    Catches all errors per-cycle so one failed sync doesn't kill the
    loop. Logs the result of each cycle.
    """
    if interval_hours <= 0:
        return
    interval_seconds = max(60.0, interval_hours * 3600.0)
    # Lazy import — pgvector connection happens lazily in the script.
    from ..scripts.ingest_templates import ingest

    try:
        await asyncio.sleep(max(0, initial_delay))
        while True:
            try:
                result = await ingest(rebuild=False)
                logger.info(
                    "Template sync cycle ok: fetched=%s indexed=%s",
                    result.get("fetched"),
                    result.get("indexed"),
                )
            except Exception:
                logger.exception("Template sync cycle failed; will retry next interval")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("Template sync loop cancelled (shutdown)")
        raise


async def _module_sync_loop(initial_delay: int, interval_hours: float) -> None:
    """Run agent.scripts.ingest_modules on a repeating schedule.

    Sibling of `_template_sync_loop`: keeps the module-catalog vector
    index fresh so `modules.search` reflects newly-listed (or de-listed)
    platform modules without a manual re-ingest. Per-cycle errors are
    swallowed so one bad sync doesn't kill the loop.
    """
    if interval_hours <= 0:
        return
    interval_seconds = max(60.0, interval_hours * 3600.0)
    # Lazy import — pgvector connection + embedder happen lazily inside.
    from ..scripts.ingest_modules import ingest

    try:
        await asyncio.sleep(max(0, initial_delay))
        while True:
            try:
                result = await ingest(rebuild=False)
                logger.info(
                    "Module sync cycle ok: fetched=%s indexed=%s",
                    result.get("fetched"),
                    result.get("indexed"),
                )
            except Exception:
                logger.exception("Module sync cycle failed; will retry next interval")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("Module sync loop cancelled (shutdown)")
        raise


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging(s.log_level)

    # Fail-fast: never boot with an insecure/misconfigured JWT setup (wrong
    # scheme, missing RS256 public key, or a placeholder session secret).
    validate_security_config(s)

    # MCP server (`/mcp`) — Streamable HTTP, stateful (per-connection session). session_manager.run()
    # lifespan'ga ulanishi SHART (aks holda handle_request RuntimeError beradi).
    mcp_asgi_app = None
    mcp_session_manager = None
    if s.mcp_enabled:
        try:
            from ..mcp.server import build_mcp_asgi_app

            mcp_asgi_app, mcp_session_manager = build_mcp_asgi_app()
        except Exception:
            logger.exception("MCP server qurib bo'lmadi — /mcp o'chiq qoldi")
            mcp_asgi_app = None
            mcp_session_manager = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with AsyncExitStack() as stack:
            if mcp_session_manager is not None:
                from ..mcp.server import mcp_lifespan

                await stack.enter_async_context(mcp_lifespan(mcp_session_manager))
            async with _core_lifespan():
                yield

    @asynccontextmanager
    async def _core_lifespan():
        # Bootstrap database schema for runs + vector store
        store = get_run_store()
        await store.bootstrap()
        # A redeploy/restart kills in-flight asyncio run tasks, but their DB status
        # stays "running"/"queued" — the UI then shows them stuck forever. Fail
        # those orphans on startup (paused waiting_* runs are left alone).
        try:
            n = await store.recover_orphaned_runs()
            if n:
                logger.info("recovered %d orphaned run(s) on startup", n)
        except Exception as exc:
            logger.warning("orphaned-run recovery failed: %s", exc)
        if s.rag_enabled:
            try:
                from ..memory import get_vector_store

                vs = get_vector_store()
                await vs.bootstrap()
            except Exception:
                # vector store optional — bootstrap failures shouldn't kill the API
                pass
        # Background scheduler: periodically refresh the template vector
        # index so newly-added templates become searchable without a
        # manual re-deploy. AGENT_TEMPLATE_SYNC_INTERVAL_HOURS=0 disables.
        sync_tasks: list[asyncio.Task] = []
        if s.rag_enabled and s.template_sync_interval_hours > 0:
            sync_tasks.append(
                asyncio.create_task(
                    _template_sync_loop(
                        s.template_sync_initial_delay_seconds,
                        s.template_sync_interval_hours,
                    ),
                    name="template-sync-loop",
                )
            )
        # Module-catalog RAG: same scheduler, separate cadence so
        # `modules.search` stays current. AGENT_MODULE_SYNC_INTERVAL_HOURS=0
        # disables it independently of the template loop.
        if s.rag_enabled and s.module_sync_interval_hours > 0:
            sync_tasks.append(
                asyncio.create_task(
                    _module_sync_loop(
                        s.module_sync_initial_delay_seconds,
                        s.module_sync_interval_hours,
                    ),
                    name="module-sync-loop",
                )
            )
        try:
            yield
        finally:
            for task in sync_tasks:
                task.cancel()
            for task in sync_tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    app = FastAPI(title="Botmother agent API", version="0.1.0", lifespan=lifespan)
    # Reflect ANY origin: browser MCP clients (e.g. the Inspector on
    # localhost:6274) reach /mcp and the /.well-known/* OAuth discovery docs
    # cross-origin from arbitrary origins, and the resource-server metadata is
    # useless if the browser blocks it on CORS. Safe because the agent
    # authenticates via the Authorization bearer header only (never cookies),
    # which browsers don't attach cross-origin — so an attacker origin still
    # can't make an authenticated call.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins_list,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Per-IP rate limiting + access logging (added after CORS → they wrap it, so
    # AccessLog is outermost and records the final status incl. 429s). /health,
    # /metrics, /.well-known, /mcp are exempt from the limiter.
    from .middleware import AccessLogMiddleware, RateLimitMiddleware

    if s.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware, limit=s.rate_limit_per_min, window_sec=60.0
        )
    if s.access_log_enabled:
        app.add_middleware(AccessLogMiddleware)
    app.state.settings = s
    app.state.registry = build_default_registry()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "agent", "provider": s.llm_provider}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/tools")
    async def list_tools() -> dict[str, object]:
        specs = [
            {
                "name": t.name,
                "description": t.description,
                "destructive": t.destructive,
                "input_schema": t.to_spec().input_schema,
            }
            for t in app.state.registry.all()
        ]
        return {"count": len(specs), "tools": specs}

    app.include_router(runs_router)
    app.include_router(threads_router)

    # MCP Streamable HTTP — auth o'z middleware'ida (Authorization header), shu
    # sabab FastAPI dependency'siz to'g'ridan-to'g'ri mount qilinadi.
    # MCPPathNormalizer `/mcp`ni (slashsiz) routing'dan oldin `/mcp/`ga rewrite
    # qiladi — aks holda Mount 307 redirect qaytaradi va TLS gateway ortida
    # Location `http://` bo'lib MCP klientlar (Claude Code) uzilib qoladi.
    if mcp_asgi_app is not None:
        from ..mcp.server import MCPPathNormalizer, protected_resource_metadata

        app.mount("/mcp", mcp_asgi_app)
        app.add_middleware(MCPPathNormalizer)

        # RFC 9728 — MCP klientlar OAuth discovery shu metadata'dan boshlaydi
        # (401'dagi WWW-Authenticate shu URL'ga ko'rsatadi).
        @app.get("/.well-known/oauth-protected-resource/mcp")
        async def oauth_protected_resource() -> dict:
            return protected_resource_metadata()

        # Some MCP clients probe the path-less form too (RFC 9728 allows both).
        @app.get("/.well-known/oauth-protected-resource")
        async def oauth_protected_resource_root() -> dict:
            return protected_resource_metadata()

    @app.get("/projects/{project_id}/settings")
    async def get_project_settings(
        project_id: str,
        user: CurrentUser = Depends(get_current_user),
        store: RunStore = Depends(get_store),
    ) -> dict:
        effort = await store.get_project_thinking_effort(project_id)
        return {"project_id": project_id, "thinking_effort": effort}

    @app.put("/projects/{project_id}/settings")
    async def set_project_settings(
        project_id: str,
        body: ProjectSettingsUpdate,
        user: CurrentUser = Depends(get_current_user),
        store: RunStore = Depends(get_store),
    ) -> dict:
        effort = body.thinking_effort
        if effort is not None and effort not in THINKING_EFFORTS:
            raise HTTPException(
                status_code=422,
                detail=f"thinking_effort must be one of {THINKING_EFFORTS} or null",
            )
        if effort == "max" and not await _max_thinking_allowed(user):
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Max fikrlash faqat pullik tarifda. Tarifni yangilang.",
                    "code": "plan.upgrade_required",
                },
            )
        # null/None clears the override → provider default; store as-is via upsert.
        await store.set_project_thinking_effort(project_id, effort or "")
        return {"project_id": project_id, "thinking_effort": effort}

    @app.get("/usage")
    async def usage(
        user: CurrentUser = Depends(get_current_user),
    ) -> dict:
        from ..integrations.backend import get_backend_client

        backend = get_backend_client()
        try:
            plan_usage = await backend.subscription_usage(auth_claims=user.claims)
            usage = plan_usage.get("ai_token_usage") or {}
        except Exception as exc:
            # Never 500 the usage widget on a backend hiccup — return unknown.
            logger.warning("usage fetch failed: %s", exc)
            usage = {}
        plan = usage.get("plan") or {}
        plan_name = (plan.get("name") if isinstance(plan, dict) else str(plan)) or ""
        # Max reasoning effort is paid-only; free plan (or internal) → locked in UI.
        max_thinking_allowed = user.user_id == "__internal__" or plan_name.strip().lower() != "free"
        return {
            "user_id": user.user_id,
            "limit": usage.get("limit", 0),
            "used": usage.get("used", 0),
            "remaining": usage.get("remaining", 0),
            "input_used": usage.get("input_used", 0),
            "output_used": usage.get("output_used", 0),
            "plan": usage.get("plan"),
            "status": usage.get("status"),
            "token_reset_at": usage.get("token_reset_at"),
            "max_thinking_allowed": max_thinking_allowed,
        }

    return app
