"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import Settings, get_settings, validate_security_config
from ..logging import configure_logging
from ..runtime.store import get_run_store
from ..tools import build_default_registry
from .deps import CurrentUser, get_current_user
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins_list,
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

    @app.get("/usage")
    async def usage(
        user: CurrentUser = Depends(get_current_user),
    ) -> dict:
        from ..integrations.backend import get_backend_client

        backend = get_backend_client()
        plan_usage = await backend.subscription_usage(auth_claims=user.claims)
        usage = plan_usage.get("ai_token_usage") or {}
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
        }

    return app
