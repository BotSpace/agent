"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, get_settings
from ..logging import configure_logging
from ..runtime.store import get_run_store
from ..tools import build_default_registry
from fastapi import Depends

from .deps import CurrentUser, get_current_user, get_store
from .routes.runs import router as runs_router


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging(s.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
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
        yield

    app = FastAPI(title="Botmother AgentV3 API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = s
    app.state.registry = build_default_registry()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "agent", "provider": s.llm_provider}

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

    @app.get("/usage")
    async def usage(
        user: CurrentUser = Depends(get_current_user),
        store=Depends(get_store),
    ) -> dict:
        used = await store.count_user_runs_today(user.user_id)
        return {
            "user_id": user.user_id,
            "used_today": used,
            "limit": user.prompt_limit,
            "remaining": max(0, user.prompt_limit - used),
        }

    return app
