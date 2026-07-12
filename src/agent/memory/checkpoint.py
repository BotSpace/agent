"""LangGraph async Postgres checkpointer with safe fallback to MemorySaver.

Worker.run_agent uses `await graph.ainvoke(...)`, so the checkpointer must be
async-capable. We use `AsyncPostgresSaver.from_conn_string(...)` which yields an
async context manager — we enter it once at startup and hold the handle.

For tests / dev without Postgres we return `None` and let `build_graph` fall
back to in-memory.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..config import normalize_pg_url as _normalize


async def create_async_checkpointer(database_url: str | None = None) -> Any | None:
    url = database_url or get_settings().database_url
    if not url or not url.startswith(("postgresql://", "postgresql+psycopg://")):
        return None
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "AsyncPostgresSaver requires `langgraph-checkpoint-postgres`."
        ) from exc

    normalized = _normalize(url)
    manager = AsyncPostgresSaver.from_conn_string(normalized)
    cp = await manager.__aenter__()
    try:
        cp._manager = manager
    except Exception:
        pass
    await cp.setup()
    return cp


async def close_async_checkpointer(checkpointer: Any | None) -> None:
    manager = getattr(checkpointer, "_manager", None)
    if manager is not None and hasattr(manager, "__aexit__"):
        await manager.__aexit__(None, None, None)
