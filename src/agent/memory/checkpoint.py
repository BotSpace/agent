"""LangGraph async Postgres checkpointer.

Worker.run_agent uses `await graph.ainvoke(...)`, so the checkpointer must be
async-capable. We use `AsyncPostgresSaver.from_conn_string(...)` which yields an
async context manager — we enter it once at startup and hold the handle.

Missing configuration or dependencies are startup errors.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..config import normalize_pg_url as _normalize


async def create_async_checkpointer(database_url: str | None = None) -> Any:
    url = database_url or get_settings().database_url
    if not url or not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("a PostgreSQL checkpointer URL is required")
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError("AsyncPostgresSaver requires `langgraph-checkpoint-postgres`.") from exc

    normalized = _normalize(url)
    manager = AsyncPostgresSaver.from_conn_string(normalized)
    cp = await manager.__aenter__()
    cp._manager = manager
    await cp.setup()
    return cp


async def close_async_checkpointer(checkpointer: Any | None) -> None:
    manager = getattr(checkpointer, "_manager", None)
    if manager is not None and hasattr(manager, "__aexit__"):
        await manager.__aexit__(None, None, None)
