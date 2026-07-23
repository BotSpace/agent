"""Per-project saved regression tests (#10) — Postgres-backed.

A user can save a named simulate scenario (steps, with `expect` assertions) as a
regression test for a project, then re-run all of them after each change so a
later edit can't silently break an earlier scenario.

The table is bootstrapped on first use. Database failures are propagated so a
caller can distinguish an outage from an actually empty test collection.

`name` is UNIQUE per project — `save` UPSERTs (re-saving the same name replaces
its steps). `steps` is the simulate step format (with optional `expect`).
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS mcp_flow_tests (
    id          BIGSERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL,
    user_id     TEXT,
    name        TEXT NOT NULL,
    steps       JSONB NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, name)
);
"""


from ..config import get_settings
from ..config import normalize_pg_url as _normalize
from ._lazy_store import LazyStore


class FlowTestStore:
    """Postgres-backed per-project saved regression tests."""

    def __init__(self, dsn: str) -> None:
        self._dsn = _normalize(dsn)
        self._bootstrapped = False

    async def _bootstrap(self) -> None:
        if self._bootstrapped:
            return
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA)
            await conn.commit()
        self._bootstrapped = True

    async def save(
        self,
        *,
        project_id: str,
        user_id: str | None,
        name: str,
        steps: list[dict[str, Any]],
        enabled: bool = True,
    ) -> bool:
        """UPSERT a named test for the project. Returns True if persisted."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO mcp_flow_tests
                        (project_id, user_id, name, steps, enabled,
                         created_at, updated_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s, NOW(), NOW())
                    ON CONFLICT (project_id, name) DO UPDATE
                        SET steps = EXCLUDED.steps,
                            enabled = EXCLUDED.enabled,
                            user_id = EXCLUDED.user_id,
                            updated_at = NOW()
                    """,
                    (
                        project_id,
                        user_id,
                        name,
                        json.dumps(steps),
                        bool(enabled),
                    ),
                )
            await conn.commit()
        return True

    async def list(self, *, project_id: str, only_enabled: bool = False) -> list[dict[str, Any]]:
        """Return tests for the project: {name, steps, enabled, updated_at}."""
        await self._bootstrap()
        sql = "SELECT name, steps, enabled, updated_at FROM mcp_flow_tests WHERE project_id = %s"
        if only_enabled:
            sql += " AND enabled = TRUE"
        sql += " ORDER BY name ASC"
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (project_id,))
                rows = await cur.fetchall()
        return [
            {
                "name": r[0],
                "steps": r[1] if isinstance(r[1], list) else [],
                "enabled": bool(r[2]),
                "updated_at": r[3].isoformat() if r[3] is not None else None,
            }
            for r in rows
        ]

    async def get(self, *, project_id: str, name: str) -> dict[str, Any] | None:
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT name, steps, enabled FROM mcp_flow_tests WHERE project_id = %s AND name = %s",
                    (project_id, name),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "steps": row[1] if isinstance(row[1], list) else [],
            "enabled": bool(row[2]),
        }

    async def delete(self, *, project_id: str, name: str) -> bool:
        """Delete a test. Returns True if a row was removed."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM mcp_flow_tests WHERE project_id = %s AND name = %s",
                    (project_id, name),
                )
                deleted = cur.rowcount
            await conn.commit()
        return bool(deleted)


_store: LazyStore[FlowTestStore] = LazyStore(lambda: FlowTestStore(get_settings().database_url))


def get_flow_test_store() -> FlowTestStore:
    return _store.get()


def set_flow_test_store(store: FlowTestStore | None) -> None:
    """Test hook — inject a fake/None store."""
    _store.reset(store)
