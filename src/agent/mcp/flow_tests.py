"""Per-project saved regression tests (#10) — Postgres-backed, graceful degrade.

A user can save a named simulate scenario (steps, with `expect` assertions) as a
regression test for a project, then re-run all of them after each change so a
later edit can't silently break an earlier scenario.

Same bootstrap/degrade philosophy as flow_versions.py / draft_store.py:
CREATE TABLE IF NOT EXISTS on first use, one short-lived connection per call
(psycopg async), and every method swallows DB errors so a DB outage can never
break flow_save or the test tools (they just report "tests unavailable").

`name` is UNIQUE per project — `save` UPSERTs (re-saving the same name replaces
its steps). `steps` is the simulate step format (with optional `expect`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

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
    """Postgres-backed per-project saved regression tests, graceful degrade."""

    def __init__(self, dsn: str) -> None:
        self._dsn = _normalize(dsn)
        self._bootstrapped = False
        self._degraded = False

    async def _bootstrap(self) -> bool:
        if self._degraded:
            return False
        if self._bootstrapped:
            return True
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(_SCHEMA)
                await conn.commit()
            self._bootstrapped = True
            return True
        except Exception as exc:
            self._degraded = True
            logger.warning(
                "MCP flow-tests store unavailable, degrading to no-op: %s", exc
            )
            return False

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
        if not await self._bootstrap():
            return False
        try:
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
        except Exception as exc:
            logger.warning(
                "MCP flow-test save failed (project=%s name=%s): %s",
                project_id, name, exc,
            )
            return False

    async def list(
        self, *, project_id: str, only_enabled: bool = False
    ) -> list[dict[str, Any]]:
        """Return tests for the project: {name, steps, enabled, updated_at}."""
        if not await self._bootstrap():
            return []
        sql = (
            "SELECT name, steps, enabled, updated_at FROM mcp_flow_tests "
            "WHERE project_id = %s"
        )
        if only_enabled:
            sql += " AND enabled = TRUE"
        sql += " ORDER BY name ASC"
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (project_id,))
                    rows = await cur.fetchall()
        except Exception as exc:
            logger.warning(
                "MCP flow-test list failed (project=%s): %s", project_id, exc
            )
            return []
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
        if not await self._bootstrap():
            return None
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT name, steps, enabled FROM mcp_flow_tests "
                        "WHERE project_id = %s AND name = %s",
                        (project_id, name),
                    )
                    row = await cur.fetchone()
        except Exception as exc:
            logger.warning(
                "MCP flow-test get failed (project=%s name=%s): %s",
                project_id, name, exc,
            )
            return None
        if not row:
            return None
        return {
            "name": row[0],
            "steps": row[1] if isinstance(row[1], list) else [],
            "enabled": bool(row[2]),
        }

    async def delete(self, *, project_id: str, name: str) -> bool:
        """Delete a test. Returns True if a row was removed."""
        if not await self._bootstrap():
            return False
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM mcp_flow_tests "
                        "WHERE project_id = %s AND name = %s",
                        (project_id, name),
                    )
                    deleted = cur.rowcount
                await conn.commit()
            return bool(deleted)
        except Exception as exc:
            logger.warning(
                "MCP flow-test delete failed (project=%s name=%s): %s",
                project_id, name, exc,
            )
            return False


_store: LazyStore[FlowTestStore] = LazyStore(
    lambda: FlowTestStore(get_settings().database_url)
)


def get_flow_test_store() -> FlowTestStore:
    return _store.get()
