"""Per-project flow version history (#3) — Postgres-backed.

Each successful `flow_save` appends a new immutable version row for the
(project_id) so a user can list past versions (`flow_history`) and roll one
back into the workspace (`flow_rollback`). The table is bootstrapped on first
use and database errors are returned to the tool caller; persistence failures
must never masquerade as an empty history or a successful save.

Versioning is per-project and monotonic: version = max(existing)+1. We keep the
last 20 versions per project (older rows pruned on each append).
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

_MAX_VERSIONS = 20

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS mcp_flow_versions (
    id          BIGSERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL,
    user_id     TEXT,
    version     INTEGER NOT NULL,
    flow        JSONB NOT NULL,
    node_count  INTEGER NOT NULL DEFAULT 0,
    edge_count  INTEGER NOT NULL DEFAULT 0,
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, version)
);
"""


from ..config import get_settings
from ..config import normalize_pg_url as _normalize
from ._lazy_store import LazyStore


class FlowVersionStore:
    """Postgres-backed per-project flow version history."""

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

    async def append(
        self,
        *,
        project_id: str,
        user_id: str | None,
        flow: dict[str, Any],
        node_count: int,
        edge_count: int,
    ) -> int:
        """Insert a new version (version = max+1) and prune to the last
        _MAX_VERSIONS. Returns the new version number.
        """
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM mcp_flow_versions WHERE project_id = %s",
                    (project_id,),
                )
                row = await cur.fetchone()
                next_version = int(row[0] if row else 0) + 1
                await cur.execute(
                    """
                    INSERT INTO mcp_flow_versions
                        (project_id, user_id, version, flow, node_count,
                         edge_count, saved_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, NOW())
                    """,
                    (
                        project_id,
                        user_id,
                        next_version,
                        json.dumps(flow),
                        node_count,
                        edge_count,
                    ),
                )
                await cur.execute(
                    """
                    DELETE FROM mcp_flow_versions
                    WHERE project_id = %s AND version <= %s
                    """,
                    (project_id, next_version - _MAX_VERSIONS),
                )
            await conn.commit()
        return next_version

    async def list(self, *, project_id: str) -> list[dict[str, Any]]:
        """Return versions newest-first: {version, saved_at, node_count, edge_count}."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT version, saved_at, node_count, edge_count
                       FROM mcp_flow_versions
                       WHERE project_id = %s
                       ORDER BY version DESC""",
                    (project_id,),
                )
                rows = await cur.fetchall()
        return [
            {
                "version": int(r[0]),
                "saved_at": r[1].isoformat() if r[1] is not None else None,
                "node_count": int(r[2]),
                "edge_count": int(r[3]),
            }
            for r in rows
        ]

    async def get(self, *, project_id: str, version: int) -> dict[str, Any] | None:
        """Return one version's full flow dict, or None if it does not exist."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT flow, node_count, edge_count, saved_at
                       FROM mcp_flow_versions
                       WHERE project_id = %s AND version = %s""",
                    (project_id, version),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return {
            "flow": row[0],
            "node_count": int(row[1]),
            "edge_count": int(row[2]),
            "saved_at": row[3].isoformat() if row[3] is not None else None,
        }


_store: LazyStore[FlowVersionStore] = LazyStore(lambda: FlowVersionStore(get_settings().database_url))


def get_flow_version_store() -> FlowVersionStore:
    return _store.get()


def set_flow_version_store(store: FlowVersionStore | None) -> None:
    """Test hook — inject a fake/None store."""
    _store.reset(store)
