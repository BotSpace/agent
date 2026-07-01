"""Per-project flow version history (#3) — Postgres-backed, graceful degrade.

Each successful `flow_save` appends a new immutable version row for the
(project_id) so a user can list past versions (`flow_history`) and roll one
back into the workspace (`flow_rollback`). Same bootstrap/degrade philosophy as
draft_store.py: CREATE TABLE IF NOT EXISTS on first use, one short-lived
connection per call (psycopg async), and every method swallows DB errors so a
DB outage can never break flow_save.

Versioning is per-project and monotonic: version = max(existing)+1. We keep the
last 20 versions per project (older rows pruned on each append).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

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
    """Postgres-backed per-project flow version history with graceful degrade."""

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
                "MCP flow-version history unavailable, degrading to no-op: %s", exc
            )
            return False

    async def append(
        self,
        *,
        project_id: str,
        user_id: str | None,
        flow: dict[str, Any],
        node_count: int,
        edge_count: int,
    ) -> int | None:
        """Insert a new version (version = max+1) and prune to the last
        _MAX_VERSIONS. Returns the new version number, or None on degrade/error.

        Caller MUST NOT let a failure here break flow_save — errors are
        swallowed and logged.
        """
        if not await self._bootstrap():
            return None
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COALESCE(MAX(version), 0) FROM mcp_flow_versions "
                        "WHERE project_id = %s",
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
                    # Prune older versions beyond the retention window.
                    await cur.execute(
                        """
                        DELETE FROM mcp_flow_versions
                        WHERE project_id = %s AND version <= %s
                        """,
                        (project_id, next_version - _MAX_VERSIONS),
                    )
                await conn.commit()
            return next_version
        except Exception as exc:
            logger.warning(
                "MCP flow-version append failed (project=%s): %s", project_id, exc
            )
            return None

    async def list(self, *, project_id: str) -> list[dict[str, Any]]:
        """Return versions newest-first: {version, saved_at, node_count, edge_count}."""
        if not await self._bootstrap():
            return []
        try:
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
        except Exception as exc:
            logger.warning(
                "MCP flow-version list failed (project=%s): %s", project_id, exc
            )
            return []
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
        """Return one version's full flow dict, or None if missing/degraded."""
        if not await self._bootstrap():
            return None
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """SELECT flow, node_count, edge_count, saved_at
                           FROM mcp_flow_versions
                           WHERE project_id = %s AND version = %s""",
                        (project_id, version),
                    )
                    row = await cur.fetchone()
        except Exception as exc:
            logger.warning(
                "MCP flow-version get failed (project=%s v%s): %s",
                project_id,
                version,
                exc,
            )
            return None
        if not row:
            return None
        return {
            "flow": row[0],
            "node_count": int(row[1]),
            "edge_count": int(row[2]),
            "saved_at": row[3].isoformat() if row[3] is not None else None,
        }


_store: LazyStore[FlowVersionStore] = LazyStore(
    lambda: FlowVersionStore(get_settings().database_url)
)


def get_flow_version_store() -> FlowVersionStore:
    return _store.get()


def set_flow_version_store(store: FlowVersionStore | None) -> None:
    """Test hook — inject a fake/None store."""
    _store.reset(store)
