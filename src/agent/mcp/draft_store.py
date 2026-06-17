"""Write-through persistence for MCP flow drafts (#1, #11).

MCP transport is stateless; the working flow lives in an in-memory Workspace
keyed by (user_id, project_id). On pod restart or load-balancer reschedule that
state is lost — and with it any UNSAVED draft. This module persists each draft
to the agent's OWN Postgres (AGENT_DATABASE_URL), bootstrapping its own table
(no ownership problems — the agent created it). It also keeps a small snapshot
ring (last 5) per draft for session_undo.

Design notes:
  - Same bootstrap pattern as runtime/store.py: CREATE TABLE IF NOT EXISTS on
    first use, one short-lived connection per call (psycopg async).
  - GRACEFUL DEGRADE: if the DB is unreachable, every method swallows the error
    and logs a warning. The MCP keeps working purely in-memory.
  - Keyed by (user_id, project_id) — survives restarts and lets a user keep
    separate drafts per project.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

_MAX_SNAPSHOTS = 5

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS mcp_flow_drafts (
    user_id       TEXT NOT NULL,
    project_id    TEXT NOT NULL,
    project_name  TEXT,
    working_flow  JSONB,
    baseline_flow JSONB,
    flow_dirty    BOOLEAN NOT NULL DEFAULT FALSE,
    collections   JSONB,
    snapshots     JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, project_id)
);
"""


from ..config import normalize_pg_url as _normalize


class DraftStore:
    """Postgres-backed (user_id, project_id) → draft row, with graceful degrade."""

    def __init__(self, dsn: str) -> None:
        self._dsn = _normalize(dsn)
        self._bootstrapped = False
        self._degraded = False  # set after the first connect failure

    async def _bootstrap(self) -> bool:
        """Returns True if the store is usable, False if degraded."""
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
                "MCP draft persistence unavailable, degrading to in-memory only: %s",
                exc,
            )
            return False

    async def upsert(
        self,
        *,
        user_id: str,
        project_id: str,
        project_name: str | None,
        working_flow: dict[str, Any] | None,
        baseline_flow: dict[str, Any] | None,
        flow_dirty: bool,
        collections: list[dict[str, Any]] | None,
        snapshots: list[dict[str, Any]] | None,
    ) -> None:
        if not await self._bootstrap():
            return
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO mcp_flow_drafts
                            (user_id, project_id, project_name, working_flow,
                             baseline_flow, flow_dirty, collections, snapshots,
                             updated_at)
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb,
                                %s::jsonb, NOW())
                        ON CONFLICT (user_id, project_id) DO UPDATE SET
                            project_name  = EXCLUDED.project_name,
                            working_flow  = EXCLUDED.working_flow,
                            baseline_flow = EXCLUDED.baseline_flow,
                            flow_dirty    = EXCLUDED.flow_dirty,
                            collections   = EXCLUDED.collections,
                            snapshots     = EXCLUDED.snapshots,
                            updated_at    = NOW()
                        """,
                        (
                            user_id,
                            project_id,
                            project_name,
                            json.dumps(working_flow) if working_flow is not None else None,
                            json.dumps(baseline_flow) if baseline_flow is not None else None,
                            flow_dirty,
                            json.dumps(collections or []),
                            json.dumps((snapshots or [])[-_MAX_SNAPSHOTS:]),
                        ),
                    )
                await conn.commit()
        except Exception as exc:
            logger.warning("MCP draft upsert failed (%s/%s): %s", user_id, project_id, exc)

    async def load(self, *, user_id: str, project_id: str) -> dict[str, Any] | None:
        if not await self._bootstrap():
            return None
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """SELECT project_name, working_flow, baseline_flow,
                                  flow_dirty, collections, snapshots
                           FROM mcp_flow_drafts
                           WHERE user_id = %s AND project_id = %s""",
                        (user_id, project_id),
                    )
                    row = await cur.fetchone()
        except Exception as exc:
            logger.warning("MCP draft load failed (%s/%s): %s", user_id, project_id, exc)
            return None
        if not row:
            return None
        return {
            "project_name": row[0],
            "working_flow": row[1],
            "baseline_flow": row[2],
            "flow_dirty": bool(row[3]),
            "collections": row[4] or [],
            "snapshots": row[5] or [],
        }

    async def load_latest(self, *, user_id: str) -> dict[str, Any] | None:
        """Most-recently-updated draft for a user across all projects (#5).

        Used for transparent session recovery: after a restart the active-project
        pointer is lost, so we fall back to "whatever the user was last editing".
        Includes `project_id` in the returned dict. Graceful degrade → None."""
        if not await self._bootstrap():
            return None
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """SELECT project_id, project_name, working_flow,
                                  baseline_flow, flow_dirty, collections, snapshots
                           FROM mcp_flow_drafts
                           WHERE user_id = %s
                           ORDER BY updated_at DESC
                           LIMIT 1""",
                        (user_id,),
                    )
                    row = await cur.fetchone()
        except Exception as exc:
            logger.warning("MCP draft load_latest failed (%s): %s", user_id, exc)
            return None
        if not row:
            return None
        return {
            "project_id": row[0],
            "project_name": row[1],
            "working_flow": row[2],
            "baseline_flow": row[3],
            "flow_dirty": bool(row[4]),
            "collections": row[5] or [],
            "snapshots": row[6] or [],
        }

    async def clear(self, *, user_id: str, project_id: str) -> None:
        """Drop the persisted draft after a successful save."""
        if not await self._bootstrap():
            return
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM mcp_flow_drafts WHERE user_id = %s AND project_id = %s",
                        (user_id, project_id),
                    )
                await conn.commit()
        except Exception as exc:
            logger.warning("MCP draft clear failed (%s/%s): %s", user_id, project_id, exc)


_store: DraftStore | None = None


def get_draft_store() -> DraftStore:
    global _store
    if _store is None:
        from ..config import get_settings

        _store = DraftStore(get_settings().database_url)
    return _store


def set_draft_store(store: DraftStore | None) -> None:
    """Test hook — inject a fake/None store."""
    global _store
    _store = store
