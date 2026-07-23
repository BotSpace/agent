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
  - Database errors propagate. The caller must never mistake an in-memory
    change for a durable draft.
  - Keyed by (user_id, project_id) — survives restarts and lets a user keep
    separate drafts per project.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

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
    active_at     TIMESTAMPTZ,
    PRIMARY KEY (user_id, project_id)
);
-- active_at — foydalanuvchi shu loyihani ohirgi marta EXPLICIT tanlagan vaqti
-- (session_use_project). Restore shuni asos qiladi, oxirgi tahrirlangan draftni
-- emas — noto'g'ri loyihaga drift'ni oldini oladi. Eski jadval uchun migratsiya:
ALTER TABLE mcp_flow_drafts ADD COLUMN IF NOT EXISTS active_at TIMESTAMPTZ;
"""


from ..config import get_settings
from ..config import normalize_pg_url as _normalize
from ._lazy_store import LazyStore


class DraftStore:
    """Postgres-backed (user_id, project_id) → durable draft row."""

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
        await self._bootstrap()
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

    async def load(self, *, user_id: str, project_id: str) -> dict[str, Any] | None:
        await self._bootstrap()
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

    async def touch_active(self, *, user_id: str, project_id: str) -> None:
        """Mark this (user, project) as the last EXPLICITLY selected one (#2).

        Called on session_use_project. Restore (load_latest) prefers the most
        recently *selected* project over the most recently *edited* one, so a
        mutation on another project can't silently drift the session. No-op if
        the draft row doesn't exist yet (upsert creates it first)."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE mcp_flow_drafts SET active_at = NOW() WHERE user_id = %s AND project_id = %s",
                    (user_id, project_id),
                )
            await conn.commit()

    async def load_latest(self, *, user_id: str) -> dict[str, Any] | None:
        """Session recovery anchor for a user across all projects (#2, #5).

        After a restart the active-project pointer is lost. We restore the
        project the user last EXPLICITLY selected (active_at), falling back to
        the most recently edited draft (updated_at) if none was ever selected.
        Includes `project_id` in the returned dict."""
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT project_id, project_name, working_flow,
                                  baseline_flow, flow_dirty, collections, snapshots
                           FROM mcp_flow_drafts
                           WHERE user_id = %s
                           ORDER BY active_at DESC NULLS LAST, updated_at DESC
                           LIMIT 1""",
                    (user_id,),
                )
                row = await cur.fetchone()
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
        await self._bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM mcp_flow_drafts WHERE user_id = %s AND project_id = %s",
                    (user_id, project_id),
                )
            await conn.commit()


_store: LazyStore[DraftStore] = LazyStore(lambda: DraftStore(get_settings().database_url))


def get_draft_store() -> DraftStore:
    return _store.get()


def set_draft_store(store: DraftStore | None) -> None:
    """Test hook — inject a fake/None store."""
    _store.reset(store)
