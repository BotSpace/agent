"""Per-run persistence + SSE pubsub via Redis (with in-process fallback).

Schema (Postgres):

    CREATE TABLE IF NOT EXISTS agent_runs (
        request_id      TEXT PRIMARY KEY,
        project_id      TEXT,
        user_id         TEXT,
        thread_id       TEXT NOT NULL,
        status          TEXT NOT NULL,
        prompt          TEXT NOT NULL,
        final_response  TEXT,
        error           TEXT,
        pending_approval JSONB,
        pending_clarification JSONB,
        current_plan    JSONB,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS agent_events (
        id           BIGSERIAL PRIMARY KEY,
        request_id   TEXT NOT NULL REFERENCES agent_runs(request_id) ON DELETE CASCADE,
        seq          INT NOT NULL,
        type         TEXT NOT NULL,
        payload      JSONB NOT NULL DEFAULT '{}',
        at           TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS agent_events_req_idx ON agent_events (request_id, seq);
    CREATE TABLE IF NOT EXISTS agent_messages (
        id          BIGSERIAL PRIMARY KEY,
        project_id  TEXT,
        user_id     TEXT,
        request_id  TEXT,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        payload     JSONB,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from uuid import uuid4

import psycopg

from ..config import get_settings


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS agent_runs (
    request_id      TEXT PRIMARY KEY,
    project_id      TEXT,
    user_id         TEXT,
    thread_id       TEXT NOT NULL,
    status          TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    final_response  TEXT,
    error           TEXT,
    pending_approval JSONB,
    pending_clarification JSONB,
    current_plan    JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS agent_events (
    id           BIGSERIAL PRIMARY KEY,
    request_id   TEXT NOT NULL REFERENCES agent_runs(request_id) ON DELETE CASCADE,
    seq          INT NOT NULL,
    type         TEXT NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{}',
    at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS agent_events_req_idx ON agent_events (request_id, seq);
CREATE TABLE IF NOT EXISTS agent_messages (
    id          BIGSERIAL PRIMARY KEY,
    project_id  TEXT,
    user_id     TEXT,
    request_id  TEXT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    payload     JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""


def _normalize(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


@dataclass
class RunRecord:
    request_id: str
    project_id: str | None
    user_id: str | None
    thread_id: str
    prompt: str
    status: str = "queued"
    final_response: str | None = None
    error: str | None = None
    pending_approval: dict[str, Any] | None = None
    pending_clarification: dict[str, Any] | None = None
    current_plan: list[dict[str, Any]] | None = None
    last_seq: int = 0
    created_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())


class RunStore:
    """Postgres-backed run store with in-memory SSE pubsub."""

    def __init__(self, dsn: str) -> None:
        self._dsn = _normalize(dsn)
        self._bootstrapped = False
        # request_id → asyncio.Queue[AgentEvent | None]
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = {}
        # request_id → running asyncio.Task — used by /cancel endpoint.
        self._tasks: dict[str, asyncio.Task] = {}

    async def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA)
            await conn.commit()
        self._bootstrapped = True

    # ── CRUD ───────────────────────────────────────────────────────────
    async def create(
        self,
        *,
        prompt: str,
        project_id: str | None,
        user_id: str | None,
        thread_id: str | None,
    ) -> RunRecord:
        await self.bootstrap()
        rec = RunRecord(
            request_id=uuid4().hex,
            project_id=project_id,
            user_id=user_id,
            thread_id=thread_id or uuid4().hex,
            prompt=prompt,
            status="queued",
        )
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_runs (request_id, project_id, user_id, thread_id, status, prompt)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (rec.request_id, rec.project_id, rec.user_id, rec.thread_id, rec.status, rec.prompt),
                )
                # message log
                if prompt:
                    await cur.execute(
                        """
                        INSERT INTO agent_messages (project_id, user_id, request_id, role, content)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (project_id, user_id, rec.request_id, "user", prompt),
                    )
            await conn.commit()
        return rec

    async def get(
        self,
        request_id: str,
        *,
        project_id: str | None = None,
        user_id: str | None = None,
    ) -> RunRecord | None:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT request_id, project_id, user_id, thread_id, status, prompt, "
                    "final_response, error, pending_approval, pending_clarification, current_plan, "
                    "created_at, updated_at "
                    "FROM agent_runs WHERE request_id = %s",
                    (request_id,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        rec = RunRecord(
            request_id=row[0],
            project_id=row[1],
            user_id=row[2],
            thread_id=row[3],
            status=row[4],
            prompt=row[5],
            final_response=row[6],
            error=row[7],
            pending_approval=row[8],
            pending_clarification=row[9],
            current_plan=row[10],
            created_at=str(row[11]),
            updated_at=str(row[12]),
        )
        if project_id is not None and rec.project_id != project_id:
            return None
        if user_id is not None and rec.user_id != user_id:
            return None
        return rec

    async def set_status(self, request_id: str, status: str) -> None:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE agent_runs SET status = %s, updated_at = NOW() WHERE request_id = %s",
                    (status, request_id),
                )
            await conn.commit()

    async def complete(self, request_id: str, final_response: str) -> None:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE agent_runs
                       SET status = 'completed', final_response = %s, updated_at = NOW()
                       WHERE request_id = %s""",
                    (final_response, request_id),
                )
                rec = await cur.execute(
                    "SELECT project_id, user_id FROM agent_runs WHERE request_id = %s",
                    (request_id,),
                )
                pid, uid = (await rec.fetchone()) if rec is not None else (None, None)
                if final_response:
                    await cur.execute(
                        """INSERT INTO agent_messages
                           (project_id, user_id, request_id, role, content)
                           VALUES (%s, %s, %s, 'assistant', %s)""",
                        (pid, uid, request_id, final_response),
                    )
            await conn.commit()
        await self._publish(request_id, None)  # close subscribers

    async def fail(self, request_id: str, error: str) -> None:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE agent_runs SET status = 'failed', error = %s, updated_at = NOW() WHERE request_id = %s",
                    (error, request_id),
                )
            await conn.commit()
        await self._publish(request_id, None)

    async def set_pending_approval(self, request_id: str, payload: dict[str, Any] | None) -> None:
        await self.bootstrap()
        status = "waiting_for_approval" if payload else "running"
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE agent_runs SET pending_approval = %s::jsonb, status = %s, updated_at = NOW() WHERE request_id = %s",
                    (json.dumps(payload) if payload else None, status, request_id),
                )
            await conn.commit()

    async def set_pending_clarification(self, request_id: str, payload: dict[str, Any] | None) -> None:
        await self.bootstrap()
        status = "waiting_for_clarification" if payload else "running"
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE agent_runs SET pending_clarification = %s::jsonb, status = %s, updated_at = NOW() WHERE request_id = %s",
                    (json.dumps(payload) if payload else None, status, request_id),
                )
            await conn.commit()

    # ── events / SSE ───────────────────────────────────────────────────
    async def emit(self, request_id: str, event_type: str, payload: dict[str, Any]) -> int:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_events WHERE request_id = %s",
                    (request_id,),
                )
                row = await cur.fetchone()
                seq = int(row[0]) if row else 1
                await cur.execute(
                    """INSERT INTO agent_events (request_id, seq, type, payload)
                       VALUES (%s, %s, %s, %s::jsonb)""",
                    (request_id, seq, event_type, json.dumps(payload)),
                )
            await conn.commit()
        from .events import AgentEvent

        ev = AgentEvent(
            seq=seq,
            type=event_type,  # type: ignore[arg-type]
            payload=payload,
            at=dt.datetime.utcnow().isoformat(),
        )
        await self._publish(request_id, ev)
        return seq

    async def list_events(self, request_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT seq, type, payload, at FROM agent_events
                       WHERE request_id = %s AND seq > %s ORDER BY seq""",
                    (request_id, since_seq),
                )
                rows = await cur.fetchall()
                return [
                    {"seq": r[0], "type": r[1], "payload": r[2], "at": str(r[3])}
                    for r in rows
                ]

    def subscribe(self, request_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(request_id, []).append(q)
        return q

    def unsubscribe(self, request_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(request_id)
        if subs and q in subs:
            subs.remove(q)
        if subs is not None and not subs:
            self._subscribers.pop(request_id, None)

    async def _publish(self, request_id: str, ev: Any | None) -> None:
        subs = list(self._subscribers.get(request_id, []))
        for q in subs:
            await q.put(ev)

    # ── cancellation ───────────────────────────────────────────────────
    def register_task(self, request_id: str, task: asyncio.Task) -> None:
        self._tasks[request_id] = task

    def unregister_task(self, request_id: str) -> None:
        self._tasks.pop(request_id, None)

    async def cancel(self, request_id: str) -> bool:
        task = self._tasks.get(request_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    # ── messages (history) ─────────────────────────────────────────────
    async def list_messages(
        self, *, project_id: str | None, user_id: str | None, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        await self.bootstrap()
        where = []
        params: list[Any] = []
        if project_id:
            where.append("project_id = %s")
            params.append(project_id)
        if user_id:
            where.append("user_id = %s")
            params.append(user_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, offset])
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""SELECT id, request_id, role, content, created_at
                        FROM agent_messages {clause}
                        ORDER BY id DESC LIMIT %s OFFSET %s""",
                    params,
                )
                rows = await cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "request_id": r[1],
                        "role": r[2],
                        "content": r[3],
                        "created_at": str(r[4]),
                    }
                    for r in rows
                ]


@lru_cache(maxsize=1)
def get_run_store() -> RunStore:
    s = get_settings()
    return RunStore(s.database_url)
