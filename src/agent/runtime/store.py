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
    input_tokens    BIGINT DEFAULT 0,
    output_tokens   BIGINT DEFAULT 0,
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

    async def _ensure_token_columns(self, conn: psycopg.AsyncConnection) -> None:
        async with conn.cursor() as cur:
            await cur.execute(
                "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS input_tokens BIGINT DEFAULT 0;"
            )
            await cur.execute(
                "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS output_tokens BIGINT DEFAULT 0;"
            )
        await conn.commit()

    async def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA)
            await conn.commit()
            await self._ensure_token_columns(conn)
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

    # ── token tracking ──────────────────────────────────────────────────
    async def record_tokens(self, request_id: str, input_tokens: int, output_tokens: int) -> None:
        """Update token usage for a completed run."""
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE agent_runs SET input_tokens = %s, output_tokens = %s WHERE request_id = %s",
                    (input_tokens, output_tokens, request_id),
                )
            await conn.commit()

    # ── threads (dashboard history) ────────────────────────────────────
    async def list_threads(
        self, *, project_id: str | None, user_id: str | None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Group ``agent_runs`` by ``thread_id`` for the dashboard history list.

        Each thread carries the earliest run's prompt as ``title`` (truncated
        to ~60 chars), the latest ``updated_at`` and a ``run_count``. Ordered
        by ``updated_at`` DESC. Filtered by project_id / user_id so a user only
        ever sees their own threads. On any error returns an empty list.
        """
        try:
            await self.bootstrap()
            where = ["thread_id IS NOT NULL"]
            params: list[Any] = []
            if project_id:
                where.append("project_id = %s")
                params.append(project_id)
            if user_id:
                where.append("user_id = %s")
                params.append(user_id)
            clause = "WHERE " + " AND ".join(where)
            params.append(limit)
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        SELECT
                            thread_id,
                            (ARRAY_AGG(prompt ORDER BY created_at ASC))[1] AS title,
                            MAX(updated_at) AS updated_at,
                            COUNT(*) AS run_count
                        FROM agent_runs
                        {clause}
                        GROUP BY thread_id
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = await cur.fetchall()
            threads: list[dict[str, Any]] = []
            for r in rows:
                title = (r[1] or "").strip()
                if len(title) > 60:
                    title = title[:60].rstrip() + "…"
                threads.append(
                    {
                        "thread_id": r[0],
                        "title": title,
                        "updated_at": str(r[2]) if r[2] is not None else "",
                        "run_count": int(r[3]),
                    }
                )
            return threads
        except Exception:  # noqa: BLE001
            return []

    async def delete_thread(
        self, *, project_id: str | None, user_id: str | None, thread_id: str
    ) -> int:
        """Delete a thread's runs (and, via FK cascade, their events) plus its
        messages. Scoped by project_id/user_id so a user can only delete their
        own threads. Returns the number of deleted runs. Never raises.
        """
        try:
            await self.bootstrap()
            where = ["thread_id = %s"]
            params: list[Any] = [thread_id]
            if project_id:
                where.append("project_id = %s")
                params.append(project_id)
            if user_id:
                where.append("user_id = %s")
                params.append(user_id)
            clause = " AND ".join(where)
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    # agent_messages has no FK cascade — remove its rows first.
                    await cur.execute(
                        f"DELETE FROM agent_messages WHERE request_id IN "
                        f"(SELECT request_id FROM agent_runs WHERE {clause})",
                        params,
                    )
                    # Deleting runs cascades to agent_events (ON DELETE CASCADE).
                    await cur.execute(f"DELETE FROM agent_runs WHERE {clause}", params)
                    deleted = cur.rowcount
                await conn.commit()
            return int(deleted or 0)
        except Exception:  # noqa: BLE001
            return 0

    async def delete_all_threads(
        self, *, project_id: str | None, user_id: str | None
    ) -> int:
        """Delete ALL of a user's threads for a project (their runs + cascaded
        events + messages). Scoped by project_id/user_id. Returns deleted run
        count. Never raises. A project_id is required — refuse a global wipe."""
        if not project_id:
            return 0
        try:
            await self.bootstrap()
            where = ["project_id = %s"]
            params: list[Any] = [project_id]
            if user_id:
                where.append("user_id = %s")
                params.append(user_id)
            clause = " AND ".join(where)
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"DELETE FROM agent_messages WHERE request_id IN "
                        f"(SELECT request_id FROM agent_runs WHERE {clause})",
                        params,
                    )
                    await cur.execute(f"DELETE FROM agent_runs WHERE {clause}", params)
                    deleted = cur.rowcount
                await conn.commit()
            return int(deleted or 0)
        except Exception:  # noqa: BLE001
            return 0

    async def get_thread_turns(
        self, *, project_id: str | None, user_id: str | None, thread_id: str
    ) -> list[dict[str, Any]]:
        """Reconstruct an ordered transcript ("turns") for one thread.

        Faithful reconstruction note: ``agent_messages`` only persists the
        ``user`` prompt and the FINAL ``assistant`` text — intermediate
        assistant steps and tool calls live in ``agent_events`` (types
        ``assistant_message`` / ``tool_call`` / ``tool_result``). So per run we
        emit the user turn from ``agent_messages`` (stable ``m{id}``), then
        walk that run's events in ``seq`` order, materialising assistant and
        tool turns. Access is scoped to the caller's project_id / user_id; a
        thread that doesn't belong to the caller yields an empty list.
        On any error returns an empty list (never raises).
        """
        try:
            await self.bootstrap()
            run_where = ["thread_id = %s"]
            run_params: list[Any] = [thread_id]
            if project_id:
                run_where.append("project_id = %s")
                run_params.append(project_id)
            if user_id:
                run_where.append("user_id = %s")
                run_params.append(user_id)
            run_clause = "WHERE " + " AND ".join(run_where)

            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    # Runs for this thread, oldest first. Scoped to caller.
                    await cur.execute(
                        f"""SELECT request_id FROM agent_runs
                            {run_clause} ORDER BY created_at ASC, request_id ASC""",
                        run_params,
                    )
                    run_rows = await cur.fetchall()
                    request_ids = [r[0] for r in run_rows]
                    if not request_ids:
                        return []

                    # user-prompt rows (stable ids) per run.
                    await cur.execute(
                        """SELECT id, request_id, content FROM agent_messages
                           WHERE request_id = ANY(%s) AND role = 'user'
                           ORDER BY id ASC""",
                        (request_ids,),
                    )
                    user_rows = await cur.fetchall()
                    user_by_req: dict[str, tuple[int, str]] = {}
                    for mid, req, content in user_rows:
                        # first user message per run is the prompt
                        user_by_req.setdefault(req, (int(mid), content or ""))

                    # all relevant events for these runs, ordered per run by seq.
                    await cur.execute(
                        """SELECT request_id, seq, type, payload FROM agent_events
                           WHERE request_id = ANY(%s)
                             AND type IN ('assistant_message', 'tool_call', 'tool_result')
                           ORDER BY request_id, seq ASC""",
                        (request_ids,),
                    )
                    event_rows = await cur.fetchall()

            events_by_req: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
            for req, seq, etype, payload in event_rows:
                events_by_req.setdefault(req, []).append(
                    (int(seq), etype, payload if isinstance(payload, dict) else {})
                )

            turns: list[dict[str, Any]] = []
            for req in request_ids:
                # 1) user turn for this run
                um = user_by_req.get(req)
                if um is not None:
                    turns.append(
                        {"kind": "user", "id": f"m{um[0]}", "text": um[1]}
                    )

                # 2) walk events in seq order: assistant text + tool call/result
                tool_results: dict[str, dict[str, Any]] = {}
                for seq, etype, payload in events_by_req.get(req, []):
                    if etype == "tool_result":
                        tid = payload.get("tool_call_id")
                        if tid is not None:
                            tool_results[str(tid)] = payload

                for seq, etype, payload in events_by_req.get(req, []):
                    if etype == "assistant_message":
                        text = (payload.get("text") or "").strip()
                        if text:
                            turns.append(
                                {
                                    "kind": "assistant",
                                    "id": f"{req}-a{seq}",
                                    "text": text,
                                }
                            )
                    elif etype == "tool_call":
                        call_id = payload.get("id")
                        res = tool_results.get(str(call_id)) if call_id is not None else None
                        result_content = ""
                        is_error = False
                        if res is not None:
                            result_content = res.get("content") or ""
                            is_error = bool(res.get("is_error"))
                        turns.append(
                            {
                                "kind": "tool",
                                "id": f"{req}-t{seq}",
                                "call": {
                                    "id": call_id,
                                    "name": payload.get("name"),
                                    "arguments": payload.get("arguments") or {},
                                    "result": result_content,
                                    "isError": is_error,
                                },
                            }
                        )
            return turns
        except Exception:  # noqa: BLE001
            return []

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
