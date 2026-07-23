"""Per-run persistence and live SSE delivery.

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
from ..mongo_log import (
    log_assistant_message,
    log_run_complete,
    log_run_fail,
    log_run_start,
    log_subagent_started,
    log_tool_call,
    log_tool_result,
)

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
CREATE TABLE IF NOT EXISTS agent_project_settings (
    project_id      TEXT PRIMARY KEY,
    thinking_effort TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Allowed DeepSeek thinking-effort values ("off" disables reasoning). Anything
# else is rejected so a bad UI value can't reach the LLM request.
THINKING_EFFORTS = ("off", "high", "max")


from ..config import normalize_pg_url as _normalize


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
        log_run_start(rec.request_id, rec.project_id, rec.user_id, rec.thread_id, prompt)
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

    async def get_project_thinking_effort(self, project_id: str | None) -> str | None:
        """Per-project DeepSeek thinking effort ('off'|'high'|'max'), or None when
        unset (→ provider default)."""
        if not project_id:
            return None
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT thinking_effort FROM agent_project_settings WHERE project_id = %s",
                    (str(project_id),),
                )
                row = await cur.fetchone()
        effort = row[0] if row else None
        return effort if effort in THINKING_EFFORTS else None

    async def set_project_thinking_effort(self, project_id: str, effort: str) -> None:
        """Upsert the per-project thinking effort. Caller validates `effort`."""
        await self.bootstrap()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO agent_project_settings (project_id, thinking_effort, updated_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (project_id)
                       DO UPDATE SET thinking_effort = EXCLUDED.thinking_effort, updated_at = NOW()""",
                    (str(project_id), effort),
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
        log_run_complete(request_id, final_response)
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
        log_run_fail(request_id, error)
        await self._publish(request_id, None)

    async def recover_orphaned_runs(self) -> int:
        """On startup, fail runs left 'running'/'queued' by a killed pod (a redeploy
        aborts their asyncio task but not the DB row → the UI shows them stuck).
        Paused waiting_for_* runs are LEFT ALONE — the user can still resume them."""
        await self.bootstrap()
        msg = "Agent qayta ishga tushdi — run to'xtadi. Iltimos, qaytadan urinib ko'ring."
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT request_id FROM agent_runs WHERE status IN ('running', 'queued')")
                ids = [r[0] for r in await cur.fetchall()]
                if ids:
                    await cur.execute(
                        "UPDATE agent_runs SET status = 'failed', error = %s, updated_at = NOW() "
                        "WHERE status IN ('running', 'queued')",
                        (msg,),
                    )
            await conn.commit()
        # Persist a terminal event per orphan so a reconnecting SSE replay unblocks
        # the UI (it stops waiting and renders the failure).
        for rid in ids:
            try:
                await self.emit(rid, "run_failed", {"error": msg})
            except Exception:
                pass
        return len(ids)

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
        # Mirror select events to MongoDB for LLM training data
        subagent = payload.get("sub_agent_name") if isinstance(payload, dict) else None
        if event_type == "assistant_message":
            log_assistant_message(
                request_id,
                payload.get("text", ""),
                model=payload.get("model"),
                usage=payload.get("usage"),
                subagent=subagent,
            )
        elif event_type == "tool_call":
            log_tool_call(
                request_id,
                call_id=payload.get("id", ""),
                name=payload.get("name", ""),
                arguments=payload.get("arguments") or {},
                subagent=subagent,
            )
        elif event_type == "tool_result":
            log_tool_result(
                request_id,
                call_id=payload.get("tool_call_id", ""),
                name=payload.get("name", ""),
                content=str(payload.get("content", "")),
                is_error=bool(payload.get("is_error", False)),
                subagent=subagent,
            )
        elif event_type == "sub_agent_started":
            name = payload.get("name") if isinstance(payload, dict) else None
            if name:
                log_subagent_started(request_id, name)
        return seq

    async def emit_ephemeral(self, request_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a LIVE-ONLY event to SSE subscribers WITHOUT persisting it.

        Used for high-frequency streaming deltas (``assistant_chunk`` /
        ``thinking_delta``) — writing every token to ``agent_events`` (a fresh
        DB connection + INSERT each) would be ruinous, and the durable
        ``assistant_message`` emitted at turn end already records the final text.
        ``seq=0`` is a sentinel: the frontend only advances its replay cursor on
        ``seq > last`` so ephemeral events never disturb reconnect/replay.
        """
        from .events import AgentEvent

        ev = AgentEvent(
            seq=0,
            type=event_type,  # type: ignore[arg-type]
            payload=payload,
            at=dt.datetime.utcnow().isoformat(),
        )
        await self._publish(request_id, ev)

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
                return [{"seq": r[0], "type": r[1], "payload": r[2], "at": str(r[3])} for r in rows]

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
        except Exception:
            return []

    async def list_threads_for_user(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Admin view: all of ONE user's threads across every project.

        Like ``list_threads`` but scoped by ``user_id`` alone (not a project),
        and each row also carries its ``project_id`` and the thread's total
        token usage so the admin panel can show which bot a conversation
        belonged to and how much it cost. Ordered by ``updated_at`` DESC. On any
        error returns an empty list (never raises)."""
        try:
            await self.bootstrap()
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT
                            thread_id,
                            (ARRAY_AGG(project_id ORDER BY created_at ASC))[1] AS project_id,
                            (ARRAY_AGG(prompt ORDER BY created_at ASC))[1] AS title,
                            MAX(updated_at) AS updated_at,
                            COUNT(*) AS run_count,
                            COALESCE(SUM(input_tokens), 0) AS input_tokens,
                            COALESCE(SUM(output_tokens), 0) AS output_tokens
                        FROM agent_runs
                        WHERE thread_id IS NOT NULL AND user_id = %s
                        GROUP BY thread_id
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit),
                    )
                    rows = await cur.fetchall()
            threads: list[dict[str, Any]] = []
            for r in rows:
                title = (r[2] or "").strip()
                if len(title) > 80:
                    title = title[:80].rstrip() + "…"
                threads.append(
                    {
                        "thread_id": r[0],
                        "project_id": r[1] or "",
                        "title": title,
                        "updated_at": str(r[3]) if r[3] is not None else "",
                        "run_count": int(r[4]),
                        "tokens_used_total": int(r[5] or 0) + int(r[6] or 0),
                    }
                )
            return threads
        except Exception:
            return []

    async def delete_thread(self, *, project_id: str | None, user_id: str | None, thread_id: str) -> int:
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
        except Exception:
            return 0

    async def delete_all_threads(self, *, project_id: str | None, user_id: str | None) -> int:
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
        except Exception:
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
                             AND type IN ('assistant_message', 'tool_call',
                                          'tool_result', 'plan_set', 'plan_update')
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
                    turns.append({"kind": "user", "id": f"m{um[0]}", "text": um[1]})

                # 2) walk events in seq order: assistant text + tool call/result
                tool_results: dict[str, dict[str, Any]] = {}
                for _seq, etype, payload in events_by_req.get(req, []):
                    if etype == "tool_result":
                        tid = payload.get("tool_call_id")
                        if tid is not None:
                            tool_results[str(tid)] = payload

                # One inline plan turn per run, materialised at the first plan_set
                # position and mutated in place by later plan_set/plan_update so
                # history shows the plan where it was created, in its final state.
                plan_turn: dict[str, Any] | None = None
                for seq, etype, payload in events_by_req.get(req, []):
                    if etype == "plan_set":
                        plan = payload.get("plan")
                        if isinstance(plan, list):
                            if plan_turn is None:
                                plan_turn = {
                                    "kind": "plan",
                                    "id": f"{req}-plan{seq}",
                                    "plan": plan,
                                }
                                turns.append(plan_turn)
                            else:
                                plan_turn["plan"] = plan
                        continue
                    if etype == "plan_update":
                        if plan_turn is None:
                            continue
                        plan = payload.get("plan")
                        if isinstance(plan, list):
                            plan_turn["plan"] = plan
                        else:
                            item = payload.get("item")
                            if isinstance(item, dict) and item.get("id"):
                                plan_turn["plan"] = [
                                    item if p.get("id") == item.get("id") else p for p in plan_turn["plan"]
                                ]
                        continue
                    if etype == "assistant_message":
                        text = (payload.get("text") or "").strip()
                        if text:
                            turn: dict[str, Any] = {
                                "kind": "assistant",
                                "id": f"{req}-a{seq}",
                                "text": text,
                            }
                            reasoning = payload.get("reasoning")
                            if isinstance(reasoning, str) and reasoning.strip():
                                turn["reasoning"] = reasoning
                            turns.append(turn)
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
        except Exception:
            return []

    async def get_active_run(
        self, *, project_id: str | None, user_id: str | None, thread_id: str
    ) -> dict[str, Any] | None:
        """Return the thread's still-in-flight run so a refreshed page can
        re-attach to its live SSE instead of losing the stream.

        "Active" = the most recent run whose status is not terminal
        (``completed`` / ``failed``; a user cancel also lands on ``failed``).
        ``last_seq`` is the highest seq among the events already materialised
        into history turns (``assistant_message`` / ``tool_call`` /
        ``tool_result``); the frontend reconnects with ``since_seq=last_seq`` so
        only unseen events (a pending tool_result, a fresh approval prompt, the
        final answer) replay — no duplicate turns. Scoped to the caller. Returns
        ``None`` when the thread has no active run. Never raises.
        """
        try:
            await self.bootstrap()
            where = ["thread_id = %s", "status NOT IN ('completed', 'failed')"]
            params: list[Any] = [thread_id]
            if project_id:
                where.append("project_id = %s")
                params.append(project_id)
            if user_id:
                where.append("user_id = %s")
                params.append(user_id)
            clause = "WHERE " + " AND ".join(where)
            async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""SELECT request_id, status, pending_approval, pending_clarification
                            FROM agent_runs {clause}
                            ORDER BY created_at DESC, request_id DESC LIMIT 1""",
                        params,
                    )
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    request_id = row[0]
                    await cur.execute(
                        """SELECT COALESCE(MAX(seq), 0) FROM agent_events
                           WHERE request_id = %s
                             AND type IN ('assistant_message', 'tool_call', 'tool_result')""",
                        (request_id,),
                    )
                    seq_row = await cur.fetchone()
            last_seq = int(seq_row[0]) if seq_row and seq_row[0] is not None else 0
            return {
                "request_id": request_id,
                "status": row[1],
                "last_seq": last_seq,
                "pending_approval": row[2],
                "pending_clarification": row[3],
            }
        except Exception:
            return None

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


# ── interrupt/approval resolution rendezvous (cross-replica) ─────────────────
class ResolutionStore:
    """Cross-process rendezvous for interrupt (approval/clarify) resolutions.

    The per-run ``asyncio.Future`` used to resume an interrupted graph is
    process-local, so in a multi-replica deployment the dashboard's
    ``/approve`` (or ``/clarify``) POST may be served by a DIFFERENT replica
    than the one running the worker coroutine. This bridges them over a
    short-lived Redis list keyed by ``request_id``: the POST-handling replica
    ``RPUSH``es the answer, the worker's replica ``BLPOP``s it and resolves its
    local future. Redis is required for cross-replica delivery.
    """

    _KEY = "agent:resolution:{}"
    _TTL_SECONDS = 3600

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url

    def _key(self, request_id: str) -> str:
        return self._KEY.format(request_id)

    def ping(self) -> None:
        """Verify the required cross-replica Redis transport at startup."""
        import redis

        client = redis.Redis.from_url(self._url)
        try:
            client.ping()
        finally:
            client.close()

    def publish(self, request_id: str, payload: Any) -> bool:
        """Durably enqueue a resolution for a (possibly remote) waiter.

        Synchronous — called from the ``/approve|/clarify`` route when no LOCAL
        future is awaiting (the worker is on another replica). Returns True
        after the answer is enqueued; connection errors propagate.
        """
        import redis

        client = redis.Redis.from_url(self._url)
        try:
            pipe = client.pipeline()
            pipe.rpush(self._key(request_id), json.dumps(payload))
            pipe.expire(self._key(request_id), self._TTL_SECONDS)
            pipe.execute()
        finally:
            client.close()
        return True

    async def wait(self, request_id: str) -> Any:
        """Block until a resolution is pushed (by ``publish`` on ANY replica).

        Raises on any Redis error so the interrupted run reports the outage.
        """
        import redis.asyncio as aioredis  # lazy: optional dep

        client = aioredis.Redis.from_url(self._url)
        try:
            while True:
                res = await client.blpop([self._key(request_id)], timeout=5)
                if res is None:
                    continue  # timeout — keep waiting until cancelled
                _, raw = res
                return json.loads(raw)
        finally:
            await client.aclose()


@lru_cache(maxsize=1)
def get_resolution_store() -> ResolutionStore:
    """Return the required Redis-backed cross-replica resolution store."""
    import redis
    import redis.asyncio  # noqa: F401

    return ResolutionStore(get_settings().redis_url)
