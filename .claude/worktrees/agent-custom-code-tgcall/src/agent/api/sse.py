"""SSE streaming generator from a RunStore subscription."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from ..runtime.events import AgentEvent
from ..runtime.store import RunStore


async def event_stream(
    store: RunStore,
    request_id: str,
    *,
    since_seq: int = 0,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """SSE generator — first replays missed events, then streams live."""
    # Replay
    for ev in await store.list_events(request_id, since_seq=since_seq):
        yield _sse_event(ev["type"], ev)

    q = store.subscribe(request_id)
    try:
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            if ev is None:
                # Stream complete
                yield _sse_event("close", {"reason": "run_finished"})
                return
            if isinstance(ev, AgentEvent):
                yield ev.to_sse()
            else:
                yield _sse_event("event", ev)
    finally:
        store.unsubscribe(request_id, q)


def _sse_event(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
