"""/projects/{project_id}/threads/* endpoints — DB-backed chat history.

Read-only views over ``agent_runs`` / ``agent_messages`` / ``agent_events``
that let the dashboard load conversation history from Postgres instead of
localStorage. Auth + scoping mirror ``runs.py``: threads are filtered by the
caller's ``user_id`` so a user only ever sees their own conversations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...runtime.store import RunStore
from ..deps import CurrentUser, get_current_user, get_store


router = APIRouter(prefix="/projects/{project_id}/threads", tags=["threads"])


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str
    run_count: int


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummary]


class ThreadTurnsResponse(BaseModel):
    thread_id: str
    turns: list[dict[str, Any]]


@router.get("", response_model=ThreadListResponse)
async def list_threads(
    project_id: str,
    limit: int = 50,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> ThreadListResponse:
    limit = max(1, min(limit, 200))
    threads = await store.list_threads(
        project_id=project_id, user_id=user.user_id, limit=limit
    )
    return ThreadListResponse(threads=[ThreadSummary(**t) for t in threads])


@router.get("/{thread_id}", response_model=ThreadTurnsResponse)
async def get_thread(
    project_id: str,
    thread_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> ThreadTurnsResponse:
    turns = await store.get_thread_turns(
        project_id=project_id, user_id=user.user_id, thread_id=thread_id
    )
    return ThreadTurnsResponse(thread_id=thread_id, turns=turns)
