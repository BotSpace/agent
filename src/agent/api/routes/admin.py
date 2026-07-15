"""/admin/* endpoints — internal, server-to-server only.

Read-only views the admin panel uses to inspect a user's AI-agent
conversations (the botspace admin's "AI suhbatlar" tab on the user detail
page). Auth is the shared internal token (``X-Internal-Token``) via
``require_internal`` — a logged-in end user can never reach these, only a
trusted service holding the token. All reads are scoped by ``user_id`` so an
admin only ever sees that one user's threads.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...runtime.store import RunStore
from ..deps import get_store, require_internal
from .threads import ThreadTurnsResponse

router = APIRouter(
    prefix="/admin/users/{user_id}/threads",
    tags=["admin"],
    dependencies=[Depends(require_internal)],
)


@router.get("")
async def list_user_threads(
    user_id: str,
    limit: int = 100,
    store: RunStore = Depends(get_store),
) -> dict[str, Any]:
    """List all of a user's threads across every project, newest first."""
    limit = max(1, min(limit, 200))
    threads = await store.list_threads_for_user(user_id=user_id, limit=limit)
    return {"threads": threads}


@router.get("/{thread_id}", response_model=ThreadTurnsResponse)
async def get_user_thread(
    user_id: str,
    thread_id: str,
    store: RunStore = Depends(get_store),
) -> ThreadTurnsResponse:
    """Full transcript for one of the user's threads. Scoped to user_id, so a
    thread that isn't theirs yields an empty ``turns`` list."""
    turns = await store.get_thread_turns(
        project_id=None, user_id=user_id, thread_id=thread_id
    )
    return ThreadTurnsResponse(thread_id=thread_id, turns=turns, active_run=None)
