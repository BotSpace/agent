"""/projects/{project_id}/runs/* endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...runtime.store import RunStore
from ...runtime.worker import start_run_task, submit_resolution
from ..deps import CurrentUser, get_current_user, get_store
from ..sse import event_stream


router = APIRouter(prefix="/projects/{project_id}/runs", tags=["runs"])


class RunCreateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    thread_id: str | None = None
    baseline_flow: dict[str, Any] | None = Field(
        default=None, description="Initial flow JSON to seed working_flow"
    )


class RunCreateResponse(BaseModel):
    request_id: str
    status: str


class RunStatusResponse(BaseModel):
    request_id: str
    status: str
    final_response: str | None = None
    error: str | None = None
    pending_approval: dict[str, Any] | None = None
    pending_clarification: dict[str, Any] | None = None
    current_plan: list[dict[str, Any]] | None = None


class ClarifyRequest(BaseModel):
    answer: str = Field(min_length=1)


class ApproveRequest(BaseModel):
    approved: bool
    reason: str | None = None


@router.post("", response_model=RunCreateResponse)
async def create_run(
    project_id: str,
    body: RunCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> RunCreateResponse:
    used = await store.count_user_runs_today(user.user_id)
    if used >= user.prompt_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Kunlik limit tugadi ({user.prompt_limit} ta prompt). Ertaga qayta urinib ko'ring yoki tarifni yangilang.",
                "used": used,
                "limit": user.prompt_limit,
            },
        )

    try:
        from ...integrations.backend import get_backend_client
        backend = get_backend_client()
        usage = await backend.get_subscription_usage(auth_claims=user.claims)
        if usage.get("tokens_remaining", 1) <= 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Oylik token limiti tugadi. Tarifni yangilang.",
                    "tokens_used": usage.get("tokens_used_output", 0),
                    "tokens_limit": usage.get("tokens_limit", 0),
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass  # Backend mavjud bo'lmasa, runni to'xtatmaymiz

    rec = await store.create(
        prompt=body.prompt,
        project_id=project_id,
        user_id=user.user_id,
        thread_id=body.thread_id,
    )
    start_run_task(
        store=store,
        request_id=rec.request_id,
        baseline_flow=body.baseline_flow,
        auth_claims=user.claims,
    )
    return RunCreateResponse(request_id=rec.request_id, status="queued")


@router.get("/{request_id}", response_model=RunStatusResponse)
async def get_run(
    project_id: str,
    request_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> RunStatusResponse:
    rec = await store.get(request_id, project_id=project_id, user_id=user.user_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(
        request_id=rec.request_id,
        status=rec.status,
        final_response=rec.final_response,
        error=rec.error,
        pending_approval=rec.pending_approval,
        pending_clarification=rec.pending_clarification,
        current_plan=rec.current_plan,
    )


@router.get("/{request_id}/events")
async def stream_events(
    project_id: str,
    request_id: str,
    since_seq: int = 0,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> StreamingResponse:
    rec = await store.get(request_id, project_id=project_id, user_id=user.user_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        event_stream(store, request_id, since_seq=since_seq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{request_id}/clarify")
async def clarify_run(
    project_id: str,
    request_id: str,
    body: ClarifyRequest,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> dict[str, str]:
    rec = await store.get(request_id, project_id=project_id, user_id=user.user_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if rec.status != "waiting_for_clarification":
        raise HTTPException(status_code=409, detail="Run is not waiting for clarification")
    if not submit_resolution(request_id, body.answer):
        raise HTTPException(status_code=409, detail="No active worker awaiting this run")
    return {"status": "resuming"}


@router.post("/{request_id}/cancel")
async def cancel_run(
    project_id: str,
    request_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> dict[str, str]:
    rec = await store.get(request_id, project_id=project_id, user_id=user.user_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if rec.status in ("completed", "failed"):
        return {"status": rec.status}
    ok = await store.cancel(request_id)
    if not ok:
        raise HTTPException(status_code=409, detail="No active worker for this run")
    return {"status": "cancelling"}


@router.post("/{request_id}/approve")
async def approve_run(
    project_id: str,
    request_id: str,
    body: ApproveRequest,
    user: CurrentUser = Depends(get_current_user),
    store: RunStore = Depends(get_store),
) -> dict[str, str]:
    rec = await store.get(request_id, project_id=project_id, user_id=user.user_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if rec.status != "waiting_for_approval":
        raise HTTPException(status_code=409, detail="Run is not waiting for approval")
    if not submit_resolution(request_id, {"approved": body.approved, "reason": body.reason}):
        raise HTTPException(status_code=409, detail="No active worker awaiting this run")
    return {"status": "resumed", "approved": "true" if body.approved else "false"}
