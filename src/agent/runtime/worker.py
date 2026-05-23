"""Agent run worker — background coroutine that drives the LangGraph for a single run.

Lifecycle:
1. Pull RunRecord from store.
2. Build graph (with checkpointer if available).
3. ainvoke with prompt; emit events through store.emit.
4. Handle interrupts (approval / clarification) by:
   - Persist pending_* in store, expose via SSE/REST.
   - Wait on the per-run async future for user resolution.
5. On final response, store.complete; on exception, store.fail.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.types import Command

from ..config import get_settings
from ..graph import GraphConfig, build_graph, initial_state
from ..llm import build_client
from ..memory import close_async_checkpointer, create_async_checkpointer
from ..tools import build_default_registry
from .store import RunStore


# Per-run resolution future map: request_id → asyncio.Future[Any]
_RESOLUTIONS: dict[str, asyncio.Future[Any]] = {}


def submit_resolution(request_id: str, payload: Any) -> bool:
    fut = _RESOLUTIONS.get(request_id)
    if fut is None or fut.done():
        return False
    fut.get_loop().call_soon_threadsafe(fut.set_result, payload)
    return True


async def _wait_resolution(request_id: str, loop: asyncio.AbstractEventLoop) -> Any:
    fut: asyncio.Future = loop.create_future()
    _RESOLUTIONS[request_id] = fut
    try:
        return await fut
    finally:
        _RESOLUTIONS.pop(request_id, None)


async def run_agent(
    *,
    store: RunStore,
    request_id: str,
    baseline_flow: dict[str, Any] | None = None,
    auth_claims: dict[str, Any] | None = None,
) -> None:
    s = get_settings()
    rec = await store.get(request_id)
    if rec is None:
        return

    async def emit(name: str, payload: dict[str, Any]) -> None:
        await store.emit(request_id, name, payload)

    await store.set_status(request_id, "running")
    await emit("run_started", {"thread_id": rec.thread_id, "provider": s.llm_provider})

    checkpointer = None
    try:
        registry = build_default_registry()
        cfg = GraphConfig(
            llm=build_client(s.llm_provider, s),
            registry=registry,
            model=s.model_executor,
            max_tokens=s.max_output_tokens,
            emit=emit,
            services={},
            approve_destructive=False,
        )
        try:
            checkpointer = await create_async_checkpointer(s.database_url)
        except Exception as cp_err:
            print(f"[agent_worker] checkpointer init failed, using in-memory: {cp_err}", flush=True)
            checkpointer = None
        graph = build_graph(cfg, checkpointer=checkpointer)

        state = initial_state(
            run_id=request_id,
            project_id=rec.project_id,
            user_id=rec.user_id,
            thread_id=rec.thread_id,
            user_prompt=rec.prompt,
            baseline_flow=baseline_flow,
        )
        if auth_claims:
            state["auth_claims"] = auth_claims  # type: ignore[typeddict-unknown-key]

        config = {
            "configurable": {"thread_id": rec.thread_id},
            "recursion_limit": 200,
        }
        loop = asyncio.get_running_loop()
        result = await graph.ainvoke(state, config=config)

        while result.get("__interrupt__"):
            interrupt_obj = result["__interrupt__"][0]
            payload = getattr(interrupt_obj, "value", interrupt_obj)
            if not isinstance(payload, dict):
                payload = {"question": str(payload)}
            kind = payload.get("kind", "clarify")
            if kind == "approval":
                await store.set_pending_approval(request_id, payload)
                await emit("approval_required", payload)
            else:
                await store.set_pending_clarification(request_id, payload)
                await emit("clarification_required", payload)

            answer = await _wait_resolution(request_id, loop)

            if kind == "approval":
                await store.set_pending_approval(request_id, None)
                await emit(
                    "approval_resolved",
                    {"tool_call_id": payload.get("tool_call_id"), "approved": bool(answer.get("approved")) if isinstance(answer, dict) else bool(answer)},
                )
            else:
                await store.set_pending_clarification(request_id, None)
                await emit(
                    "clarification_resolved",
                    {"tool_call_id": payload.get("tool_call_id"), "answer": str(answer)},
                )

            result = await graph.ainvoke(Command(resume=answer), config=config)

        # Final assistant text
        final_text = ""
        for m in reversed(result.get("messages") or []):
            if getattr(m, "role", None) == "assistant":
                final_text = getattr(m, "text", "")
                break
        # Emit the completion event BEFORE closing subscribers — otherwise the
        # SSE stream sees only `close` and the UI never marks the run complete.
        await emit("run_completed", {"final_response": final_text})
        await store.complete(request_id, final_text)
    except asyncio.CancelledError:
        # User-initiated cancellation via /cancel endpoint.
        print(f"[agent_worker] RUN CANCELLED ({request_id})", flush=True)
        await emit("run_failed", {"error": "Cancelled by user", "cancelled": True})
        await store.fail(request_id, "Cancelled by user")
        # Re-raise to honour the cancellation contract.
        raise
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        msg = f"{type(exc).__name__}: {exc}" if str(exc) else f"{type(exc).__name__} (no message)"
        print(f"[agent_worker] RUN FAILED ({request_id}):\n{tb}", flush=True)
        await emit("run_failed", {"error": msg, "traceback": tb})
        await store.fail(request_id, msg)
    finally:
        store.unregister_task(request_id)
        if checkpointer is not None:
            try:
                await close_async_checkpointer(checkpointer)
            except Exception:
                pass


def start_run_task(
    *,
    store: RunStore,
    request_id: str,
    baseline_flow: dict[str, Any] | None = None,
    auth_claims: dict[str, Any] | None = None,
) -> asyncio.Task:
    task = asyncio.create_task(
        run_agent(
            store=store,
            request_id=request_id,
            baseline_flow=baseline_flow,
            auth_claims=auth_claims,
        )
    )
    store.register_task(request_id, task)
    return task
