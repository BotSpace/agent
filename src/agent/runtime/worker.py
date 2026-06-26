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
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

from langgraph.types import Command

from ..config import get_settings
from ..graph import GraphConfig, build_graph, initial_state
from ..integrations.dynamic_config import build_effective_client, effective_model_roles
from ..memory import close_async_checkpointer, create_async_checkpointer
from ..observability.metrics import AGENT_RUN_DURATION, AGENT_RUNS
from ..tools import build_main_registry
from .store import RunStore


def _maybe_traceable(name: str):
    """Return langsmith @traceable when tracing is enabled, else a no-op.
    Lets us wrap run_agent without forcing the dependency in tests/dev."""
    enabled = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower() == "true"
    if not enabled or not os.getenv("LANGSMITH_API_KEY"):
        def _noop(fn):
            return fn
        return _noop
    try:
        from langsmith import traceable  # type: ignore[import-untyped]
    except ImportError:
        def _noop(fn):
            return fn
        return _noop
    return traceable(name=name, run_type="chain")


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


@_maybe_traceable("botmother.agent.run")
async def run_agent(
    *,
    store: RunStore,
    request_id: str,
    baseline_flow: dict[str, Any] | None = None,
    auth_claims: dict[str, Any] | None = None,
    collections: list[dict[str, Any]] | None = None,
    module_nodes: list[dict[str, Any]] | None = None,
) -> None:
    s = get_settings()
    rec = await store.get(request_id)
    if rec is None:
        return

    async def emit(name: str, payload: dict[str, Any]) -> None:
        await store.emit(request_id, name, payload)

    # Resolve effective provider/model/region ONCE per run (fetch+cache).
    # Falls back to env settings if backend is unreachable.
    llm_client = await build_effective_client()
    models = await effective_model_roles()
    effective_prov = llm_client.__class__.__name__

    await store.set_status(request_id, "running")
    await emit("run_started", {"thread_id": rec.thread_id, "provider": effective_prov})

    run_started_at = time.monotonic()
    run_outcome = "completed"
    checkpointer = None
    try:
        registry = build_main_registry()
        cfg = GraphConfig(
            llm=llm_client,
            registry=registry,
            models=models,
            max_tokens=s.max_output_tokens,
            max_tool_calls=s.max_tool_calls,
            emit=emit,
            services={},
            approve_destructive=False,
        )
        try:
            checkpointer = await create_async_checkpointer(s.database_url)
        except Exception as cp_err:
            print(f"[agent_worker] checkpointer init failed, using in-memory: {cp_err}", flush=True)
            checkpointer = None
        # Share the worker-owned checkpointer with sub-agents (task.run) so their
        # threads persist on the same Postgres saver and can resume across turns.
        # task.run must NOT close this — the worker owns its lifecycle (finally).
        cfg.services["checkpointer"] = checkpointer
        # Sub-agents (task.run) need the run store + the MAIN request_id so they
        # can drive their OWN interrupt-resume loop: a sub-graph approval /
        # clarify interrupt is registered against the SAME request_id the
        # dashboard POSTs approve/clarify to (see api/routes/runs.py), then
        # resolved via submit_resolution. Without this the sub-agent's interrupt
        # was dead-ended → approve_run saw pending_approval is None → HTTP 409.
        cfg.services["store"] = store
        cfg.services["request_id"] = request_id
        graph = build_graph(cfg, checkpointer=checkpointer)

        state = initial_state(
            run_id=request_id,
            project_id=rec.project_id,
            user_id=rec.user_id,
            thread_id=rec.thread_id,
            user_prompt=rec.prompt,
            baseline_flow=baseline_flow,
        )
        # Snapshot the user's REMAINING monthly token budget ONCE at run start.
        # llm_call compares this run's own tokens against it and stops gracefully
        # when exhausted. FAIL OPEN: any error → remaining=None → no enforcement
        # (a failed usage fetch must never block a run).
        pre_run_remaining: int | None = None
        try:
            from ..integrations.backend import get_backend_client
            # Read the budget for the BILLED user — i.e. rec.user_id, the same
            # user tokens are reported to (report_token_usage below). Always
            # prefer it over auth_claims so we don't check the budget against a
            # different (chatter) user or fail when auth_claims is absent.
            fetch_claims = {"user_id": str(rec.user_id)} if rec.user_id else auth_claims
            usage = await get_backend_client().subscription_usage(auth_claims=fetch_claims)
            remaining = (usage.get("ai_token_usage") or {}).get("remaining")
            pre_run_remaining = int(remaining) if remaining is not None else None
        except Exception:
            logger.warning("Pre-run token budget fetch failed; running without mid-run enforcement", exc_info=True)
            pre_run_remaining = None
        state["pre_run_tokens_remaining"] = pre_run_remaining
        logger.info(
            "pre-run token budget: user=%s pre_run_remaining=%s", rec.user_id, pre_run_remaining
        )
        if auth_claims:
            state["auth_claims"] = auth_claims  # type: ignore[typeddict-unknown-key]
        if collections:
            state["collections"] = collections  # type: ignore[typeddict-unknown-key]
        if module_nodes:
            state["module_nodes"] = module_nodes

        config = {
            "configurable": {"thread_id": rec.thread_id},
            "recursion_limit": 500,
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
                answer_str = (
                    ", ".join(answer) if isinstance(answer, list) else str(answer)
                )
                await emit(
                    "clarification_resolved",
                    {"tool_call_id": payload.get("tool_call_id"), "answer": answer_str},
                )

            result = await graph.ainvoke(Command(resume=answer), config=config)

        # Final assistant text
        final_text = ""
        for m in reversed(result.get("messages") or []):
            if getattr(m, "role", None) == "assistant":
                final_text = getattr(m, "text", "")
                break
        # Token usage for THIS run (includes sub-agents). Emit it in the
        # completion event so the chat UI can show per-prompt token usage.
        total_in = result.get("total_input_tokens", 0)
        total_out = result.get("total_output_tokens", 0)
        # Emit the completion event BEFORE closing subscribers — otherwise the
        # SSE stream sees only `close` and the UI never marks the run complete.
        await emit(
            "run_completed",
            {
                "final_response": final_text,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
        )
        await store.complete(request_id, final_text)

        # Record token usage
        await store.record_tokens(request_id, total_in, total_out)

        try:
            from ..integrations.backend import get_backend_client
            backend = get_backend_client()
            await backend.report_token_usage(
                user_id=rec.user_id or "",
                input_tokens=total_in,
                output_tokens=total_out,
                project_id=rec.project_id,
                auth_claims=auth_claims,
            )
        except Exception:
            logger.warning("Failed to report token usage to backend", exc_info=True)
    except asyncio.CancelledError:
        # User-initiated cancellation via /cancel endpoint.
        run_outcome = "cancelled"
        print(f"[agent_worker] RUN CANCELLED ({request_id})", flush=True)
        await emit("run_failed", {"error": "Cancelled by user", "cancelled": True})
        await store.fail(request_id, "Cancelled by user")
        # Re-raise to honour the cancellation contract.
        raise
    except Exception as exc:
        run_outcome = "failed"
        import traceback

        tb = traceback.format_exc()
        msg = f"{type(exc).__name__}: {exc}" if str(exc) else f"{type(exc).__name__} (no message)"
        print(f"[agent_worker] RUN FAILED ({request_id}):\n{tb}", flush=True)
        await emit("run_failed", {"error": msg, "traceback": tb})
        await store.fail(request_id, msg)
    finally:
        # Run-level Prometheus metrikalari — outcome bo'yicha hisob + davomiylik.
        try:
            AGENT_RUNS.labels(run_outcome).inc()
            AGENT_RUN_DURATION.observe(time.monotonic() - run_started_at)
        except Exception:  # noqa: BLE001
            logger.debug("agent run metric record failed", exc_info=True)
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
    collections: list[dict[str, Any]] | None = None,
    module_nodes: list[dict[str, Any]] | None = None,
) -> asyncio.Task:
    task = asyncio.create_task(
        run_agent(
            store=store,
            request_id=request_id,
            baseline_flow=baseline_flow,
            auth_claims=auth_claims,
            collections=collections,
            module_nodes=module_nodes,
        )
    )
    store.register_task(request_id, task)
    return task
