"""Tests for persistent, session-resumable sub-agents (task.run).

Covers:
  (a) same session_id → second task.run resumes the SAME thread_id and APPENDS
      a new user message instead of re-seeding the full system+spec history.
  (b) omitted session_id → fresh seed + a session_id is returned.
  (c) a sub-agent calling ITSELF is rejected.
  (d) recursion depth cap is enforced.

No Postgres: we inject a fake in-process checkpointer via ctx.services so the
resume-vs-fresh detection path is exercised deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent.graph as graph_mod
import agent.llm as llm_mod
from agent.llm.base import Message, TextPart
from agent.prompts.subagent_specs import REGISTRY
from agent.tools.base import ToolContext
from agent.tools.task import MAX_SUB_AGENT_DEPTH, TaskRunInput
from agent.tools.task import task_run as _task_run_tool

_handler = _task_run_tool.handler


class _FakeGraph:
    """Records the state passed to ainvoke and returns a canned final answer."""

    def __init__(self) -> None:
        self.invoked_with: list[dict] = []

    async def ainvoke(self, base, config=None):
        self.invoked_with.append({"base": base, "config": config})
        # Return a state with a final assistant message so task.run reports ok.
        return {
            "messages": [
                *(base.get("messages") or []),
                Message(role="assistant", content=[TextPart(text="done")]),
            ]
        }


class _FakeCheckpointer:
    """Minimal checkpointer: remembers which thread_ids have been seen."""

    def __init__(self) -> None:
        self.seen_threads: set[str] = set()

    async def aget(self, config):
        tid = config["configurable"]["thread_id"]
        if tid in self.seen_threads:
            return {"v": 1}  # truthy snapshot → resume
        return None


@pytest.fixture()
def patched(monkeypatch):
    """Patch the lazily-imported graph/llm pieces so task.run runs offline."""
    fake_graph = _FakeGraph()

    def _build_graph(cfg, checkpointer=None):
        # Mirror real signature; expose cfg for assertions.
        fake_graph.last_cfg = cfg
        fake_graph.last_checkpointer = checkpointer
        return fake_graph

    monkeypatch.setattr(graph_mod, "build_graph", _build_graph)
    monkeypatch.setattr(llm_mod, "build_client", lambda *a, **k: object())
    return fake_graph


def _ctx(checkpointer, *, services_extra=None):
    services = {"checkpointer": checkpointer}
    if services_extra:
        services.update(services_extra)
    emitted: list = []

    async def _emit(name, payload):
        emitted.append((name, payload))

    ctx = ToolContext(
        run_id="run-XYZ",  # changes per main turn — must NOT key the thread
        project_id="proj-1",
        user_id="user-1",
        state={"thread_id": "conv-STABLE"},
        emit=_emit,
        services=services,
    )
    ctx._emitted = emitted  # type: ignore[attr-defined]
    return ctx


def _user_texts(base) -> list[str]:
    out = []
    for m in base.get("messages") or []:
        for p in m.content:
            txt = getattr(p, "text", None)
            if txt is not None:
                out.append(txt)
    return out


@pytest.mark.asyncio
async def test_fresh_session_seeds_and_returns_session_id(patched):
    cp = _FakeCheckpointer()
    ctx = _ctx(cp)
    res = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="Build a /start menu"),
        ctx,
    )
    assert not res.is_error
    sid = res.data["session_id"]
    assert sid  # a fresh session id was generated and returned
    assert res.data["resumed"] is False
    # Stable thread id keyed on the CONVERSATION thread_id, not run_id.
    assert res.data["sub_thread_id"] == f"conv-STABLE:sub:flow-builder:{sid}"

    base = patched.invoked_with[0]["base"]
    texts = _user_texts(base)
    # The sub-agent spec prompt is now the SYSTEM message (cfg.system_block),
    # NOT seeded into the user turn. The fresh user message carries only the task.
    assert patched.last_cfg.system_block == REGISTRY["flow-builder"].full_prompt
    assert any("Build a /start menu" in t for t in texts)
    assert not any("[Sub-agent: flow-builder]" in t for t in texts)


@pytest.mark.asyncio
async def test_same_session_resumes_same_thread_and_appends(patched):
    cp = _FakeCheckpointer()
    ctx1 = _ctx(cp)
    res1 = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="Build a /start menu"),
        ctx1,
    )
    sid = res1.data["session_id"]
    thread = res1.data["sub_thread_id"]
    # Simulate the checkpointer now persisting this thread.
    cp.seen_threads.add(thread)

    # Second call with SAME session_id on a NEW main turn (different run_id).
    ctx2 = _ctx(cp)
    ctx2.run_id = "run-DIFFERENT"  # main run id changed across turns
    res2 = await _handler(
        TaskRunInput(
            subagent_type="flow-builder",
            prompt="Now make the buttons inline",
            session_id=sid,
        ),
        ctx2,
    )
    assert not res2.is_error
    # Identical thread id → checkpointer resumes the SAME thread.
    assert res2.data["sub_thread_id"] == thread
    assert res2.data["resumed"] is True

    # Resume path APPENDS only the new task — it must NOT re-seed the spec prompt.
    base = patched.invoked_with[-1]["base"]
    texts = _user_texts(base)
    assert any("Now make the buttons inline" in t for t in texts)
    assert not any("[Sub-agent: flow-builder]" in t for t in texts), (
        "resume must not re-inject the full system/spec prompt"
    )


@pytest.mark.asyncio
async def test_self_call_rejected(patched):
    cp = _FakeCheckpointer()
    # Simulate being INSIDE flow-builder (nested task.run).
    ctx = _ctx(cp, services_extra={"current_subagent_type": "flow-builder", "sub_agent_depth": 1})
    res = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="recurse into myself"),
        ctx,
    )
    assert res.is_error
    assert "cannot delegate to ITSELF" in res.content
    # No graph was invoked.
    assert patched.invoked_with == []


@pytest.mark.asyncio
async def test_depth_cap_enforced(patched):
    cp = _FakeCheckpointer()
    # Already at the max nesting depth → any further task.run must be rejected.
    ctx = _ctx(cp, services_extra={"sub_agent_depth": MAX_SUB_AGENT_DEPTH})
    res = await _handler(
        TaskRunInput(subagent_type="cms-manager", prompt="too deep"),
        ctx,
    )
    assert res.is_error
    assert "depth cap" in res.content.lower()
    assert patched.invoked_with == []


class _InterruptOnceGraph:
    """Interrupts on the FIRST ainvoke, then finishes on the Command(resume)."""

    def __init__(self) -> None:
        self.calls: list = []

    async def ainvoke(self, base, config=None):
        self.calls.append(base)
        if len(self.calls) == 1:
            # Simulate a sub-graph approval interrupt (flow.clear / always_approve).
            iobj = SimpleNamespace(
                value={
                    "kind": "approval",
                    "tool_name": "flow.clear",
                    "tool_call_id": "tu-clear",
                    "reason": "wipes the working flow",
                }
            )
            return {"__interrupt__": [iobj]}
        # Resumed: finish with a final assistant message.
        return {
            "messages": [Message(role="assistant", content=[TextPart(text="cleared")])]
        }


class _FakeStore:
    """Records pending_* writes and (for approval) auto-resolves the waiting
    future via submit_resolution — emulating the dashboard POSTing /approve."""

    def __init__(self) -> None:
        self.events: list = []

    async def set_pending_approval(self, request_id, payload):
        self.events.append(("set_pending_approval", request_id, payload))
        if payload is not None:
            # Dashboard side (async, like the real /approve HTTP POST): the
            # future is registered by _wait_resolution AFTER this returns, so
            # defer the resolution to a follow-up task that retries until the
            # waiter exists.
            self._resolve_soon(request_id, {"approved": True})

    async def set_pending_clarification(self, request_id, payload):
        self.events.append(("set_pending_clarification", request_id, payload))
        if payload is not None:
            self._resolve_soon(request_id, "yes")

    def _resolve_soon(self, request_id, answer):
        import asyncio as _asyncio

        from agent.runtime.worker import submit_resolution

        async def _deliver():
            for _ in range(100):
                if submit_resolution(request_id, answer):
                    return
                await _asyncio.sleep(0)

        _asyncio.get_running_loop().create_task(_deliver())


@pytest.mark.asyncio
async def test_subgraph_approval_interrupt_resumes_no_409(patched, monkeypatch):
    """A sub-agent approval interrupt must register pending_approval on the MAIN
    request_id (so approve_run no longer sees None → 409), await resolution, then
    RESUME the sub-graph in place — not bubble up."""
    interrupt_graph = _InterruptOnceGraph()
    monkeypatch.setattr(
        graph_mod, "build_graph", lambda cfg, checkpointer=None: interrupt_graph
    )

    cp = _FakeCheckpointer()
    store = _FakeStore()
    ctx = _ctx(cp, services_extra={"store": store, "request_id": "MAIN-REQ-1"})

    res = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="clear the flow"),
        ctx,
    )

    # Sub-graph was resumed (2 invocations: initial + Command(resume)).
    assert len(interrupt_graph.calls) == 2
    from langgraph.types import Command as _Cmd

    assert isinstance(interrupt_graph.calls[1], _Cmd)

    # Pending approval was registered on the MAIN request_id, then cleared.
    names = [(e[0], e[1], e[2]) for e in store.events]
    assert ("set_pending_approval", "MAIN-REQ-1", store.events[0][2]) in names
    assert store.events[0][2] is not None  # set
    assert any(
        e[0] == "set_pending_approval" and e[1] == "MAIN-REQ-1" and e[2] is None
        for e in store.events
    )  # cleared

    # approval_required + approval_resolved were emitted (wrapped with sub id).
    emitted_names = [n for n, _ in ctx._emitted]
    assert "approval_required" in emitted_names
    assert "approval_resolved" in emitted_names

    # The sub-agent finished successfully (no 409 dead-end, no bubble note).
    assert not res.is_error
    assert "cleared" in res.content
    # It did NOT take the legacy bubbling path (no "needs APPROVAL" note).
    assert "needs APPROVAL" not in res.content


@pytest.mark.asyncio
async def test_nested_delegation_threads_depth_and_identity(patched):
    """A first-level sub-agent call sets depth=1 and identity for nested calls."""
    cp = _FakeCheckpointer()
    ctx = _ctx(cp)
    await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="build"),
        ctx,
    )
    cfg = patched.last_cfg
    assert cfg.services["sub_agent_depth"] == 1
    assert cfg.services["current_subagent_type"] == "flow-builder"
    assert cfg.services["current_sub_agent_id"].startswith("sa-flow-builder-")
    # Checkpointer is propagated downstream for nested persistence.
    assert cfg.services["checkpointer"] is cp


# ── token usage aggregation (DELTA, resume-safe) ─────────────────────────────


class _TokenGraph:
    """ainvoke returns a final state carrying the sub-graph's CUMULATIVE token
    totals (total_in/total_out are set by the test per call)."""

    def __init__(self) -> None:
        self.total_in = 0
        self.total_out = 0
        self.invoked_with: list[dict] = []

    async def ainvoke(self, base, config=None):
        self.invoked_with.append({"base": base, "config": config})
        return {
            "messages": [
                *(base.get("messages") or []),
                Message(role="assistant", content=[TextPart(text="done")]),
            ],
            "total_input_tokens": self.total_in,
            "total_output_tokens": self.total_out,
        }


class _TokenCheckpointer:
    """aget returns a snapshot with channel_values (prior cumulative tokens) for
    threads marked seen (resume); None otherwise (fresh)."""

    def __init__(self) -> None:
        self.seen: dict[str, tuple[int, int]] = {}

    async def aget(self, config):
        tid = config["configurable"]["thread_id"]
        if tid in self.seen:
            pin, pout = self.seen[tid]
            return {"channel_values": {"total_input_tokens": pin, "total_output_tokens": pout}}
        return None


@pytest.fixture()
def token_patched(monkeypatch):
    """Offline graph + hermetic LLM/config (no env or network)."""
    graph = _TokenGraph()
    monkeypatch.setattr(graph_mod, "build_graph", lambda cfg, checkpointer=None: graph)

    import agent.integrations.dynamic_config as dyn
    from agent.llm import ModelRoles

    roles = ModelRoles(planner="m", executor="m", critic="m", embedding="m")

    async def _roles():
        return roles

    async def _client():
        return object()

    monkeypatch.setattr(dyn, "effective_model_roles", _roles)
    monkeypatch.setattr(dyn, "build_effective_client", _client)
    return graph


@pytest.mark.asyncio
async def test_subagent_token_delta_surfaced_for_parent_aggregation(token_patched):
    """A fresh sub-agent's full token usage lands in ToolResult.data as
    input_tokens/output_tokens — the exact keys the parent tool_exec node reads
    (builder.py) to aggregate sub-agent tokens into the main report."""
    graph = token_patched
    graph.total_in, graph.total_out = 120, 45
    ctx = _ctx(_TokenCheckpointer())
    res = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="build"), ctx
    )
    assert not res.is_error
    # Fresh: prior = 0 → delta == full usage.
    assert res.data["input_tokens"] == 120
    assert res.data["output_tokens"] == 45


@pytest.mark.asyncio
async def test_subagent_token_delta_not_double_counted_on_resume(token_patched):
    """On resume the sub-graph's total_* is CUMULATIVE (checkpointer reloads prior
    turns). Only the per-call DELTA must be reported, never the cumulative — else
    every resume re-counts all earlier tokens."""
    graph = token_patched
    cp = _TokenCheckpointer()

    # Turn 1 (fresh): sub-graph used 120/45.
    graph.total_in, graph.total_out = 120, 45
    ctx1 = _ctx(cp)
    res1 = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="build"), ctx1
    )
    sid = res1.data["session_id"]
    thread = res1.data["sub_thread_id"]
    assert res1.data["input_tokens"] == 120  # fresh full usage

    # Persist prior cumulative as the checkpointer would after that turn.
    cp.seen[thread] = (120, 45)

    # Turn 2 (resume, new main turn): cumulative grows to 200/80.
    graph.total_in, graph.total_out = 200, 80
    ctx2 = _ctx(cp)
    ctx2.run_id = "run-DIFFERENT"
    res2 = await _handler(
        TaskRunInput(subagent_type="flow-builder", prompt="more", session_id=sid),
        ctx2,
    )
    assert res2.data["resumed"] is True
    # DELTA only (200-120, 80-45) — NOT the cumulative 200/80.
    assert res2.data["input_tokens"] == 80
    assert res2.data["output_tokens"] == 35
