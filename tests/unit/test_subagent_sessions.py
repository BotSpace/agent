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

import pytest

import agent.graph as graph_mod
import agent.llm as llm_mod
from agent.llm.base import Message, TextPart
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
    # Fresh seed injects the full sub-agent spec prompt + the task.
    assert any("[Sub-agent: flow-builder]" in t for t in texts)
    assert any("Build a /start menu" in t for t in texts)


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
