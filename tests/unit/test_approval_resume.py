"""Regression: a destructive tool (flow.clear) must EXECUTE once the user
approves it — even though the model re-issues the call with a brand new
tool_use id after the interrupt resumes.

The old code keyed the approval flag on the per-turn `tool_use` id
(`approved_{call.id}`), which (a) was an undeclared state channel that the
checkpointer dropped, and (b) never matched the re-issued call's new id. The
result was an infinite approval loop: the user kept being asked to approve
flow.clear over and over. We now key approval on a stable (name, arguments)
signature stored in a declared `approvals` channel, consumed once.
"""
# ruff: noqa: I001
import asyncio

from langgraph.types import Command

from agent.graph.builder import GraphConfig, build_graph, initial_state
from agent.llm import ModelRoles
from agent.llm.base import GenerateResult, Message, ToolUsePart, Usage
from agent.tools import build_default_registry


class _ScriptedLLM:
    """Replays a fixed list of assistant messages, one per generate() call."""

    def __init__(self, turns: list[Message]) -> None:
        self._turns = turns
        self.calls = 0

    async def generate(self, messages, options) -> GenerateResult:  # noqa: ANN001, ARG002
        msg = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        stop = "tool_use" if msg.tool_calls else "end_turn"
        return GenerateResult(message=msg, stop_reason=stop, usage=Usage())


def _clear_call(call_id: str) -> Message:
    return Message(
        role="assistant",
        content=[ToolUsePart(id=call_id, name="flow.clear", arguments={"confirm": True})],
    )


def _baseline() -> dict:
    return {
        "nodes": [{"id": "n1", "type": "CommandTriggerNode", "data": {"command": "/start"}}],
        "edges": [],
    }


def _build(turns: list[Message]):
    cfg = GraphConfig(
        llm=_ScriptedLLM(turns),
        registry=build_default_registry(),
        models=ModelRoles(planner="m", executor="m", critic="m", embedding="m"),
        approve_destructive=False,  # isolates the always_approve path: flow.clear gates regardless
    )
    return build_graph(cfg)  # default MemorySaver checkpointer


def test_flow_clear_executes_after_approval_with_new_call_id():
    # Turn 1: model issues flow.clear (id=c1) → blocks on approval.
    # Turn 2 (post-approval): model re-issues flow.clear with a NEW id (c2).
    # Turn 3: model finishes.
    graph = _build([
        _clear_call("c1"),
        _clear_call("c2"),
        Message.assistant_text("Flow tozalandi, yangidan boshladik."),
    ])
    state = initial_state(
        run_id="r", project_id="1185", user_id="u", thread_id="t1",
        user_prompt="hammasini o'chir va yangidan boshla",
        baseline_flow=_baseline(),
    )
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 100}

    async def _drive():
        result = await graph.ainvoke(state, config=config)
        assert result.get("__interrupt__"), "first flow.clear must pause for approval"
        # User approves → re-issued call (new id) must EXECUTE, not re-prompt.
        result = await graph.ainvoke(Command(resume={"approved": True}), config=config)
        return result

    result = asyncio.run(_drive())

    assert not result.get("__interrupt__"), "approved flow.clear must not loop on approval"
    wf = result.get("working_flow") or {}
    assert wf.get("nodes") == [], f"flow.clear should have emptied the flow, got {wf}"


def test_second_flow_clear_re_gates_after_consume():
    # After one approved+executed flow.clear, a LATER flow.clear must ask again
    # (consume-once) — approval is not a permanent grant.
    graph = _build([
        _clear_call("c1"),   # blocks
        _clear_call("c2"),   # executes after approval
        _clear_call("c3"),   # a fresh clear later in the same thread → must re-gate
        Message.assistant_text("done"),
    ])
    state = initial_state(
        run_id="r", project_id="1185", user_id="u", thread_id="t2",
        user_prompt="o'chir",
        baseline_flow=_baseline(),
    )
    config = {"configurable": {"thread_id": "t2"}, "recursion_limit": 100}

    async def _drive():
        r = await graph.ainvoke(state, config=config)
        assert r.get("__interrupt__")
        r = await graph.ainvoke(Command(resume={"approved": True}), config=config)
        return r

    result = asyncio.run(_drive())
    # The c3 clear (same signature, after consume) must pause again.
    assert result.get("__interrupt__"), "a later identical flow.clear must re-request approval"
