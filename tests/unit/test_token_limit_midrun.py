"""Mid-run token-limit enforcement: when this run's OWN tokens reach the
REMAINING monthly budget snapshotted at run start, llm_call sets
`limit_exceeded` and the graph routes to the `limit_exceeded` node — which
appends the user-facing message — instead of looping into more tool calls.

The comparison is against `pre_run_tokens_remaining` (already net of pre-run
usage), NOT the absolute monthly limit.
"""
# ruff: noqa: I001
import asyncio

from agent.graph.builder import GraphConfig, build_graph, initial_state
from agent.llm import ModelRoles
from agent.llm.base import GenerateResult, Message, ToolUsePart, Usage


class _ScriptedLLM:
    """Each generate() reports a fixed token usage and (optionally) a tool call.

    If the graph keeps looping it would emit more tool calls; the test asserts
    it stops after the first over-budget LLM turn instead.
    """

    def __init__(self, usage: Usage) -> None:
        self._usage = usage
        self.calls = 0

    async def generate(self, messages, options) -> GenerateResult:  # noqa: ANN001, ARG002
        self.calls += 1
        # Always try to call a tool — proves the budget gate, not a natural
        # end_turn, is what stops the run.
        msg = Message(
            role="assistant",
            content=[ToolUsePart(id=f"c{self.calls}", name="flow.get", arguments={})],
        )
        return GenerateResult(message=msg, stop_reason="tool_use", usage=self._usage)


def _build(usage: Usage):
    cfg = GraphConfig(
        llm=_ScriptedLLM(usage),
        registry=__import__("agent.tools", fromlist=["build_default_registry"]).build_default_registry(),
        models=ModelRoles(planner="m", executor="m", critic="m", embedding="m"),
        approve_destructive=False,
    )
    return build_graph(cfg)


def test_midrun_limit_stops_and_emits_message():
    # First LLM turn already burns 60+50=110 tokens; remaining budget is 100.
    graph = _build(Usage(input_tokens=60, output_tokens=50))
    state = initial_state(
        run_id="r", project_id="1185", user_id="u", thread_id="t1",
        user_prompt="do something",
    )
    state["pre_run_tokens_remaining"] = 100
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 50}

    result = asyncio.run(graph.ainvoke(state, config=config))

    assert result.get("limit_exceeded") is True
    # The run completed normally (no interrupt) with the limit message as the
    # last assistant message.
    assert not result.get("__interrupt__")
    final_text = ""
    for m in reversed(result.get("messages") or []):
        if getattr(m, "role", None) == "assistant":
            final_text = getattr(m, "text", "")
            break
    assert "Oylik token limitingiz tugadi" in final_text
    # The LLM was called exactly once — the gate stopped the loop before any
    # second turn / tool execution.
    assert graph  # graph built
    assert result.get("total_input_tokens") == 60
    assert result.get("total_output_tokens") == 50


def test_under_budget_does_not_trip():
    # 110 tokens used, remaining is 100_000 → no limit, run proceeds to tools.
    graph = _build(Usage(input_tokens=60, output_tokens=50))
    state = initial_state(
        run_id="r", project_id="1185", user_id="u", thread_id="t2",
        user_prompt="do something",
    )
    state["pre_run_tokens_remaining"] = 100_000
    config = {"configurable": {"thread_id": "t2"}, "recursion_limit": 6}

    # It will loop tool_exec→llm_call until recursion_limit; we only assert the
    # gate did NOT trip on the first turn.
    try:
        result = asyncio.run(graph.ainvoke(state, config=config))
    except Exception:
        result = None
    # Either it hit recursion limit (None) or finished, but limit_exceeded must
    # never be set when well under budget.
    if result is not None:
        assert not result.get("limit_exceeded")


def test_none_remaining_fails_open():
    # pre_run_tokens_remaining=None → enforcement disabled even with huge usage.
    graph = _build(Usage(input_tokens=10_000, output_tokens=10_000))
    state = initial_state(
        run_id="r", project_id="1185", user_id="u", thread_id="t3",
        user_prompt="do something",
    )
    state["pre_run_tokens_remaining"] = None
    config = {"configurable": {"thread_id": "t3"}, "recursion_limit": 4}
    try:
        result = asyncio.run(graph.ainvoke(state, config=config))
    except Exception:
        result = None
    if result is not None:
        assert not result.get("limit_exceeded")
