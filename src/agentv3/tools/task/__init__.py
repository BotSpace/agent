"""task.run — sub-agent delegation (Claude Code'dagi Agent tool kabi).

Sub-agent:
- Alohida tool registry (parent tool'larining subset'i)
- Alohida thread (yangi LangGraph instance)
- Alohida system prompt (subagent_specs.py'dan)
- Parent state'ning kerakli qismi forward qilinadi (project_id, baseline_flow, ...)
- Natija — final assistant text + agar bo'lsa, working_flow patch

Bir nechta sub-agent ni parallel chaqirish uchun `task.run_parallel` keyinroq
qo'shilishi mumkin (asyncio.gather).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...prompts.subagent_specs import REGISTRY, list_subagents
from ..base import Tool, ToolContext, ToolResult, tool


class TaskRunInput(BaseModel):
    subagent_type: str = Field(
        description="Sub-agent name: flow-builder | crm-analyst | bot-deployer | broadcaster"
    )
    prompt: str = Field(description="The task description for the sub-agent (1-2 sentences + context)")
    propagate_flow: bool = Field(
        default=True,
        description="Forward the working flow to the sub-agent so it can continue editing it",
    )


@tool(
    name="task.run",
    description=(
        "Delegate a focused sub-task to a specialised sub-agent. Use this when the work splits "
        "cleanly into a sub-problem with a defined scope (e.g. 'build the /start branch', "
        "'find all VIP users', 'deploy the latest version'). Returns the sub-agent's final text "
        "and merges any flow changes back into the working flow.\n\n"
        "Available sub-agents: flow-builder, crm-analyst, bot-deployer, broadcaster."
    ),
    input_model=TaskRunInput,
)
async def task_run(input: TaskRunInput, ctx: ToolContext) -> ToolResult:
    spec = REGISTRY.get(input.subagent_type)
    if spec is None:
        available = ", ".join(r.name for r in list_subagents())
        return ToolResult.error(
            f"Unknown sub-agent `{input.subagent_type}`. Available: {available}"
        )

    # Lazy import to avoid circular dep at module load time.
    from ...graph import GraphConfig, build_graph, initial_state
    from ...llm import build_client
    from ...config import get_settings
    from ...tools import build_default_registry
    from ...tools.registry import ToolRegistry

    s = get_settings()
    full = build_default_registry()
    allowed = set(spec.allowed_tools)
    sub_registry = ToolRegistry()
    for t in full.all():
        if t.name in allowed:
            sub_registry.register(t)

    model = {
        "planner": s.model_planner,
        "executor": s.model_executor,
        "critic": s.model_critic,
    }.get(spec.model_hint, s.model_executor)

    cfg = GraphConfig(
        llm=build_client(s.llm_provider, s),
        registry=sub_registry,
        model=model,
        max_tokens=s.max_output_tokens,
        emit=ctx.emit,
        services=ctx.services,
    )
    graph = build_graph(cfg)
    seeded_prompt = f"[Sub-agent: {spec.name}]\n{spec.full_prompt}\n\nTask: {input.prompt}"
    sub_state: dict[str, Any] = {
        "messages": [
            __import__("agentv3.llm.base", fromlist=["Message"]).Message.user_text(seeded_prompt)
        ],
        "run_id": f"{ctx.run_id}-sub-{spec.name}",
        "project_id": ctx.project_id,
        "user_id": ctx.user_id,
        "thread_id": f"{ctx.run_id}-{spec.name}",
    }
    if input.propagate_flow and ctx.state.get("working_flow"):
        sub_state["working_flow"] = ctx.state["working_flow"]
        sub_state["baseline_flow"] = ctx.state.get("baseline_flow")

    # Reuse initial_state for structure consistency.
    base = initial_state(
        run_id=sub_state["run_id"],
        project_id=ctx.project_id,
        user_id=ctx.user_id,
        thread_id=sub_state["thread_id"],
        user_prompt=seeded_prompt,
        baseline_flow=sub_state.get("baseline_flow"),
    )
    if input.propagate_flow and ctx.state.get("working_flow"):
        base["working_flow"] = ctx.state["working_flow"]
    # Forward user auth so sub-agent backend calls don't fail with 401.
    auth_claims = ctx.state.get("auth_claims")
    if auth_claims:
        base["auth_claims"] = auth_claims  # type: ignore[typeddict-unknown-key]

    result = await graph.ainvoke(
        base, config={"configurable": {"thread_id": sub_state["thread_id"]}}
    )
    messages = result.get("messages") or []
    final_text = ""
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            final_text = m.text
            break

    state_updates: dict[str, Any] = {}
    sub_working = result.get("working_flow")
    if sub_working:
        state_updates["working_flow"] = sub_working
        state_updates["flow_dirty"] = True

    return ToolResult.ok(
        f"[sub-agent {spec.name}] {final_text[:500] or '(no final text)'}",
        data={"subagent": spec.name, "final_text": final_text, "messages": len(messages)},
        state_updates=state_updates,
    )


def build_task_tools() -> list[Tool]:
    return [task_run]
