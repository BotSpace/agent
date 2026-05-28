"""LangGraph orchestrator — plan/act/observe/reflect loop with approval gating.

Graph topology:

    START → llm_call
    llm_call → (tool_use?) tool_exec | (incomplete plan?) plan_reminder | final
    tool_exec → (approval needed?) approval_interrupt | (clarification?) clarify_interrupt | llm_call
    approval_interrupt → llm_call
    clarify_interrupt → llm_call
    plan_reminder → llm_call
    final → END
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

try:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[misc]

from ..llm import GenerateOptions, LLMClient, Message, ModelRoles, TextPart
from ..prompts import build_system_prompt
from ..tools import ToolContext, ToolRegistry, has_incomplete_plan
from .state import AgentState

EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None] | None]


@dataclass
class GraphConfig:
    llm: LLMClient
    registry: ToolRegistry
    models: ModelRoles
    max_tokens: int = 4096
    temperature: float = 0.7
    thinking_budget: int | None = None
    emit: EventEmitter | None = None
    services: dict[str, Any] | None = None
    approve_destructive: bool = True


async def _emit(emit: EventEmitter | None, name: str, payload: dict[str, Any]) -> None:
    if emit is None:
        return
    res = emit(name, payload)
    if hasattr(res, "__await__"):
        await res  # type: ignore[func-returns-value]


def _pick_model(state: AgentState, models: ModelRoles) -> str:
    """Main agent har doim `planner` (kuchli) modelni ishlatadi.

    Main agent — dispatcher: har bir qadam muhim (qaysi sub-agent, qanday
    prompt, project.list/get ni darhol chaqirish, va h.k.). Arzon model
    (executor=gpt-5.4-mini) bu rolda "men buni qilaman..." deb gap qilib,
    aslida tool chaqirmasdi — foydalanuvchi natija ko'rmasdi.

    Sub-agentlar `task.run` ichida `ModelRoles(planner=executor=critic=hint_model)`
    qilib collapse qilingani uchun, ular o'z hint'idagi (odatda arzon executor)
    modelni ishlatishda davom etadi.
    """
    return models.planner


def build_graph(cfg: GraphConfig, checkpointer: Any | None = None):
    # ── nodes ───────────────────────────────────────────────────────────

    async def llm_call(state: AgentState) -> dict[str, Any]:
        system = Message.system_text(build_system_prompt(state))
        history = state.get("messages") or []
        model = _pick_model(state, cfg.models)
        options = GenerateOptions(
            model=model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            tools=cfg.registry.specs(),
            thinking_budget=cfg.thinking_budget,
        )
        result = await cfg.llm.generate([system, *history], options)
        await _emit(
            cfg.emit,
            "assistant_message",
            {"text": result.message.text, "model": model, "usage": result.usage.__dict__},
        )
        return {
            "messages": [result.message],
            "total_input_tokens": state.get("total_input_tokens", 0) + (result.usage.input_tokens or 0),
            "total_output_tokens": state.get("total_output_tokens", 0) + (result.usage.output_tokens or 0),
        }

    async def tool_exec(state: AgentState) -> dict[str, Any]:
        history = state.get("messages") or []
        if not history:
            return {}
        last = history[-1]
        if not isinstance(last, Message) or not last.tool_calls:
            return {}

        updates: dict[str, Any] = {}
        tool_messages: list[Message] = []
        pending_approval: dict[str, Any] | None = None
        pending_clarification: dict[str, Any] | None = None
        current_state: dict[str, Any] = dict(state)

        for call in last.tool_calls:
            tool = cfg.registry.get(call.name)
            await _emit(
                cfg.emit,
                "tool_call",
                {"name": call.name, "id": call.id, "arguments": call.arguments},
            )

            # ── approval gate for destructive tools ───────────────────
            # IMPORTANT: emit a placeholder tool_result for every tool_use
            # in the turn even when blocking on approval. Bedrock (and
            # Anthropic) require each tool_use to be answered with a
            # corresponding tool_result in the next user message. After the
            # user approves, the model will re-issue the tool_use call.
            if (
                tool is not None
                and tool.destructive
                and cfg.approve_destructive
                and not current_state.get(f"approved_{call.id}")
            ):
                pending_approval = {
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "arguments": call.arguments,
                    "reason": f"Destructive operation `{call.name}` requires approval.",
                }
                await _emit(cfg.emit, "approval_required", pending_approval)
                placeholder = (
                    f"Awaiting user approval for `{call.name}`. "
                    "Once approved, re-issue this tool call to proceed."
                )
                await _emit(
                    cfg.emit,
                    "tool_result",
                    {
                        "tool_call_id": call.id,
                        "name": call.name,
                        "is_error": False,
                        "content": placeholder,
                        "data": None,
                    },
                )
                tool_messages.append(
                    Message.tool_result(tool_use_id=call.id, content=placeholder)
                )
                continue

            # ToolContext.emit — async wrapper. Tool'lar `await ctx.emit(...)`
            # qilib o'z event'larini chiqarishi mumkin (task.run sub_agent_started
            # / sub_agent_finished uchun ishlatadi). Sync lambda ishlatish
            # noto'g'ri edi — cfg.emit async bo'lganida coroutine yo'qotilardi.
            async def _ctx_emit(name: str, payload: dict[str, Any]) -> None:
                await _emit(cfg.emit, name, payload)

            ctx = ToolContext(
                run_id=state.get("run_id") or "",
                project_id=state.get("project_id"),
                user_id=state.get("user_id"),
                state=current_state,
                emit=_ctx_emit,
                services=cfg.services or {},
            )
            result = await cfg.registry.execute(call.name, call.arguments, ctx)
            if isinstance(result.data, dict):
                input_tokens = int(result.data.get("input_tokens") or 0)
                output_tokens = int(result.data.get("output_tokens") or 0)
                if input_tokens or output_tokens:
                    current_state["total_input_tokens"] = current_state.get("total_input_tokens", 0) + input_tokens
                    current_state["total_output_tokens"] = current_state.get("total_output_tokens", 0) + output_tokens
                    updates["total_input_tokens"] = current_state["total_input_tokens"]
                    updates["total_output_tokens"] = current_state["total_output_tokens"]
            await _emit(
                cfg.emit,
                "tool_result",
                {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "is_error": result.is_error,
                    "content": result.content,
                    "data": result.data,
                },
            )
            if result.interrupt:
                pending_clarification = {**result.interrupt, "tool_call_id": call.id}
            tool_messages.append(
                Message.tool_result(
                    tool_use_id=call.id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )
            for k, v in (result.state_updates or {}).items():
                current_state[k] = v
                updates[k] = v

        if tool_messages:
            # Bedrock (and Anthropic) want all tool_result blocks for a turn
            # in a single user message — merge per-call messages here.
            merged_parts: list[Any] = []
            for tm in tool_messages:
                merged_parts.extend(tm.content)
            updates["messages"] = [Message(role="tool", content=merged_parts)]
        if pending_approval is not None:
            updates["pending_approval"] = pending_approval
        if pending_clarification is not None:
            updates["pending_clarification"] = pending_clarification
        return updates

    async def approval_interrupt(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("pending_approval") or {})
        decision = interrupt({"kind": "approval", **payload})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        tool_name = payload.get("tool_name", "operation")
        tool_id = payload.get("tool_call_id")
        await _emit(
            cfg.emit,
            "approval_resolved",
            {"tool_call_id": tool_id, "approved": approved},
        )
        # A placeholder tool_result for this call has already been emitted in
        # `tool_exec`, so we don't owe Bedrock another tool_result here.
        # Instead we add a plain user-facing instruction telling the model
        # what to do next.
        if approved:
            note = (
                f"User approved `{tool_name}` (id={tool_id}). "
                f"You may now re-issue the same tool call to perform it."
            )
        else:
            note = (
                f"User rejected `{tool_name}` (id={tool_id}). "
                f"Do not retry this operation; consider alternatives or ask the user."
            )
        return {
            f"approved_{tool_id}": approved,
            "pending_approval": None,
            "messages": [Message.user_text(f"[system] {note}")],
        }

    async def clarify_interrupt(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("pending_clarification") or {})
        answer = interrupt({"kind": "clarify", **payload})
        text = str(answer) if not isinstance(answer, dict) else str(answer.get("answer", answer))
        await _emit(cfg.emit, "clarification_resolved", {"tool_call_id": payload.get("tool_call_id"), "answer": text})
        return {
            "pending_clarification": None,
            "messages": [
                Message.tool_result(
                    tool_use_id=str(payload.get("tool_call_id")),
                    content=f"User clarification: {text}",
                )
            ],
        }

    async def plan_reminder(_: AgentState) -> dict[str, Any]:
        reminder = Message(
            role="user",
            content=[
                TextPart(
                    text=(
                        "[system reminder] The current plan still has incomplete items. "
                        "Do not give a final answer yet. Continue with the next pending/in_progress "
                        "item via tools and mark it completed before responding."
                    )
                )
            ],
        )
        return {"messages": [reminder]}

    async def final_response(_: AgentState) -> dict[str, Any]:
        return {}

    # ── routing ────────────────────────────────────────────────────────

    def after_llm(state: AgentState) -> Literal["tool_exec", "plan_reminder", "final"]:
        msgs = state.get("messages") or []
        if msgs and isinstance(msgs[-1], Message) and msgs[-1].tool_calls:
            return "tool_exec"
        if has_incomplete_plan(state.get("current_plan")):
            return "plan_reminder"
        return "final"

    def after_tools(state: AgentState) -> Literal["approval", "clarify", "llm"]:
        if state.get("pending_approval"):
            return "approval"
        if state.get("pending_clarification"):
            return "clarify"
        return "llm"

    # ── assemble ───────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_exec", tool_exec)
    builder.add_node("approval_interrupt", approval_interrupt)
    builder.add_node("clarify_interrupt", clarify_interrupt)
    builder.add_node("plan_reminder", plan_reminder)
    builder.add_node("final", final_response)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        after_llm,
        {
            "tool_exec": "tool_exec",
            "plan_reminder": "plan_reminder",
            "final": "final",
        },
    )
    builder.add_conditional_edges(
        "tool_exec",
        after_tools,
        {
            "approval": "approval_interrupt",
            "clarify": "clarify_interrupt",
            "llm": "llm_call",
        },
    )
    builder.add_edge("approval_interrupt", "llm_call")
    builder.add_edge("clarify_interrupt", "llm_call")
    builder.add_edge("plan_reminder", "llm_call")
    builder.add_edge("final", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(
    *,
    run_id: str,
    project_id: str | None,
    user_id: str | None,
    thread_id: str,
    user_prompt: str,
    baseline_flow: dict[str, Any] | None = None,
) -> AgentState:
    state: AgentState = {
        "run_id": run_id,
        "project_id": project_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [Message.user_text(user_prompt)],
    }
    if baseline_flow:
        state["baseline_flow"] = baseline_flow
        state["working_flow"] = baseline_flow
    return state
