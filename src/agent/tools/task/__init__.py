"""task.run — sub-agent delegation (Claude Code'dagi Agent tool kabi).

Main agent — dispatcher: faqat plan.*, project.*, search.docs, task.run ga ega.
Domen ishi sub-agent'ga uzatiladi:
  - flow-builder      — flow.*
  - crm-manager       — crm.*, chat.*, broadcast.*
  - cms-manager       — cms.*
  - deploy-manager    — env.*, release.*, plans.*, subscription.*, payment.*, file.*

Sub-agent alohida thread'da ishlaydi. Asosiy state'ga faqat **qisqa xulosa matn**
qaytadi — sub-agent'ning o'nlab oraliq tool call'lari main history'ni bosib
ketmaydi. Dashboard UI sub-agent qadamlarini accordion'da ko'rsata oladi
(`sub_agent_started`/`sub_agent_finished` event'lari orqali).
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from ...prompts.subagent_specs import REGISTRY, list_subagents, tool_allowed
from ..base import Tool, ToolContext, ToolResult, tool


class TaskRunInput(BaseModel):
    subagent_type: str = Field(
        description=(
            "Sub-agent name: flow-builder | crm-manager | cms-manager | deploy-manager"
        )
    )
    prompt: str = Field(
        description=(
            "Vazifa tavsifi (1-3 jumla + kerakli kontekst). Misol: 'Loyiha 42 da "
            "/start trigger qo'sh, salom xabari ber, va flow.save. project_id=42.'"
        )
    )
    propagate_flow: bool = Field(
        default=True,
        description="working_flow ni sub-agent'ga uzatish (flow-builder uchun zarur).",
    )


@tool(
    name="task.run",
    description=(
        "Bir domen ishini specialized sub-agent'ga uzatish. Sub-agent o'z tool "
        "to'plamiga ega va alohida thread'da ishlaydi. Oxirida qisqa xulosa "
        "matn qaytaradi (oraliq tool call'lar main history'ga kirmaydi).\n\n"
        "Sub-agentlar:\n"
        "- flow-builder: visual bot flow qurish/o'zgartirish (flow.* tool'lari)\n"
        "- crm-manager: foydalanuvchi, segment, chat, broadcast (crm.*, chat.*, broadcast.*)\n"
        "- cms-manager: MongoDB collection schema va document CRUD (cms.*)\n"
        "- deploy-manager: env, release, deploy, tarif, to'lov, loglar (env.*, release.*, ...)\n\n"
        "Promptda kontekstni aniq ber: project_id, qaysi node'lar, qaysi segment va h.k. "
        "Sub-agent main suhbat tarixini ko'rmaydi."
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
    from ...llm import ModelRoles, build_client
    from ...config import get_settings
    from ...tools import build_default_registry
    from ...tools.registry import ToolRegistry

    s = get_settings()
    full = build_default_registry()
    sub_registry = ToolRegistry()
    for t in full.all():
        if tool_allowed(t.name, spec.allowed_tools):
            sub_registry.register(t)

    base_roles = ModelRoles.from_settings(s)
    hint_model = {
        "planner": base_roles.planner,
        "executor": base_roles.executor,
        "critic": base_roles.critic,
    }.get(spec.model_hint, base_roles.executor)
    sub_roles = ModelRoles(
        chat=hint_model,
        planner=hint_model,
        executor=hint_model,
        critic=hint_model,
        embedding=base_roles.embedding,
    )

    # Sub-agent unique id — UI accordion uchun. Sub-agent emit qilgan har bir
    # event payload'iga `sub_agent_id` qo'shamiz ki dashboard nested ko'rsata olsin.
    sub_id = f"sa-{spec.name}-{uuid.uuid4().hex[:8]}"
    parent_emit = ctx.emit

    tool_call_count = 0

    def _wrap_emit(name: str, payload: dict[str, Any]):
        """Sub-agent event'larini sub_agent_id bilan markirovka qilamiz."""
        nonlocal tool_call_count
        if name == "tool_call":
            tool_call_count += 1
        if isinstance(payload, dict):
            payload = {**payload, "sub_agent_id": sub_id, "sub_agent_name": spec.name}
        if parent_emit is None:
            return None
        return parent_emit(name, payload)

    # Sub-agent boshlanishini bildirish
    if parent_emit is not None:
        parent_emit(
            "sub_agent_started",
            {
                "sub_agent_id": sub_id,
                "name": spec.name,
                "description": spec.description,
                "prompt": input.prompt,
            },
        )

    cfg = GraphConfig(
        llm=build_client(s.llm_provider, s),
        registry=sub_registry,
        models=sub_roles,
        max_tokens=s.max_output_tokens,
        emit=_wrap_emit,
        services=ctx.services,
    )
    graph = build_graph(cfg)
    seeded_prompt = f"[Sub-agent: {spec.name}]\n{spec.full_prompt}\n\nTask: {input.prompt}"

    thread_id = f"{ctx.run_id}-{spec.name}-{sub_id}"
    base = initial_state(
        run_id=f"{ctx.run_id}-sub-{spec.name}",
        project_id=ctx.project_id,
        user_id=ctx.user_id,
        thread_id=thread_id,
        user_prompt=seeded_prompt,
        baseline_flow=ctx.state.get("baseline_flow") if input.propagate_flow else None,
    )
    if input.propagate_flow and ctx.state.get("working_flow"):
        base["working_flow"] = ctx.state["working_flow"]
    # Forward user auth so sub-agent backend calls don't fail with 401.
    auth_claims = ctx.state.get("auth_claims")
    if auth_claims:
        base["auth_claims"] = auth_claims  # type: ignore[typeddict-unknown-key]

    try:
        result = await graph.ainvoke(
            base, config={"configurable": {"thread_id": thread_id}}
        )
    except Exception as e:  # noqa: BLE001
        if parent_emit is not None:
            parent_emit(
                "sub_agent_finished",
                {
                    "sub_agent_id": sub_id,
                    "name": spec.name,
                    "ok": False,
                    "tool_calls": tool_call_count,
                    "summary": f"Sub-agent xatoligi: {e}",
                },
            )
        return ToolResult.error(f"[sub-agent {spec.name}] failed: {e}")

    messages = result.get("messages") or []
    final_text = ""
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            final_text = m.text
            break
    final_text = (final_text or "(no final text)").strip()

    # Flow o'zgargan bo'lsa — working_flow'ni qaytarish (UI panel uchun)
    state_updates: dict[str, Any] = {}
    sub_working = result.get("working_flow")
    if sub_working:
        state_updates["working_flow"] = sub_working
        state_updates["flow_dirty"] = bool(result.get("flow_dirty"))

    if parent_emit is not None:
        parent_emit(
            "sub_agent_finished",
            {
                "sub_agent_id": sub_id,
                "name": spec.name,
                "ok": True,
                "tool_calls": tool_call_count,
                "summary": final_text[:500],
            },
        )

    # Main agent uchun MINIMAL content — qisqa xulosa.
    # Oraliq tool call'lar main history'ga kirmaydi (ular faqat sub_agent_step
    # event'lari sifatida UI'da ko'rsatildi).
    return ToolResult.ok(
        f"[{spec.name}] {final_text[:1000]}",
        data={
            "subagent": spec.name,
            "sub_agent_id": sub_id,
            "tool_calls": tool_call_count,
        },
        state_updates=state_updates,
    )


def build_task_tools() -> list[Tool]:
    return [task_run]
