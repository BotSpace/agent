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

## Persistent, session-resumable sub-agents
`session_id` berilsa, sub-agent o'sha sessiyaga BOG'LANGAN turg'un thread'da
ishlaydi: agar thread allaqachon checkpoint'langan bo'lsa, eski history
qayta seed qilinmaydi — faqat yangi vazifa user message sifatida qo'shiladi
(LangGraph add_messages merge qiladi). Bu sub-agentni "qoldirgan joyidan"
davom ettiradi. `session_id` berilmasa, yangi sessiya yaratiladi va
natijada qaytariladi — main agent uni keyingi task.run'da qayta ishlatishi
mumkin.

## Nested delegation
Sub-agent o'z domenidan tashqari ish uchun yana `task.run` chaqirishi mumkin.
GUARD'lar: sub-agent O'ZINI chaqira olmaydi (subagent_type == joriy) va
nesting chuqurligi `MAX_SUB_AGENT_DEPTH` bilan cheklangan (runaway nesting +
token portlashidan himoya).
"""

from __future__ import annotations

import inspect
import logging
import traceback
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ...prompts.subagent_specs import REGISTRY, list_subagents, tool_allowed
from ..base import Tool, ToolContext, ToolResult, tool

logger = logging.getLogger(__name__)

# Maksimal nesting chuqurligi: main agent (depth 0) → sub-agent (depth 1) →
# nested sub-agent (depth 2). Depth 2'dagi sub-agent boshqa sub-agent chaqira
# olmaydi. Token portlashi va cheksiz rekursiyadan himoya.
MAX_SUB_AGENT_DEPTH = 2


class TaskRunInput(BaseModel):
    subagent_type: str = Field(
        description="flow-builder | crm-manager | cms-manager | deploy-manager"
    )
    prompt: str = Field(
        description="Task + needed context (project_id, ids, ...). Sub-agent has no main history."
    )
    propagate_flow: bool = Field(
        default=True,
        description="Pass working_flow to sub-agent (flow-builder).",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Resume a prior sub-agent conversation: pass the SAME session_id "
            "returned by an earlier task.run to continue where that sub-agent "
            "left off (its history is preserved). Omit to start a fresh session "
            "— a new session_id is generated and returned in the result."
        ),
    )


async def _checkpointer_has_state(checkpointer: Any, config: dict[str, Any]) -> bool:
    """Thread allaqachon checkpoint'langanmi? (resume-vs-fresh detection)."""
    if checkpointer is None:
        return False
    try:
        snapshot = await checkpointer.aget(config)
    except Exception:  # noqa: BLE001
        return False
    return snapshot is not None


@tool(
    name="task.run",
    description=(
        "Delegate a domain task to a specialized sub-agent. Returns a short "
        "summary plus a `session_id`; intermediate tool calls stay out of main "
        "history. Pass the SAME session_id back to continue that sub-agent where "
        "it left off (persisted history); omit it to start fresh. "
        "flow-builder=flow.*, crm-manager=crm/chat/broadcast, cms-manager=cms, "
        "deploy-manager=env/release/subscription/file."
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

    services = ctx.services or {}

    # ── nesting guards ──────────────────────────────────────────────────
    # Self-call: bir sub-agent o'zini qayta chaqira olmaydi (foydasiz loop).
    current_subagent = services.get("current_subagent_type")
    if current_subagent and current_subagent == input.subagent_type:
        return ToolResult.error(
            f"Sub-agent `{input.subagent_type}` cannot delegate to ITSELF. "
            "Do the work directly with your own tools, or pick a different sub-agent."
        )
    # Depth cap: main(0) → sub(1) → nested(2). depth == MAX'da yangi spawn yo'q.
    current_depth = int(services.get("sub_agent_depth", 0) or 0)
    if current_depth >= MAX_SUB_AGENT_DEPTH:
        return ToolResult.error(
            f"Sub-agent nesting depth cap reached (max {MAX_SUB_AGENT_DEPTH}). "
            "Cannot spawn another nested sub-agent — complete this task directly."
        )

    # Lazy import to avoid circular dep at module load time.
    from ...graph import GraphConfig, build_graph, initial_state
    from ...llm import Message, ModelRoles, build_client
    from ...config import get_settings
    from ...memory import close_async_checkpointer, create_async_checkpointer
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
        planner=hint_model,
        executor=hint_model,
        critic=hint_model,
        embedding=base_roles.embedding,
    )

    # ── session id + stable sub-agent thread id ─────────────────────────
    # session_id resume kaliti: bir xil (conversation, subagent_type, session_id)
    # uchun thread_id IDENTIK bo'lishi shart — shunda checkpointer o'sha
    # thread'ni davom ettiradi. Eng turg'un main id — main agent'ning
    # `thread_id`'si (run_id har main turn'da o'zgaradi; thread_id butun
    # suhbat davomida bir xil qoladi). Sub-agent state ham `thread_id` saqlaydi,
    # shuning uchun nested chaqiruvda ham o'sha turg'un asosga bog'lanamiz.
    session_id = input.session_id or uuid.uuid4().hex[:12]
    main_thread = ctx.state.get("thread_id") or ctx.run_id or "main"
    sub_thread_id = f"{main_thread}:sub:{spec.name}:{session_id}"

    # Sub-agent unique id — UI accordion uchun. Sub-agent emit qilgan har bir
    # event payload'iga `sub_agent_id` qo'shamiz ki dashboard nested ko'rsata olsin.
    sub_id = f"sa-{spec.name}-{uuid.uuid4().hex[:8]}"
    parent_emit = ctx.emit
    parent_sub_agent_id = services.get("current_sub_agent_id")

    tool_call_count = 0

    async def _emit_to_parent(name: str, payload: dict[str, Any]) -> None:
        """Parent emit'ni xavfsiz await qilish — sync yoki async farqsiz."""
        if parent_emit is None:
            return
        result = parent_emit(name, payload)
        if inspect.isawaitable(result):
            await result

    async def _wrap_emit(name: str, payload: dict[str, Any]) -> None:
        """Sub-agent event'larini sub_agent_id bilan markirovka qilamiz."""
        nonlocal tool_call_count
        if name == "tool_call":
            tool_call_count += 1
        if isinstance(payload, dict):
            payload = {**payload, "sub_agent_id": sub_id, "sub_agent_name": spec.name}
        await _emit_to_parent(name, payload)

    # Sub-agent boshlanishini bildirish — dashboard SubAgentAccordion yaratadi.
    # Nested bo'lsa, parent_sub_agent_id linkage'i orqali UI nesting ko'rsatadi.
    await _emit_to_parent(
        "sub_agent_started",
        {
            "sub_agent_id": sub_id,
            "parent_sub_agent_id": parent_sub_agent_id,
            "name": spec.name,
            "description": spec.description,
            "prompt": input.prompt,
            "session_id": session_id,
            "depth": current_depth + 1,
        },
    )

    # ── checkpointer: worker-owned (reuse) yoki per-call (own + close) ───
    # Worker o'z checkpointer'ini services'ga qo'yadi — uni QAYTA ishlatamiz va
    # ASLO yopmaymiz (worker o'zi finally'da yopadi). Yo'q bo'lsa — bu chaqiruv
    # uchun yangi yaratamiz va oxirida xavfsiz yopamiz. Postgres bo'lmasa
    # (test/dev) create_async_checkpointer None qaytaradi → build_graph
    # MemorySaver'ga tushadi (resume bu holatda jarayon ichida ishlaydi).
    checkpointer = services.get("checkpointer")
    own_checkpointer = False
    if checkpointer is None:
        try:
            checkpointer = await create_async_checkpointer(s.database_url)
            own_checkpointer = checkpointer is not None
        except Exception as cp_err:  # noqa: BLE001
            logger.warning("Sub-agent checkpointer init failed, in-memory: %s", cp_err)
            checkpointer = None

    # Nested chaqiruvlar uchun depth/identity'ni oldinga uzatamiz: sub-agent
    # ichidagi task.run shu services'ni ctx orqali oladi.
    sub_services = {
        **services,
        "sub_agent_depth": current_depth + 1,
        "current_subagent_type": spec.name,
        "current_sub_agent_id": sub_id,
        # Reuse the SAME checkpointer downstream so nested sub-agents persist too.
        "checkpointer": checkpointer,
    }

    cfg = GraphConfig(
        llm=build_client(s.llm_provider, s),
        registry=sub_registry,
        models=sub_roles,
        max_tokens=s.max_output_tokens,
        emit=_wrap_emit,
        services=sub_services,
        # Sub-agent ichida destructive tool'lar (release.deploy, flow.save,
        # cms.delete, ...) approval-interrupt orqali to'xtatilmaydi: foydalanuvchi
        # main agent'ga vazifa berganida (masalan "deploy qil") allaqachon
        # niyatini tasdiqlagan. Approval bo'lsa sub-agent yarim yo'lda tugab
        # qoladi va main agent uchun "no final text" qaytaradi — bu noto'g'ri
        # "muvaffaqiyat" xulosaga olib keladi (release-985-4 bug case).
        approve_destructive=False,
    )
    graph = build_graph(cfg, checkpointer=checkpointer)

    async def _cleanup() -> None:
        """Per-call checkpointer'ni yopish. Worker-owned bo'lsa TEGMA."""
        if own_checkpointer and checkpointer is not None:
            try:
                await close_async_checkpointer(checkpointer)
            except Exception:  # noqa: BLE001
                pass

    invoke_config = {
        "configurable": {"thread_id": sub_thread_id},
        # LangGraph default = 25; flow-builder ko'pincha 30+ qadam bajaradi
        # (har node uchun add+connect+validate). Main agent bilan teng — 500.
        "recursion_limit": 500,
    }

    # ── resume-vs-fresh ─────────────────────────────────────────────────
    # Thread allaqachon checkpoint'langan bo'lsa (resume): system+spec'ni QAYTA
    # seed QILMAYMIZ — faqat yangi vazifani user message sifatida qo'shamiz,
    # add_messages reducer uni mavjud history ustiga merge qiladi. Yangi thread
    # bo'lsa (fresh): system/spec + task seed qilamiz.
    resuming = await _checkpointer_has_state(checkpointer, invoke_config)

    try:
        if resuming:
            base: dict[str, Any] = {
                "messages": [Message.user_text(f"Yangi vazifa: {input.prompt}")],
            }
            # working_flow'ni resume turida ham yangilaymiz (panel sinxron qolsin).
            if input.propagate_flow and ctx.state.get("working_flow"):
                base["working_flow"] = ctx.state["working_flow"]
            auth_claims = ctx.state.get("auth_claims")
            if auth_claims:
                base["auth_claims"] = auth_claims
        else:
            seeded_prompt = (
                f"[Sub-agent: {spec.name}]\n{spec.full_prompt}\n\nTask: {input.prompt}"
            )
            base = initial_state(
                run_id=f"{ctx.run_id}-sub-{spec.name}",
                project_id=ctx.project_id,
                user_id=ctx.user_id,
                thread_id=sub_thread_id,
                user_prompt=seeded_prompt,
                baseline_flow=(
                    ctx.state.get("baseline_flow") if input.propagate_flow else None
                ),
            )
            if input.propagate_flow and ctx.state.get("working_flow"):
                base["working_flow"] = ctx.state["working_flow"]
            # Forward user auth so sub-agent backend calls don't fail with 401.
            auth_claims = ctx.state.get("auth_claims")
            if auth_claims:
                base["auth_claims"] = auth_claims  # type: ignore[typeddict-unknown-key]

        result = await graph.ainvoke(base, config=invoke_config)
    except Exception as e:  # noqa: BLE001
        # To'liq traceback'ni log'ga yozamiz — UI'ga faqat qisqa xulosa boradi,
        # lekin debug uchun stack pod log'larida bo'lishi shart.
        tb = traceback.format_exc()
        logger.error(
            "Sub-agent %s crashed: %s\n%s",
            spec.name,
            type(e).__name__,
            tb,
        )
        await _emit_to_parent(
            "sub_agent_finished",
            {
                "sub_agent_id": sub_id,
                "name": spec.name,
                "session_id": session_id,
                "ok": False,
                "tool_calls": tool_call_count,
                "summary": f"Sub-agent xatoligi ({type(e).__name__}): {e}",
            },
        )
        await _cleanup()
        return ToolResult.error(
            f"[sub-agent {spec.name}] crashed: {type(e).__name__}: {e}",
            data={"subagent": spec.name, "sub_agent_id": sub_id, "session_id": session_id},
        )

    messages = result.get("messages") or []
    final_text = ""
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            final_text = m.text
            break
    final_text = (final_text or "").strip()
    # Agar sub-agent yakuniy assistant matnsiz tugagan bo'lsa — bu o'zining
    # vazifasini bajara olmagani belgisi (masalan max_tool_calls limit, yoki
    # ichki xato, yoki tool real bajarilmadi). Main agent uchun bu xato.
    if not final_text:
        await _emit_to_parent(
            "sub_agent_finished",
            {
                "sub_agent_id": sub_id,
                "name": spec.name,
                "session_id": session_id,
                "ok": False,
                "tool_calls": tool_call_count,
                "summary": "Sub-agent yakuniy javobsiz tugadi — vazifa to'liq bajarilmagan bo'lishi mumkin.",
            },
        )
        await _cleanup()
        return ToolResult.error(
            f"[{spec.name}] terminated without a final answer after {tool_call_count} tool call(s). "
            "Task may be incomplete — verify state before reporting success to the user.",
            data={"subagent": spec.name, "sub_agent_id": sub_id, "session_id": session_id},
        )

    # Flow o'zgargan bo'lsa — working_flow'ni qaytarish (UI panel uchun)
    state_updates: dict[str, Any] = {}
    sub_working = result.get("working_flow")
    if sub_working:
        state_updates["working_flow"] = sub_working
        state_updates["flow_dirty"] = bool(result.get("flow_dirty"))

    await _emit_to_parent(
        "sub_agent_finished",
        {
            "sub_agent_id": sub_id,
            "name": spec.name,
            "session_id": session_id,
            "ok": True,
            "tool_calls": tool_call_count,
            "summary": final_text[:500],
        },
    )

    await _cleanup()

    # Main agent uchun MINIMAL content — qisqa xulosa + session_id ko'rsatkichi,
    # shunda main agent xohlasa shu sub-agentni davom ettira oladi (resume).
    # Oraliq tool call'lar main history'ga kirmaydi (ular faqat sub_agent_step
    # event'lari sifatida UI'da ko'rsatildi).
    resumed_note = " (resumed)" if resuming else ""
    return ToolResult.ok(
        f"[{spec.name}]{resumed_note} {final_text[:1000]}\n"
        f"(session_id={session_id} — pass it back to task.run to continue this sub-agent)",
        data={
            "subagent": spec.name,
            "sub_agent_id": sub_id,
            "session_id": session_id,
            "sub_thread_id": sub_thread_id,
            "resumed": resuming,
            "tool_calls": tool_call_count,
        },
        state_updates=state_updates,
    )


def build_task_tools() -> list[Tool]:
    return [task_run]
