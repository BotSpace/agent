"""LangGraph AgentState — message history bilan birga loyihaga oid kontekst."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..llm.base import Message


class AgentState(TypedDict, total=False):
    # ── conversation ────────────────────────────────────────────────────
    messages: Annotated[list[Message], operator.add]
    """Sistema bilan birinchi user message, keyin assistant/tool/user history."""

    # ── identity & context ──────────────────────────────────────────────
    run_id: str
    project_id: str | None
    user_id: str | None
    thread_id: str
    auth_claims: dict[str, Any] | None

    # ── flow domain ─────────────────────────────────────────────────────
    baseline_flow: dict[str, Any] | None
    working_flow: dict[str, Any] | None
    flow_dirty: bool
    pending_save: bool
    last_validation: dict[str, Any] | None

    # ── cms awareness ───────────────────────────────────────────────────
    collections: list[dict[str, Any]] | None
    """Project CMS collections + field names: [{"name", "fields":[...]}].
    Seeded at run start so the flow-builder can reference them correctly."""

    # ── planner ─────────────────────────────────────────────────────────
    current_plan: list[dict[str, Any]] | None

    # ── interrupts ──────────────────────────────────────────────────────
    pending_approval: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None

    approvals: Annotated[dict[str, bool], operator.or_]
    """Consume-once approval flags keyed by a stable tool signature
    (name + sorted arguments), NOT by tool_use id. Declared as a real channel
    so it survives the interrupt/resume cycle on the Postgres checkpointer.
    After the user approves, the model re-issues the call with a *new*
    tool_use id, so an id-based key would never match — the signature does.
    The flag is reset to False once consumed so a later identical call
    re-gates and asks for approval again."""

    # ── budget snapshot (mirrors runtime.Budget) ────────────────────────
    budget: dict[str, Any] | None

    # ── token accounting ────────────────────────────────────────────────
    total_input_tokens: int
    total_output_tokens: int

    # ── tool-call budget ────────────────────────────────────────────────
    tool_calls_used: int
    """Run davomida bajarilgan tool chaqiruvlari soni. cfg.max_tool_calls ga
    yetganda graph yangi tool bajarmaydi — modelga yakuniy javob berishni
    aytadi. recursion_limit (500) faqat oxirgi to'siq; bu byudjet xarajat va
    cheksiz sikllarni ancha oldin to'xtatadi."""
