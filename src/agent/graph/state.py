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

    # ── planner ─────────────────────────────────────────────────────────
    current_plan: list[dict[str, Any]] | None

    # ── interrupts ──────────────────────────────────────────────────────
    pending_approval: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None

    # ── budget snapshot (mirrors runtime.Budget) ────────────────────────
    budget: dict[str, Any] | None

    # ── token accounting ────────────────────────────────────────────────
    total_input_tokens: int
    total_output_tokens: int
