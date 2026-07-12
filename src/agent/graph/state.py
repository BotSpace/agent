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

    module_nodes: list[dict[str, Any]] | None
    """Per-project ENABLED external module node types (dynamic catalog), e.g.
    [{"type": "openrouter.Chat", "module": "openrouter", "is_trigger": False,
    "required_fields": [...], "credential_fields": [...]}]. Seeded at run start so
    flow.add_node / the validator accept `moduleId.NodeName` types. None/empty =
    no modules enabled."""

    # ── planner ─────────────────────────────────────────────────────────
    current_plan: list[dict[str, Any]] | None

    # ── sub-agent session continuity ────────────────────────────────────
    subagent_sessions: dict[str, str]
    """Last session_id per "{project_id}:{subagent_type}". task.run auto-resumes
    the tracked session when the model omits session_id, so a forgotten session
    no longer spawns a fresh amnesiac sub-agent (which re-plans and re-seeds —
    duplicate demo data + plan thrashing). Replace-on-write (no reducer); the
    tool always merges the full map before returning it."""

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
    # Cached prompt tokens (subset of input) — reported separately so the
    # backend can discount them against the user's quota per the plan rate.
    total_cache_read_tokens: int

    # ── mid-run token-limit enforcement ─────────────────────────────────
    pre_run_tokens_remaining: int | None
    """User's REMAINING monthly token budget snapshotted at run START
    (ai_token_usage.remaining). The pre-run usage is already subtracted, so a
    run is over budget when its OWN tokens (total_input+total_output) reach this
    number. None = no enforcement (fetch failed → fail open, never block)."""

    limit_exceeded: bool
    """Set True by llm_call when this run's own tokens reach
    pre_run_tokens_remaining. Routes the graph to the `limit_exceeded` node which
    appends a user-facing message, then ends as a NORMAL completion."""

    # ── lazy tools ──────────────────────────────────────────────────────
    activated_tools: list[str]
    """Lazy tool nomlari — `tools.search` orqali yuklangan. llm_call bu tool'larni
    CORE tool'lar bilan birga yuboradi (lazy_tools=True bo'lsa)."""

    # ── tool-call budget ────────────────────────────────────────────────
    tool_calls_used: int
    """Run davomida bajarilgan tool chaqiruvlari soni. cfg.max_tool_calls ga
    yetganda graph yangi tool bajarmaydi — modelga yakuniy javob berishni
    aytadi. recursion_limit (500) faqat oxirgi to'siq; bu byudjet xarajat va
    cheksiz sikllarni ancha oldin to'xtatadi."""
