"""Plan tools — TodoWrite ekvivalenti.

Agent har task uchun status saqlaydi (pending/in_progress/completed/blocked).
Graph har turn-da incomplete plan bo'lsa, modelga reminder yuboradi.
"""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult, tool

PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]


class PlanItemModel(BaseModel):
    id: str = Field(description="Unique item id (auto-generated if omitted)", default_factory=lambda: uuid4().hex[:8])
    title: str = Field(description="Short imperative task title (3-8 words) — the step label")
    detail: str | None = Field(
        default=None,
        description=(
            "The ACTUAL plan for this step (1-3 sentences) — WHAT will be built and "
            "HOW: the concrete nodes/tools/flow, files or collections involved, key "
            "decisions. This is the substance a reader needs to understand the step "
            "without you; not a restatement of the title. Always fill it."
        ),
    )
    status: PlanStatus = "pending"
    notes: str | None = Field(default=None, description="Optional short status/context note added later")


class PlanSetInput(BaseModel):
    items: list[PlanItemModel] = Field(description="The full plan — replaces any existing plan")


class PlanUpdateInput(BaseModel):
    id: str = Field(description="Item id to update")
    status: PlanStatus | None = None
    title: str | None = None
    detail: str | None = None
    notes: str | None = None


class PlanCompleteInput(BaseModel):
    id: str


class PlanAdvanceInput(BaseModel):
    complete: list[str] = Field(
        default_factory=list,
        description="Item id'lar — `completed` ga o'tkaziladi (odatda hozirgi qadam).",
    )
    start: list[str] = Field(
        default_factory=list,
        description="Item id'lar — `in_progress` ga o'tkaziladi (odatda keyingi qadam).",
    )


def _get_plan(ctx: ToolContext) -> list[dict[str, Any]]:
    return list(ctx.state.get("current_plan") or [])


def _resolve_id(plan: list[dict[str, Any]], given: str) -> str | None:
    """Map a referenced id to a real plan item id. Exact match first; otherwise
    accept a POSITIONAL reference like `plan_0` / `plan-2` / `2` (0-based) — the
    model routinely invents these instead of the stored ids."""
    ids = [str(item.get("id")) for item in plan]
    if given in ids:
        return given
    m = re.match(r"^(?:plan[_-]?)?(\d+)$", str(given).strip(), re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(plan):
            return ids[idx]
    return None


def _set_plan(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"current_plan": items}


async def _emit_plan(ctx: ToolContext, event: str, plan: list[dict[str, Any]]) -> None:
    """Emit a plan_set/plan_update event so the plan is persisted to agent_events
    (→ chat history) and streamed to the UI as an inline plan turn. Best-effort."""
    emit = ctx.emit
    if emit is None:
        return
    try:
        res = emit(event, {"plan": plan})
        if hasattr(res, "__await__"):
            await res
    except Exception:
        pass


@tool(
    name="plan.set",
    description=(
        "Write the task plan up front — a REAL, detailed plan, not a bare checklist. "
        "Break the goal into 3-8 steps; each step has a short `title` AND a `detail` "
        "(1-3 sentences) spelling out exactly what will be built and how — the concrete "
        "nodes/tools/flow, files or collections, and key decisions — so a reader "
        "understands the whole approach from the plan alone. REPLACES the entire plan, "
        "so call it once at the start; edit single items with plan.update afterwards "
        "(calling plan.set again wipes the rest)."
    ),
    input_model=PlanSetInput,
)
async def plan_set(input: PlanSetInput, ctx: ToolContext) -> ToolResult:
    # Assign predictable positional ids (plan_0, plan_1, …) so later
    # plan.update/advance/complete calls can reference a step by position — which
    # the model does naturally — instead of guessing a random hex id.
    items = []
    for i, item in enumerate(input.items):
        d = item.model_dump()
        d["id"] = f"plan_{i}"
        items.append(d)
    await _emit_plan(ctx, "plan_set", items)
    return ToolResult.ok(
        f"Plan set with {len(items)} item(s).",
        state_updates=_set_plan(items),
        data={"plan": items},
    )


@tool(
    name="plan.update",
    description=(
        "Update a single plan item — change its status (pending/in_progress/completed/blocked) "
        "or refine the title/notes. Mark an item in_progress before starting work and completed "
        "as soon as it's done (don't batch)."
    ),
    input_model=PlanUpdateInput,
)
async def plan_update(input: PlanUpdateInput, ctx: ToolContext) -> ToolResult:
    plan = _get_plan(ctx)
    target = _resolve_id(plan, input.id)
    for item in plan:
        if item.get("id") == target:
            if input.status is not None:
                item["status"] = input.status
            if input.title is not None:
                item["title"] = input.title
            if input.detail is not None:
                item["detail"] = input.detail
            if input.notes is not None:
                item["notes"] = input.notes
            await _emit_plan(ctx, "plan_update", plan)
            return ToolResult.ok(
                f"Updated item `{input.id}` → {item.get('status')}",
                state_updates=_set_plan(plan),
                data={"plan": plan},
            )
    return ToolResult.error(f"Plan item `{input.id}` not found")


@tool(
    name="plan.complete",
    description="Mark a plan item as completed. Shortcut for plan.update with status=completed.",
    input_model=PlanCompleteInput,
)
async def plan_complete(input: PlanCompleteInput, ctx: ToolContext) -> ToolResult:
    plan = _get_plan(ctx)
    target = _resolve_id(plan, input.id)
    for item in plan:
        if item.get("id") == target:
            item["status"] = "completed"
            await _emit_plan(ctx, "plan_update", plan)
            return ToolResult.ok(
                f"Completed `{input.id}`",
                state_updates=_set_plan(plan),
                data={"plan": plan},
            )
    return ToolResult.error(f"Plan item `{input.id}` not found")


@tool(
    name="plan.advance",
    description=(
        "Bitta chaqiruvda bir nechta plan item'ning statusini o'zgartiradi: "
        "`complete` ro'yxatidagilar `completed` ga, `start` ro'yxatidagilar "
        "`in_progress` ga o'tadi. Eski qadamni tugatib darhol yangisini "
        "boshlash uchun ideal — alohida plan.complete + plan.update ikkita "
        "chaqiruv qilish o'rniga."
    ),
    input_model=PlanAdvanceInput,
)
async def plan_advance(input: PlanAdvanceInput, ctx: ToolContext) -> ToolResult:
    plan = _get_plan(ctx)
    by_id = {item.get("id"): item for item in plan}

    not_found: list[str] = []
    completed: list[str] = []
    started: list[str] = []

    for cid in input.complete:
        item = by_id.get(_resolve_id(plan, cid))
        if item is None:
            not_found.append(cid)
            continue
        item["status"] = "completed"
        completed.append(cid)

    for sid in input.start:
        item = by_id.get(_resolve_id(plan, sid))
        if item is None:
            not_found.append(sid)
            continue
        item["status"] = "in_progress"
        started.append(sid)

    if not completed and not started:
        return ToolResult.error(
            f"Hech qanday item topilmadi/o'zgartirilmadi. Yo'qotilgan id'lar: {not_found}"
        )

    parts: list[str] = []
    if completed:
        parts.append(f"completed: {completed}")
    if started:
        parts.append(f"in_progress: {started}")
    msg = " · ".join(parts)
    if not_found:
        msg += f" (topilmadi: {not_found})"

    await _emit_plan(ctx, "plan_update", plan)
    return ToolResult.ok(
        msg,
        state_updates=_set_plan(plan),
        data={"plan": plan, "completed": completed, "started": started, "not_found": not_found},
    )


def has_incomplete_plan(plan: list[dict[str, Any]] | None) -> bool:
    if not plan:
        return False
    return any(item.get("status") in ("pending", "in_progress") for item in plan)


def build_plan_tools() -> list[Tool]:
    return [plan_set, plan_update, plan_complete, plan_advance]
