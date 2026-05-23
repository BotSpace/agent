"""Plan tools — TodoWrite ekvivalenti.

Agent har task uchun status saqlaydi (pending/in_progress/completed/blocked).
Graph har turn-da incomplete plan bo'lsa, modelga reminder yuboradi.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult, tool


PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]


class PlanItemModel(BaseModel):
    id: str = Field(description="Unique item id (auto-generated if omitted)", default_factory=lambda: uuid4().hex[:8])
    title: str = Field(description="Short imperative task title (3-8 words)")
    status: PlanStatus = "pending"
    notes: str | None = Field(default=None, description="Optional why/context")


class PlanSetInput(BaseModel):
    items: list[PlanItemModel] = Field(description="The full plan — replaces any existing plan")


class PlanUpdateInput(BaseModel):
    id: str = Field(description="Item id to update")
    status: PlanStatus | None = None
    title: str | None = None
    notes: str | None = None


class PlanCompleteInput(BaseModel):
    id: str


def _get_plan(ctx: ToolContext) -> list[dict[str, Any]]:
    return list(ctx.state.get("current_plan") or [])


def _set_plan(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"current_plan": items}


@tool(
    name="plan.set",
    description=(
        "Set or replace the current task plan. Use this at the start of any non-trivial work "
        "to break the goal into 3-8 concrete tasks. Update individual items via plan.update "
        "as work progresses."
    ),
    input_model=PlanSetInput,
)
async def plan_set(input: PlanSetInput, ctx: ToolContext) -> ToolResult:
    items = [item.model_dump() for item in input.items]
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
    for item in plan:
        if item.get("id") == input.id:
            if input.status is not None:
                item["status"] = input.status
            if input.title is not None:
                item["title"] = input.title
            if input.notes is not None:
                item["notes"] = input.notes
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
    for item in plan:
        if item.get("id") == input.id:
            item["status"] = "completed"
            return ToolResult.ok(
                f"Completed `{input.id}`",
                state_updates=_set_plan(plan),
                data={"plan": plan},
            )
    return ToolResult.error(f"Plan item `{input.id}` not found")


def has_incomplete_plan(plan: list[dict[str, Any]] | None) -> bool:
    if not plan:
        return False
    return any(item.get("status") in ("pending", "in_progress") for item in plan)


def build_plan_tools() -> list[Tool]:
    return [plan_set, plan_update, plan_complete]
