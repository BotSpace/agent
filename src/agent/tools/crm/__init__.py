"""crm.* tools — MongoDB project-scoped collections (CRM + custom data).

Backend har project'ga alohida DB yaratadi (`project_<id>`).
Built-in collections: `users`, `chat` + foydalanuvchi qo'shgan custom.

Bu tool'lar agent'ga "Vipga ega foydalanuvchilarni topib, broadcast yubor"
kabi savollarga javob berishga imkon beradi.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...integrations.mongo import MongoClient, get_mongo_client
from ..base import Tool, ToolContext, ToolResult, tool


def _mongo(ctx: ToolContext) -> MongoClient:
    mongo = ctx.services.get("mongo")
    if isinstance(mongo, MongoClient):
        return mongo
    return get_mongo_client()


def _project_id(ctx: ToolContext, override: str | int | None) -> str | int | None:
    return override or ctx.state.get("project_id")


# ── list_collections ─────────────────────────────────────────────────────────


class CrmListCollectionsInput(BaseModel):
    project_id: str | int | None = None


@tool(
    name="crm.list_collections",
    description="List MongoDB collections available in the current project (CRM tables).",
    input_model=CrmListCollectionsInput,
)
async def crm_list_collections(input: CrmListCollectionsInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    names = await _mongo(ctx).list_collections(pid)
    return ToolResult.ok(
        f"{len(names)} collection(s): {', '.join(names) or '(none)'}",
        data={"collections": names},
    )


# ── find ─────────────────────────────────────────────────────────────────────


class CrmFindInput(BaseModel):
    collection: str = Field(description="Collection name (e.g. `users`, `orders`)")
    filter: dict[str, Any] | None = Field(
        default=None,
        description="MongoDB filter expression. Examples: {age: {$gte: 18}}, {segment: 'vip'}",
    )
    sort: list[list[Any]] | None = Field(
        default=None,
        description="Sort spec, e.g. [['created_at', -1]]",
    )
    limit: int = Field(default=50, ge=1, le=500)
    skip: int = Field(default=0, ge=0)
    projection: dict[str, Any] | None = Field(
        default=None,
        description="Field projection, e.g. {first_name: 1, telegram_id: 1}",
    )
    project_id: str | int | None = None


@tool(
    name="crm.find",
    description=(
        "Query documents from a collection. Use filter/sort/limit/skip/projection "
        "just like MongoDB (projection selects which fields to return)."
    ),
    input_model=CrmFindInput,
)
async def crm_find(input: CrmFindInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    sort = [(s[0], int(s[1])) for s in input.sort] if input.sort else None
    docs = await _mongo(ctx).find(
        pid,
        input.collection,
        filter=input.filter,
        sort=sort,
        limit=input.limit,
        skip=input.skip,
        projection=input.projection,
    )
    return ToolResult.ok(f"{len(docs)} document(s).", data={"documents": docs})


# ── count ────────────────────────────────────────────────────────────────────


class CrmCountInput(BaseModel):
    collection: str
    filter: dict[str, Any] | None = None
    project_id: str | int | None = None


@tool(
    name="crm.count",
    description="Count documents matching a filter.",
    input_model=CrmCountInput,
)
async def crm_count(input: CrmCountInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    n = await _mongo(ctx).count(pid, input.collection, filter=input.filter)
    return ToolResult.ok(f"{n} document(s).", data={"count": n})


# ── aggregate (segmentation) ─────────────────────────────────────────────────


class CrmAggregateInput(BaseModel):
    collection: str
    pipeline: list[dict[str, Any]] = Field(
        description="MongoDB aggregation pipeline. Examples: $match, $group, $sort, $project."
    )
    limit: int = Field(default=200, ge=1, le=2000)
    project_id: str | int | None = None


@tool(
    name="crm.aggregate",
    description=(
        "Run a MongoDB aggregation pipeline — best for segmentation, "
        "grouping, and computing counts. Example pipeline for 'top 10 spenders': "
        "[{$group: {_id: '$user_id', total: {$sum: '$amount'}}}, {$sort: {total: -1}}, {$limit: 10}]"
    ),
    input_model=CrmAggregateInput,
)
async def crm_aggregate(input: CrmAggregateInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    docs = await _mongo(ctx).aggregate(pid, input.collection, input.pipeline, limit=input.limit)
    return ToolResult.ok(f"{len(docs)} row(s).", data={"rows": docs})


# ── insert ───────────────────────────────────────────────────────────────────


class CrmInsertInput(BaseModel):
    collection: str
    document: dict[str, Any]
    project_id: str | int | None = None


@tool(
    name="crm.insert",
    description="Insert a single document into a collection.",
    input_model=CrmInsertInput,
)
async def crm_insert(input: CrmInsertInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    inserted_id = await _mongo(ctx).insert(pid, input.collection, input.document)
    return ToolResult.ok(f"Inserted, _id={inserted_id}", data={"_id": inserted_id})


# ── update ───────────────────────────────────────────────────────────────────


class CrmUpdateInput(BaseModel):
    collection: str
    filter: dict[str, Any]
    update: dict[str, Any] = Field(
        description=(
            "Update fields. If you do not include any `$` operator, this becomes `{$set: update}`."
        )
    )
    many: bool = Field(default=False, description="Update many or just one")
    upsert: bool = Field(default=False)
    project_id: str | int | None = None


@tool(
    name="crm.update",
    description="Update document(s) matching filter.",
    input_model=CrmUpdateInput,
)
async def crm_update(input: CrmUpdateInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    modified = await _mongo(ctx).update(
        pid,
        input.collection,
        filter=input.filter,
        update=input.update,
        upsert=input.upsert,
        many=input.many,
    )
    return ToolResult.ok(f"Modified {modified} document(s).", data={"modified": modified})


# ── delete (destructive) ─────────────────────────────────────────────────────


class CrmDeleteInput(BaseModel):
    collection: str
    filter: dict[str, Any]
    many: bool = Field(default=False)
    project_id: str | int | None = None


@tool(
    name="crm.delete",
    description="Permanently delete document(s) matching filter — hard delete, not recoverable (use cms.delete for the soft-delete variant). Requires approval.",
    input_model=CrmDeleteInput,
    destructive=True,
)
async def crm_delete(input: CrmDeleteInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    n = await _mongo(ctx).delete(pid, input.collection, filter=input.filter, many=input.many)
    return ToolResult.ok(f"Deleted {n} document(s).", data={"deleted": n})


# ── segment (high-level convenience) ─────────────────────────────────────────


class CrmSegmentInput(BaseModel):
    collection: str = "users"
    filter: dict[str, Any] = Field(
        description="Filter that defines the segment, e.g. {segment: 'vip'} or {age: {$gte: 30}}"
    )
    project_id: str | int | None = None
    limit: int = Field(default=1000, ge=1, le=10000)


@tool(
    name="crm.segment",
    description=(
        "Build a target segment — returns the list of telegram_id values matching the filter. "
        "Used as input for broadcast.send."
    ),
    input_model=CrmSegmentInput,
)
async def crm_segment(input: CrmSegmentInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    docs = await _mongo(ctx).find(
        pid,
        input.collection,
        filter=input.filter,
        limit=input.limit,
        projection={"telegram_id": 1, "_id": 0},
    )
    ids = [d.get("telegram_id") for d in docs if d.get("telegram_id") is not None]
    return ToolResult.ok(
        f"Segment size: {len(ids)} user(s).",
        data={"telegram_ids": ids, "count": len(ids)},
        state_updates={"last_segment": {"filter": input.filter, "ids": ids}},
    )


def build_crm_tools() -> list[Tool]:
    return [
        crm_list_collections,
        crm_find,
        crm_count,
        crm_aggregate,
        crm_insert,
        crm_update,
        crm_delete,
        crm_segment,
    ]
