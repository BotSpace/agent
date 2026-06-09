"""cms.* tools — Django + MongoDB hybrid (schema + documents).

Replaces the older `crm.*` tools (which talked to project-scoped Mongo directly
via motor). cms goes through the backend REST API — so schema validation,
audit, soft-delete, webhooks, and indexing are all handled by the backend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ...integrations.backend import BackendClient, BackendError, get_backend_client, pagination_summary
from ..base import Tool, ToolContext, ToolResult, tool


# ── shared helpers ───────────────────────────────────────────────────────────


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def _bearer(ctx: ToolContext) -> str | None:
    claims = ctx.state.get("auth_claims") or {}
    return claims.get("access_token") if isinstance(claims, dict) else None


def _pid(ctx: ToolContext, override: str | int | None = None) -> str | int | None:
    return override or ctx.state.get("project_id")


def _error(e: BackendError, action: str | None = None) -> ToolResult:
    """Backend xatosini batafsil tarzda LLM uchun ko'rsatish."""
    from ...integrations.backend import format_backend_error
    return ToolResult.error(format_backend_error(action, e))


# ── result formatters ─────────────────────────────────────────────────────────

import json as _json

# Max docs serialised into the content string sent to the LLM.
# Keeps token usage predictable; the full set is still in result.data.
_CONTENT_DOC_LIMIT = 50


def _safe(obj: Any) -> Any:
    """Recursively make a value JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    try:
        _json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _fmt_rows(rows: list[Any], total: int | str, *, noun: str = "document") -> str:
    """Format a list of docs into a compact JSON string for the LLM."""
    header = f"{len(rows)} of {total} {noun}(s)"
    if not rows:
        return header + " — empty result."
    shown = rows[:_CONTENT_DOC_LIMIT]
    body = _json.dumps([_safe(d) for d in shown], ensure_ascii=False, separators=(",", ":"))
    tail = f"\n(first {len(shown)} shown)" if len(rows) > len(shown) else ""
    return f"{header}:\n{body}{tail}"


# ── field type catalog (matches backend FieldKind enum) ──────────────────────


FieldKindLiteral = Literal[
    "text", "longtext", "int", "float", "bool", "datetime", "date", "email",
    "url", "phone", "enum", "json", "array", "relation", "file", "image",
    "audio", "video", "geo",
]

IndexKindLiteral = Literal["none", "asc", "desc", "text", "hashed", "unique", "ttl"]


class FieldSpec(BaseModel):
    name: str = Field(description="snake_case field name")
    kind: FieldKindLiteral
    label: str = ""
    required: bool = False
    default: Any | None = None
    validators: dict[str, Any] = Field(default_factory=dict)
    relation_collection: int | None = Field(
        default=None, description="ID of the referenced collection (RELATION kind)"
    )
    index: IndexKindLiteral = "none"
    order: int = 0


# ── schema tools ─────────────────────────────────────────────────────────────


class ListCollectionsInput(BaseModel):
    project_id: str | int | None = None
    page: int | None = Field(
        default=None,
        description="Page number (10 per page). Paginated; omit for page 1, increment if total_pages > 1.",
    )


@tool(
    name="cms.list_collections",
    description=(
        "List cms collections (schemas) for the current project. Paginated (10 per page) — "
        "the result reports total_items/total_pages; pass `page` for more."
    ),
    input_model=ListCollectionsInput,
)
async def list_collections(input: ListCollectionsInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        items = await _backend(ctx).cms_list_collections(
            pid, page=input.page, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    payload = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "label": c.get("label"),
            "fields_count": c.get("fields_count"),
            "system": c.get("system"),
        }
        for c in items
    ]
    names = ", ".join(str(c.get("name")) for c in payload if c.get("name")) or "(none)"
    total = getattr(items, "total_items", len(payload))
    return ToolResult.ok(
        f"Showing {len(payload)} of {total} collection(s): {names}{pagination_summary(items)}",
        data={
            "collections": payload,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


class GetCollectionInput(BaseModel):
    name: str
    project_id: str | int | None = None


@tool(
    name="cms.get_collection",
    description="Get a single collection's schema (fields, validators, indexes).",
    input_model=GetCollectionInput,
)
async def get_collection(input: GetCollectionInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        c = await _backend(ctx).cms_get_collection(pid, input.name, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return _error(e)
    fields = c.get("fields", [])
    field_summary = ", ".join(
        f"{f.get('name')}({f.get('kind')}{'*' if f.get('required') else ''})"
        for f in fields
    ) or "(no fields)"
    return ToolResult.ok(
        f"Collection `{input.name}` — {len(fields)} field(s): {field_summary}",
        data={"collection": c},
    )


class CreateCollectionInput(BaseModel):
    name: str = Field(description="snake_case, e.g. 'orders', 'leads'")
    label: str = ""
    description: str = ""
    primary_field: str | None = Field(
        default=None,
        description=(
            "relation orqali ko'rsatilganda title sifatida ishlatiladigan field "
            "nomi, masalan 'name'. Boshqa collection bu collection'ga relation "
            "field bilan bog'langanda shu field display qiymat bo'lib chiqadi."
        ),
    )
    fields: list[FieldSpec] = Field(default_factory=list)
    project_id: str | int | None = None


@tool(
    name="cms.create_collection",
    description=(
        "Create a new collection with an optional initial schema. Field `kind` "
        "must be one of: text, longtext, int, float, bool, datetime, date, "
        "email, url, phone, enum, json, array, relation, file, image, audio, "
        "video, geo. RELATION kind requires `relation_collection` (collection id). "
        "Set `primary_field` to the field used as the display/title when this "
        "collection is referenced from another via a relation (e.g. 'name'). "
        "Returns the new collection's `id` (in content and structuredContent) — "
        "use it as `relation_collection` when adding a RELATION field elsewhere."
    ),
    input_model=CreateCollectionInput,
)
async def create_collection(input: CreateCollectionInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    body = input.model_dump(exclude={"project_id"})
    # primary_field=None bo'lsa body'dan tushirib qoldiramiz (backend default'i
    # buzilmasin); berilgan bo'lsa backend serializer'ga uzatiladi.
    if body.get("primary_field") is None:
        body.pop("primary_field", None)
    try:
        c = await _backend(ctx).cms_create_collection(pid, body, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return _error(e)
    # #6 — surface the backend-assigned id explicitly (needed for relations).
    new_id = c.get("id") if isinstance(c, dict) else None
    return ToolResult.ok(
        f"Collection `{c.get('name')}` created (id={new_id}). "
        "Use this id as `relation_collection` when linking from another collection.",
        data={"collection": c, "id": new_id},
    )


class DeleteCollectionInput(BaseModel):
    name: str
    project_id: str | int | None = None
    confirm: bool = Field(
        default=False,
        description=(
            "Tasdiqlash. False (default) — o'chirilMAYDI, nima o'chishi haqida "
            "xulosa qaytaradi. Faqat tasdiqlangach confirm=true."
        ),
    )


@tool(
    name="cms.delete_collection",
    description=(
        "Soft-delete a collection schema (documents remain in MongoDB until "
        "purged). confirm=true talab qiladi — confirm berilmasa nima o'chishini "
        "qaytaradi, o'chirmaydi."
    ),
    input_model=DeleteCollectionInput,
    destructive=True,
)
async def delete_collection(input: DeleteCollectionInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    if not input.confirm:
        return ToolResult.error(
            f"DIQQAT: `{input.name}` kolleksiyasi sxemasi o'chiriladi (hujjatlar "
            "MongoDB'da qoladi, lekin sxema soft-delete bo'ladi). Tasdiqlash uchun "
            "confirm=true bilan qayta chaqiring."
        )
    try:
        await _backend(ctx).cms_delete_collection(pid, input.name, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Collection `{input.name}` deleted.")


class AddFieldInput(BaseModel):
    collection: str
    field: FieldSpec
    project_id: str | int | None = None


@tool(
    name="cms.add_field",
    description="Add a new field to an existing collection's schema.",
    input_model=AddFieldInput,
)
async def add_field(input: AddFieldInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        f = await _backend(ctx).cms_add_field(
            pid, input.collection, input.field.model_dump(), auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Field `{f.get('name')}` added.", data={"field": f})


class UpdateFieldInput(BaseModel):
    collection: str
    field_name: str
    patch: dict[str, Any]
    project_id: str | int | None = None


@tool(
    name="cms.update_field",
    description="Patch a single field (validators, index, required, ...).",
    input_model=UpdateFieldInput,
)
async def update_field(input: UpdateFieldInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        f = await _backend(ctx).cms_update_field(
            pid, input.collection, input.field_name, input.patch, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Field `{input.field_name}` updated.", data={"field": f})


class RemoveFieldInput(BaseModel):
    collection: str
    field_name: str
    project_id: str | int | None = None


@tool(
    name="cms.remove_field",
    description="Remove a field from a collection's schema. Existing documents are not modified.",
    input_model=RemoveFieldInput,
    destructive=True,
)
async def remove_field(input: RemoveFieldInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        await _backend(ctx).cms_remove_field(
            pid, input.collection, input.field_name, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Field `{input.field_name}` removed from `{input.collection}`.")


# ── document tools ───────────────────────────────────────────────────────────


class ListDocumentsInput(BaseModel):
    collection: str
    filter: dict[str, Any] | None = None
    sort: str | None = Field(
        default=None,
        description="Comma-separated, '-' prefix = desc. e.g. '-_created_at,name'",
    )
    limit: int = Field(default=50, ge=1, le=500)
    skip: int = Field(default=0, ge=0)
    project_id: str | int | None = None


@tool(
    name="cms.find",
    description=(
        "Query documents in a collection. `filter` is a MongoDB filter (e.g. "
        "{age: {$gte: 18}}). Soft-deleted documents are excluded by default."
    ),
    input_model=ListDocumentsInput,
)
async def find(input: ListDocumentsInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        res = await _backend(ctx).cms_list_documents(
            pid,
            input.collection,
            filter=input.filter,
            sort=input.sort,
            limit=input.limit,
            skip=input.skip,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return _error(e)
    rows = res.get("results", [])
    total = res.get("total", "?")
    return ToolResult.ok(
        _fmt_rows(rows, total),
        data={"results": rows, "total": total, "limit": res.get("limit"), "skip": res.get("skip")},
    )


class GetDocumentInput(BaseModel):
    collection: str
    id: str
    project_id: str | int | None = None


@tool(
    name="cms.get_document",
    description="Fetch a single document by id.",
    input_model=GetDocumentInput,
)
async def get_document(input: GetDocumentInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_get_document(
            pid, input.collection, input.id, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    body = _json.dumps(_safe(d), ensure_ascii=False, separators=(",", ":"))
    return ToolResult.ok(f"Document `{input.id}`:\n{body}", data={"document": d})


class InsertInput(BaseModel):
    collection: str
    document: dict[str, Any]
    project_id: str | int | None = None


@tool(
    name="cms.insert",
    description=(
        "Insert a single document. The backend validates against the collection "
        "schema, stamps _created_at/_updated_at/_version, fires the doc.created webhook."
    ),
    input_model=InsertInput,
)
async def insert(input: InsertInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_insert(
            pid, input.collection, input.document, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(
        f"Inserted, _id={d.get('_id')}", data={"document": d}
    )


class BulkInsertInput(BaseModel):
    collection: str
    documents: list[dict[str, Any]]
    project_id: str | int | None = None


@tool(
    name="cms.bulk_insert",
    description="Insert many documents in one call. If any document fails validation the whole batch is rejected (nothing is inserted).",
    input_model=BulkInsertInput,
)
async def bulk_insert(input: BulkInsertInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_bulk_insert(
            pid, input.collection, input.documents, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(
        f"Inserted {d.get('count', len(input.documents))} document(s).", data=d
    )


class UpdateInput(BaseModel):
    collection: str
    id: str
    patch: dict[str, Any]
    expected_version: int | None = Field(
        default=None,
        description="Optimistic concurrency token. If set, update fails when _version differs.",
    )
    project_id: str | int | None = None


@tool(
    name="cms.update",
    description="Patch a single document by id (partial update).",
    input_model=UpdateInput,
)
async def update_doc(input: UpdateInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    patch = dict(input.patch)
    if input.expected_version is not None:
        patch["_version"] = input.expected_version
    try:
        d = await _backend(ctx).cms_update(
            pid, input.collection, input.id, patch, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Document `{input.id}` updated.", data={"document": d})


class DeleteInput(BaseModel):
    collection: str
    id: str
    project_id: str | int | None = None


@tool(
    name="cms.delete",
    description="Soft-delete a document — recoverable via cms.restore (unlike crm.delete, which is a permanent hard delete).",
    input_model=DeleteInput,
    destructive=True,
)
async def delete_doc(input: DeleteInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        await _backend(ctx).cms_delete(
            pid, input.collection, input.id, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Document `{input.id}` soft-deleted.")


class RestoreInput(BaseModel):
    collection: str
    id: str
    project_id: str | int | None = None


@tool(
    name="cms.restore",
    description="Restore a soft-deleted document.",
    input_model=RestoreInput,
)
async def restore_doc(input: RestoreInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        res = await _backend(ctx).cms_restore(
            pid, input.collection, input.id, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Document `{input.id}` restored.", data=res)


class AggregateInput(BaseModel):
    collection: str
    pipeline: list[dict[str, Any]] = Field(
        description="MongoDB aggregation pipeline. Example: [{$match:{x:1}}, {$group:{_id:'$y',n:{$sum:1}}}]"
    )
    limit: int = Field(default=500, ge=1, le=5000)
    project_id: str | int | None = None


@tool(
    name="cms.aggregate",
    description="Run a MongoDB aggregation pipeline (segmentation, group-by, joins via $lookup).",
    input_model=AggregateInput,
)
async def aggregate(input: AggregateInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_aggregate(
            pid, input.collection, input.pipeline, limit=input.limit, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    rows = d.get("results", [])
    count = d.get("count", len(rows))
    return ToolResult.ok(_fmt_rows(rows, count, noun="row"), data=d)


class AuditInput(BaseModel):
    collection: str
    action: str | None = Field(
        default=None,
        description="Filter by action: create | update | delete | restore | schema_create | ...",
    )
    document_id: str | None = None
    project_id: str | int | None = None


@tool(
    name="cms.audit_log",
    description="View the audit log of a collection (last 200 entries).",
    input_model=AuditInput,
)
async def audit_log(input: AuditInput, ctx: ToolContext) -> ToolResult:
    pid = _pid(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        items = await _backend(ctx).cms_audit(
            pid,
            input.collection,
            action=input.action,
            document_id=input.document_id,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(_fmt_rows(items, len(items), noun="audit entry"), data={"audit": items})


def build_cms_tools() -> list[Tool]:
    return [
        list_collections,
        get_collection,
        create_collection,
        delete_collection,
        add_field,
        update_field,
        remove_field,
        find,
        get_document,
        insert,
        bulk_insert,
        update_doc,
        delete_doc,
        restore_doc,
        aggregate,
        audit_log,
    ]
