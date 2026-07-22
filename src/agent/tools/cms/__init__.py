"""cms.* tools — Django + MongoDB hybrid (schema + documents).

Replaces the older `crm.*` tools (which talked to project-scoped Mongo directly
via motor). cms goes through the backend REST API — so schema validation,
audit, soft-delete, webhooks, and indexing are all handled by the backend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ...integrations.backend import BackendError, pagination_summary

# ── shared helpers ───────────────────────────────────────────────────────────
from .._common import backend_from_ctx as _backend
from .._common import resolve_project_id
from ..base import Tool, ToolContext, ToolResult, tool


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
    "text",
    "longtext",
    "int",
    "float",
    "bool",
    "datetime",
    "date",
    "email",
    "url",
    "phone",
    "enum",
    "json",
    "array",
    "relation",
    "file",
    "image",
    "audio",
    "video",
    "geo",
]

IndexKindLiteral = Literal["none", "asc", "desc", "text", "hashed", "unique", "ttl"]

# Common generic/JSON-Schema type names the model tends to invent for a field,
# mapped to our backend FieldKind. Lets `type: "string"` (etc.) work instead of
# silently dropping the whole schema. Unknown values pass through unchanged so a
# real typo still surfaces as a validation error.
_KIND_ALIASES = {
    "string": "text",
    "str": "text",
    "char": "text",
    "varchar": "text",
    "uuid": "text",
    "number": "float",
    "num": "float",
    "double": "float",
    "decimal": "float",
    "integer": "int",
    "long": "int",
    "bigint": "int",
    "boolean": "bool",
    "timestamp": "datetime",
    "time": "datetime",
    "object": "json",
    "dict": "json",
    "map": "json",
    "list": "array",
    "photo": "image",
    "picture": "image",
    "img": "image",
    "location": "geo",
    "point": "geo",
}


class FieldSpec(BaseModel):
    # Accept the model's natural aliases (`type` for `kind`) so a hallucinated key
    # name doesn't wipe the field.
    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(description="snake_case field name")
    kind: FieldKindLiteral = Field(validation_alias=AliasChoices("kind", "type"))
    label: str = ""
    required: bool = False
    default: Any | None = None
    validators: dict[str, Any] = Field(default_factory=dict)
    relation_collection: str | None = Field(
        default=None,
        description="Slug of the referenced collection (RELATION kind), e.g. courses",
    )
    index: IndexKindLiteral = "none"
    order: int = 0

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _KIND_ALIASES.get(v.strip().lower(), v.strip().lower())
        return v


# ── schema tools ─────────────────────────────────────────────────────────────


class ListCollectionsInput(BaseModel):
    project_id: str | int | None = None
    page: int | None = Field(
        default=None,
        description="Page number. Paginated; omit for page 1, increment if total_pages > 1.",
    )
    page_size: int | None = Field(
        default=None,
        description="Items per page (default 10). Increase (e.g. 100) to fetch more at once.",
    )


@tool(
    name="cms.list_collections",
    description=(
        "List cms collections (schemas) for the current project. Paginated (10/page by default) — "
        "the result reports total_items/total_pages; pass `page`/`page_size` for more."
    ),
    input_model=ListCollectionsInput,
)
async def list_collections(input: ListCollectionsInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        items = await _backend(ctx).cms_list_collections(
            pid, page=input.page, page_size=input.page_size, auth_claims=ctx.state.get("auth_claims")
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
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        c = await _backend(ctx).cms_get_collection(pid, input.name, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return _error(e)
    fields = c.get("fields", [])
    field_summary = (
        ", ".join(f"{f.get('name')}({f.get('kind')}{'*' if f.get('required') else ''})" for f in fields)
        or "(no fields)"
    )
    return ToolResult.ok(
        f"Collection `{input.name}` — {len(fields)} field(s): {field_summary}",
        data={"collection": c},
    )


class SchemaAssistantInput(BaseModel):
    collections: list[str] = Field(
        default_factory=list,
        description="Collection sluglari. Bo'sh bo'lsa loyihadagi barcha collectionlar olinadi.",
    )
    project_id: str | int | None = None


@tool(
    name="cms.schema_assistant",
    description=(
        "Return a compact, agent-oriented CMS contract: exact collection slugs, field "
        "names/types, required flags, relation targets and safe CollectionNode mappings. "
        "Call before building any flow that reads or writes CMS data."
    ),
    input_model=SchemaAssistantInput,
    core=True,
)
async def schema_assistant(input: SchemaAssistantInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    names = [name.strip() for name in input.collections if name.strip()]
    try:
        if not names:
            listed = await _backend(ctx).cms_list_collections(
                pid,
                page_size=100,
                auth_claims=ctx.state.get("auth_claims"),
            )
            names = [str(item.get("name")) for item in listed if isinstance(item, dict) and item.get("name")]
        contracts: list[dict[str, Any]] = []
        for name in names:
            collection = await _backend(ctx).cms_get_collection(
                pid,
                name,
                auth_claims=ctx.state.get("auth_claims"),
            )
            fields = []
            for raw in collection.get("fields") or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                kind = raw.get("kind") or raw.get("type") or "text"
                field = {
                    "name": str(raw["name"]),
                    "kind": str(kind),
                    "required": bool(raw.get("required")),
                }
                relation = raw.get("relation_collection") or raw.get("relationCollection")
                if relation:
                    field["relation_collection"] = str(relation)
                fields.append(field)
            contracts.append(
                {
                    "name": name,
                    "label": collection.get("label"),
                    "fields": fields,
                    "write_contract": {
                        "node_type": "CollectionNode",
                        "collection_name": name,
                        "fieldMappings": {
                            field["name"]: (
                                "{{selected_item.id}}" if field["kind"] == "relation" else field["name"]
                            )
                            for field in fields
                        },
                    },
                }
            )
    except BackendError as exc:
        return _error(exc)
    lines = [f"CMS contract — {len(contracts)} collection:"]
    for contract in contracts:
        rendered = []
        for field in contract["fields"]:
            suffix = "*" if field["required"] else ""
            relation = f"→{field['relation_collection']}" if field.get("relation_collection") else ""
            rendered.append(f"{field['name']}:{field['kind']}{relation}{suffix}")
        lines.append(f"• {contract['name']}: {', '.join(rendered) or "(field yo'q)"}")
    lines.append(
        "Mapping qoidasi: oddiy capture uchun yalang'och state key; relation uchun "
        "tanlangan full_data hujjatning `{{selected_item.id}}` qiymati. Labelni slug o'rnida ishlatmang."
    )
    return ToolResult.ok(
        "\n".join(lines),
        data={"contracts": contracts},
        data_in_content=True,
    )


class CreateCollectionInput(BaseModel):
    # `initial_schema` is a very common alias the model invents for `fields`;
    # accept it so the schema isn't silently dropped (→ fieldless collection).
    model_config = ConfigDict(populate_by_name=True)
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
    fields: list[FieldSpec] = Field(
        default_factory=list,
        validation_alias=AliasChoices("fields", "initial_schema"),
    )
    project_id: str | int | None = None


def _slugify(name: str) -> str:
    """snake_case/erkin nomdan collection slug (backend `slug` majburiy)."""
    import re

    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    return s or "collection"


@tool(
    name="cms.create_collection",
    description=(
        "Create a new collection with an optional initial schema. Field `kind` "
        "must be one of: text, longtext, int, float, bool, datetime, date, "
        "email, url, phone, enum, json, array, relation, file, image, audio, "
        "video, geo. RELATION kind requires `relation_collection` (collection slug). "
        "Set `primary_field` to the field used as the display/title when this "
        "collection is referenced from another via a relation (e.g. 'name'). "
        "RELATION `relation_collection` is the target collection SLUG, not its id "
        "(for example `courses`). Returns both the new collection's slug and id."
    ),
    input_model=CreateCollectionInput,
)
async def create_collection(input: CreateCollectionInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    # Go backend: create takes {name, slug(required), description, primary_field};
    # fields are added SEPARATELY. Derive slug from name and add each field after.
    slug = _slugify(input.name)
    create_body: dict[str, Any] = {"name": input.name, "slug": slug}
    if input.description:
        create_body["description"] = input.description
    if input.primary_field is not None:
        create_body["primary_field"] = input.primary_field

    def _is_conflict(e: BackendError, code: str) -> bool:
        return e.status == 409 or code in (e.codes or [])

    existed = False
    c: dict[str, Any] | None = None
    try:
        c = await _backend(ctx).cms_create_collection(
            pid, create_body, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        # Idempotent: if the collection already exists, reuse it and still add any
        # missing fields (don't abort — otherwise a re-run leaves a collection with
        # NO schema while documents keep flowing in schemalessly).
        if _is_conflict(e, "cms.collection_exists"):
            existed = True
            try:
                c = await _backend(ctx).cms_get_collection(
                    pid, slug, auth_claims=ctx.state.get("auth_claims")
                )
            except BackendError:
                c = None
        else:
            return _error(e)
    new_id = c.get("id") if isinstance(c, dict) else None

    # Add the initial schema fields (backend client remaps kind→type).
    added: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for f in input.fields:
        try:
            await _backend(ctx).cms_add_field(
                pid, slug, f.model_dump(), auth_claims=ctx.state.get("auth_claims")
            )
            added.append(f.name)
        except BackendError as e:
            if _is_conflict(e, "cms.field_exists"):
                skipped.append(f.name)  # already present, not a failure
            else:
                failed.append(f"{f.name}: {e.human_message()}")

    name_disp = (c.get("name") if isinstance(c, dict) else None) or input.name
    verb = "already existed — reused" if existed else "created"
    msg = f"Collection `{name_disp}` {verb} (slug={slug})."
    if added:
        msg += f" Fields added: {', '.join(added)}."
    if skipped:
        msg += f" Already present: {', '.join(skipped)}."
    if failed:
        msg += f" ⚠️ Field FAILED: {'; '.join(failed)}."
    if new_id is not None:
        msg += f" Collection id: `{new_id}`."
    msg += f" Use slug `{slug}` as `relation_collection` when linking another schema."
    return ToolResult.ok(
        msg,
        data={"collection": c, "id": new_id, "fields_added": added, "fields_failed": failed},
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
    pid = resolve_project_id(ctx, input.project_id)
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
    pid = resolve_project_id(ctx, input.project_id)
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
    pid = resolve_project_id(ctx, input.project_id)
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
    pid = resolve_project_id(ctx, input.project_id)
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
    untrusted_output=True,
)
async def find(input: ListDocumentsInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
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
        data_in_content=True,
    )


class GetDocumentInput(BaseModel):
    collection: str
    id: str
    project_id: str | int | None = None


@tool(
    name="cms.get_document",
    description="Fetch a single document by id.",
    input_model=GetDocumentInput,
    untrusted_output=True,
)
async def get_document(input: GetDocumentInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_get_document(
            pid, input.collection, input.id, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    body = _json.dumps(_safe(d), ensure_ascii=False, separators=(",", ":"))
    # content to'liq hujjatni o'z ichiga oladi → MCP YAML(data)ni takrorlamasin.
    return ToolResult.ok(f"Document `{input.id}`:\n{body}", data={"document": d}, data_in_content=True)


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
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_insert(
            pid, input.collection, input.document, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Inserted, _id={d.get('_id')}", data={"document": d})


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
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        d = await _backend(ctx).cms_bulk_insert(
            pid, input.collection, input.documents, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Inserted {d.get('count', len(input.documents))} document(s).", data=d)


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
    pid = resolve_project_id(ctx, input.project_id)
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
    pid = resolve_project_id(ctx, input.project_id)
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
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        res = await _backend(ctx).cms_restore(
            pid, input.collection, input.id, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return _error(e)
    return ToolResult.ok(f"Document `{input.id}` restored.", data=res)


# ── bulk update / delete (filter bo'yicha) ───────────────────────────────────
# ponytail: client-side loop (find → per-doc update/delete). Backend'da atomik
# bulk-update endpoint yo'q; agent uchun 1 tool chaqiruvi — asosiy og'riqni yechadi.
# Atomiklik/tezlik kerak bo'lsa backend'ga bulk endpoint qo'shiladi.


class UpdateManyInput(BaseModel):
    collection: str
    filter: dict[str, Any] = Field(
        description="Yangilanadigan hujjatlarni tanlovchi MongoDB filter, masalan {code:'123'}. "
        "Bo'sh {} RAD ETILADI (butun kolleksiyani yangilab yubormaslik uchun)."
    )
    patch: dict[str, Any] = Field(
        description="Har bir mos hujjatga qo'llaniladigan qisman yangilanish (partial merge)."
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Ko'pi bilan shuncha hujjat yangilanadi (xavfsizlik chegarasi).",
    )
    project_id: str | int | None = None


@tool(
    name="cms.update_many",
    description=(
        "Update every document matching a MongoDB `filter` with the same `patch` "
        "(partial merge). One call instead of many cms.update. Not atomic — each "
        "doc is patched in turn; partial failure reports how many succeeded. Empty "
        "filter is rejected. At most `limit` (default 500) docs are touched."
    ),
    input_model=UpdateManyInput,
)
async def update_many(input: UpdateManyInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    if not input.filter:
        return ToolResult.error(
            "filter bo'sh bo'lishi mumkin emas — butun kolleksiyani yangilab "
            'yubormaslik uchun aniq filter bering (masalan {"code":"123"}).'
        )
    if not input.patch:
        return ToolResult.error("patch bo'sh — yangilanadigan maydon yo'q.")
    claims = ctx.state.get("auth_claims")
    try:
        res = await _backend(ctx).cms_list_documents(
            pid, input.collection, filter=input.filter, limit=input.limit, auth_claims=claims
        )
    except BackendError as e:
        return _error(e)
    rows = res.get("results", [])
    ids = [str(r["_id"]) for r in rows if r.get("_id")]
    total = res.get("total", len(ids))
    updated, errors = 0, []
    for did in ids:
        try:
            await _backend(ctx).cms_update(pid, input.collection, did, dict(input.patch), auth_claims=claims)
            updated += 1
        except BackendError as e:
            errors.append(f"{did}: {e}")
    msg = f"Updated {updated}/{len(ids)} document(s) matching filter."
    if isinstance(total, int) and total > len(ids):
        msg += f" ({total} matched; capped at limit={input.limit} — increase `limit` for the rest.)"
    if errors:
        msg += f" {len(errors)} failed; first: {errors[0]}"
    return ToolResult.ok(
        msg, data={"matched": len(ids), "total": total, "updated": updated, "errors": errors[:10]}
    )


class DeleteManyInput(BaseModel):
    collection: str
    filter: dict[str, Any] = Field(
        description="O'chiriladigan hujjatlarni tanlovchi MongoDB filter. Bo'sh {} RAD ETILADI."
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Ko'pi bilan shuncha hujjat o'chiriladi (xavfsizlik chegarasi).",
    )
    project_id: str | int | None = None


@tool(
    name="cms.delete_many",
    description=(
        "Soft-delete every document matching a MongoDB `filter` (recoverable via "
        "cms.restore). One call instead of many cms.delete. Not atomic. Empty "
        "filter is rejected. At most `limit` (default 500) docs are deleted."
    ),
    input_model=DeleteManyInput,
    destructive=True,
)
async def delete_many(input: DeleteManyInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    if not input.filter:
        return ToolResult.error(
            "filter bo'sh bo'lishi mumkin emas — butun kolleksiyani o'chirib "
            "yubormaslik uchun aniq filter bering."
        )
    claims = ctx.state.get("auth_claims")
    try:
        res = await _backend(ctx).cms_list_documents(
            pid, input.collection, filter=input.filter, limit=input.limit, auth_claims=claims
        )
    except BackendError as e:
        return _error(e)
    rows = res.get("results", [])
    ids = [str(r["_id"]) for r in rows if r.get("_id")]
    total = res.get("total", len(ids))
    deleted, errors = 0, []
    for did in ids:
        try:
            await _backend(ctx).cms_delete(pid, input.collection, did, auth_claims=claims)
            deleted += 1
        except BackendError as e:
            errors.append(f"{did}: {e}")
    msg = f"Soft-deleted {deleted}/{len(ids)} document(s) matching filter."
    if isinstance(total, int) and total > len(ids):
        msg += f" ({total} matched; capped at limit={input.limit}.)"
    if errors:
        msg += f" {len(errors)} failed; first: {errors[0]}"
    return ToolResult.ok(
        msg, data={"matched": len(ids), "total": total, "deleted": deleted, "errors": errors[:10]}
    )


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
    untrusted_output=True,
)
async def aggregate(input: AggregateInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
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
    return ToolResult.ok(_fmt_rows(rows, count, noun="row"), data=d, data_in_content=True)


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
    pid = resolve_project_id(ctx, input.project_id)
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
    return ToolResult.ok(
        _fmt_rows(items, len(items), noun="audit entry"), data={"audit": items}, data_in_content=True
    )


def build_cms_tools() -> list[Tool]:
    return [
        list_collections,
        get_collection,
        schema_assistant,
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
        update_many,
        delete_doc,
        delete_many,
        restore_doc,
        aggregate,
        audit_log,
    ]
