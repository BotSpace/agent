"""credentials.* tools — read-only listing of the OWNER's stored credentials.

Credentials are n8n-style, owner-global (NOT per-project): one user's saved
API keys / OAuth connections, shared across all their bots. A flow node never
stores a secret — it references a credential by **id** in `data.credential_id`,
and the engine resolves it to a fresh token at runtime.

This tool lets the agent DISCOVER which credentials the user already has so it
can pick the right `credential_id` for credential-consuming nodes:
  - AIOpenAIProviderNode  → type_key 'openai'
  - AIOllamaProviderNode   → type_key 'ollama'
  - AIGeminiProviderNode   → type_key 'ai_studio'
  - GoogleSheetsNode       → type_key 'google_sheets'
  - GoogleDriveNode        → type_key 'google_drive'
  - HTTPRequestNode        → any type_key

The agent can list/create/update/delete credentials. NON-OAuth types
(telegram/openai/ai_studio/ollama) can be created here when the user provides the
secret. OAUTH types (google_email/google_sheets/google_drive) CANNOT be created via
the API — they need the browser "Connect with Google" consent popup the user does in
the dashboard; for those, ask the user to connect, then list.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendClient,
    BackendError,
    format_backend_error,
    get_backend_client,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


# Hint shown when no credential matches — maps a node type to its type_key so
# the LLM gives the user the right thing to create.
_NODE_TYPE_KEYS = (
    "AIOpenAIProviderNode→openai, AIOllamaProviderNode→ollama, "
    "AIGeminiProviderNode→ai_studio, GoogleSheetsNode→google_sheets, "
    "GoogleDriveNode→google_drive, HTTPRequestNode→any"
)


class CredentialsListInput(BaseModel):
    type_key: str | None = Field(
        default=None,
        description=(
            "Filter by credential type. Use the type that matches the node: "
            "openai | ai_studio | ollama | telegram | google_email | google_sheets | "
            "google_drive. Omit to list all of the user's credentials."
        ),
    )
    page: int | None = Field(default=None, description="Page number (paginated; omit for page 1).")
    page_size: int | None = Field(default=None, description="Items per page (default 20).")


@tool(
    name="credentials.list",
    description=(
        "List the USER's stored credentials (owner-global, NOT per-project) so you can "
        "pick a `credential_id` for a credential-consuming node. Secrets are masked. "
        "Filter with `type_key` to match the node "
        f"({_NODE_TYPE_KEYS}). Read-only — the user creates/connects credentials in the "
        "panel/dashboard (OAuth needs a consent popup). If none match, ask the user to "
        "create one there, then re-list."
    ),
    input_model=CredentialsListInput,
)
async def credentials_list(input: CredentialsListInput, ctx: ToolContext) -> ToolResult:
    try:
        items = await _backend(ctx).list_credentials(
            type_key=input.type_key,
            page=input.page,
            page_size=input.page_size,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("credentials.list", e))

    total = getattr(items, "total_items", len(items))
    note = pagination_summary(items)
    payload = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "type_key": c.get("type_key"),
            "label": c.get("label"),
            "mode": c.get("mode"),
        }
        for c in items
        if isinstance(c, dict)
    ]

    if not payload:
        filt = f" of type `{input.type_key}`" if input.type_key else ""
        return ToolResult.ok(
            f"No credentials{filt} found for this user. For NON-OAuth types "
            "(telegram/openai/ai_studio/ollama) you can make one with credentials.create when "
            "the user gives you the secret. For OAuth types (google_*) the user must connect them "
            "in the dashboard. node→type_key: " + _NODE_TYPE_KEYS + ".",
            data={"credentials": [], "total_items": 0},
        )

    lines = [f"Showing {len(payload)} of {total} credential(s):{note}"]
    for c in payload:
        lines.append(
            f"  id={c['id']}  name={c['name']!r}  type_key={c['type_key']}  "
            f"({c.get('label') or '—'}, mode={c.get('mode') or '—'})"
        )
    lines.append("Use an `id` above as `data.credential_id` on the matching node.")
    return ToolResult.ok(
        "\n".join(lines),
        data={
            "credentials": payload,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


# OAuth types can't be created via the API — they need the browser consent popup.
_OAUTH_TYPE_KEYS = {"google_email", "google_sheets", "google_drive"}

# Data fields per non-oauth type, to guide the LLM on what `data` to pass.
_TYPE_FIELDS = (
    "telegram→{bot_token}, openai→{api_key, base_url?}, "
    "ai_studio→{api_key}, ollama→{base_url, api_key?}"
)


class CredentialsCreateInput(BaseModel):
    type_key: str = Field(
        description=(
            "Credential type. NON-OAuth only: telegram | openai | ai_studio | ollama. "
            "OAuth types (google_*) cannot be created here — the user connects them in the "
            "dashboard."
        )
    )
    name: str = Field(description="Human label for the credential, e.g. 'My OpenAI key'.")
    data: dict = Field(
        description=(
            "Secret field values for the type. Fields per type: " + _TYPE_FIELDS + ". "
            "Only set what the user explicitly provides."
        )
    )


@tool(
    name="credentials.create",
    description=(
        "Create a NON-OAuth credential (telegram/openai/ai_studio/ollama) when the user gives "
        "you the secret (api key / bot token). Returns the new id to use as `data.credential_id`. "
        "Fields per type: " + _TYPE_FIELDS + ". OAuth types (google_*) are REJECTED — the user "
        "must connect those via the dashboard's 'Connect with Google'. Note: the secret passes "
        "through this conversation, so only do this when the user explicitly asks."
    ),
    input_model=CredentialsCreateInput,
)
async def credentials_create(input: CredentialsCreateInput, ctx: ToolContext) -> ToolResult:
    if input.type_key in _OAUTH_TYPE_KEYS:
        return ToolResult.error(
            f"`{input.type_key}` is an OAuth credential and cannot be created via the API. "
            "Ask the user to open the dashboard → Credentials → connect with Google, then list."
        )
    try:
        cred = await _backend(ctx).create_credential(
            {"type_key": input.type_key, "name": input.name, "data": input.data},
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("credentials.create", e))
    cid = cred.get("id") if isinstance(cred, dict) else None
    return ToolResult.ok(
        f"Created credential id={cid} name={input.name!r} type_key={input.type_key}. "
        "Use this id as `data.credential_id` on the matching node.",
        data={"id": cid, "credential": cred},
    )


class CredentialsUpdateInput(BaseModel):
    id: int = Field(description="Credential id to update.")
    name: str | None = Field(default=None, description="New name (optional).")
    data: dict | None = Field(
        default=None,
        description="Field values to change (optional). Empty secret fields keep the old value.",
    )


@tool(
    name="credentials.update",
    description=(
        "Update a credential's name and/or secret field values (PATCH). Omitted/empty secret "
        "fields keep their existing value. Works for any type (including renaming an OAuth "
        "credential)."
    ),
    input_model=CredentialsUpdateInput,
)
async def credentials_update(input: CredentialsUpdateInput, ctx: ToolContext) -> ToolResult:
    body: dict = {}
    if input.name is not None:
        body["name"] = input.name
    if input.data is not None:
        body["data"] = input.data
    if not body:
        return ToolResult.error("Nothing to update — provide `name` and/or `data`.")
    try:
        cred = await _backend(ctx).update_credential(
            input.id, body, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("credentials.update", e))
    return ToolResult.ok(f"Updated credential id={input.id}.", data={"credential": cred})


class CredentialsDeleteInput(BaseModel):
    id: int = Field(description="Credential id to delete (soft delete).")


@tool(
    name="credentials.delete",
    description=(
        "Delete (soft) a credential by id. Any node still referencing this credential_id will "
        "fail to resolve at runtime — warn the user / update those nodes first."
    ),
    input_model=CredentialsDeleteInput,
)
async def credentials_delete(input: CredentialsDeleteInput, ctx: ToolContext) -> ToolResult:
    try:
        await _backend(ctx).delete_credential(input.id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error("credentials.delete", e))
    return ToolResult.ok(f"Deleted credential id={input.id}.", data={"id": input.id})


def build_credentials_tools() -> list[Tool]:
    return [credentials_list, credentials_create, credentials_update, credentials_delete]
