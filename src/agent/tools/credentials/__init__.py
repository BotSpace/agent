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
the API — they need a browser "Connect with Google" consent. For those, the agent
calls `credentials.connect_oauth` to mint a one-click ULASH (connect) link the user
opens; after they grant consent the backend creates the credential under the same
owner, and the agent re-lists to read the new `credential_id`.
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
    "GoogleDriveNode→google_drive, "
    "AIGoogleSheetsToolNode→google_sheets, HTTPRequestNode→any"
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
            "the user gives you the secret. For OAuth types (google_*) call "
            "credentials.connect_oauth to give the user a connect link (they can't be created via "
            "the API); after they connect, re-list. node→type_key: " + _NODE_TYPE_KEYS + ".",
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
            "Call credentials.connect_oauth(type_key="
            f"'{input.type_key}') to give the user a connect link, then re-list."
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


# OAuth type_key → provider slug for the backend's start endpoint
# (POST /api/v1/credentials/oauth/{provider}/start/). All google_* types share
# the "google" provider (same consent screen, different scopes per type_key).
_OAUTH_PROVIDERS = {
    "google_email": "google",
    "google_sheets": "google",
    "google_drive": "google",
}


class CredentialsConnectOAuthInput(BaseModel):
    type_key: str = Field(
        description=(
            "OAuth credential type to connect: google_email | google_sheets | google_drive. "
            "Use the type that matches the node (GoogleSheetsNode/AIGoogleSheetsToolNode→"
            "google_sheets, GoogleDriveNode→google_drive). NON-OAuth types are rejected — "
            "create those with credentials.create."
        )
    )


@tool(
    name="credentials.connect_oauth",
    description=(
        "Mint a one-click ULASH (connect) link for an OAuth credential "
        "(google_email/google_sheets/google_drive) when the user has NONE for a Google node/tool "
        "(GoogleSheetsNode, GoogleDriveNode, AIGoogleSheetsToolNode). These credentials CANNOT be "
        "created via the API — they need a browser Google-consent. This returns an authorize_url; "
        "give it to the user to open and grant access, then call credentials.list to read the new "
        "credential_id. Use AFTER credentials.list returns nothing for a google_* type. NON-OAuth "
        "types (telegram/openai/ai_studio/ollama) are REJECTED — use credentials.create for those."
    ),
    input_model=CredentialsConnectOAuthInput,
)
async def credentials_connect_oauth(
    input: CredentialsConnectOAuthInput, ctx: ToolContext
) -> ToolResult:
    provider = _OAUTH_PROVIDERS.get(input.type_key)
    if provider is None:
        return ToolResult.error(
            f"`{input.type_key}` is not an OAuth credential type — connect_oauth only handles "
            f"OAuth types ({', '.join(sorted(_OAUTH_PROVIDERS))}). For non-OAuth types "
            "(telegram/openai/ai_studio/ollama) use credentials.create with the user's secret."
        )
    try:
        res = await _backend(ctx).oauth_start(
            provider,
            input.type_key,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("credentials.connect_oauth", e))

    authorize_url = res.get("authorize_url") if isinstance(res, dict) else None
    if not authorize_url:
        return ToolResult.error(
            "Backend did not return an authorize_url for the OAuth start — cannot build the "
            f"connect link for `{input.type_key}`."
        )
    return ToolResult.ok(
        f"Google hisobingizni ulash uchun shu havolani oching:\n{authorize_url}\n"
        "Ruxsat bergach menga 'ulandim' deb ayting — keyin credentials.list bilan "
        f"yangi `{input.type_key}` credential'ini topib node'da ishlataman.",
        data={"authorize_url": authorize_url, "type_key": input.type_key},
    )


def build_credentials_tools() -> list[Tool]:
    return [
        credentials_list,
        credentials_create,
        credentials_update,
        credentials_delete,
        credentials_connect_oauth,
    ]
