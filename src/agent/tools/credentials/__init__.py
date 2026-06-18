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

Read-only by design — the agent cannot CREATE/edit credentials (those need
secret entry / an OAuth consent popup the user performs in the panel/dashboard).
If the user has no matching credential, tell them to create one there first.
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
            f"No credentials{filt} found for this user. Credentials are owner-global "
            "and the user must create/connect them in the panel/dashboard "
            "(node→type_key: " + _NODE_TYPE_KEYS + "). Ask the user to add one, then "
            "re-list and use its id as data.credential_id.",
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


def build_credentials_tools() -> list[Tool]:
    return [credentials_list]
