"""Pull the module catalog from backend, embed, and upsert into pgvector.

Usage:
    python -m agent.scripts.ingest_modules [--rebuild]

This is the modules sibling of `ingest_templates`. It indexes the platform's
AVAILABLE module catalog (NOT a project's enabled modules) so any agent run can
semantically search "is there a module that does X?" regardless of project.

Idempotent: each module is upserted by (project_id="_global", kind="module",
metadata.module_slug). With --rebuild we delete the whole module namespace first,
so catalog entries that were de-listed disappear too.

The text we embed is the searchable summary the agent matches a user prompt
against — name + description + node titles/categories. The node manifests
themselves are NOT embedded; the agent fetches live node details via
flow_list_node_types after the module is enabled.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import psycopg

from ..config import get_settings
from ..integrations.backend import load_modules_catalog
from ..memory import get_retriever
from ..memory.vector import _normalize_psycopg_url

logger = logging.getLogger("ingest_modules")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Shared, non-project pseudo-pool — same sentinel the search tool passes.
MODULE_PROJECT_ID = "_global"
MODULE_KIND = "module"

# GET /api/v1/modules is behind middleware.Auth, so this server-to-server ingest
# (no user session) has no bearer token and used to 401 — leaving the module
# catalog un-indexed and modules.search permanently empty. Authenticate via the
# X-Internal-Token + X-User-Id impersonation path instead: the catalog is global
# (ListModules doesn't filter by user), so a synthetic system id is enough.
_SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000000"


def _summary_text(mod: dict[str, Any]) -> str:
    """Build the searchable text for one catalog module.

    Lead with the name (weighted heaviest), then description, then the node
    types/titles the module provides (so a query like "LLM chat" matches a
    module whose only signal is a `openrouter.Chat` node).
    """
    name = (mod.get("name") or "").strip()
    slug = (mod.get("slug") or "").strip()
    desc = (mod.get("description") or "").strip()
    nodes = mod.get("nodes") or []
    node_titles: list[str] = []
    categories: set[str] = set()
    for n in nodes:
        if not isinstance(n, dict):
            continue
        title = (n.get("title") or n.get("type") or "").strip()
        if title:
            node_titles.append(title)
        cat = (n.get("category") or "").strip()
        if cat:
            categories.add(cat)
    parts = [f"Module: {name}" + (f" ({slug})" if slug and slug != name else "")]
    if desc:
        parts.append(desc)
    if categories:
        parts.append("Categories: " + ", ".join(sorted(categories)))
    if node_titles:
        parts.append("Nodes: " + ", ".join(node_titles))
    return "\n\n".join(parts)


async def _clear_existing() -> int:
    s = get_settings()
    dsn = _normalize_psycopg_url(s.database_url)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM agent_memory WHERE project_id = %s AND kind = %s",
                (MODULE_PROJECT_ID, MODULE_KIND),
            )
            rows = cur.rowcount
        await conn.commit()
    return rows


async def _delete_one(slug: str) -> None:
    s = get_settings()
    dsn = _normalize_psycopg_url(s.database_url)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM agent_memory WHERE project_id = %s AND kind = %s "
                "AND metadata->>'module_slug' = %s",
                (MODULE_PROJECT_ID, MODULE_KIND, slug),
            )
        await conn.commit()


def _metadata(mod: dict[str, Any]) -> dict[str, Any]:
    nodes = mod.get("nodes") or []
    node_types = [
        str(n.get("type"))
        for n in nodes
        if isinstance(n, dict) and n.get("type")
    ]
    has_trigger = any(
        isinstance(n, dict) and n.get("trigger") for n in nodes
    )
    return {
        "module_slug": mod.get("slug"),
        "name": mod.get("name"),
        "icon": mod.get("icon"),
        "node_types": node_types,
        "n_nodes": len(node_types),
        "has_trigger": has_trigger,
    }


async def ingest(*, rebuild: bool = False) -> dict[str, int]:
    modules = await load_modules_catalog(
        auth_claims={"user_id": _SYSTEM_USER_ID}
    )
    logger.info("Fetched %d modules from backend catalog", len(modules))
    if not modules:
        logger.warning(
            "No modules returned — empty catalog or auth/endpoint issue. Nothing to index."
        )
        return {"fetched": 0, "indexed": 0, "deleted": 0}

    deleted = 0
    if rebuild:
        deleted = await _clear_existing()
        logger.info("Rebuild: cleared %d existing module entries", deleted)

    retriever = get_retriever()
    indexed = 0
    for mod in modules:
        slug = mod.get("slug")
        if not slug:
            continue
        # Upsert mode (no rebuild): drop the prior row so the embedding refreshes
        # instead of accumulating duplicates.
        if not rebuild:
            await _delete_one(str(slug))
        await retriever.remember(
            project_id=MODULE_PROJECT_ID,
            user_id=None,
            kind=MODULE_KIND,
            content=_summary_text(mod),
            metadata=_metadata(mod),
        )
        indexed += 1
    logger.info(
        "Done. fetched=%d indexed=%d deleted=%d", len(modules), indexed, deleted
    )
    return {"fetched": len(modules), "indexed": indexed, "deleted": deleted}


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    asyncio.run(ingest(rebuild=rebuild))


if __name__ == "__main__":
    main()
