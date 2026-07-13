"""Pull templates from backend, embed, and upsert into pgvector.

Usage:
    python -m agent.scripts.ingest_templates [--rebuild]

Idempotent: each template is upserted by (project_id="_global",
kind="template", metadata.template_id). With --rebuild we delete the
whole template namespace first, useful when the backend catalog has
been pruned and we want stale entries gone too.

The text we embed is the searchable summary the agent will match a
user prompt against — name + categories + features + descriptions.
The flow itself is NOT embedded (LLM doesn't need to see it during
search; we fetch it separately when applying a match).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import psycopg

from ..config import get_settings
from ..integrations.backend import get_backend_client
from ..memory import get_retriever
from ..memory.vector import _normalize_psycopg_url

logger = logging.getLogger("ingest_templates")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# All template embeddings live in a shared, non-project pseudo-pool so
# any agent run can search them regardless of which project the user is
# working in. Code that searches just passes this same sentinel.
TEMPLATE_PROJECT_ID = "_global"
TEMPLATE_KIND = "template"


def _summary_text(tpl: dict[str, Any]) -> str:
    """Build the searchable text representation of a template.

    Order matters: lead with what we want the embedding to weight
    heaviest (name + categories), then the descriptions, then features.
    """
    name = (tpl.get("name") or "").strip()
    short = (tpl.get("short_description") or "").strip()
    desc = (tpl.get("description") or "").strip()
    cats = ", ".join(c.get("name", "") for c in (tpl.get("categories") or []) if c.get("name"))
    feats = ", ".join(f.get("name", "") for f in (tpl.get("features") or []) if f.get("name"))
    parts = [f"Template: {name}"]
    if cats:
        parts.append(f"Categories: {cats}")
    if short:
        parts.append(short)
    if desc and desc != short:
        parts.append(desc)
    if feats:
        parts.append(f"Features: {feats}")
    return "\n\n".join(parts)


async def _clear_existing() -> int:
    s = get_settings()
    dsn = _normalize_psycopg_url(s.database_url)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM agent_memory WHERE project_id = %s AND kind = %s",
                (TEMPLATE_PROJECT_ID, TEMPLATE_KIND),
            )
            rows = cur.rowcount
        await conn.commit()
    return rows


async def _delete_one(template_id: str | int) -> None:
    s = get_settings()
    dsn = _normalize_psycopg_url(s.database_url)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM agent_memory WHERE project_id = %s AND kind = %s "
                "AND metadata->>'template_id' = %s",
                (TEMPLATE_PROJECT_ID, TEMPLATE_KIND, str(template_id)),
            )
        await conn.commit()


async def ingest(*, rebuild: bool = False) -> dict[str, int]:
    backend = get_backend_client()
    templates = await backend.list_internal_templates()
    logger.info("Fetched %d templates from backend", len(templates))
    if not templates:
        logger.warning("No templates returned — check BACKEND_INTERNAL_TOKEN / endpoint.")
        return {"fetched": 0, "indexed": 0, "deleted": 0}

    deleted = 0
    if rebuild:
        deleted = await _clear_existing()
        logger.info("Rebuild: cleared %d existing template entries", deleted)

    retriever = get_retriever()
    indexed = 0
    for tpl in templates:
        tid = tpl.get("id")
        if tid is None:
            continue
        content = _summary_text(tpl)
        # In upsert mode (no rebuild): drop the prior row for this template
        # so the embedding refreshes instead of accumulating duplicates.
        if not rebuild:
            await _delete_one(tid)
        # Only the env KEYS go into metadata (not descriptions) to keep the
        # vector row small; descriptions are fetched on apply_template. Skip
        # RELEASE_BOT_TOKEN — it has its own release flow, not a user env.
        env_keys = [
            str(e.get("key")).strip()
            for e in (tpl.get("required_envs") or [])
            if isinstance(e, dict) and str(e.get("key") or "").strip()
            and str(e.get("key")).strip() != "RELEASE_BOT_TOKEN"
        ]
        metadata = {
            # Backend template ids are UUID strings — store as-is. The old
            # int() cast crashed on UUIDs (and left stale integer ids that
            # apply_template's UUID/slug lookup couldn't resolve).
            "template_id": str(tid),
            "slug": tpl.get("slug"),
            "name": tpl.get("name"),
            "categories": [c.get("name") for c in (tpl.get("categories") or [])],
            "n_nodes": len((tpl.get("flow") or {}).get("nodes") or []),
            "n_edges": len((tpl.get("flow") or {}).get("edges") or []),
        }
        if env_keys:
            metadata["required_envs"] = env_keys
        await retriever.remember(
            project_id=TEMPLATE_PROJECT_ID,
            user_id=None,
            kind=TEMPLATE_KIND,
            content=content,
            metadata=metadata,
        )
        indexed += 1
        if indexed % 10 == 0:
            logger.info("Indexed %d / %d", indexed, len(templates))
    logger.info("Done. fetched=%d indexed=%d deleted=%d", len(templates), indexed, deleted)
    return {"fetched": len(templates), "indexed": indexed, "deleted": deleted}


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    asyncio.run(ingest(rebuild=rebuild))


if __name__ == "__main__":
    main()
