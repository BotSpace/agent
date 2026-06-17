"""pgvector store — long-term semantic memory.

Schema (psycopg orqali bootstrap qilinadi):

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS agent_memory (
        id          BIGSERIAL PRIMARY KEY,
        project_id  TEXT NOT NULL,
        user_id     TEXT,
        kind        TEXT NOT NULL,         -- 'flow' | 'message' | 'doc' | 'project_summary'
        content     TEXT NOT NULL,
        metadata    JSONB NOT NULL DEFAULT '{}',
        embedding   VECTOR(1536),
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS agent_memory_embed_idx
        ON agent_memory USING ivfflat (embedding vector_cosine_ops);

`dim` o'zgarsa, migration kerak.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import psycopg

from ..config import get_settings


_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL,
    user_id     TEXT,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   VECTOR(%(dim)s),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS agent_memory_proj_idx ON agent_memory (project_id);
CREATE INDEX IF NOT EXISTS agent_memory_kind_idx ON agent_memory (kind);
"""


from ..config import normalize_pg_url as _normalize_psycopg_url


class VectorStore:
    def __init__(self, dsn: str, dim: int) -> None:
        self._dsn = _normalize_psycopg_url(dsn)
        self._dim = dim
        self._bootstrapped = False

    async def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA % {"dim": self._dim})
            await conn.commit()
        self._bootstrapped = True

    async def upsert(
        self,
        *,
        project_id: str,
        user_id: str | None,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None,
        embedding: list[float],
    ) -> int:
        await self.bootstrap()
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: got {len(embedding)}, expected {self._dim}"
            )
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_memory
                      (project_id, user_id, kind, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        user_id,
                        kind,
                        content,
                        json.dumps(metadata or {}),
                        _vec_literal(embedding),
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
                return int(row[0]) if row else 0

    async def search(
        self,
        *,
        project_id: str,
        embedding: list[float],
        kind: str | None = None,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        await self.bootstrap()
        where = ["project_id = %s"]
        params: list[Any] = [project_id]
        if kind:
            where.append("kind = %s")
            params.append(kind)
        params.append(_vec_literal(embedding))
        params.append(top_k)
        sql = f"""
            SELECT id, kind, content, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM agent_memory
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s
            LIMIT %s
        """  # noqa: S608
        # rewrite param order: where..., search_vec, search_vec, top_k
        # Easier to rebuild query with placeholders:
        sql = f"""
            SELECT id, kind, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM agent_memory
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """  # noqa: S608
        params = []
        params.append(_vec_literal(embedding))  # similarity score
        params.append(project_id)
        if kind:
            params.append(kind)
        params.append(_vec_literal(embedding))  # ORDER BY
        params.append(top_k)

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "kind": r[1],
                        "content": r[2],
                        "metadata": r[3] or {},
                        "similarity": float(r[4]),
                    }
                    for r in rows
                ]


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    s = get_settings()
    # embedding dim — provider'ga qarab; default 1536 (OpenAI small)
    dim = 3072 if "large" in s.model_embedding else 1536
    return VectorStore(s.database_url, dim=dim)
