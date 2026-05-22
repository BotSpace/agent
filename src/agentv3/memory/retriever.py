"""Memory retriever — embedder + vector_store hovering above raw layer."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..config import get_settings
from .embedder import Embedder, HashEmbedder, get_embedder
from .vector import VectorStore, get_vector_store


class MemoryRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def remember(
        self,
        *,
        project_id: str,
        user_id: str | None,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        # Hash embedder fallback may emit non-matching dim — skip persist then.
        if isinstance(self._embedder, HashEmbedder) and self._embedder.dim != self._store._dim:
            return 0
        [vec] = await self._embedder.embed([content])
        return await self._store.upsert(
            project_id=project_id,
            user_id=user_id,
            kind=kind,
            content=content,
            metadata=metadata,
            embedding=vec,
        )

    async def recall(
        self,
        *,
        project_id: str,
        query: str,
        kind: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(self._embedder, HashEmbedder) and self._embedder.dim != self._store._dim:
            return []
        s = get_settings()
        k = top_k or s.rag_top_k
        [vec] = await self._embedder.embed([query])
        return await self._store.search(
            project_id=project_id,
            embedding=vec,
            kind=kind,
            top_k=k,
        )


@lru_cache(maxsize=1)
def get_retriever() -> MemoryRetriever:
    return MemoryRetriever(embedder=get_embedder(), store=get_vector_store())
