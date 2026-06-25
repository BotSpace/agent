"""Embeddings — provider-agnostic interface (OpenAI default, fallback hash)."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import Protocol

from ..config import get_settings


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536-dim) yoki -large (3072)."""

    def __init__(self, api_key: str | None, model: str = "text-embedding-3-small") -> None:
        from openai import AsyncOpenAI

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAIEmbedder")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self.dim = 3072 if "large" in model else 1536

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


class HashEmbedder:
    """Deterministic fallback — testing yoki API yo'q paytda.

    Sifati past, lekin schema/relevance test'lariga yetadi.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = sha256(text.encode("utf-8")).digest()
            vec = [b / 255.0 - 0.5 for b in digest]
            # repeat to fill dim
            while len(vec) < self.dim:
                vec.extend(vec[: min(len(vec), self.dim - len(vec))])
            out.append(vec[: self.dim])
        return out


@lru_cache(maxsize=2)
def get_embedder() -> Embedder:
    s = get_settings()
    if s.openai_api_key:
        return OpenAIEmbedder(api_key=s.openai_api_key, model=s.model_embedding)
    return HashEmbedder()
