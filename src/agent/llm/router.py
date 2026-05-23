"""LLM router — provider tanlash va cache."""

from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings
from .anthropic import AnthropicClient
from .base import LLMClient, LLMError
from .bedrock import BedrockClient
from .openai import OpenAIClient


def build_client(provider: str, settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    if provider == "anthropic":
        return AnthropicClient(api_key=s.anthropic_api_key)
    if provider == "openai":
        return OpenAIClient(api_key=s.openai_api_key)
    if provider == "bedrock":
        return BedrockClient(
            region=s.aws_region,
            access_key_id=s.aws_access_key_id,
            secret_access_key=s.aws_secret_access_key,
        )
    raise LLMError(f"Unknown LLM provider: {provider}", provider=provider)


@lru_cache(maxsize=4)
def get_client(provider: str | None = None) -> LLMClient:
    s = get_settings()
    return build_client(provider or s.llm_provider, s)


class ModelRoles:
    """Bir LLM run-da rol bo'yicha turli modellar ishlatilishi mumkin.

    Misol: planner — opus, executor — sonnet, critic — haiku.
    """

    def __init__(self, planner: str, executor: str, critic: str, embedding: str) -> None:
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.embedding = embedding

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ModelRoles":
        s = settings or get_settings()
        return cls(
            planner=s.model_planner,
            executor=s.model_executor,
            critic=s.model_critic,
            embedding=s.model_embedding,
        )


def reset_client_cache() -> None:
    """Tests / config reload uchun."""
    get_client.cache_clear()
