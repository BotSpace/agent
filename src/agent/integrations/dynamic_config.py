"""Dynamic LLM config resolver.

Backend public-config API'dan provider/model/region oladi va env settings
ustidan ustunlik bilan qo'llaydi. Har run boshida bir marta chaqiriladi;
natija shu run ichida barqaror (bir xil).

Public config kalitlari (backend tomonidan belgilangan):
  ai_agent.provider  — e.g. "bedrock" | "openai" | "ollama" | "gemini" | "anthropic"
  ai_agent.model     — e.g. "qwen.qwen3-coder-480b-a35b-v1:0"
  ai_agent.region    — e.g. "us-west-2"

Ustunlik tartibi (har kalit uchun):
  1. Backend config qiymati — mavjud va bo'sh bo'lmasa
  2. Env-based settings (settings.llm_provider, settings.model_*, settings.aws_region)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import get_settings
from .backend import fetch_public_config

if TYPE_CHECKING:
    from ..llm.router import ModelRoles

logger = logging.getLogger(__name__)

_PUBLIC_KEY_PROVIDER = "ai_agent.provider"
_PUBLIC_KEY_MODEL = "ai_agent.model"
_PUBLIC_KEY_REGION = "ai_agent.region"


def _nonempty(value: object) -> str | None:
    """Return the value as str if it is a non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


async def effective_provider() -> str:
    """Return the active LLM provider string, backend config over env."""
    cfg = await fetch_public_config()
    from_backend = _nonempty(cfg.get(_PUBLIC_KEY_PROVIDER))
    if from_backend:
        logger.info("LLM provider from backend config: %s", from_backend)
        return from_backend
    s = get_settings()
    return s.llm_provider


async def effective_region() -> str:
    """Return the active AWS region, backend config over env."""
    cfg = await fetch_public_config()
    from_backend = _nonempty(cfg.get(_PUBLIC_KEY_REGION))
    if from_backend:
        logger.info("LLM region from backend config: %s", from_backend)
        return from_backend
    s = get_settings()
    return s.aws_region


async def effective_model_roles() -> ModelRoles:
    """Build ModelRoles with backend config model applied to planner/executor/critic.

    When ``ai_agent.model`` is set in backend config it overrides all three
    inference roles (planner, executor, critic).  ``embedding`` always comes
    from env — it is not controlled remotely.
    """
    from ..llm.router import ModelRoles

    cfg = await fetch_public_config()
    s = get_settings()

    model_override = _nonempty(cfg.get(_PUBLIC_KEY_MODEL))
    if model_override:
        logger.info("LLM model from backend config: %s (all roles)", model_override)
        return ModelRoles(
            planner=model_override,
            executor=model_override,
            critic=model_override,
            embedding=s.model_embedding,
        )

    return ModelRoles.from_settings(s)


async def build_effective_client():
    """Build an LLMClient with effective provider + region (for bedrock).

    Uses ``build_client`` from the router but passes the dynamically resolved
    provider and — for bedrock — the effective region instead of the static
    ``settings.aws_region``.
    """
    from ..llm.bedrock import BedrockClient
    from ..llm.router import build_client

    provider = await effective_provider()
    s = get_settings()

    if provider == "bedrock":
        region = await effective_region()
        logger.info("Building BedrockClient with effective region=%s", region)
        return BedrockClient(
            region=region,
            access_key_id=s.aws_access_key_id,
            secret_access_key=s.aws_secret_access_key,
        )

    return build_client(provider, s)
