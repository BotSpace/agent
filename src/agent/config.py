from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "bedrock", "openrouter"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── server ──────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="AGENT_HOST")
    port: int = Field(default=8100, alias="AGENT_PORT")
    log_level: str = Field(default="INFO", alias="AGENT_LOG_LEVEL")
    cors_origins: str = Field(
        default="http://localhost:3010,http://localhost:3001",
        alias="AGENT_CORS_ORIGINS",
    )

    # ── llm ─────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default="openai", alias="AGENT_LLM_PROVIDER")
    model_planner: str = Field(default="gpt-5.2", alias="AGENT_MODEL_PLANNER")
    model_executor: str = Field(default="gpt-5.2", alias="AGENT_MODEL_EXECUTOR")
    model_critic: str = Field(default="gpt-5.2", alias="AGENT_MODEL_CRITIC")
    model_chat: str = Field(default="gpt-4o-mini", alias="AGENT_MODEL_CHAT")
    model_embedding: str = Field(
        default="text-embedding-3-small", alias="AGENT_MODEL_EMBEDDING"
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_site_url: str | None = Field(
        default="https://botspace.uz", alias="OPENROUTER_SITE_URL",
        description="HTTP-Referer header for OpenRouter rankings/attribution.",
    )
    openrouter_app_name: str | None = Field(
        default="Botmother Agent", alias="OPENROUTER_APP_NAME",
        description="X-Title header for OpenRouter dashboard display.",
    )
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # ── budgets ─────────────────────────────────────────────────────────
    max_input_tokens: int = Field(default=200_000, alias="AGENT_MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(default=16_000, alias="AGENT_MAX_OUTPUT_TOKENS")
    max_tool_calls: int = Field(default=80, alias="AGENT_MAX_TOOL_CALLS")
    max_wall_seconds: int = Field(default=900, alias="AGENT_MAX_WALL_SECONDS")

    # ── persistence ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg://agent:agent@localhost:5433/agent",
        alias="AGENT_DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6380/0", alias="AGENT_REDIS_URL")

    # ── external services ───────────────────────────────────────────────
    backend_base_url: str = Field(
        default="http://localhost:8000", alias="BACKEND_BASE_URL"
    )
    backend_internal_token: str | None = Field(
        default=None, alias="BACKEND_INTERNAL_TOKEN"
    )
    mongo_base_url: str = Field(
        default="mongodb://localhost:27017", alias="MONGO_BASE_URL"
    )
    mongo_db_prefix: str = Field(default="project_", alias="MONGO_DB_PREFIX")
    engine_base_url: str = Field(
        default="http://localhost:9100", alias="ENGINE_BASE_URL"
    )

    # ── auth ────────────────────────────────────────────────────────────
    session_secret: str = Field(
        default="change-me-to-a-32-char-secret-string", alias="SESSION_SECRET"
    )
    jwt_algorithm: str = Field(default="RS256", alias="JWT_ALGORITHM")
    jwt_public_key_path: str = Field(
        default="keys/public.pem", alias="JWT_PUBLIC_KEY_PATH"
    )
    internal_api_token: str | None = Field(
        default=None, alias="AGENT_INTERNAL_API_TOKEN"
    )

    # ── memory / rag ────────────────────────────────────────────────────
    rag_enabled: bool = Field(default=True, alias="AGENT_RAG_ENABLED")
    rag_top_k: int = Field(default=6, alias="AGENT_RAG_TOP_K")

    # ── observability ───────────────────────────────────────────────────
    otel_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="botmother-agent", alias="LANGSMITH_PROJECT"
    )

    # ── derived ─────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
