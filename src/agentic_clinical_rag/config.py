"""Environment-driven settings. Never hardcode secrets here."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"  # "anthropic" | "vertex" | "gemini"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    gcp_project_id: str | None = None
    gcp_location: str = "us-central1"
    vertex_model: str = "gemini-2.0-flash"

    google_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    judge_model: str | None = None

    ncbi_api_key: str | None = None
    semantic_scholar_api_key: str | None = None

    cache_dir: str = ".cache"
    cache_ttl_seconds: int = 60 * 60 * 24 * 7

    langchain_api_key: str | None = None
    langchain_project: str = "agentic-clinical-rag"

    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
