"""Provider-agnostic chat-model factory.

Swap backends purely via the LLM_PROVIDER env var — no code changes needed.
Each provider's SDK is imported lazily so installing one extra doesn't
require the others.
"""

from __future__ import annotations

import json
import re
from string import Template
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_clinical_rag.config import get_settings

_PROMPT_CACHE: dict[str, str] = {}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)

    if provider == "vertex":
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-vertexai is not installed; run "
                "`pip install agentic-clinical-rag[vertex]`"
            ) from exc
        if not settings.gcp_project_id:
            raise RuntimeError("GCP_PROJECT_ID is required when LLM_PROVIDER=vertex")
        return ChatVertexAI(
            model=settings.vertex_model,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai is not installed; run "
                "`pip install agentic-clinical-rag[gemini]`"
            ) from exc
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(model=settings.gemini_model, api_key=settings.google_api_key)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_judge_model() -> BaseChatModel:
    """A cheap model for the eval harness's LLM-as-judge metric."""
    settings = get_settings()
    if settings.judge_model:
        provider = settings.llm_provider.lower()
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=settings.judge_model, api_key=settings.anthropic_api_key)
    return get_chat_model()


def load_prompt(name: str) -> str:
    if name not in _PROMPT_CACHE:
        from importlib.resources import files

        path = files("agentic_clinical_rag.prompts").joinpath(f"{name}.txt")
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def render_prompt(name: str, **kwargs: Any) -> str:
    """Simple $-style substitution that leaves literal JSON braces alone."""
    return Template(load_prompt(name)).safe_substitute(**kwargs)


def parse_json_response(content: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in a markdown fence despite instructions not to."""
    cleaned = _FENCE_RE.sub("", content.strip())
    return json.loads(cleaned)


def token_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
