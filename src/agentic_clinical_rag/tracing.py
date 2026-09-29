"""Optional LangSmith tracing. A complete no-op when no key is configured."""

from __future__ import annotations

import os

from agentic_clinical_rag.config import get_settings


def enable_tracing() -> None:
    settings = get_settings()
    if not settings.langchain_api_key:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
