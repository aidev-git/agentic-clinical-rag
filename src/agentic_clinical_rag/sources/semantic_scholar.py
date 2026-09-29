"""Semantic Scholar Graph API search."""

from __future__ import annotations

import json

from agentic_clinical_rag.config import get_settings
from agentic_clinical_rag.sources._http import RateLimiter, fetch
from agentic_clinical_rag.state import Paper

_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,authors,year,externalIds,openAccessPdf"

_limiter_default = RateLimiter(0.9)
_limiter_keyed = RateLimiter(5.0)


def _limiter() -> RateLimiter:
    return _limiter_keyed if get_settings().semantic_scholar_api_key else _limiter_default


def search(query: str, max_results: int = 10) -> list[Paper]:
    settings = get_settings()
    headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else None

    body = fetch(
        "semantic_scholar",
        _ENDPOINT,
        {"query": query, "limit": str(max_results), "fields": _FIELDS},
        _limiter(),
        headers=headers,
    )

    data = json.loads(body).get("data", [])
    papers: list[Paper] = []
    for item in data:
        abstract = item.get("abstract")
        if not abstract:
            continue

        external = item.get("externalIds") or {}
        open_access = item.get("openAccessPdf") or {}
        url = open_access.get("url") or (
            f"https://www.semanticscholar.org/paper/{item['paperId']}" if item.get("paperId") else ""
        )

        papers.append(
            Paper(
                id=f"s2:{item.get('paperId')}",
                title=item.get("title") or "",
                abstract=abstract,
                authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                year=item.get("year"),
                source="semantic_scholar",
                url=url,
                doi=external.get("DOI"),
            )
        )
    return papers
