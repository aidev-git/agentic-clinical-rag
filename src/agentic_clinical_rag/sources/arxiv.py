"""arXiv search via the public Atom-feed API. Rate-limited per arXiv's
usage guidelines (roughly one request per 3 seconds)."""

from __future__ import annotations

import re
from xml.etree import ElementTree

from agentic_clinical_rag.sources._http import RateLimiter, fetch
from agentic_clinical_rag.state import Paper

_ENDPOINT = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_limiter = RateLimiter(1.0 / 3.0)


def _strip_version(entry_id: str) -> str:
    slug = entry_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", slug)


def _parse_entry(entry: ElementTree.Element) -> Paper | None:
    entry_id = entry.findtext(f"{_ATOM}id") or ""
    title = (entry.findtext(f"{_ATOM}title") or "").strip()
    abstract = (entry.findtext(f"{_ATOM}summary") or "").strip()
    if not title or not abstract:
        return None

    authors = [
        (a.findtext(f"{_ATOM}name") or "").strip()
        for a in entry.findall(f"{_ATOM}author")
        if (a.findtext(f"{_ATOM}name") or "").strip()
    ]

    published = entry.findtext(f"{_ATOM}published") or ""
    year_match = re.match(r"(\d{4})", published)
    year = int(year_match.group(1)) if year_match else None

    doi = entry.findtext(f"{_ARXIV_NS}doi")
    arxiv_id = _strip_version(entry_id)

    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        source="arxiv",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        doi=doi,
    )


def search(query: str, max_results: int = 10) -> list[Paper]:
    body = fetch(
        "arxiv",
        _ENDPOINT,
        {"search_query": f"all:{query}", "start": "0", "max_results": str(max_results)},
        _limiter,
    )

    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return []

    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        try:
            paper = _parse_entry(entry)
        except ElementTree.ParseError:
            paper = None
        if paper:
            papers.append(paper)
    return papers
