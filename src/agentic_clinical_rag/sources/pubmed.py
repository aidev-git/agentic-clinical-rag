"""PubMed search via NCBI E-utilities (esearch -> efetch)."""

from __future__ import annotations

import re
from xml.etree import ElementTree

from agentic_clinical_rag.config import get_settings
from agentic_clinical_rag.sources._http import RateLimiter, fetch
from agentic_clinical_rag.state import Paper

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_limiter_default = RateLimiter(3.0)
_limiter_keyed = RateLimiter(10.0)


def _limiter() -> RateLimiter:
    return _limiter_keyed if get_settings().ncbi_api_key else _limiter_default


def _year_from_text(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None


def _parse_article(article: ElementTree.Element) -> Paper | None:
    try:
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        if pmid_el is None or title_el is None or not (title_el.text or "").strip():
            return None

        abstract_parts = [
            "".join(node.itertext()).strip()
            for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(p for p in abstract_parts if p)
        if not abstract:
            return None

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName") or ""
            first = author.findtext("ForeName") or ""
            name = f"{first} {last}".strip()
            if name:
                authors.append(name)

        year_el = article.find(".//PubDate/Year")
        year = int(year_el.text) if year_el is not None and year_el.text else None
        if year is None:
            medline_date = article.findtext(".//PubDate/MedlineDate") or ""
            year = _year_from_text(medline_date)

        doi = None
        for id_el in article.findall(".//ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text

        pmid = pmid_el.text or ""
        return Paper(
            id=f"pmid:{pmid}",
            title=(title_el.text or "").strip(),
            abstract=abstract,
            authors=authors,
            year=year,
            source="pubmed",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            doi=doi,
        )
    except ElementTree.ParseError:
        return None


def search(query: str, max_results: int = 10) -> list[Paper]:
    settings = get_settings()
    common: dict[str, str] = {"db": "pubmed"}
    if settings.ncbi_api_key:
        common["api_key"] = settings.ncbi_api_key

    search_body = fetch(
        "pubmed",
        f"{_EUTILS}/esearch.fcgi",
        {**common, "term": query, "retmax": str(max_results), "retmode": "json"},
        _limiter(),
    )

    import json as _json

    ids = _json.loads(search_body).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch_body = fetch(
        "pubmed",
        f"{_EUTILS}/efetch.fcgi",
        {**common, "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"},
        _limiter(),
    )

    try:
        root = ElementTree.fromstring(fetch_body)
    except ElementTree.ParseError:
        return []

    papers = [p for article in root.findall(".//PubmedArticle") if (p := _parse_article(article))]
    return papers
