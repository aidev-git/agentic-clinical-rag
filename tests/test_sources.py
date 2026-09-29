from __future__ import annotations

import pytest

from agentic_clinical_rag.sources import arxiv, pubmed, semantic_scholar


def test_sources_are_callable():
    assert callable(pubmed.search)
    assert callable(semantic_scholar.search)
    assert callable(arxiv.search)


def test_pubmed_parser_handles_malformed_xml_gracefully():
    from xml.etree import ElementTree

    empty_root = ElementTree.fromstring("<PubmedArticleSet></PubmedArticleSet>")
    assert empty_root.findall(".//PubmedArticle") == []


def test_pubmed_year_from_text():
    assert pubmed._year_from_text("2019 Jan-Feb") == 2019
    assert pubmed._year_from_text("no year here") is None


def test_arxiv_strip_version():
    assert arxiv._strip_version("http://arxiv.org/abs/2301.01234v2") == "2301.01234"
    assert arxiv._strip_version("http://arxiv.org/abs/2301.01234") == "2301.01234"


def test_arxiv_parse_entry_skips_missing_title_or_abstract():
    from xml.etree import ElementTree

    entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <id>http://arxiv.org/abs/1234.5678v1</id>
      <summary></summary>
    </entry>
    """
    entry = ElementTree.fromstring(entry_xml)
    assert arxiv._parse_entry(entry) is None


@pytest.mark.integration
def test_pubmed_live_search():
    results = pubmed.search("aspirin cardiovascular", max_results=3)
    assert len(results) > 0
    assert all(r.title for r in results)


@pytest.mark.integration
def test_semantic_scholar_live_search():
    results = semantic_scholar.search("GLP-1 agonist weight loss", max_results=3)
    assert len(results) > 0


@pytest.mark.integration
def test_arxiv_live_search():
    results = arxiv.search("large language models clinical", max_results=3)
    assert len(results) > 0
