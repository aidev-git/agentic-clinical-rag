from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agentic_clinical_rag.agents import decomposer, extractor, reviewer, writer
from agentic_clinical_rag.agents.retriever import dedupe
from agentic_clinical_rag.graph import build_graph
from agentic_clinical_rag.state import Claim, Paper, ReviewVerdict, new_state


def fake_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=json.dumps(payload),
        usage_metadata={"input_tokens": 10, "output_tokens": 5},
    )


def fake_model(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(invoke=lambda _msgs: fake_response(payload))


def test_decomposer_parses_valid_json():
    with patch(
        "agentic_clinical_rag.agents.decomposer.get_chat_model",
        return_value=fake_model({"sub_queries": ["a", "b", "c"]}),
    ):
        state = new_state("What about X?")
        result = decomposer.run(state)
    assert result["sub_queries"] == ["a", "b", "c"]


def test_decomposer_falls_back_on_bad_json():
    bad_model = SimpleNamespace(
        invoke=lambda _msgs: SimpleNamespace(content="not json", usage_metadata={})
    )
    with patch("agentic_clinical_rag.agents.decomposer.get_chat_model", return_value=bad_model):
        state = new_state("What about X?")
        result = decomposer.run(state)
    assert result["sub_queries"] == ["What about X?"]


def test_retriever_dedupe_collapses_by_title():
    p1 = Paper("pmid:1", "Same Title", "abc", [], 2020, "pubmed", "url1")
    p2 = Paper("arxiv:2", "same title", "def", [], 2021, "arxiv", "url2")
    result = dedupe([p1, p2])
    assert len(result) == 1
    assert result[0].id == "pmid:1"


def test_extractor_skips_already_claimed_papers():
    paper = Paper("pmid:1", "Title", "Some abstract text", [], 2020, "pubmed", "url")
    state = new_state("q")
    state["papers"] = [paper]
    state["claims"] = [Claim("pmid:1", "Title", "existing claim")]

    calls = []

    def track_invoke(_msgs):
        calls.append(1)
        return fake_response({"claims": []})

    with patch(
        "agentic_clinical_rag.agents.extractor.get_chat_model",
        return_value=SimpleNamespace(invoke=track_invoke),
    ):
        result = extractor.run(state)

    assert calls == []  # no LLM call since the only paper already has a claim
    assert result["claims"] == []


def test_extractor_extracts_new_claims():
    paper = Paper("pmid:2", "Title2", "Some abstract text", [], 2020, "pubmed", "url")
    state = new_state("q")
    state["papers"] = [paper]
    state["claims"] = []

    with patch(
        "agentic_clinical_rag.agents.extractor.get_chat_model",
        return_value=fake_model({"claims": [{"text": "finding one", "confidence": 0.8}]}),
    ):
        result = extractor.run(state)

    assert len(result["claims"]) == 1
    assert result["claims"][0].text == "finding one"


def test_reviewer_continue_does_not_increment_refine_count():
    state = new_state("q")
    state["refine_count"] = 0
    with patch(
        "agentic_clinical_rag.agents.reviewer.get_chat_model",
        return_value=fake_model({"status": "sufficient", "reasoning": "ok"}),
    ):
        result = reviewer.run(state)
    assert result["refine_count"] == 0


def test_reviewer_needs_more_increments_refine_count():
    state = new_state("q")
    state["refine_count"] = 0
    with patch(
        "agentic_clinical_rag.agents.reviewer.get_chat_model",
        return_value=fake_model(
            {"status": "needs_more", "reasoning": "gap", "follow_up_queries": ["x"]}
        ),
    ):
        result = reviewer.run(state)
    assert result["refine_count"] == 1


def test_reviewer_route_forces_writer_at_max_cycles():
    state = new_state("q")
    state["verdict"] = ReviewVerdict(status="needs_more", reasoning="gap")
    state["refine_count"] = reviewer.MAX_REFINE_CYCLES
    assert reviewer.route(state) == "writer"


def test_reviewer_route_goes_to_retriever_below_max():
    state = new_state("q")
    state["verdict"] = ReviewVerdict(status="needs_more", reasoning="gap")
    state["refine_count"] = 0
    assert reviewer.route(state) == "retriever"


def test_writer_parses_valid_json():
    state = new_state("q")
    state["papers"] = [Paper("pmid:1", "T", "a", ["Jane Doe"], 2022, "pubmed", "url")]
    state["claims"] = [Claim("pmid:1", "T", "A finding")]
    with patch(
        "agentic_clinical_rag.agents.writer.get_chat_model",
        return_value=fake_model({"review": "## Summary\nDone.", "references": ["Doe (2022). T."]}),
    ):
        result = writer.run(state)
    assert result["review"] == "## Summary\nDone."
    assert result["citations"] == ["Doe (2022). T."]


def test_graph_builds_and_compiles():
    graph = build_graph()
    assert graph is not None


def test_full_graph_traversal_with_mocked_agents():
    paper = Paper("pmid:1", "T", "abstract text", ["Jane Doe"], 2023, "pubmed", "url")

    with (
        patch(
            "agentic_clinical_rag.agents.decomposer.get_chat_model",
            return_value=fake_model({"sub_queries": ["q1"]}),
        ),
        patch("agentic_clinical_rag.agents.retriever._fetch_all_sources", return_value=[paper]),
        patch(
            "agentic_clinical_rag.agents.retriever.get_chat_model",
            return_value=fake_model({"accept": True, "refined_query": None}),
        ),
        patch(
            "agentic_clinical_rag.agents.extractor.get_chat_model",
            return_value=fake_model({"claims": [{"text": "finding", "confidence": 0.9}]}),
        ),
        patch(
            "agentic_clinical_rag.agents.reviewer.get_chat_model",
            return_value=fake_model({"status": "sufficient", "reasoning": "ok"}),
        ),
        patch(
            "agentic_clinical_rag.agents.writer.get_chat_model",
            return_value=fake_model({"review": "## Summary\nDone.", "references": ["ref1"]}),
        ),
    ):
        graph = build_graph()
        result = graph.invoke(new_state("What is X?"))

    assert result["sub_queries"] == ["q1"]
    assert len(result["papers"]) == 1
    assert len(result["claims"]) == 1
    assert result["review"] == "## Summary\nDone."
