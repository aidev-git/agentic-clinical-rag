"""ReAct pattern: search, judge adequacy, optionally refine the query, repeat."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_chat_model, parse_json_response, render_prompt, token_usage
from agentic_clinical_rag.sources import arxiv, pubmed, semantic_scholar
from agentic_clinical_rag.state import Paper, ResearchState

MAX_ITERATIONS = 2
_PER_SOURCE_RESULTS = 5


def _fetch_all_sources(query: str) -> list[Paper]:
    papers: list[Paper] = []
    for source in (pubmed, semantic_scholar, arxiv):
        try:
            papers.extend(source.search(query, max_results=_PER_SOURCE_RESULTS))
        except Exception as exc:  # one source failing shouldn't sink the others
            print(f"[retriever] {source.__name__} failed: {exc}")
    return papers


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def dedupe(papers: list[Paper]) -> list[Paper]:
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        title_key = _normalize_title(paper.title)
        if paper.id in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(paper.id)
        seen_titles.add(title_key)
        result.append(paper)
    return result


def _judge_batch(query: str, papers: list[Paper]) -> dict:
    summaries = "\n".join(f"- {p.title} ({p.year}): {p.abstract[:160]}" for p in papers[:8])
    prompt = render_prompt("retriever", query=query, paper_summaries=summaries or "(none found)")
    response = get_chat_model().invoke([HumanMessage(content=prompt)])
    in_tok, out_tok = token_usage(response)
    try:
        return {**parse_json_response(response.content), "input_tokens": in_tok, "output_tokens": out_tok}
    except Exception:
        return {"accept": True, "refined_query": None, "input_tokens": in_tok, "output_tokens": out_tok}


def run(state: ResearchState) -> dict:
    verdict = state.get("verdict")
    queries = verdict.follow_up_queries if verdict else state.get("sub_queries", [])
    queries = queries or [state["question"]]

    existing_ids = {p.id for p in state.get("papers", [])}
    existing_titles = {_normalize_title(p.title) for p in state.get("papers", [])}

    collected: list[Paper] = []
    total_in = total_out = 0

    for query in queries:
        current_query = query
        for iteration in range(MAX_ITERATIONS):
            batch = _fetch_all_sources(current_query)
            is_last = iteration == MAX_ITERATIONS - 1
            if is_last:
                collected.extend(batch)
                break

            judged = _judge_batch(current_query, batch)
            total_in += judged["input_tokens"]
            total_out += judged["output_tokens"]

            if state.get("debug"):
                print(f"[retriever] '{current_query}' -> accept={judged.get('accept')}")

            collected.extend(batch)
            refined = judged.get("refined_query")
            if judged.get("accept") or not refined or refined.strip() == current_query.strip():
                break
            current_query = refined

    unique = dedupe(collected)
    fresh = [p for p in unique if p.id not in existing_ids and _normalize_title(p.title) not in existing_titles]

    return {"papers": fresh, "input_tokens": total_in, "output_tokens": total_out}
