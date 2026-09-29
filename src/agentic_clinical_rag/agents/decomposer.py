"""Hierarchical delegation: break the question into focused sub-queries."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_chat_model, parse_json_response, render_prompt, token_usage
from agentic_clinical_rag.state import ResearchState


def run(state: ResearchState) -> dict:
    question = state["question"]
    prompt = render_prompt("decomposer", question=question)
    response = get_chat_model().invoke([HumanMessage(content=prompt)])
    in_tok, out_tok = token_usage(response)

    try:
        data = parse_json_response(response.content)
        sub_queries = [q for q in data.get("sub_queries", []) if isinstance(q, str) and q.strip()]
        if not sub_queries:
            raise ValueError("empty sub_queries")
    except Exception:
        sub_queries = [question]

    if state.get("debug"):
        print(f"[decomposer] {len(sub_queries)} sub-queries: {sub_queries}")

    return {
        "sub_queries": sub_queries,
        "verdict": None,
        "refine_count": 0,
        "review": None,
        "citations": [],
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
