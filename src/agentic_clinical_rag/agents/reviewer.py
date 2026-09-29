"""Self-reflection pattern: judge evidence coverage, optionally request another pass."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_chat_model, parse_json_response, render_prompt, token_usage
from agentic_clinical_rag.state import ReviewVerdict, ResearchState

MAX_REFINE_CYCLES = 2


def _summarize_claims(state: ResearchState) -> str:
    lines = []
    for claim in state.get("claims", [])[:30]:
        lines.append(f"- ({claim.paper_title}) {claim.text[:160]}")
    return "\n".join(lines) or "(no claims gathered yet)"


def run(state: ResearchState) -> dict:
    prompt = render_prompt(
        "reviewer", question=state["question"], claim_summaries=_summarize_claims(state)
    )
    try:
        response = get_chat_model().invoke([HumanMessage(content=prompt)])
        in_tok, out_tok = token_usage(response)
        data = parse_json_response(response.content)
        verdict = ReviewVerdict(
            status=data.get("status", "sufficient"),
            reasoning=data.get("reasoning", ""),
            gaps=list(data.get("gaps", [])),
            follow_up_queries=list(data.get("follow_up_queries", [])),
        )
    except Exception:
        in_tok = out_tok = 0
        verdict = ReviewVerdict(status="sufficient", reasoning="fallback: could not evaluate")

    refine_count = state.get("refine_count", 0)
    if verdict.status == "needs_more":
        refine_count += 1

    if state.get("debug"):
        print(f"[reviewer] status={verdict.status} gaps={verdict.gaps}")

    return {
        "verdict": verdict,
        "refine_count": refine_count,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def route(state: ResearchState) -> str:
    verdict = state.get("verdict")
    if verdict and verdict.status == "needs_more" and state.get("refine_count", 0) < MAX_REFINE_CYCLES:
        return "retriever"
    return "writer"
