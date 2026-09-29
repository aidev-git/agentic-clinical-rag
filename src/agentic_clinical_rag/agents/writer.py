"""Final synthesis: turn gathered claims into a cited markdown review."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_chat_model, parse_json_response, render_prompt, token_usage
from agentic_clinical_rag.state import Paper, ResearchState

_REFERENCES_HEADING = re.compile(r"^#+\s*references", re.IGNORECASE)


def _first_author_surname(paper: Paper) -> str:
    if not paper.authors:
        return "Unknown"
    return paper.authors[0].split()[-1] if paper.authors[0] else "Unknown"


def _claim_summaries(state: ResearchState) -> str:
    papers_by_id = {p.id: p for p in state.get("papers", [])}
    lines = []
    for claim in state.get("claims", []):
        paper = papers_by_id.get(claim.paper_id)
        cite = f"{_first_author_surname(paper)}, {paper.year}" if paper else claim.paper_title
        lines.append(f"- {claim.text} ({cite})")
    return "\n".join(lines) or "(no claims gathered)"


def _reference_list(state: ResearchState) -> str:
    lines = []
    for paper in state.get("papers", []):
        authors = ", ".join(paper.authors[:3]) or "Unknown authors"
        lines.append(f"- {authors} ({paper.year}). {paper.title}. {paper.source}. {paper.url}")
    return "\n".join(lines) or "(no papers)"


def _fallback_review(state: ResearchState) -> tuple[str, list[str]]:
    n_papers = len(state.get("papers", []))
    n_claims = len(state.get("claims", []))
    return (
        f"## Summary\nUnable to synthesize a full review automatically. "
        f"Gathered {n_papers} papers and {n_claims} claims for further manual review.",
        [],
    )


def run(state: ResearchState) -> dict:
    prompt = render_prompt(
        "writer",
        question=state["question"],
        claim_summaries=_claim_summaries(state),
        reference_list=_reference_list(state),
    )

    try:
        response = get_chat_model().invoke([HumanMessage(content=prompt)])
        in_tok, out_tok = token_usage(response)
        content = response.content

        try:
            data = parse_json_response(content)
            review, references = data.get("review", ""), list(data.get("references", []))
        except Exception:
            # Some responses may already be plain markdown instead of JSON.
            review = content
            references = []
            for line in content.splitlines():
                if _REFERENCES_HEADING.match(line):
                    idx = content.splitlines().index(line)
                    references = [
                        l.strip("- ").strip()
                        for l in content.splitlines()[idx + 1 :]
                        if l.strip()
                    ]
                    break

        if not review.strip():
            review, references = _fallback_review(state)

    except Exception as exc:
        print(f"[writer] LLM call failed: {exc}")
        review, references = _fallback_review(state)
        in_tok = out_tok = 0

    if state.get("debug"):
        print(f"[writer] review length={len(review)} references={len(references)}")

    return {"review": review, "citations": references, "input_tokens": in_tok, "output_tokens": out_tok}
