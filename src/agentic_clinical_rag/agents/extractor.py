"""Tool-use pattern: pull structured claims out of each new paper's abstract."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_chat_model, parse_json_response, render_prompt, token_usage
from agentic_clinical_rag.state import Claim, ResearchState

MAX_PAPERS_PER_CALL = 8


def run(state: ResearchState) -> dict:
    already_claimed = {c.paper_id for c in state.get("claims", [])}
    candidates = [
        p for p in state.get("papers", []) if p.abstract and p.id not in already_claimed
    ]
    candidates.sort(key=lambda p: p.year or 0, reverse=True)
    candidates = candidates[:MAX_PAPERS_PER_CALL]

    model = get_chat_model()
    new_claims: list[Claim] = []
    total_in = total_out = 0

    for paper in candidates:
        prompt = render_prompt(
            "extractor",
            title=paper.title,
            authors=", ".join(paper.authors) or "Unknown",
            year=paper.year or "Unknown",
            abstract=paper.abstract,
        )
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            in_tok, out_tok = token_usage(response)
            total_in += in_tok
            total_out += out_tok
            data = parse_json_response(response.content)
            for item in data.get("claims", []):
                if isinstance(item, str):
                    text, confidence = item, 1.0
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    confidence = float(item.get("confidence", 1.0) or 1.0)
                else:
                    continue
                if text.strip():
                    new_claims.append(Claim(paper.id, paper.title, text.strip(), confidence))
        except Exception as exc:
            print(f"[extractor] failed on '{paper.title}': {exc}")

    if state.get("debug"):
        print(f"[extractor] extracted {len(new_claims)} claims from {len(candidates)} papers")

    return {"claims": new_claims, "input_tokens": total_in, "output_tokens": total_out}
