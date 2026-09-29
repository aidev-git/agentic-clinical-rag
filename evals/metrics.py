"""Scoring functions for the eval harness."""

from __future__ import annotations

import datetime as dt
import re

from langchain_core.messages import HumanMessage

from agentic_clinical_rag.llm import get_judge_model, parse_json_response, token_usage
from agentic_clinical_rag.state import Paper

_ID_PREFIX_PATTERNS = {
    "pubmed": re.compile(r"^pmid:\d+$"),
    "semantic_scholar": re.compile(r"^s2:.+$"),
    "arxiv": re.compile(r"^arxiv:.+$"),
}

# Snapshot pricing (USD per 1M tokens); update as needed.
_PRICING = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_PRICING = (2.0, 10.0)


def citation_accuracy(papers: list[Paper]) -> float:
    if not papers:
        return 0.0
    valid = sum(1 for p in papers if _ID_PREFIX_PATTERNS.get(p.source, re.compile(r".*")).match(p.id))
    return valid / len(papers)


def recency(papers: list[Paper], within_years: int = 3) -> float:
    if not papers:
        return 0.0
    current_year = dt.date.today().year
    recent = sum(1 for p in papers if p.year and current_year - p.year <= within_years)
    return recent / len(papers)


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICING.get(model_name, _DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def faithfulness(review: str, expected_themes: list[str]) -> dict:
    prompt = (
        "Score how well the review below covers the expected themes, on a 0.0-1.0 scale.\n\n"
        f"Expected themes: {expected_themes}\n\n"
        f"Review (truncated):\n{review[:4000]}\n\n"
        'Respond with ONLY JSON: {"score": 0.0-1.0, "covered": ["..."], "missing": ["..."]}'
    )
    try:
        response = get_judge_model().invoke([HumanMessage(content=prompt)])
        in_tok, out_tok = token_usage(response)
        data = parse_json_response(response.content)
        return {
            "score": float(data.get("score", 0.0)),
            "covered": data.get("covered", []),
            "missing": data.get("missing", []),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
    except Exception:
        return {"score": 0.0, "covered": [], "missing": expected_themes, "input_tokens": 0, "output_tokens": 0}
