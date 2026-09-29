"""Shared state threaded through every node of the research graph."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    source: str  # "pubmed" | "semantic_scholar" | "arxiv"
    url: str
    doi: str | None = None


@dataclass
class Claim:
    paper_id: str
    paper_title: str
    text: str
    confidence: float = 1.0


@dataclass
class ReviewVerdict:
    status: str  # "sufficient" | "needs_more"
    reasoning: str
    gaps: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)


def _keep_latest(_old: int, new: int) -> int:
    """Reducer for fields that should be overwritten, not merged."""
    return new


class ResearchState(TypedDict, total=False):
    question: str
    sub_queries: list[str]
    papers: Annotated[list[Paper], operator.add]
    claims: Annotated[list[Claim], operator.add]
    verdict: ReviewVerdict | None
    refine_count: Annotated[int, _keep_latest]
    review: str | None
    citations: list[str]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    debug: bool


def new_state(question: str, debug: bool = False) -> ResearchState:
    return ResearchState(
        question=question,
        sub_queries=[],
        papers=[],
        claims=[],
        verdict=None,
        refine_count=0,
        review=None,
        citations=[],
        input_tokens=0,
        output_tokens=0,
        debug=debug,
    )
