"""Wires the five agents into a bounded LangGraph pipeline.

decomposer -> retriever -> extractor -> reviewer -> (retriever | writer) -> END
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from agentic_clinical_rag.agents import decomposer, extractor, retriever, reviewer, writer
from agentic_clinical_rag.state import ResearchState


@lru_cache
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("decomposer", decomposer.run)
    graph.add_node("retriever", retriever.run)
    graph.add_node("extractor", extractor.run)
    graph.add_node("reviewer", reviewer.run)
    graph.add_node("writer", writer.run)

    graph.set_entry_point("decomposer")
    graph.add_edge("decomposer", "retriever")
    graph.add_edge("retriever", "extractor")
    graph.add_edge("extractor", "reviewer")
    graph.add_conditional_edges("reviewer", reviewer.route, {"retriever": "retriever", "writer": "writer"})
    graph.add_edge("writer", END)

    return graph.compile()
