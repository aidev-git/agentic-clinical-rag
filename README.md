# Agentic Clinical RAG

A multi-agent retrieval-augmented pipeline that answers a clinical research
question by searching real literature databases, extracting evidence-grounded
claims, self-critiquing coverage, and writing a cited review — orchestrated
as an explicit, bounded [LangGraph](https://github.com/langchain-ai/langgraph)
state machine.

## Pipeline

```
decomposer -> retriever -> extractor -> reviewer -+-> retriever (loop, bounded)
                                                    +-> writer -> done
```

| Agent        | Pattern                | What it does |
|--------------|-------------------------|--------------|
| `decomposer` | Hierarchical delegation | Breaks the question into 3-5 focused sub-queries |
| `retriever`  | ReAct                   | Searches PubMed, Semantic Scholar, and arXiv; judges result quality and refines the query in a bounded loop |
| `extractor`  | Tool use                | Extracts structured, confidence-scored claims from each new paper's abstract |
| `reviewer`   | Self-reflection         | Judges evidence coverage; can route back to `retriever` for up to 2 refinement cycles before forcing termination |
| `writer`     | Synthesis               | Produces a final markdown review with inline citations and a reference list |

State (papers, claims, token counts) accumulates across refinement cycles via
LangGraph reducers rather than being overwritten each pass.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in your LLM_PROVIDER and its credentials
```

The LLM backend is fully pluggable via `LLM_PROVIDER` in `.env`:

- `anthropic` — set `ANTHROPIC_API_KEY`
- `vertex` — set `GCP_PROJECT_ID`; auth via `gcloud auth application-default login`; install with `pip install -e ".[vertex]"`
- `gemini` — set `GOOGLE_API_KEY`; install with `pip install -e ".[gemini]"`

`.env` is git-ignored — never commit credentials.

## Usage

```bash
clinical-rag "What is the current evidence on GLP-1 agonists for non-diabetic weight management?"
clinical-rag "..." --debug   # print per-agent progress
```

## Evaluation

```bash
python evals/run.py               # run the full question bank
python evals/run.py --case glp1-weight
python evals/run.py --no-judge    # skip the LLM-as-judge faithfulness metric
```

Metrics: citation-format plausibility, recency ratio, LLM-judged faithfulness
to expected themes, estimated token cost, and latency. Results are written
incrementally to `eval_results.md`.

## Tests

```bash
pytest                # fast, fully mocked unit tests
pytest --integration  # also hits live PubMed/Semantic Scholar/arXiv/LLM APIs
```

## Project layout

```
src/agentic_clinical_rag/
  agents/        # decomposer, retriever, extractor, reviewer, writer
  sources/       # pubmed, semantic_scholar, arxiv + shared HTTP/cache/rate-limit layer
  prompts/       # one template per agent
  state.py       # shared ResearchState + accumulator reducers
  graph.py       # LangGraph wiring
  llm.py         # provider-agnostic chat-model factory
  cli.py         # Typer CLI
evals/           # question bank + scoring + runner
tests/           # unit + opt-in integration tests
```
