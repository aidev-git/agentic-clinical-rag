"""Standalone eval runner: `python evals/run.py [--case ID] [--no-judge]`.

Usable without installing the package (adjusts sys.path to find src/).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from agentic_clinical_rag.config import get_settings  # noqa: E402
from agentic_clinical_rag.graph import build_graph  # noqa: E402
from agentic_clinical_rag.state import new_state  # noqa: E402
from evals.metrics import citation_accuracy, estimate_cost_usd, faithfulness, recency  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.md"


def load_cases() -> list[dict]:
    with open(Path(__file__).parent / "cases.yaml") as f:
        return yaml.safe_load(f)


def run_case(case: dict, judge: bool) -> dict:
    graph = build_graph()
    start = time.monotonic()
    try:
        result = graph.invoke(new_state(case["question"]))
        latency = time.monotonic() - start
    except Exception as exc:
        return {**case, "error": str(exc)}

    papers = result.get("papers", [])
    review = result.get("review") or ""
    settings = get_settings()
    model_name = settings.anthropic_model if settings.llm_provider == "anthropic" else settings.vertex_model

    row = {
        **case,
        "papers": len(papers),
        "claims": len(result.get("claims", [])),
        "citation_accuracy": citation_accuracy(papers),
        "recency": recency(papers),
        "cost_usd": estimate_cost_usd(
            model_name, result.get("input_tokens", 0), result.get("output_tokens", 0)
        ),
        "latency_s": round(latency, 1),
        "review": review,
        "references": result.get("citations", []),
        "error": None,
    }

    if judge:
        faith = faithfulness(review, case.get("expected_themes", []))
        row["faithfulness"] = faith["score"]
    else:
        row["faithfulness"] = None

    return row


def write_results(rows: list[dict]) -> None:
    lines = ["# Eval Results\n", "| ID | Papers | Claims | Citation Acc | Recency | Faithfulness | Cost ($) | Latency (s) | Error |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row.get('papers', '-')} | {row.get('claims', '-')} | "
            f"{row.get('citation_accuracy', '-')} | {row.get('recency', '-')} | "
            f"{row.get('faithfulness', '-')} | {row.get('cost_usd', '-')} | "
            f"{row.get('latency_s', '-')} | {row.get('error') or ''} |"
        )

    valid = [r for r in rows if not r.get("error")]
    if valid:
        avg_cost = sum(r["cost_usd"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_s"] for r in valid) / len(valid)
        lines.append(f"\n**Averages** — cost: ${avg_cost:.4f}, latency: {avg_latency:.1f}s")

    for row in valid:
        lines.append(f"\n## {row['id']}\n\n{row['review']}\n\n### References\n")
        lines.extend(f"- {ref}" for ref in row.get("references", []))

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run only the case with this id")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge faithfulness metric")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    rows = []
    for case in cases:
        print(f"Running: {case['id']}")
        row = run_case(case, judge=not args.no_judge)
        rows.append(row)
        write_results(rows)  # incremental write so a killed run still leaves output

    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
