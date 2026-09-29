"""CLI entrypoint: `clinical-rag research "<question>" [--debug]`."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from agentic_clinical_rag.graph import build_graph
from agentic_clinical_rag.state import new_state
from agentic_clinical_rag.tracing import enable_tracing

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def research(
    question: str = typer.Argument(..., help="The clinical research question to investigate."),
    debug: bool = typer.Option(False, "--debug", help="Print per-agent progress."),
) -> None:
    enable_tracing()
    state = new_state(question, debug=debug)
    graph = build_graph()

    final_state = state
    try:
        for step in graph.stream(state, stream_mode="values"):
            final_state = step
    except Exception as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if debug:
        console.print(
            Panel(
                f"Sub-queries: {final_state.get('sub_queries')}\n"
                f"Papers: {len(final_state.get('papers', []))}  "
                f"Claims: {len(final_state.get('claims', []))}  "
                f"Refine cycles: {final_state.get('refine_count', 0)}\n"
                f"Tokens: in={final_state.get('input_tokens', 0)} "
                f"out={final_state.get('output_tokens', 0)}",
                title="Run summary",
            )
        )

    console.print(Panel(final_state.get("review") or "(no review generated)", title="Literature Review"))
    citations = final_state.get("citations") or []
    if citations:
        console.print("\n[bold]References[/bold]")
        for citation in citations:
            console.print(f"  - {citation}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
