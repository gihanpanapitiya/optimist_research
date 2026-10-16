"""Command-line interface.

    optimiztresearch run "What are the recent developments in planning for AI agents?"
    optimiztresearch sources
    optimiztresearch config
"""

from __future__ import annotations

import asyncio
import logging
import os

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import ALL_SOURCES, get_settings
from .runner import run_research, summarize_run
from .sources.registry import SOURCE_CLASSES

app = typer.Typer(
    add_completion=False,
    help="An iterative deep-research agent for autonomous literature research.",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )
    # These are chatty at DEBUG and drown out the research log.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def run(
    question: str = typer.Argument(..., help="The research question to investigate."),
    scope: str = typer.Option(
        "", "--scope", "-s", help="Extra scope notes or constraints for the planner."
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output directory (default: OPTIMIZTRESEARCH_OUTPUT_DIR)."
    ),
    sources: str = typer.Option(
        None,
        "--sources",
        help=f"Comma-separated sources to use. Available: {','.join(ALL_SOURCES)}",
    ),
    semantic_scholar: str = typer.Option(
        None,
        "--semantic-scholar",
        help="Use Semantic Scholar: auto (on only with an API key), on, or off.",
    ),
    max_documents: int = typer.Option(
        None, "--max-documents", help="Cap on documents read in depth."
    ),
    max_iterations: int = typer.Option(
        None, "--max-iterations", help="Cap on research iterations."
    ),
    full_text: bool = typer.Option(
        False, "--full-text", help="Fetch and read PDFs instead of abstracts only."
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Continue from a saved research_state.json."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Investigate a research question and write a literature report."""
    load_dotenv()
    _setup_logging(verbose)

    # CLI flags win over the environment, so apply them before settings load.
    if sources:
        os.environ["OPTIMIZTRESEARCH_ENABLED_SOURCES"] = sources
    if semantic_scholar:
        choice = semantic_scholar.strip().lower()
        if choice not in ("auto", "on", "off"):
            console.print(
                f"[red]--semantic-scholar must be auto, on or off[/red] (got {choice!r})"
            )
            raise typer.Exit(2)
        os.environ["OPTIMIZTRESEARCH_SEMANTIC_SCHOLAR"] = choice
    if max_documents is not None:
        os.environ["OPTIMIZTRESEARCH_BUDGET__MAX_DOCUMENTS_READ"] = str(max_documents)
    if max_iterations is not None:
        os.environ["OPTIMIZTRESEARCH_BUDGET__MAX_RESEARCH_ITERATIONS"] = str(max_iterations)
    if full_text:
        os.environ["OPTIMIZTRESEARCH_READ_FULL_TEXT"] = "true"
    if output:
        os.environ["OPTIMIZTRESEARCH_OUTPUT_DIR"] = output

    settings = get_settings(refresh=True)

    available = settings.available_sources()
    if not available:
        console.print(
            "[red]No literature sources are available.[/red] Check "
            "OPTIMIZTRESEARCH_ENABLED_SOURCES and required API keys."
        )
        raise typer.Exit(1)

    console.rule("[bold]Literature research")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Sources:[/bold] {', '.join(available)}")
    for name in settings.sources:
        reason = settings.source_unavailable_reason(name)
        if reason:
            console.print(f"[yellow]  {name} skipped:[/yellow] {reason}")
    if "semantic_scholar" not in available:
        citation_source = "openalex" if "openalex" in available else None
        console.print(
            "[bold]Citations:[/bold] "
            + (
                f"{citation_source} (both directions)"
                if citation_source
                else "[yellow]no citation source -- snowballing disabled[/yellow]"
            )
        )
    console.print(
        f"[bold]Budget:[/bold] {settings.budget.max_documents_read} documents, "
        f"{settings.budget.max_search_operations} searches, "
        f"{settings.budget.max_research_iterations} iterations"
    )
    console.print(
        f"[bold]Reading:[/bold] "
        f"{'full text (PDF)' if settings.read_full_text else 'abstracts and metadata'}"
    )
    console.rule()

    seen_log_entries = 0

    def on_progress(_event: str, snapshot: dict) -> None:
        nonlocal seen_log_entries
        entries = snapshot.get("log") or []
        for entry in entries[seen_log_entries:]:
            console.print(f"  {entry}")
        seen_log_entries = len(entries)

    try:
        state = asyncio.run(
            run_research(
                question,
                scope_notes=scope,
                output_dir=output,
                resume=resume,
                on_progress=on_progress,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Partial corpus was saved.")
        raise typer.Exit(130) from None
    except RuntimeError as exc:
        console.print(f"[red]Research could not start:[/red] {exc}")
        raise typer.Exit(1) from None

    summary = summarize_run(state, output_dir=output)

    console.rule("[bold]Complete")
    table = Table(show_header=False, box=None)
    table.add_row("Iterations", str(summary["iterations"]))
    table.add_row("Search operations", str(summary["search_operations"]))
    table.add_row("Snowball rounds", str(summary["snowball_rounds"]))
    table.add_row("LLM calls", str(summary["llm_calls"]))
    table.add_row("Documents retrieved", str(summary["documents_retrieved"]))
    table.add_row("Documents read", str(summary["documents_read"]))
    table.add_row("Evidence items", str(summary["evidence_items"]))
    table.add_row("Subquestions", str(summary["subquestions"]))
    table.add_row("Mean coverage", f"{summary['mean_coverage']:.0%}")
    console.print(table)
    console.print()
    console.print(f"[dim]{summary['terminated_because']}[/dim]")
    console.print()
    console.print(f"[bold]Report:[/bold] {summary['report_path']}")
    console.print(f"[bold]Corpus:[/bold] {summary['corpus_root']}")


@app.command()
def sources() -> None:
    """Show which literature sources are configured and usable."""
    load_dotenv()
    settings = get_settings(refresh=True)
    from .nodes.common import SOURCE_DESCRIPTIONS

    enabled = set(settings.sources)

    table = Table("source", "status", "citations", "notes")
    for name in ALL_SOURCES:
        reason = settings.source_unavailable_reason(name) if name in enabled else None
        if name not in enabled:
            status, note = "[dim]not enabled[/dim]", ""
        elif reason:
            status, note = "[yellow]unavailable[/yellow]", reason
        else:
            status, note = "[green]ready[/green]", SOURCE_DESCRIPTIONS.get(name, "")[:70]

        source_cls = SOURCE_CLASSES[name]
        directions = []
        if source_cls.supports_backward:
            directions.append("back")
        if source_cls.supports_forward:
            directions.append("fwd")
        table.add_row(name, status, "+".join(directions) or "—", note)
    console.print(table)

    console.print(
        f"\n[bold]Semantic Scholar:[/bold] mode={settings.semantic_scholar}, "
        f"key={'present' if settings.semantic_scholar_has_key else 'absent'} "
        f"-> {'enabled' if settings.use_semantic_scholar else 'disabled'}"
    )
    if not settings.use_semantic_scholar:
        console.print(
            "  Citation snowballing will run on OpenAlex, which covers both "
            "directions and needs no key. Force Semantic Scholar on anyway with "
            "[bold]--semantic-scholar on[/bold]."
        )

    if not settings.contact_email:
        console.print(
            "\n[yellow]No OPTIMIZTRESEARCH_CONTACT_EMAIL set.[/yellow] Setting one puts "
            "OpenAlex, Crossref and PubMed requests in their polite pools, which "
            "have far better rate limits."
        )


@app.command()
def config() -> None:
    """Show the effective configuration and where each layer came from."""
    load_dotenv()
    settings = get_settings(refresh=True)

    console.print("[bold]Configuration layers[/bold] (lowest precedence first)")
    console.print("  1. defaults in optimiztresearch/config.py")
    yaml_files = settings.config_files
    if yaml_files:
        for path in yaml_files:
            console.print(f"  2. [green]{path}[/green]")
    else:
        console.print(
            "  2. [dim]no optimiztresearch.yaml found[/dim] "
            "(create one beside the project, or set OPTIMIZTRESEARCH_CONFIG_FILE)"
        )
    env_files = settings.env_files
    if env_files:
        for path in env_files:
            console.print(f"  3. [green]{path}[/green] (secrets and local overrides)")
    else:
        console.print("  3. [dim]no .env found[/dim]")
    overrides = sorted(k for k in os.environ if k.startswith("OPTIMIZTRESEARCH_"))
    if overrides:
        console.print(
            f"  4. [yellow]environment: {', '.join(overrides)}[/yellow] "
            "(overrides the files above)"
        )
    else:
        console.print("  4. [dim]no OPTIMIZTRESEARCH_* environment overrides[/dim]")
    console.print()

    table = Table("setting", "value")
    for role, spec in settings.models.model_dump().items():
        if isinstance(spec, str):
            table.add_row(f"model.{role}", spec)
    for key, value in settings.budget.model_dump().items():
        table.add_row(f"budget.{key}", str(value))
    table.add_row("enabled_sources", settings.enabled_sources)
    table.add_row(
        "semantic_scholar",
        f"{settings.semantic_scholar} -> "
        f"{'enabled' if settings.use_semantic_scholar else 'disabled'}",
    )
    table.add_row("read_full_text", str(settings.read_full_text))
    table.add_row("output_dir", settings.output_dir)
    table.add_row("contact_email", settings.contact_email or "(unset)")
    console.print(table)


if __name__ == "__main__":
    app()
