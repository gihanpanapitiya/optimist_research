"""MCP server, so coding agents such as Claude Code can delegate literature research.

    optimistresearch-mcp                  # stdio transport

A run takes minutes, far longer than an agent should block on one tool call,
so ``start_research`` launches it in the background and returns at once. The
agent then polls ``research_status`` and fetches ``get_report`` when it is done.

A run is identified by its corpus directory name (the question's slug), not by
a process-local handle. Everything the status and report tools need is read
back from disk, so a corpus from an earlier session -- or one written by the
CLI -- is addressable the same way.

stdout carries the protocol, so nothing in this process may print to it. The
research log goes to stderr through ``logging``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from .config import ALL_SOURCES, get_settings
from .runner import run_research, summarize_run
from .store.corpus import ResearchCorpus

logger = logging.getLogger(__name__)

# How many research-log lines a status call returns. The full log is in
# research_state.json; an agent polling for progress only needs the tail.
STATUS_LOG_TAIL = 15

mcp = MCPServer(
    name="optimistresearch",
    instructions=(
        "Optimist Deep Research investigates a research question across scholarly "
        "sources (OpenAlex, arXiv, Crossref, PubMed, ...) and writes a cited "
        "literature report. A run takes several minutes or more. Call "
        "start_research, then poll research_status every minute or so until it "
        "reports 'completed' or 'failed', then call get_report. Only one run can "
        "be active at a time."
    ),
)


@dataclass
class _Run:
    run_id: str
    question: str
    corpus_root: Path
    task: asyncio.Task | None = None
    log: list[str] = field(default_factory=list)
    error: str | None = None


# Runs started by this process. Keeping the task referenced here also stops
# it being garbage-collected mid-run.
_runs: dict[str, _Run] = {}


def _active_run() -> _Run | None:
    return next((r for r in _runs.values() if r.task and not r.task.done()), None)


def _output_root() -> Path:
    return Path(get_settings().output_dir).expanduser().resolve()


@contextmanager
def _overrides(values: dict[str, str]):
    """Apply per-run settings overrides the way the CLI's flags do.

    Settings are a process-wide singleton read throughout the graph, so the
    overrides go in through the environment and the singleton is rebuilt.
    The environment is restored afterwards; the rebuilt singleton keeps the
    values for the run, and the next ``start_research`` rebuilds it again.
    """
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        get_settings(refresh=True)
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


async def _execute(run: _Run, scope_notes: str, resume: bool) -> None:
    def on_progress(_event: str, snapshot: dict) -> None:
        run.log = list(snapshot.get("log") or [])

    try:
        await run_research(
            run.question, scope_notes=scope_notes, resume=resume, on_progress=on_progress
        )
    except Exception as exc:  # noqa: BLE001 - reported through research_status
        logger.exception("research run %s failed", run.run_id)
        run.error = f"{type(exc).__name__}: {exc}"


def _resolve(run_id: str) -> Path:
    """Map a run id to its corpus directory, refusing anything outside the output dir."""
    root = _output_root()
    path = (root / run_id).resolve()
    if path.parent != root:
        raise ValueError(f"invalid run_id {run_id!r}")
    if not path.is_dir():
        raise ValueError(
            f"no research run {run_id!r} in {root}. Call list_research to see what exists."
        )
    return path


@mcp.tool()
async def start_research(
    question: str,
    scope_notes: str = "",
    max_documents: int | None = None,
    max_iterations: int | None = None,
    full_text: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Start a literature research run in the background and return its run_id.

    Args:
        question: The research question to investigate.
        scope_notes: Extra scope or constraints for the planner, e.g.
            "Focus on empirical evaluations since 2023."
        max_documents: Cap on documents read in depth (default from config).
        max_iterations: Cap on research iterations (default from config).
        full_text: Fetch and read PDFs instead of abstracts only. Slower.
        resume: Continue from the saved state of an earlier run of the same
            question, e.g. one interrupted when its session ended.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    active = _active_run()
    if active is not None:
        raise ValueError(
            f"run {active.run_id!r} is still in progress. Wait for it to finish "
            "(research_status) or call cancel_research first."
        )

    values: dict[str, str] = {}
    if max_documents is not None:
        values["OPTIMISTRESEARCH_BUDGET__MAX_DOCUMENTS_READ"] = str(max_documents)
    if max_iterations is not None:
        values["OPTIMISTRESEARCH_BUDGET__MAX_RESEARCH_ITERATIONS"] = str(max_iterations)
    if full_text:
        values["OPTIMISTRESEARCH_READ_FULL_TEXT"] = "true"

    with _overrides(values):
        settings = get_settings()
        available = settings.available_sources()
        if not available:
            raise ValueError(
                "no literature sources are available. Check "
                "OPTIMISTRESEARCH_ENABLED_SOURCES and required API keys."
            )
        corpus = ResearchCorpus(settings.output_dir, question)
        run = _Run(run_id=corpus.root.name, question=question, corpus_root=corpus.root)
        run.task = asyncio.create_task(_execute(run, scope_notes, resume))
        _runs[run.run_id] = run

    return {
        "run_id": run.run_id,
        "status": "running",
        "corpus_root": str(run.corpus_root),
        "sources": available,
        "budget": {
            "max_documents_read": settings.budget.max_documents_read,
            "max_search_operations": settings.budget.max_search_operations,
            "max_research_iterations": settings.budget.max_research_iterations,
        },
        "read_full_text": settings.read_full_text,
    }


@mcp.tool()
async def research_status(run_id: str) -> dict[str, Any]:
    """Report progress of a research run: status, recent log lines, and a summary once done.

    Status is one of 'running', 'completed', 'failed', 'cancelled', or
    'incomplete' (a corpus from an earlier session that never finished; pass
    resume=True to start_research to continue it).
    """
    run = _runs.get(run_id)
    if run is not None and run.task is not None and not run.task.done():
        return {"run_id": run_id, "status": "running", "recent_log": run.log[-STATUS_LOG_TAIL:]}

    # A run from this session may have ended before creating its corpus
    # directory (cancelled or failed at once), so only disk lookups resolve.
    path = run.corpus_root if run is not None else _resolve(run_id)
    state = ResearchCorpus(path).load_state()

    if run is not None and run.task is not None and run.task.cancelled():
        status = "cancelled"
    elif run is not None and run.error:
        status = "failed"
    elif (path / "final_report.md").exists():
        status = "completed"
    else:
        status = "incomplete"

    result: dict[str, Any] = {"run_id": run_id, "status": status, "corpus_root": str(path)}
    if run is not None and run.error:
        result["error"] = run.error
    if state is not None:
        result["recent_log"] = list(state.get("log") or [])[-STATUS_LOG_TAIL:]
        summary = summarize_run(state, output_dir=str(path.parent))
        result["summary"] = {k: v for k, v in summary.items() if k != "corpus_root"}
    return result


@mcp.tool()
async def cancel_research(run_id: str) -> dict[str, Any]:
    """Cancel a running research run. The partial corpus stays on disk and can be resumed."""
    run = _runs.get(run_id)
    if run is None or run.task is None or run.task.done():
        raise ValueError(f"run {run_id!r} is not running in this session")
    run.task.cancel()
    try:
        await run.task
    except asyncio.CancelledError:
        pass
    return {"run_id": run_id, "status": "cancelled", "corpus_root": str(run.corpus_root)}


@mcp.tool()
async def get_report(run_id: str) -> str:
    """Return the final literature report (Markdown) of a completed run."""
    path = _resolve(run_id) / "final_report.md"
    if not path.exists():
        raise ValueError(
            f"run {run_id!r} has no final report yet. Check research_status first."
        )
    return path.read_text(encoding="utf-8")


@mcp.tool()
async def get_corpus_file(run_id: str, relative_path: str) -> str:
    """Read a text file from a run's corpus.

    Useful paths: research_plan.md, titles.md, evidence/claims.md,
    evidence/contradictions.md, subtopics/<subquestion>.md,
    sources/doc_0001/notes.md, sources/doc_0001/metadata.json.
    """
    root = _resolve(run_id)
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("relative_path must stay inside the run's corpus")
    if not path.is_file():
        raise ValueError(f"{relative_path!r} does not exist in run {run_id!r}")
    if path.suffix not in (".md", ".json", ".txt", ".bib"):
        raise ValueError("only text files (.md, .json, .txt, .bib) can be read")
    return path.read_text(encoding="utf-8")


@mcp.tool()
async def list_research() -> list[dict[str, Any]]:
    """List research runs in the output directory, most recently updated first."""
    root = _output_root()
    if not root.is_dir():
        return []
    active = _active_run()
    runs = []
    for path in root.iterdir():
        state_path = path / "research_state.json"
        if not state_path.exists():
            continue
        if active is not None and active.run_id == path.name:
            status = "running"
        elif (path / "final_report.md").exists():
            status = "completed"
        else:
            status = "incomplete"
        runs.append(
            {
                "run_id": path.name,
                "status": status,
                "updated": state_path.stat().st_mtime,
                "corpus_root": str(path),
            }
        )
    runs.sort(key=lambda r: r["updated"], reverse=True)
    return runs


@mcp.tool()
async def list_sources() -> list[dict[str, Any]]:
    """Show which literature sources are enabled and usable with the current configuration."""
    settings = get_settings()
    enabled = set(settings.sources)
    rows = []
    for name in ALL_SOURCES:
        reason = settings.source_unavailable_reason(name) if name in enabled else None
        rows.append(
            {
                "source": name,
                "enabled": name in enabled,
                "available": name in enabled and reason is None,
                "reason": reason or ("not enabled" if name not in enabled else None),
            }
        )
    return rows


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    get_settings(refresh=True)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
