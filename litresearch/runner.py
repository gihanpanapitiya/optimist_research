"""Run the research graph and persist the corpus.

This is the programmatic entry point. ``litresearch.cli`` wraps it for the
terminal; an orchestrating agent delegating a literature task would call
:func:`run_research` directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import get_settings
from .graph import build_graph
from .sources.base import close_http_client
from .state import BudgetUsage, ResearchState, new_state
from .store.corpus import ResearchCorpus

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict], None]


async def run_research(
    question: str,
    *,
    scope_notes: str = "",
    output_dir: str | None = None,
    resume: bool = False,
    on_progress: ProgressCallback | None = None,
    recursion_limit: int = 150,
) -> ResearchState:
    """Investigate ``question`` and return the final research state.

    The corpus is written after every node, so an interrupted run leaves a
    usable partial corpus on disk rather than nothing.
    """
    settings = get_settings()
    corpus = ResearchCorpus(output_dir or settings.output_dir, question)
    corpus.ensure()

    initial: ResearchState = new_state(question, scope_notes)
    if resume:
        saved = corpus.load_state()
        if saved is not None:
            logger.info("resuming from %s", corpus.state_path)
            initial = saved
            initial["main_question"] = question
        else:
            logger.info("no saved state at %s; starting fresh", corpus.state_path)

    graph = build_graph()
    final: ResearchState = initial

    try:
        # `stream_mode="values"` gives the full state after each super-step,
        # which is what lets the corpus be checkpointed continuously.
        async for snapshot in graph.astream(
            initial,
            stream_mode="values",
            config={
                "recursion_limit": recursion_limit,
                # Bound the Send fan-outs. Without this, every selected
                # document is read concurrently, which is fine against a
                # hosted API but pointless against a single local model --
                # the requests just queue, and the memory for all those
                # in-flight prompts is spent for nothing.
                "max_concurrency": settings.max_concurrent_readers,
            },
        ):
            final = snapshot
            if on_progress is not None:
                try:
                    on_progress("step", snapshot)
                except Exception:  # noqa: BLE001 - progress must never break a run
                    logger.debug("progress callback raised", exc_info=True)
            try:
                corpus.save_state(snapshot)
            except OSError as exc:
                logger.warning("could not checkpoint state: %s", exc)
    finally:
        # The shared HTTP client outlives individual nodes, so close it here.
        await close_http_client()

    try:
        corpus.write_all(final)
    except OSError as exc:
        logger.warning("could not write corpus projections: %s", exc)

    return final


def summarize_run(state: ResearchState, output_dir: str | None = None) -> dict[str, Any]:
    """Headline numbers for a completed run."""
    settings = get_settings()
    documents = state.get("documents") or {}
    usage: BudgetUsage = state.get("budget") or BudgetUsage()
    subquestions = state.get("subquestions") or {}
    corpus = ResearchCorpus(
        output_dir or settings.output_dir, state.get("main_question", "")
    )

    considered = [sq for sq in subquestions.values() if sq.status != "deprioritized"]
    coverage = (
        sum(sq.coverage for sq in considered) / len(considered) if considered else 0.0
    )

    return {
        "question": state.get("main_question", ""),
        "iterations": usage.research_iterations,
        "search_operations": usage.search_operations,
        "snowball_rounds": usage.snowball_rounds,
        "llm_calls": usage.llm_calls,
        "documents_retrieved": len(documents),
        "documents_read": sum(1 for d in documents.values() if d.status == "read"),
        "evidence_items": len(state.get("evidence") or []),
        "contradictions": len(state.get("contradictions") or []),
        "subquestions": len(subquestions),
        "mean_coverage": coverage,
        "terminated_because": state.get("terminated_because", ""),
        "report_path": str(corpus.root / "final_report.md"),
        "corpus_root": str(corpus.root),
    }


def report_path(state: ResearchState, output_dir: str | None = None) -> Path:
    settings = get_settings()
    corpus = ResearchCorpus(
        output_dir or settings.output_dir, state.get("main_question", "")
    )
    return corpus.root / "final_report.md"
