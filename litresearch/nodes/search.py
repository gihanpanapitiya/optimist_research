"""Multi-source search execution and canonicalization.

Two nodes:

``search_all``
    Runs every planned query concurrently and collects raw hits. Concurrency
    lives inside the node rather than being fanned out with ``Send``: these
    are HTTP calls, each source already enforces its own rate limit, and one
    graph node per ``(source, query)`` pair would bloat the state with no
    benefit. The LLM-heavy stages are the ones that get real fan-out.

``canonicalize_documents``
    Collapses the hits into the canonical corpus, which is where the budget
    is actually protected: deduplicating before reading is what stops the
    same paper being read four times because four sources returned it.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import get_settings
from ..sources.registry import active_sources
from ..state import BudgetUsage, QueryPlan, RawHit, ResearchState, SearchIntent
from ..store.canonicalize import Canonicalizer
from .common import log_line

logger = logging.getLogger(__name__)


async def search_all(state: ResearchState) -> dict:
    """Execute all planned queries concurrently."""
    plans: list[QueryPlan] = state.get("query_plans") or []
    if not plans:
        return {"pending_hits": [], "log": [log_line("no queries to execute")]}

    settings = get_settings()
    sources = active_sources()
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def run(plan: QueryPlan) -> tuple[QueryPlan, list[RawHit]]:
        source = sources.get(plan.source)
        if source is None:
            return plan, []
        async with semaphore:
            try:
                hits = await source.search(plan)
            except Exception as exc:  # noqa: BLE001
                # One source failing must not take down the search round.
                logger.warning(
                    "search failed on %s for %r: %s", plan.source, plan.query, exc
                )
                return plan, []
        return plan, hits

    results = await asyncio.gather(*(run(plan) for plan in plans))

    all_hits: list[RawHit] = []
    per_source: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for plan, hits in results:
        all_hits.extend(hits)
        per_source[plan.source] = per_source.get(plan.source, 0) + len(hits)
        intent_counts[plan.intent_id] = intent_counts.get(plan.intent_id, 0) + len(hits)

    # Mark the intents that have now been searched.
    intents: dict[str, SearchIntent] = state.get("intents") or {}
    updated_intents: dict[str, SearchIntent] = {}
    for intent_id, count in intent_counts.items():
        intent = intents.get(intent_id)
        if intent is None:
            continue
        updated = intent.model_copy(deep=True)
        updated.status = "searched"
        updated.documents_found += count
        updated_intents[intent_id] = updated

    breakdown = ", ".join(f"{s}:{n}" for s, n in sorted(per_source.items())) or "nothing"

    return {
        "pending_hits": all_hits,
        "intents": updated_intents,
        "budget": BudgetUsage(search_operations=len(plans)),
        "log": [
            log_line(
                f"ran {len(plans)} queries, {len(all_hits)} raw hits ({breakdown})"
            )
        ],
    }


async def canonicalize_documents(state: ResearchState) -> dict:
    """Deduplicate raw hits into the canonical corpus."""
    hits: list[RawHit] = state.get("pending_hits") or []
    if not hits:
        return {"pending_hits": [], "log": [log_line("no hits to canonicalize")]}

    canonicalizer = Canonicalizer(
        state.get("documents") or {}, state.get("identity_index") or {}
    )
    result = canonicalizer.ingest(hits)

    total_docs = len(state.get("documents") or {}) + result.new_count

    return {
        "documents": result.documents,
        "identity_index": result.identity_index,
        # Clear the scratch channel now the hits are folded in.
        "pending_hits": [],
        "budget": BudgetUsage(documents_discovered=result.new_count),
        "log": [
            log_line(
                f"canonicalized {len(hits)} hits -> {result.new_count} new, "
                f"{result.merged_count} merged into existing, "
                f"{result.skipped_count} skipped "
                f"(corpus now {total_docs} documents)"
            )
        ],
    }
