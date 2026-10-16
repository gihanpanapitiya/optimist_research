"""Citation-based snowballing.

Runs after ranking, on documents that screened as genuinely relevant, in
whichever direction the current research objective calls for:

* **Backward** (references of a paper) finds foundational work, precursor
  methods and theoretical underpinnings.
* **Forward** (papers citing it) finds later developments, improvements,
  critiques and replications.

Depth is not fixed. Per the methodology, snowballing is governed by a budget
plus a **saturation criterion**: when a round's yield of newly relevant
documents falls below a threshold, the citation neighbourhood is treated as
exhausted and the loop stops. Declining yield across rounds is the signal
that the literature search is approaching completeness.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import get_settings
from ..sources.registry import snowball_sources
from ..state import (
    BudgetUsage,
    Document,
    RawHit,
    ResearchState,
    SnowballRound,
    Subquestion,
)
from ..store.canonicalize import Canonicalizer
from .common import active_subquestions, log_line

logger = logging.getLogger(__name__)



def plan_snowball(state: ResearchState) -> dict:
    """Choose seed documents and directions for this round.

    Seeds are picked without an LLM call: relevance and the shape of the
    corpus already say which papers are worth expanding and which direction
    is informative. Explicit requests from the critic take precedence, since
    those encode a specific research need.
    """
    settings = get_settings()
    documents: dict[str, Document] = state.get("documents") or {}
    history = state.get("snowball_history") or []
    usage = state.get("budget") or BudgetUsage()

    rounds_done = len({r.round_index for r in history})
    if rounds_done >= settings.budget.max_snowball_rounds:
        return {
            "snowball_seeds": [],
            "log": [log_line(f"snowball budget spent ({rounds_done} rounds)")],
        }

    if _is_saturated(history, settings.budget.snowball_min_yield):
        return {
            "snowball_seeds": [],
            "log": [log_line("snowball saturated: yield below threshold, stopping")],
        }

    if usage.documents_read >= settings.budget.max_documents_read:
        return {"snowball_seeds": [], "log": [log_line("read budget spent; skipping snowball")]}

    expanded = frozenset(state.get("snowball_expanded") or [])

    # Requests the critic made explicitly. A request for an expansion that
    # has already run would only re-fetch edges the corpus already holds.
    requested = [
        dict(item)
        for item in (state.get("snowball_seeds") or [])
        if item.get("doc_id") in documents
        and expansion_key(item["doc_id"], item.get("direction", "backward")) not in expanded
    ]

    max_seeds = max(1, settings.budget.max_snowball_seeds)
    seeds = requested[:max_seeds]
    if len(seeds) < max_seeds:
        seeds.extend(
            _auto_seeds(
                documents,
                active_subquestions(state),
                already={item["doc_id"] for item in seeds},
                limit=max_seeds - len(seeds),
                expanded=expanded,
            )
        )

    if not seeds:
        return {"snowball_seeds": [], "log": [log_line("no suitable snowball seeds")]}

    summary = ", ".join(f"{s['doc_id']}:{s['direction'][:3]}" for s in seeds)
    return {
        "snowball_seeds": seeds,
        "log": [log_line(f"snowball round {rounds_done + 1} seeds: {summary}")],
    }


async def expand_snowball(state: ResearchState) -> dict:
    """Fetch citation neighbours of the seeds and fold them into the corpus.

    Canonicalization happens inside this node rather than looping back
    through the shared canonicalize node, so that ``pending_hits`` keeps a
    single writer and the two paths cannot interleave.
    """
    seeds = state.get("snowball_seeds") or []
    if not seeds:
        return {"log": [log_line("snowball: nothing to expand")]}

    documents: dict[str, Document] = state.get("documents") or {}
    history = state.get("snowball_history") or []
    round_index = len({r.round_index for r in history}) + 1

    edge_limit = max(1, get_settings().budget.max_snowball_edges_per_seed)
    tasks = []
    expanded: list[str] = []
    for seed in seeds:
        doc = documents.get(seed.get("doc_id", ""))
        if doc is None:
            continue
        direction = seed.get("direction", "backward")
        tasks.append(_expand_one(doc, direction, edge_limit))
        expanded.append(expansion_key(doc.doc_id, direction))

    if not tasks:
        return {"log": [log_line("snowball: no resolvable seeds")]}

    results = await asyncio.gather(*tasks)

    all_hits: list[RawHit] = []
    per_direction: dict[str, int] = {}
    for direction, hits in results:
        all_hits.extend(hits)
        per_direction[direction] = per_direction.get(direction, 0) + len(hits)

    if not all_hits:
        # Still recorded as expanded: a seed with no citation data will not
        # grow any in a later round, so retrying it only spends a round.
        return {
            "snowball_seeds": [],
            "snowball_expanded": expanded,
            "snowball_history": [
                SnowballRound(
                    round_index=round_index,
                    direction="mixed",
                    candidates_examined=0,
                    newly_relevant=0,
                )
            ],
            "log": [log_line(f"snowball round {round_index}: no citation data returned")],
        }

    canonicalizer = Canonicalizer(documents, state.get("identity_index") or {})
    result = canonicalizer.ingest(all_hits)

    # Mark the depth of genuinely new documents so later stages can tell how
    # far from the original search a document sits.
    updates = dict(result.documents)
    for doc_id in result.new_doc_ids:
        doc = updates[doc_id].model_copy(deep=True)
        doc.snowball_depth = round_index
        updates[doc_id] = doc

    # Yield is measured in *new canonical documents*, not raw hits: a round
    # that returns 200 references we already have has saturated, even though
    # it returned plenty of data.
    rounds = [
        SnowballRound(
            round_index=round_index,
            direction="+".join(sorted(per_direction)),
            candidates_examined=len(all_hits),
            newly_relevant=result.new_count,
        )
    ]

    breakdown = ", ".join(f"{d}:{n}" for d, n in sorted(per_direction.items()))
    return {
        "documents": updates,
        "identity_index": result.identity_index,
        "snowball_seeds": [],
        "snowball_expanded": expanded,
        "snowball_history": rounds,
        "budget": BudgetUsage(
            snowball_rounds=1,
            documents_discovered=result.new_count,
        ),
        "log": [
            log_line(
                f"snowball round {round_index}: {len(all_hits)} candidates ({breakdown}) "
                f"-> {result.new_count} new, {result.merged_count} already known "
                f"(yield {rounds[0].yield_rate:.0%})"
            )
        ],
    }


async def _expand_one(
    doc: Document, direction: str, edge_limit: int
) -> tuple[str, list[RawHit]]:
    """Traverse one direction for one seed, trying each capable source."""
    sources = snowball_sources(direction)
    if not sources:
        logger.info("no source supports %s snowballing", direction)
        return direction, []

    for source in sources:
        try:
            if direction == "backward":
                hits = await source.references(doc, limit=edge_limit)
            else:
                hits = await source.citations(doc, limit=edge_limit)
        except NotImplementedError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s snowball failed on %s via %s: %s", direction, doc.doc_id, source.name, exc
            )
            continue
        if hits:
            return direction, hits
    return direction, []


def _auto_seeds(
    documents: dict[str, Document],
    active: list[Subquestion],
    already: set[str],
    limit: int,
    expanded: frozenset[str] = frozenset(),
) -> list[dict]:
    """Pick seeds and directions from the state of the corpus.

    Direction follows the research need rather than being applied uniformly:
    a subquestion whose evidence is all recent is missing its foundations, so
    expand backward; one whose evidence is all old is missing the current
    state of the art, so expand forward.

    A (document, direction) pair in ``expanded`` has already been traversed
    this run and is skipped, so a later round moves on to the next-best seed.
    """
    if not active:
        return []

    seeds: list[dict] = []
    years_by_sq: dict[str, list[int]] = {}
    for doc in documents.values():
        for sq_id, score in doc.relevance.items():
            if score >= 0.7 and doc.year:
                years_by_sq.setdefault(sq_id, []).append(doc.year)

    for sq in sorted(active, key=lambda s: -s.priority):
        if len(seeds) >= limit:
            break

        years = sorted(years_by_sq.get(sq.id, []))
        if not years:
            continue
        median = years[len(years) // 2]
        span = years[-1] - years[0]

        # A corpus clustered in the last few years with no historical spread
        # is missing its precursors; a corpus with an old median is missing
        # recent developments.
        if span <= 3 and median >= 2022:
            direction = "backward"
        elif median <= 2020:
            direction = "forward"
        else:
            # Mixed corpus: expand forward from the most-cited older work,
            # which is where the influential follow-ups will be.
            direction = "forward"

        candidates = [
            doc
            for doc in documents.values()
            if doc.relevance.get(sq.id, 0.0) >= 0.7
            and doc.doc_id not in already
            and expansion_key(doc.doc_id, direction) not in expanded
            and doc.snowball_depth == 0
            and (doc.s2_paper_id or doc.doi or doc.arxiv_id or doc.openalex_id)
        ]
        if not candidates:
            continue

        if direction == "backward":
            # Expand the most relevant recent paper: its reference list is the
            # curated path into the area's foundations.
            candidates.sort(
                key=lambda d: (d.relevance.get(sq.id, 0.0), d.year or 0), reverse=True
            )
        else:
            # Expand the most-cited paper: it has the most citing work to find.
            candidates.sort(
                key=lambda d: (d.citation_count or 0, d.relevance.get(sq.id, 0.0)),
                reverse=True,
            )

        seed = candidates[0]
        already.add(seed.doc_id)
        seeds.append(
            {
                "doc_id": seed.doc_id,
                "direction": direction,
                "reason": (
                    f"{sq.id}: evidence spans {years[0]}-{years[-1]} (median {median}); "
                    f"expanding {direction} from {seed.citation_key}"
                ),
            }
        )

    return seeds


def expansion_key(doc_id: str, direction: str) -> str:
    """Identity of one traversal, as recorded in ``snowball_expanded``."""
    return f"{doc_id}:{direction}"


def _is_saturated(history: list[SnowballRound], min_yield: float) -> bool:
    """Has snowballing stopped producing new documents?

    Requires two consecutive low-yield rounds. One bad round can be an
    artifact of a single seed with poor citation coverage; two in a row is a
    property of the neighbourhood.
    """
    if len(history) < 2:
        return False
    recent = sorted(history, key=lambda r: r.round_index)[-2:]
    return all(r.yield_rate < min_yield for r in recent)


def should_snowball(state: ResearchState) -> str:
    """Conditional edge: expand the citation graph, or move on to reading."""
    return "expand_snowball" if (state.get("snowball_seeds") or []) else "prepare_reading"
