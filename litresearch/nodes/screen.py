"""Document screening, ranking and selection.

This node is the boundary between cheap retrieval and expensive deep
reading. Search is deliberately generous; this is where the funnel narrows.

Two stages:

1. **Screening.** A cheap model scores each candidate's relevance to each
   active subquestion from title, abstract and metadata. Screening is where
   documents become shareable across subquestions: a paper retrieved for RQ1
   may screen as highly relevant to RQ3 and will then be read for both,
   without being retrieved again.

2. **Ranking and selection.** Relevance is combined with the retrieval
   objectives from the methodology -- recency, citation impact, source
   quality, primary-source character, multi-source corroboration, and
   novelty against what has already been read -- and the top documents are
   assigned to reader agents per subquestion.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import ScreeningOutput
from ..state import BudgetUsage, Document, ResearchState, SearchIntent, Subquestion
from ..store.canonicalize import significant_tokens
from .common import (
    active_subquestions,
    current_year,
    format_document_brief,
    log_line,
    resolve_subquestion_id,
)

logger = logging.getLogger(__name__)

SYSTEM = """You screen retrieved documents for a literature review, deciding \
which are worth reading in full.

For each document you are given a title, metadata and an abstract. For each \
subquestion the document plausibly bears on, score its relevance from 0 to 1:

  0.9-1.0  Directly addresses the subquestion; central evidence.
  0.7-0.9  Substantially relevant; contains evidence that bears on it.
  0.5-0.7  Partially relevant; one section or a secondary result applies.
  0.3-0.5  Tangential; shares vocabulary but not substance.
  0.0-0.3  Not relevant.

Only list subquestions the document genuinely bears on. OMIT the others \
rather than scoring them low -- an omitted subquestion is treated as \
irrelevant, and listing everything at 0.1 just adds noise.

Judge on substance, not keyword overlap. Shared terminology is the most \
common cause of a bad screening decision: a paper about "planning" in \
logistics does not bear on a subquestion about planning in LLM agents, \
however many words it shares. Conversely, a paper that never uses the \
subquestion's vocabulary may still be squarely on point.

Be decisive. Reading budget is limited and every document you pass through \
displaces another. If an abstract is missing or uninformative, score on the \
title and metadata and say so in the note; do not score generously to hedge.

Keep each note to one line: what this document would contribute, or why it is \
off-target."""

HUMAN = """Main research question:
{main_question}

Subquestions to screen against:
{subquestions}

Documents to screen:

{documents}

Screen each document by its index."""


async def screen_and_rank(state: ResearchState) -> dict:
    """Screen new candidates, rank everything, and build the reading queue."""
    settings = get_settings()
    documents: dict[str, Document] = state.get("documents") or {}
    active = active_subquestions(state)

    if not documents:
        return {"reading_queue": [], "log": [log_line("no documents to screen")]}
    if not active:
        return {"reading_queue": [], "log": [log_line("no active subquestions to screen against")]}

    active_ids = [sq.id for sq in active]

    # Group documents by which active subquestions they have not been scored
    # against yet, so a newly added subquestion re-screens the existing corpus
    # while already-scored pairs are not paid for twice.
    pending: dict[tuple[str, ...], list[Document]] = defaultdict(list)
    for doc in documents.values():
        if doc.status in ("failed", "skipped"):
            continue
        unscored = tuple(sq for sq in active_ids if sq not in doc.relevance)
        if unscored:
            pending[unscored].append(doc)

    screened: dict[str, Document] = {}
    llm_calls = 0

    if pending:
        tasks = []
        for sq_subset, docs in pending.items():
            subset = [sq for sq in active if sq.id in set(sq_subset)]
            batch_size = max(1, settings.budget.screen_batch_size)
            for start in range(0, len(docs), batch_size):
                batch = docs[start : start + batch_size]
                tasks.append(_screen_batch(state, batch, subset))

        semaphore = asyncio.Semaphore(settings.max_concurrent_readers)

        async def guarded(coro):
            async with semaphore:
                return await coro

        for batch_result in await asyncio.gather(*(guarded(t) for t in tasks)):
            llm_calls += 1
            for doc_id, doc in batch_result.items():
                existing = screened.get(doc_id)
                screened[doc_id] = existing.merge(doc) if existing else doc

    # Rank over the full corpus, using screened scores where we have them.
    merged_corpus = {**documents, **screened}
    read_titles = [
        doc.title for doc in merged_corpus.values() if doc.status == "read"
    ]
    ranked = _rank(merged_corpus, settings, read_titles)

    queue, selected_docs = _select(
        ranked, active, settings, state.get("budget") or BudgetUsage()
    )

    # Fold ranking scores and status changes back into the screened updates.
    updates: dict[str, Document] = {}
    for doc_id, doc in ranked.items():
        base = screened.get(doc_id)
        target = base if base is not None else doc
        changed = (
            base is not None
            or abs(doc.rank_score - documents[doc_id].rank_score) > 1e-9
            or doc_id in selected_docs
        )
        if not changed:
            continue
        updated = target.model_copy(deep=True)
        updated.rank_score = doc.rank_score
        if doc_id in selected_docs and updated.status == "candidate":
            updated.status = "selected"
        updates[doc_id] = updated

    note = (
        f"screened {sum(len(d) for d in pending.values())} document(s), "
        f"selected {len(queue)} for reading "
        f"({sum(len(item['subquestion_ids']) for item in queue)} document-subquestion pairs)"
    )

    return {
        "documents": updates,
        "reading_queue": queue,
        "intents": _intent_yields(
            merged_corpus, state.get("intents") or {}, settings.ranking.min_relevance_to_read
        ),
        "budget": BudgetUsage(llm_calls=llm_calls),
        "log": [log_line(note)],
    }


def _intent_yields(
    documents: dict[str, Document], intents: dict[str, SearchIntent], threshold: float
) -> dict[str, SearchIntent]:
    """Each intent's yield: distinct documents its searches found, and how
    many of those screened as relevant to its subquestion.

    This is the per-intent yield the critic reads when judging saturation: an
    intent returning hits that all screen as irrelevant is a signal that the
    query, or the intent itself, is not working.

    Recomputed from the whole corpus rather than incremented. Documents are
    re-screened whenever a subquestion is added, and an increment credited
    their intents again on every such pass; a document reached through two
    sources was likewise counted once per provenance record. Only search
    provenance counts -- snowball hits inherit their seed's intent, but say
    nothing about whether that intent's queries are working.
    """
    found: dict[str, set[str]] = defaultdict(set)
    relevant: dict[str, set[str]] = defaultdict(set)
    for doc in documents.values():
        for prov in doc.discovered_by:
            if prov.via != "search" or prov.intent_id not in intents:
                continue
            found[prov.intent_id].add(doc.doc_id)
            if doc.relevance.get(prov.subquestion_id, 0.0) >= threshold:
                relevant[prov.intent_id].add(doc.doc_id)

    updates = {}
    for intent_id, doc_ids in found.items():
        intent = intents[intent_id]
        n_found, n_relevant = len(doc_ids), len(relevant[intent_id])
        if (intent.documents_found, intent.relevant_found) == (n_found, n_relevant):
            continue
        updated = intent.model_copy(deep=True)
        updated.documents_found = n_found
        updated.relevant_found = n_relevant
        updates[intent_id] = updated
    return updates


async def _screen_batch(
    state: ResearchState, batch: list[Document], subquestions: list[Subquestion]
) -> dict[str, Document]:
    """Score one batch of documents against one set of subquestions."""
    listing = "\n\n".join(
        format_document_brief(doc, index) for index, doc in enumerate(batch)
    )
    sq_text = "\n".join(f"- {sq.id}: {sq.question}" for sq in subquestions)

    messages = [
        SystemMessage(SYSTEM),
        HumanMessage(
            HUMAN.format(
                main_question=state["main_question"],
                subquestions=sq_text,
                documents=listing,
            )
        ),
    ]

    output = await ainvoke_structured("screener", ScreeningOutput, messages)
    if output is None:
        # Unscreened documents would otherwise be re-screened forever. Record
        # a neutral score so they settle below the read threshold but remain
        # visible to a later iteration that adds subquestions.
        logger.warning("screening failed for a batch of %d documents", len(batch))
        return {
            doc.doc_id: _with_scores(
                doc, {sq.id: 0.0 for sq in subquestions}, "screening call failed"
            )
            for doc in batch
        }

    valid_ids = {sq.id for sq in subquestions}
    updates: dict[str, Document] = {}
    seen_indices = set()

    for item in output.documents:
        if not 0 <= item.index < len(batch):
            logger.debug("screener returned out-of-range index %s", item.index)
            continue
        seen_indices.add(item.index)
        doc = batch[item.index]
        scores = {sq.id: 0.0 for sq in subquestions}
        for score in item.scores:
            # Repair the id before matching: some models echo the whole
            # prompt line ("RQ1: How do agents...") as the id, and dropping
            # those would score every document zero and read nothing.
            sq_id = resolve_subquestion_id(score.subquestion_id, valid_ids)
            if sq_id is None:
                logger.debug(
                    "screener returned unknown subquestion id %r", score.subquestion_id
                )
                continue
            scores[sq_id] = max(scores[sq_id], max(0.0, min(1.0, score.relevance)))
        updates[doc.doc_id] = _with_scores(doc, scores, item.note)

    # Documents the model skipped entirely are treated as irrelevant.
    for index, doc in enumerate(batch):
        if index not in seen_indices:
            updates[doc.doc_id] = _with_scores(
                doc, {sq.id: 0.0 for sq in subquestions}, "not returned by screener"
            )

    return updates


def _with_scores(doc: Document, scores: dict[str, float], note: str) -> Document:
    updated = doc.model_copy(deep=True)
    for sq_id, score in scores.items():
        updated.relevance[sq_id] = max(updated.relevance.get(sq_id, 0.0), score)
    if note:
        updated.screening_note = note.strip()
    return updated


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank(
    documents: dict[str, Document], settings, read_titles: list[str]
) -> dict[str, Document]:
    """Score every document against the retrieval objectives."""
    weights = settings.ranking
    year_now = current_year()
    read_token_sets = [significant_tokens(t) for t in read_titles]

    ranked: dict[str, Document] = {}
    for doc_id, doc in documents.items():
        relevance = doc.max_relevance()

        score = weights.relevance * relevance
        score += weights.citation_impact * _citation_score(doc, year_now)
        score += weights.recency * _recency_score(doc, year_now)
        score += weights.source_quality * _source_quality(doc)
        score += weights.multi_source_corroboration * min(len(doc.sources) / 3.0, 1.0)
        score += weights.primary_research * (1.0 if doc.is_primary_research else 0.0)
        score += weights.novelty * _novelty(doc, read_token_sets)

        # A document with no abstract is a much weaker read candidate in
        # abstract-only mode, since there is almost nothing to extract from.
        if not doc.abstract and not settings.read_full_text:
            score *= 0.6

        updated = doc.model_copy(deep=True)
        updated.rank_score = round(score, 6)
        ranked[doc_id] = updated

    return ranked


def _citation_score(doc: Document, year_now: int) -> float:
    """Log-scaled citation count, normalized for how long the paper has existed.

    Raw counts would systematically bury work from the last year or two,
    which is exactly the work a question about recent developments needs.
    """
    if doc.citation_count is None:
        return 0.3  # unknown, not zero: absence of data is not absence of impact
    age = max(1, year_now - (doc.year or year_now) + 1)
    per_year = doc.citation_count / age
    return min(math.log1p(per_year) / math.log1p(60), 1.0)


def _recency_score(doc: Document, year_now: int) -> float:
    if not doc.year:
        return 0.4
    age = year_now - doc.year
    if age <= 1:
        return 1.0
    if age >= 15:
        return 0.0
    return max(0.0, 1.0 - (age - 1) / 14.0)


def _source_quality(doc: Document) -> float:
    """Evidentiary weight by where the document came from."""
    if "web" in doc.sources and len(doc.sources) == 1:
        return 0.35  # grey literature only
    if doc.is_preprint and not doc.doi:
        return 0.7
    if doc.doi and doc.venue:
        return 1.0
    return 0.85


def _novelty(doc: Document, read_token_sets: list[set[str]]) -> float:
    """How different this document is from what has already been read.

    Prevents a reading queue of near-identical papers once an area is
    already well covered.
    """
    if not read_token_sets:
        return 1.0
    tokens = significant_tokens(doc.title)
    if not tokens:
        return 0.5
    best_overlap = max(
        len(tokens & other) / max(len(tokens | other), 1) for other in read_token_sets
    )
    return 1.0 - best_overlap


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _select(
    ranked: dict[str, Document],
    active: list[Subquestion],
    settings,
    usage: BudgetUsage,
) -> tuple[list[dict], set[str]]:
    """Choose which documents to read, and for which subquestions.

    Allocation is round-robin across subquestions in priority order rather
    than a single global top-N: a global cut lets one well-served
    subquestion monopolize the budget and starve the others, which is the
    opposite of the coverage the methodology is after.
    """
    threshold = settings.ranking.min_relevance_to_read
    remaining_total = settings.budget.max_documents_read - usage.documents_read
    if remaining_total <= 0:
        logger.info("reading budget exhausted")
        return [], set()

    iteration_cap = min(settings.budget.max_documents_per_iteration, remaining_total)

    # Candidate list per subquestion, best first, excluding pairs already read.
    per_sq: dict[str, list[Document]] = {}
    for sq in active:
        candidates = [
            doc
            for doc in ranked.values()
            if doc.relevance.get(sq.id, 0.0) >= threshold
            and sq.id not in doc.read_for
            and doc.status not in ("failed", "skipped")
        ]
        candidates.sort(
            key=lambda d: (d.relevance.get(sq.id, 0.0) * 0.6 + d.rank_score * 0.4),
            reverse=True,
        )
        per_sq[sq.id] = candidates

    assignments: dict[str, set[str]] = defaultdict(set)
    per_sq_count: dict[str, int] = defaultdict(int)
    cursor: dict[str, int] = defaultdict(int)
    selected: set[str] = set()
    sq_cap = settings.budget.max_documents_per_subquestion

    # Priority order, highest first.
    ordered = sorted(active, key=lambda sq: -sq.priority)

    # Phase 1: allocate reads round-robin. One pass per subquestion per
    # round, highest priority first, until the iteration cap is reached or
    # every subquestion has run out of candidates.
    progress = True
    while len(selected) < iteration_cap and progress:
        progress = False
        for sq in ordered:
            if len(selected) >= iteration_cap:
                break
            if per_sq_count[sq.id] >= sq_cap:
                continue
            candidates = per_sq[sq.id]
            index = cursor[sq.id]
            while index < len(candidates):
                doc = candidates[index]
                index += 1
                if sq.id in assignments[doc.doc_id]:
                    continue
                assignments[doc.doc_id].add(sq.id)
                per_sq_count[sq.id] += 1
                selected.add(doc.doc_id)
                progress = True
                break
            cursor[sq.id] = index

    # Phase 2: attach every other subquestion each selected document is
    # relevant to. Reading one document against several subquestions is a
    # single read, so these attachments cost no document budget -- and they
    # are how a paper retrieved for RQ1 comes to serve RQ3 as well. This runs
    # after phase 1 so the iteration cap never blocks a free attachment.
    for sq in ordered:
        if per_sq_count[sq.id] >= sq_cap:
            continue
        for doc in per_sq[sq.id]:
            if per_sq_count[sq.id] >= sq_cap:
                break
            if doc.doc_id not in selected:
                continue
            if sq.id in assignments[doc.doc_id]:
                continue
            assignments[doc.doc_id].add(sq.id)
            per_sq_count[sq.id] += 1

    queue = [
        {"doc_id": doc_id, "subquestion_ids": sorted(sq_ids)}
        for doc_id, sq_ids in assignments.items()
        if sq_ids
    ]
    # Read the most promising documents first, so a mid-run budget cutoff
    # loses the least valuable work.
    queue.sort(key=lambda item: -ranked[item["doc_id"]].rank_score)
    return queue, selected
