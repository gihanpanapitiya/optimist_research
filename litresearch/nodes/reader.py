"""Parallel document reading and evidence extraction.

Each document in the reading queue is read by its own graph branch, fanned
out with ``Send``. Fan-out is real here, unlike in the search stage: these
are long LLM calls, they are the dominant cost of a run, and giving each its
own node means one malformed extraction cannot take down the others.

The reader's primary output is **structured evidence**, not a prose summary.
A summary collapses a paper into one opinion; evidence keeps the claim, its
support, its location and its type separable, which is what later lets the
critic ask whether a claim is actually established and lets the report cite
the specific thing that establishes it.

The same document is read against each subquestion it was assigned to, in a
single call, so one paper can yield different evidence for different
subquestions without being paid for twice.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import ReadingOutput
from ..state import (
    BudgetUsage,
    Contradiction,
    Document,
    DocumentReading,
    Evidence,
    ReaderState,
    ResearchState,
    Subquestion,
)
from ..store.corpus import ResearchCorpus
from .common import format_document_for_reading, log_line, resolve_subquestion_id

logger = logging.getLogger(__name__)

SYSTEM = """You are a document-reading agent in a literature research system. \
You extract structured evidence from one document.

Your primary output is EVIDENCE, not a summary. An evidence item is a claim \
the document establishes, paired with what in the document establishes it.

A good claim is a single, self-contained, checkable assertion:

  GOOD: "Tree-of-Thoughts improves success on Game of 24 from 4% to 74% \
compared to chain-of-thought prompting with GPT-4."
  BAD:  "The paper discusses tree search for reasoning."   (a topic, not a claim)
  BAD:  "The method works well."                           (not checkable)
  BAD:  "LLM agents can plan and also use tools and are evaluated on \
benchmarks."                                               (several claims at once)

A claim must stand on its own. "Our method outperforms the baseline" is \
useless once separated from the paper; name the method, the baseline and the \
task.

For each claim, record:
- support: the specific result, measurement, argument or construction behind \
it. Give numbers where the document gives numbers.
- location: where it is, e.g. "Section 4.2", "Table 3", "Abstract".
- evidence_type: experimental, theoretical, survey, benchmark, position, anecdotal.
- strength: your confidence that the document really establishes the claim. \
This is a judgement about the EVIDENCE, not about how confidently the authors \
assert it. A claim stated in an abstract with no supporting experiment is \
weak however emphatically it is phrased. Score low for: single-benchmark \
results presented as general, missing baselines, no ablation, unstated \
sample sizes, or results you can see only in the abstract.

Attribute to only ONE subquestion per evidence item -- the one it most \
directly bears on. If a finding genuinely serves two, emit two items with \
claims phrased for each.

Also report:
- limitations, including ones the authors do not acknowledge.
- contradictions with other literature, where the document explicitly \
disagrees with prior work or reports a result that conflicts with it.
- research directions the current plan does not cover, but only genuinely \
important omissions.
- specific cited works worth retrieving, by title.
- terminology notes: other names this community uses for these concepts, \
which later searches should try.

{source_caveat}

If, having read it, the document does not actually bear on any assigned \
subquestion, set relevant=false and return no evidence. A screening stage \
passed it to you on its abstract and screening is sometimes wrong; saying so \
is more useful than manufacturing evidence. Never infer findings the document \
does not report."""

ABSTRACT_CAVEAT = """IMPORTANT: you are seeing only the title, abstract and \
metadata -- not the full text. Extract only what the abstract actually \
supports, set location to "Abstract", and keep strength at or below 0.6 for \
empirical claims, since you cannot verify the experimental setup. Do not \
speculate about what the full paper probably contains."""

FULLTEXT_CAVEAT = """You are seeing the paper's text, possibly with lower-value \
sections trimmed. Cite real section headings in location. If a section you \
would need is missing, reflect that in strength rather than guessing."""

HUMAN = """Main research question:
{main_question}

Subquestions this document was selected for:
{subquestions}

Why it was selected: {screening_note}

=== DOCUMENT ({doc_id}) ===
{document}
=== END DOCUMENT ===

Extract the structured evidence."""


def dispatch_readers(state: ResearchState) -> list[Send] | str:
    """Fan out one reader branch per queued document.

    A conditional edge rather than a node: it returns ``Send`` objects, each
    carrying only the slice of state that one reader needs.
    """
    queue = state.get("reading_queue") or []
    if not queue:
        return "synthesize_dispatch"

    settings = get_settings()
    documents: dict[str, Document] = state.get("documents") or {}
    subquestions: dict[str, Subquestion] = state.get("subquestions") or {}

    sends: list[Send] = []
    for item in queue:
        doc = documents.get(item["doc_id"])
        if doc is None:
            continue
        assigned = [
            subquestions[sq_id] for sq_id in item["subquestion_ids"] if sq_id in subquestions
        ]
        if not assigned:
            continue
        sends.append(
            Send(
                "read_document",
                ReaderState(
                    main_question=state["main_question"],
                    doc=doc,
                    subquestions=assigned,
                    read_full_text=settings.read_full_text,
                    iteration=state.get("iteration", 1),
                ),
            )
        )

    return sends or "synthesize_dispatch"


async def read_document(state: ReaderState) -> dict:
    """Read one document and extract its evidence."""
    settings = get_settings()
    doc: Document = state["doc"]
    assigned: list[Subquestion] = state["subquestions"]

    full_text = None
    used_full_text = False
    if state["read_full_text"]:
        corpus = ResearchCorpus(settings.output_dir, state["main_question"])
        extracted = await _load_full_text(doc, corpus.pdf_path(doc.doc_id))
        if extracted is not None:
            full_text = extracted.text
            used_full_text = True

    if not full_text and not doc.abstract:
        # Nothing to read. Mark it so it is not queued again every iteration.
        failed = doc.model_copy(deep=True)
        failed.status = "failed"
        failed.read_for = sorted(set(failed.read_for) | {sq.id for sq in assigned})
        return {
            "documents": {doc.doc_id: failed},
            "log": [log_line(f"{doc.doc_id}: no abstract or full text available; skipped")],
        }

    caveat = FULLTEXT_CAVEAT if used_full_text else ABSTRACT_CAVEAT
    sq_text = "\n".join(f"- {sq.id}: {sq.question}" for sq in assigned)
    valid_sq_ids = {sq.id for sq in assigned}

    messages = [
        SystemMessage(SYSTEM.format(source_caveat=caveat)),
        HumanMessage(
            HUMAN.format(
                main_question=state["main_question"],
                subquestions=sq_text,
                screening_note=doc.screening_note or "(no note)",
                doc_id=doc.doc_id,
                document=format_document_for_reading(doc, full_text),
            )
        ),
    ]

    output = await ainvoke_structured("reader", ReadingOutput, messages)

    if output is None:
        failed = doc.model_copy(deep=True)
        failed.status = "failed"
        return {
            "documents": {doc.doc_id: failed},
            "budget": BudgetUsage(llm_calls=1),
            "log": [log_line(f"{doc.doc_id}: evidence extraction failed")],
        }

    updated = doc.model_copy(deep=True)
    updated.status = "read"
    updated.read_for = sorted(set(updated.read_for) | valid_sq_ids)
    updated.fulltext_chars = len(full_text or "")
    if used_full_text:
        updated.fulltext_path = str(
            ResearchCorpus(settings.output_dir, state["main_question"]).pdf_path(doc.doc_id)
        )

    if not output.relevant:
        updated.notes = "Read and judged not relevant to the assigned subquestions."
        return {
            "documents": {doc.doc_id: updated},
            "budget": BudgetUsage(llm_calls=1),
            "log": [log_line(f"{doc.doc_id}: read, judged not relevant")],
        }

    evidence: list[Evidence] = []
    for item in output.evidence:
        # Guard against evidence attributed to a subquestion this reader was
        # not given: it would otherwise pollute another subquestion's
        # synthesis. A loosely formatted id is repaired first, since models
        # sometimes return the whole prompt line instead of the bare id.
        sq_id = resolve_subquestion_id(item.subquestion_id, valid_sq_ids)
        if sq_id is None:
            if len(valid_sq_ids) == 1:
                sq_id = next(iter(valid_sq_ids))
            else:
                logger.debug(
                    "%s: dropping evidence attributed to unassigned %s",
                    doc.doc_id,
                    item.subquestion_id,
                )
                continue
        if not item.claim.strip() or not item.support.strip():
            continue
        evidence.append(
            Evidence(
                claim=item.claim.strip(),
                support=item.support.strip(),
                doc_id=doc.doc_id,
                subquestion_id=sq_id,
                location=item.location.strip(),
                evidence_type=item.evidence_type,
                strength=max(0.0, min(1.0, item.strength)),
                is_limitation=item.is_limitation,
                verbatim_quote=item.verbatim_quote,
            )
        )

    contradictions = [
        Contradiction(
            description=(
                f"{item.description}"
                + (
                    f" (other work: {item.conflicting_doc_hint})"
                    if item.conflicting_doc_hint
                    else ""
                )
            ),
            subquestion_id=(
                resolve_subquestion_id(item.subquestion_id, valid_sq_ids) or ""
            ),
            doc_ids=[doc.doc_id],
        )
        for item in output.contradictions
        if item.description.strip()
    ]

    reading = DocumentReading(
        doc_id=doc.doc_id,
        main_contribution=output.main_contribution.strip(),
        method=output.method.strip(),
        experimental_setup=output.experimental_setup.strip(),
        major_findings=output.major_findings,
        limitations=output.limitations,
        evidence_ids=[e.id for e in evidence],
        new_subquestions=output.new_subquestions,
        snowball_candidates=output.snowball_candidates,
        read_for=sorted(valid_sq_ids),
        used_full_text=used_full_text,
    )

    # Terminology notes are recorded as knowledge gaps on the subquestion so
    # the next round's query generator sees the alternative vocabulary.
    sq_updates: dict[str, Subquestion] = {}
    for sq in assigned:
        updated_sq = sq.model_copy(deep=True)
        updated_sq.documents_read = sq.documents_read + 1
        if output.terminology_notes:
            updated_sq.knowledge_gaps = list(
                dict.fromkeys(
                    [
                        *sq.knowledge_gaps,
                        *[f"terminology: {t}" for t in output.terminology_notes[:3]],
                    ]
                )
            )
        sq_updates[sq.id] = updated_sq

    mode = "full text" if used_full_text else "abstract"
    return {
        "documents": {doc.doc_id: updated},
        "evidence": evidence,
        "contradictions": contradictions,
        "readings": {doc.doc_id: reading},
        "subquestions": sq_updates,
        "budget": BudgetUsage(documents_read=1, llm_calls=1),
        "log": [
            log_line(
                f"{doc.doc_id}: read from {mode}, extracted {len(evidence)} evidence "
                f"item(s) for {', '.join(sorted(valid_sq_ids))}"
            )
        ],
    }


async def _load_full_text(doc: Document, pdf_path: Path):
    """Import the PDF path lazily so pymupdf stays an optional dependency."""
    from ..fulltext import load_full_text

    try:
        return await load_full_text(doc, pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("full-text load failed for %s: %s", doc.doc_id, exc)
        return None
