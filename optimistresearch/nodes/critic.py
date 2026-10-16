"""Research reflection and the adaptive loop.

Two nodes plus the termination edge.

``critique_research``
    Evaluates the investigation along the dimensions the methodology names:
    coverage, evidence quality, citation coverage, source diversity,
    contradictions, recency, knowledge gaps and search saturation. It does
    not return a binary verdict -- it returns *research actions*: new search
    intents, edits to the research-question tree, and snowball requests.

``apply_critique``
    Turns those actions into state: mutating the subquestion tree, queuing
    new intents, and recording contradictions. This is the step that makes
    the research plan part of the evolving research state rather than a fixed
    input.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import CritiqueOutput
from ..state import (
    BudgetUsage,
    Contradiction,
    Critique,
    Document,
    Evidence,
    PlanEdit,
    ResearchState,
    SearchIntent,
    SnowballRound,
    Subquestion,
)
from ..store.evidence import diversity_report, weakly_supported_claims
from .common import (
    current_year,
    format_subquestion_tree,
    log_line,
    resolve_subquestion_id,
    sorted_subquestions,
    truncate,
)

logger = logging.getLogger(__name__)

SYSTEM = """You are the research critic of a deep literature research system. \
You decide whether the investigation is complete, and if not, what to do next.

Assess these dimensions explicitly:

COVERAGE -- Are all subquestions important to the main question addressed? Is \
any central aspect still unexamined?

EVIDENCE QUALITY -- Are the major claims carried by adequate sources? A claim \
resting on one paper, or on assertions rather than results, is not \
established. You are given the weakly supported claims; judge whether they \
matter.

CITATION COVERAGE -- Does each major claim in the syntheses have evidence \
behind it, or are there assertions nothing supports?

SOURCE DIVERSITY -- Is the synthesis over-dependent on a few papers, one \
research group, or a single venue? You are given concentration statistics.

CONTRADICTIONS -- Are there conflicting findings that need investigating \
rather than being left unresolved?

RECENCY -- Given the question, is relevant recent work missing? Weigh this \
against the question: a question about recent developments with a 2019 median \
publication year has a real problem; a question about theoretical foundations \
does not.

KNOWLEDGE GAPS -- Are important aspects still poorly understood?

SATURATION -- Are further searches still producing new relevant evidence? You \
are given the yield of each search and snowball round. Falling yield across \
rounds means the literature is approaching saturation and further search is \
unlikely to repay its cost.

Then decide.

The number of papers retrieved is NOT a stopping criterion. A run can retrieve \
200 papers and still leave a central question unanswered; a narrow question can \
be settled by 15 highly relevant ones. Judge the evidence against the question.

Return status="sufficient" only when further search would not materially \
change the answer to the main question. Return status="continue" when a \
specific, addressable gap remains -- and then say concretely how to address it.

Your recommended_intents become actual searches, so make them retrieval goals \
specific enough to act on: "Find empirical studies evaluating replanning over \
horizons longer than 20 steps", not "search for more on replanning". Attach \
each to the subquestion it serves.

You may also restructure the research plan, and should when the evidence \
warrants it:
- add: a research direction the literature revealed that the plan missed.
- split: a subquestion whose parts turned out to need separate evidence.
- deprioritize: a subquestion the literature shows to be marginal, or a dead \
end where searching has repeatedly returned nothing relevant.
- mark_answered: a subquestion whose evidence is sufficient.
- rephrase: a subquestion whose framing the evidence shows to be wrong, or \
that used the wrong terminology for this field.

Use snowball_requests when a specific paper's citation neighbourhood is the \
right next move: backward from a key recent paper for foundations it builds \
on, forward from an influential older paper for what came after it.

Be decisive and economical. Every continuation spends budget. Do not ask for \
another round to chase marginal improvements to an answer that is already \
adequate, and do not declare sufficiency to avoid work when a central gap is \
plainly open. If you continue, the gaps you name must be ones the new intents \
would actually close."""

HUMAN = """Main research question:
{main_question}

RESEARCH PLAN
{tree}

RESEARCH BUDGET
- Iteration {iteration} of at most {max_iterations}
- Search operations: {search_ops} of {max_search_ops}
- Documents read: {docs_read} of {max_docs_read}
- Snowball rounds: {snowball_rounds} of {max_snowball_rounds}
- Documents retrieved: {doc_total} (read in depth: {docs_read})
- Evidence items: {evidence_count}

RETRIEVAL YIELD PER ROUND
{yield_history}

EVIDENCE BASE
{diversity}

WEAKLY SUPPORTED CLAIMS (one source, or low-strength support)
{weak_claims}

OPEN CONTRADICTIONS
{contradictions}

SUBQUESTION SYNTHESES
{syntheses}

GLOBAL SYNTHESIS
{global_synthesis}

Current year: {year}.

Assess this investigation and decide whether to continue."""


async def critique_research(state: ResearchState) -> dict:
    """Evaluate research completeness and produce research actions."""
    settings = get_settings()
    documents: dict[str, Document] = state.get("documents") or {}
    evidence: list[Evidence] = state.get("evidence") or []
    usage = state.get("budget") or BudgetUsage()
    iteration = state.get("iteration", 1)

    weak = weakly_supported_claims(evidence)
    weak_text = (
        "\n".join(
            f"- [{c.subquestion_id}] {c.claim} "
            f"({c.support_count} source(s), strength {c.mean_strength:.2f})"
            for c in weak[:25]
        )
        or "(none)"
    )

    open_contradictions = [c for c in (state.get("contradictions") or []) if c.status == "open"]
    contradiction_text = (
        "\n".join(f"- {c.id}: {c.description}" for c in open_contradictions[:20]) or "(none)"
    )

    synthesized = [sq for sq in sorted_subquestions(state) if sq.synthesis]
    syntheses_text = (
        "\n\n".join(
            f"### {sq.id}: {sq.question} (coverage {sq.coverage:.2f})\n"
            f"{truncate(sq.synthesis, 3000)}"
            + (
                "\nGaps noted: " + "; ".join(sq.knowledge_gaps)
                if sq.knowledge_gaps
                else ""
            )
            for sq in synthesized
        )
        or "(no syntheses produced)"
    )

    messages = [
        SystemMessage(SYSTEM),
        HumanMessage(
            HUMAN.format(
                main_question=state["main_question"],
                tree=format_subquestion_tree(state),
                iteration=iteration,
                max_iterations=settings.budget.max_research_iterations,
                search_ops=usage.search_operations,
                max_search_ops=settings.budget.max_search_operations,
                docs_read=usage.documents_read,
                max_docs_read=settings.budget.max_documents_read,
                snowball_rounds=usage.snowball_rounds,
                max_snowball_rounds=settings.budget.max_snowball_rounds,
                doc_total=len(documents),
                evidence_count=len(evidence),
                yield_history=_yield_history(state),
                diversity=diversity_report(evidence, documents).summary(),
                weak_claims=weak_text,
                contradictions=contradiction_text,
                syntheses=syntheses_text,
                global_synthesis=truncate(state.get("global_synthesis") or "(none)", 6000),
                year=current_year(),
            )
        ),
    ]

    output = await ainvoke_structured("critic", CritiqueOutput, messages)

    if output is None:
        # Without a verdict, stop rather than loop blindly: an unbounded loop
        # is the worse failure.
        logger.warning("critique failed; terminating research")
        critique = Critique(
            status="sufficient",
            reasoning="Critic call failed; terminating to avoid an unbounded loop.",
            overall_coverage=_mean_coverage(state),
        )
        return {
            "critique": critique,
            "critique_history": [critique],
            "budget": BudgetUsage(llm_calls=1),
            "log": [log_line("critique failed; terminating")],
        }

    known_ids = set(state.get("subquestions") or {})

    def canonical(raw: str) -> str:
        """Keep the critic's ids usable even when loosely formatted."""
        return resolve_subquestion_id(raw, known_ids) or raw

    critique = Critique(
        status=output.status,
        overall_coverage=max(0.0, min(1.0, output.overall_coverage)),
        reasoning=output.reasoning,
        coverage_assessment=output.coverage_assessment,
        evidence_quality_assessment=output.evidence_quality_assessment,
        source_diversity_assessment=output.source_diversity_assessment,
        recency_assessment=output.recency_assessment,
        saturation_assessment=output.saturation_assessment,
        weak_subquestions=[
            canonical(sq_id) for sq_id in output.weak_subquestions
        ],
        subquestion_coverage={
            canonical(c.subquestion_id): c.coverage
            for c in output.subquestion_coverage
        },
        knowledge_gaps=output.knowledge_gaps,
        unsupported_claims=output.unsupported_claims,
        new_contradictions=[
            Contradiction(description=c.description, subquestion_id=c.subquestion_id)
            for c in output.contradictions
            if c.description.strip()
        ],
        recommended_intents=[
            SearchIntent(
                id="",  # assigned in apply_critique, which can see existing ids
                subquestion_id=canonical(item.subquestion_id),
                intent=item.intent.strip(),
                rationale=item.rationale,
                created_in_iteration=iteration + 1,
                origin="critic",
            )
            for item in output.recommended_intents
            if item.intent.strip()
        ],
        plan_edits=[
            PlanEdit(
                action=edit.action,
                subquestion_id=(
                    canonical(edit.subquestion_id) if edit.subquestion_id else None
                ),
                question=edit.question,
                rationale=edit.rationale,
                children=edit.children,
                priority=edit.priority,
            )
            for edit in output.plan_edits
        ],
        snowball_requests=[
            {"doc_id": req.doc_id, "direction": req.direction, "reason": req.reason}
            for req in output.snowball_requests
            if req.doc_id in documents
        ],
    )

    return {
        "critique": critique,
        "critique_history": [critique],
        "contradictions": critique.new_contradictions,
        "budget": BudgetUsage(llm_calls=1),
        "log": [
            log_line(
                f"critique: {critique.status}, coverage {critique.overall_coverage:.2f}, "
                f"{len(critique.recommended_intents)} new intent(s), "
                f"{len(critique.plan_edits)} plan edit(s), "
                f"{len(critique.snowball_requests)} snowball request(s)"
            )
        ],
    }


async def apply_critique(state: ResearchState) -> dict:
    """Mutate the research plan and queue the critic's recommended work."""
    critique: Critique | None = state.get("critique")
    if critique is None:
        return {"log": [log_line("no critique to apply")]}

    subquestions: dict[str, Subquestion] = dict(state.get("subquestions") or {})
    intents: dict[str, SearchIntent] = state.get("intents") or {}
    iteration = state.get("iteration", 1)

    sq_updates: dict[str, Subquestion] = {}
    notes: list[str] = []

    # Record the critic's coverage judgements, which override the
    # synthesizers' self-assessments: the critic can compare across
    # subquestions, a synthesizer sees only its own.
    for sq_id, coverage in critique.subquestion_coverage.items():
        if sq_id in subquestions:
            updated = sq_updates.get(sq_id) or subquestions[sq_id].model_copy(deep=True)
            updated.coverage = max(0.0, min(1.0, coverage))
            sq_updates[sq_id] = updated

    for edit in critique.plan_edits:
        applied = _apply_edit(edit, subquestions, sq_updates, iteration)
        if applied:
            notes.append(applied)

    # Mark weak subquestions as needing more work so they are eligible for
    # intent generation again.
    for sq_id in critique.weak_subquestions:
        if sq_id in subquestions:
            updated = sq_updates.get(sq_id) or subquestions[sq_id].model_copy(deep=True)
            if updated.status in ("answered", "unexplored"):
                updated.status = "in_progress"
            sq_updates[sq_id] = updated

    # Attach the critic's knowledge gaps to their subquestions so the next
    # round's query generator can see them.
    for gap in critique.knowledge_gaps:
        target = _best_subquestion_for_gap(gap, critique, subquestions)
        if target and target in subquestions:
            updated = sq_updates.get(target) or subquestions[target].model_copy(deep=True)
            if gap not in updated.knowledge_gaps:
                updated.knowledge_gaps = [*updated.knowledge_gaps, gap]
            sq_updates[target] = updated

    # Queue the recommended intents.
    merged = {**subquestions, **sq_updates}
    new_intents: dict[str, SearchIntent] = {}
    for intent in critique.recommended_intents:
        sq_id = intent.subquestion_id
        if sq_id not in merged:
            # Attach an orphaned intent to the weakest subquestion rather
            # than discarding the critic's recommendation.
            fallback = min(merged.values(), key=lambda sq: sq.coverage, default=None)
            if fallback is None:
                continue
            sq_id = fallback.id
        intent_id = _mint_intent_id(sq_id, intents, new_intents)
        new_intents[intent_id] = intent.model_copy(
            update={"id": intent_id, "subquestion_id": sq_id, "status": "pending"}
        )
        updated = sq_updates.get(sq_id) or merged[sq_id].model_copy(deep=True)
        updated.intent_ids = [*updated.intent_ids, intent_id]
        if updated.status == "answered":
            updated.status = "in_progress"
        sq_updates[sq_id] = updated

    note = (
        f"applied critique: {len(new_intents)} new intent(s), "
        f"{len(critique.plan_edits)} plan edit(s)"
    )
    if notes:
        note += " — " + "; ".join(notes)

    return {
        "subquestions": sq_updates,
        "intents": new_intents,
        "snowball_seeds": list(critique.snowball_requests),
        "iteration": iteration + 1,
        "budget": BudgetUsage(research_iterations=1),
        "log": [log_line(note)],
    }


def _apply_edit(
    edit: PlanEdit,
    subquestions: dict[str, Subquestion],
    sq_updates: dict[str, Subquestion],
    iteration: int,
) -> str | None:
    """Apply one plan edit. Returns a human-readable note, or None."""
    if edit.action == "add":
        if not edit.question:
            return None
        new_id = _mint_subquestion_id(subquestions, sq_updates, parent_id=None)
        sq_updates[new_id] = Subquestion(
            id=new_id,
            question=edit.question.strip(),
            rationale=edit.rationale,
            priority=edit.priority if edit.priority is not None else 0.7,
            status="unexplored",
            created_in_iteration=iteration + 1,
            origin="critic",
        )
        return f"added {new_id}"

    target_id = edit.subquestion_id
    if not target_id or target_id not in subquestions:
        return None
    target = sq_updates.get(target_id) or subquestions[target_id].model_copy(deep=True)

    if edit.action == "deprioritize":
        target.status = "deprioritized"
        target.priority = min(target.priority, edit.priority if edit.priority is not None else 0.2)
        sq_updates[target_id] = target
        return f"deprioritized {target_id}"

    if edit.action == "mark_answered":
        target.status = "answered"
        sq_updates[target_id] = target
        return f"marked {target_id} answered"

    if edit.action == "rephrase":
        if not edit.question:
            return None
        target.question = edit.question.strip()
        if edit.rationale:
            target.rationale = edit.rationale
        sq_updates[target_id] = target
        return f"rephrased {target_id}"

    if edit.action == "split":
        if not edit.children:
            return None
        # The parent becomes a container; its evidence stays attached to it,
        # while new search is directed at the children.
        target.status = "superseded"
        created = []
        for index, child_question in enumerate(edit.children, start=1):
            child_id = f"{target_id}.{index}"
            suffix = 0
            while child_id in subquestions or child_id in sq_updates:
                suffix += 1
                child_id = f"{target_id}.{index}b{suffix}"
            sq_updates[child_id] = Subquestion(
                id=child_id,
                question=child_question.strip(),
                rationale=edit.rationale or f"Split from {target_id}.",
                priority=target.priority,
                parent_id=target_id,
                status="unexplored",
                created_in_iteration=iteration + 1,
                origin="critic",
            )
            created.append(child_id)
        target.children = list(dict.fromkeys([*target.children, *created]))
        sq_updates[target_id] = target
        return f"split {target_id} into {', '.join(created)}"

    return None


def _best_subquestion_for_gap(
    gap: str, critique: Critique, subquestions: dict[str, Subquestion]
) -> str | None:
    """Attach a free-text gap to the subquestion it most likely concerns.

    The critic states gaps as prose without saying which subquestion each
    belongs to, so this matches on content and uses the critic's own
    weak-subquestion list as a tie-breaker.
    """
    candidates = [sq for sq in subquestions.values() if sq.is_active()]
    if not candidates:
        return None

    gap_tokens = _tokens(gap)
    weak = set(critique.weak_subquestions)
    recommended = {i.subquestion_id for i in critique.recommended_intents}

    def score(sq: Subquestion) -> tuple[float, float]:
        overlap = 0.0
        if gap_tokens:
            sq_tokens = _tokens(f"{sq.question} {sq.rationale}")
            if sq_tokens:
                overlap = len(gap_tokens & sq_tokens) / len(gap_tokens)
        # A gap the critic also flagged, or wrote an intent for, almost
        # certainly belongs to that subquestion.
        if sq.id in weak:
            overlap += 0.3
        if sq.id in recommended:
            overlap += 0.2
        # Lower coverage breaks ties: an unaddressed gap belongs where the
        # evidence is thinnest.
        return (overlap, -sq.coverage)

    return max(candidates, key=score).id


_GAP_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its no not of on or
    that the to with how what why when evidence research studies study limited
    little more missing lack lacks only few""".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower())
    return {w for w in words if w not in _GAP_STOPWORDS}


def _mint_subquestion_id(
    subquestions: dict[str, Subquestion],
    pending: dict[str, Subquestion],
    parent_id: str | None,
) -> str:
    existing = set(subquestions) | set(pending)
    if parent_id:
        index = 1
        while f"{parent_id}.{index}" in existing:
            index += 1
        return f"{parent_id}.{index}"
    index = 1
    while f"RQ{index}" in existing:
        index += 1
    return f"RQ{index}"


def _mint_intent_id(
    sq_id: str, existing: dict[str, SearchIntent], pending: dict[str, SearchIntent]
) -> str:
    ordinal = 1
    while True:
        candidate = f"{sq_id}-i{ordinal}"
        if candidate not in existing and candidate not in pending:
            return candidate
        ordinal += 1


def _yield_history(state: ResearchState) -> str:
    """Per-round retrieval yield, the input to the saturation judgement."""
    lines = []
    history: list[SnowballRound] = state.get("snowball_history") or []
    for round_record in sorted(history, key=lambda r: r.round_index):
        lines.append(
            f"- Snowball round {round_record.round_index} ({round_record.direction}): "
            f"{round_record.candidates_examined} candidates examined, "
            f"{round_record.newly_relevant} new documents "
            f"(yield {round_record.yield_rate:.0%})"
        )

    # Per-intent search yield, which shows whether querying is still finding
    # anything or has started returning only documents already in the corpus.
    intents: dict[str, SearchIntent] = state.get("intents") or {}
    searched = [i for i in intents.values() if i.status != "pending"]
    if searched:
        lines.append("")
        for intent in sorted(searched, key=lambda i: i.created_in_iteration):
            lines.append(
                f"- Intent {intent.id} (iteration {intent.created_in_iteration}): "
                f"{intent.documents_found} hits -> {intent.relevant_found} screened relevant"
            )
    return "\n".join(lines) or "(no rounds completed)"


def _mean_coverage(state: ResearchState) -> float:
    subquestions = (state.get("subquestions") or {}).values()
    active = [sq for sq in subquestions if sq.status != "deprioritized"]
    if not active:
        return 0.0
    return sum(sq.coverage for sq in active) / len(active)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def should_continue(state: ResearchState) -> str:
    """Conditional edge: another research iteration, or write the report?

    The critic's verdict is authoritative on sufficiency, but the budget is
    authoritative on continuation: a critic that keeps asking for one more
    round must still be stopped.
    """
    settings = get_settings()
    critique: Critique | None = state.get("critique")
    usage = state.get("budget") or BudgetUsage()
    iteration = state.get("iteration", 1)

    if critique is None:
        return "write_report"

    if critique.status == "sufficient":
        return "write_report"

    if iteration >= settings.budget.max_research_iterations:
        logger.info("iteration budget reached at iteration %d", iteration)
        return "write_report"

    if usage.documents_read >= settings.budget.max_documents_read:
        logger.info("reading budget exhausted")
        return "write_report"

    if usage.search_operations >= settings.budget.max_search_operations:
        logger.info("search budget exhausted")
        return "write_report"

    # A continuation with no actionable work would spin the loop without
    # changing the state.
    if not critique.recommended_intents and not critique.snowball_requests:
        if not any(
            edit.action in ("add", "split") for edit in critique.plan_edits
        ):
            logger.info("critic asked to continue but recommended no actions")
            return "write_report"

    return "apply_critique"


def termination_reason(state: ResearchState) -> str:
    """Human-readable explanation of why the run stopped."""
    settings = get_settings()
    critique: Critique | None = state.get("critique")
    usage = state.get("budget") or BudgetUsage()
    iteration = state.get("iteration", 1)

    if critique is None:
        return "No critique was produced; the run terminated defensively."
    if critique.status == "sufficient":
        return f"The research critic judged the evidence sufficient: {critique.reasoning}"
    if iteration >= settings.budget.max_research_iterations:
        return (
            f"Stopped at the iteration limit ({settings.budget.max_research_iterations}) "
            f"with the critic still recommending further work. Remaining gaps: "
            + ("; ".join(critique.knowledge_gaps) or "none recorded")
        )
    if usage.documents_read >= settings.budget.max_documents_read:
        return (
            f"Stopped at the reading limit ({settings.budget.max_documents_read} documents) "
            f"with research still in progress."
        )
    if usage.search_operations >= settings.budget.max_search_operations:
        return (
            f"Stopped at the search-operation limit "
            f"({settings.budget.max_search_operations}) with research still in progress."
        )
    return (
        "The critic recommended continuing but produced no actionable next steps, "
        "so the run terminated."
    )
