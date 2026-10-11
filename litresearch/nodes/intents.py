"""Search intent generation.

Turns each active subquestion into retrieval goals. This is the step the
methodology insists on keeping separate from both the subquestion above it
and the queries below it: the intent describes *what documents would close
the gap*, which is a stable target even as query phrasings get rewritten.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import IntentOutput
from ..sources.registry import active_sources
from ..state import BudgetUsage, ResearchState, SearchIntent, Subquestion
from .common import (
    active_subquestions,
    format_intents,
    format_source_menu,
    log_line,
)

logger = logging.getLogger(__name__)

SYSTEM = """You translate a research subquestion into search intents.

A SUBQUESTION says what we need to know. A SEARCH INTENT says what evidence we \
need to retrieve in order to know it. The intent describes the DOCUMENTS that \
would close the gap -- their content, kind and character -- not the keywords \
that would find them. Query wording is a later, separate step; do not produce \
search strings here.

Subquestion: "How do agents recover from planning failures?"

Good intents:
  - Find methods for detecting plan or action execution failures in agents.
  - Find mechanisms for generating alternative plans after a failure.
  - Find empirical studies evaluating replanning strategies.
  - Find reported limitations of current replanning approaches.

Bad intents:
  - "agent replanning failure recovery"        (a query, not an intent)
  - "Find information about replanning"        (restates the subquestion)
  - "Find papers about agents"                 (not specific enough to act on)

Produce intents that partition the subquestion by the KIND of evidence \
needed, so that each one would retrieve a materially different set of \
documents. Typical useful partitions are: mechanisms and methods; empirical \
evaluations and benchmarks; surveys that map the area; reported limitations \
and negative results; foundational or precursor work.

Produce between 1 and {max_intents} intents. A narrow subquestion may need \
only one; use the full budget only when the subquestion genuinely has that \
many distinct evidence requirements. Each intent you add costs search and \
reading budget.

For each intent, name the sources best suited to it from those offered. Match \
the source to the kind of evidence: preprint servers for very recent methods, \
citation-graph indexes for established work and impact, domain databases for \
domain-specific questions, web search only for grey literature that the \
scholarly indexes would not carry."""

HUMAN = """Main research question:
{main_question}

Subquestion to generate intents for:
{sq_id}: {question}
Why it matters: {rationale}

Intents already issued for this subquestion:
{existing}

Known knowledge gaps for this subquestion:
{gaps}

Available sources:
{sources}

Generate the search intents."""


async def generate_intents(state: ResearchState) -> dict:
    """Create pending search intents for every active subquestion.

    Subquestions that already have unexhausted intents are skipped, so a
    later iteration does not regenerate work the previous one queued.
    """
    settings = get_settings()
    intents: dict[str, SearchIntent] = dict(state.get("intents") or {})
    source_names = list(active_sources().keys())
    source_menu = format_source_menu(source_names)

    targets = [sq for sq in active_subquestions(state) if _needs_intents(sq, intents)]
    if not targets:
        return {"log": [log_line("no subquestions need new intents")]}

    new_intents: dict[str, SearchIntent] = {}
    updated_sqs: dict[str, Subquestion] = {}
    llm_calls = 0

    for sq in targets:
        messages = [
            SystemMessage(
                SYSTEM.format(max_intents=settings.budget.max_intents_per_subquestion)
            ),
            HumanMessage(
                HUMAN.format(
                    main_question=state["main_question"],
                    sq_id=sq.id,
                    question=sq.question,
                    rationale=sq.rationale or "(not stated)",
                    existing=format_intents(intents, sq.id),
                    gaps="\n".join(f"- {g}" for g in sq.knowledge_gaps) or "(none)",
                    sources=source_menu,
                )
            ),
        ]
        output = await ainvoke_structured("intent", IntentOutput, messages)
        llm_calls += 1

        if output is None or not output.intents:
            # Fall back to the subquestion itself as a single intent so the
            # subquestion still gets searched.
            logger.warning("intent generation failed for %s; using fallback", sq.id)
            produced = [
                SearchIntent(
                    id=_mint_intent_id(sq.id, intents, new_intents),
                    subquestion_id=sq.id,
                    intent=f"Find research that addresses: {sq.question}",
                    rationale="Intent generation failed; using the subquestion directly.",
                    created_in_iteration=state.get("iteration", 1),
                )
            ]
        else:
            produced = []
            for item in output.intents[: settings.budget.max_intents_per_subquestion]:
                intent_id = _mint_intent_id(sq.id, intents, new_intents)
                preferred = [s for s in item.preferred_sources if s in source_names]
                produced.append(
                    SearchIntent(
                        id=intent_id,
                        subquestion_id=sq.id,
                        intent=item.intent.strip(),
                        rationale=item.rationale.strip(),
                        desired_evidence=item.desired_evidence,
                        preferred_sources=preferred,
                        created_in_iteration=state.get("iteration", 1),
                        origin="planner",
                    )
                )
                # Register immediately so ids stay unique within this loop.
                new_intents[intent_id] = produced[-1]

        for intent in produced:
            new_intents[intent.id] = intent

        updated = sq.model_copy(deep=True)
        updated.intent_ids = list(
            dict.fromkeys([*sq.intent_ids, *[i.id for i in produced]])
        )
        updated.status = "in_progress"
        updated_sqs[sq.id] = updated

    return {
        "intents": new_intents,
        "subquestions": updated_sqs,
        "budget": BudgetUsage(llm_calls=llm_calls),
        "log": [
            log_line(
                f"generated {len(new_intents)} search intents across "
                f"{len(targets)} subquestion(s)"
            )
        ],
    }


def _needs_intents(sq: Subquestion, intents: dict[str, SearchIntent]) -> bool:
    """True when this subquestion has no intent still awaiting search."""
    return not any(
        intents[i].status == "pending" for i in sq.intent_ids if i in intents
    )


def _mint_intent_id(
    sq_id: str, existing: dict[str, SearchIntent], pending: dict[str, SearchIntent]
) -> str:
    ordinal = 1
    while True:
        candidate = f"{sq_id}-i{ordinal}"
        if candidate not in existing and candidate not in pending:
            return candidate
        ordinal += 1
