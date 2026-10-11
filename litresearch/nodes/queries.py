"""Source routing and retrieval-query generation.

The plan offered three arrangements of routing and query generation. This
implements **Option 3**: route first, then generate queries specialized for
each selected source, in a single model call per intent.

Why Option 3 rather than the alternatives:

* Option 1 (generate one query, then route it) forces a single phrasing onto
  sources with incompatible query languages. arXiv wants quoted technical
  phrases and ANDs them; Semantic Scholar ranks natural language
  semantically; PubMed wants MeSH terms. One string cannot serve all three.
* Option 2 (fan out to every source, each generating its own queries) spends
  retrieval budget on sources that cannot answer the intent -- PubMed for a
  question about transformer architectures, say.
* Option 3 keeps both properties that matter: only relevant sources are
  queried, and each gets a query written in its own idiom. Doing it in one
  call also lets the model deliberately vary terminology *across* sources
  rather than producing three paraphrases of one phrasing.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import QueryPlanOutput
from ..sources.registry import active_sources
from ..state import BudgetUsage, QueryPlan, ResearchState, SearchIntent
from .common import (
    current_year,
    format_search_history,
    format_source_menu,
    issued_queries,
    log_line,
)

logger = logging.getLogger(__name__)

SYSTEM = """You are the source router and query generator of a literature \
research system.

You receive one search intent -- a description of the documents we need to \
retrieve -- and you produce concrete retrieval queries aimed at specific \
sources.

Two decisions, in order.

FIRST, route. Choose only the sources that can plausibly answer this intent. \
Querying an unsuitable source wastes budget and pollutes the candidate set. \
Most intents are well served by two or three sources; a narrow intent may \
need one. Do not query every available source by reflex.

SECOND, write each query in the idiom of its target source. The same intent \
should produce genuinely DIFFERENT query strings per source, not paraphrases:

- Keyword sources (arxiv) match literally and AND their terms. Use precise \
technical phrases in quotes. Few terms, chosen well. Too many terms returns \
nothing.
- Semantically ranked sources (semantic_scholar, openalex) handle natural \
language. A short descriptive phrase works better than a keyword soup.
- Bibliographic sources (crossref) match on title and venue text. Use the \
wording a title would actually use.
- Controlled-vocabulary sources (pubmed) reward domain terms and boolean \
operators.
- Web search rewards how a practitioner would phrase it, including product \
and framework names.

Terminology is the main failure mode in literature search. The same idea is \
named differently across communities -- "replanning" vs "plan repair", \
"tool use" vs "function calling", "chain-of-thought" vs "scratchpad \
reasoning". When an intent's concepts plausibly go by several names, issue \
separate queries using the different vocabularies rather than guessing one.

Set year_from only when the intent genuinely concerns recent developments. A \
date filter on a request for foundational work actively removes the right \
answers.

Do not repeat a query that has already been issued -- the already-searched \
list is given to you. Produce at most {max_queries} queries in total."""

HUMAN = """Main research question:
{main_question}

Subquestion {sq_id}: {question}

SEARCH INTENT (what we need to retrieve):
{intent}

Why: {rationale}
Evidence that would satisfy it:
{desired}

Sources suggested by the intent planner: {preferred}

Available sources:
{sources}

Queries already issued in this investigation (do not repeat):
{history}

Current year: {year}.

Route this intent and write the queries."""


async def generate_queries(state: ResearchState) -> dict:
    """Build the retrieval queries for every pending intent."""
    settings = get_settings()
    intents: dict[str, SearchIntent] = state.get("intents") or {}
    subquestions = state.get("subquestions") or {}
    source_names = list(active_sources().keys())
    source_menu = format_source_menu(source_names)
    history = format_search_history(state)

    pending = [i for i in intents.values() if i.status == "pending"]
    if not pending:
        return {
            "query_plans": [],
            "log": [log_line("no pending intents; nothing to query")],
        }

    # Respect the remaining search-operation budget. Intents belonging to
    # higher-priority subquestions get the remaining capacity.
    usage = state.get("budget") or BudgetUsage()
    remaining_ops = settings.budget.max_search_operations - usage.search_operations
    if remaining_ops <= 0:
        return {
            "query_plans": [],
            "log": [log_line("search-operation budget exhausted; skipping query generation")],
        }

    pending.sort(
        key=lambda i: -(
            subquestions[i.subquestion_id].priority
            if i.subquestion_id in subquestions
            else 0.0
        )
    )

    max_per_intent = max(1, min(4, len(source_names)))
    plans: list[QueryPlan] = []
    llm_calls = 0

    for intent in pending:
        if len(plans) >= remaining_ops:
            logger.info("search budget reached; %d intent(s) left unqueried", len(pending))
            break

        sq = subquestions.get(intent.subquestion_id)
        messages = [
            SystemMessage(SYSTEM.format(max_queries=max_per_intent)),
            HumanMessage(
                HUMAN.format(
                    main_question=state["main_question"],
                    sq_id=intent.subquestion_id,
                    question=sq.question if sq else "(unknown)",
                    intent=intent.intent,
                    rationale=intent.rationale or "(not stated)",
                    desired="\n".join(f"- {d}" for d in intent.desired_evidence) or "(unspecified)",
                    preferred=", ".join(intent.preferred_sources) or "(no preference)",
                    sources=source_menu,
                    history=history,
                    year=current_year(),
                )
            ),
        ]

        output = await ainvoke_structured("query", QueryPlanOutput, messages)
        llm_calls += 1

        if output is None or not output.queries:
            plans.extend(_fallback_plans(intent, source_names, settings))
            continue

        for item in output.queries[:max_per_intent]:
            if item.source not in source_names:
                logger.debug("dropping query for unavailable source %r", item.source)
                continue
            query = item.query.strip()
            if not query:
                continue
            plans.append(
                QueryPlan(
                    intent_id=intent.id,
                    subquestion_id=intent.subquestion_id,
                    source=item.source,
                    query=query,
                    year_from=item.year_from,
                    rationale=item.rationale,
                    max_results=settings.budget.max_results_per_query,
                )
            )

        if not any(p.intent_id == intent.id for p in plans):
            # Every generated query was unusable; still search something.
            plans.extend(_fallback_plans(intent, source_names, settings))

    # Drop queries already issued earlier in this run as well as duplicates
    # within this batch, so a later iteration spends its budget on new ground.
    plans = _dedupe(plans, already_issued=issued_queries(state))[:remaining_ops]

    # An intent can end up with no queries at all -- every one it generated
    # was a duplicate, or the budget cut it off. Such an intent must not be
    # left "pending": nothing downstream will ever mark it, so it would block
    # its subquestion from generating fresh intents on every later iteration.
    queried = {plan.intent_id for plan in plans}
    intent_updates: dict[str, SearchIntent] = {}
    for intent in pending:
        if intent.id in queried:
            continue
        updated = intent.model_copy(deep=True)
        # Exhausted, not searched: its ground was covered by another intent's
        # identical query, whose results screening scores against every active
        # subquestion anyway.
        updated.status = "exhausted"
        intent_updates[intent.id] = updated

    note = (
        f"routed {len(pending)} intent(s) into {len(plans)} queries across "
        f"{len({p.source for p in plans})} source(s)"
    )
    if intent_updates:
        note += f"; {len(intent_updates)} intent(s) had no new queries and were closed"

    return {
        # Replace-semantics channel: writing the full list each iteration is
        # what stops the previous iteration's searches from being re-run.
        "query_plans": plans,
        "intents": intent_updates,
        "budget": BudgetUsage(llm_calls=llm_calls),
        "log": [log_line(note)],
    }


def _fallback_plans(
    intent: SearchIntent, source_names: list[str], settings
) -> list[QueryPlan]:
    """Query the intent text verbatim against the most capable sources.

    Crude, but a failed query-generation call should not leave a subquestion
    with no retrieval at all.
    """
    preferred = [s for s in intent.preferred_sources if s in source_names]
    if not preferred:
        preferred = [
            s for s in ("semantic_scholar", "openalex", "arxiv") if s in source_names
        ][:2]
    # Strip the imperative framing; "Find methods for X" searches worse than "X".
    query = intent.intent
    for prefix in ("Find ", "Identify ", "Retrieve ", "Locate "):
        if query.startswith(prefix):
            query = query[len(prefix) :]
            break
    query = query.rstrip(".")

    return [
        QueryPlan(
            intent_id=intent.id,
            subquestion_id=intent.subquestion_id,
            source=source,
            query=query,
            rationale="Fallback: query generation failed for this intent.",
            max_results=settings.budget.max_results_per_query,
        )
        for source in preferred
    ]


def _dedupe(
    plans: list[QueryPlan], already_issued: set[tuple[str, str]] | None = None
) -> list[QueryPlan]:
    """Drop repeat queries, within this batch and against the run's history."""
    seen: set[tuple[str, str]] = set(already_issued or ())
    unique = []
    for plan in plans:
        key = (plan.source, plan.query.lower().strip())
        if key in seen:
            logger.debug("dropping duplicate query %r on %s", plan.query, plan.source)
            continue
        seen.add(key)
        unique.append(plan)
    return unique
